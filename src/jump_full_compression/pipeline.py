"""Candidate-only, restart-safe dual JPEG XL compression controller."""

# Thread caps must be established before importing numerical/codec libraries.
# ruff: noqa: E402
from __future__ import annotations

import os
from .model import THREAD_ENV

for _name in THREAD_ENV:
    os.environ[_name] = "1"

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import boto3
import imagecodecs
import numpy as np
import numcodecs
import zarr
from botocore import UNSIGNED
from botocore.config import Config
from imagecodecs.numcodecs import Jpegxl
from PIL import Image, __version__ as PILLOW_VERSION

from .inventory import audit_inventory, candidate_rows, load_audit, normalized_rows
from .model import (
    CODECS,
    COMPRESSION_CPUS,
    INITIAL_WORKERS,
    MAX_CUMULATIVE_ERRORS,
    MAX_WORKERS,
    CandidateConfig,
    assert_no_symlinks,
    assert_runtime_task_ceiling,
    atomic_json,
    fsync_dir,
    sha256_file,
)

numcodecs.register_codec(Jpegxl)
STOP = threading.Event()
_THREAD = threading.local()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _s3_client():
    client = getattr(_THREAD, "s3", None)
    if client is None:
        client = boto3.client(
            "s3",
            region_name="us-east-1",
            config=Config(
                signature_version=UNSIGNED,
                max_pool_connections=4,
                retries={"mode": "adaptive", "max_attempts": 8},
                connect_timeout=20,
                read_timeout=180,
            ),
        )
        _THREAD.s3 = client
    return client


def read_source(
    uri: str, attempts: int = 5, *, allow_file: bool = False, sleep=time.sleep
) -> tuple[bytes, dict[str, Any]]:
    parsed = urlparse(uri)
    if parsed.scheme == "file" and allow_file:
        payload = Path(parsed.path).read_bytes()
        return payload, {
            "uri": uri,
            "size": len(payload),
            "etag": hashlib.md5(payload).hexdigest(),
        }  # noqa: S324
    if (
        parsed.scheme != "s3"
        or parsed.netloc != "cellpainting-gallery"
        or not parsed.path
    ):
        raise ValueError(f"source URI outside public allowlist: {uri}")
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = _s3_client().get_object(
                Bucket=parsed.netloc, Key=parsed.path.lstrip("/")
            )
            payload = response["Body"].read()
            size = int(response.get("ContentLength", len(payload)))
            if len(payload) != size:
                raise OSError(f"truncated source: {len(payload)} != {size}")
            return payload, {
                "uri": uri,
                "size": size,
                "etag": str(response.get("ETag", "")).strip('"'),
            }
        except Exception as error:
            last = error
            if attempt < attempts:
                sleep(min(8.0, 0.25 * 2**attempt))
    assert last is not None
    raise last


def decode_stack(
    row: dict[str, Any], test_mode: bool
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    images, observations = [], []
    for uri in row["urls"]:
        payload, observation = read_source(uri, allow_file=test_mode)
        with Image.open(io.BytesIO(payload)) as image:
            array = np.asarray(image)
        if array.ndim != 2 or array.dtype != np.uint16:
            raise ValueError(
                f"source must be 2-D uint16: {uri} {array.shape} {array.dtype}"
            )
        images.append(array)
        observations.append(observation)
    if len({x.shape for x in images}) != 1:
        raise ValueError("channel shapes differ")
    return np.stack(images, axis=0), observations


def _codec(codec: str) -> Jpegxl:
    return Jpegxl(lossless=False, distance=CODECS[codec]["distance"], numthreads=1)


def _write_staged(stack: np.ndarray, staging: Path, site: str, codec: str) -> Path:
    root = staging / f"{codec}.zarr"
    zarr.create_array(
        store=zarr.storage.LocalStore(root),
        name=site,
        shape=stack.shape,
        chunks=stack.shape,
        dtype=stack.dtype,
        compressors=_codec(codec),
        zarr_format=2,
    )[:] = stack
    return root / site


def validate_site(
    path: Path, expected_shape: tuple[int, ...], codec: str
) -> dict[str, Any]:
    if path.is_symlink() or any(entry.is_symlink() for entry in path.iterdir()):
        raise RuntimeError("symlink in candidate site")
    expected = {".zarray", ".zattrs", "0.0.0"}
    actual = {entry.name for entry in path.iterdir() if entry.is_file()}
    if actual != expected or any(entry.is_dir() for entry in path.iterdir()):
        raise RuntimeError(f"site layout mismatch: {actual}")
    metadata = json.loads((path / ".zarray").read_text())
    compressor = metadata.get("compressor", {})
    if (
        metadata.get("zarr_format") != 2
        or tuple(metadata.get("shape", ())) != expected_shape
        or tuple(metadata.get("chunks", ())) != expected_shape
        or metadata.get("dtype") != "<u2"
        or compressor.get("id") != "imagecodecs_jpegxl"
        or compressor.get("distance") != CODECS[codec]["distance"]
        or compressor.get("numthreads") != 1
    ):
        raise RuntimeError("Zarr/JPEG XL metadata mismatch")
    decoded = zarr.open_array(
        store=zarr.storage.LocalStore(path), mode="r", zarr_format=2
    )[:]
    if decoded.shape != expected_shape or decoded.dtype != np.uint16:
        raise RuntimeError("chunk decode failed")
    return {
        name: {
            "bytes": (path / name).stat().st_size,
            "sha256": sha256_file(path / name),
        }
        for name in sorted(expected)
    }


def _source_tree_identity(repo: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((repo / "src/jump_full_compression").glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def software_identity(*, require_clean: bool) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[2]
    configured_git = os.environ.get("GIT_EXECUTABLE")
    git_path = Path(configured_git) if configured_git else None
    if git_path is None:
        discovered = shutil.which("git")
        git_path = Path(discovered) if discovered else None
    if (
        git_path is None
        or not git_path.is_absolute()
        or not os.access(git_path, os.X_OK)
    ):
        raise RuntimeError("an absolute executable GIT_EXECUTABLE is required")
    if require_clean and configured_git is None:
        raise RuntimeError("live apply requires explicit GIT_EXECUTABLE provenance")
    git = str(git_path.resolve())
    commit = subprocess.check_output(
        [git, "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            [git, "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        ).strip()
    )
    package_files = sorted((repo / "src/jump_full_compression").glob("*.py"))
    tracked = {
        line.strip()
        for line in subprocess.check_output(
            [git, "-C", str(repo), "ls-files", "src/jump_full_compression/*.py"],
            text=True,
        ).splitlines()
    }
    package_tracked = all(
        str(path.relative_to(repo)) in tracked for path in package_files
    )
    if require_clean and (dirty or not package_tracked):
        raise RuntimeError(
            "live apply requires committed package files and a clean tracked tree"
        )
    versions = imagecodecs.version("jpegxl")
    libjxl = next(
        (item for item in versions if item.lower().startswith("libjxl ")), None
    )
    if not libjxl:
        raise RuntimeError("libjxl version unavailable")
    return {
        "git_commit": commit,
        "git_executable": git,
        "git_executable_sha256": sha256_file(Path(git)),
        "tracked_tree_clean": not dirty,
        "package_files_tracked": package_tracked,
        "package_source_sha256": _source_tree_identity(repo),
        "uv_lock_sha256": sha256_file(repo / "uv.lock"),
        "python": sys.version,
        "numpy": np.__version__,
        "pillow": PILLOW_VERSION,
        "numcodecs": numcodecs.__version__,
        "imagecodecs": imagecodecs.__version__,
        "libjxl": libjxl,
        "zarr": zarr.__version__,
        "thread_env": {name: os.environ.get(name) for name in THREAD_ENV},
    }


def _fsync_tree(path: Path) -> None:
    for item in path.iterdir():
        if item.is_file():
            with item.open("rb") as stream:
                os.fsync(stream.fileno())
    fsync_dir(path)


def receipt_path(config: CandidateConfig, site: str) -> Path:
    return config.output_root / "receipts" / f"{site}.json"


def _check_component(path: Path, boundary: Path, *, directory: bool = True) -> None:
    assert_no_symlinks(path, boundary)
    if path.is_symlink() or (path.exists() and directory and not path.is_dir()):
        raise RuntimeError(f"invalid candidate component: {path}")


def _validate_root_layout(config: CandidateConfig) -> None:
    _check_component(config.output_root, config.output_root.parent)
    _check_component(config.state_root, config.state_root.parent)
    if config.output_root.exists():
        allowed = {"codecs", "receipts", ".staging"}
        actual = {path.name for path in config.output_root.iterdir()}
        if not actual <= allowed:
            raise RuntimeError(
                f"unexpected candidate output entries: {actual - allowed}"
            )
        for name in actual:
            _check_component(config.output_root / name, config.output_root)
    if config.state_root.exists():
        allowed = {
            "control.json",
            "compression.json",
            "checkpoint.json",
            "controller.lock",
            "governor_snapshots",
        }
        actual = {path.name for path in config.state_root.iterdir()}
        if not actual <= allowed:
            raise RuntimeError(
                f"unexpected candidate state entries: {actual - allowed}"
            )
        for path in config.state_root.iterdir():
            assert_no_symlinks(path, config.state_root)


def _enumerate_candidate_entries(
    config: CandidateConfig, valid_site_keys: set[str], *, require_exact: bool
) -> tuple[set[str], dict[str, set[str]]]:
    _validate_root_layout(config)
    staging_root = config.output_root / ".staging"
    if staging_root.exists() or staging_root.is_symlink():
        _check_component(staging_root, config.output_root)
        entries = list(staging_root.iterdir())
        if entries:
            raise RuntimeError("candidate staging root is not empty")
    receipts_root = config.output_root / "receipts"
    receipt_sites: set[str] = set()
    if receipts_root.exists():
        _check_component(receipts_root, config.output_root)
        for path in receipts_root.iterdir():
            assert_no_symlinks(path, config.output_root)
            if not path.is_file() or path.suffix != ".json":
                raise RuntimeError(f"unexpected receipt entry: {path}")
            receipt_sites.add(path.stem)
    codecs_parent = config.output_root / "codecs"
    codec_sites: dict[str, set[str]] = {}
    if codecs_parent.exists():
        _check_component(codecs_parent, config.output_root)
        expected_roots = {f"{codec}.zarr" for codec in CODECS}
        actual_roots = {path.name for path in codecs_parent.iterdir()}
        if not actual_roots <= expected_roots:
            raise RuntimeError(
                f"unexpected codec roots: {actual_roots - expected_roots}"
            )
        for codec in CODECS:
            root = codecs_parent / f"{codec}.zarr"
            sites: set[str] = set()
            if root.exists() or root.is_symlink():
                _check_component(root, config.output_root)
                for path in root.iterdir():
                    assert_no_symlinks(path, config.output_root)
                    if not path.is_dir() or path.name.startswith("."):
                        raise RuntimeError(f"unexpected codec entry: {path}")
                    sites.add(path.name)
            codec_sites[codec] = sites
    else:
        codec_sites = {codec: set() for codec in CODECS}
    sets = [receipt_sites, *codec_sites.values()]
    if any(not sites <= valid_site_keys for sites in sets):
        raise RuntimeError("candidate roots contain unknown site keys")
    if require_exact and any(sites != valid_site_keys for sites in sets):
        raise RuntimeError("candidate roots are not exactly complete")
    return receipt_sites, codec_sites


def _validate_receipt(
    config: CandidateConfig,
    row: dict[str, Any],
    producer: dict[str, Any],
    expected_sha256: str | None = None,
) -> bool:
    path = receipt_path(config, row["Metadata_Site_Key"])
    if not path.is_file() or path.is_symlink():
        return False
    assert_no_symlinks(path, config.output_root)
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise RuntimeError("checkpoint receipt hash drift")
    payload = json.loads(path.read_text())
    required = {
        "format_version",
        "candidate_only",
        "adoption_requires_frozen_manifest_revalidation",
        "site_key",
        "source_row_sha256",
        "channels",
        "sources",
        "outputs",
        "shape",
        "dtype",
        "codec_json",
        "inventory_digest",
        "config_sha256",
        "software",
        "completed_at",
    }
    if set(payload) != required:
        raise RuntimeError("receipt fields are not exact")
    shape = payload.get("shape")
    sources = payload.get("sources")
    if (
        payload.get("format_version") != "full-jump-candidate-receipt-v2"
        or payload.get("candidate_only") is not True
        or payload.get("adoption_requires_frozen_manifest_revalidation") is not True
        or payload.get("dtype") != "<u2"
        or not isinstance(shape, list)
        or len(shape) != 3
        or shape[0] != len(row["channels"])
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in shape
        )
        or not isinstance(sources, list)
        or len(sources) != len(row["urls"])
    ):
        raise RuntimeError("receipt format/shape/source contract drift")
    for expected_uri, observation in zip(row["urls"], sources):
        if (
            set(observation) != {"uri", "size", "etag"}
            or observation.get("uri") != expected_uri
            or not isinstance(observation.get("size"), int)
            or observation["size"] <= 0
            or not isinstance(observation.get("etag"), str)
            or not re.fullmatch(r"[A-Fa-f0-9]{16,64}(?:-\d+)?", observation["etag"])
        ):
            raise RuntimeError("receipt source observation drift")
    if (
        payload["site_key"] != row["Metadata_Site_Key"]
        or payload["source_row_sha256"] != row["source_row_sha256"]
        or payload["channels"] != row["channels"]
        or [x.get("uri") for x in payload["sources"]] != row["urls"]
        or payload["config_sha256"] != config.digest
        or payload["inventory_digest"] != config.inventory_digest
        or set(payload["outputs"]) != set(CODECS)
        or payload["codec_json"] != CODECS
        or payload["software"] != producer
    ):
        raise RuntimeError("receipt identity/producer drift")
    for codec in CODECS:
        site_path = (
            config.output_root / "codecs" / f"{codec}.zarr" / row["Metadata_Site_Key"]
        )
        if (
            validate_site(site_path, tuple(payload["shape"]), codec)
            != payload["outputs"][codec]
        ):
            raise RuntimeError("receipt output checksum drift")
    return True


def _remove_site(config: CandidateConfig, site: str) -> None:
    for codec in CODECS:
        target = config.output_root / "codecs" / f"{codec}.zarr" / site
        if target.exists() or target.is_symlink():
            if target.is_symlink():
                raise RuntimeError("refuse to remove symlink site")
            shutil.rmtree(target)
    receipt_path(config, site).unlink(missing_ok=True)


def build_site(
    config: CandidateConfig, row: dict[str, Any], producer: dict[str, Any]
) -> dict[str, Any]:
    site = row["Metadata_Site_Key"]
    if _validate_receipt(config, row, producer):
        return {"site": site, "status": "skipped"}
    _remove_site(config, site)
    staging_parent = config.output_root / ".staging"
    assert_no_symlinks(staging_parent, config.output_root)
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = staging_parent / f"{site}.{uuid4().hex}"
    staging.mkdir()
    try:
        stack, observations = decode_stack(row, config.test_mode)
        staged = {}
        outputs = {}
        for codec in CODECS:
            staged[codec] = _write_staged(stack, staging, site, codec)
            outputs[codec] = validate_site(staged[codec], tuple(stack.shape), codec)
            _fsync_tree(staged[codec])
        for codec in CODECS:
            target = config.output_root / "codecs" / f"{codec}.zarr" / site
            assert_no_symlinks(target, config.output_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            fsync_dir(target.parent)
            os.replace(staged[codec], target)
            fsync_dir(target.parent)
        receipt = {
            "format_version": "full-jump-candidate-receipt-v2",
            "candidate_only": True,
            "adoption_requires_frozen_manifest_revalidation": True,
            "site_key": site,
            "source_row_sha256": row["source_row_sha256"],
            "channels": row["channels"],
            "shape": list(stack.shape),
            "dtype": stack.dtype.str,
            "sources": observations,
            "outputs": outputs,
            "codec_json": CODECS,
            "inventory_digest": config.inventory_digest,
            "config_sha256": config.digest,
            "software": producer,
            "completed_at": now(),
        }
        final_receipt = receipt_path(config, site)
        assert_no_symlinks(final_receipt, config.output_root)
        atomic_json(final_receipt, receipt)
        return {
            "site": site,
            "status": "created",
            "receipt_sha256": sha256_file(final_receipt),
        }
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _control(
    config: CandidateConfig,
    timestamp: float | None = None,
    cumulative_errors: int | None = None,
) -> dict[str, Any]:
    now_ts = time.time() if timestamp is None else timestamp
    path = config.state_root / "control.json"
    if not path.is_file() or path.is_symlink():
        return {
            "paused": True,
            "desired_workers": INITIAL_WORKERS,
            "reason": "missing control",
        }
    try:
        payload = json.loads(path.read_text())
        acknowledged = int(payload.get("acknowledged_error_count", -1))
        if (
            payload.get("format_version") != "full-jump-compression-control-v2"
            or payload.get("candidate_id") != config.candidate_id
            or payload.get("config_sha256") != config.digest
            or payload.get("compression_cpus") != list(COMPRESSION_CPUS)
            or payload.get("max_workers") != MAX_WORKERS
            or payload.get("feature_processes_mutated") is not False
            or payload.get("governor_evaluation_required", False) is not False
            or not 0 <= acknowledged <= MAX_CUMULATIVE_ERRORS
            or (
                acknowledged > 0
                and payload.get("acknowledgement_source") != "explicit-cli"
            )
            or not 1 <= int(payload.get("desired_workers", 0)) <= config.max_workers
            or now_ts - float(payload.get("observed_at_unix", 0)) > 4 * 3600
            or now_ts < float(payload.get("observed_at_unix", 0))
        ):
            raise ValueError("control identity/age drift")
        if cumulative_errors is not None and acknowledged != cumulative_errors:
            raise ValueError(
                "control acknowledgement must equal persistent cumulative errors"
            )
        return payload
    except Exception as error:
        return {
            "paused": True,
            "desired_workers": INITIAL_WORKERS,
            "reason": f"malformed/stale control: {error}",
        }


def _write_state(config: CandidateConfig, **fields: Any) -> None:
    old = {}
    path = config.state_root / "compression.json"
    if path.exists() or path.is_symlink():
        assert_no_symlinks(path, config.state_root)
        old = json.loads(path.read_text())
        if (
            old.get("format_version") != "full-jump-compression-state-v2"
            or old.get("candidate_id") != config.candidate_id
            or old.get("config_sha256") != config.digest
        ):
            raise RuntimeError("persistent compression state identity drift")
    cumulative = max(
        int(old.get("cumulative_errors", 0)), int(fields.pop("cumulative_errors", 0))
    )
    if not 0 <= cumulative <= MAX_CUMULATIVE_ERRORS:
        raise RuntimeError("cumulative error count outside bounded range")
    assert_no_symlinks(path, config.state_root)
    atomic_json(
        path,
        {
            "format_version": "full-jump-compression-state-v2",
            "candidate_id": config.candidate_id,
            "config_sha256": config.digest,
            "cumulative_errors": cumulative,
            "heartbeat_unix": time.time(),
            **fields,
        },
    )


def _load_checkpoint(
    config: CandidateConfig, rows: list[dict[str, Any]], producer: dict[str, Any]
) -> tuple[int, int, int]:
    path = config.state_root / "checkpoint.json"
    if not path.is_file():
        return 0, 0, 0
    assert_no_symlinks(path, config.state_root)
    payload = json.loads(path.read_text())
    if (
        payload.get("format_version") != "full-jump-candidate-checkpoint-v2"
        or payload.get("candidate_id") != config.candidate_id
        or payload.get("config_sha256") != config.digest
        or payload.get("inventory_digest") != config.inventory_digest
    ):
        raise RuntimeError("checkpoint identity drift")
    index = int(payload.get("next_index", -1))
    if not 0 <= index <= len(rows) or payload.get("last_site_key") != (
        rows[index - 1]["Metadata_Site_Key"] if index else None
    ):
        raise RuntimeError("checkpoint ordering drift")
    receipt_hashes = payload.get("receipt_sha256")
    expected_keys = {row["Metadata_Site_Key"] for row in rows[:index]}
    if not isinstance(receipt_hashes, dict) or set(receipt_hashes) != expected_keys:
        raise RuntimeError("checkpoint receipt hash set drift")
    invalid = index
    for position, row in enumerate(rows[:index]):
        try:
            if not _validate_receipt(
                config,
                row,
                producer,
                receipt_hashes[row["Metadata_Site_Key"]],
            ):
                invalid = min(invalid, position)
                break
        except Exception:
            invalid = min(invalid, position)
            break
    if invalid < index:
        for row in rows[invalid:index]:
            _remove_site(config, row["Metadata_Site_Key"])
        index = invalid
        assert_no_symlinks(path, config.state_root)
        atomic_json(
            path,
            {
                **payload,
                "next_index": index,
                "last_site_key": rows[index - 1]["Metadata_Site_Key"]
                if index
                else None,
                "created": index,
                "skipped": 0,
                "complete": False,
                "receipt_sha256": {
                    row["Metadata_Site_Key"]: receipt_hashes[row["Metadata_Site_Key"]]
                    for row in rows[:index]
                },
                "rolled_back_at": now(),
            },
        )
        return index, index, 0
    return index, int(payload.get("created", 0)), int(payload.get("skipped", 0))


def _prepare(
    config: CandidateConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config.validate()
    if not config.test_mode:
        assert_runtime_task_ceiling()
    audit = load_audit(
        config.audit_report, config.manifest, config.inventory_digest, kind="candidate"
    )
    if (
        audit["manifest"]["sha256"] != config.manifest_sha256
        or audit["manifest"]["bytes"] != config.manifest_size
    ):
        raise RuntimeError("configured manifest identity drift")
    rows = list(
        candidate_rows(config.manifest, expected_count=int(audit["site_count"]))
    )
    producer = software_identity(require_clean=not config.test_mode)
    return audit, rows, producer


def _paused_control(
    config: CandidateConfig, acknowledged: int, reason: str
) -> dict[str, Any]:
    return {
        "format_version": "full-jump-compression-control-v2",
        "candidate_id": config.candidate_id,
        "config_sha256": config.digest,
        "paused": True,
        "desired_workers": INITIAL_WORKERS,
        "max_workers": MAX_WORKERS,
        "consecutive_healthy_windows": 0,
        "compression_cpus": list(COMPRESSION_CPUS),
        "acknowledged_error_count": acknowledged,
        "acknowledgement_source": "explicit-cli" if acknowledged else "zero",
        "reasons": [reason],
        "observed_at_unix": time.time(),
        "feature_processes_mutated": False,
        "governor_evaluation_required": True,
    }


def bootstrap_candidate(config: CandidateConfig, apply: bool) -> dict[str, Any]:
    _, rows, _ = _prepare(config)
    result = {
        "status": "bootstrap-dry-run" if not apply else "bootstrapped-paused",
        "candidate_id": config.candidate_id,
        "config_sha256": config.digest,
        "sites": len(rows),
    }
    if not apply:
        return result
    _validate_root_layout(config)
    config.state_root.mkdir(parents=True, exist_ok=True)
    lock_path = config.state_root / "controller.lock"
    assert_no_symlinks(lock_path, config.state_root)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        occupied = {
            name
            for name in ("control.json", "compression.json", "checkpoint.json")
            if (config.state_root / name).exists()
        }
        if occupied:
            raise RuntimeError(f"bootstrap refuses initialized state: {occupied}")
        _write_state(
            config,
            state="bootstrap-paused",
            next_index=0,
            processed=0,
            sites=len(rows),
            cumulative_errors=0,
            bootstrap=True,
        )
        atomic_json(
            config.state_root / "control.json",
            _paused_control(config, 0, "bootstrap requires governor re-evaluation"),
        )
    return result


def acknowledge_errors(
    config: CandidateConfig, expected_count: int, apply: bool
) -> dict[str, Any]:
    _prepare(config)
    if not 0 <= expected_count <= MAX_CUMULATIVE_ERRORS:
        raise ValueError("expected cumulative error count outside bounded range")
    _validate_root_layout(config)
    state_path = config.state_root / "compression.json"
    if not state_path.is_file() or state_path.is_symlink():
        raise RuntimeError("compression state unavailable for acknowledgement")
    state = json.loads(state_path.read_text())
    if (
        state.get("format_version") != "full-jump-compression-state-v2"
        or state.get("candidate_id") != config.candidate_id
        or state.get("config_sha256") != config.digest
    ):
        raise RuntimeError("compression state identity drift")
    observed = int(state.get("cumulative_errors", -1))
    if observed != expected_count:
        raise RuntimeError(
            f"expected count {expected_count} does not equal observed {observed}"
        )
    result = {
        "status": "acknowledgement-dry-run" if not apply else "acknowledged-paused",
        "candidate_id": config.candidate_id,
        "config_sha256": config.digest,
        "acknowledged_error_count": observed,
        "governor_evaluation_required": True,
    }
    if not apply:
        return result
    lock_path = config.state_root / "controller.lock"
    assert_no_symlinks(lock_path, config.state_root)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("controller active during acknowledgement") from None
        atomic_json(
            config.state_root / "control.json",
            _paused_control(
                config,
                observed,
                "explicit error acknowledgement; governor re-evaluation required",
            ),
        )
    return result


def run_candidate(config: CandidateConfig, apply: bool) -> dict[str, Any]:
    _, rows, producer = _prepare(config)
    if not apply:
        return {"status": "dry-run", "sites": len(rows), "config_sha256": config.digest}
    _validate_root_layout(config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    config.state_root.mkdir(parents=True, exist_ok=True)
    assert_no_symlinks(config.output_root, config.output_root.parent)
    assert_no_symlinks(config.state_root, config.state_root.parent)
    lock_path = config.state_root / "controller.lock"
    assert_no_symlinks(lock_path, config.state_root)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "another candidate controller owns the state root"
            ) from None
        valid_site_keys = {row["Metadata_Site_Key"] for row in rows}
        _enumerate_candidate_entries(config, valid_site_keys, require_exact=False)
        for codec in CODECS:
            root = config.output_root / "codecs" / f"{codec}.zarr"
            assert_no_symlinks(root, config.output_root)
            root.mkdir(parents=True, exist_ok=True)
            if (
                root.is_symlink()
                or (root / ".zgroup").exists()
                or (root / ".zattrs").exists()
            ):
                raise RuntimeError("invalid candidate codec root")
        index, created, skipped = _load_checkpoint(config, rows, producer)
        state_path = config.state_root / "compression.json"
        old_state = json.loads(state_path.read_text()) if state_path.is_file() else {}
        if old_state and (
            old_state.get("format_version") != "full-jump-compression-state-v2"
            or old_state.get("candidate_id") != config.candidate_id
            or old_state.get("config_sha256") != config.digest
        ):
            raise RuntimeError("persistent compression state identity drift")
        cumulative_errors = int(old_state.get("cumulative_errors", 0))
        if not 0 <= cumulative_errors <= MAX_CUMULATIVE_ERRORS:
            raise RuntimeError("cumulative error count outside bounded range")
        while index < len(rows) and not STOP.is_set():
            control = _control(config, cumulative_errors=cumulative_errors)
            if control.get("paused"):
                _write_state(
                    config,
                    state="paused",
                    next_index=index,
                    cumulative_errors=cumulative_errors,
                    control_reason=control.get("reason") or control.get("reasons"),
                )
                STOP.wait(60)
                continue
            workers = min(int(control["desired_workers"]), config.max_workers)
            end = min(index + max(config.batch_size, workers), len(rows))
            batch = rows[index:end]
            _write_state(
                config,
                state="compressing",
                next_index=index,
                processed=index,
                sites=len(rows),
                desired_workers=workers,
                cumulative_errors=cumulative_errors,
            )
            try:
                if not config.test_mode:
                    assert_runtime_task_ceiling(
                        additional_tasks=min(workers, len(batch))
                    )
                with ThreadPoolExecutor(
                    max_workers=min(workers, len(batch)),
                    initializer=(
                        assert_runtime_task_ceiling if not config.test_mode else None
                    ),
                ) as executor:
                    results = list(
                        executor.map(
                            lambda row: build_site(config, row, producer), batch
                        )
                    )
            except Exception:
                cumulative_errors += 1
                _write_state(
                    config,
                    state="failed",
                    next_index=index,
                    cumulative_errors=cumulative_errors,
                )
                raise
            created += sum(x["status"] == "created" for x in results)
            skipped += sum(x["status"] == "skipped" for x in results)
            index = end
            checkpoint_path = config.state_root / "checkpoint.json"
            assert_no_symlinks(checkpoint_path, config.state_root)
            atomic_json(
                checkpoint_path,
                {
                    "format_version": "full-jump-candidate-checkpoint-v2",
                    "candidate_id": config.candidate_id,
                    "config_sha256": config.digest,
                    "inventory_digest": config.inventory_digest,
                    "next_index": index,
                    "last_site_key": rows[index - 1]["Metadata_Site_Key"],
                    "created": created,
                    "skipped": skipped,
                    "processed": index,
                    "complete": index == len(rows),
                    "receipt_sha256": {
                        row["Metadata_Site_Key"]: sha256_file(
                            receipt_path(config, row["Metadata_Site_Key"])
                        )
                        for row in rows[:index]
                    },
                    "updated_at": now(),
                },
            )
        status = "stopped" if STOP.is_set() else "complete"
        _write_state(
            config,
            state=status,
            next_index=index,
            processed=index,
            sites=len(rows),
            cumulative_errors=cumulative_errors,
        )
        return {
            "status": status,
            "sites": len(rows),
            "next_index": index,
            "created": created,
            "skipped": skipped,
            "cumulative_errors": cumulative_errors,
        }


def validate_adoption_seam(
    config: CandidateConfig,
    frozen_manifest: Path,
    frozen_audit: Path,
    frozen_inventory_digest: str,
    exclusion_policy: Path,
    damaged_objects: Path,
    damaged_sites: Path,
    qc_plates: Path,
    build_report: Path,
) -> dict[str, Any]:
    _, candidate_rows_original, producer = _prepare(config)
    audit = audit_inventory(
        frozen_manifest,
        kind="frozen",
        exclusion_policy=exclusion_policy,
        damaged_objects=damaged_objects,
        damaged_sites=damaged_sites,
        qc_plates=qc_plates,
        build_report=build_report,
    )
    recorded = load_audit(
        frozen_audit, frozen_manifest, frozen_inventory_digest, kind="frozen"
    )
    if (
        audit["inventory_digest"] != frozen_inventory_digest
        or recorded["full_row_sha256"] != audit["full_row_sha256"]
        or recorded["site_key_sha256"] != audit["site_key_sha256"]
    ):
        raise RuntimeError("frozen manifest/audit digest mismatch")
    expected = {row["Metadata_Site_Key"] for row in candidate_rows_original}
    checkpoint_path = config.state_root / "checkpoint.json"
    assert_no_symlinks(checkpoint_path, config.state_root)
    if not checkpoint_path.is_file():
        raise RuntimeError("complete candidate checkpoint missing")
    checkpoint = json.loads(checkpoint_path.read_text())
    receipt_hashes = checkpoint.get("receipt_sha256")
    if (
        checkpoint.get("format_version") != "full-jump-candidate-checkpoint-v2"
        or checkpoint.get("candidate_id") != config.candidate_id
        or checkpoint.get("config_sha256") != config.digest
        or checkpoint.get("inventory_digest") != config.inventory_digest
        or checkpoint.get("complete") is not True
        or checkpoint.get("next_index") != len(candidate_rows_original)
        or checkpoint.get("processed") != len(candidate_rows_original)
        or checkpoint.get("last_site_key")
        != candidate_rows_original[-1]["Metadata_Site_Key"]
        or int(checkpoint.get("created", -1)) + int(checkpoint.get("skipped", -1))
        != len(candidate_rows_original)
        or not isinstance(receipt_hashes, dict)
        or set(receipt_hashes) != expected
    ):
        raise RuntimeError("candidate checkpoint is not exactly complete")
    _enumerate_candidate_entries(config, expected, require_exact=True)
    frozen_rows = {
        row["Metadata_Site_Key"]: row
        for row in normalized_rows(frozen_manifest)
        if row["Metadata_Site_Key"] in expected
    }
    if set(frozen_rows) != expected:
        raise RuntimeError("candidate absent from frozen manifest")
    matched = set()
    for original in candidate_rows_original:
        key = original["Metadata_Site_Key"]
        frozen_row = frozen_rows[key]
        if original["source_row_sha256"] != frozen_row["source_row_sha256"]:
            raise RuntimeError("candidate row changed in frozen manifest")
        if not _validate_receipt(config, frozen_row, producer, receipt_hashes[key]):
            raise RuntimeError("candidate receipt missing")
        matched.add(key)
    return {
        "candidate_only": True,
        "promotion_performed": False,
        "frozen_inventory_digest": frozen_inventory_digest,
        "validated_receipts": len(matched),
        "site_keys": sorted(matched),
    }


def install_signal_handlers() -> None:
    def stop(_signum, _frame):
        STOP.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
