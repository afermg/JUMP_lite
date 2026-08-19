"""Streaming, tranche-committed full-JUMP production compressor."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import re
import resource
import shutil
import stat
import threading
import time
from datetime import datetime
from typing import Any, Callable, Iterator
from uuid import uuid4

import pyarrow.parquet as pq

from .inventory import (
    _canonical_row,
    _physical_order_key,
    inventory_digest_from_report,
)
from .model import (
    CODECS,
    COMPRESSION_CPUS,
    INITIAL_WORKERS,
    MAX_CUMULATIVE_ERRORS,
    MAX_WORKERS,
    ProductionConfig,
    assert_runtime_task_ceiling,
    atomic_json,
    digest_json,
    fsync_dir,
    sha256_file,
)
from .pipeline import (
    STOP,
    _control,
    _fsync_tree,
    _write_staged,
    decode_stack,
    now,
    software_identity,
    validate_site,
)

CHECKPOINT_FORMAT = "full-jump-production-checkpoint-v1"
RECEIPT_FORMAT = "full-jump-production-site-receipt-v1"
TRANCHE_FORMAT = "full-jump-production-tranche-v1"
STATE_FORMAT = "full-jump-compression-state-v2"
PRODUCER_FORMAT = "full-jump-production-producer-v1"
ZERO_CHAIN = "0" * 64
HEARTBEAT_SECONDS = 30
CONTINUOUS_AUTH_FORMAT = "full-jump-production-continuous-authorization-v1"
ONE_TRANCHE_ACCEPTANCE_FORMAT = "full-jump-one-tranche-acceptance-v1"
MIGRATION_ACCEPTANCE_FORMAT = "producer-migration-acceptance-v1"
PRODUCER_TRANSITION_FORMAT = "full-jump-production-producer-transition-v1"
AUTHORIZED_FIRST_TRANCHE_DIGEST = (
    "8431770a92782eaeab5e7e41f430aec53d67ab58909aa1bac8ec1d8889654a77"
)
AUTHORIZED_PREDECESSOR_PRODUCER_SHA256 = (
    "eea9ed8964f7d2f3ce9a164becdfa0530818b07855cde1b578f22e8c686d469a"
)
AUTHORIZED_PREDECESSOR_COMMIT = "75b18904ea0fe18610feb840888794733fea2fd0"
IDLE_TERMINAL_STATES = {"session-complete", "complete", "stopped", "paused"}


def _no_fault(_point: str) -> None:
    return None


FAULT_HOOK: Callable[[str], None] = _no_fault


class ManifestSnapshot:
    """One stable manifest inode used for authentication and every Parquet scan."""

    def __init__(self, path: Path):
        self.path = path
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        self.fd = os.open(path, flags)
        observed = os.fstat(self.fd)
        if not stat.S_ISREG(observed.st_mode):
            os.close(self.fd)
            raise RuntimeError("production manifest snapshot is not a regular file")
        self.stat = observed
        self.sha256 = self._hash()
        self.binding = {"bytes": observed.st_size, "sha256": self.sha256}
        self.columns: tuple[str, ...] | None = None
        self.row_groups: tuple[int, ...] | None = None
        self.row_count: int | None = None

    def _handle(self):
        duplicate = os.dup(self.fd)
        os.lseek(duplicate, 0, os.SEEK_SET)
        return os.fdopen(duplicate, "rb", closefd=True)

    def _hash(self) -> str:
        digest = hashlib.sha256()
        with self._handle() as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def load_metadata(self) -> None:
        if self.row_count is not None:
            return
        with self._handle() as handle:
            parquet = pq.ParquetFile(handle)
            self.columns = tuple(parquet.schema_arrow.names)
            self.row_groups = tuple(
                parquet.metadata.row_group(index).num_rows
                for index in range(parquet.metadata.num_row_groups)
            )
            self.row_count = int(parquet.metadata.num_rows)

    def iter_rows(self, start: int = 0) -> Iterator[dict[str, Any]]:
        if self.row_count is None or self.columns is None or self.row_groups is None:
            raise RuntimeError("manifest snapshot metadata was not authenticated")
        if not 0 <= start <= self.row_count:
            raise ValueError("manifest snapshot start outside row count")
        group = 0
        preceding = 0
        while (
            group < len(self.row_groups) and preceding + self.row_groups[group] <= start
        ):
            preceding += self.row_groups[group]
            group += 1
        skip = start - preceding
        with self._handle() as handle:
            parquet = pq.ParquetFile(handle)
            for group_index in range(group, len(self.row_groups)):
                for batch in parquet.iter_batches(
                    batch_size=1024,
                    row_groups=[group_index],
                    columns=list(self.columns),
                    use_threads=False,
                ):
                    values = batch.to_pydict()
                    for index in range(batch.num_rows):
                        if skip:
                            skip -= 1
                            continue
                        row = {column: values[column][index] for column in self.columns}
                        _physical_order_key(row)
                        yield _canonical_row(row, self.columns)

    def close(self) -> None:
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1

    def __enter__(self) -> "ManifestSnapshot":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def _binding(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _validate_identity(
    config: ProductionConfig, snapshot: ManifestSnapshot
) -> dict[str, Any]:
    config.validate()
    if snapshot.binding != {
        "bytes": config.manifest_size,
        "sha256": config.manifest_sha256,
    }:
        raise RuntimeError("stable production manifest binding drift")
    snapshot.load_metadata()
    if snapshot.row_count != config.site_count:
        raise RuntimeError("stable production manifest row-count drift")
    expected = {
        config.audit_report: config.audit_sha256,
        config.build_report: config.build_report_sha256,
        config.exclusion_policy: config.exclusion_policy_sha256,
        config.damaged_objects: config.damaged_objects_sha256,
        config.damaged_sites: config.damaged_sites_sha256,
        config.qc_plates: config.qc_plates_sha256,
    }
    for path, digest in expected.items():
        if _binding(path)["sha256"] != digest:
            raise RuntimeError(f"production identity binding drift: {path}")
    audit = json.loads(config.audit_report.read_text())
    try:
        recomputed = inventory_digest_from_report(audit)
        local_digest = digest_json(audit["local_observation"])
    except Exception as error:
        raise RuntimeError("frozen production audit fields malformed") from error
    if (
        audit.get("format_version") != "full-jump-inventory-audit-v2"
        or audit.get("audit_kind") != "frozen"
        or audit.get("release_identity_frozen") is not True
        or audit.get("audit_success") is not True
        or audit.get("site_count") != config.site_count
        or audit.get("inventory_digest") != config.inventory_digest
        or recomputed != config.inventory_digest
        or audit.get("local_observation_digest") != local_digest
        or audit.get("manifest") != snapshot.binding
    ):
        raise RuntimeError("frozen production audit identity drift")
    build = json.loads(config.build_report.read_text())
    if (
        build.get("format_version") != "full-jump-production-manifest-build-v1"
        or build.get("build_success") is not True
        or build.get("output", {}).get("rows") != config.site_count
        or build.get("output", {}).get("bytes") != config.manifest_size
        or build.get("output", {}).get("sha256") != config.manifest_sha256
        or build.get("policy_action") != "exclude_red_include_gray"
    ):
        raise RuntimeError("production build report drift")
    return audit


def _allowed_entries(root: Path, allowed: set[str]) -> None:
    if root.is_symlink():
        raise RuntimeError(f"structural root symlink rejected: {root}")
    if root.exists():
        unknown = {p.name for p in root.iterdir()} - allowed
        if unknown:
            raise RuntimeError(
                f"unknown structural entries in {root}: {sorted(unknown)}"
            )


def _validate_structure(config: ProductionConfig) -> None:
    _allowed_entries(
        config.output_root,
        {
            "codecs",
            "receipts",
            "tranches",
            "producers",
            "transitions",
            ".staging",
            "producer.json",
            ".lock",
        },
    )
    _allowed_entries(
        config.state_root,
        {
            "checkpoint.json",
            "compression.json",
            "control.json",
            "continuous-authorization.json",
            "governor_snapshots",
            ".lock",
        },
    )
    codecs = config.output_root / "codecs"
    _allowed_entries(codecs, {f"{codec}.zarr" for codec in CODECS})
    for codec in CODECS:
        root = codecs / f"{codec}.zarr"
        if (root / ".zgroup").exists() or (root / ".zattrs").exists():
            raise RuntimeError(
                "production codec root metadata is forbidden pre-publication"
            )
    receipts = config.output_root / "receipts"
    tranches = config.output_root / "tranches"
    if receipts.exists():
        bad = [p.name for p in receipts.iterdir() if not re.fullmatch(r"\d{8}", p.name)]
        if bad:
            raise RuntimeError(f"unknown receipt tranche entries: {bad[:10]}")
    if tranches.exists():
        bad = [
            p.name
            for p in tranches.iterdir()
            if not re.fullmatch(r"\d{8}\.json", p.name)
        ]
        if bad:
            raise RuntimeError(f"unknown tranche record entries: {bad[:10]}")


def _validate_progress_structure(
    config: ProductionConfig, checkpoint: dict[str, Any]
) -> None:
    committed = int(checkpoint["completed_tranches"])
    record_root = config.output_root / "tranches"
    receipt_root = config.output_root / "receipts"
    observed_records = {path.name for path in record_root.iterdir()}
    required_records = {f"{index:08d}.json" for index in range(committed)}
    allowed_records = (
        required_records
        if checkpoint["complete"]
        else required_records | {f"{committed:08d}.json"}
    )
    if (
        not required_records <= observed_records
        or not observed_records <= allowed_records
    ):
        raise RuntimeError("production tranche structural inventory drift")
    observed_receipts = {path.name for path in receipt_root.iterdir()}
    required_receipts = {f"{index:08d}" for index in range(committed)}
    allowed_receipts = (
        required_receipts
        if checkpoint["complete"]
        else required_receipts | {f"{committed:08d}"}
    )
    if (
        not required_receipts <= observed_receipts
        or not observed_receipts <= allowed_receipts
    ):
        raise RuntimeError("production receipt structural inventory drift")
    for path in receipt_root.iterdir():
        if not path.is_dir() or path.is_symlink():
            raise RuntimeError("production receipt tranche root must be a directory")
    for path in record_root.iterdir():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError("production tranche record must be a regular file")


def _producer_path(config: ProductionConfig) -> Path:
    return config.output_root / "producer.json"


def _read_regular_bytes(path: Path, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"{label} missing or unsafe")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise RuntimeError(f"{label} is not a regular file")
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _validate_producer_payload(
    config: ProductionConfig, payload: dict[str, Any]
) -> None:
    required = {
        "format_version",
        "production_id",
        "config_sha256",
        "inventory_digest",
        "software",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != required
        or payload.get("format_version") != PRODUCER_FORMAT
        or payload.get("production_id") != config.production_id
        or payload.get("config_sha256") != config.digest
        or payload.get("inventory_digest") != config.inventory_digest
        or not isinstance(payload.get("software"), dict)
    ):
        raise RuntimeError("production producer identity drift")


def _load_producer_file(
    config: ProductionConfig, path: Path
) -> tuple[dict[str, Any], str, bytes]:
    producer_bytes = _read_regular_bytes(path, "production producer identity")
    try:
        payload = json.loads(producer_bytes)
    except Exception as error:
        raise RuntimeError("production producer identity malformed") from error
    _validate_producer_payload(config, payload)
    return payload, hashlib.sha256(producer_bytes).hexdigest(), producer_bytes


def _load_producer(config: ProductionConfig) -> tuple[dict[str, Any], str]:
    payload, digest, _ = _load_producer_file(config, _producer_path(config))
    if not config.test_mode:
        current = software_identity(require_clean=True)
        if current != payload.get("software"):
            raise RuntimeError(
                "live production checkout/interpreter/dependency identity drift"
            )
    return payload, digest


def _checkpoint_path(config: ProductionConfig) -> Path:
    return config.state_root / "checkpoint.json"


def _initial_checkpoint(config: ProductionConfig) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT,
        "production_id": config.production_id,
        "config_sha256": config.digest,
        "inventory_digest": config.inventory_digest,
        "next_index": 0,
        "completed_tranches": 0,
        "chain_head": ZERO_CHAIN,
        "last_site_key": None,
        "created": 0,
        "skipped": 0,
        "cumulative_errors": 0,
        "complete": False,
        "updated_at": now(),
    }


def _load_checkpoint(config: ProductionConfig) -> dict[str, Any]:
    path = _checkpoint_path(config)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("production checkpoint missing")
    value = json.loads(path.read_text())
    required = {
        "format_version",
        "production_id",
        "config_sha256",
        "inventory_digest",
        "producer_sha256",
        "next_index",
        "completed_tranches",
        "chain_head",
        "last_site_key",
        "created",
        "skipped",
        "cumulative_errors",
        "complete",
        "updated_at",
    }
    if (
        set(value) != required
        or value.get("format_version") != CHECKPOINT_FORMAT
        or value.get("production_id") != config.production_id
        or value.get("config_sha256") != config.digest
        or value.get("inventory_digest") != config.inventory_digest
        or not 0 <= int(value.get("next_index", -1)) <= config.site_count
        or int(value.get("completed_tranches", -1))
        != (int(value.get("next_index", -1)) + config.tranche_size - 1)
        // config.tranche_size
        or int(value.get("created", -1)) + int(value.get("skipped", -1))
        != int(value.get("next_index", -1))
        or not 0 <= int(value.get("cumulative_errors", -1)) <= MAX_CUMULATIVE_ERRORS
        or value.get("complete") is not (value.get("next_index") == config.site_count)
    ):
        raise RuntimeError("production checkpoint identity/count drift")
    return value


def _record_path(config: ProductionConfig, tranche: int) -> Path:
    return config.output_root / "tranches" / f"{tranche:08d}.json"


def _receipt_path(config: ProductionConfig, tranche: int, site: str) -> Path:
    return config.output_root / "receipts" / f"{tranche:08d}" / f"{site}.json"


def _site_path(config: ProductionConfig, codec: str, site: str) -> Path:
    return config.output_root / "codecs" / f"{codec}.zarr" / site


def _rows_from(snapshot: ManifestSnapshot, start: int) -> Iterator[dict[str, Any]]:
    return snapshot.iter_rows(start)


def _rows_slice(
    snapshot: ManifestSnapshot, start: int, count: int
) -> list[dict[str, Any]]:
    return list(islice(snapshot.iter_rows(start), count))


def _source_observation_valid(value: dict[str, Any], uri: str, test_mode: bool) -> bool:
    legacy = {"uri", "size", "etag"}
    extended = legacy | {"version_id", "last_modified"}
    keys = set(value)
    if test_mode:
        allowed = keys == legacy
    else:
        allowed = keys == extended
    return bool(
        allowed
        and value.get("uri") == uri
        and isinstance(value.get("size"), int)
        and value["size"] > 0
        and isinstance(value.get("etag"), str)
        and value["etag"]
        and (
            test_mode
            or (
                isinstance(value.get("version_id"), str)
                and bool(value["version_id"])
                and isinstance(value.get("last_modified"), str)
                and bool(value["last_modified"])
            )
        )
    )


def _validate_site_receipt(
    config: ProductionConfig,
    tranche: int,
    row: dict[str, Any],
    producer_sha256: str,
    expected_hash: str | None = None,
) -> bool:
    path = _receipt_path(config, tranche, row["Metadata_Site_Key"])
    if not path.is_file() or path.is_symlink():
        return False
    if expected_hash is not None and sha256_file(path) != expected_hash:
        raise RuntimeError("committed production receipt hash drift")
    value = json.loads(path.read_text())
    required = {
        "format_version",
        "production_not_published",
        "production_id",
        "tranche",
        "site_key",
        "source_row_sha256",
        "channels",
        "sources",
        "shape",
        "dtype",
        "outputs",
        "codec_json",
        "inventory_digest",
        "config_sha256",
        "producer_sha256",
        "completed_at",
    }
    if (
        set(value) != required
        or value.get("format_version") != RECEIPT_FORMAT
        or value.get("production_not_published") is not True
        or value.get("production_id") != config.production_id
        or value.get("tranche") != tranche
        or value.get("site_key") != row["Metadata_Site_Key"]
        or value.get("source_row_sha256") != row["source_row_sha256"]
        or value.get("channels") != row["channels"]
        or value.get("dtype") != "<u2"
        or value.get("codec_json") != CODECS
        or value.get("inventory_digest") != config.inventory_digest
        or value.get("config_sha256") != config.digest
        or value.get("producer_sha256") != producer_sha256
        or len(value.get("sources", [])) != len(row["urls"])
    ):
        raise RuntimeError("production site receipt identity drift")
    if any(
        not _source_observation_valid(obs, uri, config.test_mode)
        for obs, uri in zip(value["sources"], row["urls"])
    ):
        raise RuntimeError("production source observation drift")
    shape = tuple(value.get("shape", ()))
    if len(shape) != 3 or shape[0] != len(row["channels"]):
        raise RuntimeError("production receipt shape drift")
    if set(value.get("outputs", {})) != set(CODECS):
        raise RuntimeError("production receipt output set drift")
    for codec in CODECS:
        if (
            validate_site(
                _site_path(config, codec, row["Metadata_Site_Key"]), shape, codec
            )
            != value["outputs"][codec]
        ):
            raise RuntimeError("committed production chunk drift")
    return True


def _validate_uncommitted_inventory(
    config: ProductionConfig, tranche: int, rows: list[dict[str, Any]]
) -> None:
    sites = {row["Metadata_Site_Key"] for row in rows}
    receipt_root = config.output_root / "receipts" / f"{tranche:08d}"
    if receipt_root.exists():
        if not receipt_root.is_dir() or receipt_root.is_symlink():
            raise RuntimeError("uncommitted receipt root drift")
        extras = {
            path.name
            for path in receipt_root.iterdir()
            if not path.is_file()
            or path.is_symlink()
            or not path.name.endswith(".json")
            or path.name[:-5] not in sites
        }
        if extras:
            raise RuntimeError(
                f"unknown uncommitted receipt entries: {sorted(extras)[:10]}"
            )
    staging = config.output_root / ".staging"
    for path in list(staging.iterdir()):
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError("unknown production staging entry")
        parts = path.name.rsplit(".", 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{32}", parts[1]):
            raise RuntimeError("malformed production staging entry")
        # Staging is never published identity. A SIGKILL may leave a directory
        # for the last committed or next tranche, so any well-formed staging
        # directory is safe to remove before inspecting durable destinations.
        shutil.rmtree(path)
    fsync_dir(staging)


def _remove_uncommitted(config: ProductionConfig, tranche: int, site: str) -> None:
    for codec in CODECS:
        path = _site_path(config, codec, site)
        if path.is_symlink():
            raise RuntimeError("uncommitted site symlink rejected")
        if path.exists():
            shutil.rmtree(path)
    _receipt_path(config, tranche, site).unlink(missing_ok=True)


def _build_site(
    config: ProductionConfig,
    tranche: int,
    row: dict[str, Any],
    producer_sha256: str,
) -> dict[str, Any]:
    site = row["Metadata_Site_Key"]
    try:
        if _validate_site_receipt(config, tranche, row, producer_sha256):
            return {
                "site": site,
                "status": "skipped",
                "receipt_sha256": sha256_file(_receipt_path(config, tranche, site)),
            }
    except Exception:
        _remove_uncommitted(config, tranche, site)
    receipt = _receipt_path(config, tranche, site)
    if not receipt.exists() and any(
        _site_path(config, codec, site).exists() for codec in CODECS
    ):
        _remove_uncommitted(config, tranche, site)
    staging_root = config.output_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = staging_root / f"{site}.{uuid4().hex}"
    staging.mkdir()
    try:
        stack, observations = decode_stack(
            row, config.test_mode, extended_observation=True
        )
        if len(observations) != len(row["urls"]) or any(
            not _source_observation_valid(observation, uri, config.test_mode)
            for observation, uri in zip(observations, row["urls"])
        ):
            raise RuntimeError("source observations invalid before production write")
        outputs: dict[str, Any] = {}
        staged: dict[str, Path] = {}
        for codec in CODECS:
            staged[codec] = _write_staged(stack, staging, site, codec)
            outputs[codec] = validate_site(staged[codec], tuple(stack.shape), codec)
            _fsync_tree(staged[codec])
        for codec in CODECS:
            target = _site_path(config, codec, site)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise RuntimeError("uncommitted destination appeared concurrently")
            os.replace(staged[codec], target)
            fsync_dir(target.parent)
        FAULT_HOOK("before_site_receipt")
        value = {
            "format_version": RECEIPT_FORMAT,
            "production_not_published": True,
            "production_id": config.production_id,
            "tranche": tranche,
            "site_key": site,
            "source_row_sha256": row["source_row_sha256"],
            "channels": row["channels"],
            "sources": observations,
            "shape": list(stack.shape),
            "dtype": stack.dtype.str,
            "outputs": outputs,
            "codec_json": CODECS,
            "inventory_digest": config.inventory_digest,
            "config_sha256": config.digest,
            "producer_sha256": producer_sha256,
            "completed_at": now(),
        }
        atomic_json(receipt, value)
        FAULT_HOOK("after_site_receipt")
        return {
            "site": site,
            "status": "created",
            "receipt_sha256": sha256_file(receipt),
        }
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _record_digest(value: dict[str, Any]) -> str:
    return digest_json({k: value[k] for k in value if k != "tranche_digest"})


def _load_record(config: ProductionConfig, tranche: int) -> dict[str, Any]:
    path = _record_path(config, tranche)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"committed tranche record missing: {tranche}")
    value = json.loads(path.read_text())
    required = {
        "format_version",
        "production_id",
        "config_sha256",
        "inventory_digest",
        "producer_sha256",
        "tranche",
        "start_index",
        "end_index",
        "site_count",
        "first_site_key",
        "last_site_key",
        "manifest_slice_digest",
        "site_receipt_hash_digest",
        "previous_tranche_digest",
        "created",
        "skipped",
        "completed_at",
        "tranche_digest",
    }
    if set(value) != required or value.get("tranche_digest") != _record_digest(value):
        raise RuntimeError(f"tranche record digest/field drift: {tranche}")
    return value


def _production_control(
    config: ProductionConfig, cumulative_errors: int
) -> dict[str, Any]:
    value = _control(config, cumulative_errors=cumulative_errors)
    try:
        workers = int(value.get("desired_workers", 0))
    except Exception:
        workers = 0
    if value.get("paused") is False and not INITIAL_WORKERS <= workers <= MAX_WORKERS:
        return {"paused": True, "reason": "production worker count must remain 4..16"}
    return value


def _production_task_check(config: ProductionConfig, additional: int) -> None:
    if not config.test_mode:
        assert_runtime_task_ceiling(additional)


def _transition_digest(value: dict[str, Any]) -> str:
    return digest_json({k: value[k] for k in value if k != "transition_digest"})


def _load_transition_plan(
    config: ProductionConfig, current_producer_sha: str
) -> list[dict[str, Any]]:
    root = config.output_root / "transitions"
    if not root.exists():
        return []
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("producer transition root unsafe")
    paths = sorted(root.iterdir())
    if any(not re.fullmatch(r"\d{8}\.json", path.name) for path in paths):
        raise RuntimeError("unknown producer transition entry")
    required = {
        "format_version",
        "production_id",
        "config_sha256",
        "inventory_digest",
        "boundary_tranche",
        "from_producer_sha256",
        "to_producer_sha256",
        "checkpoint_next_index",
        "checkpoint_completed_tranches",
        "checkpoint_chain_head",
        "pre_migration_checkpoint_sha256",
        "one_tranche_acceptance_sha256",
        "migration_acceptance_sha256",
        "successor_software",
        "transitioned_at",
        "transition_digest",
    }
    transitions = []
    preceding_to = None
    preceding_boundary = 0
    seen_producers: set[str] = set()
    for path in paths:
        value = json.loads(_read_regular_bytes(path, "producer transition record"))
        boundary = value.get("boundary_tranche", -1)
        if (
            not isinstance(value, dict)
            or set(value) != required
            or value.get("format_version") != PRODUCER_TRANSITION_FORMAT
            or value.get("production_id") != config.production_id
            or value.get("config_sha256") != config.digest
            or value.get("inventory_digest") != config.inventory_digest
            or not isinstance(boundary, int)
            or boundary <= preceding_boundary
            or path.name != f"{boundary:08d}.json"
            or value.get("checkpoint_completed_tranches") != boundary
            or value.get("checkpoint_next_index")
            != min(boundary * config.tranche_size, config.site_count)
            or value.get("transition_digest") != _transition_digest(value)
            or value.get("from_producer_sha256") == value.get("to_producer_sha256")
            or value.get("to_producer_sha256") in seen_producers
            or (
                preceding_to is not None
                and value.get("from_producer_sha256") != preceding_to
            )
        ):
            raise RuntimeError("producer transition identity/chain drift")
        before = _load_record(config, boundary - 1)
        if before["tranche_digest"] != value.get("checkpoint_chain_head"):
            raise RuntimeError("producer transition checkpoint chain drift")
        producer_values = {}
        for producer_sha in (
            value["from_producer_sha256"],
            value["to_producer_sha256"],
        ):
            history = config.output_root / "producers" / f"{producer_sha}.json"
            producer_value, observed_sha, _ = _load_producer_file(config, history)
            if observed_sha != producer_sha:
                raise RuntimeError("producer history filename/hash drift")
            producer_values[producer_sha] = producer_value
        if (
            producer_values[value["to_producer_sha256"]]["software"]
            != value["successor_software"]
        ):
            raise RuntimeError("producer transition successor software drift")
        seen_producers.add(value["from_producer_sha256"])
        seen_producers.add(value["to_producer_sha256"])
        preceding_to = value["to_producer_sha256"]
        preceding_boundary = boundary
        transitions.append(value)
    if transitions and transitions[-1]["to_producer_sha256"] != current_producer_sha:
        raise RuntimeError("current producer does not equal transition chain head")
    return transitions


def _expected_producer(
    tranche: int, current_producer_sha: str, transitions: list[dict[str, Any]]
) -> str:
    if not transitions:
        return current_producer_sha
    expected = transitions[0]["from_producer_sha256"]
    for transition in transitions:
        if tranche >= transition["boundary_tranche"]:
            expected = transition["to_producer_sha256"]
        else:
            break
    return expected


def _validate_chain(config: ProductionConfig, checkpoint: dict[str, Any]) -> None:
    current_producer_sha = checkpoint["producer_sha256"]
    transitions = _load_transition_plan(config, current_producer_sha)
    previous = ZERO_CHAIN
    next_index = 0
    last = None
    for tranche in range(checkpoint["completed_tranches"]):
        value = _load_record(config, tranche)
        count = min(config.tranche_size, config.site_count - next_index)
        expected_producer = _expected_producer(
            tranche, current_producer_sha, transitions
        )
        if (
            value.get("format_version") != TRANCHE_FORMAT
            or value.get("production_id") != config.production_id
            or value.get("config_sha256") != config.digest
            or value.get("inventory_digest") != config.inventory_digest
            or value.get("tranche") != tranche
            or value.get("start_index") != next_index
            or value.get("end_index") != next_index + count
            or value.get("site_count") != count
            or value.get("previous_tranche_digest") != previous
            or value.get("producer_sha256") != expected_producer
            or value.get("created", -1) + value.get("skipped", -1) != count
        ):
            raise RuntimeError(f"tranche chain/index drift: {tranche}")
        previous = value["tranche_digest"]
        next_index += count
        last = value.get("last_site_key")
    if (
        next_index != checkpoint["next_index"]
        or previous != checkpoint["chain_head"]
        or last != checkpoint["last_site_key"]
    ):
        raise RuntimeError("checkpoint/tranche chain head drift")


def _verify_tranche(
    config: ProductionConfig,
    snapshot: ManifestSnapshot,
    tranche: int,
    producer_sha: str,
) -> dict[str, Any]:
    record = _load_record(config, tranche)
    rows = _rows_slice(snapshot, record["start_index"], record["site_count"])
    if len(rows) != record["site_count"]:
        raise RuntimeError("manifest tranche slice short")
    expected_names = {f"{row['Metadata_Site_Key']}.json" for row in rows}
    receipt_root = config.output_root / "receipts" / f"{tranche:08d}"
    if (
        not receipt_root.is_dir()
        or receipt_root.is_symlink()
        or {path.name for path in receipt_root.iterdir()} != expected_names
    ):
        raise RuntimeError("committed tranche receipt inventory drift")
    row_digest = digest_json([row["source_row_sha256"] for row in rows])
    hashes = []
    for row in rows:
        path = _receipt_path(config, tranche, row["Metadata_Site_Key"])
        digest = sha256_file(path)
        _validate_site_receipt(config, tranche, row, producer_sha, digest)
        hashes.append(digest)
    if (
        row_digest != record["manifest_slice_digest"]
        or digest_json(hashes) != record["site_receipt_hash_digest"]
        or rows[0]["Metadata_Site_Key"] != record["first_site_key"]
        or rows[-1]["Metadata_Site_Key"] != record["last_site_key"]
    ):
        raise RuntimeError("committed tranche manifest/receipt digest drift")
    return {
        "status": "valid",
        "tranche": tranche,
        "sites": len(rows),
        "tranche_digest": record["tranche_digest"],
    }


def verify_tranche(config: ProductionConfig, tranche: int) -> dict[str, Any]:
    with ManifestSnapshot(config.manifest) as snapshot:
        _validate_identity(config, snapshot)
        _, producer_sha = _load_producer(config)
        checkpoint = _load_checkpoint(config)
        _validate_chain(config, checkpoint)
        if not 0 <= tranche < checkpoint["completed_tranches"]:
            raise ValueError("selected tranche is not committed")
        transitions = _load_transition_plan(config, producer_sha)
        return _verify_tranche(
            config,
            snapshot,
            tranche,
            _expected_producer(tranche, producer_sha, transitions),
        )


def _resource_telemetry() -> dict[str, Any]:
    task_count = len(list(Path("/proc/self/task").iterdir()))
    page_size = os.sysconf("SC_PAGE_SIZE")
    rss_bytes = int(Path("/proc/self/statm").read_text().split()[1]) * page_size
    max_rss_bytes = max(
        rss_bytes, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    return {
        "current_tasks": task_count,
        "rss_bytes": rss_bytes,
        "max_rss_bytes": max_rss_bytes,
        "affinity": sorted(os.sched_getaffinity(0)),
    }


class Heartbeat:
    def __init__(self, config: ProductionConfig, fields: dict[str, Any]):
        self.config = config
        self.fields = fields
        self.peak_tasks = 0
        self.terminal = False
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="production-heartbeat", daemon=True
        )

    def write(self, **changes: Any) -> None:
        with self.lock:
            self.fields.update(changes)
            telemetry = _resource_telemetry()
            self.peak_tasks = max(self.peak_tasks, telemetry["current_tasks"])
            self.terminal = self.fields.get("state") in IDLE_TERMINAL_STATES | {"error"}
            atomic_json(
                self.config.state_root / "compression.json",
                {
                    "format_version": STATE_FORMAT,
                    "candidate_id": self.config.production_id,
                    "config_sha256": self.config.digest,
                    "heartbeat_unix": time.time(),
                    **self.fields,
                    **telemetry,
                    "peak_tasks": self.peak_tasks,
                },
            )

    def _run(self) -> None:
        while not self.stop.wait(HEARTBEAT_SECONDS):
            self.write()

    def __enter__(self):
        self.write()
        self.thread.start()
        return self

    def __exit__(self, exc_type, *_args):
        self.stop.set()
        self.thread.join(timeout=HEARTBEAT_SECONDS + 5)
        if not self.terminal:
            self.write(
                state="error" if exc_type else self.fields.get("state", "running")
            )


def _lock(config: ProductionConfig):
    path = config.state_root / ".lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def bootstrap_production(config: ProductionConfig, apply: bool) -> dict[str, Any]:
    with ManifestSnapshot(config.manifest) as snapshot:
        return _bootstrap_with_snapshot(config, snapshot, apply)


def _bootstrap_with_snapshot(
    config: ProductionConfig, snapshot: ManifestSnapshot, apply: bool
) -> dict[str, Any]:
    audit = _validate_identity(config, snapshot)
    if config.state_root.exists() and any(config.state_root.iterdir()):
        raise RuntimeError("production state is already initialized")
    if config.output_root.exists() and any(config.output_root.iterdir()):
        raise RuntimeError("production output is already initialized")
    producer = software_identity(require_clean=not config.test_mode)
    preview = {
        "status": "would-bootstrap",
        "production_id": config.production_id,
        "sites": config.site_count,
        "config_sha256": config.digest,
    }
    if not apply:
        return preview
    for root in (config.output_root, config.state_root):
        root.mkdir(parents=True, exist_ok=False)
    for path in (
        config.output_root / "codecs",
        config.output_root / "receipts",
        config.output_root / "tranches",
        config.output_root / ".staging",
    ):
        path.mkdir()
    for codec in CODECS:
        (config.output_root / "codecs" / f"{codec}.zarr").mkdir()
    producer_value = {
        "format_version": PRODUCER_FORMAT,
        "production_id": config.production_id,
        "config_sha256": config.digest,
        "inventory_digest": config.inventory_digest,
        "software": producer,
    }
    atomic_json(_producer_path(config), producer_value)
    checkpoint = {
        **_initial_checkpoint(config),
        "producer_sha256": sha256_file(_producer_path(config)),
    }
    atomic_json(_checkpoint_path(config), checkpoint)
    atomic_json(
        config.state_root / "control.json",
        {
            "format_version": "full-jump-compression-control-v2",
            "candidate_id": config.production_id,
            "config_sha256": config.digest,
            "paused": True,
            "desired_workers": INITIAL_WORKERS,
            "max_workers": MAX_WORKERS,
            "consecutive_healthy_windows": 0,
            "compression_cpus": list(COMPRESSION_CPUS),
            "acknowledged_error_count": 0,
            "acknowledgement_source": "zero",
            "reasons": ["production bootstrap requires governor evaluation"],
            "observed_at_unix": time.time(),
            "feature_processes_mutated": False,
            "governor_evaluation_required": True,
        },
    )
    telemetry = _resource_telemetry()
    atomic_json(
        config.state_root / "compression.json",
        {
            "format_version": STATE_FORMAT,
            "candidate_id": config.production_id,
            "config_sha256": config.digest,
            "state": "paused",
            "next_index": 0,
            "processed": 0,
            "sites": config.site_count,
            "cumulative_errors": 0,
            "heartbeat_unix": time.time(),
            **telemetry,
            "peak_tasks": telemetry["current_tasks"],
        },
    )
    _validate_structure(config)
    return {
        **preview,
        "status": "bootstrapped",
        "audit_digest": audit["inventory_digest"],
    }


def acknowledge_production_errors(
    config: ProductionConfig, expected: int, apply: bool
) -> dict[str, Any]:
    with ManifestSnapshot(config.manifest) as snapshot:
        return _acknowledge_with_snapshot(config, snapshot, expected, apply)


def _acknowledge_with_snapshot(
    config: ProductionConfig,
    snapshot: ManifestSnapshot,
    expected: int,
    apply: bool,
) -> dict[str, Any]:
    _validate_identity(config, snapshot)
    checkpoint = _load_checkpoint(config)
    if (
        expected != checkpoint["cumulative_errors"]
        or not 0 <= expected <= MAX_CUMULATIVE_ERRORS
    ):
        raise RuntimeError(
            "expected error count does not equal persistent production count"
        )
    result = {"status": "would-acknowledge", "expected_count": expected}
    if not apply:
        return result
    atomic_json(
        config.state_root / "control.json",
        {
            "format_version": "full-jump-compression-control-v2",
            "candidate_id": config.production_id,
            "config_sha256": config.digest,
            "paused": True,
            "desired_workers": INITIAL_WORKERS,
            "max_workers": MAX_WORKERS,
            "consecutive_healthy_windows": 0,
            "compression_cpus": list(COMPRESSION_CPUS),
            "acknowledged_error_count": expected,
            "acknowledgement_source": "explicit-cli" if expected else "zero",
            "reasons": ["error acknowledgement requires governor evaluation"],
            "observed_at_unix": time.time(),
            "feature_processes_mutated": False,
            "governor_evaluation_required": True,
        },
    )
    return {**result, "status": "acknowledged-paused"}


def _continuous_authorization_path(config: ProductionConfig) -> Path:
    return config.state_root / "continuous-authorization.json"


def _strict_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise RuntimeError(f"{label} schema drift")
    return value


def _valid_review(value: Any, label: str) -> dict[str, Any]:
    review = _strict_keys(value, {"identifier", "reviewed_at"}, label)
    if not isinstance(review["identifier"], str) or not review["identifier"].strip():
        raise RuntimeError(f"{label} identifier invalid")
    try:
        parsed = datetime.fromisoformat(review["reviewed_at"].replace("Z", "+00:00"))
    except Exception as error:
        raise RuntimeError(f"{label} timestamp invalid") from error
    if parsed.tzinfo is None:
        raise RuntimeError(f"{label} timestamp must be timezone-aware")
    return review


def _artifact_binding(value: Any, label: str) -> tuple[Path, str, Any]:
    binding = _strict_keys(value, {"path", "sha256"}, label)
    path = Path(binding["path"])
    digest = binding["sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or path.is_symlink()
        or not path.is_file()
        or sha256_file(path) != digest
    ):
        raise RuntimeError(f"{label} binding drift")
    try:
        payload = json.loads(_read_regular_bytes(path, label))
    except Exception as error:
        raise RuntimeError(f"{label} malformed") from error
    return path, digest, payload


def _prevalidate_one_tranche_acceptance(path: Path, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise RuntimeError("one-tranche acceptance receipt binding drift")
    try:
        value = json.loads(_read_regular_bytes(path, "one-tranche acceptance receipt"))
    except Exception as error:
        raise RuntimeError("one-tranche acceptance receipt malformed") from error
    required = {
        "format_version",
        "decision",
        "production_id",
        "config_sha256",
        "inventory_digest",
        "frozen_manifest",
        "checkpoint",
        "tranche0",
        "verification",
        "governor",
        "predecessor_producer",
        "reviews",
        "accepted_at",
    }
    _strict_keys(value, required, "one-tranche acceptance receipt")
    if value["format_version"] != ONE_TRANCHE_ACCEPTANCE_FORMAT:
        raise RuntimeError("one-tranche acceptance format invalid")
    if value["decision"] != "GO":
        raise RuntimeError("one-tranche acceptance decision is not GO")


def _validate_one_tranche_acceptance(
    config: ProductionConfig,
    path: Path,
    expected_sha256: str,
    checkpoint: dict[str, Any],
    predecessor_sha: str,
    predecessor: dict[str, Any],
    accepted_checkpoint_sha256: str,
) -> dict[str, Any]:
    _prevalidate_one_tranche_acceptance(path, expected_sha256)
    try:
        value = json.loads(_read_regular_bytes(path, "one-tranche acceptance receipt"))
    except Exception as error:
        raise RuntimeError("one-tranche acceptance receipt malformed") from error
    required = {
        "format_version",
        "decision",
        "production_id",
        "config_sha256",
        "inventory_digest",
        "frozen_manifest",
        "checkpoint",
        "tranche0",
        "verification",
        "governor",
        "predecessor_producer",
        "reviews",
        "accepted_at",
    }
    _strict_keys(value, required, "one-tranche acceptance receipt")
    frozen = _strict_keys(
        value["frozen_manifest"], {"sha256", "bytes", "site_count"}, "frozen manifest"
    )
    accepted_checkpoint = _strict_keys(
        value["checkpoint"],
        {
            "sha256",
            "next_index",
            "completed_tranches",
            "cumulative_errors",
            "chain_head",
        },
        "accepted checkpoint",
    )
    tranche = _strict_keys(
        value["tranche0"], {"record_sha256", "tranche_digest", "site_count"}, "tranche0"
    )
    predecessor_binding = _strict_keys(
        value["predecessor_producer"], {"sha256", "git_commit"}, "predecessor producer"
    )
    reviews = _strict_keys(value["reviews"], {"code", "science", "ops"}, "reviews")
    identifiers = {
        _valid_review(reviews[name], f"{name} review")["identifier"]
        for name in ("code", "science", "ops")
    }
    if len(identifiers) != 3:
        raise RuntimeError("independent review identifiers must be distinct")
    _valid_review(
        {"identifier": "acceptance", "reviewed_at": value["accepted_at"]},
        "acceptance",
    )
    if (
        value["format_version"] != ONE_TRANCHE_ACCEPTANCE_FORMAT
        or value["decision"] != "GO"
        or value["production_id"] != config.production_id
        or value["config_sha256"] != config.digest
        or value["inventory_digest"] != config.inventory_digest
        or frozen
        != {
            "sha256": config.manifest_sha256,
            "bytes": config.manifest_size,
            "site_count": config.site_count,
        }
        or accepted_checkpoint["sha256"] != accepted_checkpoint_sha256
        or accepted_checkpoint["next_index"] != config.tranche_size
        or accepted_checkpoint["completed_tranches"] != 1
        or accepted_checkpoint["cumulative_errors"] != 0
        or accepted_checkpoint["chain_head"] != AUTHORIZED_FIRST_TRANCHE_DIGEST
        or tranche["record_sha256"] != sha256_file(_record_path(config, 0))
        or tranche["tranche_digest"] != AUTHORIZED_FIRST_TRANCHE_DIGEST
        or tranche["site_count"] != config.tranche_size
        or predecessor_binding["sha256"] != predecessor_sha
        or predecessor_binding["git_commit"]
        != predecessor["software"].get("git_commit")
    ):
        raise RuntimeError("one-tranche acceptance semantic drift")
    if not config.test_mode and (
        predecessor_sha != AUTHORIZED_PREDECESSOR_PRODUCER_SHA256
        or predecessor["software"].get("git_commit") != AUTHORIZED_PREDECESSOR_COMMIT
    ):
        raise RuntimeError("one-tranche predecessor is not accepted 75b1890 identity")
    verification = _strict_keys(
        value["verification"],
        {"artifact", "status", "tranche", "sites", "tranche_digest"},
        "verification",
    )
    _, _, verified_artifact = _artifact_binding(
        verification["artifact"], "tranche verification artifact"
    )
    expected_verification = {
        "status": "valid",
        "tranche": 0,
        "sites": config.tranche_size,
        "tranche_digest": AUTHORIZED_FIRST_TRANCHE_DIGEST,
    }
    if {
        key: verification[key] for key in expected_verification
    } != expected_verification:
        raise RuntimeError("tranche verification receipt semantic drift")
    if not isinstance(verified_artifact, dict) or any(
        verified_artifact.get(key) != expected
        for key, expected in expected_verification.items()
    ):
        raise RuntimeError("tranche verification artifact semantic drift")
    governor = _strict_keys(
        value["governor"],
        {"before", "post", "feature_deltas", "io_pressure"},
        "governor evidence",
    )
    _, _, before = _artifact_binding(governor["before"], "governor before artifact")
    _, _, post = _artifact_binding(governor["post"], "governor post artifact")
    deltas = _strict_keys(
        governor["feature_deltas"], {"MQ", "lossless"}, "feature deltas"
    )
    for codec in ("MQ", "lossless"):
        delta = _strict_keys(
            deltas[codec],
            {"receipt_backed_masks", "canonical_profiles"},
            f"{codec} feature delta",
        )
        try:
            observed_before = before["metrics"]["authoritative_progress"][codec]
            observed_post = post["metrics"]["authoritative_progress"][codec]
        except Exception as error:
            raise RuntimeError("governor progress evidence malformed") from error
        computed = {
            key: observed_post[key] - observed_before[key]
            for key in ("receipt_backed_masks", "canonical_profiles")
        }
        if delta != computed or any(
            not isinstance(v, int) or v <= 0 for v in delta.values()
        ):
            raise RuntimeError("feature progress delta is not explicit and positive")
    pressure = _strict_keys(
        governor["io_pressure"],
        {"before_some_avg10", "after_some_avg10", "max_some_avg10"},
        "I/O pressure evidence",
    )
    if any(
        not isinstance(value, (int, float)) or value != 0 for value in pressure.values()
    ):
        raise RuntimeError("I/O pressure evidence is not zero")
    return value


def _validate_migration_acceptance(
    config: ProductionConfig,
    path: Path,
    expected_sha256: str,
    one_tranche_path: Path,
    one_tranche_sha256: str,
    checkpoint: dict[str, Any],
    predecessor: dict[str, Any],
    predecessor_sha: str,
    successor_software: dict[str, Any],
    accepted_checkpoint_sha256: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected_sha256:
        raise RuntimeError("producer migration acceptance binding drift")
    try:
        value = json.loads(_read_regular_bytes(path, "migration acceptance receipt"))
    except Exception as error:
        raise RuntimeError("producer migration acceptance malformed") from error
    _strict_keys(
        value,
        {
            "format_version",
            "decision",
            "production_id",
            "config_sha256",
            "inventory_digest",
            "checkpoint_sha256",
            "tranche0_record_sha256",
            "tranche0_digest",
            "one_tranche_acceptance",
            "predecessor",
            "successor",
            "review",
            "approved_at",
        },
        "migration acceptance receipt",
    )
    one = _strict_keys(
        value["one_tranche_acceptance"],
        {"path", "sha256"},
        "one-tranche migration binding",
    )
    pred = _strict_keys(
        value["predecessor"], {"producer_sha256", "software"}, "predecessor"
    )
    succ = _strict_keys(value["successor"], {"software"}, "successor")
    _valid_review(value["review"], "migration review")
    _valid_review(
        {"identifier": "approval", "reviewed_at": value["approved_at"]},
        "migration approval",
    )
    if (
        value["format_version"] != MIGRATION_ACCEPTANCE_FORMAT
        or value["decision"] != "GO"
        or value["production_id"] != config.production_id
        or value["config_sha256"] != config.digest
        or value["inventory_digest"] != config.inventory_digest
        or value["checkpoint_sha256"] != accepted_checkpoint_sha256
        or value["tranche0_record_sha256"] != sha256_file(_record_path(config, 0))
        or value["tranche0_digest"] != AUTHORIZED_FIRST_TRANCHE_DIGEST
        or one
        != {"path": str(one_tranche_path.resolve()), "sha256": one_tranche_sha256}
        or pred
        != {"producer_sha256": predecessor_sha, "software": predecessor["software"]}
        or succ != {"software": successor_software}
        or checkpoint["next_index"] != config.tranche_size
        or checkpoint["completed_tranches"] != 1
        or checkpoint["chain_head"] != AUTHORIZED_FIRST_TRANCHE_DIGEST
        or checkpoint["cumulative_errors"] != 0
    ):
        raise RuntimeError("producer migration acceptance semantic drift")
    return value


def _atomic_json_create(
    path: Path, value: dict[str, Any], label: str = "continuous authorization"
) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(f"{label} already exists") from error
        fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable_bytes(path: Path, encoded: bytes, label: str) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or _read_regular_bytes(path, label) != encoded:
            raise RuntimeError(f"existing {label} drift")
        return
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(f"{label} already exists") from error
        fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable_json(path: Path, value: dict[str, Any], label: str) -> None:
    encoded = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    _write_immutable_bytes(path, encoded, label)


def _normalize_migration_telemetry(
    config: ProductionConfig,
    checkpoint: dict[str, Any],
    *,
    state: str = "session-complete",
    reason: str = "producer migration requires governor evaluation",
) -> None:
    telemetry = _resource_telemetry()
    atomic_json(
        config.state_root / "compression.json",
        {
            "format_version": STATE_FORMAT,
            "candidate_id": config.production_id,
            "config_sha256": config.digest,
            "state": state,
            "next_index": checkpoint["next_index"],
            "processed": checkpoint["next_index"],
            "sites": config.site_count,
            "cumulative_errors": checkpoint["cumulative_errors"],
            "heartbeat_unix": time.time(),
            **telemetry,
            "peak_tasks": telemetry["current_tasks"],
        },
    )
    atomic_json(
        config.state_root / "control.json",
        {
            "format_version": "full-jump-compression-control-v2",
            "candidate_id": config.production_id,
            "config_sha256": config.digest,
            "paused": True,
            "desired_workers": INITIAL_WORKERS,
            "max_workers": MAX_WORKERS,
            "consecutive_healthy_windows": 0,
            "compression_cpus": list(COMPRESSION_CPUS),
            "acknowledged_error_count": 0,
            "acknowledgement_source": "zero",
            "reasons": [reason],
            "observed_at_unix": time.time(),
            "feature_processes_mutated": False,
            "governor_evaluation_required": True,
        },
    )


def migrate_producer(
    config: ProductionConfig,
    one_tranche_acceptance: Path,
    one_tranche_acceptance_sha256: str,
    migration_acceptance: Path,
    migration_acceptance_sha256: str,
    apply: bool,
) -> dict[str, Any]:
    successor_software = software_identity(require_clean=not config.test_mode)
    with ManifestSnapshot(config.manifest) as snapshot:
        _validate_identity(config, snapshot)
        with _lock(config):
            try:
                return _migrate_producer_locked(
                    config,
                    snapshot,
                    one_tranche_acceptance,
                    one_tranche_acceptance_sha256,
                    migration_acceptance,
                    migration_acceptance_sha256,
                    successor_software,
                    apply,
                )
            except Exception:
                # The lock proves no bounded or continuous compressor is active. Keep
                # a safe paused control after any fail-closed migration preflight/fault.
                if apply and _checkpoint_path(config).is_file():
                    try:
                        checkpoint = _load_checkpoint(config)
                        _normalize_migration_telemetry(
                            config,
                            checkpoint,
                            state="error",
                            reason="producer migration failed; review and retry required",
                        )
                    except Exception:
                        pass
                raise


def _migrate_producer_locked(
    config: ProductionConfig,
    snapshot: ManifestSnapshot,
    one_tranche_acceptance: Path,
    one_tranche_acceptance_sha256: str,
    migration_acceptance: Path,
    migration_acceptance_sha256: str,
    successor_software: dict[str, Any],
    apply: bool,
) -> dict[str, Any]:
    if _continuous_authorization_path(config).exists():
        raise RuntimeError(
            "producer migration forbidden after continuous authorization"
        )
    current, current_sha, current_bytes = _load_producer_file(
        config, _producer_path(config)
    )
    checkpoint = _load_checkpoint(config)
    transition_path = config.output_root / "transitions" / "00000001.json"
    existing_transition = None
    if transition_path.exists():
        existing_transition = json.loads(
            _read_regular_bytes(transition_path, "producer transition record")
        )
        predecessor_sha = existing_transition.get("from_producer_sha256")
        accepted_checkpoint_sha256 = existing_transition.get(
            "pre_migration_checkpoint_sha256"
        )
    else:
        predecessor_sha = checkpoint["producer_sha256"]
        accepted_checkpoint_sha256 = sha256_file(_checkpoint_path(config))
    predecessor_path = config.output_root / "producers" / f"{predecessor_sha}.json"
    if predecessor_path.exists():
        predecessor, observed_predecessor_sha, predecessor_bytes = _load_producer_file(
            config, predecessor_path
        )
    elif current_sha == predecessor_sha:
        predecessor, observed_predecessor_sha, predecessor_bytes = (
            current,
            current_sha,
            current_bytes,
        )
    else:
        raise RuntimeError("predecessor producer history missing")
    if observed_predecessor_sha != predecessor_sha:
        raise RuntimeError("predecessor producer hash drift")
    if not config.test_mode and (
        predecessor_sha != AUTHORIZED_PREDECESSOR_PRODUCER_SHA256
        or predecessor["software"].get("git_commit") != AUTHORIZED_PREDECESSOR_COMMIT
    ):
        raise RuntimeError("predecessor producer is not the accepted 75b1890 identity")
    successor_value = {
        "format_version": PRODUCER_FORMAT,
        "production_id": config.production_id,
        "config_sha256": config.digest,
        "inventory_digest": config.inventory_digest,
        "software": successor_software,
    }
    successor_bytes = (
        json.dumps(successor_value, sort_keys=True, indent=2) + "\n"
    ).encode()
    successor_sha = hashlib.sha256(successor_bytes).hexdigest()
    if current_sha not in {predecessor_sha, successor_sha}:
        raise RuntimeError(
            "current producer is neither migration predecessor nor successor"
        )
    if checkpoint["producer_sha256"] not in {predecessor_sha, successor_sha}:
        raise RuntimeError("checkpoint producer is neither predecessor nor successor")
    # Fully authenticate the accepted old chain with its original producer, even on
    # convergence runs where current producer/checkpoint already name the successor.
    if (
        checkpoint["next_index"] != config.tranche_size
        or checkpoint["completed_tranches"] != 1
        or checkpoint["cumulative_errors"] != 0
        or checkpoint["chain_head"] != AUTHORIZED_FIRST_TRANCHE_DIGEST
        or checkpoint["complete"]
    ):
        raise RuntimeError(
            "checkpoint is not the accepted one-tranche migration boundary"
        )
    record = _load_record(config, 0)
    if record["producer_sha256"] != predecessor_sha:
        raise RuntimeError("tranche0 predecessor producer drift")
    verified = _verify_tranche(config, snapshot, 0, predecessor_sha)
    if verified["tranche_digest"] != AUTHORIZED_FIRST_TRANCHE_DIGEST:
        raise RuntimeError("tranche0 digest drift")
    one = _validate_one_tranche_acceptance(
        config,
        one_tranche_acceptance,
        one_tranche_acceptance_sha256,
        checkpoint,
        predecessor_sha,
        predecessor,
        accepted_checkpoint_sha256,
    )
    migration = _validate_migration_acceptance(
        config,
        migration_acceptance,
        migration_acceptance_sha256,
        one_tranche_acceptance,
        one_tranche_acceptance_sha256,
        checkpoint,
        predecessor,
        predecessor_sha,
        successor_software,
        accepted_checkpoint_sha256,
    )
    transition = {
        "format_version": PRODUCER_TRANSITION_FORMAT,
        "production_id": config.production_id,
        "config_sha256": config.digest,
        "inventory_digest": config.inventory_digest,
        "boundary_tranche": 1,
        "from_producer_sha256": predecessor_sha,
        "to_producer_sha256": successor_sha,
        "checkpoint_next_index": config.tranche_size,
        "checkpoint_completed_tranches": 1,
        "checkpoint_chain_head": AUTHORIZED_FIRST_TRANCHE_DIGEST,
        "pre_migration_checkpoint_sha256": accepted_checkpoint_sha256,
        "one_tranche_acceptance_sha256": one_tranche_acceptance_sha256,
        "migration_acceptance_sha256": migration_acceptance_sha256,
        "successor_software": successor_software,
        "transitioned_at": migration["approved_at"],
    }
    transition["transition_digest"] = _transition_digest(transition)
    result = {
        "status": "would-migrate-producer",
        "config_sha256": config.digest,
        "boundary_tranche": 1,
        "from_producer_sha256": predecessor_sha,
        "to_producer_sha256": successor_sha,
        "transition_digest": transition["transition_digest"],
        "one_tranche_decision": one["decision"],
    }
    if not apply:
        return result
    producers = config.output_root / "producers"
    transitions = config.output_root / "transitions"
    producers.mkdir(exist_ok=True)
    transitions.mkdir(exist_ok=True)
    _write_immutable_bytes(
        predecessor_path, predecessor_bytes, "predecessor producer history"
    )
    successor_path = producers / f"{successor_sha}.json"
    _write_immutable_json(successor_path, successor_value, "successor producer history")
    FAULT_HOOK("after_migration_history")
    _write_immutable_json(transition_path, transition, "producer transition")
    FAULT_HOOK("after_migration_transition")
    atomic_json(_producer_path(config), successor_value)
    FAULT_HOOK("after_migration_current_producer")
    if checkpoint["producer_sha256"] != successor_sha:
        checkpoint = {
            **checkpoint,
            "producer_sha256": successor_sha,
            "updated_at": now(),
        }
        atomic_json(_checkpoint_path(config), checkpoint)
    FAULT_HOOK("after_migration_checkpoint")
    _normalize_migration_telemetry(config, checkpoint)
    _validate_structure(config)
    _validate_chain(config, checkpoint)
    return {**result, "status": "producer-migrated"}


def authorize_continuous(
    config: ProductionConfig,
    acceptance_receipt: Path,
    acceptance_receipt_sha256: str,
    apply: bool,
) -> dict[str, Any]:
    _prevalidate_one_tranche_acceptance(acceptance_receipt, acceptance_receipt_sha256)
    with ManifestSnapshot(config.manifest) as snapshot:
        _validate_identity(config, snapshot)
        with _lock(config):
            try:
                return _authorize_continuous_locked(
                    config,
                    snapshot,
                    acceptance_receipt,
                    acceptance_receipt_sha256,
                    apply,
                )
            except Exception:
                if apply and _checkpoint_path(config).is_file():
                    try:
                        checkpoint = _load_checkpoint(config)
                        _normalize_migration_telemetry(
                            config,
                            checkpoint,
                            state="error",
                            reason="continuous authorization failed; review required",
                        )
                    except Exception:
                        pass
                raise


def _authorize_continuous_locked(
    config: ProductionConfig,
    snapshot: ManifestSnapshot,
    acceptance_receipt: Path,
    acceptance_receipt_sha256: str,
    apply: bool,
) -> dict[str, Any]:
    marker = _continuous_authorization_path(config)
    if marker.exists() or marker.is_symlink():
        raise RuntimeError("continuous authorization already exists")
    if (
        not acceptance_receipt.is_file()
        or acceptance_receipt.is_symlink()
        or sha256_file(acceptance_receipt) != acceptance_receipt_sha256
    ):
        raise RuntimeError("one-tranche acceptance receipt binding drift")
    _, producer_sha = _load_producer(config)
    checkpoint = _load_checkpoint(config)
    _validate_progress_structure(config, checkpoint)
    _validate_chain(config, checkpoint)
    transitions = _load_transition_plan(config, producer_sha)
    if len(transitions) != 1 or transitions[0]["boundary_tranche"] != 1:
        raise RuntimeError("exact applied producer migration transition required")
    transition = transitions[0]
    predecessor_sha = transition["from_producer_sha256"]
    predecessor, observed_predecessor_sha, _ = _load_producer_file(
        config, config.output_root / "producers" / f"{predecessor_sha}.json"
    )
    if observed_predecessor_sha != predecessor_sha:
        raise RuntimeError("predecessor producer history drift")
    accepted = _validate_one_tranche_acceptance(
        config,
        acceptance_receipt,
        acceptance_receipt_sha256,
        checkpoint,
        predecessor_sha,
        predecessor,
        transition["pre_migration_checkpoint_sha256"],
    )
    if (
        checkpoint["next_index"] != config.tranche_size
        or checkpoint["completed_tranches"] != 1
        or checkpoint["cumulative_errors"] != 0
        or checkpoint["complete"]
        or checkpoint["chain_head"] != AUTHORIZED_FIRST_TRANCHE_DIGEST
    ):
        raise RuntimeError("checkpoint is not the accepted one-tranche gate")
    verified = _verify_tranche(config, snapshot, 0, predecessor_sha)
    if verified["tranche_digest"] != AUTHORIZED_FIRST_TRANCHE_DIGEST:
        raise RuntimeError("accepted first-tranche digest drift")
    value = {
        "format_version": CONTINUOUS_AUTH_FORMAT,
        "production_id": config.production_id,
        "config_sha256": config.digest,
        "inventory_digest": config.inventory_digest,
        "producer_sha256": producer_sha,
        "authorized_next_index": config.tranche_size,
        "authorized_completed_tranches": 1,
        "authorized_chain_head": AUTHORIZED_FIRST_TRANCHE_DIGEST,
        "tranche": 0,
        "tranche_digest": AUTHORIZED_FIRST_TRANCHE_DIGEST,
        "acceptance_receipt": str(acceptance_receipt.resolve()),
        "acceptance_receipt_sha256": acceptance_receipt_sha256,
        "transition_digest": transition["transition_digest"],
        "migration_acceptance_sha256": transition["migration_acceptance_sha256"],
        "authorized_at": now(),
    }
    if accepted["decision"] != "GO":
        raise RuntimeError("one-tranche acceptance decision is not GO")
    result = {"status": "would-authorize-continuous", **value}
    if not apply:
        return result
    # Pause first: a crash may leave no marker, but can never leave a new marker
    # paired with the pre-authorization unpaused control.
    atomic_json(
        config.state_root / "control.json",
        {
            "format_version": "full-jump-compression-control-v2",
            "candidate_id": config.production_id,
            "config_sha256": config.digest,
            "paused": True,
            "desired_workers": INITIAL_WORKERS,
            "max_workers": MAX_WORKERS,
            "consecutive_healthy_windows": 0,
            "compression_cpus": list(COMPRESSION_CPUS),
            "acknowledged_error_count": 0,
            "acknowledgement_source": "zero",
            "reasons": ["continuous authorization requires governor evaluation"],
            "observed_at_unix": time.time(),
            "feature_processes_mutated": False,
            "governor_evaluation_required": True,
        },
    )
    _atomic_json_create(marker, value)
    return {**result, "status": "authorized-continuous"}


def _validate_continuous_authorization(
    config: ProductionConfig, checkpoint: dict[str, Any], producer_sha: str
) -> dict[str, Any]:
    path = _continuous_authorization_path(config)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("continuous authorization missing")
    value = json.loads(path.read_text())
    required = {
        "format_version",
        "production_id",
        "config_sha256",
        "inventory_digest",
        "producer_sha256",
        "authorized_next_index",
        "authorized_completed_tranches",
        "authorized_chain_head",
        "tranche",
        "tranche_digest",
        "acceptance_receipt",
        "acceptance_receipt_sha256",
        "transition_digest",
        "migration_acceptance_sha256",
        "authorized_at",
    }
    if (
        set(value) != required
        or value.get("format_version") != CONTINUOUS_AUTH_FORMAT
        or value.get("production_id") != config.production_id
        or value.get("config_sha256") != config.digest
        or value.get("inventory_digest") != config.inventory_digest
        or value.get("producer_sha256") != producer_sha
        or value.get("authorized_next_index") != config.tranche_size
        or value.get("authorized_completed_tranches") != 1
        or value.get("authorized_chain_head") != AUTHORIZED_FIRST_TRANCHE_DIGEST
        or value.get("tranche") != 0
        or value.get("tranche_digest") != AUTHORIZED_FIRST_TRANCHE_DIGEST
        or checkpoint["next_index"] < value.get("authorized_next_index", -1)
        or checkpoint["completed_tranches"] < 1
    ):
        raise RuntimeError("continuous authorization identity/checkpoint drift")
    transitions = _load_transition_plan(config, producer_sha)
    if (
        len(transitions) != 1
        or transitions[0]["boundary_tranche"] != 1
        or transitions[0]["transition_digest"] != value["transition_digest"]
        or transitions[0]["migration_acceptance_sha256"]
        != value["migration_acceptance_sha256"]
    ):
        raise RuntimeError("continuous producer transition drift")
    receipt = Path(value["acceptance_receipt"])
    predecessor_sha = transitions[0]["from_producer_sha256"]
    predecessor, observed_sha, _ = _load_producer_file(
        config, config.output_root / "producers" / f"{predecessor_sha}.json"
    )
    if observed_sha != predecessor_sha:
        raise RuntimeError("continuous predecessor producer drift")
    _validate_one_tranche_acceptance(
        config,
        receipt,
        value["acceptance_receipt_sha256"],
        checkpoint,
        predecessor_sha,
        predecessor,
        transitions[0]["pre_migration_checkpoint_sha256"],
    )
    first = _load_record(config, 0)
    if first["tranche_digest"] != AUTHORIZED_FIRST_TRANCHE_DIGEST:
        raise RuntimeError("continuous authorized chain ancestor drift")
    return value


def run_production(
    config: ProductionConfig,
    max_tranches: int | None,
    apply: bool,
    *,
    continuous: bool = False,
) -> dict[str, Any]:
    if continuous == (max_tranches is not None):
        raise ValueError("select exactly one of --continuous or --max-tranches")
    if not continuous and max_tranches != 1:
        raise ValueError("bounded --max-tranches must equal 1")
    with ManifestSnapshot(config.manifest) as snapshot:
        return _run_with_snapshot(config, snapshot, max_tranches, apply, continuous)


def _run_with_snapshot(
    config: ProductionConfig,
    snapshot: ManifestSnapshot,
    max_tranches: int | None,
    apply: bool,
    continuous: bool = False,
) -> dict[str, Any]:
    _validate_identity(config, snapshot)
    if not apply and not continuous:
        return {
            "status": "would-run",
            "max_tranches": max_tranches,
            "config_sha256": config.digest,
        }
    if not config.test_mode:
        assert_runtime_task_ceiling()
    _validate_structure(config)
    with _lock(config):
        _, producer_sha = _load_producer(config)
        checkpoint = _load_checkpoint(config)
        _validate_progress_structure(config, checkpoint)
        if checkpoint.get("producer_sha256") != producer_sha:
            raise RuntimeError("checkpoint producer binding drift")
        _validate_chain(config, checkpoint)
        if continuous:
            _validate_continuous_authorization(config, checkpoint, producer_sha)
            if not apply:
                return {
                    "status": "would-run-continuous",
                    "config_sha256": config.digest,
                    "next_index": checkpoint["next_index"],
                }
        if checkpoint["completed_tranches"]:
            last_tranche = checkpoint["completed_tranches"] - 1
            transitions = _load_transition_plan(config, producer_sha)
            _verify_tranche(
                config,
                snapshot,
                last_tranche,
                _expected_producer(last_tranche, producer_sha, transitions),
            )
        # A durable record written before a crash is adopted only after full validation.
        # Adoption consumes this invocation's one-tranche allowance: a restart must not
        # both adopt tranche N and create tranche N+1.
        committed = 0
        ahead = _record_path(config, checkpoint["completed_tranches"])
        if ahead.exists():
            tranche = checkpoint["completed_tranches"]
            record = _load_record(config, tranche)
            expected_count = min(
                config.tranche_size, config.site_count - checkpoint["next_index"]
            )
            if (
                record["production_id"] != config.production_id
                or record["config_sha256"] != config.digest
                or record["inventory_digest"] != config.inventory_digest
                or record["producer_sha256"] != producer_sha
                or record["tranche"] != tranche
                or record["start_index"] != checkpoint["next_index"]
                or record["end_index"] != checkpoint["next_index"] + expected_count
                or record["site_count"] != expected_count
                or record["previous_tranche_digest"] != checkpoint["chain_head"]
                or record["created"] + record["skipped"] != expected_count
            ):
                raise RuntimeError("ahead tranche identity/chain drift")
            _verify_tranche(config, snapshot, tranche, producer_sha)
            checkpoint = {
                **checkpoint,
                "next_index": record["end_index"],
                "completed_tranches": checkpoint["completed_tranches"] + 1,
                "chain_head": record["tranche_digest"],
                "last_site_key": record["last_site_key"],
                "created": checkpoint["created"] + record["created"],
                "skipped": checkpoint["skipped"] + record["skipped"],
                "complete": record["end_index"] == config.site_count,
                "updated_at": now(),
            }
            atomic_json(_checkpoint_path(config), checkpoint)
            committed = 1
        control = _production_control(config, checkpoint["cumulative_errors"])
        heartbeat_fields = {
            "state": "running",
            "next_index": checkpoint["next_index"],
            "processed": checkpoint["next_index"],
            "sites": config.site_count,
            "cumulative_errors": checkpoint["cumulative_errors"],
        }
        if control.get("paused") is not False:
            Heartbeat(config, heartbeat_fields).write(state="paused")
            return {
                "status": "paused",
                "reason": control.get("reason", control.get("reasons")),
                "next_index": checkpoint["next_index"],
            }
        iterator = _rows_from(snapshot, checkpoint["next_index"])
        heartbeat_fields = {
            "state": "running",
            "next_index": checkpoint["next_index"],
            "processed": checkpoint["next_index"],
            "sites": config.site_count,
            "cumulative_errors": checkpoint["cumulative_errors"],
        }
        with Heartbeat(config, heartbeat_fields) as heartbeat:
            while checkpoint["next_index"] < config.site_count and (
                continuous or committed < (max_tranches or 0)
            ):
                if STOP.is_set():
                    heartbeat.write(state="stopped")
                    return {
                        "status": "stopped",
                        "next_index": checkpoint["next_index"],
                        "committed_tranches": committed,
                    }
                between_control = _production_control(
                    config, checkpoint["cumulative_errors"]
                )
                if between_control.get("paused") is not False:
                    heartbeat.write(state="paused")
                    return {
                        "status": "paused",
                        "next_index": checkpoint["next_index"],
                        "committed_tranches": committed,
                    }
                tranche = checkpoint["completed_tranches"]
                rows = list(islice(iterator, config.tranche_size))
                expected = min(
                    config.tranche_size, config.site_count - checkpoint["next_index"]
                )
                if len(rows) != expected:
                    raise RuntimeError(
                        "production manifest ended before declared site count"
                    )
                _validate_uncommitted_inventory(config, tranche, rows)
                results = []
                cursor = 0
                try:
                    allocation = _production_control(
                        config, checkpoint["cumulative_errors"]
                    )
                    if allocation.get("paused") is not False:
                        heartbeat.write(state="paused")
                        return {
                            "status": "paused",
                            "next_index": checkpoint["next_index"],
                            "committed_tranches": committed,
                        }
                    workers = int(allocation["desired_workers"])
                    _production_task_check(config, workers)
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        while cursor < len(rows):
                            if STOP.is_set():
                                heartbeat.write(state="stopped")
                                return {
                                    "status": "stopped",
                                    "next_index": checkpoint["next_index"],
                                    "committed_tranches": committed,
                                }
                            control = _production_control(
                                config, checkpoint["cumulative_errors"]
                            )
                            if control.get("paused") is not False:
                                heartbeat.write(state="paused")
                                return {
                                    "status": "paused",
                                    "next_index": checkpoint["next_index"],
                                    "committed_tranches": committed,
                                }
                            batch = rows[cursor : cursor + workers]
                            results.extend(
                                pool.map(
                                    lambda row: _build_site(
                                        config, tranche, row, producer_sha
                                    ),
                                    batch,
                                )
                            )
                            cursor += len(batch)
                            _production_task_check(config, 0)
                            heartbeat.write()
                    FAULT_HOOK("before_tranche_validation")
                    expected_hashes = [result["receipt_sha256"] for result in results]
                    receipt_hashes = []
                    for row, expected_hash in zip(rows, expected_hashes):
                        receipt = _receipt_path(
                            config, tranche, row["Metadata_Site_Key"]
                        )
                        observed_hash = sha256_file(receipt)
                        if observed_hash != expected_hash:
                            raise RuntimeError(
                                "ordered production receipt hash changed before commit"
                            )
                        _validate_site_receipt(
                            config,
                            tranche,
                            row,
                            producer_sha,
                            observed_hash,
                        )
                        receipt_hashes.append(observed_hash)
                    if receipt_hashes != expected_hashes or len(receipt_hashes) != len(
                        rows
                    ):
                        raise RuntimeError("ordered production receipt list drift")
                    record = {
                        "format_version": TRANCHE_FORMAT,
                        "production_id": config.production_id,
                        "config_sha256": config.digest,
                        "inventory_digest": config.inventory_digest,
                        "producer_sha256": producer_sha,
                        "tranche": tranche,
                        "start_index": checkpoint["next_index"],
                        "end_index": checkpoint["next_index"] + len(rows),
                        "site_count": len(rows),
                        "first_site_key": rows[0]["Metadata_Site_Key"],
                        "last_site_key": rows[-1]["Metadata_Site_Key"],
                        "manifest_slice_digest": digest_json(
                            [row["source_row_sha256"] for row in rows]
                        ),
                        "site_receipt_hash_digest": digest_json(receipt_hashes),
                        "previous_tranche_digest": checkpoint["chain_head"],
                        "created": sum(r["status"] == "created" for r in results),
                        "skipped": sum(r["status"] == "skipped" for r in results),
                        "completed_at": now(),
                    }
                    record["tranche_digest"] = _record_digest(record)
                    atomic_json(_record_path(config, tranche), record)
                    FAULT_HOOK("after_tranche_record")
                    checkpoint = {
                        **checkpoint,
                        "next_index": record["end_index"],
                        "completed_tranches": tranche + 1,
                        "chain_head": record["tranche_digest"],
                        "last_site_key": record["last_site_key"],
                        "created": checkpoint["created"] + record["created"],
                        "skipped": checkpoint["skipped"] + record["skipped"],
                        "complete": record["end_index"] == config.site_count,
                        "updated_at": now(),
                    }
                    atomic_json(_checkpoint_path(config), checkpoint)
                    committed += 1
                    heartbeat.write(
                        next_index=checkpoint["next_index"],
                        processed=checkpoint["next_index"],
                        state="complete" if checkpoint["complete"] else "running",
                    )
                except Exception:
                    checkpoint["cumulative_errors"] += 1
                    checkpoint["updated_at"] = now()
                    atomic_json(_checkpoint_path(config), checkpoint)
                    heartbeat.write(
                        state="error", cumulative_errors=checkpoint["cumulative_errors"]
                    )
                    raise
            terminal = "complete" if checkpoint["complete"] else "session-complete"
            heartbeat.write(
                state=terminal,
                next_index=checkpoint["next_index"],
                processed=checkpoint["next_index"],
            )
            return {
                "status": terminal,
                "next_index": checkpoint["next_index"],
                "committed_tranches": committed,
                "completed_tranches": checkpoint["completed_tranches"],
            }


def finalize_validation(config: ProductionConfig) -> dict[str, Any]:
    with ManifestSnapshot(config.manifest) as snapshot:
        _validate_identity(config, snapshot)
        checkpoint = _load_checkpoint(config)
        _validate_progress_structure(config, checkpoint)
        if (
            checkpoint.get("complete") is not True
            or checkpoint.get("next_index") != config.site_count
        ):
            raise RuntimeError("production output is not complete")
        _validate_chain(config, checkpoint)
        _, producer_sha = _load_producer(config)
        codec_counts = {}
        for codec in CODECS:
            root = config.output_root / "codecs" / f"{codec}.zarr"
            count = 0
            for entry in root.iterdir():
                if (
                    entry.name in {".zgroup", ".zattrs"}
                    or entry.is_symlink()
                    or not entry.is_dir()
                ):
                    raise RuntimeError("unsafe/unknown flat codec-root entry")
                count += 1
            if count != config.site_count:
                raise RuntimeError(f"flat codec-root site count drift: {codec} {count}")
            codec_counts[codec] = count
        staging = config.output_root / ".staging"
        if any(staging.iterdir()):
            raise RuntimeError("final production staging root is not empty")
        transitions = _load_transition_plan(config, producer_sha)
        for tranche in range(checkpoint["completed_tranches"]):
            _verify_tranche(
                config,
                snapshot,
                tranche,
                _expected_producer(tranche, producer_sha, transitions),
            )
        return {
            "status": "exhaustively-valid",
            "sites": config.site_count,
            "codec_site_counts": codec_counts,
            "tranches": checkpoint["completed_tranches"],
            "chain_head": checkpoint["chain_head"],
            "production_not_published": True,
        }


def production_status(config: ProductionConfig) -> dict[str, Any]:
    with ManifestSnapshot(config.manifest) as snapshot:
        _validate_identity(config, snapshot)
        result = {
            "production_id": config.production_id,
            "config_sha256": config.digest,
        }
        for name in (
            "checkpoint.json",
            "compression.json",
            "control.json",
            "continuous-authorization.json",
        ):
            path = config.state_root / name
            result[name] = json.loads(path.read_text()) if path.is_file() else None
        return result
