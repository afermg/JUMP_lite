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
import shutil
import stat
import threading
import time
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
        {"codecs", "receipts", "tranches", ".staging", "producer.json", ".lock"},
    )
    _allowed_entries(
        config.state_root,
        {
            "checkpoint.json",
            "compression.json",
            "control.json",
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


def _load_producer(config: ProductionConfig) -> tuple[dict[str, Any], str]:
    path = _producer_path(config)
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("production producer identity missing")
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise RuntimeError("production producer identity is not a regular file")
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as handle:
            producer_bytes = handle.read()
    finally:
        os.close(descriptor)
    payload = json.loads(producer_bytes)
    required = {
        "format_version",
        "production_id",
        "config_sha256",
        "inventory_digest",
        "software",
    }
    if (
        set(payload) != required
        or payload.get("format_version") != PRODUCER_FORMAT
        or payload.get("production_id") != config.production_id
        or payload.get("config_sha256") != config.digest
        or payload.get("inventory_digest") != config.inventory_digest
    ):
        raise RuntimeError("production producer identity drift")
    if not config.test_mode:
        current = software_identity(require_clean=True)
        if current != payload.get("software"):
            raise RuntimeError(
                "live production checkout/interpreter/dependency identity drift"
            )
    return payload, hashlib.sha256(producer_bytes).hexdigest()


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


def _validate_chain(config: ProductionConfig, checkpoint: dict[str, Any]) -> None:
    previous = ZERO_CHAIN
    next_index = 0
    last = None
    for tranche in range(checkpoint["completed_tranches"]):
        value = _load_record(config, tranche)
        count = min(config.tranche_size, config.site_count - next_index)
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
            or value.get("producer_sha256") != checkpoint.get("producer_sha256")
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
        return _verify_tranche(config, snapshot, tranche, producer_sha)


class Heartbeat:
    def __init__(self, config: ProductionConfig, fields: dict[str, Any]):
        self.config = config
        self.fields = fields
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="production-heartbeat", daemon=True
        )

    def write(self, **changes: Any) -> None:
        with self.lock:
            self.fields.update(changes)
            atomic_json(
                self.config.state_root / "compression.json",
                {
                    "format_version": STATE_FORMAT,
                    "candidate_id": self.config.production_id,
                    "config_sha256": self.config.digest,
                    "heartbeat_unix": time.time(),
                    **self.fields,
                },
            )

    def _run(self) -> None:
        while not self.stop.wait(HEARTBEAT_SECONDS):
            self.write()

    def __enter__(self):
        self.write()
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.stop.set()
        self.thread.join(timeout=HEARTBEAT_SECONDS + 5)
        self.write()


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


def run_production(
    config: ProductionConfig, max_tranches: int, apply: bool
) -> dict[str, Any]:
    if max_tranches != 1:
        raise ValueError(
            "--max-tranches must equal 1 until a reviewed authorization marker exists"
        )
    with ManifestSnapshot(config.manifest) as snapshot:
        return _run_with_snapshot(config, snapshot, max_tranches, apply)


def _run_with_snapshot(
    config: ProductionConfig,
    snapshot: ManifestSnapshot,
    max_tranches: int,
    apply: bool,
) -> dict[str, Any]:
    _validate_identity(config, snapshot)
    if not apply:
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
        if checkpoint["completed_tranches"]:
            _verify_tranche(
                config,
                snapshot,
                checkpoint["completed_tranches"] - 1,
                producer_sha,
            )
        # A durable record written before a crash is adopted only after full validation.
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
        control = _production_control(config, checkpoint["cumulative_errors"])
        if control.get("paused") is not False:
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
        committed = 0
        with Heartbeat(config, heartbeat_fields) as heartbeat:
            while (
                checkpoint["next_index"] < config.site_count
                and committed < max_tranches
            ):
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
                    while cursor < len(rows):
                        if STOP.is_set():
                            heartbeat.write(state="stopped-partial")
                            return {
                                "status": "stopped-partial",
                                "next_index": checkpoint["next_index"],
                                "committed_tranches": committed,
                            }
                        control = _production_control(
                            config, checkpoint["cumulative_errors"]
                        )
                        if control.get("paused") is not False:
                            heartbeat.write(state="paused-partial")
                            return {
                                "status": "paused-partial",
                                "next_index": checkpoint["next_index"],
                                "committed_tranches": committed,
                            }
                        workers = int(control["desired_workers"])
                        if not config.test_mode:
                            assert_runtime_task_ceiling(workers)
                        batch = rows[cursor : cursor + workers]
                        with ThreadPoolExecutor(max_workers=workers) as pool:
                            results.extend(
                                pool.map(
                                    lambda row: _build_site(
                                        config, tranche, row, producer_sha
                                    ),
                                    batch,
                                )
                            )
                        cursor += len(batch)
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
            return {
                "status": "complete" if checkpoint["complete"] else "session-complete",
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
        for tranche in range(checkpoint["completed_tranches"]):
            _verify_tranche(config, snapshot, tranche, producer_sha)
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
        for name in ("checkpoint.json", "compression.json", "control.json"):
            path = config.state_root / name
            result[name] = json.loads(path.read_text()) if path.is_file() else None
        return result
