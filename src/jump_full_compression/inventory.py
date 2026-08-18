"""Streaming raw-inventory and bounded candidate-manifest audits."""

from __future__ import annotations

import base64
from collections import Counter
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator, Mapping

import pyarrow.parquet as pq

from .model import (
    CHANNELS,
    IDENTITY_COLUMNS,
    MAX_CANDIDATE_ROWS,
    channels_for_source,
    digest_json,
    sha256_file,
    site_key,
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite manifest value")
        return value
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode()}
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in sorted(value.items())}
    return str(value)


def _canonical_row(row: Mapping[str, Any], columns: tuple[str, ...]) -> dict[str, Any]:
    source = str(row["Metadata_Source"])
    channels = channels_for_source(source)
    key = site_key(row)
    full = {column: _json_safe(row.get(column)) for column in columns}
    urls = [row.get(f"URL_Orig{channel}") for channel in channels]
    for channel, value in zip(channels, urls):
        if value is None or not str(value).strip():
            raise ValueError(f"missing required channel {channel}")
    if source == "source_15" and row.get("URL_OrigRNA") not in (None, ""):
        raise ValueError("source_15 has unsupported separate RNA channel")
    result = {
        "Metadata_Site_Key": key,
        "channels": list(channels),
        "urls": [str(value) for value in urls],
        "full_row": full,
    }
    result["source_row_sha256"] = digest_json(result)
    return result


def manifest_stat(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.absolute()),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "sha256": sha256_file(path),
    }


def inventory_digest_from_report(report: Mapping[str, Any]) -> str:
    """Recompute portable inventory identity from every content-bound field."""
    excluded = {"inventory_digest", "local_observation", "local_observation_digest"}
    return digest_json({key: report[key] for key in sorted(set(report) - excluded)})


def schema(path: Path) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    parquet = pq.ParquetFile(path)
    arrow_schema = parquet.schema_arrow
    description = [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
            "metadata": _json_safe(field.metadata),
        }
        for field in arrow_schema
    ]
    return description, tuple(arrow_schema.names)


def row_count(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _physical_order_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the typed canonical identity tuple used for physical-order checks."""
    site_key(row)  # validates missing, separators, and path-like identity values
    return tuple(row[column] for column in IDENTITY_COLUMNS)


def iter_inventory(
    path: Path, columns: tuple[str, ...] | None = None, batch_size: int = 1024
) -> Iterator[dict[str, Any]]:
    """Stream a physically identity-ordered Parquet without sorting or threads."""
    parquet = pq.ParquetFile(path)
    columns = columns or tuple(parquet.schema_arrow.names)
    previous: tuple[Any, ...] | None = None
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=list(columns), use_threads=False
    ):
        values = batch.to_pydict()
        for index in range(batch.num_rows):
            row = {column: values[column][index] for column in columns}
            try:
                current = _physical_order_key(row)
                out_of_order = previous is not None and current <= previous
            except Exception as error:
                raise RuntimeError(
                    "manifest identity values are invalid or not order-comparable"
                ) from error
            if out_of_order:
                raise RuntimeError(
                    "manifest is not in strict physical canonical identity order"
                )
            previous = current
            yield row


def audit_inventory(
    path: Path, report_path: Path | None = None, *, kind: str = "raw"
) -> dict[str, Any]:
    if kind not in {"raw", "candidate", "frozen"}:
        raise ValueError("audit kind must be raw, candidate, or frozen")
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    description, columns = schema(path)
    count_expected = row_count(path)
    if kind == "candidate" and not 1 <= count_expected <= MAX_CANDIDATE_ROWS:
        raise RuntimeError(
            f"candidate manifest must contain 1..{MAX_CANDIDATE_ROWS} rows, got {count_expected}"
        )
    required = set(IDENTITY_COLUMNS) | {f"URL_Orig{x}" for x in CHANNELS}
    missing_columns = sorted(required - set(columns))
    extra_url_columns = sorted(
        name for name in columns if name.startswith("URL_Orig") and name not in required
    )
    key_hash, row_hash = hashlib.sha256(), hashlib.sha256()
    anomalies: list[dict[str, Any]] = []
    anomaly_count = 0
    counts: Counter[str] = Counter()
    processed = 0
    if not missing_columns:
        for raw in iter_inventory(path, columns):
            processed += 1
            try:
                normalized = _canonical_row(raw, columns)
                key = normalized["Metadata_Site_Key"]
                key_hash.update(key.encode() + b"\n")
                row_hash.update(
                    json.dumps(
                        normalized["full_row"], sort_keys=True, separators=(",", ":")
                    ).encode()
                    + b"\n"
                )
                counts[str(raw["Metadata_Source"])] += 1
            except Exception as error:
                anomaly_count += 1
                if len(anomalies) < 1000:
                    anomalies.append(
                        {
                            "row": processed,
                            "error": str(error),
                            "identity": [
                                _json_safe(raw.get(column))
                                for column in IDENTITY_COLUMNS
                            ],
                        }
                    )
    observation = manifest_stat(path)
    summary = {
        "format_version": "full-jump-inventory-audit-v2",
        "audit_kind": kind,
        "manifest": {
            "bytes": observation["bytes"],
            "sha256": observation["sha256"],
        },
        "local_observation": observation,
        "local_observation_digest": digest_json(observation),
        "schema": description,
        "columns": list(columns),
        "schema_sha256": digest_json(description),
        "site_count": processed,
        "site_key_sha256": key_hash.hexdigest(),
        "full_row_sha256": row_hash.hexdigest(),
        "source_counts": dict(sorted(counts.items())),
        "missing_columns": missing_columns,
        "unsupported_url_columns": extra_url_columns,
        "anomaly_count": anomaly_count,
        "anomalies": anomalies,
        "release_identity_frozen": kind == "frozen",
        "policy_note": "Raw audit and bounded candidate selection are distinct; QC policy is external.",
    }
    summary["inventory_digest"] = inventory_digest_from_report(summary)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if (
        processed != count_expected
        or missing_columns
        or extra_url_columns
        or anomaly_count
    ):
        raise RuntimeError(f"inventory audit failed; report={report_path}")
    return summary


def load_audit(
    path: Path, manifest: Path, expected_digest: str, *, kind: str
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    actual = manifest_stat(manifest)
    if (
        payload.get("format_version") != "full-jump-inventory-audit-v2"
        or payload.get("audit_kind") != kind
    ):
        raise RuntimeError("audit format/kind drift")
    try:
        recomputed = inventory_digest_from_report(payload)
        observation_digest = digest_json(payload["local_observation"])
        content = payload["manifest"]
    except Exception as error:
        raise RuntimeError("audit report fields malformed") from error
    if (
        payload.get("inventory_digest") != expected_digest
        or recomputed != expected_digest
        or payload.get("local_observation_digest") != observation_digest
        or content != {"bytes": actual["bytes"], "sha256": actual["sha256"]}
    ):
        raise RuntimeError("audit/manifest identity drift")
    return payload


def normalized_rows(path: Path) -> Iterator[dict[str, Any]]:
    columns = schema(path)[1]
    for raw in iter_inventory(path, columns):
        yield _canonical_row(raw, columns)


def candidate_rows(path: Path, *, expected_count: int) -> Iterator[dict[str, Any]]:
    if (
        not 1 <= expected_count <= MAX_CANDIDATE_ROWS
        or row_count(path) != expected_count
    ):
        raise RuntimeError("candidate row ceiling/count drift")
    yield from normalized_rows(path)
