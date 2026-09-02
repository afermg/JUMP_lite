"""Streaming raw-inventory and bounded candidate-manifest audits."""

from __future__ import annotations

import base64
from collections import Counter
from datetime import date, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Iterator, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from .model import (
    CHANNELS,
    IDENTITY_COLUMNS,
    MAX_CANDIDATE_ROWS,
    channels_for_source,
    digest_json,
    site_key,
)

KNOWN_DAMAGED_OBJECTS = (
    (
        2,
        "ER",
        "URL_OrigER",
        "CP3-SC1-18_I22_T0001F003L01A03Z01C04.tif",
    ),
    (
        3,
        "DNA",
        "URL_OrigDNA",
        "CP3-SC1-18_I22_T0001F004L01A01Z01C01.tif",
    ),
    (
        3,
        "Mito",
        "URL_OrigMito",
        "CP3-SC1-18_I22_T0001F004L01A01Z01C02.tif",
    ),
    (
        3,
        "RNA",
        "URL_OrigRNA",
        "CP3-SC1-18_I22_T0001F004L01A02Z01C03.tif",
    ),
)
KNOWN_DAMAGED_SITE_KEYS = frozenset(
    {
        "source_7__20210727_Run3__CP3-SC1-18__I22__2",
        "source_7__20210727_Run3__CP3-SC1-18__I22__3",
    }
)
_DAMAGED_PREFIX = (
    "s3://cellpainting-gallery/cpg0016-jump/source_7/images/20210727_Run3/"
    "images/CP3-SC1-18/"
)
_DAMAGED_BYTES = 2_768_896
_DAMAGED_ETAG = "d4ffe90e54a5af4e2009e5984da69f03"
_DAMAGED_SHA256 = "5943e9d0a21bc6cc913afecb6e36f2e53d57813d5cd486557a855905293cfe0f"
_CANONICAL_DAMAGED_OBJECT_LEDGER = {
    "bytes": 4_561,
    "sha256": "5666af0c025a0fd97381ab831c49de7bbadf45747c5a97a0e19930a4c469857d",
}
_CANONICAL_DAMAGED_SITE_LEDGER = {
    "bytes": 1_434,
    "sha256": "b5ab319a6a7b910ef82423243ec3dadd731deb89eab614f632f8d29e60dfe15d",
}
_CANONICAL_QC_PLATE_LEDGER = {
    "bytes": 18_229,
    "sha256": "57552bafb6a0065f2de45facc09c7cddc06074924da80b00943c4b8edc804b8f",
}
_CANONICAL_OBJECT_LEDGER_PATH = (
    "metadata/full_jump_compression/known_damaged_objects_v1.json"
)
_CANONICAL_SITE_LEDGER_PATH = (
    "metadata/full_jump_compression/known_damaged_sites_v1.json"
)
_CANONICAL_QC_PLATE_LEDGER_PATH = (
    "metadata/full_jump_compression/qc_plate_classification_v1.json"
)
_QC_UPSTREAM_COMMIT = "016e865fa0691244e0860943e41c7d6a88ed2580"
_QC_PLATE_CSV_SHA256 = (
    "541ada1f64816166509a4e2328316d2a6662ba67e257b7ae134cbec9d7079319"
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
    """Capture the current manifest path for post-audit identity validation."""
    content, observation = _capture_regular_bytes(path, "manifest")
    del content
    return observation


def inventory_digest_from_report(report: Mapping[str, Any]) -> str:
    """Recompute portable inventory identity from every content-bound field."""
    excluded = {"inventory_digest", "local_observation", "local_observation_digest"}
    return digest_json({key: report[key] for key in sorted(set(report) - excluded)})


def _parquet(source: Path | bytes) -> pq.ParquetFile:
    return pq.ParquetFile(
        pa.BufferReader(source) if isinstance(source, bytes) else source
    )


def schema(path: Path | bytes) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    parquet = _parquet(path)
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


def row_count(path: Path | bytes) -> int:
    return int(_parquet(path).metadata.num_rows)


def _physical_order_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return the typed canonical identity tuple used for physical-order checks."""
    site_key(row)  # validates missing, separators, and path-like identity values
    return tuple(row[column] for column in IDENTITY_COLUMNS)


def iter_inventory(
    path: Path | bytes, columns: tuple[str, ...] | None = None, batch_size: int = 1024
) -> Iterator[dict[str, Any]]:
    """Stream a physically identity-ordered Parquet without sorting or threads."""
    parquet = _parquet(path)
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


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _capture_regular_bytes(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    """Capture one stable regular-file snapshot without following the leaf link."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(
            f"{label} must be a regular non-symlink file: {path}"
        ) from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError(f"{label} changed during capture: {path}") from error
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or _stat_identity(before) != _stat_identity(after)
        or _stat_identity(after) != _stat_identity(current)
    ):
        raise RuntimeError(f"{label} changed during capture: {path}")
    content = b"".join(chunks)
    if len(content) != before.st_size:
        raise RuntimeError(f"{label} byte-count drift: {path}")
    return content, {
        "path": str(path.absolute()),
        "bytes": len(content),
        "mtime_ns": before.st_mtime_ns,
        "ctime_ns": before.st_ctime_ns,
        "device": before.st_dev,
        "inode": before.st_ino,
        "sha256": digest.hexdigest(),
    }


def _capture_json_artifact(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    content, observation = _capture_regular_bytes(path, "policy artifact")
    try:
        payload = json.loads(content)
    except Exception as error:
        raise RuntimeError(f"policy artifact is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"policy artifact must contain a JSON object: {path}")
    return {
        "bytes": observation["bytes"],
        "sha256": observation["sha256"],
    }, payload


def _path_matches_observation(path: Path, observation: Mapping[str, Any]) -> bool:
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return (
        stat.S_ISREG(current.st_mode)
        and not stat.S_ISLNK(current.st_mode)
        and current.st_dev == observation["device"]
        and current.st_ino == observation["inode"]
        and current.st_size == observation["bytes"]
        and current.st_mtime_ns == observation["mtime_ns"]
        and current.st_ctime_ns == observation["ctime_ns"]
    )


def _validate_frozen_policy(
    policy_path: Path | None,
    damaged_objects_path: Path | None,
    damaged_sites_path: Path | None,
    qc_plates_path: Path | None,
) -> dict[str, Any]:
    if (
        policy_path is None
        or damaged_objects_path is None
        or damaged_sites_path is None
        or qc_plates_path is None
    ):
        raise RuntimeError(
            "frozen audit requires explicit exclusion policy, damaged-object ledger, "
            "damaged-site ledger, and QC plate-classification ledger"
        )
    policy_binding, policy = _capture_json_artifact(policy_path)
    object_binding, objects = _capture_json_artifact(damaged_objects_path)
    site_binding, sites = _capture_json_artifact(damaged_sites_path)
    qc_binding, qc_plates = _capture_json_artifact(qc_plates_path)

    # These independent pins bind every canonical ledger byte, including JSON
    # key sets, list order, scope, evidence strings, paths, and all values.
    if (
        object_binding != _CANONICAL_DAMAGED_OBJECT_LEDGER
        or site_binding != _CANONICAL_DAMAGED_SITE_LEDGER
    ):
        raise RuntimeError("canonical damaged-ledger binding drift")
    if qc_binding != _CANONICAL_QC_PLATE_LEDGER:
        raise RuntimeError("canonical QC plate-ledger binding drift")

    expected_objects = {
        (
            site,
            channel,
            column,
            _DAMAGED_PREFIX + filename,
        )
        for site, channel, column, filename in KNOWN_DAMAGED_OBJECTS
    }
    try:
        observed_objects = {
            (
                item["site"],
                item["channel"],
                item["url_column"],
                item["uri"],
            )
            for item in objects["objects"]
        }
        object_common_valid = all(
            item["source"] == "source_7"
            and item["batch"] == "20210727_Run3"
            and item["plate"] == "CP3-SC1-18"
            and item["well"] == "I22"
            and item["site_key"]
            == f"source_7__20210727_Run3__CP3-SC1-18__I22__{item['site']}"
            and item["bytes"] == _DAMAGED_BYTES
            and item["etag"] == _DAMAGED_ETAG
            and item["sha256"] == _DAMAGED_SHA256
            and "all-zero" in item["sha256_basis"]
            and item["damage_class"] == "public_object_all_zero_not_valid_tiff"
            and item["action"] == "exclude_entire_site_from_full_jump_compression"
            and isinstance(item["evidence"], list)
            and item["evidence"]
            for item in objects["objects"]
        )
    except Exception as error:
        raise RuntimeError("damaged-object ledger schema malformed") from error
    if (
        set(objects)
        != {
            "format_version",
            "object_count",
            "objects",
            "scope",
            "site_count",
            "unknown_decode_failure_action",
        }
        or objects.get("format_version") != "full-jump-known-damaged-objects-v1"
        or objects.get("scope") != "full_jump_compression_production_inventory"
        or objects.get("object_count") != 4
        or objects.get("site_count") != 2
        or objects.get("unknown_decode_failure_action") != "fatal"
        or len(objects.get("objects", [])) != 4
        or observed_objects != expected_objects
        or not object_common_valid
    ):
        raise RuntimeError("damaged-object ledger content drift")

    try:
        observed_sites = {item["site_key"] for item in sites["sites"]}
        sites_valid = all(
            item["source"] == "source_7"
            and item["batch"] == "20210727_Run3"
            and item["plate"] == "CP3-SC1-18"
            and item["well"] == "I22"
            and item["site"] in (2, 3)
            and item["action"] == "exclude_entire_site_from_full_jump_compression"
            and item["damaged_object_uris"]
            == [
                entry["uri"]
                for entry in objects["objects"]
                if entry["site"] == item["site"]
            ]
            for item in sites["sites"]
        )
    except Exception as error:
        raise RuntimeError("damaged-site ledger schema malformed") from error
    if (
        sites.get("format_version") != "full-jump-known-damaged-sites-v1"
        or sites.get("derived_from") != "known_damaged_objects_v1.json"
        or sites.get("derived_from_sha256") != object_binding["sha256"]
        or sites.get("object_count") != 4
        or sites.get("site_count") != 2
        or len(sites.get("sites", [])) != 2
        or observed_sites != KNOWN_DAMAGED_SITE_KEYS
        or not sites_valid
    ):
        raise RuntimeError("damaged-site ledger content drift")

    try:
        red_plates = {
            (item["source"], item["batch"], item["plate"])
            for item in qc_plates["red_plates"]
        }
        gray_plates = {
            (item["source"], item["batch"], item["plate"])
            for item in qc_plates["gray_plates"]
        }
    except Exception as error:
        raise RuntimeError("QC plate-classification ledger schema malformed") from error
    if (
        qc_plates.get("format_version") != "full-jump-qc-plate-classification-v1"
        or qc_plates.get("classification_only") is not True
        or qc_plates.get("red_plate_count") != 169
        or qc_plates.get("gray_plate_count") != 6
        or len(qc_plates.get("red_plates", [])) != 169
        or len(qc_plates.get("gray_plates", [])) != 6
        or len(red_plates) != 169
        or len(gray_plates) != 6
        or red_plates & gray_plates
        or qc_plates.get("upstream")
        != {
            "commit": _QC_UPSTREAM_COMMIT,
            "plate_csv_path": "metadata/plate.csv.gz",
            "plate_csv_sha256": _QC_PLATE_CSV_SHA256,
            "repository": "jump-cellpainting/datasets",
            "row_count": 2525,
        }
        or qc_plates.get("rules", {}).get("source") != "prep/build_jl_index.sql"
    ):
        raise RuntimeError("QC plate-classification ledger content drift")

    try:
        red_gray = policy["red_gray_release_policy"]
    except Exception as error:
        raise RuntimeError("production exclusion policy schema malformed") from error
    expected_policy = {
        "audit_behavior": "validate_only_never_filter",
        "format_version": "full-jump-production-exclusion-policy-v1",
        "known_damaged_objects": {
            "action": "exclude_entire_affected_site",
            "object_ledger": {
                **_CANONICAL_DAMAGED_OBJECT_LEDGER,
                "path": _CANONICAL_OBJECT_LEDGER_PATH,
            },
            "site_ledger": {
                **_CANONICAL_DAMAGED_SITE_LEDGER,
                "path": _CANONICAL_SITE_LEDGER_PATH,
            },
            "status": "resolved",
        },
        "red_gray_release_policy": red_gray,
        "source_15": {"action": "exclude_all_rows", "status": "resolved"},
        "unknown_decode_failures": {"action": "fatal", "status": "resolved"},
    }
    if policy != expected_policy:
        raise RuntimeError("production exclusion policy content/binding drift")
    expected_classification = {
        **_CANONICAL_QC_PLATE_LEDGER,
        "path": _CANONICAL_QC_PLATE_LEDGER_PATH,
    }
    if red_gray.get("classification_ledger") != expected_classification:
        raise RuntimeError("production exclusion policy QC binding drift")
    unresolved = {
        "action": None,
        "classification_ledger": expected_classification,
        "release_identity_blocked": True,
        "status": "unresolved",
    }
    if red_gray == unresolved:
        raise RuntimeError(
            "red/gray release policy unresolved; frozen identity blocked"
        )
    valid_resolved = {
        (
            "exclude_red_include_gray",
            False,
            "resolved",
        ),
        (
            "exclude_red_and_gray",
            False,
            "resolved",
        ),
    }
    declaration = (
        red_gray.get("action"),
        red_gray.get("release_identity_blocked"),
        red_gray.get("status"),
    )
    if (
        set(red_gray)
        != {
            "action",
            "classification_ledger",
            "release_identity_blocked",
            "status",
        }
        or declaration not in valid_resolved
    ):
        raise RuntimeError("production exclusion policy red/gray content drift")
    return {
        "policy": policy_binding,
        "damaged_objects": object_binding,
        "damaged_sites": site_binding,
        "qc_plates": qc_binding,
        "red_gray_action": red_gray["action"],
        "known_damaged_site_keys": sorted(KNOWN_DAMAGED_SITE_KEYS),
        "red_plate_keys": red_plates,
        "gray_plate_keys": gray_plates,
    }


def _validate_manifest_build_report(
    path: Path | None,
    manifest: Mapping[str, Any],
    row_total: int,
    columns: tuple[str, ...],
    schema_string: str,
    frozen_policy: Mapping[str, Any],
) -> dict[str, Any]:
    if path is None:
        raise RuntimeError("frozen audit requires an explicit manifest build report")
    binding, report = _capture_json_artifact(path)
    if report.get("format_version") != "full-jump-production-manifest-build-v1":
        raise RuntimeError("manifest build report format drift")
    claimed_digest = report.get("build_digest")
    unsigned = {key: value for key, value in report.items() if key != "build_digest"}
    computed_digest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    try:
        inputs = report["inputs"]
        output = report["output"]
        counts = report["counts"]
        if not all(isinstance(value, dict) for value in (inputs, output, counts)):
            raise TypeError("build report sections must be objects")
    except Exception as error:
        raise RuntimeError("manifest build report fields malformed") from error
    expected_output = {
        "bytes": manifest["bytes"],
        "sha256": manifest["sha256"],
        "rows": row_total,
        "columns": list(columns),
        "schema": schema_string,
        "strict_identity_order": True,
        "unique_identities": True,
    }
    if (
        not isinstance(claimed_digest, str)
        or claimed_digest != computed_digest
        or report.get("build_success") is not True
        or report.get("release_identity_frozen") is not False
        or report.get("policy_action") != frozen_policy["red_gray_action"]
        or output != expected_output
        or counts.get("final_rows") != row_total
        or inputs.get("exclusion_policy") != frozen_policy["policy"]
        or inputs.get("damaged_objects") != frozen_policy["damaged_objects"]
        or inputs.get("damaged_sites") != frozen_policy["damaged_sites"]
        or inputs.get("qc_plates") != frozen_policy["qc_plates"]
    ):
        raise RuntimeError("manifest build report identity/completion drift")
    return {
        "artifact": binding,
        "build_digest": computed_digest,
        "format_version": report["format_version"],
    }


def audit_inventory(
    path: Path,
    report_path: Path | None = None,
    *,
    kind: str = "raw",
    exclusion_policy: Path | None = None,
    damaged_objects: Path | None = None,
    damaged_sites: Path | None = None,
    qc_plates: Path | None = None,
    build_report: Path | None = None,
) -> dict[str, Any]:
    if kind not in {"raw", "candidate", "frozen"}:
        raise ValueError("audit kind must be raw, candidate, or frozen")
    manifest_bytes, observation = _capture_regular_bytes(path, "manifest")
    frozen_policy = None
    frozen_red_plates: set[tuple[str, str, str]] = set()
    frozen_gray_plates: set[tuple[str, str, str]] = set()
    if kind == "frozen":
        frozen_policy = _validate_frozen_policy(
            exclusion_policy, damaged_objects, damaged_sites, qc_plates
        )
        frozen_red_plates = frozen_policy.pop("red_plate_keys")
        frozen_gray_plates = frozen_policy.pop("gray_plate_keys")
    description, columns = schema(manifest_bytes)
    manifest_schema_string = str(_parquet(manifest_bytes).schema_arrow)
    count_expected = row_count(manifest_bytes)
    frozen_build = None
    if kind == "frozen":
        frozen_build = _validate_manifest_build_report(
            build_report,
            {"bytes": observation["bytes"], "sha256": observation["sha256"]},
            count_expected,
            columns,
            manifest_schema_string,
            frozen_policy,
        )
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
    frozen_source_15_rows = 0
    frozen_damaged_site_rows = 0
    frozen_red_plate_rows = 0
    frozen_gray_plate_rows = 0
    processed = 0
    if not missing_columns:
        for raw in iter_inventory(manifest_bytes, columns):
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
                source = str(raw["Metadata_Source"])
                counts[source] += 1
                if kind == "frozen":
                    plate_key = (
                        source,
                        str(raw["Metadata_Batch"]),
                        str(raw["Metadata_Plate"]),
                    )
                    frozen_source_15_rows += int(source == "source_15")
                    frozen_damaged_site_rows += int(key in KNOWN_DAMAGED_SITE_KEYS)
                    frozen_red_plate_rows += int(plate_key in frozen_red_plates)
                    frozen_gray_plate_rows += int(plate_key in frozen_gray_plates)
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
    path_stable = _path_matches_observation(path, observation)
    audit_success = not (
        not path_stable
        or processed != count_expected
        or missing_columns
        or extra_url_columns
        or anomaly_count
        or (kind == "frozen" and frozen_source_15_rows)
        or (kind == "frozen" and frozen_damaged_site_rows)
        or (kind == "frozen" and frozen_red_plate_rows)
        or (
            kind == "frozen"
            and frozen_policy["red_gray_action"] == "exclude_red_and_gray"
            and frozen_gray_plate_rows
        )
    )
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
        "audit_success": audit_success,
        "release_identity_frozen": kind == "frozen" and audit_success,
        "manifest_path_stable_through_audit": path_stable,
        "policy_note": "Raw audit and bounded candidate selection are distinct; frozen audits validate but never filter exclusions.",
    }
    if kind == "frozen":
        summary["frozen_exclusion_policy"] = {
            **frozen_policy,
            "source_15_rows_present": frozen_source_15_rows,
            "known_damaged_site_rows_present": frozen_damaged_site_rows,
            "red_plate_rows_present": frozen_red_plate_rows,
            "gray_plate_rows_present": frozen_gray_plate_rows,
            "manifest_build_report": frozen_build,
        }
    summary["inventory_digest"] = inventory_digest_from_report(summary)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if not audit_success:
        raise RuntimeError(f"inventory audit failed; report={report_path}")
    return summary


def load_audit(
    path: Path, manifest: Path, expected_digest: str, *, kind: str
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    actual_bytes, actual = _capture_regular_bytes(manifest, "manifest")
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
    audit_success = payload.get("audit_success")
    if kind == "frozen":
        if (
            audit_success is not True
            or payload.get("release_identity_frozen") is not True
        ):
            raise RuntimeError(
                "frozen audit report does not record a successful freeze"
            )
    elif audit_success is not True:
        legacy_success = (
            "audit_success" not in payload
            and row_count(actual_bytes) == payload.get("site_count")
            and payload.get("missing_columns") == []
            and payload.get("unsupported_url_columns") == []
            and payload.get("anomalies") == []
            and payload.get("anomaly_count") == 0
            and payload.get("release_identity_frozen") is False
        )
        if not legacy_success:
            raise RuntimeError("audit report records an unsuccessful audit")
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
