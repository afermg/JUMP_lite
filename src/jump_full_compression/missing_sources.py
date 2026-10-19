"""Conservative missing-object classification and durable whole-site exclusions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .incidents import (
    MAX_READ_ATTEMPTS,
    MAX_RECORD_BYTES,
    SOURCE_READ_ATTEMPTS,
    _read,
    retry_evidence,
)
from .model import CODECS, ProductionConfig, atomic_json, fsync_dir

MISSING_RECEIPT_FORMAT = "full-jump-production-excluded-source-v1"
EVIDENCE_KEYS = {
    "attempt",
    "error_code",
    "http_status",
    "sdk_retry_attempts",
    "retry_delay_seconds",
}


def _public_uri(uri: Any) -> bool:
    if not isinstance(uri, str):
        return False
    parsed = urlparse(uri)
    return bool(
        parsed.scheme == "s3"
        and parsed.netloc == "cellpainting-gallery"
        and parsed.path
        and not parsed.query
        and not parsed.fragment
    )


def _exhausted_missing(failure: dict[str, Any]) -> bool:
    if not isinstance(failure, dict):
        return False
    attempts = failure.get("attempts")
    count = failure.get("attempt_count")
    return bool(
        _public_uri(failure.get("uri"))
        and type(count) is int
        and SOURCE_READ_ATTEMPTS <= count <= MAX_READ_ATTEMPTS
        and isinstance(attempts, list)
        and len(attempts) == count
        and all(
            isinstance(item, dict)
            and set(item) == EVIDENCE_KEYS
            and type(item.get("attempt")) is int
            and item["attempt"] == index
            and item.get("error_code") == "NoSuchKey"
            and type(item.get("http_status")) is int
            and item["http_status"] == 404
            and (
                item.get("sdk_retry_attempts") is None
                or (
                    type(item["sdk_retry_attempts"]) is int
                    and 0 <= item["sdk_retry_attempts"] <= 100
                )
            )
            and type(item.get("retry_delay_seconds")) in (int, float)
            and item["retry_delay_seconds"]
            == (min(8.0, 0.25 * 2**index) if index < count else 0.0)
            for index, item in enumerate(attempts, 1)
        )
    )


def confirm_missing_sources(
    failures: list[dict[str, Any]],
    client_factory: Callable,
) -> list[dict[str, Any]]:
    # A timeout, access denial, missing bucket/version, mixed retry history, or
    # unknown error is not proof of a missing object. Never downgrade it to a skip.
    if not failures or not all(_exhausted_missing(failure) for failure in failures):
        return failures
    observed = []
    for failure in failures:
        parsed = urlparse(failure["uri"])
        try:
            client_factory().head_object(
                Bucket=parsed.netloc, Key=parsed.path.lstrip("/")
            )
        except Exception as error:
            confirmation = retry_evidence(error, 1, 0.0)
        else:
            confirmation = {"result": "object-present"}
        observed.append({**failure, "head_confirmation": confirmation})
    return observed


def confirmed_missing(failures: list[dict[str, Any]]) -> bool:
    if not isinstance(failures, list) or not failures:
        return False
    for failure in failures:
        if not isinstance(failure, dict) or not isinstance(failure.get("channel"), str):
            return False
        if set(failure) != {
            "channel",
            "uri",
            "attempt_count",
            "attempts",
            "head_confirmation",
        } or not _exhausted_missing(failure):
            return False
        head = failure["head_confirmation"]
        if not (
            isinstance(head, dict)
            and set(head) == EVIDENCE_KEYS
            and type(head.get("attempt")) is int
            and head["attempt"] == 1
            and type(head.get("retry_delay_seconds")) in (int, float)
            and head["retry_delay_seconds"] == 0.0
            and type(head.get("http_status")) is int
            and head["http_status"] == 404
            and isinstance(head.get("error_code"), str)
            and head["error_code"] in {"NoSuchKey", "NotFound", "404"}
            and (
                head.get("sdk_retry_attempts") is None
                or (
                    type(head["sdk_retry_attempts"]) is int
                    and 0 <= head["sdk_retry_attempts"] <= 100
                )
            )
        ):
            return False
    return True


def read_receipt(path: Path) -> dict[str, Any]:
    return json.loads(_read(path, MAX_RECORD_BYTES))


def validate_missing_receipt(
    value: dict[str, Any],
    config: ProductionConfig,
    tranche: int,
    row: dict[str, Any],
    producer_sha256: str,
) -> None:
    required = {
        "format_version",
        "production_not_published",
        "production_id",
        "tranche",
        "manifest_index",
        "site_key",
        "source_row_sha256",
        "channels",
        "missing_sources",
        "object_count",
        "status",
        "inventory_digest",
        "config_sha256",
        "producer_sha256",
        "completed_at",
    }
    failures = value.get("missing_sources")
    if (
        set(value) != required
        or value.get("format_version") != MISSING_RECEIPT_FORMAT
        or config.missing_source_policy != "record-and-skip"
        or value.get("production_not_published") is not True
        or value.get("production_id") != config.production_id
        or value.get("config_sha256") != config.digest
        or value.get("inventory_digest") != config.inventory_digest
        or value.get("producer_sha256") != producer_sha256
        or type(value.get("tranche")) is not int
        or value["tranche"] != tranche
        or type(value.get("manifest_index")) is not int
        or value["manifest_index"] != row["manifest_index"]
        or not 0 <= value["manifest_index"] < config.site_count
        or value["manifest_index"] // config.tranche_size != tranche
        or value.get("site_key") != row["Metadata_Site_Key"]
        or value.get("source_row_sha256") != row["source_row_sha256"]
        or value.get("channels") != row["channels"]
        or value.get("status") != "excluded-missing-source"
        or not isinstance(failures, list)
        or not confirmed_missing(failures)
        or type(value.get("object_count")) is not int
        or value["object_count"] != len(failures)
    ):
        raise RuntimeError("missing-source receipt identity/evidence drift")
    channels = [failure["channel"] for failure in failures]
    if (
        len(set(channels)) != len(channels)
        or channels != [channel for channel in row["channels"] if channel in channels]
        or any(
            failure["uri"] != row["urls"][row["channels"].index(failure["channel"])]
            for failure in failures
        )
    ):
        raise RuntimeError("missing-source receipt channel/URI order drift")
    for codec in CODECS:
        path = (
            config.output_root / "codecs" / f"{codec}.zarr" / row["Metadata_Site_Key"]
        )
        if path.exists() or path.is_symlink():
            raise RuntimeError("excluded site unexpectedly has image output")


def publish_missing_receipt(
    path: Path,
    config: ProductionConfig,
    tranche: int,
    row: dict[str, Any],
    producer_sha256: str,
    failures: list[dict[str, Any]],
    completed_at: str,
) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(
            "refusing to replace an existing site receipt with an exclusion"
        )
    value = {
        "format_version": MISSING_RECEIPT_FORMAT,
        "production_not_published": True,
        "production_id": config.production_id,
        "tranche": tranche,
        "manifest_index": row["manifest_index"],
        "site_key": row["Metadata_Site_Key"],
        "source_row_sha256": row["source_row_sha256"],
        "channels": row["channels"],
        "missing_sources": failures,
        "object_count": len(failures),
        "status": "excluded-missing-source",
        "inventory_digest": config.inventory_digest,
        "config_sha256": config.digest,
        "producer_sha256": producer_sha256,
        "completed_at": completed_at,
    }
    validate_missing_receipt(value, config, tranche, row, producer_sha256)
    if len(json.dumps(value, sort_keys=True, indent=2).encode()) + 1 > MAX_RECORD_BYTES:
        raise RuntimeError("missing-source receipt exceeds size bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    fsync_dir(path.parent.parent)
    atomic_json(path, value)
