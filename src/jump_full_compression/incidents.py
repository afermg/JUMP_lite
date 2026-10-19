"""Bounded, credential-free source-read evidence and a fail-closed incident ledger.

The production lock must be held by callers that append or acknowledge. Records are
published before their index: an interrupted append leaves an unknown entry, which
blocks resume rather than silently losing evidence. Nothing here changes progress.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any

from botocore.exceptions import ClientError

from .model import ProductionConfig, atomic_json, digest_json, fsync_dir

SOURCE_READ_ATTEMPTS = 5
MAX_READ_ATTEMPTS = 8
MAX_INCIDENT_RECORDS = 10_000
MAX_RECORD_BYTES = 256 * 1024
MAX_INDEX_BYTES = 2 * 1024 * 1024
INDEX_FORMAT = "full-jump-source-incident-index-v1"
RECORD_FORMAT = "full-jump-source-incident-v1"
ACK_FORMAT = "full-jump-source-incident-ack-v1"
S3_CODES = {
    "404",
    "NoSuchKey",
    "NotFound",
    "NoSuchBucket",
    "NoSuchVersion",
    "AccessDenied",
    "InvalidAccessKeyId",
    "SignatureDoesNotMatch",
    "ExpiredToken",
    "InvalidToken",
    "RequestTimeout",
    "SlowDown",
    "ServiceUnavailable",
    "InternalError",
    "PermanentRedirect",
    "AuthorizationHeaderMalformed",
}
TRANSPORT_CODES = {
    "OSError",
    "FileNotFoundError",
    "TimeoutError",
    "EndpointConnectionError",
    "ConnectionClosedError",
    "ConnectTimeoutError",
    "ReadTimeoutError",
    "ResponseStreamingError",
    "IncompleteReadError",
}


class SourceReadError(RuntimeError):
    def __init__(self, uri: str, attempts: list[dict[str, Any]]):
        self.failure = {
            "uri": uri,
            "attempt_count": len(attempts),
            "attempts": attempts,
        }
        super().__init__("source read exhausted its retry budget")


class SiteSourceReadError(RuntimeError):
    def __init__(self, failures: list[dict[str, Any]]):
        self.failures = failures
        super().__init__(f"source reads failed for {len(failures)} channel(s)")


class SourceIncidentError(RuntimeError):
    """A source failure must not enter the generic checkpoint-error update path."""


def retry_evidence(error: Exception, attempt: int, delay: float) -> dict[str, Any]:
    # Never serialize str(error), exception args, request IDs, headers, endpoint
    # URLs, or the full botocore response. Unknown codes are deliberately collapsed.
    code = type(error).__name__
    status = retries = None
    if isinstance(error, ClientError):
        response = error.response if isinstance(error.response, dict) else {}
        detail = response.get("Error", {})
        metadata = response.get("ResponseMetadata", {})
        code = detail.get("Code") if isinstance(detail, dict) else None
        code = (
            code if isinstance(code, str) and code in S3_CODES else "OtherClientError"
        )
        if isinstance(metadata, dict):
            value = metadata.get("HTTPStatusCode")
            status = value if type(value) is int and 100 <= value <= 599 else None
            value = metadata.get("RetryAttempts")
            retries = value if type(value) is int and 0 <= value <= 100 else None
    elif code not in TRANSPORT_CODES:
        code = "OtherReadError"
    return {
        "attempt": attempt,
        "error_code": code,
        "http_status": status,
        "sdk_retry_attempts": retries,
        "retry_delay_seconds": delay,
    }


def _identity(config: ProductionConfig) -> dict[str, str]:
    return {
        "production_id": config.production_id,
        "config_sha256": config.digest,
        "inventory_digest": config.inventory_digest,
        "manifest_sha256": config.manifest_sha256,
    }


def _root(config: ProductionConfig) -> Path:
    return config.state_root / "source-incidents"


def _read(path: Path, maximum: int) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        observed = os.fstat(fd)
        if not stat.S_ISREG(observed.st_mode) or observed.st_size > maximum:
            raise RuntimeError("source incident file is not bounded regular data")
        with os.fdopen(os.dup(fd), "rb") as stream:
            data = stream.read(maximum + 1)
        if len(data) > maximum:
            raise RuntimeError("source incident file exceeds size bound")
        return data
    finally:
        os.close(fd)


def _write(path: Path, value: dict[str, Any], maximum: int) -> None:
    if len(json.dumps(value, sort_keys=True, indent=2).encode()) + 1 > maximum:
        raise RuntimeError("source incident evidence exceeds size bound")
    if path.is_symlink():
        raise RuntimeError("source incident symlink rejected")
    atomic_json(path, value)


def load_incidents(config: ProductionConfig) -> dict[str, Any]:
    root = _root(config)
    entries: list[dict[str, str]] = []
    acknowledged = 0
    if root.is_symlink():
        raise RuntimeError("source incident root symlink rejected")
    if root.exists():
        if not root.is_dir():
            raise RuntimeError("source incident root is not a directory")
        index = json.loads(_read(root / "index.json", MAX_INDEX_BYTES))
        entries = index.get("records")
        if (
            set(index) != {"format_version", "records", *_identity(config)}
            or index.get("format_version") != INDEX_FORMAT
            or any(index.get(k) != v for k, v in _identity(config).items())
            or not isinstance(entries, list)
            or not 1 <= len(entries) <= MAX_INCIDENT_RECORDS
        ):
            raise RuntimeError("source incident index identity/count drift")
        expected = {"index.json", "acknowledgement.json"}
        for sequence, entry in enumerate(entries, 1):
            name = f"{sequence:08d}.json"
            if (
                not isinstance(entry, dict)
                or set(entry) != {"name", "sha256"}
                or entry.get("name") != name
            ):
                raise RuntimeError("source incident record order drift")
            expected.add(name)
            data = _read(root / name, MAX_RECORD_BYTES)
            record = json.loads(data)
            if (
                hashlib.sha256(data).hexdigest() != entry["sha256"]
                or record.get("format_version") != RECORD_FORMAT
                or record.get("sequence") != sequence
                or any(record.get(k) != v for k, v in _identity(config).items())
            ):
                raise RuntimeError("source incident record identity/hash drift")
        if {p.name for p in root.iterdir()} - expected:
            raise RuntimeError(
                "unknown or interrupted source incident entry; review required"
            )
        ack_path = root / "acknowledgement.json"
        if ack_path.exists() or ack_path.is_symlink():
            ack = json.loads(_read(ack_path, MAX_RECORD_BYTES))
            acknowledged = ack.get("count")
            if (
                set(ack)
                != {"format_version", "count", "list_sha256", *_identity(config)}
                or ack.get("format_version") != ACK_FORMAT
                or any(ack.get(k) != v for k, v in _identity(config).items())
                or type(acknowledged) is not int
                or not 0 <= acknowledged <= len(entries)
                or ack.get("list_sha256") != digest_json(entries[:acknowledged])
            ):
                raise RuntimeError("source incident acknowledgement drift")
    return {
        "count": len(entries),
        "records": entries,
        "list_sha256": digest_json(entries),
        "acknowledged_count": acknowledged,
        "unacknowledged_count": len(entries) - acknowledged,
    }


def record_incident(
    config: ProductionConfig,
    checkpoint: dict[str, Any],
    checkpoint_sha256: str,
    producer_sha256: str,
    sites: list[dict[str, Any]],
    created_at: str,
    *,
    missing_limit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ledger = load_incidents(config)
    if ledger["unacknowledged_count"]:
        raise RuntimeError("source incidents must be acknowledged before retry")
    sequence = ledger["count"] + 1
    if sequence > MAX_INCIDENT_RECORDS or not 1 <= len(sites) <= config.max_workers:
        raise RuntimeError("source incident record/count limit exceeded")
    root = _root(config)
    if not root.exists():
        root.mkdir(mode=0o700)
        fsync_dir(root.parent)
    name = f"{sequence:08d}.json"
    path = root / name
    if path.exists() or path.is_symlink():
        raise RuntimeError("source incident record already exists")
    _write(
        path,
        {
            "format_version": RECORD_FORMAT,
            **_identity(config),
            "sequence": sequence,
            "producer_sha256": producer_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "boundary": {
                k: checkpoint[k]
                for k in ("next_index", "completed_tranches", "chain_head")
            },
            "sites": sites,
            "object_count": sum(len(site["failures"]) for site in sites),
            "created_at": created_at,
            "reason": "missing-source-rate-exceeded"
            if missing_limit
            else "source-read-failure",
            "missing_limit": missing_limit,
        },
        MAX_RECORD_BYTES,
    )
    entries = ledger["records"] + [
        {
            "name": name,
            "sha256": hashlib.sha256(_read(path, MAX_RECORD_BYTES)).hexdigest(),
        }
    ]
    _write(
        root / "index.json",
        {
            "format_version": INDEX_FORMAT,
            **_identity(config),
            "records": entries,
        },
        MAX_INDEX_BYTES,
    )
    return load_incidents(config)


def acknowledge_incidents(
    config: ProductionConfig,
    expected_count: int,
    expected_list_sha256: str,
    *,
    apply: bool = False,
) -> dict[str, Any]:
    ledger = load_incidents(config)
    if (
        type(expected_count) is not int
        or expected_count < 1
        or expected_count != ledger["count"]
        or expected_list_sha256 != ledger["list_sha256"]
    ):
        raise RuntimeError("exact source incident count and list SHA-256 are required")
    if apply:
        _write(
            _root(config) / "acknowledgement.json",
            {
                "format_version": ACK_FORMAT,
                **_identity(config),
                "count": expected_count,
                "list_sha256": expected_list_sha256,
            },
            MAX_RECORD_BYTES,
        )
    return ledger
