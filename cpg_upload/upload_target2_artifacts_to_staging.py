#!/usr/bin/env python3
"""Validate and upload Target-2 publication artifacts to CPG staging.

The command is a dry run unless ``--apply`` is supplied. Object/profile data
are uploaded first through a resumable checkpoint. Package README and manifests
are withheld until recursive remote count/byte verification of those data
prefixes succeeds. The complete version prefix is then verified again.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import boto3
import pyarrow.parquet as pq
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from build_target2_artifacts import (
    DEFAULT_CP_PROFILE_ROOT,
    DEFAULT_DL_PROFILE_ROOT,
    DEFAULT_MASK_ROOT,
    DEFAULT_RELEASE_ROOT,
    DESTINATION_ROOT,
    EXPECTED_MASK_BYTES,
    EXPECTED_MASK_FILE_COUNT,
    EXPECTED_OBJECT_FEATURE_BYTES,
    EXPECTED_OBJECT_FEATURE_FILE_COUNT,
    EXPECTED_CP_ROOT_BYTES,
    EXPECTED_CP_ROOT_FILE_COUNT,
    EXPECTED_DL_ROOT_BYTES,
    EXPECTED_DL_ROOT_FILE_COUNT,
    sha256_file,
)

ACCOUNT_ID = "309624411020"
REGION = "us-east-1"
BUCKET = "staging-cellpainting-gallery"
GRANT_TARGET = "s3://staging-cellpainting-gallery/cpg0016-jump/*"
DEFAULT_CHECKPOINT = Path(
    "/work/datasets/jump_lite/cpg_upload_state/target_2/v1.0/checkpoint.json"
)
METADATA_RELATIVE_KEYS = (
    "README.md",
    "manifests/masks.parquet",
    "manifests/object_features.parquet",
    "manifests/profiles.parquet",
    "manifests/provenance.json",
)
AUTH_REFRESH_CODES = {
    "AccessDenied",
    "ExpiredToken",
    "InvalidAccessKeyId",
    "RequestExpired",
    "TokenRefreshRequired",
}


@dataclass(frozen=True)
class UploadEntry:
    source: Path
    relative_key: str
    size_bytes: int
    sha256: str
    content_type: str

    @property
    def destination_key(self) -> str:
        return f"{DESTINATION_ROOT}/{self.relative_key}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--cp-profile-root", type=Path, default=DEFAULT_CP_PROFILE_ROOT)
    parser.add_argument("--dl-profile-root", type=Path, default=DEFAULT_DL_PROFILE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument(
        "--adopt-existing-verified",
        action="store_true",
        help=(
            "when extending the package, verify and adopt the already-complete "
            "mask/compact-profile upload before transferring new object features"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the staging upload after validation",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _source_for_mask(row: Mapping[str, Any], mask_root: Path) -> Path:
    return mask_root / str(row["source_relative_path"])


def _source_for_object_feature(row: Mapping[str, Any], mask_root: Path) -> Path:
    return mask_root / str(row["source_relative_path"])


def _source_for_profile(
    row: Mapping[str, Any], cp_profile_root: Path, dl_profile_root: Path
) -> Path:
    group = row["source_group"]
    if group == "compact_cp":
        return cp_profile_root / str(row["source_relative_path"])
    if group == "compact_dl":
        return dl_profile_root / str(row["source_relative_path"])
    raise RuntimeError(f"unknown profile source group: {group!r}")


def _content_type(path: Path) -> str:
    if path.suffix == ".parquet":
        return "application/vnd.apache.parquet"
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".md":
        return "text/markdown; charset=utf-8"
    return "application/octet-stream"


def _metadata_entry(release_root: Path, relative_key: str) -> UploadEntry:
    source = release_root / relative_key
    if not source.is_file():
        raise RuntimeError(f"missing package metadata: {source}")
    return UploadEntry(
        source=source,
        relative_key=relative_key,
        size_bytes=source.stat().st_size,
        sha256=sha256_file(source),
        content_type=_content_type(source),
    )


def inventory_digest(entries: Sequence[UploadEntry]) -> str:
    payload = [
        {
            "relative_key": row.relative_key,
            "size_bytes": row.size_bytes,
            "sha256": row.sha256,
        }
        for row in entries
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_package(
    *,
    release_root: Path,
    mask_root: Path,
    cp_profile_root: Path,
    dl_profile_root: Path,
) -> tuple[list[UploadEntry], list[UploadEntry], dict[str, Any]]:
    provenance_path = release_root / "manifests/provenance.json"
    if not provenance_path.is_file():
        raise RuntimeError(
            f"missing Target-2 provenance; run build_target2_artifacts.py first: {provenance_path}"
        )
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("destination_root") != DESTINATION_ROOT:
        raise RuntimeError("Target-2 provenance has the wrong destination root")

    mask_manifest = release_root / "manifests/masks.parquet"
    object_feature_manifest = release_root / "manifests/object_features.parquet"
    profile_manifest = release_root / "manifests/profiles.parquet"
    expected_mask_manifest_hash = provenance.get("mask_inventory", {}).get(
        "manifest_sha256"
    )
    expected_object_feature_manifest_hash = provenance.get(
        "object_feature_inventory", {}
    ).get("manifest_sha256")
    expected_profile_manifest_hash = provenance.get("profile_inventory", {}).get(
        "manifest_sha256"
    )
    if sha256_file(mask_manifest) != expected_mask_manifest_hash:
        raise RuntimeError("Target-2 mask manifest hash mismatch")
    if sha256_file(object_feature_manifest) != expected_object_feature_manifest_hash:
        raise RuntimeError("Target-2 object-feature manifest hash mismatch")
    if sha256_file(profile_manifest) != expected_profile_manifest_hash:
        raise RuntimeError("Target-2 profile manifest hash mismatch")
    if sha256_file(release_root / "README.md") != provenance.get(
        "package_metadata", {}
    ).get("readme_sha256"):
        raise RuntimeError("Target-2 README hash mismatch")

    mask_rows = _rows(mask_manifest)
    object_feature_rows = _rows(object_feature_manifest)
    profile_rows = _rows(profile_manifest)
    if len(mask_rows) != EXPECTED_MASK_FILE_COUNT:
        raise RuntimeError(
            f"mask manifest has {len(mask_rows):,} rows; expected {EXPECTED_MASK_FILE_COUNT:,}"
        )
    mask_bytes = sum(int(row["size_bytes"]) for row in mask_rows)
    if mask_bytes != EXPECTED_MASK_BYTES:
        raise RuntimeError(
            f"mask manifest has {mask_bytes:,} bytes; expected {EXPECTED_MASK_BYTES:,}"
        )
    if len(object_feature_rows) != EXPECTED_OBJECT_FEATURE_FILE_COUNT:
        raise RuntimeError(
            f"object-feature manifest has {len(object_feature_rows):,} rows; "
            f"expected {EXPECTED_OBJECT_FEATURE_FILE_COUNT:,}"
        )
    object_feature_bytes = sum(
        int(row["size_bytes"]) for row in object_feature_rows
    )
    if object_feature_bytes != EXPECTED_OBJECT_FEATURE_BYTES:
        raise RuntimeError(
            f"object-feature manifest has {object_feature_bytes:,} bytes; "
            f"expected {EXPECTED_OBJECT_FEATURE_BYTES:,}"
        )
    expected_profile_count = EXPECTED_CP_ROOT_FILE_COUNT + EXPECTED_DL_ROOT_FILE_COUNT
    if len(profile_rows) != expected_profile_count:
        raise RuntimeError(
            f"profile manifest has {len(profile_rows)} rows; expected {expected_profile_count}"
        )
    expected_profile_bytes = EXPECTED_CP_ROOT_BYTES + EXPECTED_DL_ROOT_BYTES
    profile_bytes = sum(int(row["size_bytes"]) for row in profile_rows)
    if profile_bytes != expected_profile_bytes:
        raise RuntimeError(
            f"profile manifest has {profile_bytes:,} bytes; expected {expected_profile_bytes:,}"
        )

    existing_data_entries = [
        UploadEntry(
            source=_source_for_mask(row, mask_root),
            relative_key=str(row["relative_key"]),
            size_bytes=int(row["size_bytes"]),
            sha256=str(row["sha256"]),
            content_type="application/octet-stream",
        )
        for row in mask_rows
    ]
    existing_data_entries.extend(
        UploadEntry(
            source=_source_for_profile(row, cp_profile_root, dl_profile_root),
            relative_key=str(row["relative_key"]),
            size_bytes=int(row["size_bytes"]),
            sha256=str(row["sha256"]),
            content_type="application/vnd.apache.parquet",
        )
        for row in profile_rows
    )
    # Keep the previously published inventory as an exact leading segment so a
    # verified completed checkpoint can be adopted without re-uploading it.
    existing_data_entries.sort(key=lambda entry: entry.relative_key)
    object_feature_entries = [
        UploadEntry(
            source=_source_for_object_feature(row, mask_root),
            relative_key=str(row["relative_key"]),
            size_bytes=int(row["size_bytes"]),
            sha256=str(row["sha256"]),
            content_type="application/vnd.apache.parquet",
        )
        for row in object_feature_rows
    ]
    object_feature_entries.sort(key=lambda entry: entry.relative_key)
    data_entries = [*existing_data_entries, *object_feature_entries]
    keys = [entry.relative_key for entry in data_entries]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Target-2 package has duplicate data destination keys")
    if any(key.startswith("manifests/") or key == "README.md" for key in keys):
        raise RuntimeError("Target-2 data inventory overlaps withheld metadata keys")

    metadata_entries = [
        _metadata_entry(release_root, key) for key in METADATA_RELATIVE_KEYS
    ]
    return data_entries, metadata_entries, provenance


def verify_local_entry(entry: UploadEntry) -> None:
    if not entry.source.is_file():
        raise RuntimeError(f"missing Target-2 source artifact: {entry.source}")
    size = entry.source.stat().st_size
    if size != entry.size_bytes:
        raise RuntimeError(
            f"Target-2 source size changed: {entry.source} ({size} != {entry.size_bytes})"
        )
    digest = sha256_file(entry.source)
    if digest != entry.sha256:
        raise RuntimeError(f"Target-2 source hash changed: {entry.source}")


def verify_local_inventory(entries: Sequence[UploadEntry], workers: int) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(verify_local_entry, entries))


def validate_checkpoint(
    checkpoint: Mapping[str, Any], entries: Sequence[UploadEntry], expected_digest: str
) -> tuple[int, int]:
    recorded_destination = checkpoint.get("destination_prefix")
    if recorded_destination != DESTINATION_ROOT:
        raise RuntimeError(
            "checkpoint destination mismatch; use a new checkpoint path "
            f"(recorded={recorded_destination!r}, expected={DESTINATION_ROOT!r})"
        )
    if checkpoint.get("inventory_sha256") != expected_digest:
        raise RuntimeError("checkpoint inventory digest mismatch")
    if int(checkpoint.get("file_count", -1)) != len(entries):
        raise RuntimeError("checkpoint file_count does not match the artifact inventory")
    start = int(checkpoint.get("next_index", 0))
    if start < 0 or start > len(entries):
        raise RuntimeError("checkpoint next_index is outside the artifact inventory")
    if start and checkpoint.get("last_relative_key") != entries[start - 1].relative_key:
        raise RuntimeError("checkpoint ordering mismatch")
    uploaded_bytes = int(checkpoint.get("uploaded_bytes", -1))
    expected_uploaded_bytes = sum(entry.size_bytes for entry in entries[:start])
    if uploaded_bytes != expected_uploaded_bytes:
        raise RuntimeError(
            "checkpoint uploaded_bytes does not match its completed prefix"
        )
    return start, uploaded_bytes


class RefreshingS3Client:
    """Vend and proactively refresh a thread-safe S3 client."""

    def __init__(self, workers: int) -> None:
        self.workers = workers
        self._lock = threading.Lock()
        self._client: Any = None
        self._expiration = 0.0

    @staticmethod
    def _base_credentials() -> tuple[str, str]:
        key_path = Path(os.environ.get("CPG_KEY_ID_FILE", Path.home() / ".cpg_key_id"))
        secret_path = Path(
            os.environ.get("CPG_SECRET_FILE", Path.home() / ".cpg_access_key")
        )
        return key_path.read_text().strip(), secret_path.read_text().strip()

    def _refresh(self) -> None:
        access_key, secret_key = self._base_credentials()
        control = boto3.client(
            "s3control",
            region_name=REGION,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(retries={"mode": "adaptive", "max_attempts": 10}),
        )
        response = control.get_data_access(
            AccountId=ACCOUNT_ID,
            Target=GRANT_TARGET,
            Permission="READWRITE",
            DurationSeconds=43_200,
        )
        credentials = response["Credentials"]
        expiration = credentials["Expiration"]
        self._client = boto3.client(
            "s3",
            region_name=REGION,
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
            config=Config(
                max_pool_connections=self.workers + 16,
                retries={"mode": "adaptive", "max_attempts": 12},
                connect_timeout=20,
                read_timeout=120,
            ),
        )
        self._expiration = expiration.timestamp()
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] refreshed temporary "
            f"Target-2 upload credentials; expires={expiration.isoformat()}",
            flush=True,
        )

    def client(self, *, force_refresh: bool = False) -> Any:
        if (
            not force_refresh
            and self._client is not None
            and time.time() < self._expiration - 900
        ):
            return self._client
        with self._lock:
            if (
                force_refresh
                or self._client is None
                or time.time() >= self._expiration - 900
            ):
                self._refresh()
            return self._client


def upload_entry(
    manager: RefreshingS3Client,
    entry: UploadEntry,
    *,
    max_attempts: int = 20,
    in_memory: bool = False,
) -> int:
    expected_checksum = base64.b64encode(bytes.fromhex(entry.sha256)).decode("ascii")
    for attempt in range(1, max_attempts + 1):
        try:
            if in_memory:
                body = entry.source.read_bytes()
                if (
                    len(body) != entry.size_bytes
                    or hashlib.sha256(body).hexdigest() != entry.sha256
                ):
                    raise RuntimeError(
                        f"metadata source changed before upload: {entry.source}"
                    )
                manager.client().put_object(
                    Bucket=BUCKET,
                    Key=entry.destination_key,
                    Body=body,
                    ContentLength=entry.size_bytes,
                    ContentType=entry.content_type,
                    ChecksumSHA256=expected_checksum,
                    ExpectedBucketOwner=ACCOUNT_ID,
                )
            else:
                with entry.source.open("rb") as stream:
                    manager.client().put_object(
                        Bucket=BUCKET,
                        Key=entry.destination_key,
                        Body=stream,
                        ContentLength=entry.size_bytes,
                        ContentType=entry.content_type,
                        ChecksumSHA256=expected_checksum,
                        ExpectedBucketOwner=ACCOUNT_ID,
                    )
            return entry.size_bytes
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "unknown")
            if code in AUTH_REFRESH_CODES:
                manager.client(force_refresh=True)
            if attempt == max_attempts:
                raise
            time.sleep(min(60.0, 0.5 * (2 ** min(attempt, 7))))
        except (BotoCoreError, OSError):
            if attempt == max_attempts:
                raise
            time.sleep(min(60.0, 0.5 * (2 ** min(attempt, 7))))
    raise AssertionError("unreachable")


def remote_inventory(client: Any, prefix: str) -> dict[str, int]:
    paginator = client.get_paginator("list_objects_v2")
    objects: dict[str, int] = {}
    for page in paginator.paginate(
        Bucket=BUCKET, Prefix=prefix, ExpectedBucketOwner=ACCOUNT_ID
    ):
        for row in page.get("Contents", []):
            key = str(row["Key"])
            if key in objects:
                raise RuntimeError(f"duplicate key returned by S3 listing: {key}")
            objects[key] = int(row["Size"])
    return objects


def _verify_remote_checksum(
    manager: RefreshingS3Client,
    entry: UploadEntry,
    *,
    max_attempts: int = 20,
) -> None:
    expected = base64.b64encode(bytes.fromhex(entry.sha256)).decode("ascii")
    for attempt in range(1, max_attempts + 1):
        try:
            response = manager.client().head_object(
                Bucket=BUCKET,
                Key=entry.destination_key,
                ChecksumMode="ENABLED",
                ExpectedBucketOwner=ACCOUNT_ID,
            )
            actual = response.get("ChecksumSHA256")
            if actual != expected:
                raise RuntimeError(
                    f"remote SHA-256 mismatch for s3://{BUCKET}/{entry.destination_key}: "
                    f"actual={actual!r}, expected={expected!r}"
                )
            return
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "unknown")
            if code in AUTH_REFRESH_CODES:
                manager.client(force_refresh=True)
            if attempt == max_attempts:
                raise
            time.sleep(min(60.0, 0.5 * (2 ** min(attempt, 7))))
        except BotoCoreError:
            if attempt == max_attempts:
                raise
            time.sleep(min(60.0, 0.5 * (2 ** min(attempt, 7))))
    raise AssertionError("unreachable")


def verify_remote_entries(
    manager: RefreshingS3Client,
    *,
    entries: Sequence[UploadEntry],
    prefix: str,
    workers: int,
) -> dict[str, Any]:
    expected = {entry.destination_key: entry.size_bytes for entry in entries}
    if len(expected) != len(entries):
        raise RuntimeError("expected remote inventory contains duplicate keys")
    actual = remote_inventory(manager.client(), prefix)
    if actual != expected:
        expected_keys = set(expected)
        actual_keys = set(actual)
        wrong_size = sorted(
            key
            for key in expected_keys & actual_keys
            if expected[key] != actual[key]
        )
        raise RuntimeError(
            f"remote inventory mismatch for s3://{BUCKET}/{prefix}: "
            f"missing={len(expected_keys - actual_keys)}, "
            f"extra={len(actual_keys - expected_keys)}, "
            f"wrong_size={len(wrong_size)}"
        )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(lambda entry: _verify_remote_checksum(manager, entry), entries))
    return {
        "prefix": prefix,
        "object_count": len(actual),
        "bytes": sum(actual.values()),
        "sha256_checksums_verified": len(entries),
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }


def verify_root_ready_for_metadata(
    manager: RefreshingS3Client,
    *,
    data_entries: Sequence[UploadEntry],
    metadata_entries: Sequence[UploadEntry],
) -> None:
    data = {entry.destination_key: entry.size_bytes for entry in data_entries}
    metadata_keys = {entry.destination_key for entry in metadata_entries}
    actual = remote_inventory(manager.client(), f"{DESTINATION_ROOT}/")
    actual_data = {key: size for key, size in actual.items() if key in data}
    unexpected = set(actual) - set(data) - metadata_keys
    missing = set(data) - set(actual_data)
    wrong_size = {
        key for key in actual_data if actual_data[key] != data[key]
    }
    if unexpected or missing or wrong_size:
        raise RuntimeError(
            "Target-2 root is not safe for metadata replacement: "
            f"unexpected={len(unexpected)}, missing_data={len(missing)}, "
            f"wrong_size={len(wrong_size)}"
        )


def adopt_existing_verified_data(
    *,
    manager: RefreshingS3Client,
    entries: Sequence[UploadEntry],
    checkpoint_path: Path,
    workers: int,
) -> None:
    if not checkpoint_path.is_file():
        raise RuntimeError(
            "--adopt-existing-verified requires the completed prior checkpoint"
        )
    checkpoint = json.loads(checkpoint_path.read_text())
    new_digest = inventory_digest(entries)
    if checkpoint.get("inventory_sha256") == new_digest:
        return
    if checkpoint.get("destination_prefix") != DESTINATION_ROOT:
        raise RuntimeError("prior checkpoint destination does not match Target-2")
    if not checkpoint.get("complete") or not checkpoint.get("metadata_published"):
        raise RuntimeError("prior Target-2 checkpoint is not complete")

    existing_entries = [
        entry
        for entry in entries
        if not entry.relative_key.startswith("object_features/")
    ]
    object_feature_entries = [
        entry for entry in entries if entry.relative_key.startswith("object_features/")
    ]
    if not existing_entries or not object_feature_entries:
        raise RuntimeError("extended Target-2 inventory is missing an expected component")
    if entries[: len(existing_entries)] != existing_entries:
        raise RuntimeError("existing Target-2 inventory is not the leading checkpoint segment")
    if checkpoint.get("inventory_sha256") != inventory_digest(existing_entries):
        raise RuntimeError("prior checkpoint does not bind the existing data inventory")
    object_prefix = f"{DESTINATION_ROOT}/object_features/"
    unexpected = remote_inventory(manager.client(), object_prefix)
    if unexpected:
        raise RuntimeError(
            "cannot adopt existing data because the new object-feature prefix "
            f"is not empty ({len(unexpected):,} objects)"
        )

    segmentation_entries = [
        entry
        for entry in existing_entries
        if entry.relative_key.startswith("segmentation/")
    ]
    compact_profile_entries = [
        entry
        for entry in existing_entries
        if entry.relative_key.startswith("profiles/")
    ]
    verification = {
        "segmentation": verify_remote_entries(
            manager,
            entries=segmentation_entries,
            prefix=f"{DESTINATION_ROOT}/segmentation/",
            workers=workers,
        ),
        "compact_profiles": verify_remote_entries(
            manager,
            entries=compact_profile_entries,
            prefix=f"{DESTINATION_ROOT}/profiles/",
            workers=workers,
        ),
    }
    payload = {
        "destination_prefix": DESTINATION_ROOT,
        "inventory_sha256": new_digest,
        "next_index": len(existing_entries),
        "file_count": len(entries),
        "last_relative_key": existing_entries[-1].relative_key,
        "uploaded_bytes": sum(entry.size_bytes for entry in existing_entries),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "data_upload_complete": False,
        "metadata_published": False,
        "complete": False,
        "adopted_existing_remote_verification": verification,
    }
    atomic_write_json(checkpoint_path, payload)
    print(
        f"Adopted {len(existing_entries):,} exactly reverified existing data "
        f"objects; {len(object_feature_entries):,} new object-feature objects remain",
        flush=True,
    )


def upload_data(
    *,
    manager: RefreshingS3Client,
    entries: Sequence[UploadEntry],
    checkpoint_path: Path,
    workers: int,
    batch_size: int,
) -> dict[str, Any]:
    digest = inventory_digest(entries)
    start = 0
    uploaded_bytes = 0
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        start, uploaded_bytes = validate_checkpoint(checkpoint, entries, digest)
    if start == len(entries):
        print("Target-2 data inventory already checkpointed complete", flush=True)
        return json.loads(checkpoint_path.read_text())

    completed = start
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while completed < len(entries):
            end = min(completed + batch_size, len(entries))
            batch = entries[completed:end]
            sizes = executor.map(lambda entry: upload_entry(manager, entry), batch)
            for entry, size in zip(batch, sizes, strict=True):
                completed += 1
                uploaded_bytes += size
                if completed % 1_000 == 0 or completed == len(entries):
                    elapsed = max(time.monotonic() - started, 0.001)
                    payload = {
                        "destination_prefix": DESTINATION_ROOT,
                        "inventory_sha256": digest,
                        "next_index": completed,
                        "file_count": len(entries),
                        "last_relative_key": entry.relative_key,
                        "uploaded_bytes": uploaded_bytes,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "data_upload_complete": completed == len(entries),
                        "metadata_published": False,
                        "complete": False,
                    }
                    atomic_write_json(checkpoint_path, payload)
                    if completed % 10_000 == 0 or completed == len(entries):
                        rate = (completed - start) / elapsed
                        print(
                            f"[{payload['updated_at']}] Target-2 data "
                            f"uploaded={completed:,}/{len(entries):,} rate={rate:,.1f} files/s",
                            flush=True,
                        )
    return json.loads(checkpoint_path.read_text())


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.batch_size < 1:
        raise ValueError("workers and batch-size must be positive")
    data_entries, metadata_entries, _ = load_package(
        release_root=args.release_root,
        mask_root=args.mask_root,
        cp_profile_root=args.cp_profile_root,
        dl_profile_root=args.dl_profile_root,
    )
    print(
        f"Validating {len(data_entries):,} Target-2 data objects and "
        f"{len(metadata_entries)} withheld metadata objects...",
        flush=True,
    )
    verify_local_inventory([*data_entries, *metadata_entries], args.workers)
    expected_data_bytes = sum(entry.size_bytes for entry in data_entries)
    expected_metadata_bytes = sum(entry.size_bytes for entry in metadata_entries)
    print(
        f"Validated local package: data={len(data_entries):,} objects/"
        f"{expected_data_bytes:,} bytes; metadata={len(metadata_entries)} objects/"
        f"{expected_metadata_bytes:,} bytes",
        flush=True,
    )

    if not args.apply:
        profile_samples = [
            entry
            for entry in data_entries
            if entry.relative_key.startswith("profiles/")
        ][:3]
        object_feature_samples = [
            entry
            for entry in data_entries
            if entry.relative_key.startswith("object_features/")
        ][:3]
        mask_samples = [
            entry
            for entry in data_entries
            if entry.relative_key.startswith("segmentation/")
        ][:3]
        for entry in [*profile_samples, *object_feature_samples, *mask_samples]:
            print(f"DRY-RUN {entry.source} -> s3://{BUCKET}/{entry.destination_key}")
        print("DRY-RUN package metadata remain withheld until data verification")
        return 0

    manager = RefreshingS3Client(args.workers)
    manager.client()
    if args.adopt_existing_verified:
        adopt_existing_verified_data(
            manager=manager,
            entries=data_entries,
            checkpoint_path=args.checkpoint,
            workers=args.workers,
        )
    checkpoint = upload_data(
        manager=manager,
        entries=data_entries,
        checkpoint_path=args.checkpoint,
        workers=args.workers,
        batch_size=args.batch_size,
    )

    segmentation_entries = [
        entry
        for entry in data_entries
        if entry.relative_key.startswith("segmentation/")
    ]
    profile_entries = [
        entry for entry in data_entries if entry.relative_key.startswith("profiles/")
    ]
    object_feature_entries = [
        entry
        for entry in data_entries
        if entry.relative_key.startswith("object_features/")
    ]
    verification = {
        "segmentation": verify_remote_entries(
            manager,
            entries=segmentation_entries,
            prefix=f"{DESTINATION_ROOT}/segmentation/",
            workers=args.workers,
        ),
        "profiles": verify_remote_entries(
            manager,
            entries=profile_entries,
            prefix=f"{DESTINATION_ROOT}/profiles/",
            workers=args.workers,
        ),
        "object_features": verify_remote_entries(
            manager,
            entries=object_feature_entries,
            prefix=f"{DESTINATION_ROOT}/object_features/",
            workers=args.workers,
        ),
    }
    checkpoint["data_remote_verification"] = verification
    atomic_write_json(args.checkpoint, checkpoint)

    # Reject unknown whole-root objects before replacing any package metadata.
    verify_root_ready_for_metadata(
        manager,
        data_entries=data_entries,
        metadata_entries=metadata_entries,
    )
    # Publish manifests/provenance first and README last as the discoverability
    # marker. Any interruption leaves the old README in place and the
    # checkpoint incomplete; a rerun safely overwrites the metadata set.
    ordered_metadata = sorted(
        metadata_entries,
        key=lambda entry: (entry.relative_key == "README.md", entry.relative_key),
    )
    for entry in ordered_metadata:
        # Small metadata objects are uploaded from verified immutable byte
        # strings. This avoids streaming-checksum ambiguity and makes each
        # replacement payload atomic from the client's perspective.
        upload_entry(manager, entry, in_memory=True)

    final = verify_remote_entries(
        manager,
        entries=[*data_entries, *metadata_entries],
        prefix=f"{DESTINATION_ROOT}/",
        workers=args.workers,
    )
    checkpoint = json.loads(args.checkpoint.read_text())
    checkpoint["data_upload_complete"] = True
    checkpoint["metadata_published"] = True
    checkpoint["complete"] = True
    checkpoint["final_remote_verification"] = final
    checkpoint["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(args.checkpoint, checkpoint)
    print(
        f"Target-2 staging upload COMPLETE: objects={final['object_count']:,}, "
        f"bytes={final['bytes']:,}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
