#!/usr/bin/env python3
"""Stream checkpoint-confirmed JUMP-Lite Zstd arrays to CPG staging.

The lossless Zstd store is built concurrently by
``rebuild_zstd_from_originals.py``.  That builder writes its checkpoint only
after every array in a manifest batch is complete.  This uploader therefore
uploads only canonical sites at or before the checkpoint's ``last_site_key``;
it never scans the currently active batch.

The destination group-level ``zarr.json`` is deliberately withheld until the
builder has finalized and revalidated all 655,101 release arrays.  Consequently the
staging prefix cannot be opened as a Zarr group while it is incomplete.  After
finalization, this process validates the local store again, uploads the root
metadata, and verifies the complete remote object count and byte total.
"""

from __future__ import annotations

import argparse
import bisect
import fcntl
import hashlib
import json
import mimetypes
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import duckdb
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, ConnectionClosedError

from rebuild_zstd_from_originals import (
    CANONICAL_DIGEST,
    DEFAULT_FINAL,
    DEFAULT_INDEX,
    DEFAULT_OUTPUT,
    EXPECTED_SITE_COUNT,
    output_complete,
    validate_building_store,
)

ACCOUNT_ID = "309624411020"
REGION = "us-east-1"
BUCKET = "staging-cellpainting-gallery"
GRANT_TARGET = "s3://staging-cellpainting-gallery/cpg0016-jump/*"
RELEASE_BATCH = "2026_jump_lite_v1.0"
DESTINATION_PREFIX = (
    f"cpg0016-jump/source_all/images/{RELEASE_BATCH}/images_compressed/zstd.zarr"
)
DEFAULT_REBUILD_CHECKPOINT = Path(
    "/work/datasets/jump_lite/zstd_rebuild_state/v1.0/checkpoint.json"
)
DEFAULT_UPLOAD_STATE = Path(
    "/work/datasets/jump_lite/cpg_upload_state/v1.0/zstd/checkpoint.json"
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--building-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--final-root", type=Path, default=DEFAULT_FINAL)
    parser.add_argument(
        "--rebuild-checkpoint", type=Path, default=DEFAULT_REBUILD_CHECKPOINT
    )
    parser.add_argument("--state", type=Path, default=DEFAULT_UPLOAD_STATE)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="upload to CPG staging; otherwise validate and print current safe progress",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def canonical_keys(index: Path) -> list[str]:
    if not index.is_file():
        raise RuntimeError(f"canonical site index is missing: {index}")
    connection = duckdb.connect()
    try:
        keys = [
            str(row[0])
            for row in connection.execute(
                f"SELECT Metadata_Site_Key FROM read_parquet('{sql_path(index)}') "
                "ORDER BY Metadata_Site_Key"
            ).fetchall()
        ]
    finally:
        connection.close()
    digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    if len(keys) != EXPECTED_SITE_COUNT or digest != CANONICAL_DIGEST:
        raise RuntimeError(
            "canonical manifest mismatch: "
            f"count={len(keys):,} digest={digest}"
        )
    if len(keys) != len(set(keys)):
        raise RuntimeError("canonical manifest contains duplicate site keys")
    return keys


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON checkpoint: {path}") from error


def safe_rebuild_progress(
    keys: list[str], checkpoint_path: Path
) -> tuple[int, bool, dict[str, Any] | None]:
    checkpoint = read_json(checkpoint_path)
    if checkpoint is None:
        return 0, False, None
    if int(checkpoint.get("target_sites", -1)) != EXPECTED_SITE_COUNT:
        raise RuntimeError(f"rebuild checkpoint has wrong target: {checkpoint_path}")
    last_key = checkpoint.get("last_site_key")
    if not isinstance(last_key, str):
        return 0, False, checkpoint
    safe_count = bisect.bisect_right(keys, last_key)
    if safe_count == 0 or keys[safe_count - 1] != last_key:
        raise RuntimeError(f"rebuild checkpoint key is not canonical: {last_key}")
    reported = int(checkpoint.get("processed_manifest_sites", -1))
    # A restarted builder scans from the beginning, so this equality is an
    # important guard against using an unrelated or malformed checkpoint.
    if reported != safe_count:
        raise RuntimeError(
            "rebuild checkpoint ordering mismatch: "
            f"reported={reported:,} key_index={safe_count:,}"
        )
    complete = bool(checkpoint.get("complete", False))
    if complete and safe_count != EXPECTED_SITE_COUNT:
        raise RuntimeError("rebuild checkpoint claims completion before the final site")
    return safe_count, complete, checkpoint


def source_root(args: argparse.Namespace) -> Path:
    # During finalization the building directory is atomically renamed before
    # the checkpoint is marked complete, so prefer the final path if present.
    if args.final_root.is_dir():
        return args.final_root
    return args.building_root


def stable_site_files(root: Path, site_key: str) -> list[Path]:
    site = root / site_key
    if not output_complete(root, site_key):
        raise RuntimeError(f"checkpoint-confirmed site is incomplete: {site}")
    metadata = site / "zarr.json"
    chunks = sorted(
        entry
        for entry in (site / "c").rglob("*")
        if entry.is_file() and not entry.is_symlink()
    )
    if not metadata.is_file() or metadata.is_symlink() or len(chunks) != 1:
        raise RuntimeError(
            f"unexpected one-chunk Zarr layout for {site_key}: chunks={len(chunks)}"
        )
    all_files = sorted(
        entry
        for entry in site.rglob("*")
        if entry.is_file() and not entry.is_symlink()
    )
    expected = set(chunks + [metadata])
    if set(all_files) != expected:
        unexpected = sorted(str(path.relative_to(site)) for path in set(all_files) - expected)
        raise RuntimeError(f"unexpected files in {site_key}: {unexpected[:10]}")
    # Publish chunk data before array metadata.
    return chunks + [metadata]


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
        if not key_path.is_file() or not secret_path.is_file():
            raise RuntimeError("CPG Access Grant credential files are unavailable")
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
                read_timeout=300,
                tcp_keepalive=True,
            ),
        )
        self._expiration = expiration.timestamp()
        print(
            f"[{now()}] refreshed temporary Zstd-upload credentials; "
            f"expires={expiration.isoformat()}",
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


def upload_file(
    manager: RefreshingS3Client,
    source: Path,
    key: str,
    *,
    max_attempts: int = 20,
) -> int:
    size = source.stat().st_size
    content_type = "application/json" if source.name == "zarr.json" else (
        mimetypes.guess_type(source.name)[0] or "application/octet-stream"
    )
    for attempt in range(1, max_attempts + 1):
        try:
            with source.open("rb") as stream:
                manager.client().put_object(
                    Bucket=BUCKET,
                    Key=key,
                    Body=stream,
                    ContentLength=size,
                    ContentType=content_type,
                    ExpectedBucketOwner=ACCOUNT_ID,
                )
            return size
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "unknown")
            if code in {
                "AccessDenied",
                "ExpiredToken",
                "InvalidAccessKeyId",
                "RequestExpired",
                "TokenRefreshRequired",
            }:
                manager.client(force_refresh=True)
            if attempt == max_attempts:
                raise
            time.sleep(min(60.0, 0.5 * (2 ** min(attempt, 7))))
        except FileNotFoundError:
            # The builder atomically renames the complete store during finalization;
            # let upload_site immediately retry from the final root.
            raise
        except (BotoCoreError, ConnectionClosedError, OSError):
            if attempt == max_attempts:
                raise
            time.sleep(min(60.0, 0.5 * (2 ** min(attempt, 7))))
    raise AssertionError("unreachable")


def upload_site(
    manager: RefreshingS3Client,
    building_root: Path,
    final_root: Path,
    site_key: str,
) -> tuple[int, int]:
    root = final_root if (final_root / site_key).is_dir() else building_root
    site = root / site_key
    uploaded_bytes = 0
    files = stable_site_files(root, site_key)
    relatives = [source.relative_to(site) for source in files]
    for relative in relatives:
        source = root / site_key / relative
        key = f"{DESTINATION_PREFIX}/{site_key}/{relative.as_posix()}"
        try:
            uploaded_bytes += upload_file(manager, source, key)
        except FileNotFoundError:
            # Finalization renames the complete store atomically.  If it happened
            # after this task listed the site, continue from the new root.
            source = final_root / site_key / relative
            uploaded_bytes += upload_file(manager, source, key)
    return len(files), uploaded_bytes


def load_upload_checkpoint(
    path: Path, keys: list[str]
) -> tuple[int, int, int, bool]:
    checkpoint = read_json(path)
    if checkpoint is None:
        return 0, 0, 0, False
    if checkpoint.get("canonical_site_key_sha256") != CANONICAL_DIGEST:
        raise RuntimeError(f"upload checkpoint digest mismatch: {path}")
    if checkpoint.get("destination_prefix") != DESTINATION_PREFIX:
        raise RuntimeError(f"upload checkpoint destination mismatch: {path}")
    next_index = int(checkpoint.get("next_index", -1))
    if not 0 <= next_index <= len(keys):
        raise RuntimeError(f"upload checkpoint index is invalid: {path}")
    expected_last = keys[next_index - 1] if next_index else None
    if checkpoint.get("last_site_key") != expected_last:
        raise RuntimeError(f"upload checkpoint ordering mismatch: {path}")
    return (
        next_index,
        int(checkpoint.get("uploaded_objects", 0)),
        int(checkpoint.get("uploaded_bytes", 0)),
        bool(checkpoint.get("complete", False)),
    )


def write_upload_checkpoint(
    path: Path,
    keys: list[str],
    *,
    next_index: int,
    uploaded_objects: int,
    uploaded_bytes: int,
    complete: bool,
    remote_verification: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "destination": f"s3://{BUCKET}/{DESTINATION_PREFIX}/",
        "destination_prefix": DESTINATION_PREFIX,
        "canonical_site_key_sha256": CANONICAL_DIGEST,
        "next_index": next_index,
        "site_count": len(keys),
        "last_site_key": keys[next_index - 1] if next_index else None,
        "uploaded_objects": uploaded_objects,
        "uploaded_bytes": uploaded_bytes,
        "root_metadata_published": complete,
        "complete": complete,
        "updated_at": now(),
    }
    if remote_verification is not None:
        payload["remote_verification"] = remote_verification
        payload["completed_at"] = now()
    atomic_write_json(path, payload)


def verify_remote(
    manager: RefreshingS3Client,
    *,
    expected_objects: int,
    expected_bytes: int,
) -> dict[str, Any]:
    prefix = f"{DESTINATION_PREFIX}/"
    count = 0
    size = 0
    continuation: str | None = None
    while True:
        kwargs: dict[str, Any] = {
            "Bucket": BUCKET,
            "Prefix": prefix,
            "MaxKeys": 1000,
            "ExpectedBucketOwner": ACCOUNT_ID,
        }
        if continuation:
            kwargs["ContinuationToken"] = continuation
        response = manager.client().list_objects_v2(**kwargs)
        contents = response.get("Contents", ())
        count += len(contents)
        size += sum(int(entry["Size"]) for entry in contents)
        if not response.get("IsTruncated"):
            break
        continuation = response.get("NextContinuationToken")
        if not continuation:
            raise RuntimeError("truncated S3 listing did not return a continuation token")
    if count != expected_objects or size != expected_bytes:
        raise RuntimeError(
            "remote Zstd verification failed: "
            f"objects={count:,}/{expected_objects:,} bytes={size:,}/{expected_bytes:,}"
        )
    return {
        "bucket": BUCKET,
        "prefix": prefix,
        "object_count": count,
        "total_bytes": size,
        "verified_at": now(),
    }


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.batch_size < 1 or args.poll_seconds <= 0:
        raise ValueError("workers, batch-size, and poll-seconds must be positive")
    if args.building_root == args.final_root:
        raise ValueError("building and final roots must differ")

    keys = canonical_keys(args.index)
    safe_count, rebuild_complete, _ = safe_rebuild_progress(
        keys, args.rebuild_checkpoint
    )
    root = source_root(args)
    if not root.is_dir():
        raise RuntimeError(f"Zstd source store is unavailable: {root}")

    next_index, uploaded_objects, uploaded_bytes, upload_complete = (
        load_upload_checkpoint(args.state, keys)
    )
    if upload_complete:
        print(
            f"Zstd staging upload already complete: {next_index:,}/{len(keys):,}",
            flush=True,
        )
        return 0

    print(
        f"[{now()}] canonical_sites={len(keys):,} safe_sites={safe_count:,} "
        f"resume_index={next_index:,} rebuild_complete={rebuild_complete} "
        f"destination=s3://{BUCKET}/{DESTINATION_PREFIX}/",
        flush=True,
    )
    if next_index < safe_count:
        sample_key = keys[next_index]
        files = stable_site_files(root, sample_key)
        print(
            f"DRY-RUN sample={sample_key} files="
            f"{[path.relative_to(root / sample_key).as_posix() for path in files]}",
            flush=True,
        )
    if not args.apply:
        return 0

    args.state.parent.mkdir(parents=True, exist_ok=True)
    lock_path = args.state.with_suffix(".lock")
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another Zstd staging uploader holds the lock") from None

        manager = RefreshingS3Client(args.workers)
        manager.client()
        started = time.monotonic()
        process_start = next_index

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            while next_index < len(keys):
                safe_count, rebuild_complete, _ = safe_rebuild_progress(
                    keys, args.rebuild_checkpoint
                )
                if safe_count <= next_index:
                    print(
                        f"[{now()}] waiting for rebuild: uploaded={next_index:,} "
                        f"safe={safe_count:,}/{len(keys):,}",
                        flush=True,
                    )
                    time.sleep(args.poll_seconds)
                    continue

                end = min(next_index + args.batch_size, safe_count)
                batch = keys[next_index:end]
                results = executor.map(
                    lambda site_key: upload_site(
                        manager, args.building_root, args.final_root, site_key
                    ),
                    batch,
                )
                batch_objects = 0
                batch_bytes = 0
                for object_count, byte_count in results:
                    batch_objects += object_count
                    batch_bytes += byte_count

                next_index = end
                uploaded_objects += batch_objects
                uploaded_bytes += batch_bytes
                write_upload_checkpoint(
                    args.state,
                    keys,
                    next_index=next_index,
                    uploaded_objects=uploaded_objects,
                    uploaded_bytes=uploaded_bytes,
                    complete=False,
                )
                if next_index % (args.batch_size * 10) == 0 or next_index == len(keys):
                    elapsed = max(time.monotonic() - started, 0.001)
                    rate = (next_index - process_start) / elapsed
                    print(
                        f"[{now()}] uploaded={next_index:,}/{len(keys):,} "
                        f"objects={uploaded_objects:,} bytes={uploaded_bytes:,} "
                        f"rate={rate:,.2f} sites/s safe={safe_count:,}",
                        flush=True,
                    )

        # All site arrays may be uploaded while the builder is performing its
        # final validation.  Do not publish the group root until that succeeds.
        while True:
            safe_count, rebuild_complete, _ = safe_rebuild_progress(
                keys, args.rebuild_checkpoint
            )
            if rebuild_complete:
                break
            print(f"[{now()}] all sites uploaded; waiting for rebuild validation", flush=True)
            time.sleep(args.poll_seconds)

        if not args.final_root.is_dir():
            raise RuntimeError(f"finalized Zstd store is missing: {args.final_root}")
        print(f"[{now()}] validating finalized local Zstd store", flush=True)
        validate_building_store(args.final_root, EXPECTED_SITE_COUNT, CANONICAL_DIGEST)

        root_metadata = args.final_root / "zarr.json"
        if not root_metadata.is_file() or root_metadata.is_symlink():
            raise RuntimeError(f"final root metadata is missing: {root_metadata}")
        root_size = upload_file(
            manager,
            root_metadata,
            f"{DESTINATION_PREFIX}/zarr.json",
        )
        expected_objects = uploaded_objects + 1
        expected_bytes = uploaded_bytes + root_size
        print(
            f"[{now()}] root metadata published; verifying remote objects and bytes",
            flush=True,
        )
        verification = verify_remote(
            manager,
            expected_objects=expected_objects,
            expected_bytes=expected_bytes,
        )
        write_upload_checkpoint(
            args.state,
            keys,
            next_index=next_index,
            uploaded_objects=expected_objects,
            uploaded_bytes=expected_bytes,
            complete=True,
            remote_verification=verification,
        )
        print(
            f"[{now()}] Zstd staging upload complete: "
            f"objects={expected_objects:,} bytes={expected_bytes:,}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
