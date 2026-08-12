#!/usr/bin/env python3
"""Rebuild lossless Zstd images from original CPG TIFFs on demand.

The exact site manifest and site-major array organization come from the frozen
MQ release. Original public TIFFs are streamed into memory, stacked in
AGP/DNA/ER/Mito/RNA order, compressed as one Zarr v3 chunk per site, and then
discarded. No raw TIFF cache is created.

A clean building store is resumable. The legacy ``zstd.zarr`` is preserved and
replaced only after the complete building store matches the canonical MQ site
count and digest.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
import shutil
import sys
import threading
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import duckdb
import numpy as np
import tifffile
import zarr
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, ConnectionClosedError
from zarr.codecs import BloscCodec

from validate_release import (
    CANONICAL_CODEC,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_METADATA_ROOT,
    EXPECTED_SITE_COUNT,
)

CHANNELS = ("AGP", "DNA", "ER", "Mito", "RNA")
# This public object is permanently zero-filled rather than a valid TIFF. Its
# local raw copy and S3 ETag/size agree. Reconstruct only this exact channel as
# a zero plane using the canonical site schema; any other invalid TIFF remains
# a hard failure.
_ZERO_FILLED_SOURCE_7_PREFIX = (
    "s3://cellpainting-gallery/cpg0016-jump/source_7/images/20210727_Run3/"
    "images/CP3-SC1-18/"
)
_ZERO_FILLED_SOURCE_7_NAMES = (
    "CP3-SC1-18_I22_T0001F003L01A03Z01C04.tif",  # site 2, ER
    "CP3-SC1-18_I22_T0001F004L01A01Z01C01.tif",  # site 3, DNA
    "CP3-SC1-18_I22_T0001F004L01A01Z01C02.tif",  # site 3, Mito
    "CP3-SC1-18_I22_T0001F004L01A02Z01C03.tif",  # site 3, RNA
)
KNOWN_ZERO_FILLED_TIFFS = {
    _ZERO_FILLED_SOURCE_7_PREFIX + name: {
        "etag": "d4ffe90e54a5af4e2009e5984da69f03",
        "size": 2_768_896,
    }
    for name in _ZERO_FILLED_SOURCE_7_NAMES
}
CANONICAL_DIGEST = "4ea6ea3f5457c33a1412a80a89d8696d4f8e77474cf449e75db7ce6ba98685e2"
DEFAULT_INDEX = DEFAULT_METADATA_ROOT / "jump_lite_site_index.parquet"
DEFAULT_REBUILD_ROOT = Path("/work/datasets/jump_lite/zstd_rebuild/v1.0")
DEFAULT_OUTPUT = DEFAULT_REBUILD_ROOT / "zstd.building.zarr"
DEFAULT_FINAL = DEFAULT_REBUILD_ROOT / "zstd.zarr"
DEFAULT_STATE = Path("/work/datasets/jump_lite/zstd_rebuild_state/v1.0")
DEFAULT_QUARANTINE = Path("/work/datasets/jump_lite/quarantine")
_thread_local = threading.local()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--mq-root", type=Path, default=DEFAULT_IMAGE_ROOT / CANONICAL_CODEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--final-path", type=Path, default=DEFAULT_FINAL)
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--quarantine-root", type=Path, default=DEFAULT_QUARANTINE)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-sites", type=int, help="test only; disables final replacement")
    parser.add_argument("--apply", action="store_true", help="download and write data")
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def iter_manifest(index: Path, batch_size: int, max_sites: int | None) -> Iterator[list[tuple[Any, ...]]]:
    limit = f" LIMIT {max_sites}" if max_sites is not None else ""
    columns = ", ".join(
        ["Metadata_Site_Key"] + [f"URL_Orig{channel}" for channel in CHANNELS]
    )
    connection = duckdb.connect()
    cursor = connection.execute(
        f"SELECT {columns} FROM read_parquet('{sql_path(index)}') "
        f"ORDER BY Metadata_Site_Key{limit}"
    )
    try:
        while rows := cursor.fetchmany(batch_size):
            yield rows
    finally:
        connection.close()


def manifest_summary(index: Path) -> tuple[int, str]:
    connection = duckdb.connect()
    try:
        keys = [
            row[0]
            for row in connection.execute(
                f"SELECT Metadata_Site_Key FROM read_parquet('{sql_path(index)}') "
                "ORDER BY Metadata_Site_Key"
            ).fetchall()
        ]
    finally:
        connection.close()
    digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    return len(keys), digest


def unsigned_s3_client() -> Any:
    client = getattr(_thread_local, "s3_client", None)
    if client is None:
        client = boto3.client(
            "s3",
            region_name="us-east-1",
            config=Config(
                signature_version=UNSIGNED,
                max_pool_connections=8,
                retries={"mode": "adaptive", "max_attempts": 12},
                connect_timeout=20,
                read_timeout=180,
            ),
        )
        _thread_local.s3_client = client
    return client


def reset_unsigned_s3_client() -> None:
    """Discard a worker's client after a corrupt or broken response."""
    client = getattr(_thread_local, "s3_client", None)
    if client is not None:
        try:
            client.close()
        finally:
            delattr(_thread_local, "s3_client")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path:
        raise ValueError(f"invalid S3 URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def download_tiff(
    uri: str,
    max_attempts: int = 20,
    *,
    expected_shape: Sequence[int] | None = None,
    expected_dtype: str | None = None,
) -> tuple[np.ndarray, int]:
    bucket, key = parse_s3_uri(uri)
    for attempt in range(1, max_attempts + 1):
        try:
            response = unsigned_s3_client().get_object(Bucket=bucket, Key=key)
            payload = response["Body"].read()
            expected_length = int(response.get("ContentLength", len(payload)))
            if len(payload) != expected_length:
                raise OSError(
                    f"truncated S3 payload for {uri}: {len(payload)} != {expected_length}"
                )
            known_zero = KNOWN_ZERO_FILLED_TIFFS.get(uri)
            if known_zero is not None and payload and not any(payload):
                etag = str(response.get("ETag", "")).strip('"')
                if (
                    len(payload) != known_zero["size"]
                    or etag != known_zero["etag"]
                    or expected_shape is None
                    or expected_dtype is None
                ):
                    raise RuntimeError(
                        f"zero-filled TIFF identity/schema mismatch for {uri}"
                    )
                print(
                    f"[{now()}] reconstructing known zero-filled TIFF as a zero plane: {uri}",
                    file=sys.stderr,
                    flush=True,
                )
                return np.zeros(tuple(expected_shape), dtype=np.dtype(expected_dtype)), len(payload)
            image = tifffile.imread(io.BytesIO(payload))
            if image.ndim != 2:
                raise ValueError(f"expected a 2D TIFF at {uri}; got shape {image.shape}")
            return image, len(payload)
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "unknown")
            if code in {"NoSuchKey", "AccessDenied", "InvalidObjectState"}:
                raise
            if attempt == max_attempts:
                raise
        except (BotoCoreError, ConnectionClosedError, tifffile.TiffFileError, OSError) as error:
            # A truncated or transiently corrupt payload can return HTTP 200 but
            # fail TIFF decoding. Reusing that connection repeatedly produced
            # the same invalid zero header, so force a fresh client and endpoint.
            reset_unsigned_s3_client()
            if attempt == max_attempts:
                raise RuntimeError(
                    f"failed to download a valid TIFF after {max_attempts} attempts: {uri}"
                ) from error
            print(
                f"[{now()}] retrying TIFF attempt={attempt}/{max_attempts} "
                f"error={type(error).__name__} uri={uri}",
                file=sys.stderr,
                flush=True,
            )
        time.sleep(min(60.0, 0.5 * (2 ** min(attempt, 7))))
    raise AssertionError("unreachable")


def mq_schema(mq_root: Path, site_key: str) -> tuple[tuple[int, ...], str]:
    metadata_path = mq_root / site_key / ".zarray"
    metadata = json.loads(metadata_path.read_text())
    return tuple(metadata["shape"]), np.dtype(metadata["dtype"]).str


def output_complete(output: Path, site_key: str, expected_shape: Sequence[int] | None = None) -> bool:
    site_path = output / site_key
    metadata_path = site_path / "zarr.json"
    chunk_root = site_path / "c"
    if not metadata_path.is_file() or not chunk_root.is_dir():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if metadata.get("node_type") != "array":
        return False
    if expected_shape is not None and tuple(metadata.get("shape", ())) != tuple(expected_shape):
        return False
    return any(entry.is_file() for entry in chunk_root.rglob("*"))


def compressed_bytes(output: Path, site_key: str) -> int:
    return sum(
        entry.stat().st_size
        for entry in (output / site_key).rglob("*")
        if entry.is_file()
    )


def build_site(
    row: tuple[Any, ...],
    *,
    output: Path,
    mq_root: Path,
    compressor: BloscCodec,
) -> dict[str, Any]:
    site_key = str(row[0])
    urls = tuple(str(value) for value in row[1:])
    expected_shape, expected_dtype = mq_schema(mq_root, site_key)
    if output_complete(output, site_key, expected_shape):
        return {"site_key": site_key, "status": "skipped", "downloaded_bytes": 0}

    partial = output / site_key
    if partial.exists():
        shutil.rmtree(partial)

    images: list[np.ndarray] = []
    downloaded = 0
    for uri in urls:
        image, size = download_tiff(
            uri,
            expected_shape=expected_shape[1:],
            expected_dtype=expected_dtype,
        )
        images.append(image)
        downloaded += size

    shapes = {image.shape for image in images}
    dtypes = {image.dtype.str for image in images}
    if len(shapes) != 1 or len(dtypes) != 1:
        raise ValueError(
            f"channel mismatch for {site_key}: shapes={sorted(shapes)}, dtypes={sorted(dtypes)}"
        )
    stack = np.stack(images, axis=0)
    if tuple(stack.shape) != expected_shape:
        raise ValueError(
            f"MQ/original shape mismatch for {site_key}: {stack.shape} != {expected_shape}"
        )
    if stack.dtype.str != expected_dtype:
        raise ValueError(
            f"MQ/original dtype mismatch for {site_key}: {stack.dtype.str} != {expected_dtype}"
        )

    store = zarr.storage.LocalStore(output)
    array = zarr.create_array(
        store=store,
        name=site_key,
        shape=stack.shape,
        chunks=stack.shape,
        dtype=stack.dtype,
        compressors=compressor,
        zarr_format=3,
    )
    array[:] = stack
    del stack, images

    if not output_complete(output, site_key, expected_shape):
        raise RuntimeError(f"site write did not produce a complete array: {site_key}")
    return {
        "site_key": site_key,
        "status": "created",
        "downloaded_bytes": downloaded,
        "compressed_bytes": compressed_bytes(output, site_key),
    }


def validate_building_store(output: Path, expected_count: int, expected_digest: str) -> None:
    keys: list[str] = []
    incomplete: list[str] = []
    for entry in os.scandir(output):
        if not entry.is_dir(follow_symlinks=False):
            continue
        keys.append(entry.name)
        if not output_complete(output, entry.name):
            incomplete.append(entry.name)
    digest = hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
    if len(keys) != expected_count:
        raise RuntimeError(
            f"building Zstd has {len(keys):,} site arrays; expected {expected_count:,}"
        )
    if digest != expected_digest:
        raise RuntimeError(f"building Zstd site digest mismatch: {digest}")
    if incomplete:
        raise RuntimeError(f"building Zstd has incomplete arrays: {incomplete[:10]}")


def finalize(args: argparse.Namespace) -> Path | None:
    if args.max_sites is not None:
        return None
    validate_building_store(args.output, EXPECTED_SITE_COUNT, CANONICAL_DIGEST)
    quarantined: Path | None = None
    if args.final_path.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.quarantine_root.mkdir(parents=True, exist_ok=True)
        quarantined = args.quarantine_root / f"legacy_zstd_incomplete_{timestamp}.zarr"
        if quarantined.exists():
            raise RuntimeError(f"quarantine destination already exists: {quarantined}")
        os.replace(args.final_path, quarantined)
    os.replace(args.output, args.final_path)
    return quarantined


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.batch_size < 1:
        raise ValueError("workers and batch-size must be positive")
    if not args.index.is_file() or not args.mq_root.is_dir():
        raise RuntimeError("site index or canonical MQ store is missing")
    if args.output == args.final_path:
        raise RuntimeError("building output must differ from the final zstd.zarr path")

    args.state_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.state_root / "rebuild.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another Zstd rebuild already holds the lock") from None

        count, digest = manifest_summary(args.index)
        if count != EXPECTED_SITE_COUNT or digest != CANONICAL_DIGEST:
            raise RuntimeError(
                f"input manifest mismatch: count={count:,}, digest={digest}"
            )
        print(
            f"[{now()}] canonical manifest sites={count:,} digest={digest}",
            flush=True,
        )
        if not args.apply:
            dry_run_sites = min(args.max_sites or 5, 5)
            rows = next(
                iter_manifest(args.index, min(args.batch_size, dry_run_sites), dry_run_sites)
            )
            for row in rows:
                print(
                    f"DRY-RUN site={row[0]} channels={len(row) - 1} "
                    f"output={args.output / str(row[0])}"
                )
            return 0

        args.output.mkdir(parents=True, exist_ok=True)
        zarr.open_group(store=zarr.storage.LocalStore(args.output), mode="a", zarr_format=3)
        compressor = BloscCodec(cname="zstd", clevel=9, shuffle="bitshuffle")
        checkpoint_path = args.state_root / "checkpoint.json"
        started = time.monotonic()
        processed = created = skipped = downloaded = compressed = 0

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for rows in iter_manifest(args.index, args.batch_size, args.max_sites):
                results = executor.map(
                    lambda row: build_site(
                        row,
                        output=args.output,
                        mq_root=args.mq_root,
                        compressor=compressor,
                    ),
                    rows,
                )
                for result in results:
                    processed += 1
                    created += result["status"] == "created"
                    skipped += result["status"] == "skipped"
                    downloaded += int(result.get("downloaded_bytes", 0))
                    compressed += int(result.get("compressed_bytes", 0))
                elapsed = max(time.monotonic() - started, 0.001)
                payload = {
                    "processed_manifest_sites": processed,
                    "target_sites": args.max_sites or EXPECTED_SITE_COUNT,
                    "created_sites": created,
                    "skipped_complete_sites": skipped,
                    "downloaded_bytes_this_run": downloaded,
                    "compressed_bytes_created_this_run": compressed,
                    "last_site_key": rows[-1][0],
                    "sites_per_second": processed / elapsed,
                    "updated_at": now(),
                    "output": str(args.output),
                    "complete": False,
                }
                atomic_write_json(checkpoint_path, payload)
                if processed % (args.batch_size * 10) == 0:
                    print(
                        f"[{payload['updated_at']}] processed={processed:,}/"
                        f"{payload['target_sites']:,} created={created:,} skipped={skipped:,} "
                        f"rate={payload['sites_per_second']:.2f} sites/s",
                        flush=True,
                    )

        if args.max_sites is not None:
            print(f"[{now()}] test build complete: {processed:,} sites", flush=True)
            return 0

        print(f"[{now()}] validating complete building store", flush=True)
        quarantined = finalize(args)
        payload = json.loads(checkpoint_path.read_text())
        payload.update(
            {
                "complete": True,
                "final_path": str(args.final_path),
                "legacy_quarantine_path": str(quarantined) if quarantined else None,
                "completed_at": now(),
            }
        )
        atomic_write_json(checkpoint_path, payload)
        print(
            f"[{now()}] complete final={args.final_path} legacy={quarantined}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        raise
