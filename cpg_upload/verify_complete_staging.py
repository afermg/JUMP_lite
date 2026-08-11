#!/usr/bin/env python3
"""Verify all staged JUMP-Lite JPEG XL and embedding prefixes.

The verifier compares recursive S3 object counts and byte totals with the local
JPEG XL stores and the completed per-variant upload checkpoints. It performs no
S3 writes. Zstd is verified separately by upload_zstd_to_staging.py because its
root metadata is published only after that verifier succeeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from upload_profiles_to_staging import (
    BUCKET,
    CANONICAL_DIGEST,
    EXPECTED_PROFILE_VARIANTS,
    EXPECTED_SITE_COUNT,
    RELEASE_BATCH,
    RefreshingS3Client,
    destination_prefix,
    feature_set_name,
)

IMAGE_ROOT = Path(
    "/work/datasets/jump_lite/images/compressed/compressed_test/"
    "jump_lite_updated"
)
CHECKPOINT_ROOT = Path("/work/datasets/jump_lite/cpg_upload_state/v1.0/profiles")
VALIDATION_REPORT = Path(
    "/work/datasets/jump_lite/cpg_release/final_validation_report.json"
)
OUTPUT_REPORT = Path(
    "/work/datasets/jump_lite/cpg_release/staging_bulk_verification.json"
)
IMAGE_PREFIX_ROOT = (
    "cpg0016-jump/source_all/images/"
    f"{RELEASE_BATCH}/images_compressed"
)
IMAGE_CODECS = (
    "jpegxl_lossy_mq",
    "jpegxl_lossy_hq",
    "jpegxl_lossy_d20",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=IMAGE_ROOT)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--validation-report", type=Path, default=VALIDATION_REPORT)
    parser.add_argument("--output", type=Path, default=OUTPUT_REPORT)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def check_validation_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    if report.get("status") != "ready" or report.get("errors") != []:
        raise RuntimeError(f"final validation report is not successful: {path}")
    if report.get("canonical_site_count") != EXPECTED_SITE_COUNT:
        raise RuntimeError("final validation report has the wrong site count")
    if report.get("canonical_site_key_sha256") != CANONICAL_DIGEST:
        raise RuntimeError("final validation report has the wrong site-key digest")
    if "zstd.zarr" not in report.get("images", {}):
        raise RuntimeError("final validation report does not include Zstd")
    return report


def scan_local_tree(root: Path) -> tuple[int, int]:
    if not root.is_dir():
        raise RuntimeError(f"missing local image store: {root}")
    object_count = 0
    total_bytes = 0
    for directory, _, filenames in os.walk(root, followlinks=False):
        for filename in filenames:
            path = Path(directory, filename)
            if path.is_symlink():
                continue
            stat = path.stat()
            object_count += 1
            total_bytes += stat.st_size
    return object_count, total_bytes


def image_targets(image_root: Path) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(IMAGE_CODECS)) as executor:
        pending = {
            executor.submit(scan_local_tree, image_root / f"{codec}.zarr"): codec
            for codec in IMAGE_CODECS
        }
        results = {
            pending[future]: future.result() for future in as_completed(pending)
        }
    for codec in IMAGE_CODECS:
        count, size = results[codec]
        targets.append(
            {
                "kind": "image",
                "name": codec,
                "prefix": f"{IMAGE_PREFIX_ROOT}/{codec}.zarr/",
                "expected_objects": count,
                "expected_bytes": size,
            }
        )
    return targets


def profile_targets(checkpoint_root: Path) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for model, codecs in sorted(EXPECTED_PROFILE_VARIANTS.items()):
        for codec in sorted(codecs):
            checkpoint_path = checkpoint_root / model / f"{codec}.json"
            checkpoint = json.loads(checkpoint_path.read_text())
            if checkpoint.get("model") != model or checkpoint.get("codec") != codec:
                raise RuntimeError(f"checkpoint identity mismatch: {checkpoint_path}")
            if not checkpoint.get("complete"):
                raise RuntimeError(f"checkpoint is incomplete: {checkpoint_path}")
            if (
                checkpoint.get("file_count") != EXPECTED_SITE_COUNT
                or checkpoint.get("next_index") != EXPECTED_SITE_COUNT
            ):
                raise RuntimeError(f"checkpoint has the wrong object count: {checkpoint_path}")
            feature_set = feature_set_name(model, codec)
            targets.append(
                {
                    "kind": "embedding",
                    "name": feature_set,
                    "prefix": f"{destination_prefix(model, codec)}/",
                    "expected_objects": EXPECTED_SITE_COUNT,
                    "expected_bytes": int(checkpoint["uploaded_bytes"]),
                }
            )
    return targets


def list_prefix(client: Any, target: dict[str, Any]) -> tuple[int, int, int]:
    count = 0
    size = 0
    pages = 0
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=BUCKET,
        Prefix=target["prefix"],
        ExpectedBucketOwner="309624411020",
        PaginationConfig={"PageSize": 1000},
    ):
        pages += 1
        contents = page.get("Contents", ())
        count += len(contents)
        size += sum(int(item["Size"]) for item in contents)
        if pages % 100 == 0:
            print(
                f"[{datetime.now(timezone.utc).isoformat()}] {target['name']}: "
                f"listed {count:,} objects",
                flush=True,
            )
    return count, size, pages


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    validation = check_validation_report(args.validation_report)

    print("Scanning local JPEG XL stores ...", flush=True)
    targets = image_targets(args.image_root)
    targets.extend(profile_targets(args.checkpoint_root))

    results_by_name: dict[tuple[str, str], dict[str, Any]] = {}
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        for result in previous.get("targets", []):
            key = (result.get("kind"), result.get("name"))
            results_by_name[key] = result

    pending_targets = []
    for target in targets:
        key = (target["kind"], target["name"])
        previous = results_by_name.get(key)
        if (
            previous
            and previous.get("match") is True
            and previous.get("expected_objects") == target["expected_objects"]
            and previous.get("expected_bytes") == target["expected_bytes"]
            and previous.get("prefix") == target["prefix"]
        ):
            print(f"Reusing verified checkpoint: {target['name']}", flush=True)
        else:
            results_by_name.pop(key, None)
            pending_targets.append(target)

    def write_progress(status: str) -> None:
        current = sorted(
            results_by_name.values(), key=lambda item: (item["kind"], item["name"])
        )
        atomic_write_json(
            args.output,
            {
                "status": status,
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "bucket": BUCKET,
                "canonical_site_count": EXPECTED_SITE_COUNT,
                "canonical_site_key_sha256": CANONICAL_DIGEST,
                "local_validation_report": str(args.validation_report),
                "local_validated_at": validation.get("validated_at"),
                "target_count": len(targets),
                "verified_target_count": len(current),
                "targets": current,
            },
        )

    print(
        f"Recursively listing {len(pending_targets)} of {len(targets)} staging prefixes ...",
        flush=True,
    )
    if pending_targets:
        manager = RefreshingS3Client(args.workers)
        client = manager.client()
        print_lock = threading.Lock()
        with ThreadPoolExecutor(
            max_workers=min(args.workers, len(pending_targets))
        ) as executor:
            pending = {
                executor.submit(list_prefix, client, target): target
                for target in pending_targets
            }
            for future in as_completed(pending):
                target = pending[future]
                remote_count, remote_bytes, pages = future.result()
                result = {
                    **target,
                    "remote_objects": remote_count,
                    "remote_bytes": remote_bytes,
                    "pages": pages,
                    "objects_match": remote_count == target["expected_objects"],
                    "bytes_match": remote_bytes == target["expected_bytes"],
                }
                result["match"] = result["objects_match"] and result["bytes_match"]
                results_by_name[(target["kind"], target["name"])] = result
                write_progress("running")
                with print_lock:
                    print(
                        f"{target['kind']:9s} {target['name']:40s} "
                        f"objects={remote_count:,}/{target['expected_objects']:,} "
                        f"bytes={remote_bytes:,}/{target['expected_bytes']:,} "
                        f"match={result['match']}",
                        flush=True,
                    )

    results = sorted(
        results_by_name.values(), key=lambda item: (item["kind"], item["name"])
    )
    complete = len(results) == len(targets) and all(item["match"] for item in results)
    write_progress("ready" if complete else "error")
    print(f"Wrote verification report: {args.output}", flush=True)
    if not complete:
        print("ERROR: one or more staging prefixes do not match", file=sys.stderr)
        return 2
    print("All JPEG XL and embedding staging prefixes match local counts and bytes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
