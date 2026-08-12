#!/usr/bin/env python3
"""Reconcile JUMP-Lite HQ/D20 site sets against the canonical MQ site set.

The command is a dry run unless ``--apply`` is supplied. Applying the repair
moves stale image arrays and their corresponding per-site Parquet files into a
reversible quarantine; it never deletes data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_IMAGE_ROOT = Path(
    "/work/datasets/jump_lite/images/compressed/compressed_test/jump_lite_updated"
)
DEFAULT_PROFILE_ROOT = Path(
    "/work/datasets/jump_lite/aliby_output/jump_lite_rerun/jump_lite_updated"
)
DEFAULT_QUARANTINE_ROOT = Path("/work/datasets/jump_lite/quarantine")
CANONICAL_CODEC = "jpegxl_lossy_mq.zarr"
TARGET_CODECS = ("jpegxl_lossy_hq.zarr", "jpegxl_lossy_d20.zarr")


def child_directory_names(path: Path) -> set[str]:
    return {
        entry.name
        for entry in os.scandir(path)
        if entry.is_dir(follow_symlinks=False)
    }


def parquet_stems(path: Path) -> set[str]:
    return {
        entry.name.removesuffix(".parquet")
        for entry in os.scandir(path)
        if entry.is_file(follow_symlinks=False)
        and entry.name.endswith(".parquet")
    }


def digest(keys: set[str]) -> str:
    payload = "\n".join(sorted(keys)).encode()
    return hashlib.sha256(payload).hexdigest()


def profile_directories(profile_root: Path, codec: str) -> list[tuple[str, Path]]:
    result = []
    for model in sorted(profile_root.iterdir()):
        profiles = model / codec / "profiles"
        if model.is_dir() and profiles.is_dir():
            result.append((model.name, profiles))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument(
        "--quarantine-root", type=Path, default=DEFAULT_QUARANTINE_ROOT
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="move surplus entries to quarantine (default: audit only)",
    )
    parser.add_argument(
        "--print-surplus-image-paths",
        action="store_true",
        help="print surplus HQ/D20 image directories, one per line, then exit",
    )
    return parser.parse_args()


def fail(message: str) -> None:
    raise RuntimeError(message)


def main() -> int:
    args = parse_args()
    image_root: Path = args.image_root
    profile_root: Path = args.profile_root

    canonical_path = image_root / CANONICAL_CODEC
    canonical_keys = child_directory_names(canonical_path)

    target_keys: dict[str, set[str]] = {}
    extra_keys: dict[str, set[str]] = {}
    for codec in TARGET_CODECS:
        keys = child_directory_names(image_root / codec)
        missing = canonical_keys - keys
        extra = keys - canonical_keys
        if missing:
            fail(f"{codec} is missing {len(missing):,} canonical sites")
        target_keys[codec] = keys
        extra_keys[codec] = extra

    if extra_keys[TARGET_CODECS[0]] != extra_keys[TARGET_CODECS[1]]:
        fail("HQ and D20 do not have the same surplus site keys")

    surplus = extra_keys[TARGET_CODECS[0]]
    if args.print_surplus_image_paths:
        for codec in TARGET_CODECS:
            for key in sorted(surplus):
                print(image_root / codec / key)
        return 0

    print(f"Canonical {CANONICAL_CODEC}: {len(canonical_keys):,} sites")
    for codec in TARGET_CODECS:
        print(
            f"Target {codec}: {len(target_keys[codec]):,} sites; "
            f"{len(extra_keys[codec]):,} surplus; 0 canonical sites missing"
        )

    if not surplus:
        print("Site sets are already reconciled; nothing to do.")
        return 0

    by_source = Counter(key.split("__", 1)[0] for key in surplus)
    print(f"Shared surplus: {len(surplus):,} sites; by source: {dict(sorted(by_source.items()))}")

    profile_sets: dict[tuple[str, str], set[str]] = {}
    for codec in TARGET_CODECS:
        for model, profiles in profile_directories(profile_root, codec):
            keys = parquet_stems(profiles)
            expected = target_keys[codec]
            missing = expected - keys
            extra = keys - expected
            if missing or extra:
                fail(
                    f"{model}/{codec}/profiles differs from its image set: "
                    f"missing={len(missing):,}, extra={len(extra):,}"
                )
            profile_sets[(model, codec)] = keys
            print(
                f"Per-site Parquets {model}/{codec}: {len(keys):,}; "
                f"{len(keys & surplus):,} correspond to surplus sites"
            )

    expected_profile_variants = len(profile_sets)
    expected_moves = len(surplus) * (len(TARGET_CODECS) + expected_profile_variants)
    print(
        f"Planned moves: {len(surplus) * len(TARGET_CODECS):,} image arrays + "
        f"{len(surplus) * expected_profile_variants:,} Parquets = "
        f"{expected_moves:,} entries"
    )

    if not args.apply:
        print("DRY RUN: no files changed. Re-run with --apply after granting write access.")
        return 0

    writable_paths = [image_root / codec for codec in TARGET_CODECS]
    writable_paths.extend(
        image_root / codec / key
        for codec in TARGET_CODECS
        for key in surplus
    )
    writable_paths.extend(
        profiles
        for codec in TARGET_CODECS
        for _, profiles in profile_directories(profile_root, codec)
    )
    not_writable = [str(path) for path in writable_paths if not os.access(path, os.W_OK)]
    if not_writable:
        examples = "\n  ".join(not_writable[:10])
        remainder = len(not_writable) - min(10, len(not_writable))
        suffix = f"\n  ... and {remainder:,} more" if remainder else ""
        fail(
            f"Cannot apply; {len(not_writable):,} source directories are not writable:\n  "
            f"{examples}{suffix}\nUse --print-surplus-image-paths to list the required image ACL paths."
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = args.quarantine_root / f"extra_sites_{timestamp}"
    if quarantine.exists():
        fail(f"Quarantine destination already exists: {quarantine}")
    quarantine.mkdir(parents=True)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": "HQ and D20 contained stale sites absent from the canonical MQ site set",
        "canonical_codec": CANONICAL_CODEC,
        "canonical_site_count": len(canonical_keys),
        "canonical_site_keys_sha256": digest(canonical_keys),
        "target_codecs": list(TARGET_CODECS),
        "surplus_site_count": len(surplus),
        "surplus_site_keys_sha256": digest(surplus),
        "surplus_site_keys": sorted(surplus),
        "profile_variants": [
            {"model": model, "codec": codec}
            for model, codec in sorted(profile_sets)
        ],
        "planned_move_count": expected_moves,
        "status": "moving",
    }
    manifest_path = quarantine / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    moves = 0
    for codec in TARGET_CODECS:
        for key in sorted(surplus):
            source = image_root / codec / key
            destination = quarantine / "images" / codec / key
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Both paths are on the same dataset filesystem. Use rename rather
            # than shutil.move so a permission failure cannot leave a partial copy.
            source.rename(destination)
            moves += 1

    for model, codec in sorted(profile_sets):
        profiles = profile_root / model / codec / "profiles"
        for key in sorted(surplus):
            source = profiles / f"{key}.parquet"
            destination = (
                quarantine
                / "workspace_dl"
                / "embeddings"
                / model
                / codec
                / "profiles"
                / source.name
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.rename(destination)
            moves += 1

    if moves != expected_moves:
        fail(f"Moved {moves:,} entries, expected {expected_moves:,}")

    for codec in TARGET_CODECS:
        reconciled = child_directory_names(image_root / codec)
        if reconciled != canonical_keys:
            fail(f"Post-move image verification failed for {codec}")
        for model, profiles in profile_directories(profile_root, codec):
            if parquet_stems(profiles) != canonical_keys:
                fail(f"Post-move Parquet verification failed for {model}/{codec}")

    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["moved_entry_count"] = moves
    manifest["quarantine_path"] = str(quarantine)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Reconciliation complete: moved {moves:,} entries to {quarantine}")
    print(f"All three image sets now contain {len(canonical_keys):,} identical sites.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
