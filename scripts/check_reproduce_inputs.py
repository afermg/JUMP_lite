#!/usr/bin/env python3
"""Fail-closed preflight for the post-sweep paper reproduction DAG."""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

EXPECTED_METRIC_COUNTS = {
    Path("data/intermediate/sweep_v11_lite"): 7_285,
    Path("data/features/variance_first_v11"): 2_860,
    Path("data/intermediate/motive_eval/large_strict"): 1_055,
}
EXPECTED_FILES = {
    Path("data/intermediate/image_quality/quality_metrics.csv"): (
        224_818,
        "1bac02ffc9190dc8dab4ef2ec6a01dc2b02240b03211bfc7560bc1ab583483eb",
    ),
    Path(
        "data/intermediate/segmentation_comparison/detailed_results/"
        "segment_cell_jpegxl_lossy_effort_3.csv"
    ): (4_956_736, "dd621918dce6b9e3d4d2970c3c1775b2f80f3ed03692e263fb7ec056eb2dd72c"),
    Path(
        "data/intermediate/segmentation_comparison/detailed_results/"
        "segment_cell_jpegxl_lossy_hq.csv"
    ): (4_777_896, "4c6e5fc9f662926ef0c551a9de08aa3c9c3f617b40a035d058cee878dce4b584"),
    Path(
        "data/intermediate/segmentation_comparison/detailed_results/"
        "segment_nuclei_jpegxl_lossy_effort_3.csv"
    ): (4_687_505, "6624823230ce7e5a1f53c147fa5f90a7f520b9679d4a2f49616c9748fda7a622"),
    Path(
        "data/intermediate/segmentation_comparison/detailed_results/"
        "segment_nuclei_jpegxl_lossy_hq.csv"
    ): (4_597_989, "4010099b2fbec3dc33119b245bd1064c726675be23b21b009232fde11bfb9498"),
}
REQUIRED_NONEMPTY = (
    Path("metadata/metadata_dataset_filtered_4reps.parquet"),
    Path("data/intermediate/saturation_proper/saturation_results.csv"),
)
REQUIRED_GLOBS = {
    Path("data/intermediate/segmentation_comparison/instance_mappings"): "*.parquet",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_named_files(root: Path, name: str) -> int:
    total = 0
    for _directory, _subdirectories, files in os.walk(root, followlinks=True):
        total += files.count(name)
    return total


def validate(repo: Path) -> list[str]:
    errors: list[str] = []
    for relative, expected in EXPECTED_METRIC_COUNTS.items():
        path = repo / relative
        if not path.exists():
            errors.append(f"missing checkpoint root: {relative}")
            continue
        observed = count_named_files(path, "metrics.json")
        if observed != expected:
            errors.append(
                f"checkpoint count mismatch for {relative}: {observed} != {expected}"
            )
    for relative, (expected_size, expected_hash) in EXPECTED_FILES.items():
        path = repo / relative
        if not path.is_file():
            errors.append(f"missing checkpoint file: {relative}")
            continue
        observed_size = path.stat().st_size
        observed_hash = sha256(path)
        if (observed_size, observed_hash) != (expected_size, expected_hash):
            errors.append(
                f"checkpoint identity mismatch for {relative}: "
                f"{observed_size}/{observed_hash}"
            )
    for relative in REQUIRED_NONEMPTY:
        path = repo / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty required file: {relative}")
    for relative, pattern in REQUIRED_GLOBS.items():
        path = repo / relative
        if not path.is_dir() or not any(
            item.is_file() and item.stat().st_size > 0 for item in path.glob(pattern)
        ):
            errors.append(f"no non-empty {pattern} inputs under {relative}")
    return errors


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    errors = validate(repo)
    if errors:
        for error in errors:
            print(f"MISSING/INVALID  {error}", file=sys.stderr)
        print("See REPRODUCE.md for canonical checkpoint staging.", file=sys.stderr)
        return 1
    print("All post-sweep checkpoint identities and completeness gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
