#!/usr/bin/env python3
"""Generate the pinned full-JUMP red/gray plate classification ledger offline."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess

UPSTREAM_COMMIT = "016e865fa0691244e0860943e41c7d6a88ed2580"
PLATE_CSV_SHA256 = "541ada1f64816166509a4e2328316d2a6662ba67e257b7ae134cbec9d7079319"
FORMAT_VERSION = "full-jump-qc-plate-classification-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_red(row: dict[str, str]) -> bool:
    source = row["Metadata_Source"]
    batch = row["Metadata_Batch"]
    plate = row["Metadata_Plate"]
    return (
        (source == "source_4" and batch.endswith("Batch12"))
        or (
            source == "source_3"
            and (
                re.fullmatch(r"CP_3[23456]_all_Phenix1", batch) is not None
                or batch in {"CP59", "CP60"}
            )
        )
        or (source == "source_15" and plate in {"PEP00004458", "PEP00004421"})
    )


def is_gray(row: dict[str, str]) -> bool:
    return re.fullmatch(r"CP-CC9-R[123456]-28", row["Metadata_Plate"]) is not None


def build(checkout: Path) -> dict[str, object]:
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream checkout commit drift: {commit}")
    plate_csv = checkout / "metadata/plate.csv.gz"
    if sha256_file(plate_csv) != PLATE_CSV_SHA256:
        raise RuntimeError("upstream metadata/plate.csv.gz SHA-256 drift")
    with gzip.open(plate_csv, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    identities = [
        {
            "source": row["Metadata_Source"],
            "batch": row["Metadata_Batch"],
            "plate": row["Metadata_Plate"],
        }
        for row in rows
    ]
    red = sorted(
        (identity for identity, row in zip(identities, rows) if is_red(row)),
        key=lambda item: (item["source"], item["batch"], item["plate"]),
    )
    gray = sorted(
        (identity for identity, row in zip(identities, rows) if is_gray(row)),
        key=lambda item: (item["source"], item["batch"], item["plate"]),
    )
    if len(red) != 169 or len(gray) != 6:
        raise RuntimeError(
            f"classification count drift: red={len(red)} gray={len(gray)}"
        )
    return {
        "classification_only": True,
        "format_version": FORMAT_VERSION,
        "gray_plate_count": len(gray),
        "gray_plates": gray,
        "red_plate_count": len(red),
        "red_plates": red,
        "rules": {
            "gray": "Metadata_Plate matches ^CP-CC9-R[123456]-28$",
            "red": (
                "(source_4 and Metadata_Batch suffix Batch12) or "
                "(source_3 and Metadata_Batch matches ^CP_3[23456]_all_Phenix1$ "
                "or is CP59/CP60) or (source_15 and Metadata_Plate is "
                "PEP00004458/PEP00004421)"
            ),
            "source": "prep/build_jl_index.sql",
        },
        "upstream": {
            "commit": UPSTREAM_COMMIT,
            "plate_csv_path": "metadata/plate.csv.gz",
            "plate_csv_sha256": PLATE_CSV_SHA256,
            "repository": "jump-cellpainting/datasets",
            "row_count": len(rows),
        },
    }


def serialized(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = serialized(build(args.upstream_checkout.resolve()))
    if args.check:
        if not args.output.is_file() or args.output.read_text() != expected:
            raise RuntimeError(f"generated ledger drift: {args.output}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
