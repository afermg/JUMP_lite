"""Fetch raw JUMP CellProfiler profiles.parquet from cellpainting-gallery.

Pulls the v1.0c assembled-profiles file consumed by ``just extract-cp-lite``
(via ``src/reformat_raw_cp_profiles.py``). The destination matches the
``cp_profiles`` path resolved in the justfile.

Source (anonymous S3, no AWS credentials needed):
  s3://cellpainting-gallery/cpg0016-jump-assembled/source_all/workspace/
      profiles_assembled/ALL/v1.0c/profiles.parquet

The upstream object is a multipart upload, so its ETag is not a plain MD5.
We verify by exact byte size (13,550,356,031 bytes). Existing files of the
expected size are skipped.

Usage:
    python prep/fetch_cp_profiles.py \\
        --output data/jump_core_annotated/raw_jump_CP_profiles/profiles.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "cellpainting-gallery"
KEY = "cpg0016-jump-assembled/source_all/workspace/profiles_assembled/ALL/v1.0c/profiles.parquet"
EXPECTED_SIZE = 13_550_356_031


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination parquet path")
    args = parser.parse_args()

    if args.output.exists() and args.output.stat().st_size == EXPECTED_SIZE:
        print(f"  ✓ {args.output} already present and size-verified")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    cfg = TransferConfig(multipart_threshold=64 * 1024 * 1024,
                         multipart_chunksize=64 * 1024 * 1024,
                         max_concurrency=16,
                         use_threads=True)
    print(f"  ↓ {args.output}  <— s3://{BUCKET}/{KEY}")
    print(f"    expected size: {EXPECTED_SIZE:,} bytes (~13.5 GB)")
    s3.download_file(BUCKET, KEY, str(args.output), Config=cfg)

    actual = args.output.stat().st_size
    if actual != EXPECTED_SIZE:
        print(f"  ✗ size mismatch: expected {EXPECTED_SIZE:,}, got {actual:,}", file=sys.stderr)
        sys.exit(1)
    print(f"  ✓ downloaded and size-verified ({actual:,} bytes)")


if __name__ == "__main__":
    main()
