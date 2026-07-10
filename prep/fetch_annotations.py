"""Fetch the upstream annotation bundle into data/annotations/.

Pulls five files from two public sources (verified byte-identical to the
versions used by this project):

  - 4 parquets from Zenodo record 18197517 (MOTIVE drug-target annotations,
    Arevalo / Su / et al. — DOI 10.5281/zenodo.18197517)
  - jump_metadata.duckdb from cellpainting-gallery (cpg0042-chandrasekaran-jump v0.13)

Each download is md5-verified against the known hash. Existing files with the
correct hash are skipped.

Usage:
    python prep/fetch_annotations.py --output-dir data/annotations
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from urllib.request import urlretrieve

# (filename, url, md5) — md5s captured 2026-06-26 from the byte-identical local copies.
ZENODO_RECORD = "18197517"
SOURCES: list[tuple[str, str, str]] = [
    (
        "annotations_compound_compound.parquet",
        f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/annotations_compound_compound.parquet/content",
        "8fe7d7cd3555630e11455457c186c913",
    ),
    (
        "annotations_compound_gene.parquet",
        f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/annotations_compound_gene.parquet/content",
        "3cf2c1bfdd2320fe240ed9c850dd3f36",
    ),
    (
        "annotations_compound_gene_curated.parquet",
        f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/annotations_compound_gene_curated.parquet/content",
        "db250d52d0a2d63ed7a9a3e85e09c441",
    ),
    (
        "annotations_gene_gene.parquet",
        f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/annotations_gene_gene.parquet/content",
        "fcd6018dd42a0451ab3983e07ef87fac",
    ),
    (
        "jump_metadata.duckdb",
        "https://cellpainting-gallery.s3.amazonaws.com/cpg0042-chandrasekaran-jump/source_all/workspace/publication_data/datasets/v0.13/jump_metadata.duckdb",
        "e56186987126683d5bafa9cc18b3657d",
    ),
]


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, url: str, expected_md5: str, out_dir: Path) -> None:
    dst = out_dir / name
    if dst.exists():
        actual = md5_of(dst)
        if actual == expected_md5:
            print(f"  ✓ {name} already present and verified")
            return
        else:
            print(f"  ! {name} present but md5 mismatch ({actual}); re-downloading")
            dst.unlink()
    print(f"  ↓ {name}  <— {url}")
    urlretrieve(url, dst)
    actual = md5_of(dst)
    if actual != expected_md5:
        raise RuntimeError(
            f"md5 mismatch for {name}: expected {expected_md5}, got {actual}"
        )
    print(f"  ✓ {name} downloaded and verified ({dst.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("data/annotations"),
                        help="Destination directory (default: data/annotations)")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target: {args.output_dir.resolve()}")
    for name, url, md5 in SOURCES:
        try:
            fetch(name, url, md5, args.output_dir)
        except Exception as e:
            print(f"  ✗ {name} FAILED: {e}", file=sys.stderr)
            sys.exit(1)
    print("All annotation files present and verified.")


if __name__ == "__main__":
    main()
