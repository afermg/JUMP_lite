"""Download raw JUMP TIFFs from cellpainting-gallery S3.

Reads the URI manifest produced by prep/build_jl_index.sql and downloads each
TIFF into ``out_dir``, named as ``{source}__{batch}__{plate}__{well}__{site}__{channel}.tif``.

Anonymous S3 access — no AWS credentials required.

Usage:
    python prep/download_raw.py \\
        --manifest data/manifest/jl_index_tidy.parquet \\
        --out-dir $DATA_ROOT/jump_lite/imgs/raw \\
        --n-jobs 16
"""

from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

import boto3
import duckdb
from botocore import UNSIGNED
from botocore.config import Config
from joblib import Parallel, delayed


def load_uris(manifest: Path) -> list[tuple]:
    with duckdb.connect() as con:
        rows = con.sql(f"FROM read_parquet('{manifest}')").to_arrow_table().to_pylist()
    return [
        (
            *list(x.values())[:-3],
            str(x["Metadata_Site"]),
            x["Metadata_Channel"].removeprefix("URL_Orig"),
            x["uri"].removeprefix("s3://cellpainting-gallery/"),
        )
        for x in rows
    ]


def download_uri(meta: tuple, out_dir: Path) -> None:
    *location, key = meta
    local_file = out_dir / ("__".join(location) + ".tif")
    if local_file.exists():
        return
    local_file.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    s3.download_file("cellpainting-gallery", key, str(local_file))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="URI manifest parquet (jl_index_tidy.parquet) from prep/build_jl_index.sql")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Destination directory for downloaded TIFFs")
    parser.add_argument("--n-jobs", type=int, default=16,
                        help="Parallel download workers (default: 16)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading URI list from {args.manifest}")
    uris = load_uris(args.manifest)
    print(f"{len(uris)} TIFFs to fetch into {args.out_dir}")

    fn = partial(download_uri, out_dir=args.out_dir)
    Parallel(n_jobs=args.n_jobs, verbose=10)(delayed(fn)(x) for x in uris)


if __name__ == "__main__":
    main()
