#!/usr/bin/env python3
"""Run normalization pipeline on all extracted feature files."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run normalization on all features")
    parser.add_argument("--input", type=Path, default=Path("data/raw_features"))
    parser.add_argument("--output", type=Path, default=Path("data/normalized"))
    parser.add_argument("--preset", default="standardized")
    parser.add_argument("--filter", type=str, default=None, help="Filter files by prefix (e.g., dinov2, cp_measure)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = sorted(args.input.glob("*_raw_features.parquet"))
    if args.filter:
        files = [f for f in files if f.name.startswith(args.filter)]
    if not files:
        print(f"No feature files found in {args.input}")
        return 1

    print(f"Found {len(files)} files")
    args.output.mkdir(parents=True, exist_ok=True)

    for f in files:
        print(f"\nProcessing: {f.name}")
        cmd = [
            sys.executable, "src/norm/run_pipeline.py",
            f"+preset={args.preset}",
            f"input_override={f}",
            f"output.path={args.output / f.stem}/processed.parquet",
        ]
        if args.dry_run:
            print(f"  {' '.join(cmd)}")
        else:
            subprocess.run(cmd, cwd=Path(__file__).parent.parent)


if __name__ == "__main__":
    sys.exit(main() or 0)
