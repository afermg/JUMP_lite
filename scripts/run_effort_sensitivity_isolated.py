#!/usr/bin/env python3
"""Run the frozen effort-sensitivity producer against an isolated output root.

The scientific runner binds its own source bytes in provenance. Keeping that
runner unchanged preserves the accepted identity; this narrow wrapper only
redirects its module-level output location before calling its public entrypoint.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO / "rebuttal/mq_d2e8_explanation/effort_sensitivity/analyze.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("effort_sensitivity_frozen", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    module = load_runner()
    module.OUTPUT_DIR = output_dir
    module.main(["--verify-only"] if args.verify_only else [])


if __name__ == "__main__":
    main()
