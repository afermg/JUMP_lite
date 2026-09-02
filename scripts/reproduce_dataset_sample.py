#!/usr/bin/env python3
"""Reproduce a source-stratified JUMP-Lite v1.0 image sample.

The smoke test validates the complete 655,101-site release index, selects the
lexical middle site from each of the six release sources, reads all five source
TIFF channels, and creates Zstd, HQ, MQ, and D20 Zarr arrays in a new isolated
output root. Optional canonical stores enable exact per-file comparisons.

This is a structural generation test, not a statistically representative
biological sample and not a substitute for whole-release validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import duckdb
import numpy as np
import zarr
from botocore import UNSIGNED
from botocore.config import Config

sys.dont_write_bytecode = True

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.compress_tif_release import (  # noqa: E402
    CHANNEL_ORDER,
    available_compressors,
    compress_tif,
    read_stack,
)

EXPECTED_SITE_COUNT = 655_101
EXPECTED_SITE_INDEX_BYTES = 6_914_554
EXPECTED_SITE_INDEX_SHA256 = (
    "ec0e8a87ee66d640025617c360165d20d16a35bde655e7f78854a69651e87eec"
)
EXPECTED_SITE_DIGEST = (
    "4ea6ea3f5457c33a1412a80a89d8696d4f8e77474cf449e75db7ce6ba98685e2"
)
EXPECTED_ATTESTATION = REPO / "reproducibility/validation/dataset-smoke-20260901.json"
EXPECTED_SOURCE_COUNTS = {
    "source_2": 14_075,
    "source_4": 326_650,
    "source_6": 14_160,
    "source_7": 96_308,
    "source_8": 8_384,
    "source_13": 195_524,
}
CODECS = ("zstd", "jpegxl_lossy_hq", "jpegxl_lossy_mq", "jpegxl_lossy_d20")
URL_COLUMNS = {channel: f"URL_Orig{channel}" for channel in CHANNEL_ORDER}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_output_root(value: Path) -> Path:
    lexical = value if value.is_absolute() else REPO / value
    if ".." in value.parts:
        raise ValueError("output root may not traverse '..'")
    allowed = REPO / "data/generated/dataset-smoke"
    resolved = lexical.resolve(strict=False)
    try:
        resolved.relative_to(allowed.resolve())
    except ValueError as error:
        raise ValueError(f"output root must remain under {allowed}") from error

    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError(f"symlinked output parent is forbidden: {current}")
    if lexical.exists() or lexical.is_symlink():
        raise ValueError(f"output root must not exist: {lexical}")
    return lexical


def load_expected_attestation() -> dict[str, Any]:
    if not EXPECTED_ATTESTATION.is_file():
        raise ValueError(f"missing dataset-smoke attestation: {EXPECTED_ATTESTATION}")
    attestation = json.loads(EXPECTED_ATTESTATION.read_text())
    if (
        attestation.get("format_version")
        != "jump-lite-source-stratified-validation-attestation-v1"
    ):
        raise ValueError("dataset-smoke attestation protocol drift")
    return attestation


def validate_and_select(site_index: Path) -> tuple[dict[str, dict[str, Any]], str]:
    if not site_index.is_file():
        raise ValueError(f"site index is not a file: {site_index}")
    if (
        site_index.stat().st_size != EXPECTED_SITE_INDEX_BYTES
        or sha256_file(site_index) != EXPECTED_SITE_INDEX_SHA256
    ):
        raise ValueError("release site-index file identity mismatch")
    quoted = str(site_index).replace("'", "''")
    connection = duckdb.connect()
    columns = {
        row[0]
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{quoted}')"
        ).fetchall()
    }
    required = {
        "Metadata_Site_Key",
        "Metadata_Source",
        "Metadata_Batch",
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Site",
        *URL_COLUMNS.values(),
    }
    if not required <= columns:
        raise ValueError(f"site index missing columns: {sorted(required - columns)}")

    count, distinct, malformed = connection.execute(
        f"""
        SELECT
          count(*),
          count(DISTINCT Metadata_Site_Key),
          count(*) FILTER (
            WHERE Metadata_Site_Key != concat_ws(
              '__', Metadata_Source, Metadata_Batch, Metadata_Plate,
              Metadata_Well, cast(Metadata_Site AS VARCHAR)
            )
          )
        FROM read_parquet('{quoted}')
        """
    ).fetchone()
    if (count, distinct, malformed) != (
        EXPECTED_SITE_COUNT,
        EXPECTED_SITE_COUNT,
        0,
    ):
        raise ValueError(
            "release site-index identity mismatch: "
            f"rows={count}, distinct={distinct}, malformed={malformed}"
        )

    source_counts = dict(
        connection.execute(
            f"""
            SELECT Metadata_Source, count(*)
            FROM read_parquet('{quoted}')
            GROUP BY Metadata_Source
            ORDER BY Metadata_Source
            """
        ).fetchall()
    )
    if source_counts != EXPECTED_SOURCE_COUNTS:
        raise ValueError(f"release source counts changed: {source_counts}")

    cursor = connection.execute(
        f"""
        SELECT Metadata_Site_Key
        FROM read_parquet('{quoted}')
        ORDER BY Metadata_Site_Key
        """
    )
    digest = hashlib.sha256()
    first = True
    while rows := cursor.fetchmany(10_000):
        for (key,) in rows:
            if not first:
                digest.update(b"\n")
            digest.update(str(key).encode())
            first = False
    site_digest = digest.hexdigest()
    if site_digest != EXPECTED_SITE_DIGEST:
        raise ValueError(f"release site-key digest changed: {site_digest}")

    selected_columns = [
        "Metadata_Site_Key",
        "Metadata_Source",
        "Metadata_Batch",
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Site",
        *URL_COLUMNS.values(),
    ]
    rows = connection.execute(
        f"""
        WITH ranked AS (
          SELECT
            *,
            row_number() OVER (
              PARTITION BY Metadata_Source ORDER BY Metadata_Site_Key
            ) AS source_rank,
            count(*) OVER (PARTITION BY Metadata_Source) AS source_count
          FROM read_parquet('{quoted}')
        )
        SELECT {", ".join(selected_columns)}
        FROM ranked
        WHERE source_rank = (source_count + 1) // 2
        ORDER BY Metadata_Source
        """
    ).fetchall()
    connection.close()
    selected = {
        str(row[1]): dict(zip(selected_columns, row, strict=True)) for row in rows
    }
    if set(selected) != set(EXPECTED_SOURCE_COUNTS):
        raise ValueError(
            f"source-stratified selection is incomplete: {sorted(selected)}"
        )
    return selected, site_digest


def public_s3_client():
    return boto3.client(
        "s3",
        region_name="us-east-1",
        config=Config(
            signature_version=UNSIGNED,
            retries={"mode": "adaptive", "max_attempts": 8},
            connect_timeout=20,
            read_timeout=180,
        ),
    )


def acquire_sources(
    selected: dict[str, dict[str, Any]],
    source_root: Path | None,
    download_root: Path,
) -> tuple[dict[tuple[str, ...], list[Path]], list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[Path]] = {}
    records: list[dict[str, Any]] = []
    client = None if source_root is not None else public_s3_client()
    if client is not None:
        download_root.mkdir(parents=True)

    for source, row in selected.items():
        site_key = str(row["Metadata_Site_Key"])
        site_parts = tuple(site_key.split("__"))
        if len(site_parts) != 5 or site_parts[0] != source:
            raise ValueError(f"invalid selected site key: {site_key}")
        files: list[Path] = []
        for channel in CHANNEL_ORDER:
            url = str(row[URL_COLUMNS[channel]])
            parsed = urlparse(url)
            if (
                parsed.scheme != "s3"
                or parsed.netloc != "cellpainting-gallery"
                or not parsed.path
            ):
                raise ValueError(f"source URL outside public allowlist: {url}")
            filename = f"{site_key}__{channel}.tif"
            path = (
                source_root / filename
                if source_root is not None
                else download_root / filename
            )
            if source_root is None:
                assert client is not None
                partial = path.with_suffix(path.suffix + ".partial")
                with partial.open("xb") as stream:
                    client.download_fileobj(
                        parsed.netloc, parsed.path.lstrip("/"), stream
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                partial.rename(path)
            if not path.is_file():
                raise ValueError(f"missing source TIFF: {path}")
            files.append(path)
            records.append(
                {
                    "site_key": site_key,
                    "source": source,
                    "channel": channel,
                    "url": url,
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        groups[site_parts] = files
    return groups, records


def tree_inventory(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"invalid Zarr site directory: {path}")
    inventory: dict[str, dict[str, Any]] = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError(f"symlink in Zarr site: {item}")
        if item.is_file():
            inventory[str(item.relative_to(path))] = {
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
    return inventory


def verify_source_attestation(
    sources: list[dict[str, Any]], attestation: dict[str, Any]
) -> None:
    observed = [
        {key: value for key, value in row.items() if key != "path"} for row in sources
    ]
    if observed != attestation.get("source_tiffs"):
        raise RuntimeError("selected source-TIFF identity drift")


def reproduce(
    groups: dict[tuple[str, ...], list[Path]],
    output_root: Path,
    jxl_reference_root: Path | None,
    zstd_reference_root: Path | None,
    attestation: dict[str, Any],
) -> list[dict[str, Any]]:
    compressors = available_compressors()
    missing = set(CODECS) - set(compressors)
    if missing:
        raise RuntimeError(f"required codecs unavailable: {sorted(missing)}")

    expected_outputs = {
        (str(row["codec"]), str(row["site_key"])): row["generated_files"]
        for row in attestation.get("outputs", [])
    }
    expected_keys = {
        (codec, "__".join(site_parts)) for codec in CODECS for site_parts in groups
    }
    if set(expected_outputs) != expected_keys:
        raise RuntimeError("attested generated-array inventory is incomplete")

    outputs = output_root / "images_compressed"
    outputs.mkdir()
    records: list[dict[str, Any]] = []
    for codec in CODECS:
        result = compress_tif(
            codec,
            compressors[codec],
            outputs,
            groups,
            n_jobs_inner=min(6, len(groups)),
        )
        if result["errors"] or result["success"] != len(groups):
            raise RuntimeError(f"{codec} sample generation failed: {result}")

        for site_parts, source_paths in groups.items():
            site_key = "__".join(site_parts)
            generated = outputs / f"{codec}.zarr" / site_key
            generated_inventory = tree_inventory(generated)
            exact_attestation_match = (
                generated_inventory == expected_outputs[(codec, site_key)]
            )
            if not exact_attestation_match:
                raise RuntimeError(
                    f"generated files differ from public attestation: "
                    f"{codec}/{site_key}"
                )
            decoded = zarr.open_array(generated, mode="r")[:]
            source_stack = read_stack(source_paths)
            if decoded.shape != source_stack.shape or decoded.dtype != np.uint16:
                raise RuntimeError(f"decoded array contract failed: {codec}/{site_key}")
            if codec == "zstd" and not np.array_equal(decoded, source_stack):
                raise RuntimeError(f"lossless round-trip failed: {site_key}")

            reference_root = (
                zstd_reference_root
                if codec == "zstd"
                else (
                    jxl_reference_root / f"{codec}.zarr"
                    if jxl_reference_root is not None
                    else None
                )
            )
            reference_inventory = None
            exact_reference_match = None
            if reference_root is not None:
                reference = reference_root / site_key
                reference_inventory = tree_inventory(reference)
                exact_reference_match = generated_inventory == reference_inventory
                if not exact_reference_match:
                    raise RuntimeError(
                        f"generated files differ from canonical reference: "
                        f"{codec}/{site_key}"
                    )

            absolute_error = np.abs(
                decoded.astype(np.int64) - source_stack.astype(np.int64)
            )
            records.append(
                {
                    "codec": codec,
                    "site_key": site_key,
                    "shape": list(decoded.shape),
                    "dtype": str(decoded.dtype),
                    "generated_files": generated_inventory,
                    "exact_attestation_match": exact_attestation_match,
                    "reference_files": reference_inventory,
                    "exact_reference_match": exact_reference_match,
                    "source_roundtrip_exact": bool(
                        np.array_equal(decoded, source_stack)
                    ),
                    "mean_absolute_error": float(absolute_error.mean()),
                    "max_absolute_error": int(absolute_error.max()),
                }
            )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-index", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/generated/dataset-smoke/source-stratified"),
    )
    parser.add_argument(
        "--source-tiff-root",
        type=Path,
        help="optional flat local TIFF cache; otherwise download 30 public TIFFs",
    )
    parser.add_argument(
        "--jxl-reference-root",
        type=Path,
        help="optional root containing canonical HQ/MQ/D20 Zarr stores",
    )
    parser.add_argument(
        "--zstd-reference-root",
        type=Path,
        help="optional canonical Zstd Zarr store",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = safe_output_root(args.output_root)
    output_root.mkdir(parents=True)
    attestation = load_expected_attestation()
    selected, site_digest = validate_and_select(args.site_index)
    groups, sources = acquire_sources(
        selected, args.source_tiff_root, output_root / "source_tiffs"
    )
    verify_source_attestation(sources, attestation)
    outputs = reproduce(
        groups,
        output_root,
        args.jxl_reference_root,
        args.zstd_reference_root,
        attestation,
    )
    compared = [row for row in outputs if row["exact_reference_match"] is not None]
    report = {
        "format_version": "jump-lite-source-stratified-reproduction-v1",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "site_index": {
            "path": str(args.site_index),
            "bytes": args.site_index.stat().st_size,
            "sha256": sha256_file(args.site_index),
            "site_count": EXPECTED_SITE_COUNT,
            "site_key_sha256": site_digest,
            "source_counts": EXPECTED_SOURCE_COUNTS,
        },
        "sampling": {
            "method": "lexical middle site within each release source",
            "scope": "structural smoke test, not a biological population sample",
            "sites": [selected[source]["Metadata_Site_Key"] for source in selected],
            "site_count": len(selected),
            "tiff_count": len(sources),
            "channel_order": list(CHANNEL_ORDER),
            "codecs": list(CODECS),
        },
        "source_tiffs": sources,
        "outputs": outputs,
        "summary": {
            "generated_site_arrays": len(outputs),
            "exact_attested_array_trees": sum(
                row["exact_attestation_match"] for row in outputs
            ),
            "lossless_roundtrips": sum(
                row["source_roundtrip_exact"]
                for row in outputs
                if row["codec"] == "zstd"
            ),
            "canonical_arrays_compared": len(compared),
            "exact_canonical_array_trees": sum(
                row["exact_reference_match"] is True for row in compared
            ),
        },
    }
    report_path = output_root / "reproduction_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
