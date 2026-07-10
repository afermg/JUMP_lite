"""Build ref_chem_overlap.parquet — RefChemDB filtered to compounds present in JUMP.

Ports the RefChemDB-overlap step from JUMP_ADDON's 3.prep-aux-data.py:
joins raw RefChemDB (Judson et al. 2019 ALTEX, PMID 30570668) to JUMP
compound metadata via InChI, keeping rows whose ``Metadata_InChI`` matches
RefChemDB's ``InChI_standardized``.

Reads:
  - refchemdb_inchikey.parquet — raw RefChemDB with standardized InChI
  - jump_metadata.duckdb       — broad_babel JUMP compound table

Writes:
  - ref_chem_overlap.parquet — RefChemDB rows for compounds present in JUMP

Usage:
    python prep/build_refchemdb_overlap.py \\
        --raw data/refchemdb/refchemdb_inchikey.parquet \\
        --jump-duckdb /path/to/jump_metadata.duckdb \\
        --output data/refchemdb/ref_chem_overlap.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import polars as pl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, required=True,
                        help="refchemdb_inchikey.parquet (raw RefChemDB)")
    parser.add_argument("--jump-duckdb", type=Path, required=True,
                        help="jump_metadata.duckdb (broad_babel JUMP compound table)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination parquet (ref_chem_overlap.parquet)")
    args = parser.parse_args()

    print(f"Loading raw RefChemDB: {args.raw}")
    raw = pl.read_parquet(args.raw).unique()
    print(f"  rows={raw.height:,} (after dedup)  unique InChI_standardized={raw['InChI_standardized'].n_unique():,}")

    print(f"Loading JUMP compounds from: {args.jump_duckdb}")
    con = duckdb.connect(str(args.jump_duckdb), read_only=True)
    compounds = pl.from_arrow(
        con.sql("SELECT Metadata_JCP2022, Metadata_InChI FROM compound").to_arrow_table()
    )
    con.close()
    print(f"  rows={compounds.height:,}")

    matched_compounds = compounds.filter(
        pl.col("Metadata_InChI").is_in(raw["InChI_standardized"].unique())
    )
    print(f"JUMP compounds with InChI in RefChemDB: {matched_compounds.height:,}")

    overlap = matched_compounds.join(
        raw, left_on="Metadata_InChI", right_on="InChI_standardized", how="left"
    ).select([
        "Metadata_JCP2022", "Metadata_InChI", "DTXSID", "casrn", "name",
        "target", "target_type", "mode", "activity_class", "support", "in_cmap",
    ])
    print(f"Overlap rows: {overlap.height:,}  cols={overlap.width}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    overlap.write_parquet(args.output)
    print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
