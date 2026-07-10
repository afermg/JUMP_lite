"""Build refchemdb_conf_jump_matched.parquet from the RefChemDB JUMP-overlap subset.

Ported from archive/analysis/04_refchemdb_match.py (which itself derives from the
RefChemDB confidence-tier logic, citing Judson et al. 2019 ALTEX, PMID 30570668).

Reads:
  - ref_chem_overlap.parquet     — RefChemDB filtered to compounds present in JUMP
  - jump_metadata.duckdb         — broad_babel-style JUMP perturbation tables
                                   (compound / crispr / orf with standard_key + modality)

Writes:
  - refchemdb_conf_jump_matched.parquet — tier-annotated, joined to JUMP CRISPR/ORF JCPs

Usage:
    python prep/build_refchemdb_matched.py \\
        --overlap data/refchemdb/ref_chem_overlap.parquet \\
        --jump-duckdb /path/to/jump_metadata.duckdb \\
        --output data/refchemdb/refchemdb_conf_jump_matched.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import polars as pl


def load_jump_metadata(duckdb_path: Path) -> pl.DataFrame:
    """Build a (Metadata_JCP2022, standard_key, modality) frame from jump_metadata.duckdb.

    Matches the schema the legacy 04_refchemdb_match.py expected from
    perturbation_metadata.parquet: compound/crispr/orf perturbations unioned, with
    standard_key = the gene symbol (compounds have null) and modality in
    {compound, crispr, orf}.
    """
    con = duckdb.connect(str(duckdb_path), read_only=True)
    sql = """
        SELECT Metadata_JCP2022, NULL AS standard_key, 'compound' AS modality, 'trt' AS pert_type
        FROM compound
        UNION ALL
        SELECT Metadata_JCP2022, Metadata_Symbol AS standard_key, 'crispr' AS modality, 'trt' AS pert_type
        FROM crispr
        UNION ALL
        SELECT Metadata_JCP2022, Metadata_Symbol AS standard_key, 'orf' AS modality, Metadata_pert_type AS pert_type
        FROM orf
    """
    df = pl.from_arrow(con.sql(sql).to_arrow_table())
    con.close()
    return df


def tier_annotate(overlap: pl.DataFrame) -> pl.DataFrame:
    """Apply the RefChemDB confidence-tier logic from 04_refchemdb_match.py."""
    df = overlap.filter(pl.col("target_type") == "gene")
    df = df.filter(pl.col("support") > 1)

    df = df.join(
        df.filter(pl.col("support") >= 5)
          .group_by("Metadata_JCP2022").agg(pl.len())
          .rename({"len": "Num_Cmpd2Target_Interactions_geq5"}),
        on="Metadata_JCP2022", how="left",
    ).join(
        df.group_by("Metadata_JCP2022").agg(pl.len())
          .rename({"len": "Num_Cmpd2Target_Interactions_gt1"}),
        on="Metadata_JCP2022", how="left",
    )

    df = df.join(
        df.filter(pl.col("support") >= 5)
          .group_by("target").agg(pl.len())
          .rename({"len": "Num_TargetByCmpd_Interactions_geq5"}),
        on="target", how="left",
    ).join(
        df.group_by("target").agg(pl.len())
          .rename({"len": "Num_TargetByCmpd_Interactions_gt1"}),
        on="target", how="left",
    )

    df = df.with_columns(
        pl.when(
            (pl.col("Num_Cmpd2Target_Interactions_gt1") == 1)
            & (pl.col("Num_Cmpd2Target_Interactions_geq5") == 1)
            & (pl.col("Num_TargetByCmpd_Interactions_gt1") == 1)
            & (pl.col("Num_TargetByCmpd_Interactions_geq5") == 1)
            & (pl.col("mode") != "unspecified")
        ).then(pl.lit("Tier0"))
        .when(
            (pl.col("Num_Cmpd2Target_Interactions_geq5") == 1)
            & (pl.col("Num_Cmpd2Target_Interactions_gt1") == 1)
            & (pl.col("mode") != "unspecified")
        ).then(pl.lit("Tier1"))
        .when(
            (pl.col("Num_Cmpd2Target_Interactions_geq5") < 3)
            & (pl.col("mode") != "unspecified")
        ).then(pl.lit("Tier2"))
        .otherwise(pl.lit("Tier3"))
        .alias("CrossModalityTier")
    )

    df = df.with_columns([
        (
            (pl.col("mode") != "unspecified")
            & (pl.col("mode").n_unique().over("target") == 1)
            & (pl.col("Num_Cmpd2Target_Interactions_geq5") == 1)
            & (pl.col("Num_Cmpd2Target_Interactions_gt1") == 1)
        ).alias("tier0_eligible"),
        (
            (pl.col("mode") != "unspecified")
            & (pl.col("mode").n_unique().over("target") == 1)
            & (pl.col("Num_Cmpd2Target_Interactions_geq5") == 1)
        ).alias("tier1_eligible"),
        (
            (pl.col("mode") != "unspecified")
            & (pl.col("mode").n_unique().over("target") == 1)
        ).alias("tier2_eligible"),
        (pl.col("target").is_duplicated()).alias("tier3_eligible"),
    ])

    df = df.with_columns([
        pl.col("tier0_eligible").sum().over("target").alias("n_tier0"),
        pl.col("tier1_eligible").sum().over("target").alias("n_tier1"),
        pl.col("tier2_eligible").sum().over("target").alias("n_tier2"),
        pl.col("tier3_eligible").sum().over("target").alias("n_tier3"),
    ])

    df = df.with_columns(
        pl.when((pl.col("tier0_eligible")) & (pl.col("n_tier0") >= 2)).then(pl.lit("Tier0"))
        .when((pl.col("tier1_eligible")) & (pl.col("n_tier1") >= 2)).then(pl.lit("Tier1"))
        .when((pl.col("tier2_eligible")) & (pl.col("n_tier2") >= 2)).then(pl.lit("Tier2"))
        .when((pl.col("tier3_eligible")) & (pl.col("n_tier3") >= 2)).then(pl.lit("Tier3"))
        .otherwise(pl.lit("Excluded"))
        .alias("WithinModalityTier")
    )

    return df.drop([
        "tier0_eligible", "tier1_eligible", "tier2_eligible", "tier3_eligible",
        "n_tier0", "n_tier1", "n_tier2", "n_tier3",
    ])


def match_to_jump(tiered: pl.DataFrame, jump_metadata: pl.DataFrame) -> pl.DataFrame:
    """Join tier-annotated RefChemDB to JUMP CRISPR/ORF perturbations on gene symbol."""
    non_compound = (
        jump_metadata.filter(pl.col("modality") != "compound")
        .rename({"Metadata_JCP2022": "Metadata_JCP2022_target"})
        .with_columns(
            pl.when(pl.col("modality") == "orf").then(pl.lit("Positive"))
              .otherwise(pl.lit("Negative")).alias("modality_clean")
        )
    )

    by_dir = (
        tiered.filter(pl.col("mode") != "unspecified")
        .join(non_compound, left_on=["target", "mode"],
              right_on=["standard_key", "modality_clean"])
        .with_columns(
            pl.col("mode").alias("modality_clean"),
            pl.lit(True).alias("cmpd_pert_dir_matched"),
        )
    )

    no_dir = (
        tiered.filter(pl.col("mode") == "unspecified")
        .join(non_compound, left_on="target", right_on="standard_key")
        .with_columns(pl.lit(False).alias("cmpd_pert_dir_matched"))
    )

    return pl.concat([by_dir, no_dir], how="vertical")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--overlap", type=Path, required=True,
                        help="ref_chem_overlap.parquet (RefChemDB ∩ JUMP compounds)")
    parser.add_argument("--jump-duckdb", type=Path, required=True,
                        help="jump_metadata.duckdb (broad_babel JUMP tables)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Destination parquet (refchemdb_conf_jump_matched.parquet)")
    args = parser.parse_args()

    print(f"Loading overlap: {args.overlap}")
    overlap = pl.read_parquet(args.overlap)
    print(f"  rows={overlap.height:,}  cols={overlap.width}")

    print(f"Loading JUMP perturbation metadata from: {args.jump_duckdb}")
    jump_metadata = load_jump_metadata(args.jump_duckdb)
    print(f"  rows={jump_metadata.height:,}  modalities={jump_metadata['modality'].value_counts()}")

    print("Applying confidence-tier logic...")
    tiered = tier_annotate(overlap)
    print(f"  tiered rows={tiered.height:,}")

    print("Matching to JUMP CRISPR/ORF perturbations...")
    matched = match_to_jump(tiered, jump_metadata)
    print(f"  matched rows={matched.height:,}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    matched.write_parquet(args.output)
    print(f"Wrote {args.output} ({args.output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
