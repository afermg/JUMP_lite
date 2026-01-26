"""
Compare JCP ID overlap between filtered JUMP metadata and RefChemDB annotations.

This script:
1. Loads filtered metadata from analyze_metadata.py output (with various filtering stages)
2. Loads RefChemDB matched annotations
3. Computes overlap statistics between the two datasets
4. Shows which JCP IDs are unique to each dataset
5. Breaks down overlap by perturbation type
"""

import pandas as pd
from pathlib import Path
from broad_babel.data import get_table


# =============================================================================
# Configuration
# =============================================================================

# Paths to data sources
ORIGINAL_METADATA_PATH = Path("/work/datasets/jump_core/metadata.parquet")
FILTERED_METADATA_PATH = Path("/home/jfredinh/projects/JUMP_core/metadata/metadata_filtered.parquet")
FILTERED_WITH_TARGET2_PATH = Path("/home/jfredinh/projects/JUMP_core/metadata/metadata_filtered_with_target2.parquet")
REFCHEMDB_PATH = Path("/home/jfredinh/projects/JUMP_core/analysis/annotations_tiers/refchemdb_conf_jump_matched.parquet")

WELLS_PER_384_PLATE = 384


# =============================================================================
# Data Loading
# =============================================================================

def load_refchemdb():
    """Load RefChemDB matched annotations."""
    df = pd.read_parquet(REFCHEMDB_PATH)
    return df


def load_original_metadata():
    """Load the original unfiltered metadata."""
    return pd.read_parquet(ORIGINAL_METADATA_PATH)


def load_filtered_metadata():
    """Load filtered metadata (no TARGET2, 25% min fill rate)."""
    return pd.read_parquet(FILTERED_METADATA_PATH)


def load_filtered_with_target2():
    """Load filtered metadata (with TARGET2, 25% min fill rate)."""
    if FILTERED_WITH_TARGET2_PATH.exists():
        return pd.read_parquet(FILTERED_WITH_TARGET2_PATH)
    return None


def load_perturbation_lists():
    """Load sets of JCP IDs for each perturbation type."""
    crispr = set(get_table("crispr").to_pandas()["Metadata_JCP2022"].dropna().unique())
    orf = set(get_table("orf").to_pandas()["Metadata_JCP2022"].dropna().unique())
    compound = set(get_table("compound").to_pandas()["Metadata_JCP2022"].dropna().unique())
    return {"CRISPR": crispr, "ORF": orf, "COMPOUND": compound}


def load_plate_metadata():
    """Load plate metadata from broad_babel."""
    return get_table("plate").to_pandas()


# =============================================================================
# Filtering Functions (mirror analyze_metadata.py)
# =============================================================================

def get_source_filtered_jcp_ids(df):
    """Get JCP IDs after removing source_7 and source_9."""
    sources_to_exclude = ["source_7", "source_9"]
    mask = ~df["Metadata_Source"].isin(sources_to_exclude)
    return set(df[mask]["Metadata_JCP2022"].dropna().unique())


def get_source_and_target2_filtered_jcp_ids(df, plate_meta):
    """Get JCP IDs after removing source_7, source_9, and TARGET2 plates."""
    sources_to_exclude = ["source_7", "source_9"]
    plates_target2 = set(
        plate_meta[plate_meta["Metadata_PlateType"].str.contains("TARGET2", case=False, na=False)]["Metadata_Plate"]
    )

    source_mask = ~df["Metadata_Source"].isin(sources_to_exclude)
    plate_mask = ~df["Metadata_Plate"].isin(plates_target2)

    return set(df[source_mask & plate_mask]["Metadata_JCP2022"].dropna().unique())


def get_fill_rate_filtered_jcp_ids(df, plate_meta, min_fill_rate=25):
    """Get JCP IDs after all filtering (source, TARGET2, fill rate)."""
    # First apply source and TARGET2 filter
    sources_to_exclude = ["source_7", "source_9"]
    plates_target2 = set(
        plate_meta[plate_meta["Metadata_PlateType"].str.contains("TARGET2", case=False, na=False)]["Metadata_Plate"]
    )

    source_mask = ~df["Metadata_Source"].isin(sources_to_exclude)
    plate_mask = ~df["Metadata_Plate"].isin(plates_target2)
    df_filtered = df[source_mask & plate_mask].copy()

    # Calculate plate fill rates
    plate_fills = (
        df_filtered.groupby(["Metadata_Source", "Metadata_Plate"])["Metadata_Well"]
        .nunique()
        .reset_index(name="Wells_Present")
    )
    plate_fills["Fill_Rate_Percent"] = (
        plate_fills["Wells_Present"] / WELLS_PER_384_PLATE * 100
    )

    # Filter by fill rate
    valid_plates = plate_fills[plate_fills["Fill_Rate_Percent"] >= min_fill_rate]
    valid_plate_set = set(zip(valid_plates["Metadata_Source"], valid_plates["Metadata_Plate"]))

    mask = df_filtered.apply(
        lambda row: (row["Metadata_Source"], row["Metadata_Plate"]) in valid_plate_set,
        axis=1
    )

    return set(df_filtered[mask]["Metadata_JCP2022"].dropna().unique())


# =============================================================================
# Overlap Analysis
# =============================================================================

def compute_overlap_stats(set_a, set_b, name_a, name_b):
    """Compute overlap statistics between two sets."""
    intersection = set_a & set_b
    only_a = set_a - set_b
    only_b = set_b - set_a
    union = set_a | set_b

    stats = {
        f"{name_a}_total": len(set_a),
        f"{name_b}_total": len(set_b),
        "intersection": len(intersection),
        f"only_{name_a}": len(only_a),
        f"only_{name_b}": len(only_b),
        "union": len(union),
        f"pct_{name_a}_in_{name_b}": (len(intersection) / len(set_a) * 100) if set_a else 0,
        f"pct_{name_b}_in_{name_a}": (len(intersection) / len(set_b) * 100) if set_b else 0,
        "jaccard_index": (len(intersection) / len(union) * 100) if union else 0,
    }

    return stats, intersection, only_a, only_b


def classify_jcp_ids_by_type(jcp_ids, perturbation_lists):
    """Classify a set of JCP IDs by perturbation type."""
    result = {"CRISPR": set(), "ORF": set(), "COMPOUND": set(), "UNKNOWN": set()}

    for jcp_id in jcp_ids:
        if jcp_id in perturbation_lists["CRISPR"]:
            result["CRISPR"].add(jcp_id)
        elif jcp_id in perturbation_lists["ORF"]:
            result["ORF"].add(jcp_id)
        elif jcp_id in perturbation_lists["COMPOUND"]:
            result["COMPOUND"].add(jcp_id)
        else:
            result["UNKNOWN"].add(jcp_id)

    return result


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_overlap_table(stats, name_a, name_b):
    """Print a nicely formatted overlap statistics table."""
    print(f"\n{'Metric':<40} {'Value':>15}")
    print("-" * 55)
    print(f"{name_a + ' total JCP IDs':<40} {stats[f'{name_a}_total']:>15,}")
    print(f"{name_b + ' total JCP IDs':<40} {stats[f'{name_b}_total']:>15,}")
    print("-" * 55)
    print(f"{'Intersection (in both)':<40} {stats['intersection']:>15,}")
    print(f"{'Only in ' + name_a:<40} {stats[f'only_{name_a}']:>15,}")
    print(f"{'Only in ' + name_b:<40} {stats[f'only_{name_b}']:>15,}")
    print(f"{'Union (in either)':<40} {stats['union']:>15,}")
    print("-" * 55)
    print(f"{'% of ' + name_a + ' found in ' + name_b:<40} {stats[f'pct_{name_a}_in_{name_b}']:>14.1f}%")
    print(f"{'% of ' + name_b + ' found in ' + name_a:<40} {stats[f'pct_{name_b}_in_{name_a}']:>14.1f}%")
    print(f"{'Jaccard similarity index':<40} {stats['jaccard_index']:>14.1f}%")


def print_type_breakdown(jcp_ids, perturbation_lists, title):
    """Print breakdown of JCP IDs by perturbation type."""
    by_type = classify_jcp_ids_by_type(jcp_ids, perturbation_lists)
    total = len(jcp_ids)

    print(f"\n{title}:")
    print(f"  {'COMPOUND:':<15} {len(by_type['COMPOUND']):>8,} ({len(by_type['COMPOUND'])/total*100 if total else 0:>5.1f}%)")
    print(f"  {'ORF:':<15} {len(by_type['ORF']):>8,} ({len(by_type['ORF'])/total*100 if total else 0:>5.1f}%)")
    print(f"  {'CRISPR:':<15} {len(by_type['CRISPR']):>8,} ({len(by_type['CRISPR'])/total*100 if total else 0:>5.1f}%)")
    print(f"  {'UNKNOWN:':<15} {len(by_type['UNKNOWN']):>8,} ({len(by_type['UNKNOWN'])/total*100 if total else 0:>5.1f}%)")

    return by_type


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("JCP ID OVERLAP ANALYSIS: JUMP Metadata vs RefChemDB Annotations")
    print("=" * 70)

    # Load RefChemDB data
    print("\nLoading RefChemDB annotations...")
    refchemdb_df = load_refchemdb()
    refchemdb_jcp_ids = set(refchemdb_df["Metadata_JCP2022"].dropna().unique())
    print(f"  RefChemDB file: {REFCHEMDB_PATH}")
    print(f"  Total rows: {len(refchemdb_df):,}")
    print(f"  Unique JCP IDs: {len(refchemdb_jcp_ids):,}")

    # Load perturbation lists for classification
    print("\nLoading perturbation type lists...")
    perturbation_lists = load_perturbation_lists()

    # Load plate metadata for filtering
    print("Loading plate metadata...")
    plate_meta = load_plate_metadata()

    # Load original metadata
    print("\nLoading original JUMP metadata...")
    original_df = load_original_metadata()
    original_jcp_ids = set(original_df["Metadata_JCP2022"].dropna().unique())
    print(f"  Total rows: {len(original_df):,}")
    print(f"  Unique JCP IDs: {len(original_jcp_ids):,}")

    # ==========================================================================
    # Analysis 1: RefChemDB vs Original (unfiltered) metadata
    # ==========================================================================
    print_section_header("1. RefChemDB vs ORIGINAL METADATA (no filtering)")

    stats, intersection, only_refchemdb, only_original = compute_overlap_stats(
        refchemdb_jcp_ids, original_jcp_ids, "RefChemDB", "Original"
    )
    print_overlap_table(stats, "RefChemDB", "Original")

    print_type_breakdown(intersection, perturbation_lists, "Intersection by type")
    print_type_breakdown(only_refchemdb, perturbation_lists, "Only in RefChemDB by type")

    # ==========================================================================
    # Analysis 2: RefChemDB vs Source-filtered metadata
    # ==========================================================================
    print_section_header("2. RefChemDB vs SOURCE-FILTERED (excluding source_7, source_9)")

    source_filtered_jcp = get_source_filtered_jcp_ids(original_df)
    print(f"\nSource-filtered unique JCP IDs: {len(source_filtered_jcp):,}")

    stats, intersection, only_refchemdb, only_filtered = compute_overlap_stats(
        refchemdb_jcp_ids, source_filtered_jcp, "RefChemDB", "SourceFiltered"
    )
    print_overlap_table(stats, "RefChemDB", "SourceFiltered")

    print_type_breakdown(intersection, perturbation_lists, "Intersection by type")

    # ==========================================================================
    # Analysis 3: RefChemDB vs Source+TARGET2 filtered
    # ==========================================================================
    print_section_header("3. RefChemDB vs SOURCE+TARGET2 FILTERED")

    source_target2_filtered_jcp = get_source_and_target2_filtered_jcp_ids(original_df, plate_meta)
    print(f"\nSource+TARGET2 filtered unique JCP IDs: {len(source_target2_filtered_jcp):,}")

    stats, intersection, only_refchemdb, only_filtered = compute_overlap_stats(
        refchemdb_jcp_ids, source_target2_filtered_jcp, "RefChemDB", "SourceTarget2Filt"
    )
    print_overlap_table(stats, "RefChemDB", "SourceTarget2Filt")

    print_type_breakdown(intersection, perturbation_lists, "Intersection by type")

    # ==========================================================================
    # Analysis 4: RefChemDB vs Fully filtered (25% fill rate)
    # ==========================================================================
    print_section_header("4. RefChemDB vs FULLY FILTERED (25% min fill rate)")

    if FILTERED_METADATA_PATH.exists():
        filtered_df = load_filtered_metadata()
        filtered_jcp_ids = set(filtered_df["Metadata_JCP2022"].dropna().unique())
        print(f"\nLoaded pre-computed filtered metadata from: {FILTERED_METADATA_PATH}")
    else:
        filtered_jcp_ids = get_fill_rate_filtered_jcp_ids(original_df, plate_meta, min_fill_rate=25)
        print("\nComputed filtered JCP IDs on the fly (25% min fill rate)")

    print(f"Fully filtered unique JCP IDs: {len(filtered_jcp_ids):,}")

    stats, intersection, only_refchemdb, only_filtered = compute_overlap_stats(
        refchemdb_jcp_ids, filtered_jcp_ids, "RefChemDB", "FullyFiltered"
    )
    print_overlap_table(stats, "RefChemDB", "FullyFiltered")

    by_type_intersection = print_type_breakdown(intersection, perturbation_lists, "Intersection by type")
    by_type_only_refchemdb = print_type_breakdown(only_refchemdb, perturbation_lists, "Only in RefChemDB by type")

    # ==========================================================================
    # Summary Table: Filtering Impact on RefChemDB Coverage
    # ==========================================================================
    print_section_header("5. SUMMARY: RefChemDB Coverage at Each Filtering Stage")

    filtering_stages = [
        ("Original (no filter)", original_jcp_ids),
        ("Source filtered", source_filtered_jcp),
        ("Source + TARGET2 filtered", source_target2_filtered_jcp),
        ("Fully filtered (25% fill)", filtered_jcp_ids),
    ]

    print(f"\n{'Filtering Stage':<35} {'JUMP JCPs':>12} {'RefChemDB':>12} {'Overlap':>12} {'% RefChemDB':>12}")
    print("-" * 85)

    for stage_name, jcp_set in filtering_stages:
        overlap = len(refchemdb_jcp_ids & jcp_set)
        pct = (overlap / len(refchemdb_jcp_ids) * 100) if refchemdb_jcp_ids else 0
        print(f"{stage_name:<35} {len(jcp_set):>12,} {len(refchemdb_jcp_ids):>12,} {overlap:>12,} {pct:>11.1f}%")

    # ==========================================================================
    # RefChemDB-specific breakdowns
    # ==========================================================================
    print_section_header("6. RefChemDB ANNOTATION DETAILS")

    # Show unique annotation tiers
    if "CrossModalityTier" in refchemdb_df.columns:
        print("\nCrossModalityTier distribution:")
        tier_counts = refchemdb_df["CrossModalityTier"].value_counts()
        for tier, count in tier_counts.items():
            print(f"  {tier}: {count:,}")

    if "WithinModalityTier" in refchemdb_df.columns:
        print("\nWithinModalityTier distribution:")
        tier_counts = refchemdb_df["WithinModalityTier"].value_counts()
        for tier, count in tier_counts.items():
            print(f"  {tier}: {count:,}")

    # Show coverage by tier in fully filtered data
    if "CrossModalityTier" in refchemdb_df.columns:
        print("\nRefChemDB JCPs found in fully filtered data, by CrossModalityTier:")
        for tier in sorted(refchemdb_df["CrossModalityTier"].dropna().unique(), key=str):
            tier_jcps = set(refchemdb_df[refchemdb_df["CrossModalityTier"] == tier]["Metadata_JCP2022"].unique())
            overlap_count = len(tier_jcps & filtered_jcp_ids)
            total_tier = len(tier_jcps)
            pct = (overlap_count / total_tier * 100) if total_tier else 0
            print(f"  {str(tier):<20}: {overlap_count:>6,} / {total_tier:>6,} ({pct:>5.1f}%)")

    if "WithinModalityTier" in refchemdb_df.columns:
        print("\nRefChemDB JCPs found in fully filtered data, by WithinModalityTier:")
        for tier in sorted(refchemdb_df["WithinModalityTier"].dropna().unique(), key=str):
            tier_jcps = set(refchemdb_df[refchemdb_df["WithinModalityTier"] == tier]["Metadata_JCP2022"].unique())
            overlap_count = len(tier_jcps & filtered_jcp_ids)
            total_tier = len(tier_jcps)
            pct = (overlap_count / total_tier * 100) if total_tier else 0
            print(f"  {str(tier):<20}: {overlap_count:>6,} / {total_tier:>6,} ({pct:>5.1f}%)")

    # ==========================================================================
    # Final Summary
    # ==========================================================================
    print_section_header("7. KEY FINDINGS")

    final_overlap = len(refchemdb_jcp_ids & filtered_jcp_ids)
    final_coverage_of_refchemdb = (final_overlap / len(refchemdb_jcp_ids) * 100) if refchemdb_jcp_ids else 0
    final_coverage_of_jump = (final_overlap / len(filtered_jcp_ids) * 100) if filtered_jcp_ids else 0

    print(f"""
    RefChemDB annotations available:        {len(refchemdb_jcp_ids):>8,} unique JCP IDs
    JUMP fully filtered dataset:            {len(filtered_jcp_ids):>8,} unique JCP IDs

    Overlap (annotated & in filtered JUMP): {final_overlap:>8,} unique JCP IDs

    Coverage metrics:
      - {final_coverage_of_refchemdb:.1f}% of RefChemDB annotations are in the filtered JUMP data
      - {final_coverage_of_jump:.1f}% of filtered JUMP data has RefChemDB annotations

    RefChemDB JCPs missing from filtered:   {len(only_refchemdb):>8,} JCP IDs
      Breakdown: {len(by_type_only_refchemdb['COMPOUND']):,} COMPOUND, {len(by_type_only_refchemdb['ORF']):,} ORF, {len(by_type_only_refchemdb['CRISPR']):,} CRISPR
    """)


if __name__ == "__main__":
    main()
