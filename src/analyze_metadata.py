"""
Analyze JUMP metadata: filter, count, and visualize plate fill rates.

This script:
1. Filters out source_7 and 1536-well plates
2. Counts unique JCP IDs by perturbation type (CRISPR, ORF, COMPOUND)
3. Counts unique plates by category
4. Calculates fill rate per plate (wells present / 384)
5. Visualizes fill rates ordered by percentage
"""

import pandas as pd
import matplotlib.pyplot as plt
from broad_babel.data import get_table
from pathlib import Path


# =============================================================================
# Configuration
# =============================================================================

METADATA_PATH = Path("/work/datasets/jump_core/metadata.parquet")
OUTPUT_DIR = Path("/home/jfredinh/projects/JUMP_core/metadata_analysis")
WELLS_PER_384_PLATE = 384


# =============================================================================
# Step 1: Load data
# =============================================================================

def load_metadata():
    """Load the main metadata parquet file."""
    return pd.read_parquet(METADATA_PATH)


def load_perturbation_lists():
    """Load sets of JCP IDs for each perturbation type."""
    # get_table returns polars DataFrames, convert to pandas
    crispr = set(get_table("crispr").to_pandas()["Metadata_JCP2022"].dropna().unique())
    orf = set(get_table("orf").to_pandas()["Metadata_JCP2022"].dropna().unique())
    compound = set(get_table("compound").to_pandas()["Metadata_JCP2022"].dropna().unique())

    return {"CRISPR": crispr, "ORF": orf, "COMPOUND": compound}


def load_plate_metadata():
    """Load plate metadata from broad_babel."""
    plate_df = get_table("plate").to_pandas()
    print(f"Plate metadata columns: {plate_df.columns.tolist()}")
    return plate_df


# =============================================================================
# Step 2: Filter data
# =============================================================================

def filter_metadata(df, plate_meta):
    """
    Remove:
    - source_7 (lower concentration data)
    - source_9 (1536-well plates)
    - Plates with Metadata_PlateType containing 'TARGET2'

    Returns:
        tuple: (df_with_target2_filtered, df_without_target2_filtered)
    """
    # Get TARGET2 plates to exclude based on plate metadata
    plates_target2 = set(
        plate_meta[plate_meta["Metadata_PlateType"].str.contains("TARGET2", case=False, na=False)]["Metadata_Plate"]
    )

    #sources_to_exclude = ["source_7", "source_9"]
    sources_to_exclude = ["source_9"]

    print(f"Excluding sources: {sources_to_exclude}")
    print(f"Excluding TARGET2 plates: {len(plates_target2)}")

    source_mask = ~df["Metadata_Source"].isin(sources_to_exclude)
    plate_mask = ~df["Metadata_Plate"].isin(plates_target2)

    df_source_filtered = df[source_mask].copy()
    df_full_filtered = df[source_mask & plate_mask].copy()

    return df_full_filtered, df_source_filtered


# =============================================================================
# Step 3: Classify perturbation types
# =============================================================================

def classify_jcp_ids(df, perturbation_lists):
    """Add perturbation type column based on JCP ID lookups."""

    def get_type(jcp_id):
        if jcp_id in perturbation_lists["CRISPR"]:
            return "CRISPR"
        if jcp_id in perturbation_lists["ORF"]:
            return "ORF"
        if jcp_id in perturbation_lists["COMPOUND"]:
            return "COMPOUND"
        return "UNKNOWN"

    # Build mapping for unique JCP IDs only
    unique_jcp = df["Metadata_JCP2022"].unique()
    type_map = {jcp: get_type(jcp) for jcp in unique_jcp}

    df["Perturbation_Type"] = df["Metadata_JCP2022"].map(type_map)
    return df


# =============================================================================
# Step 4: Compute statistics
# =============================================================================

def count_unique_jcp_ids(df):
    """Count unique JCP IDs by perturbation type."""
    return df.groupby("Perturbation_Type")["Metadata_JCP2022"].nunique()


def count_unique_plates(df):
    """Count unique plates by source."""
    return df.groupby("Metadata_Source")["Metadata_Plate"].nunique()


def count_unique_plates_by_modality(df):
    """Count unique plates by perturbation type (modality)."""
    return df.groupby("Perturbation_Type")["Metadata_Plate"].nunique()


def count_unique_wells(df):
    """Count unique wells (plate + well combinations) by source."""
    return df.groupby("Metadata_Source").apply(
        lambda x: x[["Metadata_Plate", "Metadata_Well"]].drop_duplicates().shape[0]
    )


def count_unique_well_names(df):
    """Count unique well names (e.g., A01, B02) by source."""
    return df.groupby("Metadata_Source")["Metadata_Well"].nunique()


def calculate_plate_fill_rates(df):
    """Calculate fill rate for each plate (wells present / 384)."""
    plate_fills = (
        df.groupby(["Metadata_Source", "Metadata_Plate"])["Metadata_Well"]
        .nunique()
        .reset_index(name="Wells_Present")
    )

    plate_fills["Fill_Rate_Percent"] = (
        plate_fills["Wells_Present"] / WELLS_PER_384_PLATE * 100
    ).round(2)

    return plate_fills.sort_values("Fill_Rate_Percent", ascending=False)


def filter_by_fill_rate(df, plate_fills, min_fill_rate=25):
    """Filter metadata to only include wells from plates above minimum fill rate."""
    valid_plates = plate_fills[plate_fills["Fill_Rate_Percent"] >= min_fill_rate]
    valid_plate_set = set(zip(valid_plates["Metadata_Source"], valid_plates["Metadata_Plate"]))

    mask = df.apply(
        lambda row: (row["Metadata_Source"], row["Metadata_Plate"]) in valid_plate_set,
        axis=1
    )
    return df[mask].copy()


def compute_compound_overlap_matrix(df):
    """
    Compute percentage overlap of unique compounds between each source.
    Entry [i,j] = % of source i's compounds that are also in source j.
    """
    # Get compounds per source
    compound_df = df[df["Perturbation_Type"] == "COMPOUND"]
    sources = sorted(compound_df["Metadata_Source"].unique())

    compounds_by_source = {
        src: set(compound_df[compound_df["Metadata_Source"] == src]["Metadata_JCP2022"].unique())
        for src in sources
    }

    # Debug: print compound counts per source
    print("\nDebug - Unique compounds per source:")
    for src in sources:
        print(f"  {src}: {len(compounds_by_source[src])} compounds")

    # Build overlap matrix
    overlap_matrix = pd.DataFrame(index=sources, columns=sources, dtype=float)

    for src_i in sources:
        for src_j in sources:
            compounds_i = compounds_by_source[src_i]
            compounds_j = compounds_by_source[src_j]

            if len(compounds_i) == 0:
                overlap_matrix.loc[src_i, src_j] = 0.0
            else:
                overlap = len(compounds_i & compounds_j)
                overlap_matrix.loc[src_i, src_j] = (overlap / len(compounds_i)) * 100

    # Debug: print a few example overlaps
    print("\nDebug - Example overlaps (absolute counts):")
    for i, src_i in enumerate(sources[:3]):
        for j, src_j in enumerate(sources[:3]):
            overlap = len(compounds_by_source[src_i] & compounds_by_source[src_j])
            print(f"  {src_i} & {src_j}: {overlap} compounds in common")

    return overlap_matrix.round(3)


def plot_compound_overlap_heatmap(overlap_matrix, output_path):
    """Plot heatmap of compound overlap between sources."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(overlap_matrix.values.astype(float), cmap="YlOrRd", vmin=0, vmax=100)

    # Set ticks and labels
    ax.set_xticks(range(len(overlap_matrix.columns)))
    ax.set_yticks(range(len(overlap_matrix.index)))
    ax.set_xticklabels(overlap_matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(overlap_matrix.index)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Overlap %", rotation=270, labelpad=15)

    # Add text annotations
    for i in range(len(overlap_matrix.index)):
        for j in range(len(overlap_matrix.columns)):
            value = overlap_matrix.iloc[i, j]
            color = "white" if value > 50 else "black"
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", color=color, fontsize=8)

    ax.set_xlabel("Source")
    ax.set_ylabel("Source")
    ax.set_title("Compound Overlap Between Sources\n(% of row source's compounds found in column source)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Overlap heatmap saved to: {output_path}")


def compare_compounds_per_source(before_df, after_df):
    """Compare unique compounds per source before and after filtering."""
    # Count unique compounds (COMPOUND type only) per source
    before_counts = (
        before_df[before_df["Perturbation_Type"] == "COMPOUND"]
        .groupby("Metadata_Source")["Metadata_JCP2022"]
        .nunique()
        .rename("Before")
    )

    after_counts = (
        after_df[after_df["Perturbation_Type"] == "COMPOUND"]
        .groupby("Metadata_Source")["Metadata_JCP2022"]
        .nunique()
        .rename("After")
    )

    # Combine into comparison table
    comparison = pd.concat([before_counts, after_counts], axis=1).fillna(0).astype(int)
    comparison["Diff"] = comparison["After"] - comparison["Before"]
    comparison["Retained_%"] = (comparison["After"] / comparison["Before"] * 100).round(1)
    comparison = comparison.reset_index()

    # Add totals row
    totals = pd.DataFrame([{
        "Metadata_Source": "TOTAL",
        "Before": comparison["Before"].sum(),
        "After": comparison["After"].sum(),
        "Diff": comparison["Diff"].sum(),
        "Retained_%": (comparison["After"].sum() / comparison["Before"].sum() * 100).round(1)
    }])
    comparison = pd.concat([comparison, totals], ignore_index=True)

    return comparison


def verify_saved_metadata(original_df, saved_path):
    """Verify that saved parquet matches the original dataframe."""
    print("\nVerifying saved metadata...")

    loaded_df = pd.read_parquet(saved_path)

    # Check row count
    assert len(loaded_df) == len(original_df), \
        f"Row count mismatch: {len(loaded_df)} vs {len(original_df)}"

    # Check column names
    assert list(loaded_df.columns) == list(original_df.columns), \
        f"Column mismatch: {list(loaded_df.columns)} vs {list(original_df.columns)}"

    # Check unique JCP IDs
    loaded_jcp = loaded_df["Metadata_JCP2022"].nunique()
    original_jcp = original_df["Metadata_JCP2022"].nunique()
    assert loaded_jcp == original_jcp, \
        f"Unique JCP ID mismatch: {loaded_jcp} vs {original_jcp}"

    # Check unique plates
    loaded_plates = loaded_df["Metadata_Plate"].nunique()
    original_plates = original_df["Metadata_Plate"].nunique()
    assert loaded_plates == original_plates, \
        f"Unique plate mismatch: {loaded_plates} vs {original_plates}"

    # Check data content (sample comparison)
    original_sorted = original_df.sort_values(
        ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]
    ).reset_index(drop=True)
    loaded_sorted = loaded_df.sort_values(
        ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]
    ).reset_index(drop=True)

    # Compare key columns
    for col in ["Metadata_Source", "Metadata_Plate", "Metadata_Well", "Metadata_JCP2022"]:
        assert (original_sorted[col] == loaded_sorted[col]).all(), \
            f"Data mismatch in column: {col}"

    print("  All verification checks passed!")


# =============================================================================
# Step 5: Visualization
# =============================================================================

def compute_threshold_table(df, plate_fills):
    """
    Compute table showing JCP IDs and replicates at different fill rate thresholds.
    Shows what happens when dropping plates below X% fill rate.
    """
    thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    results = []

    for threshold in thresholds:
        # Get plates above threshold
        valid_plates = plate_fills[plate_fills["Fill_Rate_Percent"] >= threshold]
        valid_plate_set = set(zip(valid_plates["Metadata_Source"], valid_plates["Metadata_Plate"]))

        # Filter data to only include wells from valid plates
        mask = df.apply(
            lambda row: (row["Metadata_Source"], row["Metadata_Plate"]) in valid_plate_set,
            axis=1
        )
        filtered = df[mask]

        if len(filtered) == 0:
            results.append({
                "Min_Fill_Rate": threshold,
                "Plates_Remaining": 0,
                "Unique_JCP_IDs": 0,
                "Unique_COMPOUND": 0,
                "Unique_CRISPR": 0,
                "Unique_ORF": 0,
                "Median_Replicates": 0,
                "Total_Wells": 0
            })
            continue

        # Count unique JCP IDs (total)
        unique_jcp_ids = filtered["Metadata_JCP2022"].nunique()

        # Count unique JCP IDs by perturbation type
        unique_compound = filtered[filtered["Perturbation_Type"] == "COMPOUND"]["Metadata_JCP2022"].nunique()
        unique_crispr = filtered[filtered["Perturbation_Type"] == "CRISPR"]["Metadata_JCP2022"].nunique()
        unique_orf = filtered[filtered["Perturbation_Type"] == "ORF"]["Metadata_JCP2022"].nunique()

        # Calculate replicates per JCP ID (all types)
        replicates = filtered.groupby("Metadata_JCP2022").size()
        median_replicates = replicates.median()
        p25_replicates = replicates.quantile(0.25)

        # Calculate replicates for compounds only
        compound_data = filtered[filtered["Perturbation_Type"] == "COMPOUND"]
        if len(compound_data) > 0:
            compound_replicates = compound_data.groupby("Metadata_JCP2022").size()
            median_compound_replicates = compound_replicates.median()
            p25_compound_replicates = compound_replicates.quantile(0.25)
        else:
            median_compound_replicates = 0
            p25_compound_replicates = 0

        results.append({
            "Min_Fill_Rate": threshold,
            "Plates_Remaining": len(valid_plates),
            "Unique_JCP_IDs": unique_jcp_ids,
            "Unique_COMPOUND": unique_compound,
            "Unique_CRISPR": unique_crispr,
            "Unique_ORF": unique_orf,
            "Median_Replicates": median_replicates,
            "P25_Replicates": p25_replicates,
            "Median_COMPOUND_Replicates": median_compound_replicates,
            "P25_COMPOUND_Replicates": p25_compound_replicates,
            "Total_Wells": len(filtered)
        })

    return pd.DataFrame(results)


def plot_fill_rates(plate_fills, output_path):
    """Create a bar chart of plate fill rates, ordered by percentage."""

    df = plate_fills.sort_values("Fill_Rate_Percent")

    # Color by source
    unique_sources = sorted(df["Metadata_Source"].unique())
    colors = plt.cm.tab10.colors
    source_colors = {src: colors[i % len(colors)] for i, src in enumerate(unique_sources)}
    bar_colors = df["Metadata_Source"].map(source_colors)

    fig, ax = plt.subplots(figsize=(14, 8))

    ax.barh(range(len(df)), df["Fill_Rate_Percent"], color=bar_colors, height=0.8)

    ax.set_xlabel("Fill Rate (%)", fontsize=12)
    ax.set_ylabel("Plate Index (ordered by fill rate)", fontsize=12)
    ax.set_title("Plate Fill Rates (Wells Present / 384)", fontsize=14)
    ax.set_xlim(0, 105)

    # Reference line at 100%
    ax.axvline(x=100, color="red", linestyle="--", linewidth=1, alpha=0.7)

    # Legend for sources
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=source_colors[src], label=src)
        for src in unique_sources
    ]
    ax.legend(handles=handles, loc="lower right", title="Source")

    # Hide y-ticks if too many plates
    if len(df) > 50:
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    print("Loading metadata...")
    df = load_metadata()
    print(f"  Total rows: {len(df):,}")

    print("Loading perturbation type lists...")
    perturbation_lists = load_perturbation_lists()

    print("Loading plate metadata...")
    plate_meta = load_plate_metadata()

    # Filter
    print("\nFiltering data...")
    df, df_with_target2 = filter_metadata(df, plate_meta)
    print(f"  Rows after filtering (no TARGET2): {len(df):,}")
    print(f"  Rows after filtering (with TARGET2): {len(df_with_target2):,}")

    # Classify
    print("\nClassifying perturbation types...")
    df = classify_jcp_ids(df, perturbation_lists)
    df_with_target2 = classify_jcp_ids(df_with_target2, perturbation_lists)

    # Statistics
    print("\n" + "=" * 60)
    print("UNIQUE JCP IDs BY PERTURBATION TYPE")
    print("=" * 60)
    jcp_counts = count_unique_jcp_ids(df)
    print(jcp_counts)
    print(f"\nTotal unique JCP IDs: {df['Metadata_JCP2022'].nunique():,}")

    print("\n" + "=" * 60)
    print("UNIQUE PLATES BY SOURCE")
    print("=" * 60)
    plate_counts = count_unique_plates(df)
    print(plate_counts)
    print(f"\nTotal unique plates: {df['Metadata_Plate'].nunique():,}")

    print("\n" + "=" * 60)
    print("UNIQUE WELLS BY SOURCE")
    print("=" * 60)
    well_counts = count_unique_wells(df)
    print(well_counts)
    print(f"\nTotal unique wells: {len(df.drop_duplicates(['Metadata_Plate', 'Metadata_Well'])):,}")

    print("\n" + "=" * 60)
    print("UNIQUE WELL NAMES BY SOURCE")
    print("=" * 60)
    well_name_counts = count_unique_well_names(df)
    print(well_name_counts)
    print(f"\nTotal unique well names: {df['Metadata_Well'].nunique():,}")

    print("\n" + "=" * 60)
    print("PLATE FILL RATES")
    print("=" * 60)
    plate_fills = calculate_plate_fill_rates(df)

    print("\nFill rate statistics:")
    print(plate_fills["Fill_Rate_Percent"].describe())

    print("\nLowest fill rates:")
    print(plate_fills.tail(10).to_string())

    print("\nHighest fill rates:")
    print(plate_fills.head(10).to_string())

    # Visualize
    print("\nGenerating plot...")
    plot_fill_rates(plate_fills, OUTPUT_DIR / "plate_fill_rates.png")

    # Threshold analysis
    print("\n" + "=" * 60)
    print("FILL RATE THRESHOLD ANALYSIS")
    print("=" * 60)
    print("Effect of dropping plates below X% fill rate:\n")
    threshold_table = compute_threshold_table(df, plate_fills)
    print(threshold_table.to_string(index=False))

    # Save results
    plate_fills.to_csv(OUTPUT_DIR / "plate_fill_rates.csv", index=False)
    threshold_table.to_csv(OUTPUT_DIR / "threshold_analysis.csv", index=False)
    print(f"\nFill rates saved to: {OUTPUT_DIR / 'plate_fill_rates.csv'}")
    print(f"Threshold analysis saved to: {OUTPUT_DIR / 'threshold_analysis.csv'}")

    # Save filtered metadata with 25% minimum fill rate
    print("\n" + "=" * 60)
    print("SAVING FILTERED METADATA (25% MIN FILL RATE)")
    print("=" * 60)

    metadata_output_dir = Path("/home/jfredinh/projects/JUMP_core/metadata")
    metadata_output_dir.mkdir(parents=True, exist_ok=True)

    # Save version WITHOUT TARGET2 plates
    filtered_metadata = filter_by_fill_rate(df, plate_fills, min_fill_rate=25)
    output_file = metadata_output_dir / "metadata_filtered.parquet"
    filtered_metadata.to_parquet(output_file, index=False)
    print(f"\nFiltered metadata (no TARGET2) saved to: {output_file}")
    print(f"  Rows: {len(filtered_metadata):,}")
    print(f"  Unique JCP IDs: {filtered_metadata['Metadata_JCP2022'].nunique():,}")
    print(f"  Unique plates: {filtered_metadata['Metadata_Plate'].nunique():,}")
    plates_by_modality = count_unique_plates_by_modality(filtered_metadata)
    for modality in ["COMPOUND", "CRISPR", "ORF", "UNKNOWN"]:
        if modality in plates_by_modality.index:
            print(f"    {modality}: {plates_by_modality[modality]:,} plates")

    # Save version WITH TARGET2 plates
    plate_fills_with_target2 = calculate_plate_fill_rates(df_with_target2)
    filtered_metadata_with_target2 = filter_by_fill_rate(df_with_target2, plate_fills_with_target2, min_fill_rate=25)
    output_file_with_target2 = metadata_output_dir / "metadata_filtered_with_target2.parquet"
    filtered_metadata_with_target2.to_parquet(output_file_with_target2, index=False)
    print(f"\nFiltered metadata (with TARGET2) saved to: {output_file_with_target2}")
    print(f"  Rows: {len(filtered_metadata_with_target2):,}")
    print(f"  Unique JCP IDs: {filtered_metadata_with_target2['Metadata_JCP2022'].nunique():,}")
    print(f"  Unique plates: {filtered_metadata_with_target2['Metadata_Plate'].nunique():,}")
    plates_by_modality_t2 = count_unique_plates_by_modality(filtered_metadata_with_target2)
    for modality in ["COMPOUND", "CRISPR", "ORF", "UNKNOWN"]:
        if modality in plates_by_modality_t2.index:
            print(f"    {modality}: {plates_by_modality_t2[modality]:,} plates")

    # Compare unique compounds per source before and after filtering
    print("\n" + "=" * 60)
    print("UNIQUE COMPOUNDS PER SOURCE: BEFORE vs AFTER 25% FILTER")
    print("=" * 60)
    comparison = compare_compounds_per_source(df, filtered_metadata)
    print(comparison.to_string(index=False))

    # Compound overlap heatmap between sources
    print("\n" + "=" * 60)
    print("COMPOUND OVERLAP BETWEEN SOURCES (AFTER FILTERING)")
    print("=" * 60)
    overlap_matrix = compute_compound_overlap_matrix(filtered_metadata)
    print(overlap_matrix.to_string())
    plot_compound_overlap_heatmap(overlap_matrix, OUTPUT_DIR / "compound_overlap_heatmap.png")
    overlap_matrix.to_csv(OUTPUT_DIR / "compound_overlap_matrix.csv")

    # Verify saved files match
    verify_saved_metadata(filtered_metadata, output_file)
    verify_saved_metadata(filtered_metadata_with_target2, output_file_with_target2)


if __name__ == "__main__":
    main()
