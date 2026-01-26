"""
Prepare negative control samples for downloading.

This script:
1. Loads the full JUMP well metadata from broad_babel
2. Gets unique plates from the filtered metadata subset
3. For each plate in the subset, retrieves negative control wells
4. Selects half of the negative control wells per plate (random sample)
5. Creates a metadata.parquet file compatible with download_images.py
"""

import pandas as pd
import polars as pl
import duckdb
from pathlib import Path
from broad_babel.data import get_table


# =============================================================================
# Configuration
# =============================================================================

FILTERED_METADATA_PATH = Path("/home/jfredinh/projects/JUMP_core/metadata/metadata_filtered.parquet")
OUTPUT_DIR = Path("/home/jfredinh/projects/JUMP_core/metadata")

# Negative control JCP IDs mapped to their modality
# Each modality has its own negative controls
MODALITY_NEGCONS = {
    "COMPOUND": ["JCP2022_033924"],  # DMSO
    "CRISPR": ["JCP2022_800001"],    # Non-targeting guide
    "ORF": ["JCP2022_805264", "JCP2022_915128"],  # LacZ/untreated
}

SEED = 42

# Fraction of negative controls to sample per plate, by modality
MODALITY_FRACTION = {
    "COMPOUND": 0.5,  # Take half
    "CRISPR": 0.5,    # Take half
    "ORF": 1.0,       # Take all (few negcons available)
}


# =============================================================================
# Data Loading
# =============================================================================

def load_full_well_metadata():
    """
    Load full JUMP well metadata from broad_babel and join with plate info.
    Returns DataFrame with columns needed for download_images.py.
    """
    print("  Loading well metadata from broad_babel...")
    meta_wells = get_table("well")

    print("  Loading plate metadata from broad_babel...")
    meta_plate = get_table("plate")

    print("  Joining well and plate metadata...")
    con = duckdb.connect()

    # Join wells with plates to get Metadata_Batch
    full_metadata = con.sql("""
        SELECT
            w.Metadata_Source,
            p.Metadata_Batch,
            w.Metadata_Plate,
            w.Metadata_Well,
            w.Metadata_JCP2022
        FROM meta_wells w
        JOIN meta_plate p
            ON w.Metadata_Source = p.Metadata_Source
            AND w.Metadata_Plate = p.Metadata_Plate
    """).df()

    con.close()

    return full_metadata


def load_filtered_metadata():
    """Load filtered metadata to get plate list."""
    return pd.read_parquet(FILTERED_METADATA_PATH)


# =============================================================================
# Negative Control Selection
# =============================================================================

def get_plates_by_modality(filtered_df):
    """
    Get plate sets for each modality from the filtered metadata.

    Args:
        filtered_df: Filtered metadata DataFrame with Perturbation_Type column

    Returns:
        Dict mapping modality to set of plate IDs
    """
    plates_by_modality = {}
    for modality in ["COMPOUND", "CRISPR", "ORF"]:
        plates = set(
            filtered_df[filtered_df["Perturbation_Type"] == modality]["Metadata_Plate"].unique()
        )
        plates_by_modality[modality] = plates
    return plates_by_modality


def get_negative_controls_by_modality(full_metadata, plates_by_modality, modality_negcons):
    """
    Get negative control wells, matching each modality to its appropriate controls.

    Args:
        full_metadata: Full well metadata DataFrame (from broad_babel)
        plates_by_modality: Dict mapping modality to set of plate IDs
        modality_negcons: Dict mapping modality to list of negative control JCP IDs

    Returns:
        DataFrame with negative control wells
    """
    neg_control_dfs = []

    for modality, plates in plates_by_modality.items():
        if modality not in modality_negcons:
            continue

        neg_jcps = modality_negcons[modality]
        mask = (
            full_metadata["Metadata_Plate"].isin(plates) &
            full_metadata["Metadata_JCP2022"].isin(neg_jcps)
        )
        modality_negcons_df = full_metadata[mask].copy()
        modality_negcons_df["Modality"] = modality
        neg_control_dfs.append(modality_negcons_df)

    if neg_control_dfs:
        return pd.concat(neg_control_dfs, ignore_index=True)
    return pd.DataFrame()


def sample_per_plate_by_modality(neg_controls_df, modality_fraction, seed=42):
    """
    Sample a fraction of negative controls from each plate, with modality-specific fractions.

    Args:
        neg_controls_df: DataFrame with negative control wells (must have 'Modality' column)
        modality_fraction: Dict mapping modality to fraction to sample
        seed: Random seed for reproducibility

    Returns:
        DataFrame with sampled negative controls
    """
    sampled_dfs = []

    for plate, group in neg_controls_df.groupby("Metadata_Plate"):
        # Get the modality for this plate (should be single modality per plate)
        modality = group["Modality"].iloc[0]
        fraction = modality_fraction.get(modality, 0.5)

        if fraction >= 1.0:
            # Take all
            sampled_dfs.append(group)
        else:
            n_samples = max(1, int(len(group) * fraction))  # At least 1 sample
            sampled = group.sample(n=n_samples, random_state=seed)
            sampled_dfs.append(sampled)

    if sampled_dfs:
        return pd.concat(sampled_dfs, ignore_index=True)
    return pd.DataFrame(columns=neg_controls_df.columns)


def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("NEGATIVE CONTROL SAMPLE PREPARATION")
    print("=" * 70)

    # Load filtered metadata to get plate list by modality
    print("\nLoading filtered metadata (for plate list)...")
    filtered_df = load_filtered_metadata()
    print(f"  Filtered metadata: {len(filtered_df):,} rows")
    print(f"  Unique plates: {filtered_df['Metadata_Plate'].nunique():,}")

    # Get plates by modality
    plates_by_modality = get_plates_by_modality(filtered_df)
    print("\n  Plates by modality:")
    for modality, plates in plates_by_modality.items():
        print(f"    {modality}: {len(plates)} plates")

    # Load full well metadata from broad_babel
    print("\nLoading full JUMP well metadata from broad_babel...")
    full_metadata = load_full_well_metadata()
    print(f"  Full metadata: {len(full_metadata):,} rows")
    print(f"  Total unique plates: {full_metadata['Metadata_Plate'].nunique():,}")

    # Get negative controls matched to their modality plates
    print_section_header("NEGATIVE CONTROLS BY MODALITY")

    neg_controls = get_negative_controls_by_modality(
        full_metadata, plates_by_modality, MODALITY_NEGCONS
    )

    print(f"\nTotal negative control wells found: {len(neg_controls):,}")

    print("\nBreakdown by modality and control type:")
    for modality in ["COMPOUND", "CRISPR", "ORF"]:
        modality_data = neg_controls[neg_controls["Modality"] == modality]
        if len(modality_data) == 0:
            continue
        print(f"\n  {modality}:")
        for jcp in MODALITY_NEGCONS.get(modality, []):
            jcp_data = modality_data[modality_data["Metadata_JCP2022"] == jcp]
            count = len(jcp_data)
            plates_with = jcp_data["Metadata_Plate"].nunique()
            if count > 0:
                print(f"    {jcp}: {count:,} wells across {plates_with} plates")

    plates_with_negcon = neg_controls["Metadata_Plate"].nunique()
    total_plates = sum(len(p) for p in plates_by_modality.values())
    plates_without_negcon = total_plates - plates_with_negcon
    print(f"\nPlates with negative controls: {plates_with_negcon}")
    print(f"Plates WITHOUT negative controls: {plates_without_negcon}")

    # Sample per plate with modality-specific fractions
    print_section_header("SAMPLING NEGATIVE CONTROLS PER PLATE")

    print("\nSampling fractions by modality:")
    for modality, frac in MODALITY_FRACTION.items():
        print(f"  {modality}: {frac*100:.0f}%")

    sampled = sample_per_plate_by_modality(neg_controls, MODALITY_FRACTION, seed=SEED)

    print(f"\nSampled negative control wells: {len(sampled):,}")
    print(f"  (from {len(neg_controls):,} total)")

    print("\nSampled breakdown by modality and control type:")
    for modality in ["COMPOUND", "CRISPR", "ORF"]:
        modality_data = sampled[sampled["Modality"] == modality]
        if len(modality_data) == 0:
            continue
        print(f"\n  {modality}:")
        for jcp in MODALITY_NEGCONS.get(modality, []):
            jcp_data = modality_data[modality_data["Metadata_JCP2022"] == jcp]
            count = len(jcp_data)
            plates_with = jcp_data["Metadata_Plate"].nunique()
            if count > 0:
                print(f"    {jcp}: {count:,} wells across {plates_with} plates")

    print(f"\nPlates represented: {sampled['Metadata_Plate'].nunique()}")

    # Distribution of samples per plate
    samples_per_plate = sampled.groupby("Metadata_Plate").size()
    print(f"\nSamples per plate statistics:")
    print(f"  Min: {samples_per_plate.min()}")
    print(f"  Max: {samples_per_plate.max()}")
    print(f"  Mean: {samples_per_plate.mean():.1f}")
    print(f"  Median: {samples_per_plate.median():.1f}")

    # Save as parquet (same format as download_images.py expects)
    print_section_header("SAVING METADATA FOR DOWNLOAD")

    # Select columns in the format expected by download_images.py
    output_cols = [
        "Metadata_Source",
        "Metadata_Batch",
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_JCP2022"
    ]

    output_df = sampled[output_cols].copy()

    # Save as parquet using polars for consistency with download_images.py
    output_file = OUTPUT_DIR / "metadata_negative_controls.parquet"
    output_pl = pl.from_pandas(output_df)
    output_pl.write_parquet(output_file)

    print(f"\nMetadata saved to: {output_file}")
    print(f"  Rows: {len(output_df):,}")
    print(f"  Unique wells: {len(output_df):,}")
    print(f"  Unique plates: {output_df['Metadata_Plate'].nunique():,}")

    # Verify the saved file
    print("\nVerifying saved file...")
    verify_df = pl.read_parquet(output_file)
    print(f"  Loaded {len(verify_df):,} rows")
    print(f"  Columns: {verify_df.columns}")

    # Summary
    print_section_header("SUMMARY")

    channels = 5  # DNA, AGP, Mito, RNA, ER
    sites = 4     # Sites 1-4
    total_images = len(output_df) * channels * sites

    print(f"""
    Negative control metadata prepared for download:

      Wells to download:     {len(output_df):,}
      Plates covered:        {output_df['Metadata_Plate'].nunique():,}

      Estimated images:      {total_images:,}
        (assuming {channels} channels x {sites} sites per well)

    Output file: {output_file}

    To download images, copy this file to /work/datasets/jump_core/metadata.parquet
    or modify download_images.py to use this file path.
    """)


if __name__ == "__main__":
    main()
