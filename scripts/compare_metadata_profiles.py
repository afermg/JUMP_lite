#!/usr/bin/env python
"""
Compare wells and plates between metadata files and profiles.

This script loads metadata parquet files and compares them to the
raw JUMP CP profiles to summarize the overlap.
"""

import polars as pl
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils.redlisted_plates import excluded_plates


def main():
    # Define paths
    metadata_dir = Path("/home/jfredinh/projects/JUMP_core/metadata")
    profiles_path = Path("/work/datasets/jump_core_annotated/raw_jump_CP_profiles/profiles.parquet")

    # Load metadata files
    print("Loading metadata files...")
    df_filtered = pl.read_parquet(metadata_dir / "metadata_filtered.parquet")
    df_negative = pl.read_parquet(metadata_dir / "metadata_negative_controls.parquet")

    print(f"  metadata_filtered.parquet: {df_filtered.shape[0]:,} rows")
    print(f"  metadata_negative_controls.parquet: {df_negative.shape[0]:,} rows")

    # Combine metadata (use common columns)
    common_cols = ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"]
    df_metadata = pl.concat([
        df_filtered.select(common_cols),
        df_negative.select(common_cols)
    ])
    print(f"  Combined metadata: {df_metadata.shape[0]:,} rows")

    # Load profiles (only metadata columns needed)
    print("\nLoading profiles.parquet...")
    df_profiles = pl.scan_parquet(profiles_path).select([
        "Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"
    ]).collect()
    print(f"  profiles.parquet: {df_profiles.shape[0]:,} rows")

    # === PLATE ANALYSIS PER SOURCE ===
    print("\n" + "=" * 60)
    print("PLATE ANALYSIS PER SOURCE")
    print("=" * 60)

    # Get all unique sources from both datasets
    all_sources = sorted(
        set(df_metadata.select("Metadata_Source").unique()["Metadata_Source"].to_list()) 
    )

    for source in all_sources:
        print(f"\n--- {source} ---")

        # Get unique plates for this source
        metadata_plates_src = set(
            df_metadata.filter(pl.col("Metadata_Source") == source)
            .select("Metadata_Plate").unique()["Metadata_Plate"].to_list()
        )
        profiles_plates_src = set(
            df_profiles.filter(pl.col("Metadata_Source") == source)
            .select("Metadata_Plate").unique()["Metadata_Plate"].to_list()
        )

        plates_intersection_src = metadata_plates_src & profiles_plates_src
        plates_union_src = metadata_plates_src | profiles_plates_src
        plates_only_metadata_src = metadata_plates_src - profiles_plates_src
        plates_only_profiles_src = profiles_plates_src - metadata_plates_src

        print(f"Unique plates in metadata:      {len(metadata_plates_src):,}")
        print(f"Unique plates in profiles:      {len(profiles_plates_src):,}")
        print(f"Plates in BOTH (intersection):  {len(plates_intersection_src):,}")
        print(f"Plates in EITHER (union):       {len(plates_union_src):,}")
        print(f"Plates ONLY in metadata:        {len(plates_only_metadata_src):,}")
        print(f"Plates ONLY in profiles:        {len(plates_only_profiles_src):,}")

        if plates_only_metadata_src:
            
            # Per-batch comparison within this source
            df_metadata_src = df_metadata.filter(pl.col("Metadata_Source") == source)
            df_profiles_src = df_profiles.filter(pl.col("Metadata_Source") == source)

            all_batches = sorted(
                set(df_metadata_src.select("Metadata_Batch").unique()["Metadata_Batch"].to_list()) |
                set(df_profiles_src.select("Metadata_Batch").unique()["Metadata_Batch"].to_list())
            )

            for batch in all_batches:
                metadata_plates_batch = set(
                    df_metadata_src.filter(pl.col("Metadata_Batch") == batch)
                    .select("Metadata_Plate").unique()["Metadata_Plate"].to_list()
                )
                profiles_plates_batch = set(
                    df_profiles_src.filter(pl.col("Metadata_Batch") == batch)
                    .select("Metadata_Plate").unique()["Metadata_Plate"].to_list()
                )

                plates_only_metadata_batch = metadata_plates_batch - profiles_plates_batch
                plates_only_profiles_batch = profiles_plates_batch - metadata_plates_batch

                if plates_only_metadata_batch:
                    print(f"\n  Batch: {batch}")
                    print(f"    Plates in metadata: {len(metadata_plates_batch):,}, in profiles: {len(profiles_plates_batch):,}")
                    print(f"    Only in metadata: {len(plates_only_metadata_batch):,}, only in profiles: {len(plates_only_profiles_batch):,}")

                    # Check which missing plates are in redlist
                    redlisted = {p: excluded_plates[p] for p in plates_only_metadata_batch if p in excluded_plates}
                    not_redlisted = plates_only_metadata_batch - set(excluded_plates.keys())
                    if redlisted:
                        print(f"      Redlisted ({len(redlisted)}): {list(redlisted.keys())}")
                        # for plate, reason in redlisted.items():
                        #     print(f"        {plate}: {reason}")
                    if not_redlisted:
                        print(f"      NOT in redlist ({len(not_redlisted)}): {sorted(not_redlisted)}")
                    
    # === PLATE ANALYSIS ===
    print("\n" + "=" * 60)
    print("PLATE ANALYSIS (ALL SOURCES)")
    print("=" * 60)

    # Get unique plates
    metadata_plates = set(df_metadata.select("Metadata_Plate").unique()["Metadata_Plate"].to_list())
    profiles_plates = set(df_profiles.select("Metadata_Plate").unique()["Metadata_Plate"].to_list())

    plates_intersection = metadata_plates & profiles_plates
    plates_union = metadata_plates | profiles_plates
    plates_only_metadata = metadata_plates - profiles_plates
    plates_only_profiles = profiles_plates - metadata_plates

    print(f"\nUnique plates in metadata:      {len(metadata_plates):,}")
    print(f"Unique plates in profiles:      {len(profiles_plates):,}")
    print(f"Plates in BOTH (intersection):  {len(plates_intersection):,}")
    print(f"Plates in EITHER (union):       {len(plates_union):,}")
    print(f"Plates ONLY in metadata:        {len(plates_only_metadata):,}")
    print(f"Plates ONLY in profiles:        {len(plates_only_profiles):,}")

    if len(plates_union) > 0:
        overlap_pct = len(plates_intersection) / len(plates_union) * 100
        print(f"Overlap (Jaccard index):        {overlap_pct:.1f}%")

    # === MISSING PLATE ANALYSIS ===
    print("\n" + "=" * 60)
    print("MISSING PLATE ANALYSIS")
    print("=" * 60)

    # Missing plates explained by redlist
    print("\nChecking missing plates against redlist...")
    print(f"  Plates only in metadata: {len(plates_only_metadata):,}, redlisted plates: {len(excluded_plates):,}")
    
    missing_explination = [ p for p in plates_only_metadata if (p not in excluded_plates) ]
    print(f"  Plates only in metadata not explained by redlist: {len([p for p in missing_explination if p is None]):,}")

    if len(missing_explination) > 0:
        print("\nWARNING: The following plates are missing from profiles but NOT in redlist:")
        

    # === WELL ANALYSIS (Plate + Well combination) ===
    print("\n" + "=" * 60)
    print("WELL ANALYSIS (Plate + Well unique combinations)")
    print("=" * 60)

    # Filter out excluded plates from metadata
    df_metadata = df_metadata.filter(~pl.col("Metadata_Plate").is_in(list(excluded_plates.keys())))
    print(f"\n(Filtered out {len(excluded_plates)} redlisted plates from metadata)")

    # Create plate_well identifier
    metadata_wells = set(
        df_metadata.select(
            pl.concat_str(["Metadata_Plate", "Metadata_Well"], separator="_").alias("plate_well")
        ).unique()["plate_well"].to_list()
    )
    profiles_wells = set(
        df_profiles.select(
            pl.concat_str(["Metadata_Plate", "Metadata_Well"], separator="_").alias("plate_well")
        ).unique()["plate_well"].to_list()
    )

    wells_intersection = metadata_wells & profiles_wells
    wells_union = metadata_wells | profiles_wells
    wells_only_metadata = metadata_wells - profiles_wells
    wells_only_profiles = profiles_wells - metadata_wells

    print(f"\nUnique wells in metadata:       {len(metadata_wells):,}")
    print(f"Unique wells in profiles:       {len(profiles_wells):,}")
    print(f"Wells in BOTH (intersection):   {len(wells_intersection):,}")
    print(f"Wells in EITHER (union):        {len(wells_union):,}")
    print(f"Wells ONLY in metadata:         {len(wells_only_metadata):,}")
    print(f"Wells ONLY in profiles:         {len(wells_only_profiles):,}")

    if len(wells_union) > 0:
        overlap_pct = len(wells_intersection) / len(wells_union) * 100
        print(f"Overlap (Jaccard index):        {overlap_pct:.1f}%")

    # === SOURCE + PLATE + WELL ANALYSIS ===
    print("\n" + "=" * 60)
    print("FULL KEY ANALYSIS (Source + Plate + Well)")
    print("=" * 60)

    metadata_full = set(
        df_metadata.select(
            pl.concat_str(["Metadata_Source", "Metadata_Plate", "Metadata_Well"], separator="_").alias("full_key")
        ).unique()["full_key"].to_list()
    )
    profiles_full = set(
        df_profiles.select(
            pl.concat_str(["Metadata_Source", "Metadata_Plate", "Metadata_Well"], separator="_").alias("full_key")
        ).unique()["full_key"].to_list()
    )

    full_intersection = metadata_full & profiles_full
    full_union = metadata_full | profiles_full
    full_only_metadata = metadata_full - profiles_full
    full_only_profiles = profiles_full - metadata_full

    print(f"\nUnique keys in metadata:        {len(metadata_full):,}")
    print(f"Unique keys in profiles:        {len(profiles_full):,}")
    print(f"Keys in BOTH (intersection):    {len(full_intersection):,}")
    print(f"Keys in EITHER (union):         {len(full_union):,}")
    print(f"Keys ONLY in metadata:          {len(full_only_metadata):,}")
    print(f"Keys ONLY in profiles:          {len(full_only_profiles):,}")

    if len(full_union) > 0:
        overlap_pct = len(full_intersection) / len(full_union) * 100
        print(f"Overlap (Jaccard index):        {overlap_pct:.1f}%")

    # Breakdown of keys only in metadata
    if full_only_metadata:
        print("\n--- Breakdown of Keys ONLY in metadata ---")

        # Add full_key column to filter
        metadata_with_key = df_metadata.with_columns(
            pl.concat_str(["Metadata_Source", "Metadata_Plate", "Metadata_Well"], separator="_").alias("full_key")
        )

        # Filter to only keys missing from profiles
        only_in_metadata_df = metadata_with_key.filter(pl.col("full_key").is_in(full_only_metadata))

        # Show number value_counts for only_in_metadata_df by Source
        print(only_in_metadata_df.group_by("Metadata_Source").agg(pl.len().alias("n_samples")).sort("n_samples"))
        # Per batch breakdown
        print(only_in_metadata_df.group_by(["Metadata_Batch"]).agg(pl.len().alias("n_samples")).sort(["n_samples"]))
        # Per plate breakdown
        print(only_in_metadata_df.group_by(["Metadata_Plate"]).agg(pl.len().alias("n_samples")).sort(["n_samples"]))
        # Per well breakdown
        print(only_in_metadata_df.group_by(["Metadata_Well"]).agg(pl.len().alias("n_samples")).sort(["n_samples"]))

        # Group by Source, Batch, Plate
        breakdown = (
            only_in_metadata_df
            .group_by(["Metadata_Source", "Metadata_Batch", "Metadata_Plate"])
            .agg(pl.len().alias("n_samples"))
            .sort(["Metadata_Source", "Metadata_Batch", "Metadata_Plate"])
        )

        print(f"\nBy Source/Batch/Plate ({breakdown.height} groups):")
        print(breakdown)

        # Summary by source
        by_source = (
            only_in_metadata_df
            .group_by("Metadata_Source")
            .agg(pl.len().alias("n_samples"))
            .sort("Metadata_Source")
        )
        print(f"\nSummary by Source:")
        print(by_source)

    # === SOURCE BREAKDOWN ===
    print("\n" + "=" * 60)
    print("SOURCE BREAKDOWN")
    print("=" * 60)

    metadata_sources = set(df_metadata.select("Metadata_Source").unique()["Metadata_Source"].to_list())
    profiles_sources = set(df_profiles.select("Metadata_Source").unique()["Metadata_Source"].to_list())

    print(f"\nSources in metadata: {sorted(metadata_sources)}")
    print(f"Sources in profiles: {sorted(profiles_sources)}")
    print(f"Sources in both:     {sorted(metadata_sources & profiles_sources)}")
    print(f"Sources only in metadata: {sorted(metadata_sources - profiles_sources)}")
    print(f"Sources only in profiles: {sorted(profiles_sources - metadata_sources)}")

    # === SAMPLE OF MISSING WELLS ===
    if wells_only_metadata:
        print("\n" + "=" * 60)
        print("SAMPLE: Wells only in metadata (first 10)")
        print("=" * 60)
        for i, w in enumerate(sorted(wells_only_metadata)[:10]):
            print(f"  {w}")

    if wells_only_profiles:
        print("\n" + "=" * 60)
        print("SAMPLE: Wells only in profiles (first 10)")
        print("=" * 60)
        for i, w in enumerate(sorted(wells_only_profiles)[:10]):
            print(f"  {w}")

    # === DETAILED ANALYSIS OF MISSING WELLS ===
    print("\n" + "=" * 60)
    print("DETAILED: Wells only in metadata - by Source and Plate")
    print("=" * 60)

    # Create full keys for filtering
    metadata_with_key = df_metadata.with_columns(
        pl.concat_str(["Metadata_Source", "Metadata_Plate", "Metadata_Well"], separator="_").alias("full_key")
    )
    profiles_keys = profiles_full  # reuse from above

    # Find wells only in metadata
    only_in_metadata = metadata_with_key.filter(~pl.col("full_key").is_in(profiles_keys))

    if only_in_metadata.height > 0:
        summary = (
            only_in_metadata
            .group_by(["Metadata_Source", "Metadata_Plate"])
            .agg(pl.count().alias("n_wells"))
            .sort(["Metadata_Source", "Metadata_Plate"])
        )
        print(f"\nSummary by Source and Plate ({summary.height} plate groups):")
        print(summary)

        print("\n\nSample rows (first 30):")
        print(only_in_metadata.select(["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"]).head(30))
    else:
        print("\nNo wells found only in metadata.")

    # === EXPORT FILTERED METADATA (wells in both) ===
    print("\n" + "=" * 60)
    print("EXPORTING FILTERED METADATA (wells in both metadata and profiles)")
    print("=" * 60)

    # Reload original metadata files with all columns
    df_filtered_full = pl.read_parquet(metadata_dir / "metadata_filtered.parquet")
    df_negative_full = pl.read_parquet(metadata_dir / "metadata_negative_controls.parquet")

    # Add full_key to each and filter to intersection
    df_filtered_matched = df_filtered_full.with_columns(
        pl.concat_str(["Metadata_Source", "Metadata_Plate", "Metadata_Well"], separator="_").alias("full_key")
    ).filter(pl.col("full_key").is_in(full_intersection)).drop("full_key")

    df_negative_matched = df_negative_full.with_columns(
        pl.concat_str(["Metadata_Source", "Metadata_Plate", "Metadata_Well"], separator="_").alias("full_key")
    ).filter(pl.col("full_key").is_in(full_intersection)).drop("full_key")

    # Combine into single dataframe using common columns
    df_matched = pl.concat([
        df_filtered_matched.select(common_cols),
        df_negative_matched.select(common_cols)
    ])

    # drop any duplicate rows
    df_matched = df_matched.unique()
    
    print(f"\nOriginal metadata_filtered:          {df_filtered_full.shape[0]:,} rows")
    print(f"Original metadata_negative_controls: {df_negative_full.shape[0]:,} rows")
    print(f"Combined matched metadata:           {df_matched.shape[0]:,} rows")

    # Assert well count matches intersection
    matched_wells = set(
        df_matched.select(
            pl.concat_str(["Metadata_Source", "Metadata_Plate", "Metadata_Well"], separator="_").alias("full_key")
        ).unique()["full_key"].to_list()
    )
    assert len(matched_wells) == len(full_intersection), \
        f"Well count mismatch: {len(matched_wells)} vs {len(full_intersection)}"
    print(f"Well count verified: {len(matched_wells):,} unique wells (matches intersection)")

    # Load profiles with additional columns for join
    df_profiles_for_join = pl.scan_parquet(profiles_path).select([
        "Metadata_Source", "Metadata_Plate", "Metadata_Well",
        "Metadata_JCP2022", "Metadata_broad_sample", "Metadata_Symbol", "Metadata_pert_type", "Metadata_Perturbation_Type"
    ]).unique().collect()

    # Cast join columns to string to avoid type mismatch (cat vs str)
    join_cols = ["Metadata_Source", "Metadata_Plate", "Metadata_Well"]
    df_profiles_for_join = df_profiles_for_join.with_columns([
        pl.col(c).cast(pl.Utf8) for c in join_cols
    ])

    # Change "Metadata_Perturbation_Type" for non orf and crispr to compound
    # Cast to string first, then fill nulls with "compound"
    df_profiles_for_join = df_profiles_for_join.with_columns(
        pl.col("Metadata_Perturbation_Type").cast(pl.Utf8).fill_null("compound").alias("Metadata_Perturbation_Type")
    )
    
    # Replace all values that aren't orf or crispr with compound
    df_profiles_for_join = df_profiles_for_join.with_columns(
        pl.when(pl.col("Metadata_Perturbation_Type").is_in(["orf", "crispr"]))
        .then(pl.col("Metadata_Perturbation_Type"))
        .otherwise(pl.lit("compound"))
        .alias("Metadata_Perturbation_Type")
    )
    
    # Left join to add JCP columns to matched metadata
    df_matched_with_jcp = df_matched.join(
        df_profiles_for_join,
        on=join_cols,
        how="left"
    )

    print(f"Matched metadata with JCP columns: {df_matched_with_jcp.shape[0]:,} rows, {df_matched_with_jcp.shape[1]} columns")
    print(f"Columns: {df_matched_with_jcp.columns}")

    print("\nSample rows:")
    print(df_matched_with_jcp.head(10))
    
    # Save combined file
    output_path = metadata_dir / "metadata_dataset.parquet"
    df_matched_with_jcp.write_parquet(output_path)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
