"""
Reformat Raw CellProfiler Profiles to Match Target Format

Takes raw JUMP CellProfiler profiles, filters to wells of interest from metadata,
and reformats to match the standard output structure.

Output format:
- Metadata_id (first column): {source}__{batch}__{plate}__{well}
- Feature columns (middle)
- Metadata columns at end: Metadata_Source, Metadata_Batch, Metadata_Plate,
  Metadata_Well, Metadata_model, Metadata_dataset, Metadata_compression

Usage:
    python src/reformat_raw_cp_profiles.py \
        --source data/raw/raw_jump_CP_profiles/profiles.parquet \
        --metadata metadata/metadata_dataset_filtered_4reps.parquet \
        --output data/features/cellprofiler_raw_jump_core_raw_features.parquet
"""

import argparse
from pathlib import Path

import polars as pl


def reformat_profiles(
    source_path: Path,
    metadata_path: Path,
    output_path: Path,
    model: str = "cellprofiler_raw",
    dataset: str = "jump_core",
    compression: str = "none",
) -> None:
    """
    Load source profiles, filter to wells of interest, and reformat.

    Args:
        source_path: Path to source profiles parquet
        metadata_path: Path to metadata parquet with wells to include
        output_path: Path for output parquet file
        model: Value for Metadata_model column
        dataset: Value for Metadata_dataset column
        compression: Value for Metadata_compression column
    """
    join_cols = ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"]

    # Load metadata (wells of interest + metadata columns to keep)
    print(f"Loading metadata: {metadata_path}")
    meta_df = pl.read_parquet(metadata_path)
    # Cast join columns to string for consistent types
    meta_df = meta_df.with_columns([pl.col(c).cast(pl.Utf8) for c in join_cols])
    print(f"  Metadata shape: {meta_df.shape}")
    print(f"  Metadata columns: {meta_df.columns}")

    # Rename Metadata_pert_type to Metadata_control_type for pipeline compatibility
    if "Metadata_pert_type" in meta_df.columns:
        meta_df = meta_df.rename({"Metadata_pert_type": "Metadata_control_type"})
        print("  Renamed Metadata_pert_type -> Metadata_control_type")

    # Load source profiles - only feature columns + join columns
    print(f"Loading source profiles: {source_path}")
    source_df = pl.read_parquet(source_path)
    print(f"  Source shape: {source_df.shape}")

    if join_cols not in source_df.columns: 
        if "site" in source_df.columns: 
            source_df = source_df.with_columns([
                pl.col("site").str.split("__").list.get(0).alias("Metadata_Source"),
                pl.col("site").str.split("__").list.get(1).alias("Metadata_Batch"),
                pl.col("site").str.split("__").list.get(2).alias("Metadata_Plate"),
                pl.col("site").str.split("__").list.get(3).alias("Metadata_Well"),
                pl.col("site").str.split("__").list.get(4).alias("Metadata_Site"),
            ])
            # Drop model and filename columns if they exist
            if "model" in source_df.columns:
                source_df = source_df.drop("model")
            if "filename" in source_df.columns:
                source_df = source_df.drop("filename")
            
            # # rename site model to Metadata_model also rename filename to Metadata_filename
            # source_df = source_df.rename({"model": "Metadata_model", "filename": "Metadata_filename"})
            
            # Rename site to Metadata_id
            source_df = source_df.rename({"site": "Metadata_id"})
            
    
    # Keep only feature columns (non-metadata) + join columns from source
    feature_cols = [c for c in source_df.columns if not c.startswith("Metadata_")]
    source_df = source_df.select(join_cols + feature_cols)
    # Cast join columns to string
    source_df = source_df.with_columns([pl.col(c).cast(pl.Utf8) for c in join_cols])
    print(f"  Keeping {len(feature_cols)} feature columns from source")

    # If Metadata_Site in source_df repeat each row of meta_df for each site
    if "Metadata_Site" in source_df.columns:
        print("Expanding metadata for multiple sites...")
        meta_expanded = []
        unique_sites = source_df.select("Metadata_Site").unique().to_series().to_list()
        for site in unique_sites:
            site_meta = meta_df.with_columns(pl.lit(site).alias("Metadata_Site"))
            meta_expanded.append(site_meta)
        meta_df = pl.concat(meta_expanded)
        print(f"  Expanded metadata shape: {meta_df.shape}")
        # add Metadata_Site to join columns
        join_cols.append("Metadata_Site")
    
    # Join: source features + metadata columns
    print("Joining source features with metadata...")
    filtered_df = source_df.join(meta_df, on=join_cols, how="inner")
    print(f"  Joined shape: {filtered_df.shape}")

    if len(filtered_df) == 0:
        raise ValueError("No matching wells found after joining!")

    # Create Metadata_id column (Source__Batch__Plate__Well)
    if  "Metadata_id" in filtered_df.columns:
        filtered_df = filtered_df.drop("Metadata_id")
    else:
        filtered_df = filtered_df.with_columns(
            (
                pl.col("Metadata_Source") + "__" +
                pl.col("Metadata_Batch") + "__" +
                pl.col("Metadata_Plate") + "__" +
                pl.col("Metadata_Well")
            ).alias("Metadata_id")
        )

    # Add model/dataset/compression metadata
    filtered_df = filtered_df.with_columns([
        pl.lit(model).alias("Metadata_model"),
        pl.lit(dataset).alias("Metadata_dataset"),
        pl.lit(compression).alias("Metadata_compression"),
    ])

    # Define trailing metadata order (matching target format)
    trailing_metadata = [
        "Metadata_Source",
        "Metadata_Batch",
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_control_type",
        "Metadata_Perturbation_Type",
        "Metadata_JCP2022",
        "Metadata_broad_sample",
        "Metadata_Symbol",
        "Metadata_model",
        "Metadata_dataset",
        "Metadata_compression",
    ]

    # Final column order: Metadata_id, features, then metadata at end
    final_order = ["Metadata_id"] + feature_cols + trailing_metadata

    # Only select columns that exist
    final_order = [c for c in final_order if c in filtered_df.columns]
    filtered_df = filtered_df.select(final_order)

    # Save output
    print(f"Saving output: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.write_parquet(output_path)

    # Summary
    n_meta = len([c for c in filtered_df.columns if c.startswith("Metadata_")])
    n_feat = len(filtered_df.columns) - n_meta
    print(f"\nDone!")
    print(f"  Output rows: {len(filtered_df)}")
    print(f"  Metadata columns: {n_meta}")
    print(f"  Feature columns: {n_feat}")


def main():
    parser = argparse.ArgumentParser(
        description="Reformat raw CellProfiler profiles to match target format",
    )
    parser.add_argument("--source", type=Path, required=True, help="Source profiles parquet")
    parser.add_argument("--metadata", type=Path, required=True, help="Metadata parquet with wells of interest")
    parser.add_argument("--output", type=Path, required=True, help="Output parquet file")
    parser.add_argument("--model", type=str, default="cellprofiler_raw", help="Metadata_model value")
    parser.add_argument("--dataset", type=str, default="jump_core_annotated", help="Metadata_dataset value")
    parser.add_argument("--compression", type=str, default="none", help="Metadata_compression value")

    args = parser.parse_args()

    if not args.source.exists():
        print(f"Error: Source file not found: {args.source}")
        return 1
    if not args.metadata.exists():
        print(f"Error: Metadata file not found: {args.metadata}")
        return 1

    reformat_profiles(
        args.source,
        args.metadata,
        args.output,
        model=args.model,
        dataset=args.dataset,
        compression=args.compression,
    )
    return 0


if __name__ == "__main__":
    exit(main())
