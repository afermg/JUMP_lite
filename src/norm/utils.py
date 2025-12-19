"""
Utility functions for normalization pipeline.

Contains helper functions for metadata loading and data preparation.
"""

import logging
from pathlib import Path

import polars as pl


def load_metadata(metadata_dir: Path, plate_col: str = "Metadata_Plate", well_col: str = "Metadata_Well") -> pl.DataFrame:
    """
    Load and merge well metadata for Target-2 plates.

    Args:
        metadata_dir: Directory containing metadata TSV files
        plate_col: Column name for plate identifier
        well_col: Column name for well identifier

    Returns:
        Polars DataFrame with merged metadata
    """
    try:
        import pandas as pd

        compound_meta = pd.read_csv(
            metadata_dir / "JUMP-Target-2_compound_metadata.tsv", sep="\t"
        ).drop_duplicates()
        plate_map = pd.read_csv(
            metadata_dir / "JUMP-Target-2_compound_platemap.tsv", sep="\t"
        )

        plate_map_annotated = plate_map.merge(compound_meta, on="broad_sample", how="left")

        # Rename columns to have Metadata_ prefix
        rename_cols = {
            col: f"Metadata_{col}"
            for col in plate_map_annotated.columns
            if not col.startswith("Metadata_") and col not in ["well_position", "broad_sample"]
        }

        rename_cols["well_position"] = "Metadata_Well"
        rename_cols["broad_sample"] = "Metadata_broad_sample"

        plate_map_annotated = plate_map_annotated.rename(columns=rename_cols)

        logging.info(f"Loaded metadata for {len(plate_map_annotated)} wells")
        return pl.from_pandas(plate_map_annotated)

    except Exception as e:
        logging.error(f"Failed to load metadata: {e}")
        raise
