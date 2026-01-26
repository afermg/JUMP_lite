"""
Move image files based on metadata parquet file.

Moves all images matching wells in the metadata file from input folder to output folder.
"""

import argparse
import shutil
from pathlib import Path

import polars as pl
from tqdm import tqdm


def get_file_patterns(metadata: pl.DataFrame) -> list[str]:
    """
    Generate file prefix patterns for each well in metadata.

    Files are named: {source}__{batch}__{plate}__{well}__{channel}__{site}__{correction}.tif
    We match on: {source}__{batch}__{plate}__{well}__*
    """
    prefixes = []
    for row in metadata.iter_rows(named=True):
        prefix = f"{row['Metadata_Source']}__{row['Metadata_Batch']}__{row['Metadata_Plate']}__{row['Metadata_Well']}__"
        prefixes.append(prefix)
    return prefixes


def main():
    parser = argparse.ArgumentParser(description="Move images based on metadata file")
    parser.add_argument("--metadata", type=str, required=True,
                        help="Path to metadata parquet file")
    parser.add_argument("--input", type=str, required=True,
                        help="Input folder containing images")
    parser.add_argument("--output", type=str, required=True,
                        help="Output folder to move images to")
    parser.add_argument("--copy", action="store_true",
                        help="Copy instead of move")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be moved without actually moving")
    parser.add_argument("--sites", type=str, nargs="+", default=["1", "2", "3", "4"],
                        help="Only move files for specific sites (default: 1 2 3 4)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    metadata_path = Path(args.metadata)

    # Validate paths
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        return 1

    if not input_path.exists():
        print(f"Error: Input folder not found: {input_path}")
        return 1

    # Create output directory
    if not args.dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    # Load metadata
    print(f"Loading metadata from: {metadata_path}")
    metadata = pl.read_parquet(metadata_path)
    print(f"  {len(metadata):,} wells in metadata")

    # Generate file prefixes to match
    print("Generating file patterns...")
    prefixes = set(get_file_patterns(metadata))
    print(f"  {len(prefixes):,} unique well prefixes")

    # Find matching files
    print(f"Scanning input folder: {input_path}")
    all_files = list(input_path.glob("*.tif"))
    print(f"  {len(all_files):,} .tif files found")

    # Match files to metadata
    sites_filter = set(args.sites) if args.sites else None
    if sites_filter:
        print(f"  Filtering for sites: {sorted(sites_filter)}")

    files_to_move = []
    for f in all_files:
        # File format: {source}__{batch}__{plate}__{well}__{channel}__{site}__{correction}.tif
        parts = f.name.split("__")
        if len(parts) >= 6:
            prefix = "__".join(parts[:4]) + "__"
            site = parts[5]  # Site is the 6th element (index 5)
            if prefix in prefixes:
                if sites_filter is None or site in sites_filter:
                    files_to_move.append(f)

    print(f"  {len(files_to_move):,} files match metadata")

    if not files_to_move:
        print("No files to move.")
        return 0

    # Move or copy files
    action = "Copying" if args.copy else "Moving"
    if args.dry_run:
        print(f"\nDry run - would {action.lower()} {len(files_to_move):,} files:")
        for f in files_to_move[:10]:
            print(f"  {f.name}")
        if len(files_to_move) > 10:
            print(f"  ... and {len(files_to_move) - 10} more")
    else:
        print(f"\n{action} {len(files_to_move):,} files...")
        for f in tqdm(files_to_move):
            dest = output_path / f.name
            if args.copy:
                shutil.copy2(f, dest)
            else:
                shutil.move(f, dest)

        print(f"\nDone! {len(files_to_move):,} files moved to: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
