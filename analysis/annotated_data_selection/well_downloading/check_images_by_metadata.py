"""
Check image file completeness based on metadata parquet files.

Checks how many expected images (based on metadata) are present in a folder.
"""

import argparse
from pathlib import Path

import polars as pl


CHANNELS = ["DNA", "AGP", "Mito", "RNA", "ER"]


def get_expected_files(metadata: pl.DataFrame, sites: list[str], channels: list[str]) -> set[str]:
    """
    Generate expected file names for each well in metadata.

    Files are named: {source}__{batch}__{plate}__{well}__{channel}__{site}__Orig.tif
    """
    expected = set()
    for row in metadata.iter_rows(named=True):
        for channel in channels:
            for site in sites:
                filename = f"{row['Metadata_Source']}__{row['Metadata_Batch']}__{row['Metadata_Plate']}__{row['Metadata_Well']}__{channel}__{site}__Orig.tif"
                expected.add(filename)
    return expected


def print_section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Check image completeness based on metadata")
    parser.add_argument("--metadata", type=str, nargs="+", required=True,
                        help="Path(s) to metadata parquet file(s)")
    parser.add_argument("--folder", type=str, required=True,
                        help="Folder to check for images")
    parser.add_argument("--sites", type=str, nargs="+", default=["1", "2", "3", "4"],
                        help="Sites to check (default: 1 2 3 4)")
    parser.add_argument("--channels", type=str, nargs="+", default=CHANNELS,
                        help=f"Channels to check (default: {' '.join(CHANNELS)})")
    parser.add_argument("--show-missing", action="store_true",
                        help="Show list of missing files")
    parser.add_argument("--limit-missing", type=int, default=20,
                        help="Limit number of missing files to show (default: 20)")
    parser.add_argument("--save-missing", type=str, default=None,
                        help="Save wells with missing files to parquet (e.g., --save-missing missing_wells.parquet)")
    parser.add_argument("--remove-partial", action="store_true",
                        help="Remove all files for wells that have missing channels (for re-download)")
    args = parser.parse_args()

    folder_path = Path(args.folder)

    if not folder_path.exists():
        print(f"Error: Folder not found: {folder_path}")
        return 1

    # Load and combine metadata files
    print_section("LOADING METADATA")

    # Columns needed for file name generation and identification
    required_cols = ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well", "Metadata_JCP2022"]

    all_metadata = []
    for meta_path in args.metadata:
        meta_path = Path(meta_path)
        if not meta_path.exists():
            print(f"Error: Metadata file not found: {meta_path}")
            return 1
        df = pl.read_parquet(meta_path).select(required_cols)
        print(f"  {meta_path.name}: {len(df):,} wells")
        all_metadata.append(df)

    metadata = pl.concat(all_metadata).unique()
    total_wells = len(metadata)
    print(f"\n  Total unique wells: {total_wells:,}")

    # Generate expected files
    print_section("CHECKING FILES")
    print(f"Folder: {folder_path}")
    print(f"Sites: {args.sites}")
    print(f"Channels: {args.channels}")

    expected_files = get_expected_files(metadata, args.sites, args.channels)
    print(f"\nExpected files: {len(expected_files):,}")
    print(f"  ({total_wells:,} wells x {len(args.channels)} channels x {len(args.sites)} sites)")

    # Check which files exist
    existing_files = set(f.name for f in folder_path.glob("*.tif"))
    print(f"Total .tif files in folder: {len(existing_files):,}")

    # Calculate overlap
    found = expected_files & existing_files
    missing = expected_files - existing_files
    extra = existing_files - expected_files

    print_section("RESULTS")
    print(f"{'Expected files:':<30} {len(expected_files):>10,}")
    print(f"{'Found:':<30} {len(found):>10,} ({len(found)/len(expected_files)*100:.1f}%)")
    print(f"{'Missing:':<30} {len(missing):>10,} ({len(missing)/len(expected_files)*100:.1f}%)")
    print(f"{'Extra (not in metadata):':<30} {len(extra):>10,}")

    # Breakdown by metadata file
    if len(args.metadata) > 1:
        print_section("BREAKDOWN BY METADATA FILE")
        for meta_path in args.metadata:
            meta_path = Path(meta_path)
            df = pl.read_parquet(meta_path).select(required_cols)
            expected = get_expected_files(df, args.sites, args.channels)
            found_for_file = expected & existing_files
            missing_for_file = expected - existing_files
            pct = len(found_for_file) / len(expected) * 100 if expected else 0
            print(f"\n  {meta_path.name}:")
            print(f"    Expected: {len(expected):,}")
            print(f"    Found:    {len(found_for_file):,} ({pct:.1f}%)")
            print(f"    Missing:  {len(missing_for_file):,}")

    # Show missing files if requested
    if args.show_missing and missing:
        print_section("MISSING FILES")
        missing_sorted = sorted(missing)
        for f in missing_sorted[:args.limit_missing]:
            print(f"  {f}")
        if len(missing) > args.limit_missing:
            print(f"  ... and {len(missing) - args.limit_missing:,} more")

    # Summary by well
    print_section("WELL COMPLETENESS")
    wells_complete = 0
    wells_partial = 0
    wells_missing = 0
    expected_per_well = len(args.channels) * len(args.sites)

    # Track wells with missing channels
    wells_with_missing = []
    wells_completely_missing = []

    for row in metadata.iter_rows(named=True):
        well_files = 0
        missing_channels = []
        for channel in args.channels:
            for site in args.sites:
                filename = f"{row['Metadata_Source']}__{row['Metadata_Batch']}__{row['Metadata_Plate']}__{row['Metadata_Well']}__{channel}__{site}__Orig.tif"
                if filename in existing_files:
                    well_files += 1
                else:
                    missing_channels.append(f"{channel}__{site}")

        if well_files == expected_per_well:
            wells_complete += 1
        elif well_files == 0:
            wells_missing += 1
            wells_completely_missing.append({
                "Metadata_Source": row["Metadata_Source"],
                "Metadata_Batch": row["Metadata_Batch"],
                "Metadata_Plate": row["Metadata_Plate"],
                "Metadata_Well": row["Metadata_Well"],
                "Metadata_JCP2022": row["Metadata_JCP2022"],
            })
        else:
            wells_partial += 1
            # Track existing files for this partial well (for potential removal)
            existing_for_well = []
            for channel in args.channels:
                for site in args.sites:
                    filename = f"{row['Metadata_Source']}__{row['Metadata_Batch']}__{row['Metadata_Plate']}__{row['Metadata_Well']}__{channel}__{site}__Orig.tif"
                    if filename in existing_files:
                        existing_for_well.append(filename)

            wells_with_missing.append({
                "source": row["Metadata_Source"],
                "batch": row["Metadata_Batch"],
                "plate": row["Metadata_Plate"],
                "well": row["Metadata_Well"],
                "jcp": row["Metadata_JCP2022"],
                "missing": missing_channels,
                "found": well_files,
                "expected": expected_per_well,
                "existing_files": existing_for_well
            })

    print(f"{'Complete wells:':<30} {wells_complete:>10,} ({wells_complete/total_wells*100:.1f}%)")
    print(f"{'Partial wells:':<30} {wells_partial:>10,} ({wells_partial/total_wells*100:.1f}%)")
    print(f"{'Missing wells:':<30} {wells_missing:>10,} ({wells_missing/total_wells*100:.1f}%)")

    # Show wells with missing channels
    if wells_with_missing:
        print_section("WELLS WITH MISSING CHANNELS")
        print(f"Total wells with partial data: {len(wells_with_missing)}")
        print()
        for w in wells_with_missing[:args.limit_missing]:
            print(f"  {w['source']}__{w['plate']}__{w['well']} ({w['jcp']})")
            print(f"    Found {w['found']}/{w['expected']} files")
            print(f"    Missing: {', '.join(w['missing'][:5])}" + (f" (+{len(w['missing'])-5} more)" if len(w['missing']) > 5 else ""))
        if len(wells_with_missing) > args.limit_missing:
            print(f"\n  ... and {len(wells_with_missing) - args.limit_missing} more wells with missing channels")

    # Remove files for partial wells if requested
    if args.remove_partial and wells_with_missing:
        print_section("REMOVING FILES FOR PARTIAL WELLS")
        files_to_remove = []
        for w in wells_with_missing:
            files_to_remove.extend(w["existing_files"])

        print(f"Wells with partial data: {len(wells_with_missing):,}")
        print(f"Files to remove: {len(files_to_remove):,}")

        confirm = input("\nAre you sure you want to remove these files? (yes/no): ")
        if confirm.lower() == "yes":
            removed = 0
            for filename in files_to_remove:
                filepath = folder_path / filename
                if filepath.exists():
                    filepath.unlink()
                    removed += 1
            print(f"Removed {removed:,} files")
        else:
            print("Aborted. No files removed.")

    # Save missing wells to parquet
    if args.save_missing:
        print_section("SAVING MISSING WELLS")

        # Combine partial and completely missing wells
        all_missing_wells = []

        # Add completely missing wells
        all_missing_wells.extend(wells_completely_missing)

        # Add partial wells (convert to same format)
        for w in wells_with_missing:
            all_missing_wells.append({
                "Metadata_Source": w["source"],
                "Metadata_Batch": w["batch"],
                "Metadata_Plate": w["plate"],
                "Metadata_Well": w["well"],
                "Metadata_JCP2022": w["jcp"],
            })

        if all_missing_wells:
            missing_df = pl.DataFrame(all_missing_wells)
            output_path = Path(args.save_missing)
            missing_df.write_parquet(output_path)
            print(f"Saved {len(all_missing_wells):,} wells with missing files to: {output_path}")
            print(f"  Completely missing: {len(wells_completely_missing):,}")
            print(f"  Partially missing:  {len(wells_with_missing):,}")
        else:
            print("No missing wells to save.")

    return 0


if __name__ == "__main__":
    exit(main())
