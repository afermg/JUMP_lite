#!/usr/bin/env python3
"""
Map CellProfiler features between two different naming conventions:
1. Traditional CellProfiler format: Compartment_Category_FeatureName_Parameters
2. cp_measure format: compartment_channel/aggregation/featuretype_FeatureName

This script creates a 1:1 mapping between features and reports overlaps and missing features.
"""

import polars as pl
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set
import re
import json


class FeatureMapper:
    """Map features between traditional CellProfiler and cp_measure formats."""

    def __init__(self, file1_path: str, file2_path: str):
        """
        Initialize the mapper with paths to both parquet files.

        Args:
            file1_path: Path to traditional CellProfiler features (reformatted_filtered)
            file2_path: Path to cp_measure features (zstd_raw_features)
        """
        self.file1_path = file1_path
        self.file2_path = file2_path

        # Load the data
        print("Loading parquet files...")
        self.df1 = pl.read_parquet(file1_path)
        self.df2 = pl.read_parquet(file2_path)

        # Extract feature columns
        self.features1 = [c for c in self.df1.columns if not c.startswith('Metadata_')]
        self.features2 = [c for c in self.df2.columns if not c.startswith('Metadata_')]
        self.features1_set = set(self.features1)  # For faster lookup

        print(f"File 1 features: {len(self.features1)}")
        print(f"File 2 features: {len(self.features2)}")

        # Mapping dictionaries
        self.compartment_map = {
            'cell': 'Cells',
            'nuclei': 'Nuclei',
            'cytoplasm': 'Cytoplasm'
        }

        # Channel number to name mapping (from src/download_images.py: ["DNA", "AGP", "Mito", "RNA", "ER"])
        self.channel_map = {
            '0': 'DNA',
            '1': 'AGP',
            '2': 'Mito',
            '3': 'RNA',
            '4': 'ER'
        }

        # Define the feature type to category mapping
        self.feature_type_to_category = {
            'sizeshape': 'AreaShape',
            'feret': 'AreaShape',
            'zernike': 'AreaShape',
            'intensity': 'Intensity',  # Note: intensityLocation maps to 'Location' - handled specially
            'radial_distribution': 'RadialDistribution',
            'radial_zernikes': 'RadialDistribution',
            'texture': 'Texture',
            'correlation': 'Correlation',
            'granularity': 'Granularity',
            'neighbors': 'Neighbors'
        }

        # Feature name mappings (cp_measure -> CellProfiler)
        self.feature_name_mappings = {
            # AreaShape/SizeShape mappings
            'sizeshapeArea': 'Area',
            'sizeshapeBoundingBoxArea': 'BoundingBoxArea',
            'sizeshapeBoundingBoxMaximum_X': 'BoundingBoxMaximum_X',
            'sizeshapeBoundingBoxMaximum_Y': 'BoundingBoxMaximum_Y',
            'sizeshapeBoundingBoxMinimum_X': 'BoundingBoxMinimum_X',
            'sizeshapeBoundingBoxMinimum_Y': 'BoundingBoxMinimum_Y',
            'sizeshapeCenter_X': 'Center_X',
            'sizeshapeCenter_Y': 'Center_Y',
            'sizeshapeCompactness': 'Compactness',
            'sizeshapeEccentricity': 'Eccentricity',
            'sizeshapeEquivalentDiameter': 'EquivalentDiameter',
            'sizeshapeEulerNumber': 'EulerNumber',
            'sizeshapeExtent': 'Extent',
            'sizeshapeFormFactor': 'FormFactor',
            'sizeshapeMajorAxisLength': 'MajorAxisLength',
            'sizeshapeMaximumRadius': 'MaximumRadius',
            'sizeshapeMeanRadius': 'MeanRadius',
            'sizeshapeMedianRadius': 'MedianRadius',
            'sizeshapeMinorAxisLength': 'MinorAxisLength',
            'sizeshapeOrientation': 'Orientation',
            'sizeshapePerimeter': 'Perimeter',
            'sizeshapeSolidity': 'Solidity',
            'sizeshapeConvexArea': 'ConvexArea',
            'sizeshapeFilledArea': 'FilledArea',
            'sizeshapePerimeterCrofton': 'PerimeterCrofton',

            # Feret diameter mappings
            'ferretMaxFeretDiameter': 'MaxFeretDiameter',
            'ferretMinFeretDiameter': 'MinFeretDiameter',

            # Zernike mappings - these map to AreaShape_Zernike_*
            'zernikeZernike_0_0': 'Zernike_0_0',
            'zernikeZernike_1_1': 'Zernike_1_1',
            'zernikeZernike_2_0': 'Zernike_2_0',
            'zernikeZernike_2_2': 'Zernike_2_2',
            'zernikeZernike_3_1': 'Zernike_3_1',
            'zernikeZernike_3_3': 'Zernike_3_3',
            'zernikeZernike_4_0': 'Zernike_4_0',
            'zernikeZernike_4_2': 'Zernike_4_2',
            'zernikeZernike_4_4': 'Zernike_4_4',
            'zernikeZernike_5_1': 'Zernike_5_1',
            'zernikeZernike_5_3': 'Zernike_5_3',
            'zernikeZernike_5_5': 'Zernike_5_5',
            'zernikeZernike_6_0': 'Zernike_6_0',
            'zernikeZernike_6_2': 'Zernike_6_2',
            'zernikeZernike_6_4': 'Zernike_6_4',
            'zernikeZernike_6_6': 'Zernike_6_6',
            'zernikeZernike_7_1': 'Zernike_7_1',
            'zernikeZernike_7_3': 'Zernike_7_3',
            'zernikeZernike_7_5': 'Zernike_7_5',
            'zernikeZernike_7_7': 'Zernike_7_7',
            'zernikeZernike_8_0': 'Zernike_8_0',
            'zernikeZernike_8_2': 'Zernike_8_2',
            'zernikeZernike_8_4': 'Zernike_8_4',
            'zernikeZernike_8_6': 'Zernike_8_6',
            'zernikeZernike_8_8': 'Zernike_8_8',
            'zernikeZernike_9_1': 'Zernike_9_1',
            'zernikeZernike_9_3': 'Zernike_9_3',
            'zernikeZernike_9_5': 'Zernike_9_5',
            'zernikeZernike_9_7': 'Zernike_9_7',
            'zernikeZernike_9_9': 'Zernike_9_9',
        }

    def parse_cp_measure_feature(self, feature: str) -> Tuple[str, str, str, str]:
        """
        Parse a cp_measure format feature name.

        Format: compartment_channel/aggregation/featuretype_FeatureName
        Example: cell_0/max/sizeshapeArea -> ('cell', '0', 'max', 'sizeshapeArea')

        Returns:
            Tuple of (compartment, channel, aggregation, full_feature_name)
        """
        parts = feature.split('/')
        if len(parts) != 3:
            return None, None, None, None

        # Parse compartment and channel
        comp_channel = parts[0].rsplit('_', 1)
        if len(comp_channel) != 2:
            return None, None, None, None

        compartment = comp_channel[0]
        channel = comp_channel[1]
        aggregation = parts[1]
        full_feature_name = parts[2]

        return compartment, channel, aggregation, full_feature_name

    def parse_cellprofiler_feature(self, feature: str) -> Tuple[str, str, str]:
        """
        Parse a traditional CellProfiler format feature name.

        Format: Compartment_Category_FeatureName_Parameters
        Example: Cells_AreaShape_Area -> ('Cells', 'AreaShape', 'Area')

        Returns:
            Tuple of (compartment, category, feature_name_with_params)
        """
        parts = feature.split('_')
        if len(parts) < 3:
            return None, None, None

        compartment = parts[0]
        category = parts[1]
        feature_name_with_params = '_'.join(parts[2:])

        return compartment, category, feature_name_with_params

    def map_cp_measure_to_cellprofiler(self, cp_measure_feature: str) -> List[str]:
        """
        Map a cp_measure feature to possible CellProfiler features.

        Args:
            cp_measure_feature: Feature in cp_measure format

        Returns:
            List of possible matching CellProfiler feature names
        """
        compartment, channel, aggregation, full_feature_name = self.parse_cp_measure_feature(cp_measure_feature)

        if compartment is None:
            return []

        # Map compartment
        cp_compartment = self.compartment_map.get(compartment)
        if cp_compartment is None:
            return []

        # Map channel if present
        cp_channel = self.channel_map.get(channel) if channel else None

        # Try to find the feature type and feature name
        # Check if we have a direct mapping
        if full_feature_name in self.feature_name_mappings:
            cp_feature_name = self.feature_name_mappings[full_feature_name]

            # Determine the category based on the feature type prefix
            feature_type = None
            for ft in self.feature_type_to_category.keys():
                if full_feature_name.startswith(ft):
                    feature_type = ft
                    break

            if feature_type:
                category = self.feature_type_to_category[feature_type]
                # AreaShape features don't have channel parameter
                if feature_type in ['sizeshape', 'feret', 'zernike']:
                    return [f"{cp_compartment}_{category}_{cp_feature_name}"]
                # Other features include channel
                elif cp_channel:
                    return [f"{cp_compartment}_{category}_{cp_feature_name}_{cp_channel}"]

        # Try pattern matching for features we don't have explicit mappings for
        possible_matches = []

        # Extract feature type prefix
        feature_type = None
        for ft in self.feature_type_to_category.keys():
            if full_feature_name.startswith(ft):
                feature_type = ft
                break

        if feature_type:
            category = self.feature_type_to_category[feature_type]
            # Remove the feature type prefix
            remaining_name = full_feature_name[len(feature_type):]

            # Special handling for intensity features - check if it's actually a Location feature
            if feature_type == 'intensity' and remaining_name.startswith('Location'):
                category = 'Location'
                # Remove the "Location_" prefix to get the actual feature name
                cp_feature_name = remaining_name[9:] if remaining_name.startswith('Location_') else remaining_name
            # For features with channel information
            elif feature_type in ['intensity', 'radial_distribution', 'radial_zernikes', 'texture']:
                # These often keep their structure after the prefix
                # Example: intensityIntensity_IntegratedIntensity -> Intensity_IntegratedIntensity

                # Check for pattern like "intensityIntensity_*" -> "Intensity_*"
                # After removing prefix "intensity", we get "Intensity_FeatureName"
                # This should map to "Compartment_Intensity_FeatureName_Channel"
                # So we need to remove the duplicate category name
                if remaining_name.startswith(category + '_'):
                    # Remove the duplicate category prefix
                    cp_feature_name = remaining_name[len(category) + 1:]
                elif remaining_name.startswith(category):
                    cp_feature_name = remaining_name[len(category):]
                    if cp_feature_name.startswith('_'):
                        cp_feature_name = cp_feature_name[1:]
                else:
                    cp_feature_name = remaining_name
            else:
                cp_feature_name = remaining_name

            # Build the pattern with channel (applies to all channel-based features)
            # Note: Location features are derived from intensity features
            if feature_type in ['intensity', 'radial_distribution', 'radial_zernikes', 'texture'] or category == 'Location':
                # For RadialDistribution and Texture, the channel appears BEFORE the final parameters
                # Format: Compartment_Category_FeatureName_Channel_Parameters
                if cp_channel:
                    # For these features, we need to insert channel before the last numeric parts
                    # Location and Intensity features have channel at the end
                    # RadialDistribution and Texture features have channel before parameters
                    if feature_type in ['radial_distribution', 'radial_zernikes', 'texture']:
                        # Split the feature name into parts
                        name_parts = cp_feature_name.split('_')
                        # Find where the numeric/parameter part starts
                        # Parameters typically start with numbers or specific patterns like "1of4"
                        split_idx = len(name_parts)
                        for i, part in enumerate(name_parts):
                            # Check if this part looks like a parameter (starts with digit or contains "of")
                            if part and (part[0].isdigit() or 'of' in part):
                                split_idx = i
                                break

                        # Insert channel before the parameters
                        if split_idx < len(name_parts):
                            feature_base = '_'.join(name_parts[:split_idx])
                            feature_params = '_'.join(name_parts[split_idx:])
                            pattern = f"{cp_compartment}_{category}_{feature_base}_{cp_channel}_{feature_params}"
                        else:
                            pattern = f"{cp_compartment}_{category}_{cp_feature_name}_{cp_channel}"
                    else:
                        # For Intensity and other features, channel comes at the end
                        pattern = f"{cp_compartment}_{category}_{cp_feature_name}_{cp_channel}"

                    # Exact match
                    if pattern in self.features1_set:
                        possible_matches.append(pattern)
                else:
                    # No channel provided, search for all channel variants
                    pattern = f"{cp_compartment}_{category}_{cp_feature_name}"
                    for f1 in self.features1:
                        if f1.startswith(pattern):
                            possible_matches.append(f1)
            else:
                # For features without channel (AreaShape, etc.)
                pattern = f"{cp_compartment}_{category}_{remaining_name}"
                if pattern in self.features1_set:
                    possible_matches.append(pattern)

        return possible_matches

    def create_mapping(self) -> Dict[str, str]:
        """
        Create a 1:1 mapping between cp_measure and CellProfiler features.

        Returns:
            Dictionary mapping cp_measure features to CellProfiler features
        """
        mapping = {}
        ambiguous = {}  # Features with multiple possible matches

        print("\nCreating feature mapping...")
        for f2 in self.features2:
            matches = self.map_cp_measure_to_cellprofiler(f2)

            if len(matches) == 1:
                mapping[f2] = matches[0]
            elif len(matches) > 1:
                ambiguous[f2] = matches
                # For now, take the first match
                mapping[f2] = matches[0]

        return mapping, ambiguous

    def analyze_coverage(self, mapping: Dict[str, str]) -> Dict:
        """
        Analyze the mapping coverage and identify missing features.

        Args:
            mapping: Feature mapping dictionary

        Returns:
            Dictionary with coverage statistics
        """
        mapped_f2 = set(mapping.keys())
        mapped_f1 = set(mapping.values())

        unmapped_f2 = set(self.features2) - mapped_f2
        unmapped_f1 = set(self.features1) - mapped_f1

        # Group unmapped features by category
        unmapped_f1_by_category = {}
        for f1 in unmapped_f1:
            comp, cat, feat = self.parse_cellprofiler_feature(f1)
            if cat not in unmapped_f1_by_category:
                unmapped_f1_by_category[cat] = []
            unmapped_f1_by_category[cat].append(f1)

        unmapped_f2_by_type = {}
        for f2 in unmapped_f2:
            comp, chan, agg, feat = self.parse_cp_measure_feature(f2)
            feature_type = None
            for ft in self.feature_type_to_category.keys():
                if feat and feat.startswith(ft):
                    feature_type = ft
                    break
            if feature_type:
                if feature_type not in unmapped_f2_by_type:
                    unmapped_f2_by_type[feature_type] = []
                unmapped_f2_by_type[feature_type].append(f2)

        return {
            'total_f1': len(self.features1),
            'total_f2': len(self.features2),
            'mapped_f1': len(mapped_f1),
            'mapped_f2': len(mapped_f2),
            'unmapped_f1': len(unmapped_f1),
            'unmapped_f2': len(unmapped_f2),
            'unmapped_f1_by_category': unmapped_f1_by_category,
            'unmapped_f2_by_type': unmapped_f2_by_type,
            'coverage_f1': len(mapped_f1) / len(self.features1) * 100,
            'coverage_f2': len(mapped_f2) / len(self.features2) * 100,
        }

    def generate_report(self, mapping: Dict[str, str], ambiguous: Dict[str, List[str]],
                       coverage: Dict, output_dir: str = '.'):
        """
        Generate comprehensive mapping report.

        Args:
            mapping: Feature mapping dictionary
            ambiguous: Ambiguous mappings
            coverage: Coverage statistics
            output_dir: Directory to save reports
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Save mapping to JSON
        mapping_file = output_dir / 'feature_mapping.json'
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"\nMapping saved to: {mapping_file}")

        # Save mapping to CSV for easier viewing
        mapping_csv = output_dir / 'feature_mapping.csv'
        df_mapping = pd.DataFrame([
            {'cp_measure_feature': k, 'cellprofiler_feature': v}
            for k, v in sorted(mapping.items())
        ])
        df_mapping.to_csv(mapping_csv, index=False)
        print(f"Mapping CSV saved to: {mapping_csv}")

        # Save ambiguous mappings
        if ambiguous:
            ambiguous_file = output_dir / 'ambiguous_mappings.json'
            with open(ambiguous_file, 'w') as f:
                json.dump(ambiguous, f, indent=2)
            print(f"Ambiguous mappings saved to: {ambiguous_file}")

        # Generate text report
        report_file = output_dir / 'mapping_report.txt'
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("CellProfiler Feature Mapping Report\n")
            f.write("="*80 + "\n\n")

            f.write("FILES:\n")
            f.write(f"  File 1 (CellProfiler): {self.file1_path}\n")
            f.write(f"  File 2 (cp_measure):   {self.file2_path}\n\n")

            f.write("SUMMARY:\n")
            f.write(f"  Total features in File 1: {coverage['total_f1']}\n")
            f.write(f"  Total features in File 2: {coverage['total_f2']}\n")
            f.write(f"  Mapped features from File 1: {coverage['mapped_f1']} ({coverage['coverage_f1']:.1f}%)\n")
            f.write(f"  Mapped features from File 2: {coverage['mapped_f2']} ({coverage['coverage_f2']:.1f}%)\n")
            f.write(f"  Unmapped features in File 1: {coverage['unmapped_f1']}\n")
            f.write(f"  Unmapped features in File 2: {coverage['unmapped_f2']}\n")
            if ambiguous:
                f.write(f"  Ambiguous mappings: {len(ambiguous)}\n")
            f.write("\n")

            # Unmapped File 1 features by category
            f.write("UNMAPPED FILE 1 FEATURES (by category):\n")
            for cat, features in sorted(coverage['unmapped_f1_by_category'].items()):
                f.write(f"\n  {cat} ({len(features)} features):\n")
                for feat in sorted(features)[:10]:  # Show first 10
                    f.write(f"    - {feat}\n")
                if len(features) > 10:
                    f.write(f"    ... and {len(features) - 10} more\n")
            f.write("\n")

            # Unmapped File 2 features by type
            f.write("UNMAPPED FILE 2 FEATURES (by type):\n")
            for ftype, features in sorted(coverage['unmapped_f2_by_type'].items()):
                f.write(f"\n  {ftype} ({len(features)} features):\n")
                for feat in sorted(features)[:10]:  # Show first 10
                    f.write(f"    - {feat}\n")
                if len(features) > 10:
                    f.write(f"    ... and {len(features) - 10} more\n")
            f.write("\n")

            # Sample mappings
            f.write("SAMPLE MAPPINGS (first 20):\n")
            for i, (f2, f1) in enumerate(sorted(mapping.items())[:20]):
                f.write(f"  {f2}\n")
                f.write(f"    -> {f1}\n")
            f.write("\n")

            if ambiguous:
                f.write("AMBIGUOUS MAPPINGS (first 10):\n")
                for i, (f2, matches) in enumerate(list(ambiguous.items())[:10]):
                    f.write(f"  {f2}\n")
                    for match in matches:
                        f.write(f"    -> {match}\n")

        print(f"Report saved to: {report_file}")

        # Print summary to console
        print("\n" + "="*80)
        print("MAPPING SUMMARY")
        print("="*80)
        print(f"Total features in File 1 (CellProfiler): {coverage['total_f1']}")
        print(f"Total features in File 2 (cp_measure):   {coverage['total_f2']}")
        print(f"\nMapped features from File 1: {coverage['mapped_f1']} ({coverage['coverage_f1']:.1f}%)")
        print(f"Mapped features from File 2: {coverage['mapped_f2']} ({coverage['coverage_f2']:.1f}%)")
        print(f"\nUnmapped features in File 1: {coverage['unmapped_f1']}")
        print(f"Unmapped features in File 2: {coverage['unmapped_f2']}")
        if ambiguous:
            print(f"\nAmbiguous mappings: {len(ambiguous)}")
        print("="*80)


def main():
    """Main execution function."""
    # File paths
    file1 = 'data/features/raw_jump_cp_profiles_reformatted_filtered.parquet'
    file2 = 'data/features/cp_measure_jump_target2_4plate_zstd_raw_features.parquet'

    # Create mapper
    mapper = FeatureMapper(file1, file2)

    # Create mapping
    mapping, ambiguous = mapper.create_mapping()

    # Analyze coverage
    coverage = mapper.analyze_coverage(mapping)

    # Generate report
    mapper.generate_report(mapping, ambiguous, coverage,
                          output_dir='data/results/feature_mapping')

    print("\nMapping complete!")


if __name__ == '__main__':
    main()
