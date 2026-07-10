#!/usr/bin/env python3
"""Evaluate Phenotypic Activity per compound.

Takes processed profile output and calculates Phenotypic Activity
(compound replicate retrieval) for each compound.

Supports:
1. Pre-annotated data with Metadata_JCP2022 and Metadata_negcon columns
2. Raw profiles that need metadata merged

Usage:
    # For pre-annotated data (like cellprofiler_raw_jump_core_annotated)
    python scripts/evaluate_phenotypic_activity.py \
        --input data/features/cellprofiler_raw_jump_core_annotated_raw_features/agg/OUTPUT.parquet

    # For raw profiles (will merge metadata)
    python scripts/evaluate_phenotypic_activity.py --input output/profiles.parquet

    # With custom annotations file
    python scripts/evaluate_phenotypic_activity.py --input output/profiles.parquet \
        --annotations metadata/dataset_overlaps/refchemdb_targets_by_jcp.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

from evaluate_cross_modality_retrieval import evaluate_cross_modality_retrieval


def load_profiles(path: Path) -> pl.DataFrame:
    """Load profiles from parquet file."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    return pl.read_parquet(path)


def infer_columns(
    df: pl.DataFrame,
    metadata_prefix: str = "Metadata_",
) -> tuple[list[str], list[str]]:
    """Infer feature and metadata columns based on prefix."""
    metadata_cols = [c for c in df.columns if c.startswith(metadata_prefix)]
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    return feature_cols, metadata_cols


def get_numeric_features(df: pl.DataFrame, features: list[str]) -> list[str]:
    """Filter to only numeric features."""
    numeric_types = (
        pl.Float32, pl.Float64,
        pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    )
    return [f for f in features if df[f].dtype in numeric_types]


def merge_metadata(
    df: pl.DataFrame,
    metadata_path: Path,
) -> pl.DataFrame:
    """Merge profiles with metadata to get JCP2022 IDs."""
    print(f"Loading metadata from: {metadata_path}")
    metadata = pl.read_parquet(metadata_path)

    # Find common join columns
    profile_cols = set(df.columns)
    metadata_cols = set(metadata.columns)

    # Prefer specific join columns
    join_candidates = ["Metadata_Plate", "Metadata_Well", "Metadata_Source", "Metadata_Batch"]
    join_cols = [c for c in join_candidates if c in profile_cols and c in metadata_cols]

    if not join_cols:
        raise ValueError("No common columns found for joining profiles with metadata")

    print(f"  Joining on: {join_cols}")

    # Ensure consistent types for join columns
    for col in join_cols:
        df = df.with_columns(pl.col(col).cast(pl.Utf8))
        metadata = metadata.with_columns(pl.col(col).cast(pl.Utf8))

    # Select only columns we need from metadata (avoid duplicates)
    metadata_select_cols = join_cols + [
        c for c in metadata.columns
        if c not in profile_cols and c not in join_cols
    ]
    metadata = metadata.select(metadata_select_cols)

    df = df.join(metadata, on=join_cols, how="left")
    print(f"  After merge: {df.shape}")

    return df


def merge_annotations(
    df: pl.DataFrame,
    annotations_path: Path,
) -> pl.DataFrame:
    """Merge profiles with annotations file on Metadata_JCP2022."""
    print(f"Loading annotations from: {annotations_path}")
    annotations = pl.read_parquet(annotations_path)

    if "Metadata_JCP2022" not in df.columns:
        raise ValueError("Metadata_JCP2022 not found in profiles. Run merge_metadata first.")

    if "Metadata_JCP2022" not in annotations.columns:
        raise ValueError("Metadata_JCP2022 not found in annotations file.")

    # Ensure consistent types
    df = df.with_columns(pl.col("Metadata_JCP2022").cast(pl.Utf8))
    annotations = annotations.with_columns(pl.col("Metadata_JCP2022").cast(pl.Utf8))

    # Select only annotation columns that aren't already in df
    existing_cols = set(df.columns)
    annotation_select_cols = ["Metadata_JCP2022"] + [
        c for c in annotations.columns
        if c not in existing_cols and c != "Metadata_JCP2022"
    ]
    annotations = annotations.select(annotation_select_cols)

    df = df.join(annotations, on="Metadata_JCP2022", how="left")
    print(f"  After annotation merge: {df.shape}")

    return df


def load_tier_annotations(
    annotations_path: Path,
) -> pl.DataFrame:
    """
    Load tier annotations from refchemdb_conf_jump_matched.parquet.

    Returns DataFrame with columns:
    - Metadata_JCP2022
    - target
    - WithinModalityTier (Tier1, Tier2, Tier3, Excluded)
    - modality_clean (Positive, Negative)
    """
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    ann = pl.read_parquet(annotations_path)

    # Ensure required columns exist
    required_cols = ["Metadata_JCP2022", "target", "WithinModalityTier", "modality_clean"]
    missing = [c for c in required_cols if c not in ann.columns]
    if missing:
        raise ValueError(f"Annotations file missing columns: {missing}")

    # Select and return relevant columns
    return ann.select(required_cols).unique()


def _filter_targets_by_compound_count(
    df_consensus,
    min_compounds_per_target: int = 3,
    max_targets_per_compound: int = 50,
    include_unknown_as_background: bool = False,
    compound_col: str = "Metadata_JCP2022",
    negcon_col: str = "Metadata_negcon",
):
    """
    Filter targets based on minimum number of compounds.

    Args:
        df_consensus: DataFrame with Metadata_target column
        min_compounds_per_target: Minimum compounds required per target
        max_targets_per_compound: Maximum targets per compound
        include_unknown_as_background: If True, keep compounds with "unknown" targets
            as background (negatives) but don't calculate consistency for "unknown"
        compound_col: Column containing compound identifier
        negcon_col: Column containing negative control flag

    Returns:
        Filtered df_consensus DataFrame
    """
    # Filter out negative controls
    if negcon_col in df_consensus.columns:
        df_consensus = df_consensus[df_consensus[negcon_col] == False].copy()

    # Filter out promiscuous compounds (but don't count "unknown" toward this limit)
    if max_targets_per_compound is not None:
        target_counts_per_compound = df_consensus["Metadata_target"].apply(
            lambda x: len([t for t in x if t != "unknown"]) if isinstance(x, list) else 0
        )
        n_promiscuous = (target_counts_per_compound > max_targets_per_compound).sum()
        if n_promiscuous > 0:
            print(f"    Filtering out {n_promiscuous} promiscuous compounds")
        df_consensus = df_consensus[
            target_counts_per_compound <= max_targets_per_compound
        ].copy()

    # Explode targets for counting (exclude "unknown" from counts)
    df_exploded = df_consensus.explode("Metadata_target")
    df_exploded_known = df_exploded[df_exploded["Metadata_target"] != "unknown"].copy()

    # Count unique compounds per target (only for known targets)
    target_counts = df_exploded_known.groupby("Metadata_target")[compound_col].nunique()
    target_counts = target_counts.sort_values(ascending=False)

    valid_targets = target_counts[
        target_counts >= min_compounds_per_target
    ].index.tolist()
    print(f"    Targets with >={min_compounds_per_target} compounds: {len(valid_targets)}")

    # Filter compounds: keep those with valid targets OR unknown-only if including as background
    def should_keep_compound(target_list):
        if not isinstance(target_list, list):
            return False
        has_valid = any(t in valid_targets for t in target_list)
        if has_valid:
            return True
        if include_unknown_as_background:
            # Keep if it has "unknown" (will serve as background)
            return "unknown" in target_list
        return False

    df_consensus = df_consensus[
        df_consensus["Metadata_target"].apply(should_keep_compound)
    ].copy()

    # Update target lists: keep only valid targets, but preserve "unknown" for background compounds
    # Compounds with only "unknown" keep ["unknown"] so copairs includes them as negatives
    def filter_targets(targets):
        if not isinstance(targets, list):
            return ["unknown"] if include_unknown_as_background else []
        filtered = [t for t in targets if t in valid_targets]
        # If no valid targets but we're including unknown as background, keep "unknown"
        if not filtered and include_unknown_as_background:
            return ["unknown"]
        return filtered

    df_consensus["Metadata_target"] = df_consensus["Metadata_target"].apply(filter_targets)

    # Count compounds with actual valid targets vs unknown-only (background)
    def has_valid_targets(targets):
        if not isinstance(targets, list) or len(targets) == 0:
            return False
        return targets != ["unknown"]

    n_with_valid_targets = df_consensus["Metadata_target"].apply(has_valid_targets).sum()
    n_background = len(df_consensus) - n_with_valid_targets

    print(f"    Compounds with valid targets: {n_with_valid_targets}")
    if include_unknown_as_background and n_background > 0:
        print(f"    Compounds as background (unknown only): {n_background}")

    return df_consensus


def calculate_phenotypic_consistency(
    df: pl.DataFrame,
    features: list[str],
    null_size: int = 10_000,
    p_threshold: float = 0.05,
    seed: int = 0,
    min_compounds_per_target: int = 3,
    max_targets_per_compound: int = 50,
    include_unknown_as_background: bool = False,
    compound_col: str = "Metadata_JCP2022",
    target_col: str = "Metadata_target_list",
    negcon_col: str = "Metadata_negcon",
) -> dict[str, Any]:
    """
    Calculate Phenotypic Consistency (target-level retrieval).

    Measures how well compounds targeting the same protein cluster together.

    Args:
        df: Profiles with metadata
        features: Feature column names
        null_size: Size of null distribution
        p_threshold: Significance threshold
        seed: Random seed
        min_compounds_per_target: Minimum compounds required per target
        max_targets_per_compound: Maximum targets per compound
        include_unknown_as_background: If True, keep compounds with "unknown" targets
            as background (negatives) but don't calculate consistency for "unknown"
        compound_col: Column containing compound identifier
        target_col: Column containing target identifier (pipe-separated for multiple)
        negcon_col: Column containing negative control flag

    Returns:
        Dictionary with:
        - target_consistency: Per-target consistency DataFrame
        - pct_targets_active: % of targets below corrected p-value
        - n_targets_active: Number of significant targets
        - n_targets_total: Total number of targets
    """
    from copairs import map as copairs_map

    df_pd = df.to_pandas()

    # Fill null target values with "unknown"
    if target_col in df_pd.columns:
        df_pd[target_col] = df_pd[target_col].fillna("unknown")

    # Get consensus profiles per compound
    df_consensus = (
        df_pd.groupby(
            [compound_col, target_col, negcon_col],
            as_index=False,
        )[features]
        .median()
        .copy()
    )
    df_consensus["Metadata_target"] = df_consensus[target_col].str.split("|")

    df_consensus = _filter_targets_by_compound_count(
        df_consensus,
        min_compounds_per_target=min_compounds_per_target,
        max_targets_per_compound=max_targets_per_compound,
        include_unknown_as_background=include_unknown_as_background,
        compound_col=compound_col,
        negcon_col=negcon_col,
    )

    if len(df_consensus) < 2:
        print(f"    Warning: Not enough compounds ({len(df_consensus)}) for PC")
        return {
            "target_consistency": None,
            "pct_targets_active": 0.0,
            "n_targets_active": 0,
            "n_targets_total": 0,
            "median_n_total_pairs": 0,
        }

    metadata = df_consensus.filter(regex="^Metadata")
    profiles = df_consensus[features].values

    pos_sameby_target = ["Metadata_target"]
    pos_diffby_target = []
    neg_sameby_target = []
    neg_diffby_target = ["Metadata_target"]

    try:
        target_ap = copairs_map.multilabel.average_precision(
            metadata,
            profiles,
            pos_sameby_target,
            pos_diffby_target,
            neg_sameby_target,
            neg_diffby_target,
            multilabel_col="Metadata_target",
        )

        # Filter out "unknown" target from AP results - they were only used as background
        # Unknown compounds are included in n_total_pairs as negatives for known targets
        n_unknown_rows = (target_ap["Metadata_target"] == "unknown").sum()
        if n_unknown_rows > 0:
            target_ap = target_ap[target_ap["Metadata_target"] != "unknown"].copy()

        if len(target_ap) == 0:
            print(f"    Warning: No known target compounds for PC calculation")
            return {
                "target_consistency": None,
                "pct_targets_active": 0.0,
                "n_targets_active": 0,
                "n_targets_total": 0,
                "median_n_total_pairs": 0,
            }

        target_map = copairs_map.mean_average_precision(
            target_ap,
            pos_sameby_target,
            null_size=null_size,
            threshold=p_threshold,
            seed=seed,
        ).copy()

        target_map["below_corrected_p"] = target_map["corrected_p_value"] < p_threshold

        n_targets_active = target_map["below_corrected_p"].sum()
        n_targets_total = len(target_map)
        pct_targets_active = (
            (n_targets_active / n_targets_total * 100) if n_targets_total > 0 else 0.0
        )

        # Calculate median n_total_pairs to show how many compounds were compared
        median_n_total_pairs = int(target_ap["n_total_pairs"].median()) if "n_total_pairs" in target_ap.columns else 0

        return {
            "target_consistency": target_map,
            "pct_targets_active": float(pct_targets_active),
            "n_targets_active": int(n_targets_active),
            "n_targets_total": int(n_targets_total),
            "median_n_total_pairs": median_n_total_pairs,
        }
    except Exception as e:
        print(f"    Warning: Target consistency failed: {e}")
        return {
            "target_consistency": None,
            "pct_targets_active": 0.0,
            "n_targets_active": 0,
            "n_targets_total": 0,
            "median_n_total_pairs": 0,
        }


def evaluate_phenotypic_consistency_by_subsets(
    df: pl.DataFrame,
    features: list[str],
    annotations: pl.DataFrame,
    output_dir: Path,
    activity_map: "pd.DataFrame | None" = None,
    null_size: int = 10_000,
    p_threshold: float = 0.05,
    seed: int = 0,
    min_compounds_per_target: int = 3,
    include_unknown_as_background: bool = False,
    compound_col: str = "Metadata_JCP2022",
    negcon_col: str = "Metadata_negcon",
    all_tier_combinations: bool = False,
) -> dict[str, Any]:
    """
    Evaluate Phenotypic Consistency for different data subsets.

    Subsets are based on (loop order):
    1. Active filter mode (all compounds vs active-only)
    2. Metadata_Group (if present in df)
    3. WithinModalityTier (Tier1, Tier2, Tier3, combinations)
    4. modality_clean (all, Positive, Negative)

    Args:
        df: Profiles with metadata
        features: Feature column names
        annotations: Tier annotations from refchemdb_conf_jump_matched.parquet
        output_dir: Directory to save results
        activity_map: Per-compound phenotypic activity results (from calculate_phenotypic_activity)
            with columns: Metadata_JCP2022, Metadata_Group, below_corrected_p
        null_size: Size of null distribution
        p_threshold: Significance threshold
        seed: Random seed
        min_compounds_per_target: Minimum compounds required per target
        include_unknown_as_background: If True, include compounds with unknown targets
            as background (negatives) for the consistency calculation
        compound_col: Column containing compound identifier
        negcon_col: Column containing negative control flag
        all_tier_combinations: If True, run PC on all tier combinations (Tier1, Tier2,
            Tier3, Tier1+Tier2, Tier1+Tier2+Tier3). If False (default), only run on
            the combined tiers (all tiers together).

    Returns:
        Dictionary with results for each subset
    """
    import pandas as pd

    results = {}
    all_pc_results = []

    # Get unique tiers (excluding "Excluded")
    base_tiers = [t for t in annotations["WithinModalityTier"].unique().to_list() if t != "Excluded"]
    base_tiers = sorted(base_tiers)  # Tier1, Tier2, Tier3

    # Create tier options
    tier_options = []
    tier_to_filter = {}  # Maps tier option name to list of tiers to include

    if all_tier_combinations:
        # Run on all tier combinations: Tier1, Tier2, Tier3, Tier1+Tier2, Tier1+Tier2+Tier3
        for i, tier in enumerate(base_tiers):
            tier_options.append(tier)
            tier_to_filter[tier] = [tier]

        # Add cumulative combinations
        if len(base_tiers) >= 2:
            combo_name = "+".join(base_tiers[:2])  # Tier1+Tier2
            tier_options.append(combo_name)
            tier_to_filter[combo_name] = base_tiers[:2]

        if len(base_tiers) >= 3:
            combo_name = "+".join(base_tiers[:3])  # Tier1+Tier2+Tier3
            tier_options.append(combo_name)
            tier_to_filter[combo_name] = base_tiers[:3]
    else:
        # Default: only run on combined tiers (all tiers together)
        if base_tiers:
            combo_name = "+".join(base_tiers)  # e.g., Tier1+Tier2+Tier3
            tier_options = [combo_name]
            tier_to_filter = {combo_name: base_tiers}

    # modality_clean options: "all" (no filter), plus actual values from annotations
    modality_clean_values = annotations["modality_clean"].unique().to_list()
    modality_clean_options = ["all"] + sorted(modality_clean_values)

    # Get unique Metadata_Group values from profiles
    if "Metadata_Group" in df.columns:
        groups = df["Metadata_Group"].unique().to_list()
    else:
        groups = ["all"]  # Fallback if no Metadata_Group column

    print(f"\n=== Phenotypic Consistency Evaluation by Subsets ===")
    print(f"Groups: {groups}")
    print(f"Tiers: {tier_options}")
    print(f"Modality Clean: {modality_clean_options}")
    print(f"Min compounds per target: {min_compounds_per_target}")

    # Pre-compute ALL annotated JCPs across all tiers (for truly unknown background)
    # A compound is only "unknown" if it has no annotations in ANY tier
    all_annotated_jcps = set(
        annotations.filter(pl.col("WithinModalityTier") != "Excluded")[compound_col]
        .unique()
        .to_list()
    )
    print(f"Total annotated JCPs across all tiers: {len(all_annotated_jcps)}")

    # Run evaluation for both modes: without and with unknown background
    # FIXME TEMPORARILY FORCE ONLY WITHOUT UNKNOWN
    #background_modes = [False, True] if include_unknown_as_background else [False]
    background_modes = [False]

    print(f"Background modes to run: {['without_unknown', 'with_unknown'] if include_unknown_as_background else ['without_unknown']}")

    # Active filter modes: always run both with and without active-only filtering
    # "all_compounds" = include all compounds, "active_only" = only phenotypically active compounds
    if activity_map is not None and len(activity_map) > 0:
        active_filter_modes = [False, True]  # [all_compounds, active_only]
        print("Active filter modes: ['all_compounds', 'active_only']")

        # Pre-compute active JCPs per group for quick lookup
        active_jcps_by_group = {}
        for grp in activity_map["Metadata_Group"].unique():
            grp_active = activity_map[
                (activity_map["Metadata_Group"] == grp) &
                (activity_map["below_corrected_p"] == True)
            ][compound_col].tolist()
            active_jcps_by_group[grp] = set(grp_active)
            print(f"  Active JCPs in {grp}: {len(grp_active)}")
    else:
        active_filter_modes = [False]  # Only run without filtering
        active_jcps_by_group = {}
        print("Active filter modes: ['all_compounds'] (no activity_map provided)")

    for use_unknown_bg in background_modes:
        bg_mode_name = "with_unknown_bg" if use_unknown_bg else "without_unknown_bg"
        print(f"\n{'='*50}")
        print(f"Running PC evaluation: {bg_mode_name}")
        print(f"{'='*50}")

        for group in groups:
            print(f"\n--- Group: {group} ---")

            for filter_active_only in active_filter_modes:
                active_mode_name = "active_only" if filter_active_only else "all_compounds"
                print(f"\n  Active filter: {active_mode_name}")

                # Get active JCPs for this group if filtering
                if filter_active_only:
                    active_jcps_for_group = active_jcps_by_group.get(group, set())
                    if len(active_jcps_for_group) == 0:
                        print(f"    Skipping (no active compounds in this group)")
                        continue
                else:
                    active_jcps_for_group = None  # No filtering

                for tier_name in tier_options:
                    print(f"    Tier: {tier_name}")

                    # Get annotations for this tier (or combination of tiers)
                    tiers_to_include = tier_to_filter[tier_name]
                    tier_ann = annotations.filter(pl.col("WithinModalityTier").is_in(tiers_to_include))

                    if tier_ann.height == 0:
                        continue

                    for modality_clean in modality_clean_options:
                        # Filter annotations by modality_clean (or use all if "all")
                        if modality_clean == "all":
                            subset_ann = tier_ann
                        else:
                            subset_ann = tier_ann.filter(pl.col("modality_clean") == modality_clean)

                        if subset_ann.height == 0:
                            continue

                        # Create target list per JCP (pipe-separated)
                        jcp_targets = (
                            subset_ann
                            .group_by(compound_col)
                            .agg(pl.col("target").unique().alias("target_list"))
                            .with_columns(
                                pl.col("target_list").list.join("|").alias("Metadata_target_list")
                            )
                            .select([compound_col, "Metadata_target_list"])
                        )

                        # Get JCPs with known targets in this tier/modality subset
                        annotated_jcps_this_tier = set(jcp_targets[compound_col].to_list())

                        # Apply active-only filter if enabled
                        if filter_active_only and active_jcps_for_group is not None:
                            annotated_jcps_this_tier = annotated_jcps_this_tier & active_jcps_for_group

                        subset_name = f"{group}_{tier_name}_{modality_clean}_{active_mode_name}_{bg_mode_name}"

                        # Filter profiles to this subset
                        if use_unknown_bg:
                            # Include compounds with annotations for this tier
                            # PLUS compounds that are truly unknown (not in ANY tier)
                            # Exclude compounds that have annotations in other tiers but not this one
                            if group == "all" or "Metadata_Group" not in df.columns:
                                df_group = df
                            else:
                                df_group = df.filter(pl.col("Metadata_Group") == group)

                            # Get all JCPs in this group
                            group_jcps = set(df_group[compound_col].unique().to_list())

                            # Truly unknown = in group but not annotated in ANY tier
                            truly_unknown_jcps = group_jcps - all_annotated_jcps

                            # Apply active-only filter to unknowns as well
                            if filter_active_only and active_jcps_for_group is not None:
                                truly_unknown_jcps = truly_unknown_jcps & active_jcps_for_group

                            # Include: annotated in this tier + truly unknown
                            jcps_to_include = annotated_jcps_this_tier | truly_unknown_jcps

                            df_subset = df_group.filter(pl.col(compound_col).is_in(list(jcps_to_include)))
                        else:
                            # Only include compounds with known targets in this tier
                            if group == "all" or "Metadata_Group" not in df.columns:
                                df_subset = df.filter(pl.col(compound_col).is_in(list(annotated_jcps_this_tier)))
                            else:
                                df_subset = df.filter(
                                    (pl.col(compound_col).is_in(list(annotated_jcps_this_tier))) &
                                    (pl.col("Metadata_Group") == group)
                                )

                        if df_subset.height == 0:
                            continue

                        # Merge target info (compounds not in this tier's annotations will get null).
                        # Cast join key to Utf8 on both sides — normalized inputs from norm_3
                        # carry Metadata_JCP2022 as Categorical while annotations are String,
                        # which polars refuses to auto-cast across.
                        df_subset = df_subset.with_columns(pl.col(compound_col).cast(pl.Utf8))
                        jcp_targets_str = jcp_targets.with_columns(pl.col(compound_col).cast(pl.Utf8))
                        df_subset = df_subset.join(jcp_targets_str, on=compound_col, how="left")

                        # Fill null targets with "unknown" for truly unknown compounds
                        df_subset = df_subset.with_columns(
                            pl.col("Metadata_target_list").fill_null("unknown")
                        )

                        n_jcps = df_subset[compound_col].n_unique()
                        n_profiles = df_subset.height
                        n_annotated = df_subset.filter(pl.col("Metadata_target_list") != "unknown")[compound_col].n_unique()
                        n_unknown = n_jcps - n_annotated

                        if n_annotated < min_compounds_per_target:
                            print(f"      {modality_clean}: Skipping (only {n_annotated} annotated JCPs)")
                            continue

                        if use_unknown_bg and n_unknown > 0:
                            print(f"      {modality_clean}: {n_annotated} annotated + {n_unknown} unknown JCPs, {n_profiles} profiles")
                        else:
                            print(f"      {modality_clean}: {n_annotated} JCPs, {n_profiles} profiles")

                        # Calculate PC
                        pc_result = calculate_phenotypic_consistency(
                            df_subset,
                            features,
                            null_size=null_size,
                            p_threshold=p_threshold,
                            seed=seed,
                            min_compounds_per_target=min_compounds_per_target,
                            include_unknown_as_background=use_unknown_bg,
                            compound_col=compound_col,
                            target_col="Metadata_target_list",
                            negcon_col=negcon_col,
                        )

                        results[subset_name] = pc_result

                        # Add to summary
                        all_pc_results.append({
                            "subset": subset_name,
                            "group": group,
                            "tier": tier_name,
                            "modality_clean": modality_clean,
                            "active_filter": active_mode_name,
                            "background_mode": bg_mode_name,
                            "n_jcps_annotated": n_annotated,
                            "n_jcps_unknown": n_unknown,
                            "n_profiles": n_profiles,
                            "PC": pc_result["pct_targets_active"],
                            "n_targets_active": pc_result["n_targets_active"],
                            "n_targets_total": pc_result["n_targets_total"],
                            "median_n_total_pairs": pc_result["median_n_total_pairs"],
                        })

                        print(f"        PC: {pc_result['pct_targets_active']:.1f}% ({pc_result['n_targets_active']}/{pc_result['n_targets_total']} targets, median pairs: {pc_result['median_n_total_pairs']})")

                        # Save per-target results if available
                        if pc_result["target_consistency"] is not None:
                            target_path = output_dir / f"pc_{subset_name}_per_target.csv"
                            pc_result["target_consistency"].to_csv(target_path, index=False)

    # Save summary
    if all_pc_results:
        summary_df = pd.DataFrame(all_pc_results)
        summary_path = output_dir / "phenotypic_consistency_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Saved PC summary to: {summary_path}")

    return {
        "subset_results": results,
        "summary": all_pc_results,
    }


def calculate_batch_effects(
    df: pl.DataFrame,
    features: list[str],
    null_size: int = 10_000,
    p_threshold: float = 0.05,
    seed: int = 0,
    compound_col: str = "Metadata_JCP2022",
    negcon_col: str = "Metadata_negcon",
    plate_col: str = "Metadata_Plate",
    well_col: str = "Metadata_Well",
    group_col: str = "Metadata_Group",
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Calculate batch effect metrics.

    Two metrics are computed:
    1. Well Position Effect (within groups): Do wells in the same position cluster together?
       - Positive: same group, same well location, different compounds
       - Negative: same group, same plate, different well location

    2. Plate Batch Effect: Are plates distinguishable using only negative controls?
       - Positive: negative controls from same plate
       - Negative: negative controls from different plate AND different well location

    Args:
        df: Profiles with metadata
        features: Feature column names
        null_size: Size of null distribution
        p_threshold: Significance threshold
        seed: Random seed
        compound_col: Column containing compound identifier
        negcon_col: Column containing negative control flag
        plate_col: Column containing plate identifier
        well_col: Column containing well identifier (e.g., "A01")
        group_col: Column containing group identifier (e.g., "Metadata_Group")
        verbose: If True, print detailed information about each step

    Returns:
        Dictionary with batch effect metrics
    """
    from copairs import map as copairs_map

    results = {}

    df_pd = df.to_pandas()

    if well_col not in df_pd.columns:
        print(f"  Warning: {well_col} not found, skipping batch effect analysis")
        return {"well_position_effect": None, "plate_batch_effect": None}

    if verbose:
        print("\n" + "=" * 60)
        print("BATCH EFFECT ANALYSIS")
        print("=" * 60)
        print(f"\nInput data:")
        print(f"  Total samples: {len(df_pd):,}")
        print(f"  Total features: {len(features):,}")
        print(f"  Unique plates ({plate_col}): {df_pd[plate_col].nunique():,}")
        print(f"  Unique wells ({well_col}): {df_pd[well_col].nunique():,}")
        print(f"  Unique compounds ({compound_col}): {df_pd[compound_col].nunique():,}")
        if group_col in df_pd.columns:
            print(f"  Unique groups ({group_col}): {df_pd[group_col].nunique()}")
            for grp in df_pd[group_col].unique():
                n_grp = len(df_pd[df_pd[group_col] == grp])
                print(f"    - {grp}: {n_grp:,} samples")
        if negcon_col in df_pd.columns:
            n_negcon = (df_pd[negcon_col] == True).sum()
            print(f"  Negative controls: {n_negcon:,}")

    # =========================================
    # 1. Well Position Effect (within groups)
    # =========================================
    print("\n  Calculating Well Position Effect...")

    if verbose:
        print("\n  --- Well Position Effect ---")
        print("  Purpose: Detect if wells at the same plate position cluster together")
        print("           (indicates well position bias like edge effects)")
        print("  Note: Only non-control compounds are used")
        print("\n  Comparison design:")
        print("    Positive pairs: Same group + Same well position + Different compound")
        print("    Negative pairs: Same group + Same plate + Different well position + Different compound")

    try:
        # Check if group column exists
        has_groups = group_col in df_pd.columns

        # Filter out negative controls - only use treatment compounds
        if negcon_col in df_pd.columns:
            df_treatments = df_pd[df_pd[negcon_col] == False].copy()
            n_excluded = len(df_pd) - len(df_treatments)
            if verbose:
                print(f"\n  Filtering to non-control compounds:")
                print(f"    Total samples: {len(df_pd):,}")
                print(f"    Negative controls excluded: {n_excluded:,}")
                print(f"    Treatment samples: {len(df_treatments):,}")
        else:
            df_treatments = df_pd.copy()
            if verbose:
                print(f"\n  No {negcon_col} column found, using all samples")

        if len(df_treatments) < 10:
            print(f"    Not enough treatment samples ({len(df_treatments)}) for well position analysis")
            results["well_position_effect"] = {
                "pct_active": 0.0,
                "n_active": 0,
                "n_total": 0,
                "mean_map": 0.0,
            }
            return results

        # Positive: same group, same well location, different compound
        # Negative: same group, same plate, different well location, different compound
        if has_groups:
            pos_sameby_well = [group_col, well_col]
            neg_sameby_well = [group_col, plate_col]
        else:
            pos_sameby_well = [well_col]
            neg_sameby_well = [plate_col]

        pos_diffby_well = [compound_col]
        neg_diffby_well = [well_col, compound_col]

        if verbose:
            print(f"\n  Copairs configuration:")
            print(f"    pos_sameby: {pos_sameby_well}")
            print(f"    pos_diffby: {pos_diffby_well}")
            print(f"    neg_sameby: {neg_sameby_well}")
            print(f"    neg_diffby: {neg_diffby_well}")

        # Filter to only include rows where we can make valid comparisons
        # Need at least 2 different compounds per well location (within each group)
        if has_groups:
            # Count unique compounds per (group, well) combination
            well_loc_counts = df_treatments.groupby([group_col, well_col])[compound_col].nunique()
            valid_combinations = well_loc_counts[well_loc_counts >= 2].index.tolist()

            if verbose:
                print(f"\n  Filtering:")
                print(f"    Total (group, well) combinations: {len(well_loc_counts):,}")
                print(f"    Combinations with >= 2 compounds: {len(valid_combinations):,}")

            if len(valid_combinations) < 2:
                print("    Not enough valid (group, well) combinations for analysis")
                results["well_position_effect"] = {
                    "pct_active": 0.0,
                    "n_active": 0,
                    "n_total": 0,
                    "mean_map": 0.0,
                }
            else:
                # Filter to valid combinations
                df_well = df_treatments[
                    df_treatments.apply(lambda r: (r[group_col], r[well_col]) in valid_combinations, axis=1)
                ].copy()

                if verbose:
                    print(f"    Samples after filtering: {len(df_well):,}")
                    print(f"    Unique well positions: {df_well[well_col].nunique():,}")

                metadata_well = df_well.filter(regex="^Metadata")
                profiles_well = df_well[features].values

                if verbose:
                    print(f"\n  Computing average precision...")

                well_ap = copairs_map.average_precision(
                    metadata_well, profiles_well,
                    pos_sameby_well, pos_diffby_well,
                    neg_sameby_well, neg_diffby_well
                )

                if verbose:
                    print(f"    AP results: {len(well_ap):,} entries")

                if len(well_ap) > 0:
                    # MAP grouped by (group, well) - each combination gets its own MAP score
                    well_map = copairs_map.mean_average_precision(
                        well_ap, pos_sameby_well,
                        null_size=null_size, threshold=p_threshold, seed=seed
                    )
                    well_map["below_corrected_p"] = well_map["corrected_p_value"] < p_threshold

                    pct_active = (well_map["below_corrected_p"].sum() / len(well_map)) * 100 if len(well_map) > 0 else 0
                    mean_map_val = well_map["mean_average_precision"].mean() if len(well_map) > 0 else 0

                    n_groups = df_well[group_col].nunique()
                    n_wells = df_well[well_col].nunique()

                    # Calculate per-group statistics
                    per_group_stats = {}
                    for grp in well_map[group_col].unique():
                        grp_data = well_map[well_map[group_col] == grp]
                        grp_active = grp_data["below_corrected_p"].sum()
                        grp_total = len(grp_data)
                        grp_pct = (grp_active / grp_total * 100) if grp_total > 0 else 0
                        per_group_stats[grp] = {
                            "pct_active": float(grp_pct),
                            "n_active": int(grp_active),
                            "n_total": int(grp_total),
                            "mean_map": float(grp_data["mean_average_precision"].mean()),
                        }

                    results["well_position_effect"] = {
                        "pct_active": float(pct_active),
                        "n_active": int(well_map["below_corrected_p"].sum()),
                        "n_total": int(len(well_map)),
                        "mean_map": float(mean_map_val),
                        "n_samples_used": int(len(df_well)),
                        "n_groups": int(n_groups),
                        "n_well_positions": int(n_wells),
                        "n_group_well_combinations": int(len(well_map)),
                        "per_group": per_group_stats,
                    }

                    if verbose:
                        print(f"\n  Results:")
                        print(f"    MAP entries (group, well combinations): {len(well_map):,}")
                        print(f"    Groups: {n_groups}, Well positions: {n_wells}")
                        print(f"    Significant (p < {p_threshold}): {well_map['below_corrected_p'].sum():,}")
                        print(f"\n    Per-group breakdown:")
                        for grp, stats in per_group_stats.items():
                            print(f"      {grp}: {stats['pct_active']:.2f}% ({stats['n_active']}/{stats['n_total']} wells)")

                    print(f"    Well Position Effect: {pct_active:.2f}% of (group, well) combinations show effect")
                    for grp, stats in per_group_stats.items():
                        print(f"      {grp}: {stats['pct_active']:.2f}%")
                else:
                    results["well_position_effect"] = {
                        "pct_active": 0.0,
                        "n_active": 0,
                        "n_total": 0,
                        "mean_map": 0.0,
                    }
        else:
            # No groups - original logic
            well_loc_counts = df_treatments.groupby(well_col)[compound_col].nunique()
            valid_well_locs = well_loc_counts[well_loc_counts >= 2].index.tolist()

            if verbose:
                print(f"\n  Filtering (no groups):")
                print(f"    Total well positions: {len(well_loc_counts):,}")
                print(f"    Positions with >= 2 compounds: {len(valid_well_locs):,}")

            if len(valid_well_locs) < 2:
                print("    Not enough valid well locations for analysis")
                results["well_position_effect"] = {
                    "pct_active": 0.0,
                    "n_active": 0,
                    "n_total": 0,
                    "mean_map": 0.0,
                }
            else:
                df_well = df_treatments[df_treatments[well_col].isin(valid_well_locs)].copy()

                if verbose:
                    print(f"    Samples after filtering: {len(df_well):,}")

                metadata_well = df_well.filter(regex="^Metadata")
                profiles_well = df_well[features].values

                well_ap = copairs_map.average_precision(
                    metadata_well, profiles_well,
                    pos_sameby_well, pos_diffby_well,
                    neg_sameby_well, neg_diffby_well
                )

                if len(well_ap) > 0:
                    well_map = copairs_map.mean_average_precision(
                        well_ap, pos_sameby_well,
                        null_size=null_size, threshold=p_threshold, seed=seed
                    )
                    well_map["below_corrected_p"] = well_map["corrected_p_value"] < p_threshold

                    pct_active = (well_map["below_corrected_p"].sum() / len(well_map)) * 100 if len(well_map) > 0 else 0
                    mean_map_val = well_map["mean_average_precision"].mean() if len(well_map) > 0 else 0

                    results["well_position_effect"] = {
                        "pct_active": float(pct_active),
                        "n_active": int(well_map["below_corrected_p"].sum()),
                        "n_total": int(len(well_map)),
                        "mean_map": float(mean_map_val),
                        "n_samples_used": int(len(df_well)),
                        "n_well_positions": int(len(valid_well_locs)),
                    }
                    print(f"    Well Position Effect: {pct_active:.2f}% of positions show effect")
                else:
                    results["well_position_effect"] = {
                        "pct_active": 0.0,
                        "n_active": 0,
                        "n_total": 0,
                        "mean_map": 0.0,
                    }
    except Exception as e:
        print(f"    Warning: Well position effect calculation failed: {e}")
        results["well_position_effect"] = {
            "pct_active": 0.0,
            "n_active": 0,
            "n_total": 0,
            "mean_map": 0.0,
            "error": str(e),
        }

    # =========================================
    # 2. Plate Batch Effect (using only negcons)
    # =========================================
    print("\n  Calculating Plate Batch Effect (using negative controls only)...")

    if verbose:
        print("\n  --- Plate Batch Effect ---")
        print("  Purpose: Detect if plates are distinguishable from each other")
        print("           (indicates plate-level batch effects)")
        print("\n  Comparison design:")
        print("    Positive pairs: Same plate (using only negative controls)")
        print("    Negative pairs: Same well position + Different plate")
        print("                    (controls for well position effects)")

    try:
        # Filter to negative controls only
        df_negcon = df_pd[df_pd[negcon_col] == True].copy()

        if verbose:
            print(f"\n  Filtering to negative controls:")
            print(f"    Total samples: {len(df_pd):,}")
            print(f"    Negative controls: {len(df_negcon):,}")

        if len(df_negcon) < 10:
            print(f"    Not enough negative controls ({len(df_negcon)}) for plate batch analysis")
            results["plate_batch_effect"] = {
                "pct_active": 0.0,
                "n_active": 0,
                "n_total": 0,
                "mean_map": 0.0,
            }
        else:
            # Check we have multiple plates
            n_plates = df_negcon[plate_col].nunique()
            n_wells = df_negcon[well_col].nunique()

            if verbose:
                print(f"    Unique plates: {n_plates:,}")
                print(f"    Unique well positions: {n_wells:,}")
                print(f"    Avg negcons per plate: {len(df_negcon) / n_plates:.1f}")

            if n_plates < 2:
                print(f"    Not enough plates ({n_plates}) for batch effect analysis")
                results["plate_batch_effect"] = {
                    "pct_active": 0.0,
                    "n_active": 0,
                    "n_total": 0,
                    "mean_map": 0.0,
                }
            else:
                metadata_negcon = df_negcon.filter(regex="^Metadata")
                profiles_negcon = df_negcon[features].values

                # Positive: same plate
                # Negative: same well position + different plate
                pos_sameby_plate = [plate_col]
                pos_diffby_plate = []
                neg_sameby_plate = [well_col]  # Same well position
                neg_diffby_plate = [plate_col]  # Different plate

                if verbose:
                    print(f"\n  Copairs configuration:")
                    print(f"    pos_sameby: {pos_sameby_plate}")
                    print(f"    pos_diffby: {pos_diffby_plate}")
                    print(f"    neg_sameby: {neg_sameby_plate}")
                    print(f"    neg_diffby: {neg_diffby_plate}")
                    print(f"\n  Computing average precision...")

                plate_ap = copairs_map.average_precision(
                    metadata_negcon, profiles_negcon,
                    pos_sameby_plate, pos_diffby_plate,
                    neg_sameby_plate, neg_diffby_plate
                )

                if verbose:
                    print(f"    AP results: {len(plate_ap):,} entries")

                if len(plate_ap) > 0:
                    plate_map = copairs_map.mean_average_precision(
                        plate_ap, pos_sameby_plate,
                        null_size=null_size, threshold=p_threshold, seed=seed
                    )
                    plate_map["below_corrected_p"] = plate_map["corrected_p_value"] < p_threshold

                    pct_active = (plate_map["below_corrected_p"].sum() / len(plate_map)) * 100 if len(plate_map) > 0 else 0
                    mean_map_val = plate_map["mean_average_precision"].mean() if len(plate_map) > 0 else 0

                    results["plate_batch_effect"] = {
                        "pct_active": float(pct_active),
                        "n_active": int(plate_map["below_corrected_p"].sum()),
                        "n_total": int(len(plate_map)),
                        "mean_map": float(mean_map_val),
                        "n_plates": int(n_plates),
                        "n_negcons": int(len(df_negcon)),
                    }

                    if verbose:
                        print(f"\n  Results:")
                        print(f"    MAP entries (plates): {len(plate_map):,}")
                        print(f"    Significant (p < {p_threshold}): {plate_map['below_corrected_p'].sum():,}")

                    print(f"    Plate Batch Effect: {pct_active:.2f}% of plates are distinguishable ({n_plates} plates, {len(df_negcon)} negcons)")
                else:
                    results["plate_batch_effect"] = {
                        "pct_active": 0.0,
                        "n_active": 0,
                        "n_total": 0,
                        "mean_map": 0.0,
                    }
    except Exception as e:
        print(f"    Warning: Plate batch effect calculation failed: {e}")
        results["plate_batch_effect"] = {
            "pct_active": 0.0,
            "n_active": 0,
            "n_total": 0,
            "mean_map": 0.0,
            "error": str(e),
        }

    if verbose:
        print("\n" + "=" * 60)

    return results


def setup_control_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Set up negative control columns if not present."""
    # Check for negcon column
    if "Metadata_negcon" not in df.columns:
        # Try to infer from pert_type or control_type
        if "Metadata_pert_type" in df.columns:
            df = df.with_columns(
                (pl.col("Metadata_pert_type") == "negcon").alias("Metadata_negcon")
            )
        elif "Metadata_control_type" in df.columns:
            df = df.with_columns(
                (pl.col("Metadata_control_type") == "negcon").alias("Metadata_negcon")
            )
        elif "Metadata_JCP2022" in df.columns:
            # DMSO JCP ID
            df = df.with_columns(
                (pl.col("Metadata_JCP2022") == "JCP2022_085227").alias("Metadata_negcon")
            )
        else:
            print("  Warning: Could not identify negative controls, setting all to False")
            df = df.with_columns(pl.lit(False).alias("Metadata_negcon"))

    n_negcon = df.filter(pl.col("Metadata_negcon") == True).height
    print(f"  Negative controls: {n_negcon}")

    return df


def calculate_phenotypic_activity(
    df: pl.DataFrame,
    features: list[str],
    null_size: int = 10_000,
    p_threshold: float = 0.05,
    seed: int = 0,
    compound_col: str = "Metadata_JCP2022",
    negcon_col: str = "Metadata_negcon",
    batch_col: str = "Metadata_Plate",
) -> dict[str, Any]:
    """
    Calculate Phenotypic Activity (compound replicate retrieval).

    Measures how well replicates of the same compound cluster together
    compared to random compounds.

    Args:
        df: Profiles with metadata
        features: Feature column names
        null_size: Size of null distribution
        p_threshold: Significance threshold
        seed: Random seed
        compound_col: Column containing compound identifier
        negcon_col: Column containing negative control flag
        batch_col: Column containing batch/plate identifier

    Returns:
        Dictionary with activity metrics and per-compound results
    """
    from copairs import map as copairs_map
    from copairs.map.average_precision import p_values
    from copairs.matching import assign_reference_index

    df_pd = df.to_pandas()

    # Define indexing for copairs
    if negcon_col in df_pd.columns:
        df_pd = assign_reference_index(
            df_pd,
            f"{negcon_col} == True",
            reference_col="Metadata_reference_index",
            default_value=-1,
        )
    else:
        df_pd = assign_reference_index(
            df_pd,
            f"{compound_col} == 'JCP2022_085227'",  # DMSO
            reference_col="Metadata_reference_index",
            default_value=-1,
        )
    
    pos_sameby = [compound_col, "Metadata_Group", "Metadata_reference_index"]
    pos_diffby = []
    neg_sameby = [batch_col,  "Metadata_Group"]
    neg_diffby = [compound_col, negcon_col, "Metadata_reference_index"]

    
    
    metadata = df_pd.filter(regex="^Metadata")
    profiles = df_pd[features].values

    print(f"  Computing average precision for {len(df_pd[compound_col].unique())} compounds...")
    activity_ap = copairs_map.average_precision(
        metadata, profiles, pos_sameby, pos_diffby, neg_sameby, neg_diffby
    )

    # Filter out negative controls
    if negcon_col in activity_ap.columns:
        activity_ap = activity_ap.query(f"{negcon_col} == False").copy()
    else:
        activity_ap = activity_ap.query(f"{compound_col} != 'JCP2022_085227'").copy()

    # Calculate p-values
    print("  Calculating p-values...")
    activity_ap["p_value"] = p_values(activity_ap, null_size=null_size, seed=seed)
    activity_ap["below_p"] = activity_ap["p_value"] < p_threshold

    # Summarize number of replicates per compound group pair
    replicate_counts = activity_ap.groupby(pos_sameby).size()

    # Calculate mean average precision
    print("  Calculating mean average precision...")
    activity_map = copairs_map.mean_average_precision(
        activity_ap, pos_sameby, null_size=null_size, threshold=p_threshold, seed=seed
    ).copy()
    activity_map["below_corrected_p"] = activity_map["corrected_p_value"] < p_threshold

    pct_compounds_active = (
        activity_map["below_corrected_p"].sum() / len(activity_map)
    ) * 100

    # Merge replicate counts into activity_map
    activity_map = activity_map.merge(
        replicate_counts.rename("n_replicate_pairs"),
        on=pos_sameby,
        how="left",
    )
    
    # Get the per GROUP summary statistics
    group_summary = activity_map.groupby("Metadata_Group").agg(  pct_active=("below_corrected_p", "mean"),
                                                               num_active=("below_corrected_p", "sum"),
                                                               mean_normalized_map=("mean_normalized_average_precision", "mean"), 
                                                               median_normalized_map=("mean_normalized_average_precision", "median"),
                                                               mean_n_replicates=("n_replicate_pairs", "mean"),
                                                               median_n_replicates=("n_replicate_pairs", "median"),
                                                               n_unique_jcp=("Metadata_JCP2022", "nunique")).reset_index()      
    group_summary["pct_active"] *= 100   # Multiply by 100 to get percentage
    
    return {
        "activity_ap": activity_ap,
        "activity_map": activity_map,
        "group_summary": group_summary,
        "pct_compounds_active": float(pct_compounds_active),
        "n_compounds": int(len(activity_map)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Phenotypic Activity per JCP ID"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Path to input profiles parquet file"
    )
    parser.add_argument(
        "--annotations", "-a",
        type=Path,
        default=Path("metadata/refchemdb_conf_jump_matched.parquet"),
        help="Path to annotations parquet file (default: metadata/refchemdb_conf_jump_matched.parquet)"
    )
    parser.add_argument(
        "--metadata", "-m",
        type=Path,
        default=Path("metadata/metadata_dataset.parquet"),
        help="Path to metadata parquet file (default: metadata/metadata_dataset.parquet)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory (default: same directory as input)"
    )
    parser.add_argument(
        "--compound-col",
        type=str,
        default="Metadata_JCP2022",
        help="Column containing compound identifier (default: Metadata_JCP2022)"
    )
    parser.add_argument(
        "--batch-col",
        type=str,
        default="Metadata_Plate",
        help="Column containing batch identifier (default: Metadata_Plate)"
    )
    parser.add_argument(
        "--null-size",
        type=int,
        default=10_000,
        help="Size of null distribution for p-value calculation (default: 10000)"
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.05,
        help="Significance threshold (default: 0.05)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--min-compounds-per-target",
        dest="min_compounds_per_target",
        type=int,
        default=3,
        help="Minimum compounds per target for PC evaluation (default: 3)"
    )
    parser.add_argument(
        "--include-unknown-as-background",
        dest="include_unknown_as_background",
        action="store_true",
        help="Include compounds with unknown targets as background (negatives) for PC calculation"
    )
    parser.add_argument(
        "--all-tier-combinations",
        dest="all_tier_combinations",
        action="store_true",
        help="Run PC on all tier combinations (Tier1, Tier2, etc.) instead of only combined tiers"
    )
    parser.add_argument(
        "--verbose-batch-effects",
        dest="verbose_batch_effects",
        action="store_true",
        help="Print detailed information about batch effect calculations"
    )
    parser.add_argument(
        "--skip-batch-effects",
        dest="skip_batch_effects",
        action="store_true",
        help="Skip batch effect calculation entirely"
    )
    parser.add_argument(
        "--no-annotations",
        action="store_true",
        help="Skip merging annotations (use when data is already annotated)"
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Skip merging metadata (use when data already has compound IDs)"
    )
    parser.add_argument(
        "--skip-cross-modality",
        dest="skip_cross_modality",
        action="store_true",
        help="Skip cross-modality retrieval evaluation"
    )
    parser.add_argument(
        "--recall-k-percentages",
        dest="recall_k_percentages",
        type=str,
        default="1,5,10",
        help="Comma-separated list of k percentages for recall@k (default: 1,5,10)"
    )
    parser.add_argument(
        "--similarity-directions",
        dest="similarity_directions",
        type=str,
        default="top,bottom",
        help="Comma-separated similarity directions: top (most similar), bottom (least similar) (default: top,bottom)"
    )
    parser.add_argument(
        "--include-genetic-pairs",
        dest="include_genetic_pairs",
        action="store_true",
        help="Include ORF vs CRISPR cross-modality retrieval (matching on gene symbol)"
    )

    args = parser.parse_args()

    # Set output directory
    if args.output is None:
        output_dir = args.input.parent / f"{args.input.stem}_evaluation"
    else:
        output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Phenotypic Activity Evaluation ===")
    print(f"Input: {args.input}")
    print(f"Output: {output_dir}")
    print()

    # Load profiles
    print("Loading profiles...")
    df = load_profiles(args.input)
    print(f"  Shape: {df.shape}")

    # Check if data is pre-annotated (has compound_col and negcon column)
    has_compound_col = args.compound_col in df.columns
    has_negcon_col = "Metadata_negcon" in df.columns

    # Check if we need to merge metadata
    if not has_compound_col and not args.no_metadata:
        print(f"\n{args.compound_col} not in profiles, merging metadata...")
        df = merge_metadata(df, args.metadata)

    # # Merge annotations
    # if not args.no_annotations and "Metadata_JCP2022" in df.columns:
    #     if args.annotations.exists():
    #         print("\nMerging annotations...")
    #         df = merge_annotations(df, args.annotations)
    #     else:
    #         print(f"\nWarning: Annotations file not found: {args.annotations}")
    # elif args.no_annotations:
    #     print("\nSkipping annotation merge (--no-annotations).")

    # Setup control columns
    print("\nSetting up control columns...")
    df = setup_control_columns(df)

    # Filter to only rows with valid compound IDs
    n_before = len(df)
    df = df.filter(pl.col(args.compound_col).is_not_null())
    n_after = len(df)
    if n_before != n_after:
        print(f"  Filtered {n_before - n_after} rows without {args.compound_col}")

    # Get features
    features, _ = infer_columns(df)
    features = get_numeric_features(df, features)
    print(f"\nFeatures: {len(features)}")

    
    
    # Calculate Phenotypic Activity
    print("\nCalculating Phenotypic Activity...")
    pa_results = calculate_phenotypic_activity(
        df,
        features,
        null_size=args.null_size,
        p_threshold=args.p_threshold,
        seed=args.seed,
        compound_col=args.compound_col,
        batch_col=args.batch_col,
    )

    # Calculate batch effects (unless skipped)
    if args.skip_batch_effects:
        print("\nSkipping Batch Effects calculation (--skip-batch-effects)")
        batch_effects = {}
    else:
        print("\nCalculating Batch Effects...")
        batch_effects = calculate_batch_effects(
            df,
            features,
            null_size=args.null_size,
            p_threshold=args.p_threshold,
            seed=args.seed,
            compound_col=args.compound_col,
            plate_col=args.batch_col,
            verbose=args.verbose_batch_effects,
        )

    # Print summary
    print("\n=== Results ===")
    print(f"Total compounds: {pa_results['n_compounds']}")
    print(f"Phenotypic Activity (PA): {pa_results['pct_compounds_active']:.2f}%")
    if batch_effects.get("well_position_effect"):
        print(f"Well Position Effect: {batch_effects['well_position_effect']['pct_active']:.2f}%")
    if batch_effects.get("plate_batch_effect"):
        print(f"Plate Batch Effect: {batch_effects['plate_batch_effect']['pct_active']:.2f}%")

    # Save results
    print(f"\nSaving results to {output_dir}...")

    # Save per-compound average precision
    if pa_results["activity_ap"] is not None and len(pa_results["activity_ap"]) > 0:
        ap_path = output_dir / "phenotypic_activity_per_compound.csv"
        pa_results["activity_ap"].to_csv(ap_path, index=False)
        print(f"  Saved: {ap_path}")

    # Save mean average precision per compound
    if pa_results["activity_map"] is not None and len(pa_results["activity_map"]) > 0:
        map_path = output_dir / "phenotypic_activity_map.csv"
        pa_results["activity_map"].to_csv(map_path, index=False)
        print(f"  Saved: {map_path}")

    # Save summary metrics
    metrics = {
        "PA": pa_results["pct_compounds_active"],
        "n_compounds": pa_results["n_compounds"],
        "input_file": str(args.input),
        "annotations_file": str(args.annotations),
        "compound_col": args.compound_col,
        "batch_col": args.batch_col,
        "null_size": args.null_size,
        "p_threshold": args.p_threshold,
        "seed": args.seed,
    }

    # Add group summary if available
    if "group_summary" in pa_results:
        metrics["group_summary"] = pa_results["group_summary"].set_index("Metadata_Group").to_dict(orient="index")

    # Add batch effects
    if batch_effects:
        metrics["batch_effects"] = batch_effects

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {metrics_path}")

    # Phenotypic Consistency evaluation by subsets
    if not args.no_annotations and args.annotations.exists():
        print("\n" + "=" * 50)
        print("Loading tier annotations for PC evaluation...")
        annotations = load_tier_annotations(args.annotations)
        print(f"  Loaded {annotations.height} annotations")

        pc_results = evaluate_phenotypic_consistency_by_subsets(
            df,
            features,
            annotations,
            output_dir,
            activity_map=pa_results.get("activity_map"),
            null_size=args.null_size,
            p_threshold=args.p_threshold,
            seed=args.seed,
            min_compounds_per_target=args.min_compounds_per_target,
            include_unknown_as_background=args.include_unknown_as_background,
            compound_col=args.compound_col,
            negcon_col="Metadata_negcon",
            all_tier_combinations=args.all_tier_combinations,
        )

        # Add PC summary to metrics
        if pc_results["summary"]:
            metrics["PC_summary"] = pc_results["summary"]
            # Update metrics.json with PC results
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  Updated: {metrics_path}")
    else:
        if args.no_annotations:
            print("\nSkipping PC evaluation (--no-annotations).")
        else:
            print(f"\nWarning: Annotations file not found: {args.annotations}")
            print("Skipping PC evaluation.")

    # Cross-Modality Retrieval evaluation
    if not args.skip_cross_modality and not args.no_annotations and args.annotations.exists():
        print("\n" + "=" * 50)
        print("Loading annotations for Cross-Modality Retrieval evaluation...")

        # Load full annotations (not just tier annotations)
        full_annotations = pl.read_parquet(args.annotations)
        print(f"  Loaded {full_annotations.height} annotation rows")

        # Parse recall k percentages
        k_percentages = [float(k) for k in args.recall_k_percentages.split(",")]
        similarity_directions = [s.strip() for s in args.similarity_directions.split(",")]

        cmr_results = evaluate_cross_modality_retrieval(
            df,
            features,
            full_annotations,
            output_dir,
            activity_map=pa_results.get("activity_map"),
            compound_col=args.compound_col,
            group_col="Metadata_Group",
            negcon_col="Metadata_negcon",
            target_col="target",
            k_percentages=k_percentages,
            similarity_directions=similarity_directions,
            include_genetic_pairs=args.include_genetic_pairs,
        )

        # Add CMR summary to metrics
        if cmr_results["summary"]:
            metrics["CMR_summary"] = cmr_results["summary"]
            # Update metrics.json with CMR results
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"  Updated: {metrics_path}")
    elif args.skip_cross_modality:
        print("\nSkipping Cross-Modality Retrieval evaluation (--skip-cross-modality).")

    print("\nDone!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
