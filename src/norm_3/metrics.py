"""Metrics for normalization quality evaluation.

This module provides:
- Phenotypic Activity (PA): compound replicate retrieval
- Phenotypic Consistency (PC): target-level retrieval

Uses copairs library (CPU) for the actual metrics computation.
GPU is used for preprocessing only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from norm_3.io import get_numeric_features, infer_columns


def calculate_phenotypic_activity(
    df: pl.DataFrame,
    features: list[str],
    null_size: int = 10_000,
    p_threshold: float = 0.05,
    seed: int = 0,
    compound_col: str = "Metadata_pert_iname",
    negcon_col: str = "Metadata_negcon",
    negcon_value: str = "DMSO",
    batch_col: str = "Metadata_Plate",
    group_col: str = "Metadata_Group",
    distance: str = "cosine",
) -> dict[str, Any]:
    """Calculate Phenotypic Activity (compound replicate retrieval).

    Measures how well replicates of the same compound cluster together
    compared to random compounds.

    Args:
        df: Normalized profiles with metadata
        features: Feature column names
        null_size: Size of null distribution
        p_threshold: Significance threshold
        seed: Random seed
        compound_col: Column containing compound/perturbation identifier
        negcon_col: Column containing negative control flag
        negcon_value: Value in compound_col that represents negative controls
        batch_col: Column containing batch/plate identifier
        group_col: Column containing group identifier for per-group statistics

    Returns:
        Dictionary with metrics including per-group summary
    """
    import pandas as pd
    from copairs import map as copairs_map
    from copairs.matching import assign_reference_index

    df_pd = df.to_pandas()

    # Check if group column exists for per-group statistics
    has_groups = group_col in df_pd.columns

    if has_groups:
        pos_sameby = [compound_col, group_col, "Metadata_reference_index"]
        neg_sameby = [batch_col, group_col]
    else:
        pos_sameby = [compound_col, "Metadata_reference_index"]
        neg_sameby = [batch_col]

    pos_diffby = []
    neg_diffby = [compound_col, negcon_col, "Metadata_reference_index"]

    # Split by group and compute AP separately per group.
    # Since group_col is in both pos_sameby and neg_sameby, copairs only
    # compares within groups. Splitting first avoids the overhead of
    # processing all 163K rows in a single call.
    if has_groups:
        groups = sorted(df_pd[group_col].unique())
        activity_ap_parts = []
        for grp in groups:
            grp_df = df_pd[df_pd[group_col] == grp].copy().reset_index(drop=True)
            if negcon_col in grp_df.columns:
                grp_df = assign_reference_index(
                    grp_df, f"{negcon_col} == True",
                    reference_col="Metadata_reference_index", default_value=-1,
                )
            else:
                grp_df = assign_reference_index(
                    grp_df, f"{compound_col} == '{negcon_value}'",
                    reference_col="Metadata_reference_index", default_value=-1,
                )
            grp_meta = grp_df.filter(regex="^Metadata")
            grp_profiles = grp_df[features].values
            print(f"  PA {grp}: {len(grp_df)} rows")
            grp_ap = copairs_map.average_precision(
                grp_meta, grp_profiles, pos_sameby, pos_diffby, neg_sameby, neg_diffby,
                distance=distance,
            )
            activity_ap_parts.append(grp_ap)
        activity_ap = pd.concat(activity_ap_parts, ignore_index=True)
    else:
        if negcon_col in df_pd.columns:
            df_pd = assign_reference_index(
                df_pd, f"{negcon_col} == True",
                reference_col="Metadata_reference_index", default_value=-1,
            )
        else:
            df_pd = assign_reference_index(
                df_pd, f"{compound_col} == '{negcon_value}'",
                reference_col="Metadata_reference_index", default_value=-1,
            )
        metadata = df_pd.filter(regex="^Metadata")
        profiles = df_pd[features].values
        activity_ap = copairs_map.average_precision(
            metadata, profiles, pos_sameby, pos_diffby, neg_sameby, neg_diffby,
            distance=distance,
        )

    # Filter out negative controls
    if negcon_col in activity_ap.columns:
        activity_ap = activity_ap.query(f"{negcon_col} == False").copy()
    else:
        activity_ap = activity_ap.query(f"{compound_col} != '{negcon_value}'").copy()

    # Calculate replicate counts per compound (or compound+group)
    replicate_counts = activity_ap.groupby(pos_sameby).size()

    # Calculate mean average precision
    activity_map = copairs_map.mean_average_precision(
        activity_ap, pos_sameby, null_size=null_size, threshold=p_threshold, seed=seed
    ).copy()
    activity_map["below_corrected_p"] = activity_map["corrected_p_value"] < p_threshold

    # Merge replicate counts into activity_map
    activity_map = activity_map.merge(
        replicate_counts.rename("n_replicate_pairs"),
        on=pos_sameby,
        how="left",
    )

    pct_compounds_active = (
        activity_map["below_corrected_p"].sum() / len(activity_map)
    ) * 100

    # Calculate per-group summary if groups exist
    group_summary = None
    if has_groups and group_col in activity_map.columns:
        group_summary = activity_map.groupby(group_col).agg(
            pct_active=("below_corrected_p", "mean"),
            num_active=("below_corrected_p", "sum"),
            mean_normalized_average_precision=("mean_normalized_average_precision", "mean"),
            median_normalized_average_precision=("mean_normalized_average_precision", "median"),
            mean_normalized_average_precision_clipped=("mean_normalized_average_precision", lambda s: float(s.clip(lower=0).mean())),
            mean_n_replicates=("n_replicate_pairs", "mean"),
            median_n_replicates=("n_replicate_pairs", "median"),
            n_unique_compounds=(compound_col, "nunique"),
        ).reset_index()
        group_summary["pct_active"] *= 100  # Convert to percentage

    # Calculate overall mean normalized average precision
    mean_nap = float(activity_map["mean_normalized_average_precision"].mean())
    median_nap = float(activity_map["mean_normalized_average_precision"].median())
    mean_nap_clipped = float(activity_map["mean_normalized_average_precision"].clip(lower=0).mean())

    return {
        "activity_ap": activity_ap,
        "activity_map": activity_map,
        "group_summary": group_summary,
        "pct_compounds_active": float(pct_compounds_active),
        "n_compounds": int(len(activity_map)),
        "mean_normalized_average_precision": mean_nap,
        "median_normalized_average_precision": median_nap,
        "mean_normalized_average_precision_clipped": mean_nap_clipped,
    }


def _filter_targets_by_compound_count(
    df_consensus,
    min_compounds_per_target: int = 3,
    max_targets_per_compound: int = 50,
    exclude_unknown: bool = True,
    compound_col: str = "Metadata_pert_iname",
    negcon_col: str = "Metadata_negcon",
):
    """Filter targets based on minimum number of compounds."""
    # Filter out negative controls
    df_consensus = df_consensus[df_consensus[negcon_col] == False].copy()

    # Filter out promiscuous compounds
    if max_targets_per_compound is not None:
        target_counts_per_compound = df_consensus["Metadata_target"].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )
        n_promiscuous = (target_counts_per_compound > max_targets_per_compound).sum()
        if n_promiscuous > 0:
            print(f"  Filtering out {n_promiscuous} promiscuous compounds")
        df_consensus = df_consensus[
            target_counts_per_compound <= max_targets_per_compound
        ].copy()

    # Explode targets
    df_exploded = df_consensus.explode("Metadata_target")

    if exclude_unknown:
        df_exploded = df_exploded[df_exploded["Metadata_target"] != "unknown"].copy()

    # Count unique compounds per target
    target_counts = df_exploded.groupby("Metadata_target")[
        compound_col
    ].nunique()
    target_counts = target_counts.sort_values(ascending=False)

    valid_targets = target_counts[
        target_counts >= min_compounds_per_target
    ].index.tolist()
    print(f"  Using min_compounds_per_target={min_compounds_per_target}: {len(valid_targets)} targets")

    def has_valid_target(target_list):
        if not isinstance(target_list, list):
            return False
        return any(t in valid_targets for t in target_list)

    df_consensus = df_consensus[
        df_consensus["Metadata_target"].apply(has_valid_target)
    ].copy()

    df_consensus["Metadata_target"] = df_consensus["Metadata_target"].apply(
        lambda targets: [t for t in targets if t in valid_targets]
        if isinstance(targets, list)
        else []
    )

    print(f"  Compounds remaining after filtering: {len(df_consensus)}")
    return df_consensus


def calculate_phenotypic_consistency(
    df: pl.DataFrame,
    features: list[str],
    null_size: int = 10_000,
    p_threshold: float = 0.05,
    seed: int = 0,
    min_compounds_per_target: int = 3,
    max_targets_per_compound: int = 50,
    compound_col: str = "Metadata_pert_iname",
    target_col: str = "Metadata_target_list",
    negcon_col: str = "Metadata_negcon",
    group_col: str = "Metadata_Group",
    pc_groups: list[str] | None = None,
    distance: str = "cosine",
) -> dict[str, Any]:
    """Calculate Phenotypic Consistency (target-level retrieval).

    Measures how well compounds targeting the same protein cluster together.

    Args:
        df: Normalized profiles with metadata
        features: Feature column names
        null_size: Size of null distribution
        p_threshold: Significance threshold
        seed: Random seed
        min_compounds_per_target: Minimum compounds per target
        max_targets_per_compound: Maximum targets per compound
        compound_col: Column containing compound identifier
        target_col: Column containing target identifier (pipe-separated for multiple)
        negcon_col: Column containing negative control flag
        group_col: Column containing group identifier for per-group statistics
        pc_groups: If set, only compute PC for these groups (e.g. ["group_high", "group_low"])

    Returns:
        Dictionary with metrics including per-group summary
    """
    from copairs import map as copairs_map

    # Filter to specified groups before computing PC
    if pc_groups and group_col in df.columns:
        before = len(df)
        df = df.filter(pl.col(group_col).is_in(pc_groups))
        print(f"  PC: filtered to groups {pc_groups}: {before} -> {len(df)} rows")

    df_pd = df.to_pandas()

    # Check if group column exists for per-group statistics
    has_groups = group_col in df_pd.columns

    # Fill null target values with "unknown"
    if target_col in df_pd.columns:
        df_pd[target_col] = df_pd[target_col].fillna("unknown")

    # Get consensus profiles per compound (and per group if groups exist)
    if has_groups:
        groupby_cols = [compound_col, target_col, negcon_col, group_col]
    else:
        groupby_cols = [compound_col, target_col, negcon_col]

    df_consensus = (
        df_pd.groupby(
            groupby_cols,
            as_index=False,
            observed=True,
        )[features]
        .median()
        .copy()
    )
    df_consensus["Metadata_target"] = df_consensus[target_col].str.split("|")

    df_consensus = _filter_targets_by_compound_count(
        df_consensus,
        min_compounds_per_target=min_compounds_per_target,
        max_targets_per_compound=max_targets_per_compound,
        compound_col=compound_col,
        negcon_col=negcon_col,
    )

    if len(df_consensus) < 2:
        print(f"  Warning: Not enough compounds ({len(df_consensus)}) for PC")
        return {
            "target_consistency": None,
            "pct_targets_active": 0.0,
            "n_targets_active": 0,
            "n_targets_total": 0,
            "group_summary": None,
        }

    # Include group in sameby columns if groups exist
    if has_groups:
        pos_sameby_target = ["Metadata_target", group_col]
        neg_sameby_target = [group_col]
    else:
        pos_sameby_target = ["Metadata_target"]
        neg_sameby_target = []

    pos_diffby_target = []
    neg_diffby_target = ["Metadata_target"]

    try:
        # Split by group for faster copairs computation
        if has_groups:
            import pandas as pd
            groups = sorted(df_consensus[group_col].unique())
            target_ap_parts = []
            for grp in groups:
                grp_df = df_consensus[df_consensus[group_col] == grp].copy().reset_index(drop=True)
                if len(grp_df) < 2:
                    continue
                grp_meta = grp_df.filter(regex="^Metadata")
                grp_profiles = grp_df[features].values
                print(f"  PC {grp}: {len(grp_df)} rows")
                grp_ap = copairs_map.multilabel.average_precision(
                    grp_meta, grp_profiles,
                    pos_sameby_target, pos_diffby_target,
                    neg_sameby_target, neg_diffby_target,
                    multilabel_col="Metadata_target",
                    distance=distance,
                )
                target_ap_parts.append(grp_ap)
            target_ap = pd.concat(target_ap_parts, ignore_index=True)
        else:
            metadata = df_consensus.filter(regex="^Metadata")
            profiles = df_consensus[features].values
            target_ap = copairs_map.multilabel.average_precision(
                metadata, profiles,
                pos_sameby_target, pos_diffby_target,
                neg_sameby_target, neg_diffby_target,
                multilabel_col="Metadata_target",
                distance=distance,
            )

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

        # Calculate per-group summary if groups exist
        group_summary = None
        if has_groups and group_col in target_map.columns:
            group_summary = target_map.groupby(group_col).agg(
                pct_active=("below_corrected_p", "mean"),
                num_active=("below_corrected_p", "sum"),
                mean_normalized_average_precision=("mean_normalized_average_precision", "mean"),
                median_normalized_average_precision=("mean_normalized_average_precision", "median"),
                mean_normalized_average_precision_clipped=("mean_normalized_average_precision", lambda s: float(s.clip(lower=0).mean())),
                n_targets=("Metadata_target", "nunique"),
            ).reset_index()
            group_summary["pct_active"] *= 100  # Convert to percentage

        # Calculate overall mean normalized average precision
        mean_nap = float(target_map["mean_normalized_average_precision"].mean())
        median_nap = float(target_map["mean_normalized_average_precision"].median())
        mean_nap_clipped = float(target_map["mean_normalized_average_precision"].clip(lower=0).mean())

        return {
            "target_consistency": target_map,
            "pct_targets_active": float(pct_targets_active),
            "n_targets_active": int(n_targets_active),
            "n_targets_total": int(n_targets_total),
            "group_summary": group_summary,
            "mean_normalized_average_precision": mean_nap,
            "median_normalized_average_precision": median_nap,
            "mean_normalized_average_precision_clipped": mean_nap_clipped,
        }
    except Exception as e:
        print(f"Warning: Target consistency failed: {e}")
        return {
            "target_consistency": None,
            "pct_targets_active": 0.0,
            "n_targets_active": 0,
            "n_targets_total": 0,
            "group_summary": None,
        }


def calculate_batch_effects(
    df: pl.DataFrame,
    features: list[str],
    null_size: int = 10_000,
    p_threshold: float = 0.05,
    seed: int = 0,
    n_random_groups: int = 15,
    compound_col: str = "Metadata_pert_iname",
    negcon_col: str = "Metadata_negcon",
    batch_col: str = "Metadata_Plate",
    well_col: str = "Metadata_Well",
    group_col: str = "Metadata_Group",
    distance: str = "cosine",
) -> dict[str, Any]:
    """Calculate batch effect metrics using random sampling for speed.

    Based on scripts/analyze_batch_effects.py::calculate_batch_effects_fast().

    Two metrics are computed (both on treatments only, negcons excluded):
    1. Well Position Effect: Do wells at the same position cluster across plates?
       Positive: same group + same well + different plate + different compound
       Negative: same group + same plate + same _random_index + different well + different compound
    2. Plate Batch Effect: Are plates distinguishable?
       Positive: same group + same plate + same _random_index_pos + different compound
       Negative: same group + same _random_index_neg + different plate + different well + different compound

    Args:
        df: Profiles with metadata
        features: Feature column names
        null_size: Size of null distribution
        p_threshold: Significance threshold
        seed: Random seed
        n_random_groups: Max random groups for well effect subsampling
        compound_col: Column containing compound identifier
        negcon_col: Column containing negative control flag
        batch_col: Column containing plate identifier
        well_col: Column containing well identifier
        group_col: Column containing group identifier

    Returns:
        Dictionary with well_position_effect and plate_batch_effect results
    """
    import pandas as pd
    from copairs import map as copairs_map

    results = {}
    rng = np.random.default_rng(seed)
    df_pd = df.to_pandas()

    if well_col not in df_pd.columns:
        print(f"  Warning: {well_col} not found, skipping batch effect analysis")
        return {"well_position_effect": None, "plate_batch_effect": None}

    has_groups = group_col in df_pd.columns

    # Filter out negative controls for both analyses
    if negcon_col in df_pd.columns:
        df_treatments = df_pd[df_pd[negcon_col] == False].copy()
    else:
        df_treatments = df_pd.copy()

    if len(df_treatments) < 10:
        print(f"    Not enough treatment samples ({len(df_treatments)})")
        _empty = {"pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0}
        return {"well_position_effect": _empty, "plate_batch_effect": _empty}

    # Compute adaptive plate effect random indices on df_treatments
    n_plates = df_treatments[batch_col].nunique()
    n_groups = df_treatments[group_col].nunique() if has_groups else 1
    rows_per_plate_group = len(df_treatments) / max(n_plates * n_groups, 1)
    rows_per_group = len(df_treatments) / max(n_groups, 1)

    n_pos_groups = max(2, int(rows_per_plate_group / 15))
    n_neg_groups = max(2, int(rows_per_group / 25))

    df_treatments["_random_index_pos"] = rng.integers(1, n_pos_groups + 1, size=len(df_treatments))
    df_treatments["_random_index_neg"] = rng.integers(1, n_neg_groups + 1, size=len(df_treatments))

    # =========================================
    # 1. Well Position Effect
    # =========================================
    # Positive: same well + different plate + different compound
    # Negative: same plate + same _random_index + different well + different compound
    # When groups exist, split by group first to avoid O(n^2) on the full dataset.
    print("  Calculating Well Position Effect...")
    try:
        pos_sameby_well = [well_col]
        neg_sameby_well = [batch_col, "_random_index"]
        if has_groups:
            pos_sameby_well = [group_col] + pos_sameby_well
            neg_sameby_well = [group_col] + neg_sameby_well
        pos_diffby_well = [batch_col, compound_col]
        neg_diffby_well = [well_col, compound_col]

        if has_groups:
            import pandas as pd
            groups = sorted(df_treatments[group_col].unique())
            well_ap_parts = []
            for grp in groups:
                grp_df = df_treatments[df_treatments[group_col] == grp].copy().reset_index(drop=True)
                well_plate_counts = grp_df.groupby(well_col)[batch_col].nunique()
                valid_wells = well_plate_counts[well_plate_counts >= 2].index.tolist()
                if len(valid_wells) < 2:
                    continue
                grp_well = grp_df[grp_df[well_col].isin(valid_wells)].copy()
                n_well_plates = grp_well[batch_col].nunique()
                rows_per_pg = len(grp_well) / max(n_well_plates, 1)
                n_ri = max(2, int(rows_per_pg / 15))
                grp_well["_random_index"] = rng.integers(1, n_ri + 1, size=len(grp_well))
                print(f"    Well {grp}: {len(grp_well)} samples, {len(valid_wells)} wells, {n_well_plates} plates")
                grp_meta = grp_well.filter(regex="^Metadata|^_random_index")
                grp_profiles = grp_well[features].values
                grp_ap = copairs_map.average_precision(
                    grp_meta, grp_profiles,
                    pos_sameby_well, pos_diffby_well,
                    neg_sameby_well, neg_diffby_well,
                    distance=distance,
                )
                well_ap_parts.append(grp_ap)

            if well_ap_parts:
                well_ap = pd.concat(well_ap_parts, ignore_index=True)
            else:
                well_ap = pd.DataFrame()
        else:
            well_plate_counts = df_treatments.groupby(well_col)[batch_col].nunique()
            valid_wells = well_plate_counts[well_plate_counts >= 2].index.tolist()
            if len(valid_wells) >= 2:
                df_well = df_treatments[df_treatments[well_col].isin(valid_wells)].copy()
                n_well_plates = df_well[batch_col].nunique()
                rows_per_pg = len(df_well) / max(n_well_plates, 1)
                n_ri = max(2, int(rows_per_pg / 15))
                df_well["_random_index"] = rng.integers(1, n_ri + 1, size=len(df_well))
                print(f"    {len(df_well)} samples, {len(valid_wells)} wells, {n_well_plates} plates")
                metadata_well = df_well.filter(regex="^Metadata|^_random_index")
                profiles_well = df_well[features].values
                well_ap = copairs_map.average_precision(
                    metadata_well, profiles_well,
                    pos_sameby_well, pos_diffby_well,
                    neg_sameby_well, neg_diffby_well,
                    distance=distance,
                )
            else:
                well_ap = pd.DataFrame()

        if len(well_ap) > 0:
            well_map = copairs_map.mean_average_precision(
                well_ap, pos_sameby_well,
                null_size=null_size, threshold=p_threshold, seed=seed
            )
            well_map["below_corrected_p"] = well_map["corrected_p_value"] < p_threshold

            pct_active = (well_map["below_corrected_p"].sum() / len(well_map)) * 100 if len(well_map) > 0 else 0
            mean_map_val = well_map["mean_average_precision"].mean() if len(well_map) > 0 else 0
            mean_nap_val = well_map["mean_normalized_average_precision"].mean() if len(well_map) > 0 else 0

            result_dict = {
                "pct_active": float(pct_active),
                "n_active": int(well_map["below_corrected_p"].sum()),
                "n_total": int(len(well_map)),
                "mean_map": float(mean_map_val),
                "mean_nap": float(mean_nap_val),
            }

            if has_groups and group_col in well_map.columns:
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
                        "mean_nap": float(grp_data["mean_normalized_average_precision"].mean()),
                    }
                result_dict["per_group"] = per_group_stats

            results["well_position_effect"] = result_dict
            print(f"    Well Position Effect: {pct_active:.2f}%")
            if has_groups:
                for grp, stats in result_dict.get("per_group", {}).items():
                    print(f"      {grp}: {stats['pct_active']:.2f}%")
        else:
            results["well_position_effect"] = {
                "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0
            }

    except Exception as e:
        print(f"    Warning: Well position effect failed: {e}")
        results["well_position_effect"] = {
            "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0, "error": str(e)
        }

    # =========================================
    # 2. Plate Batch Effect (treatments, random sampling)
    # =========================================
    # Positive: same plate + same _random_index_pos + different compound
    # Negative: same _random_index_neg + different plate + different well + different compound
    # When groups exist, split by group first for performance.
    print("  Calculating Plate Batch Effect...")
    try:
        if n_plates < 2:
            print(f"    Not enough plates ({n_plates})")
            results["plate_batch_effect"] = {
                "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0
            }
        else:
            if has_groups:
                pos_sameby_plate = [group_col, batch_col, "_random_index_pos"]
                neg_sameby_plate = [group_col, "_random_index_neg"]
            else:
                pos_sameby_plate = [batch_col, "_random_index_pos"]
                neg_sameby_plate = ["_random_index_neg"]

            pos_diffby_plate = [compound_col]
            neg_diffby_plate = [batch_col, well_col, compound_col]

            if has_groups:
                import pandas as pd
                groups = sorted(df_treatments[group_col].unique())
                plate_ap_parts = []
                for grp in groups:
                    grp_df = df_treatments[df_treatments[group_col] == grp].copy().reset_index(drop=True)
                    grp_n_plates = grp_df[batch_col].nunique()
                    if grp_n_plates < 2:
                        continue
                    grp_rows_per_plate = len(grp_df) / max(grp_n_plates, 1)
                    grp_n_pos = max(2, int(grp_rows_per_plate / 15))
                    grp_n_neg = max(2, int(len(grp_df) / 25))
                    grp_df["_random_index_pos"] = rng.integers(1, grp_n_pos + 1, size=len(grp_df))
                    grp_df["_random_index_neg"] = rng.integers(1, grp_n_neg + 1, size=len(grp_df))
                    print(f"    Plate {grp}: {len(grp_df)} treatments, {grp_n_plates} plates")
                    grp_meta = grp_df.filter(regex="^Metadata|^_random_index")
                    grp_profiles = grp_df[features].values
                    grp_ap = copairs_map.average_precision(
                        grp_meta, grp_profiles,
                        pos_sameby_plate, pos_diffby_plate,
                        neg_sameby_plate, neg_diffby_plate,
                        distance=distance,
                    )
                    plate_ap_parts.append(grp_ap)

                if plate_ap_parts:
                    plate_ap = pd.concat(plate_ap_parts, ignore_index=True)
                else:
                    plate_ap = pd.DataFrame()
            else:
                print(f"    {len(df_treatments)} treatments, {n_plates} plates, pos_groups={n_pos_groups}, neg_groups={n_neg_groups}")
                metadata_treat = df_treatments.filter(regex="^Metadata|^_random_index")
                profiles_treat = df_treatments[features].values
                plate_ap = copairs_map.average_precision(
                    metadata_treat, profiles_treat,
                    pos_sameby_plate, pos_diffby_plate,
                    neg_sameby_plate, neg_diffby_plate,
                    distance=distance,
                )

            if len(plate_ap) > 0:
                map_groupby = [group_col, batch_col] if has_groups else [batch_col]
                plate_map = copairs_map.mean_average_precision(
                    plate_ap, map_groupby,
                    null_size=null_size, threshold=p_threshold, seed=seed
                )
                plate_map["below_corrected_p"] = plate_map["corrected_p_value"] < p_threshold

                pct_active = (plate_map["below_corrected_p"].sum() / len(plate_map)) * 100 if len(plate_map) > 0 else 0
                mean_map_val = plate_map["mean_average_precision"].mean() if len(plate_map) > 0 else 0
                mean_nap_val = plate_map["mean_normalized_average_precision"].mean() if len(plate_map) > 0 else 0

                result_dict = {
                    "pct_active": float(pct_active),
                    "n_active": int(plate_map["below_corrected_p"].sum()),
                    "n_total": int(len(plate_map)),
                    "mean_map": float(mean_map_val),
                    "mean_nap": float(mean_nap_val),
                    "n_plates": int(n_plates),
                    "n_samples": int(len(df_treatments)),
                }

                if has_groups and group_col in plate_map.columns:
                    per_group_stats = {}
                    for grp in plate_map[group_col].unique():
                        grp_data = plate_map[plate_map[group_col] == grp]
                        grp_active = grp_data["below_corrected_p"].sum()
                        grp_total = len(grp_data)
                        grp_pct = (grp_active / grp_total * 100) if grp_total > 0 else 0
                        per_group_stats[grp] = {
                            "pct_active": float(grp_pct),
                            "n_active": int(grp_active),
                            "n_total": int(grp_total),
                            "mean_map": float(grp_data["mean_average_precision"].mean()),
                            "mean_nap": float(grp_data["mean_normalized_average_precision"].mean()),
                        }
                    result_dict["per_group"] = per_group_stats

                results["plate_batch_effect"] = result_dict
                print(f"    Plate Batch Effect: {pct_active:.2f}% ({n_plates} plates)")
            else:
                results["plate_batch_effect"] = {
                    "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0
                }

    except Exception as e:
        error_msg = str(e)
        if "No data left" in error_msg:
            print("    Warning: Plate batch effect failed: not enough valid pairs")
        else:
            print(f"    Warning: Plate batch effect failed: {e}")
        results["plate_batch_effect"] = {
            "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0, "error": error_msg
        }

    return results


def evaluate_all(
    df: pl.DataFrame,
    features: list[str],
    output_dir: Path | str | None = None,
    skip_visualization: bool = False,
    skip_umap: bool = False,
    n_top_compounds: int = 20,
    min_compounds_per_target: int = 3,
    compound_col: str = "Metadata_pert_iname",
    target_col: str = "Metadata_target_list",
    negcon_col: str = "Metadata_negcon",
    batch_col: str = "Metadata_Plate",
    group_col: str = "Metadata_Group",
    pc_groups: list[str] | None = None,
    skip_batch_effects: bool = True,
    well_col: str = "Metadata_Well",
    distance: str | None = None,
) -> dict[str, Any]:
    """Run all metrics and optionally save results.

    Args:
        df: Normalized profiles
        features: Feature column names
        output_dir: Directory to save results (None = don't save)
        skip_visualization: Skip visualization generation
        skip_umap: Skip UMAP in visualization
        n_top_compounds: Number of compounds to highlight
        min_compounds_per_target: Minimum compounds required per target for PC
        compound_col: Column for compound identifier
        target_col: Column for target identifier
        negcon_col: Boolean column for negative control flag
        batch_col: Column for batch/plate identifier
        group_col: Column for group identifier (for per-group PA and PC statistics)
        skip_batch_effects: Skip batch effect calculation (default True for backward compat)
        well_col: Column for well position identifier
        pc_groups: If set, only compute PC for these groups (e.g. ["group_high", "group_low"])
        distance: Distance metric for copairs ("cosine", "euclidean", etc.).
            If None, auto-selects: "euclidean" for <=2 features, "cosine" otherwise.

    Returns:
        Dictionary with all metrics including per-group summaries
    """
    # Auto-select distance metric: cosine degenerates to sign(a)*sign(b) with
    # <=2 features (especially when highly correlated), so use euclidean instead.
    if distance is None:
        distance = "euclidean" if len(features) <= 2 else "cosine"
        print(f"  Distance metric: {distance} (auto, {len(features)} features)")
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    results = {}
    pa = {}  # Initialized here so PC_replicable can check it even if PA errors

    # Phenotypic Activity
    try:
        pa = calculate_phenotypic_activity(
            df, features,
            compound_col=compound_col,
            negcon_col=negcon_col,
            batch_col=batch_col,
            group_col=group_col,
            distance=distance,
        )
        results["PA"] = pa["pct_compounds_active"]
        results["n_compounds"] = pa["n_compounds"]
        results["PA_mean_nap"] = pa["mean_normalized_average_precision"]
        results["PA_median_nap"] = pa["median_normalized_average_precision"]
        results["PA_mean_nap_clipped"] = pa.get("mean_normalized_average_precision_clipped", 0.0)
        print(f"  PA: {pa['pct_compounds_active']:.2f}%")
        print(f"  PA mean NAP: {pa['mean_normalized_average_precision']:.4f}")
        print(f"  PA mean NAP (clipped): {results['PA_mean_nap_clipped']:.4f}")

        # Add per-group PA summary to results
        if pa.get("group_summary") is not None:
            results["PA_group_summary"] = pa["group_summary"].set_index(group_col).to_dict(orient="index")
            for group_name, group_stats in results["PA_group_summary"].items():
                print(f"    {group_name}: {group_stats['pct_active']:.2f}%")

        if output_dir is not None:
            if pa.get("activity_ap") is not None and len(pa["activity_ap"]) > 0:
                pa["activity_ap"].to_csv(output_dir / "phenotypic_activity_per_compound.csv", index=False)
            if pa.get("activity_map") is not None and len(pa["activity_map"]) > 0:
                pa["activity_map"].to_csv(output_dir / "phenotypic_activity_map.csv", index=False)
            if pa.get("group_summary") is not None and len(pa["group_summary"]) > 0:
                pa["group_summary"].to_csv(output_dir / "phenotypic_activity_group_summary.csv", index=False)
    except Exception as e:
        print(f"  PA ERROR: {e}")
        results["PA"] = 0.0
        results["n_compounds"] = 0

    # Phenotypic Consistency
    try:
        pc = calculate_phenotypic_consistency(
            df, features,
            min_compounds_per_target=min_compounds_per_target,
            compound_col=compound_col,
            target_col=target_col,
            negcon_col=negcon_col,
            group_col=group_col,
            pc_groups=pc_groups,
            distance=distance,
        )
        results["PC"] = pc["pct_targets_active"]
        results["n_targets_active"] = pc["n_targets_active"]
        results["n_targets_total"] = pc["n_targets_total"]
        results["PC_mean_nap"] = pc.get("mean_normalized_average_precision", 0.0)
        results["PC_median_nap"] = pc.get("median_normalized_average_precision", 0.0)
        results["PC_mean_nap_clipped"] = pc.get("mean_normalized_average_precision_clipped", 0.0)
        print(f"  PC: {pc['pct_targets_active']:.1f}%")
        if pc.get("mean_normalized_average_precision") is not None:
            print(f"  PC mean NAP: {pc['mean_normalized_average_precision']:.4f}")
            print(f"  PC mean NAP (clipped): {results['PC_mean_nap_clipped']:.4f}")

        # Add per-group PC summary to results
        if pc.get("group_summary") is not None:
            results["PC_group_summary"] = pc["group_summary"].set_index(group_col).to_dict(orient="index")
            for group_name, group_stats in results["PC_group_summary"].items():
                print(f"    {group_name}: {group_stats['pct_active']:.2f}%")

        if output_dir is not None:
            if pc.get("target_consistency") is not None:
                pc["target_consistency"].to_csv(output_dir / "phenotypic_consistency_per_target.csv", index=False)
            if pc.get("group_summary") is not None and len(pc["group_summary"]) > 0:
                pc["group_summary"].to_csv(output_dir / "phenotypic_consistency_group_summary.csv", index=False)
    except Exception as e:
        print(f"  PC ERROR: {e}")
        results["PC"] = 0.0
        results["n_targets_active"] = 0
        results["n_targets_total"] = 0
        results["PC_mean_nap"] = 0.0
        results["PC_median_nap"] = 0.0
        results["PC_mean_nap_clipped"] = 0.0

    # Phenotypic Consistency on PA-replicable compounds only (Chandrasekaran-style gating)
    _pc_rep_defaults = {
        "PC_replicable": 0.0,
        "PC_replicable_n_targets_active": 0,
        "PC_replicable_n_targets_total": 0,
        "PC_replicable_mean_nap": 0.0,
        "PC_replicable_median_nap": 0.0,
        "PC_replicable_n_compounds": 0,
    }
    try:
        if pa.get("activity_map") is not None and len(pa["activity_map"]) > 0:
            replicable = pa["activity_map"].query("below_corrected_p == True")[compound_col].tolist()
            if len(replicable) >= 2:
                df_replicable = df.filter(pl.col(compound_col).is_in(replicable))
                print(f"  PC_replicable: {len(replicable)} PA-significant compounds")
                pc_rep = calculate_phenotypic_consistency(
                    df_replicable, features,
                    min_compounds_per_target=min_compounds_per_target,
                    compound_col=compound_col,
                    target_col=target_col,
                    negcon_col=negcon_col,
                    group_col=group_col,
                    pc_groups=pc_groups,
                    distance=distance,
                )
                results["PC_replicable"] = pc_rep["pct_targets_active"]
                results["PC_replicable_n_targets_active"] = pc_rep["n_targets_active"]
                results["PC_replicable_n_targets_total"] = pc_rep["n_targets_total"]
                results["PC_replicable_mean_nap"] = pc_rep.get("mean_normalized_average_precision", 0.0)
                results["PC_replicable_median_nap"] = pc_rep.get("median_normalized_average_precision", 0.0)
                results["PC_replicable_n_compounds"] = len(replicable)
                print(f"  PC_replicable: {pc_rep['pct_targets_active']:.1f}% ({pc_rep['n_targets_active']}/{pc_rep['n_targets_total']} targets)")
                if pc_rep.get("mean_normalized_average_precision") is not None:
                    print(f"  PC_replicable mean NAP: {pc_rep['mean_normalized_average_precision']:.4f}")

                if pc_rep.get("group_summary") is not None:
                    results["PC_replicable_group_summary"] = pc_rep["group_summary"].set_index(group_col).to_dict(orient="index")
                    for group_name, group_stats in results["PC_replicable_group_summary"].items():
                        print(f"    {group_name}: {group_stats['pct_active']:.2f}%")

                if output_dir is not None:
                    if pc_rep.get("target_consistency") is not None:
                        pc_rep["target_consistency"].to_csv(
                            output_dir / "phenotypic_consistency_replicable_per_target.csv", index=False
                        )
                    if pc_rep.get("group_summary") is not None and len(pc_rep["group_summary"]) > 0:
                        pc_rep["group_summary"].to_csv(
                            output_dir / "phenotypic_consistency_replicable_group_summary.csv", index=False
                        )
            else:
                print(f"  PC_replicable: skipped (only {len(replicable)} PA-significant compounds)")
                results.update(_pc_rep_defaults)
                results["PC_replicable_n_compounds"] = len(replicable)
        else:
            results.update(_pc_rep_defaults)
    except Exception as e:
        print(f"  PC_replicable ERROR: {e}")
        results.update(_pc_rep_defaults)

    # Add TVN state
    from norm_3.core import get_tvn_state
    tvn_ill_conditioned, tvn_max_condition_number = get_tvn_state()
    results["tvn_ill_conditioned"] = tvn_ill_conditioned
    results["tvn_max_condition_number"] = float(tvn_max_condition_number) if tvn_max_condition_number > 0 else None
    if tvn_ill_conditioned:
        print(f"  WARNING: TVN encountered ill-conditioned matrix (condition number: {tvn_max_condition_number:.2e})")

    # Add spherize truncation state
    from norm_3.core import get_spherize_truncation_state
    trunc_state = get_spherize_truncation_state()
    results.update(trunc_state)
    if trunc_state["spherize_truncation_k"] is not None:
        print(f"  Spherize truncation: method={trunc_state['spherize_truncation_method']}, "
              f"k={trunc_state['spherize_truncation_k']}/{trunc_state['spherize_truncation_input_dims']} "
              f"({trunc_state['spherize_truncation_k_pct']:.1f}%), "
              f"variance_removed={trunc_state['spherize_truncation_variance_removed']*100:.1f}%")

    # Compute PCA variance (PC1 and PC2 explained variance)
    try:
        from norm_3.core import PCATransform
        from norm_3.utils import to_gpu, to_cpu

        X = df.select(features).to_numpy()
        if np.isnan(X).any() or np.isinf(X).any():
            print("  WARNING: NaN/Inf detected in features for PCA computation")
            results["PC1_variance"] = None
            results["PC2_variance"] = None
        else:
            X_gpu = to_gpu(X)
            pca = PCATransform(n_components=2)
            pca.fit(X_gpu)
            results["PC1_variance"] = float(pca.explained_variance_ratio_[0])
            results["PC2_variance"] = float(pca.explained_variance_ratio_[1])
            print(f"  PC1 variance: {results['PC1_variance']*100:.2f}%")
            print(f"  PC2 variance: {results['PC2_variance']*100:.2f}%")
    except Exception as e:
        print(f"  PCA variance ERROR: {e}")
        results["PC1_variance"] = None
        results["PC2_variance"] = None

    # Batch effects (well position + plate)
    _batch_defaults = {
        "well_effect_pct": None,
        "well_effect_mean_nap": None,
        "well_effect_n_active": None,
        "well_effect_n_total": None,
        "plate_effect_pct": None,
        "plate_effect_mean_nap": None,
        "plate_effect_n_active": None,
        "plate_effect_n_total": None,
    }
    if skip_batch_effects:
        print("  Batch effects: skipped")
        results.update(_batch_defaults)
    else:
        try:
            batch_results = calculate_batch_effects(
                df, features,
                null_size=10_000,
                p_threshold=0.05,
                seed=0,
                compound_col=compound_col,
                negcon_col=negcon_col,
                batch_col=batch_col,
                well_col=well_col,
                group_col=group_col,
                distance=distance,
            )

            well_effect = batch_results.get("well_position_effect") or {}
            plate_effect = batch_results.get("plate_batch_effect") or {}

            results["well_effect_pct"] = well_effect.get("pct_active")
            results["well_effect_mean_nap"] = well_effect.get("mean_nap")
            results["well_effect_n_active"] = well_effect.get("n_active")
            results["well_effect_n_total"] = well_effect.get("n_total")

            results["plate_effect_pct"] = plate_effect.get("pct_active")
            results["plate_effect_mean_nap"] = plate_effect.get("mean_nap")
            results["plate_effect_n_active"] = plate_effect.get("n_active")
            results["plate_effect_n_total"] = plate_effect.get("n_total")

            # Flatten per-group batch effects
            for effect_name, effect_data in [("well_effect", well_effect), ("plate_effect", plate_effect)]:
                per_group = effect_data.get("per_group", {})
                if per_group:
                    results[f"{effect_name}_group_summary"] = per_group

            if output_dir is not None:
                batch_path = output_dir / "batch_effects.json"
                with open(batch_path, "w") as f:
                    json.dump(batch_results, f, indent=2)

        except Exception as e:
            print(f"  Batch effects ERROR: {e}")
            results.update(_batch_defaults)

    # Add feature space size
    results["n_features"] = len(features)
    print(f"  Feature space size: {len(features)}")

    # Save metrics
    if output_dir is not None:
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)

    return results
