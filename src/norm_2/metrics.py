"""Metrics for normalization quality evaluation.

This module provides:
- Phenotypic Activity (PA): compound replicate retrieval
- Phenotypic Consistency (PC): target-level retrieval
- Batch effect metrics: silhouette, kBET

Based on JUMP profiling recipe and copairs library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def calculate_phenotypic_activity(
    df: pl.DataFrame,
    features: list[str],
    null_size: int = 10_000,
    p_threshold: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Calculate Phenotypic Activity (compound replicate retrieval).

    Measures how well replicates of the same compound cluster together
    compared to random compounds.

    Args:
        df: Normalized profiles with metadata
        features: Feature column names
        null_size: Size of null distribution
        p_threshold: Significance threshold
        seed: Random seed

    Returns:
        Dictionary with:
        - activity_ap: Per-compound average precision DataFrame
        - activity_map: Mean average precision DataFrame
        - pct_compounds_active: % compounds below corrected p-value
        - n_compounds: Total number of compounds
    """
    from copairs import map as copairs_map
    from copairs.map.average_precision import p_values

    df_pd = df.to_pandas()

    pos_sameby = ["Metadata_pert_iname"]
    pos_diffby = []
    neg_sameby = ["Metadata_Plate"]
    neg_diffby = ["Metadata_pert_iname", "Metadata_negcon"]

    metadata = df_pd.filter(regex="^Metadata")
    profiles = df_pd[features].values

    activity_ap = copairs_map.average_precision(
        metadata, profiles, pos_sameby, pos_diffby, neg_sameby, neg_diffby
    )

    # Filter out DMSO
    activity_ap = activity_ap.query("Metadata_pert_iname != 'DMSO'").copy()

    # Calculate p-values
    activity_ap["p_value"] = p_values(activity_ap, null_size=null_size, seed=seed)
    activity_ap["below_p"] = activity_ap["p_value"] < p_threshold

    # Calculate mean average precision
    activity_map = copairs_map.mean_average_precision(
        activity_ap, pos_sameby, null_size=null_size, threshold=p_threshold, seed=seed
    ).copy()
    activity_map["below_corrected_p"] = activity_map["corrected_p_value"] < p_threshold

    pct_compounds_active = (
        activity_map["below_corrected_p"].sum() / len(activity_map)
    ) * 100

    return {
        "activity_ap": activity_ap,
        "activity_map": activity_map,
        "pct_compounds_active": float(pct_compounds_active),
        "n_compounds": int(len(activity_map)),
    }


def _filter_targets_by_compound_count(
    df_consensus,
    min_compounds_per_target: int = 2,
    max_targets_per_compound: int = 50,
    exclude_unknown: bool = True,
):
    """
    Filter targets based on minimum number of compounds.

    Args:
        df_consensus: DataFrame with Metadata_target column
        min_compounds_per_target: Minimum compounds required per target
        max_targets_per_compound: Maximum targets per compound
        exclude_unknown: Whether to exclude "unknown" targets

    Returns:
        Filtered df_consensus DataFrame
    """
    # Filter out negative controls
    df_consensus = df_consensus[df_consensus["Metadata_negcon"] == False].copy()

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
        "Metadata_pert_iname"
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
    min_compounds_per_target: int = 2,
    max_targets_per_compound: int = 50,
) -> dict[str, Any]:
    """
    Calculate Phenotypic Consistency (target-level retrieval).

    Measures how well compounds targeting the same protein cluster together.

    Args:
        df: Normalized profiles with metadata
        features: Feature column names
        null_size: Size of null distribution
        p_threshold: Significance threshold
        seed: Random seed
        min_compounds_per_target: Minimum compounds required per target
        max_targets_per_compound: Maximum targets per compound

    Returns:
        Dictionary with:
        - target_consistency: Per-target consistency DataFrame
        - pct_targets_active: % of targets below corrected p-value
        - n_targets_active: Number of significant targets
        - n_targets_total: Total number of targets
    """
    from copairs import map as copairs_map

    df_pd = df.to_pandas()

    # Get consensus profiles per compound
    df_consensus = (
        df_pd.groupby(
            ["Metadata_pert_iname", "Metadata_target_list", "Metadata_negcon"],
            as_index=False,
        )[features]
        .median()
        .copy()
    )
    df_consensus["Metadata_target"] = df_consensus["Metadata_target_list"].str.split("|")

    df_consensus = _filter_targets_by_compound_count(
        df_consensus,
        min_compounds_per_target=min_compounds_per_target,
        max_targets_per_compound=max_targets_per_compound,
    )

    if len(df_consensus) < 2:
        print(f"  Warning: Not enough compounds ({len(df_consensus)}) for PC")
        return {
            "target_consistency": None,
            "pct_targets_active": 0.0,
            "n_targets_active": 0,
            "n_targets_total": 0,
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

        return {
            "target_consistency": target_map,
            "pct_targets_active": float(pct_targets_active),
            "n_targets_active": int(n_targets_active),
            "n_targets_total": int(n_targets_total),
        }
    except Exception as e:
        print(f"Warning: Target consistency failed: {e}")
        return {
            "target_consistency": None,
            "pct_targets_active": 0.0,
            "n_targets_active": 0,
            "n_targets_total": 0,
        }


def calculate_batch_metrics(
    df: pl.DataFrame,
    features: list[str],
    batch_col: str = "Metadata_Batch",
    kbet_k_neighbors: int = 25,
) -> dict[str, float]:
    """
    Calculate batch mixing quality metrics.

    Lower scores indicate better batch mixing (less batch effect).

    Args:
        df: Normalized profiles
        features: Feature column names
        batch_col: Column containing batch labels
        kbet_k_neighbors: Number of neighbors for kBET

    Returns:
        Dictionary with:
        - silhouette_batch: Silhouette score (lower = better)
        - kbet_score: kBET rejection rate (lower = better)
    """
    from scib_metrics import kbet
    from scib_metrics.nearest_neighbors import pynndescent
    from sklearn.metrics import silhouette_score

    if batch_col not in df.columns:
        print(f"Warning: {batch_col} not found, skipping batch metrics")
        return {"silhouette_batch": np.nan, "kbet_score": np.nan}

    X = df.select(features).to_numpy()
    batch_labels = df[batch_col].to_numpy()

    unique_batches = np.unique(batch_labels)
    if len(unique_batches) < 2:
        print(f"Warning: Only {len(unique_batches)} batch(es), skipping batch metrics")
        return {"silhouette_batch": np.nan, "kbet_score": np.nan}

    # Silhouette score
    try:
        silhouette_batch = silhouette_score(X, batch_labels)
    except Exception as e:
        print(f"Warning: Silhouette failed: {e}")
        silhouette_batch = np.nan

    # kBET
    try:
        neighbors = pynndescent(
            X, n_neighbors=kbet_k_neighbors, random_state=0, n_jobs=1
        )
        acceptance_rate, _, _ = kbet(neighbors, batch_labels)
        kbet_score = 1.0 - acceptance_rate
    except Exception as e:
        print(f"Warning: kBET failed: {e}")
        kbet_score = np.nan

    return {
        "silhouette_batch": float(silhouette_batch),
        "kbet_score": float(kbet_score),
    }


def evaluate_all(
    df: pl.DataFrame,
    features: list[str],
    output_dir: Path | str | None = None,
    skip_visualization: bool = False,
    skip_umap: bool = False,
    n_top_compounds: int = 20,
) -> dict[str, Any]:
    """
    Run all metrics and optionally save results.

    Args:
        df: Normalized profiles
        features: Feature column names
        output_dir: Directory to save results (None = don't save)
        skip_visualization: Skip visualization generation
        skip_umap: Skip UMAP in visualization
        n_top_compounds: Number of compounds to highlight

    Returns:
        Dictionary with all metrics
    """
    import json

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    results = {}
    full_results = {}

    # Phenotypic Activity
    try:
        pa = calculate_phenotypic_activity(df, features)
        results["PA"] = pa["pct_compounds_active"]
        results["n_compounds"] = pa["n_compounds"]
        full_results["phenotypic_activity"] = pa
        print(f"  PA: {pa['pct_compounds_active']:.2f}%")

        if output_dir is not None:
            if pa.get("activity_ap") is not None and len(pa["activity_ap"]) > 0:
                pa["activity_ap"].to_csv(output_dir / "phenotypic_activity_per_compound.csv", index=False)
            if pa.get("activity_map") is not None and len(pa["activity_map"]) > 0:
                pa["activity_map"].to_csv(output_dir / "phenotypic_activity_map.csv", index=False)
    except Exception as e:
        print(f"  PA ERROR: {e}")
        full_results["phenotypic_activity"] = {}

    # Phenotypic Consistency
    try:
        pc = calculate_phenotypic_consistency(df, features)
        results["PC"] = pc["pct_targets_active"]
        results["n_targets_active"] = pc["n_targets_active"]
        results["n_targets_total"] = pc["n_targets_total"]
        print(f"  PC: {pc['pct_targets_active']:.1f}%")

        if output_dir is not None and pc.get("target_consistency") is not None:
            pc["target_consistency"].to_csv(output_dir / "phenotypic_consistency_per_target.csv", index=False)
    except Exception as e:
        print(f"  PC ERROR: {e}")

    # Batch metrics
    try:
        batch = calculate_batch_metrics(df, features)
        results["Silhouette"] = batch["silhouette_batch"]
        results["kBET"] = batch["kbet_score"]
        print(f"  Silhouette: {batch['silhouette_batch']:.4f}")
        print(f"  kBET: {batch['kbet_score']:.4f}")
    except Exception as e:
        print(f"  Batch ERROR: {e}")

    # Add TVN ill-conditioning state to results
    from .core import get_tvn_state
    tvn_ill_conditioned, tvn_max_condition_number = get_tvn_state()
    results["tvn_ill_conditioned"] = tvn_ill_conditioned
    results["tvn_max_condition_number"] = float(tvn_max_condition_number) if tvn_max_condition_number > 0 else None
    if tvn_ill_conditioned:
        print(f"  WARNING: TVN encountered ill-conditioned matrix (condition number: {tvn_max_condition_number:.2e})")

    # Save metrics
    if output_dir is not None:
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2)

    # Visualization
    if not skip_visualization and output_dir is not None:
        try:
            from .visualization import plot_dimensionality_reduction_extended

            print("  Creating visualization...")
            plot_path = output_dir / "dimreduction.png"
            plot_dimensionality_reduction_extended(
                df,
                features,
                full_results,
                plot_path,
                n_top_compounds=n_top_compounds,
                skip_umap=skip_umap,
            )
            print(f"  Saved visualization to: {plot_path}")
        except Exception as e:
            print(f"  Visualization ERROR: {e}")

    return results
