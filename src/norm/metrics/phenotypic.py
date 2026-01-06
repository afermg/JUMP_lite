"""
Phenotypic activity metrics for normalization quality evaluation.

Based on JUMP profiling recipe and copairs library.
"""

import polars as pl
from copairs import map as copairs_map
from copairs.map.average_precision import p_values


def calculate_phenotypic_activity(
    df: pl.DataFrame,
    features: list[str],
    null_size: int = 10_000,
    p_threshold: float = 0.05,
    seed: int = 0,
) -> dict:
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
    # Convert to pandas for copairs compatibility
    df_pd = df.to_pandas()

    pos_sameby = ["Metadata_pert_iname"]
    pos_diffby = []
    neg_sameby = ["Metadata_Plate"]
    neg_diffby = ["Metadata_pert_iname", "Metadata_negcon"]

    metadata = df_pd.filter(regex="^Metadata")
    profiles = df_pd[features].values

    # Calculate average precision
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

    # Extract metrics
    pct_compounds_active = (
        activity_map["below_corrected_p"].sum() / len(activity_map)
    ) * 100

    return {
        "activity_ap": activity_ap,
        "activity_map": activity_map,
        "pct_compounds_active": float(pct_compounds_active),
        "n_compounds": int(len(activity_map)),
    }


def filter_targets_by_compound_count(
    df_consensus,
    min_compounds_per_target: int = 3,
    exclude_unknown: bool = True,
):
    """
    Filter targets based on minimum number of compounds.

    Args:
        df_consensus: DataFrame with Metadata_target column (list of targets per compound)
        min_compounds_per_target: Minimum compounds required per target (default: 2)
        exclude_unknown: Whether to exclude "unknown" targets (default: True)

    Returns:
        Filtered df_consensus DataFrame
    """
    # Filter out negative controls
    df_consensus = df_consensus[df_consensus["Metadata_negcon"] == False].copy()

    # Explode targets to count compounds per target
    df_exploded = df_consensus.explode("Metadata_target")

    # Filter out "unknown" targets if requested
    if exclude_unknown:
        df_exploded = df_exploded[df_exploded["Metadata_target"] != "unknown"].copy()

    # Count unique compounds per target
    target_counts = df_exploded.groupby("Metadata_target")["Metadata_pert_iname"].nunique()
    target_counts = target_counts.sort_values(ascending=False)

    # Report how many targets would remain at different thresholds
    print(f"  Target filtering statistics:")
    for threshold in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        n_targets = (target_counts >= threshold).sum()
        print(f"    min_compounds={threshold}: {n_targets} targets")

    # Apply the chosen threshold
    valid_targets = target_counts[target_counts >= min_compounds_per_target].index.tolist()

    print(f"  Using min_compounds_per_target={min_compounds_per_target}: {len(valid_targets)} targets")

    # Filter df_consensus to only include compounds with valid targets
    def has_valid_target(target_list):
        """Check if any target in the list is valid."""
        if not isinstance(target_list, list):
            return False
        return any(t in valid_targets for t in target_list)

    df_consensus = df_consensus[
        df_consensus["Metadata_target"].apply(has_valid_target)
    ].copy()

    # Update Metadata_target to only include valid targets
    df_consensus["Metadata_target"] = df_consensus["Metadata_target"].apply(
        lambda targets: [t for t in targets if t in valid_targets] if isinstance(targets, list) else []
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
) -> dict:
    """
    Calculate Phenotypic Consistency (target-level retrieval).

    Measures how well compounds targeting the same protein cluster together.

    Args:
        df: Normalized profiles with metadata
        features: Feature column names
        null_size: Size of null distribution
        p_threshold: Significance threshold
        seed: Random seed
        min_compounds_per_target: Minimum compounds required per target (default: 2)

    Returns:
        Dictionary with:
        - target_consistency: Per-target consistency DataFrame
        - n_targets_active: Number of significant targets
        - n_targets_total: Total number of targets
    """
    # Convert to pandas
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
    df_consensus["Metadata_target"] = df_consensus["Metadata_target_list"].str.split(
        "|"
    )

    # Filter targets by minimum compound count
    df_consensus = filter_targets_by_compound_count(
        df_consensus, min_compounds_per_target=min_compounds_per_target
    )

    metadata = df_consensus.filter(regex="^Metadata")
    profiles = df_consensus[features].values

    pos_sameby_target = ["Metadata_target"]
    pos_diffby_target = []
    neg_sameby_target = []
    neg_diffby_target = ["Metadata_target"]

    try:
        # Calculate target-based mAP
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

        return {
            "target_consistency": target_map,
            "n_targets_active": int(n_targets_active),
            "n_targets_total": int(n_targets_total),
        }
    except Exception as e:
        print(f"Warning: Target consistency failed: {e}")
        return {
            "target_consistency": None,
            "n_targets_active": 0,
            "n_targets_total": 0,
        }
