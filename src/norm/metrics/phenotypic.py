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


def calculate_phenotypic_consistency(
    df: pl.DataFrame,
    features: list[str],
    null_size: int = 10_000,
    p_threshold: float = 0.05,
    seed: int = 0,
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
