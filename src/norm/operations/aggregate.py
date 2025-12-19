"""Aggregation operations for morphological profiles."""

import polars as pl


def aggregate_profiles(
    df: pl.DataFrame,
    features: list[str],
    metadata: list[str],
    strata: list[str] = None,
    method: str = "median",
    remove_keys: list[str] | None = None,
) -> pl.DataFrame:
    """
    Aggregate replicates to single profiles.

    Args:
        df: Input DataFrame
        features: Feature columns to aggregate
        metadata: Metadata columns to preserve
        strata: Grouping columns (default: ["Metadata_Plate", "Metadata_Well"])
        method: Aggregation method (median or mean)
        remove_keys: Perturbations to exclude (e.g., ["DMSO"])

    Returns:
        Aggregated profiles
    """
    if strata is None:
        strata = ["Metadata_Plate", "Metadata_Well"]

    # Validate strata columns exist
    missing_cols = [col for col in strata if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Strata columns not found: {missing_cols}")

    # Filter out remove_keys if specified
    if remove_keys:
        for key in remove_keys:
            df = df.filter(~pl.any_horizontal(pl.all().eq(key)))

    # Select aggregation function
    if method == "median":
        agg_func = pl.median
    elif method == "mean":
        agg_func = pl.mean
    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    # Aggregate features and keep first metadata value
    metadata_to_keep = [m for m in metadata if m not in strata]

    agg_exprs = [agg_func(feat).alias(feat) for feat in features]
    meta_exprs = [pl.first(meta).alias(meta) for meta in metadata_to_keep]

    df_agg = df.group_by(strata).agg(agg_exprs + meta_exprs)

    return df_agg
