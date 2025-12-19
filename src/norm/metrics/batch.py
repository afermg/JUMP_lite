"""Batch effect metrics for normalization quality evaluation."""

import numpy as np
import polars as pl
from scib_metrics import kbet
from scib_metrics.nearest_neighbors import pynndescent
from sklearn.metrics import silhouette_score


def calculate_batch_metrics(
    df: pl.DataFrame,
    features: list[str],
    batch_col: str = "Metadata_Batch",
    kbet_k_neighbors: int = 25,
) -> dict:
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
    if batch_col not in df.columns:
        print(f"Warning: {batch_col} not found, skipping batch metrics")
        return {"silhouette_batch": np.nan, "kbet_score": np.nan}

    # Convert to numpy
    X = df.select(features).to_numpy()
    batch_labels = df[batch_col].to_numpy()

    unique_batches = np.unique(batch_labels)
    if len(unique_batches) < 2:
        print(f"Warning: Only {len(unique_batches)} batch(es), skipping batch metrics")
        return {"silhouette_batch": np.nan, "kbet_score": np.nan}

    # Silhouette score (lower = better batch mixing)
    try:
        silhouette_batch = silhouette_score(X, batch_labels)
    except Exception as e:
        print(f"Warning: Silhouette failed: {e}")
        silhouette_batch = np.nan

    # kBET (rejection rate, lower = better batch mixing)
    try:
        neighbors = pynndescent(
            X, n_neighbors=kbet_k_neighbors, random_state=0, n_jobs=1
        )
        acceptance_rate, _, _ = kbet(neighbors, batch_labels)
        kbet_score = 1.0 - acceptance_rate
    except Exception as e:
        print(f"Warning: kBET failed: {e}")
        kbet_score = np.nan

    return {"silhouette_batch": float(silhouette_batch), "kbet_score": float(kbet_score)}
