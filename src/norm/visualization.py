"""
Visualization functions for normalization analysis.

Separated from data processing and evaluation.
Includes dimensionality reduction plots and result comparisons.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from umap import UMAP


def plot_dimensionality_reduction_extended(
    df: pl.DataFrame,
    features: list[str],
    evaluation_results: dict,
    output_path: Path,
    n_top_compounds: int = 20,
    pert_iname_col: str = "Metadata_pert_iname",
    negcon_col: str = "Metadata_negcon",
    plate_col: str = "Metadata_Plate",
    figsize: tuple = (14, 18),
    random_state: int = 42,
) -> None:
    """
    Create extended PCA and UMAP visualization plots (3x2 layout).

    Args:
        df: Normalized DataFrame
        features: List of feature column names
        evaluation_results: Results from evaluate_normalization()
        output_path: Path to save the plot
        n_top_compounds: Number of top compounds to highlight
        pert_iname_col: Column with perturbation names
        negcon_col: Column with negative control flag
        plate_col: Column with plate identifiers
        figsize: Figure size (width, height)
        random_state: Random seed for reproducibility
    """
    # Filter features to only numeric ones that exist
    numeric_features = [
        f for f in features
        if f in df.columns and df[f].dtype in (pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32,
                                                pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ]

    if len(numeric_features) == 0:
        logging.warning("No numeric features found for visualization")
        return

    # Convert to numpy
    X = df.select(numeric_features).to_numpy()
    is_negcon = df[negcon_col].to_numpy()
    plate_ids = df[plate_col].to_numpy()

    # Get phenotypic activity results
    pa_metrics = evaluation_results.get("phenotypic_activity", {})
    activity_ap = pa_metrics.get("activity_ap")

    if activity_ap is None or len(activity_ap) == 0:
        logging.warning("No active compounds found")
        top_compounds = []
    else:
        top_compounds = activity_ap.nlargest(n_top_compounds, "average_precision")[pert_iname_col].unique()

    compound_ids = df[pert_iname_col].to_numpy() if pert_iname_col in df.columns else np.array(["Unknown"] * len(df))

    # Compute PCA
    logging.info("  Computing PCA...")
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X)

    # Compute UMAP
    logging.info("  Computing UMAP...")
    umap = UMAP(n_components=2, random_state=random_state, n_jobs=1)
    X_umap = umap.fit_transform(X)

    # Create 3x2 subplot
    fig, axes = plt.subplots(3, 2, figsize=figsize)

    # Row 1: Negcon vs Treatment
    # PCA - Negcon
    ax = axes[0, 0]
    for is_neg, label, color in [(True, "Negcon", "#1f77b4"), (False, "Treatment", "#ff7f0e")]:
        mask = is_negcon == is_neg
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label, alpha=0.5, s=20)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA: Negcon vs Treatment")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # UMAP - Negcon
    ax = axes[0, 1]
    for is_neg, label, color in [(True, "Negcon", "#1f77b4"), (False, "Treatment", "#ff7f0e")]:
        mask = is_negcon == is_neg
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=color, label=label, alpha=0.5, s=20)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title("UMAP: Negcon vs Treatment")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 2: Top 20 Compounds
    cmap = plt.colormaps.get_cmap("tab20")

    # PCA - Top compounds
    ax = axes[1, 0]
    if len(top_compounds) > 0:
        # Plot gray background
        gray_mask = ~np.isin(compound_ids, top_compounds)
        ax.scatter(X_pca[gray_mask, 0], X_pca[gray_mask, 1], c="#cccccc", alpha=0.2, s=10, label="Other")
        # Plot colored compounds
        for i, compound in enumerate(top_compounds):
            mask = compound_ids == compound
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[cmap(i)], label=compound, alpha=0.7, s=30)
        ax.set_title(f"PCA: Top {len(top_compounds)} Compounds (by PA)")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=6)
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], c="#cccccc", alpha=0.3, s=10)
        ax.set_title("PCA: No Active Compounds")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.grid(True, alpha=0.3)

    # UMAP - Top compounds
    ax = axes[1, 1]
    if len(top_compounds) > 0:
        gray_mask = ~np.isin(compound_ids, top_compounds)
        ax.scatter(X_umap[gray_mask, 0], X_umap[gray_mask, 1], c="#cccccc", alpha=0.2, s=10, label="Other")
        for i, compound in enumerate(top_compounds):
            mask = compound_ids == compound
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=[cmap(i)], label=compound, alpha=0.7, s=30)
        ax.set_title(f"UMAP: Top {len(top_compounds)} Compounds (by PA)")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=6)
    else:
        ax.scatter(X_umap[:, 0], X_umap[:, 1], c="#cccccc", alpha=0.3, s=10)
        ax.set_title("UMAP: No Active Compounds")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.grid(True, alpha=0.3)

    # Row 3: Plate ID
    unique_plates = np.unique(plate_ids)
    n_plates = len(unique_plates)
    plate_cmap = plt.colormaps.get_cmap("tab10" if n_plates <= 10 else "tab20")
    plate_to_color = {plate: plate_cmap(i % 20) for i, plate in enumerate(unique_plates)}

    # PCA - Plate ID
    ax = axes[2, 0]
    for i, plate in enumerate(unique_plates):
        mask = plate_ids == plate
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[plate_to_color[plate]], label=plate, alpha=0.6, s=20)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title(f"PCA: Plate ID ({n_plates} plates)")
    if n_plates <= 10:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # UMAP - Plate ID
    ax = axes[2, 1]
    for i, plate in enumerate(unique_plates):
        mask = plate_ids == plate
        ax.scatter(X_umap[mask, 0], X_umap[mask, 1], c=[plate_to_color[plate]], label=plate, alpha=0.6, s=20)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(f"UMAP: Plate ID ({n_plates} plates)")
    if n_plates <= 10:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logging.info(f"  Saved extended dimensionality reduction plot to {output_path}")
