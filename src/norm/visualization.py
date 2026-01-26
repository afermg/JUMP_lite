"""
Visualization functions for normalization analysis.

Separated from data processing and evaluation.
Includes dimensionality reduction plots and result comparisons.
"""

import logging
from pathlib import Path

# Use non-interactive backend to avoid issues in parallel processes
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def plot_dimensionality_reduction_extended(
    df: pl.DataFrame,
    features: list[str],
    evaluation_results: dict,
    output_path: Path,
    n_top_compounds: int = 20,
    pert_iname_col: str = "Metadata_pert_iname",
    negcon_col: str = "Metadata_negcon",
    plate_col: str = "Metadata_Plate",
    figsize: tuple = (14, 24),
    random_state: int = 42,
    skip_umap: bool = False,
) -> None:
    """
    Create extended PCA and UMAP visualization plots (4x2 layout).

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
        skip_umap: If True, skip UMAP computation (useful for parallel runs)
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

    # Compute UMAP (optional - can cause issues in parallel execution)
    X_umap = None
    if skip_umap:
        logging.info("  Skipping UMAP (skip_umap=True)")
    else:
        logging.info("  Computing UMAP...")
        try:
            from umap import UMAP
            umap = UMAP(n_components=2, random_state=random_state, n_jobs=1)
            X_umap = umap.fit_transform(X)
        except Exception as e:
            logging.warning(f"  UMAP failed: {e}. Using PCA only.")

    # Create layout based on whether UMAP is available (4 rows now for heatmap)
    if X_umap is not None:
        fig, axes = plt.subplots(4, 2, figsize=figsize)
    else:
        # PCA only - use 4x1 layout
        fig, axes = plt.subplots(4, 1, figsize=(figsize[0] // 2, figsize[1]))
        # Reshape to 2D for consistent indexing
        axes = axes.reshape(-1, 1)

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

    # UMAP - Negcon (if available)
    if X_umap is not None:
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

    # UMAP - Top compounds (if available)
    if X_umap is not None:
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

    # UMAP - Plate ID (if available)
    if X_umap is not None:
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

    # Row 4: Cosine similarity heatmap of top compounds' samples + negcon
    if len(top_compounds) > 0:
        logging.info("  Computing cosine similarity heatmap for top compounds + negcon...")

        # Get samples for top compounds
        top_mask = np.isin(compound_ids, top_compounds)
        top_X = X[top_mask]
        top_compound_ids = compound_ids[top_mask]

        # Get negcon samples (randomly sample 4)
        negcon_mask = is_negcon == True
        negcon_indices = np.where(negcon_mask)[0]
        n_negcon_samples = min(4, len(negcon_indices))
        rng = np.random.default_rng(random_state)
        sampled_negcon_indices = rng.choice(negcon_indices, size=n_negcon_samples, replace=False)
        negcon_X = X[sampled_negcon_indices]
        negcon_ids = np.array(["negcon"] * n_negcon_samples)

        # Sort top compounds by name to group samples together
        sort_idx = np.argsort(top_compound_ids)
        top_X = top_X[sort_idx]
        top_compound_ids = top_compound_ids[sort_idx]

        # Get unique compound names (sorted) - negcon will be added last
        unique_compounds = list(dict.fromkeys(top_compound_ids))
        unique_top = unique_compounds + ["negcon"]

        # Combine: top compounds first, then negcon at the end
        combined_X = np.vstack([top_X, negcon_X])
        combined_ids = np.concatenate([top_compound_ids, negcon_ids])

        # Compute cosine similarity matrix
        corr_matrix = cosine_similarity(combined_X)

        # Create color labels for compounds (negcon gets a distinct gray color)
        compound_to_idx = {c: i for i, c in enumerate(unique_top)}
        # Use gray for negcon
        def get_color(c, idx):
            if c == "negcon":
                return (0.5, 0.5, 0.5, 1.0)  # Gray for negcon
            return cmap(idx % 20)
        sample_colors = [get_color(c, compound_to_idx[c]) for c in combined_ids]

        # Use combined_ids for counting samples per compound
        sample_ids = combined_ids

        # Plot heatmap spanning both columns
        if X_umap is not None:
            # Merge the two bottom axes into one for the heatmap
            ax_left = axes[3, 0]
            ax_right = axes[3, 1]
            ax_left.remove()
            ax_right.remove()
            ax = fig.add_subplot(4, 1, 4)
        else:
            ax = axes[3, 0]

        # Plot correlation heatmap
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        # Add compound boundaries and labels
        boundaries = [0]
        for compound in unique_top:
            count = np.sum(sample_ids == compound)
            boundaries.append(boundaries[-1] + count)

        # Draw lines at compound boundaries
        for b in boundaries[1:-1]:
            ax.axhline(y=b - 0.5, color='black', linewidth=0.5, alpha=0.5)
            ax.axvline(x=b - 0.5, color='black', linewidth=0.5, alpha=0.5)

        # Add compound name labels at midpoints
        midpoints = [(boundaries[i] + boundaries[i+1]) / 2 for i in range(len(boundaries) - 1)]
        ax.set_yticks(midpoints)
        ax.set_yticklabels(unique_top, fontsize=6)
        ax.set_xticks(midpoints)
        ax.set_xticklabels(unique_top, fontsize=6, rotation=45, ha='right')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Cosine Similarity', fontsize=9)

        n_compounds = len(unique_top) - 1 if "negcon" in unique_top else len(unique_top)
        ax.set_title(f"Sample Cosine Similarity: Top {n_compounds} Compounds + Negcon ({len(top_compound_ids)} samples)")

    else:
        # No top compounds - show empty plot with message
        if X_umap is not None:
            ax_left = axes[3, 0]
            ax_right = axes[3, 1]
            ax_left.remove()
            ax_right.remove()
            ax = fig.add_subplot(4, 1, 4)
        else:
            ax = axes[3, 0]
        ax.text(0.5, 0.5, "No active compounds found for heatmap",
                ha='center', va='center', fontsize=12, transform=ax.transAxes)
        ax.set_title("Sample Cosine Similarity")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logging.info(f"  Saved extended dimensionality reduction plot to {output_path}")
