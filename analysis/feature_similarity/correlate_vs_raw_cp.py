#!/usr/bin/env python3
"""
Compare extracted features (filtered and non-filtered) against raw CellProfiler reference.

Computes per-feature Spearman correlations between:
1. Raw CellProfiler reference vs filtered (border+size) extracted features
2. Raw CellProfiler reference vs non-filtered extracted features

Includes a focused analysis on cell count, area, and diameter features.

Uses FeatureMapper from src/utils/map_cellprofiler_features.py to map between naming conventions.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy import stats

# Add project root to path so we can import the mapper
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # Two levels up from analysis/feature_similarity/

sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.map_cellprofiler_features import FeatureMapper


# --- Paths ---
RAW_CP_PATH = PROJECT_ROOT / "output" / "raw_jump_cp_profiles_reformatted_filtered.parquet"
FILTERED_DIR = PROJECT_ROOT / "data" / "features" / "jump_target2_4plate_filtered"
NONFILTERED_DIR = PROJECT_ROOT / "data" / "features" / "jump_target2_4plate"
OUTPUT_DIR = SCRIPT_DIR / "output"

def discover_compressions(base_dir: Path, dataset: str = "jump_target2_4plate") -> list[str]:
    """Discover available compressions from feature files."""
    pattern = f"cp_measure_{dataset}_*_raw_features.parquet"
    files = list(base_dir.glob(pattern))
    compressions = []
    for f in files:
        # Extract compression name: cp_measure_{dataset}_{compression}_raw_features.parquet
        name = f.stem  # e.g., cp_measure_jump_target2_4plate_zstd_raw_features
        prefix = f"cp_measure_{dataset}_"
        suffix = "_raw_features"
        if name.startswith(prefix) and name.endswith(suffix):
            compression = name[len(prefix):-len(suffix)]
            compressions.append(compression)
    # Sort: zstd first, then others alphabetically
    compressions.sort(key=lambda x: (x != "zstd", x))
    return compressions

# Features of special interest for the focused analysis (exact feature name matches)
FOCUS_CP_FEATURES = {
    "Cells_AreaShape_Area",
    "Cells_AreaShape_EquivalentDiameter",
    "Cells_AreaShape_MajorAxisLength",
    "Cells_AreaShape_MinorAxisLength",
    "Cells_AreaShape_Perimeter",
    "Cells_AreaShape_MaximumRadius",
    "Cells_AreaShape_MeanRadius",
    "Cells_AreaShape_MedianRadius",
    "Cells_AreaShape_BoundingBoxArea",
    "Cells_AreaShape_ConvexArea",
    "Cells_AreaShape_FilledArea",
    "Nuclei_AreaShape_Area",
    "Nuclei_AreaShape_EquivalentDiameter",
    "Nuclei_AreaShape_MajorAxisLength",
    "Nuclei_AreaShape_MinorAxisLength",
    "Nuclei_AreaShape_Perimeter",
    "Nuclei_AreaShape_MaximumRadius",
    "Nuclei_AreaShape_MeanRadius",
    "Nuclei_AreaShape_MedianRadius",
    "Nuclei_AreaShape_BoundingBoxArea",
    "Nuclei_AreaShape_ConvexArea",
    "Nuclei_AreaShape_FilledArea",
}


def build_join_key(df: pl.DataFrame, kind: str) -> pl.DataFrame:
    """Add a join key (source__plate__well) to the dataframe."""
    return df.with_columns(
        (pl.col("Metadata_Source") + "__" + pl.col("Metadata_Plate") + "__" + pl.col("Metadata_Well"))
        .alias("_join_key")
    )


def compute_correlations(
    raw_cp: pl.DataFrame,
    extracted: pl.DataFrame,
    mapping: dict[str, str],
) -> pl.DataFrame:
    """
    Compute per-feature Spearman correlations between raw CP and extracted features.

    Returns:
        DataFrame with columns: cp_measure_feature, cellprofiler_feature, correlation
    """
    # Join on the key — select unique CP feature columns to avoid duplicates
    cp_cols = list(dict.fromkeys(mapping.values()))  # unique, preserving order
    joined = extracted.join(raw_cp.select(["_join_key"] + cp_cols), on="_join_key", how="inner")
    n_rows = len(joined)
    print(f"  Joined rows: {n_rows}")

    results = []
    for cp_measure_feat, cp_feat in mapping.items():
        if cp_measure_feat not in joined.columns or cp_feat not in joined.columns:
            continue
        x = joined[cp_measure_feat].to_numpy().astype(np.float64)
        y = joined[cp_feat].to_numpy().astype(np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if len(x) < 10 or np.std(x) == 0 or np.std(y) == 0:
            corr = np.nan
        else:
            corr, _ = stats.spearmanr(x, y)
        results.append({
            "cp_measure_feature": cp_measure_feat,
            "cellprofiler_feature": cp_feat,
            "correlation": corr,
        })

    return pl.DataFrame(results)


def get_cell_counts(df: pl.DataFrame) -> pl.DataFrame | None:
    """Extract cell/nuclei count columns from extracted data."""
    cols = ["_join_key"]
    if "Metadata_n_cells" in df.columns:
        cols.append("Metadata_n_cells")
    if "Metadata_n_nuclei" in df.columns:
        cols.append("Metadata_n_nuclei")
    if len(cols) <= 1:
        return None
    return df.select(cols)


def get_feature_category(cp_feat: str) -> str:
    """Extract feature category from CellProfiler feature name."""
    parts = cp_feat.split("_")
    return parts[1] if len(parts) >= 2 else "Unknown"


def get_compartment(cp_feat: str) -> str:
    """Extract compartment from CellProfiler feature name."""
    parts = cp_feat.split("_")
    return parts[0] if parts else "Unknown"


def is_focus_feature(cp_measure_feat: str, cp_feat: str) -> bool:
    """Check if a feature is one of the focus features (area, diameter, count)."""
    return cp_feat in FOCUS_CP_FEATURES


def main():
    # Discover available compressions dynamically
    COMPRESSIONS = discover_compressions(NONFILTERED_DIR)
    print(f"Discovered {len(COMPRESSIONS)} compressions: {COMPRESSIONS}")

    # Load raw CP reference
    print("Loading raw CellProfiler reference...")
    raw_cp = pl.read_parquet(RAW_CP_PATH)
    print(f"  Shape: {raw_cp.shape}")
    raw_cp = build_join_key(raw_cp, "raw_cp")

    # Check if count columns exist
    print(f"  Has Metadata_Count_Cells: {'Metadata_Count_Cells' in raw_cp.columns}")
    print(f"  Has Metadata_Count_Nuclei: {'Metadata_Count_Nuclei' in raw_cp.columns}")

    # Build feature mapping
    zstd_nonfiltered = NONFILTERED_DIR / "cp_measure_jump_target2_4plate_zstd_raw_features.parquet"
    print("\nBuilding feature mapping using FeatureMapper...")
    mapper = FeatureMapper(str(RAW_CP_PATH), str(zstd_nonfiltered))
    mapping, ambiguous = mapper.create_mapping()
    coverage = mapper.analyze_coverage(mapping)
    print(f"  Mapped features: {coverage['mapped_f2']} / {coverage['total_f2']} ({coverage['coverage_f2']:.1f}%)")

    # Filter and deduplicate
    raw_cp_cols = set(raw_cp.columns)
    valid_mapping = {k: v for k, v in mapping.items() if v in raw_cp_cols}
    print(f"  Valid mappings (in raw CP): {len(valid_mapping)}")

    seen_cp_feats: dict[str, str] = {}
    deduped_mapping: dict[str, str] = {}
    for cp_m, cp_f in valid_mapping.items():
        if cp_f not in seen_cp_feats:
            seen_cp_feats[cp_f] = cp_m
            deduped_mapping[cp_m] = cp_f
        else:
            existing = seen_cp_feats[cp_f]
            if "/max/" in cp_m and "/max/" not in existing:
                del deduped_mapping[existing]
                seen_cp_feats[cp_f] = cp_m
                deduped_mapping[cp_m] = cp_f
    valid_mapping = deduped_mapping
    print(f"  After deduplication: {len(valid_mapping)}")

    # Identify focus features in the mapping
    focus_mapping = {k: v for k, v in valid_mapping.items() if is_focus_feature(k, v)}
    print(f"\n  Focus features (area/diameter): {len(focus_mapping)}")
    for k, v in sorted(focus_mapping.items()):
        print(f"    {k} -> {v}")

    # Process each compression
    all_results = []
    cell_count_data = {}  # (compression, filter_type) -> DataFrame with counts

    for compression in COMPRESSIONS:
        print(f"\n{'='*60}")
        print(f"Compression: {compression}")

        for filter_type, base_dir, suffix in [
            ("non-filtered", NONFILTERED_DIR, f"cp_measure_jump_target2_4plate_{compression}_raw_features.parquet"),
            ("filtered", FILTERED_DIR, f"cp_measure_jump_target2_4plate_{compression}_filtered_border_size_raw_features.parquet"),
        ]:
            fpath = base_dir / suffix
            if not fpath.exists():
                print(f"\n  {filter_type}: NOT FOUND")
                continue

            print(f"\n  {filter_type}: {fpath.name}")
            df = pl.read_parquet(fpath)
            df = build_join_key(df, "extracted")
            feat_mapping = {k: v for k, v in valid_mapping.items() if k in df.columns}
            print(f"  Features to correlate: {len(feat_mapping)}")
            corrs = compute_correlations(raw_cp, df, feat_mapping)
            corrs = corrs.with_columns([
                pl.lit(compression).alias("compression"),
                pl.lit(filter_type).alias("filter_type"),
            ])
            all_results.append(corrs)

            # Collect cell count data
            cc = get_cell_counts(df)
            if cc is not None:
                cell_count_data[(compression, filter_type)] = cc

    # Combine all results
    combined = pl.concat(all_results)
    combined = combined.with_columns([
        pl.col("cellprofiler_feature").map_elements(get_feature_category, return_dtype=pl.Utf8).alias("category"),
        pl.col("cellprofiler_feature").map_elements(get_compartment, return_dtype=pl.Utf8).alias("compartment"),
    ])

    # Save raw results
    combined.write_parquet(OUTPUT_DIR / "raw_cp_spearman_correlation_results.parquet")
    print(f"\n\nSaved results to {OUTPUT_DIR / 'raw_cp_spearman_correlation_results.parquet'}")
    print(f"Total rows: {len(combined)}")

    # =====================================================================
    # SUMMARY TABLE
    # =====================================================================
    print(f"\n{'='*80}")
    print("SUMMARY: Median Spearman correlation with raw CellProfiler reference")
    print(f"{'='*80}")

    summary = (
        combined.group_by(["compression", "filter_type"])
        .agg([
            pl.col("correlation").median().alias("median_corr"),
            pl.col("correlation").mean().alias("mean_corr"),
            pl.col("correlation").quantile(0.25).alias("q25_corr"),
            pl.col("correlation").quantile(0.75).alias("q75_corr"),
            pl.col("correlation").count().alias("n_features"),
            (pl.col("correlation") > 0.9).sum().alias("n_above_0.9"),
        ])
        .sort(["compression", "filter_type"])
    )
    print(summary)

    pivot_summary = (
        combined.group_by(["compression", "filter_type"])
        .agg(pl.col("correlation").median().alias("median_corr"))
        .pivot(on="filter_type", index="compression", values="median_corr")
        .sort("compression")
    )
    if "filtered" in pivot_summary.columns and "non-filtered" in pivot_summary.columns:
        pivot_summary = pivot_summary.with_columns(
            (pl.col("filtered") - pl.col("non-filtered")).alias("diff")
        )
    print(f"\n{'='*80}")
    print("Per-compression comparison (median Spearman with raw CP)")
    print(f"{'='*80}")
    print(pivot_summary)

    # =====================================================================
    # PLOTS
    # =====================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Violin plot ---
    fig, axes = plt.subplots(1, len(COMPRESSIONS), figsize=(3 * len(COMPRESSIONS), 7), sharey=True)
    if len(COMPRESSIONS) == 1:
        axes = [axes]

    for ax, comp in zip(axes, COMPRESSIONS):
        nf_data = combined.filter(
            (pl.col("compression") == comp) & (pl.col("filter_type") == "non-filtered")
        )["correlation"].drop_nulls().to_numpy()
        f_data = combined.filter(
            (pl.col("compression") == comp) & (pl.col("filter_type") == "filtered")
        )["correlation"].drop_nulls().to_numpy()

        parts = ax.violinplot([nf_data, f_data], positions=[0, 1], showmedians=True, widths=0.8)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(["#1f77b4", "#ff7f0e"][i])
            pc.set_alpha(0.7)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-filt", "Filtered"], rotation=45, ha="right", fontsize=8)
        ax.set_title(comp.replace("jpegxl_lossy_", "jxl_"), fontsize=9)
        ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        nf_med = np.median(nf_data) if len(nf_data) > 0 else 0
        f_med = np.median(f_data) if len(f_data) > 0 else 0
        ax.text(0, -0.25, f"{nf_med:.3f}", ha="center", fontsize=7, color="#1f77b4", fontweight="bold")
        ax.text(1, -0.25, f"{f_med:.3f}", ha="center", fontsize=7, color="#ff7f0e", fontweight="bold")
        ax.set_ylim(-0.35, 1.05)

    axes[0].set_ylabel("Spearman correlation with raw CellProfiler")
    fig.suptitle("Feature Correlation with Raw CellProfiler Reference\n(Filtered vs Non-Filtered, Spearman)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "raw_cp_spearman_violin.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved violin plot to {OUTPUT_DIR / 'raw_cp_spearman_violin.png'}")

    # --- 2. Scatter plot (zstd) ---
    fig, ax = plt.subplots(figsize=(8, 8))
    zstd_nf = combined.filter(
        (pl.col("compression") == "zstd") & (pl.col("filter_type") == "non-filtered")
    ).select(["cp_measure_feature", "correlation"]).rename({"correlation": "corr_nf"})
    zstd_f = combined.filter(
        (pl.col("compression") == "zstd") & (pl.col("filter_type") == "filtered")
    ).select(["cp_measure_feature", "correlation"]).rename({"correlation": "corr_f"})

    scatter_df = zstd_nf.join(zstd_f, on="cp_measure_feature", how="inner").drop_nulls()
    x = scatter_df["corr_nf"].to_numpy()
    y = scatter_df["corr_f"].to_numpy()

    # Color focus features differently
    is_focus = np.array([
        is_focus_feature(f, valid_mapping.get(f, ""))
        for f in scatter_df["cp_measure_feature"].to_list()
    ])
    ax.scatter(x[~is_focus], y[~is_focus], alpha=0.3, s=10, c="#1f77b4", label="Other features")
    ax.scatter(x[is_focus], y[is_focus], alpha=0.9, s=40, c="red", marker="*", label="Area/Diameter", zorder=5)
    ax.plot([-.3, 1], [-.3, 1], "k--", alpha=0.4, label="y=x")
    ax.set_xlabel("Non-filtered Spearman r with raw CP")
    ax.set_ylabel("Filtered (border+size) Spearman r with raw CP")
    ax.set_title(f"zstd: Filtered vs Non-Filtered (n={len(scatter_df)} features)")
    ax.legend(loc="lower right")
    ax.set_xlim(-0.3, 1.05)
    ax.set_ylim(-0.3, 1.05)
    ax.set_aspect("equal")

    improved = (y > x).sum()
    worsened = (y < x).sum()
    mean_diff = np.nanmean(y - x)
    ax.text(0.05, 0.95, f"Improved: {improved}\nWorsened: {worsened}\nMean diff: {mean_diff:+.4f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "raw_cp_spearman_scatter_zstd.png", dpi=150, bbox_inches="tight")
    print(f"Saved scatter plot to {OUTPUT_DIR / 'raw_cp_spearman_scatter_zstd.png'}")

    # --- 3. Bar chart by category (zstd) ---
    zstd_combined = combined.filter(pl.col("compression") == "zstd")
    cat_summary = (
        zstd_combined.group_by(["category", "filter_type"])
        .agg(pl.col("correlation").median().alias("median_corr"))
        .sort("category")
    )
    categories = sorted(cat_summary["category"].unique().to_list())
    nf_vals, f_vals = [], []
    for cat in categories:
        nf_row = cat_summary.filter((pl.col("category") == cat) & (pl.col("filter_type") == "non-filtered"))
        f_row = cat_summary.filter((pl.col("category") == cat) & (pl.col("filter_type") == "filtered"))
        nf_vals.append(nf_row["median_corr"][0] if len(nf_row) > 0 else 0)
        f_vals.append(f_row["median_corr"][0] if len(f_row) > 0 else 0)

    x_pos = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x_pos - width / 2, nf_vals, width, label="Non-filtered", color="#1f77b4")
    ax.bar(x_pos + width / 2, f_vals, width, label="Filtered", color="#ff7f0e")
    ax.set_xlabel("Feature Category")
    ax.set_ylabel("Median Spearman Correlation with Raw CP")
    ax.set_title("zstd: Median Spearman Correlation by Feature Category")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.legend()
    ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "raw_cp_spearman_by_category_zstd.png", dpi=150, bbox_inches="tight")
    print(f"Saved category bar chart to {OUTPUT_DIR / 'raw_cp_spearman_by_category_zstd.png'}")

    # =====================================================================
    # FOCUSED ANALYSIS: Cell Count, Area, Diameter
    # =====================================================================
    print(f"\n\n{'#'*80}")
    print("FOCUSED ANALYSIS: Cell Count, Area, and Diameter")
    print(f"{'#'*80}")

    # --- Cell/nuclei count analysis (filtered data only has counts) ---
    filtered_count_data = {k: v for k, v in cell_count_data.items() if k[1] == "filtered"}
    if filtered_count_data:
        print(f"\n--- Cell/Nuclei Counts After Filtering (per compression) ---")
        count_stats = []
        for (compression, _), cc in sorted(filtered_count_data.items()):
            for col, label in [("Metadata_n_cells", "cell_count"), ("Metadata_n_nuclei", "nuclei_count")]:
                if col in cc.columns:
                    vals = cc[col].drop_nulls().to_numpy().astype(np.float64)
                    count_stats.append({
                        "compression": compression, "metric": label,
                        "median": float(np.median(vals)), "mean": float(np.mean(vals)),
                        "std": float(np.std(vals)), "min": float(np.min(vals)), "max": float(np.max(vals)),
                    })
        if count_stats:
            cs_df = pl.DataFrame(count_stats)
            print(cs_df)

        # Plot: cell count distributions across compressions (filtered)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, col, title in zip(axes,
            ["Metadata_n_cells", "Metadata_n_nuclei"],
            ["Cell Count per Well (Filtered)", "Nuclei Count per Well (Filtered)"]):
            data_by_comp = []
            labels = []
            for comp in COMPRESSIONS:
                key = (comp, "filtered")
                if key in filtered_count_data and col in filtered_count_data[key].columns:
                    vals = filtered_count_data[key][col].drop_nulls().to_numpy()
                    data_by_comp.append(vals)
                    labels.append(comp.replace("jpegxl_lossy_", "jxl_"))
            if data_by_comp:
                parts = ax.violinplot(data_by_comp, positions=range(len(data_by_comp)), showmedians=True, widths=0.8)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
                ax.set_title(title)
                ax.set_ylabel("Count")
                for i, vals in enumerate(data_by_comp):
                    ax.text(i, np.median(vals) - 10, f"{np.median(vals):.0f}", ha="center", fontsize=7, fontweight="bold")

        fig.suptitle("Cell/Nuclei Counts per Well After Border+Size Filtering", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "raw_cp_cell_count_by_compression.png", dpi=150, bbox_inches="tight")
        print(f"\nSaved cell count plot to {OUTPUT_DIR / 'raw_cp_cell_count_by_compression.png'}")

    # --- Area/Diameter feature correlations across compressions ---
    focus_corrs = combined.filter(
        pl.col("cellprofiler_feature").is_in(list(FOCUS_CP_FEATURES))
    )
    if len(focus_corrs) > 0:
        print(f"\n--- Area/Diameter Feature Correlations (zstd) ---")
        zstd_focus = focus_corrs.filter(pl.col("compression") == "zstd")
        focus_pivot = (
            zstd_focus
            .pivot(on="filter_type", index=["cp_measure_feature", "cellprofiler_feature"], values="correlation")
            .sort("cellprofiler_feature")
        )
        if "filtered" in focus_pivot.columns and "non-filtered" in focus_pivot.columns:
            focus_pivot = focus_pivot.with_columns(
                (pl.col("filtered") - pl.col("non-filtered")).alias("diff")
            )
        print(focus_pivot)

        # Heatmap-style plot: area/diameter features across compressions
        focus_features = sorted(focus_corrs["cellprofiler_feature"].unique().to_list())
        fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(focus_features) * 0.35)))

        for ax, ft in zip(axes, ["non-filtered", "filtered"]):
            ft_data = focus_corrs.filter(pl.col("filter_type") == ft)

            # Build matrix: features x compressions
            matrix = np.full((len(focus_features), len(COMPRESSIONS)), np.nan)
            for i, feat in enumerate(focus_features):
                for j, comp in enumerate(COMPRESSIONS):
                    row = ft_data.filter(
                        (pl.col("cellprofiler_feature") == feat) & (pl.col("compression") == comp)
                    )
                    if len(row) > 0:
                        matrix[i, j] = row["correlation"][0]

            im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-0.2, vmax=1.0)
            ax.set_xticks(range(len(COMPRESSIONS)))
            ax.set_xticklabels([c.replace("jpegxl_lossy_", "jxl_") for c in COMPRESSIONS], rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(focus_features)))
            ax.set_yticklabels(focus_features, fontsize=6)
            ax.set_title(f"{ft.title()}", fontsize=10)

            # Annotate cells
            for i in range(len(focus_features)):
                for j in range(len(COMPRESSIONS)):
                    val = matrix[i, j]
                    if np.isfinite(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5,
                                color="white" if val < 0.3 else "black")

        fig.colorbar(im, ax=axes, shrink=0.6, label="Spearman r")
        fig.suptitle("Area/Diameter Features: Spearman Correlation with Raw CellProfiler", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "raw_cp_area_diameter_heatmap.png", dpi=150, bbox_inches="tight")
        print(f"Saved area/diameter heatmap to {OUTPUT_DIR / 'raw_cp_area_diameter_heatmap.png'}")

        # Bar chart: focus features, filtered vs non-filtered (zstd only)
        if len(focus_pivot) > 0:
            fig, ax = plt.subplots(figsize=(14, 6))
            feat_labels = focus_pivot["cellprofiler_feature"].to_list()
            x_pos = np.arange(len(feat_labels))
            width = 0.35

            nf_col = "non-filtered" if "non-filtered" in focus_pivot.columns else None
            f_col = "filtered" if "filtered" in focus_pivot.columns else None

            if nf_col:
                nf_v = focus_pivot[nf_col].to_numpy()
                ax.bar(x_pos - width / 2, nf_v, width, label="Non-filtered", color="#1f77b4")
            if f_col:
                f_v = focus_pivot[f_col].to_numpy()
                ax.bar(x_pos + width / 2, f_v, width, label="Filtered", color="#ff7f0e")

            ax.set_xticks(x_pos)
            ax.set_xticklabels(feat_labels, rotation=60, ha="right", fontsize=7)
            ax.set_ylabel("Spearman r with Raw CP")
            ax.set_title("zstd: Area/Diameter Features - Filtered vs Non-Filtered")
            ax.legend()
            ax.axhline(y=0.9, color="gray", linestyle="--", alpha=0.5)
            ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "raw_cp_area_diameter_bar_zstd.png", dpi=150, bbox_inches="tight")
            print(f"Saved area/diameter bar chart to {OUTPUT_DIR / 'raw_cp_area_diameter_bar_zstd.png'}")


if __name__ == "__main__":
    main()
