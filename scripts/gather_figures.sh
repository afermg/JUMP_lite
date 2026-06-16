#!/usr/bin/env bash
set -euo pipefail

# Gather key figures into main_figures/ at repo root.
# Usage: just gather-figures

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${REPO_ROOT}/main_figures"

mkdir -p "$OUT"

FIGURES=(
    # Feature correlation: full dataset
    "analysis/feature_similarity/output/correlation_boxenplot.png"
    "analysis/feature_similarity/output/feature_group_boxplot.png"
    "analysis/feature_similarity/output/feature_group_by_compartment.png"
    "analysis/feature_similarity/output/feature_subgroups_by_compartment.png"

    # Feature correlation: greenlist-filtered
    "analysis/feature_similarity/output/correlation_boxenplot_greenlist.png"

    # Cross-well consistency
    "analysis/feature_similarity/output/cross_well_vs_codec_ks_stat.png"
    "analysis/feature_similarity/output/replicate_vs_codec_correlation.png"

    # Codec feature correlation (cell vs site)
    "analysis/output/codec_feature_correlation_cell_vs_site.png"

    # Image quality
    "analysis/image_quality/output/quality_metrics_violin.png"
    "analysis/image_quality/output/ssim_violin.png"

    # Segmentation
    "analysis/segmentation/output/segmentation_comparison/segmentation_comparison_iou_boxenplot_combined.png"
    "analysis/segmentation/output/segmentation_comparison/segmentation_comparison_cell_level_iou_boxenplot_all_matched.png"
    "analysis/segmentation/output/segmentation_comparison/segmentation_comparison_cell_level_iou_boxenplot_tp_only.png"

    # Segmentation: Instance AP@0.5
    "analysis/segmentation/output/segmentation_comparison/segmentation_comparison_inst_ap_iou50_boxenplot_combined.png"

    # Normalization sweep v10: NAP PA vs PC
    "src/norm_3/data/features/variance_first_v10/plots/sweep_nap_pa_vs_pc_best_balanced.png"
    "src/norm_3/data/features/variance_first_v10/plots/sweep_nap_pa_vs_pc_best_balanced_best_any_codec.png"
    "src/norm_3/data/features/variance_first_v10/plots/sweep_nap_pa_vs_pc_best_balanced_best_avg_codec.png"
    "src/norm_3/data/features/variance_first_v10/plots/sweep_nap_pa_vs_pc_best_balanced_zstd_pinned.png"
)

copied=0
missing=0
for fig in "${FIGURES[@]}"; do
    src="${REPO_ROOT}/${fig}"
    if [[ -f "$src" ]]; then
        cp "$src" "$OUT/"
        copied=$((copied + 1))
    else
        echo "WARNING: missing ${fig}"
        missing=$((missing + 1))
    fi
done

echo "Gathered ${copied} figures into ${OUT}/ (${missing} missing)"
