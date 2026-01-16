# Codec Comparison Analysis

This directory contains scripts to generate codec comparison visualizations.

## Scripts

### 1. plot_codec_comparison.py
Generates PA/PC comparison plots from sweep results:
- `output/codec_comparison_highest_pa.png` - Best PA per codec (basic norm only)
- `output/codec_comparison_best_balance.png` - Best balance scores

### 2. plot_feature_correlation_corrected.py
Generates feature correlation plot:
- `output/codec_feature_correlation.png` - Median feature correlation with ZSTD

### 3. plot_segmentation_iou.py
Generates segmentation overlap plot:
- `output/codec_segmentation_iou.png` - Median segmentation IoU

### 4. plot_standardized_config.py
Generates standardized configuration results:
- `output/codec_standardized_pa_pc.png` - PA & PC with identical settings
- `output/codec_standardized_balance.png` - Balance scores

## Usage

Run all scripts from this directory:

```bash
cd analysis/codec_comparison

# Generate all plots
python plot_codec_comparison.py
python plot_feature_correlation_corrected.py
python plot_segmentation_iou.py
python plot_standardized_config.py
```

Or run all at once:
```bash
python plot_codec_comparison.py && \
python plot_feature_correlation_corrected.py && \
python plot_segmentation_iou.py && \
python plot_standardized_config.py
```

All outputs will be saved to the `output/` subdirectory.

## Dependencies

- matplotlib
- numpy
- pandas
- pathlib

## Data Sources

- Sweep results: `../../data/features/codec_sweeps/*/`
- Segmentation data: `../../result_summary/segmentation_comparison_summary.csv`
- Standardized run results: Hardcoded from actual pipeline runs
