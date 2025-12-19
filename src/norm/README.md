# Norm - Minimal Normalization Pipeline

Clean, minimal implementation of the morphological profile normalization pipeline.

## Structure

```
src/norm/
├── README.md                      # This file
├── run_pipeline.py                # Main pipeline orchestrator
├── aggregate_results.py           # Sweep results aggregation and visualization
├── utils.py                       # Metadata loading utility
├── visualization.py               # Dimensionality reduction plotting
├── conf/                          # Hydra configurations
│   ├── pipeline.yaml              # Base configuration
│   └── sweep/
│       └── correlation_sweep.yaml # Multi-parameter sweep config
├── data/
│   ├── __init__.py
│   └── load.py                    # Data I/O functions
├── operations/
│   ├── __init__.py
│   ├── select.py                  # Feature selection
│   ├── normalize.py               # Profile normalization
│   └── aggregate.py               # Profile aggregation
└── metrics/
    ├── __init__.py
    ├── phenotypic.py              # PA/PC metrics
    └── batch.py                   # Batch effect metrics
```

## Quick Start

### Single Run

```bash
# Using default configuration
python src/norm/run_pipeline.py

# Override parameters
python src/norm/run_pipeline.py \
  steps[4].params.threshold=0.9 \
  steps[5].params.method=robustmad
```

### Parameter Sweep

```bash
python src/norm/run_pipeline.py --multirun \
  hydra/launcher=joblib \
  hydra.launcher.n_jobs=8 \
  +sweep=correlation_sweep \
  corr_thresh=0.8,0.9,0.95 \
  var_thresh=0.01,0.05,0.1 \
  outlier_cut=50,100 \
  norm_method=standardize,robustmad
```

### Analyze Results

```bash
python src/norm/aggregate_results.py data/features/INPUT_NAME
```

## Pipeline Steps

1. **clean_nans** - Remove NaN/inf columns and rows
2. **merge_metadata** - Add JUMP metadata (compounds, genes, targets)
3. **filter_features** - Variance and outlier filtering
4. **prune_correlated** - Remove highly correlated features
5. **normalize** - Normalize profiles (standardize/robustmad/robustize/tvn/spherize)
6. **aggregate_wells** - Aggregate to well level
7. **evaluate_metrics** - Calculate PA, PC, Silhouette, kBET

## Key Features

### Feature Selection (`operations/select.py`)
- `variance_threshold` - Statistical variance filtering
- `correlation_threshold` - Greedy independent set for correlation removal
- `drop_na_columns` - Remove features with excessive NaN/inf
- `drop_outliers` - Z-score based outlier filtering
- `blocklist_filter` - Remove known problematic features

### Normalization (`operations/normalize.py`)
- **Standardize** - Z-score normalization
- **RobustMAD** - Median Absolute Deviation normalization
- **Robustize** - Robust scaling
- **TVN** - Typical Variation Normalization
- **Spherize** - ZCA/PCA whitening

All methods support:
- Batch-wise normalization (e.g., per-plate)
- Control-based fitting (fit on negative controls)

### Metrics (`metrics/`)
- **PA (Phenotypic Activity)** - % compounds with significant signal
- **PC (Phenotypic Consistency)** - # genes with consistent signal across perturbations
- **Silhouette Score** - Batch separation quality (lower = better correction)
- **kBET** - Batch mixing quality (lower = better mixing)

### Aggregation & Visualization
- **Sweep aggregation** - Collects metrics from all parameter combinations
- **Heatmaps** - Visual comparison of parameter effects
- **Consistent color scales** - Easy visual comparison across metrics
- **Logical grouping** - Parameters organized by relationship

## Configuration

### Base Config (`conf/pipeline.yaml`)
Defines default values for all parameters and pipeline steps.

### Sweep Config (`conf/sweep/correlation_sweep.yaml`)
Defines sweep variables using Hydra interpolation:

```yaml
corr_thresh: 0.9
var_thresh: 0.01
outlier_cut: 50
norm_method: standardize

steps:
  - name: prune_correlated
    params:
      threshold: ${corr_thresh}
```

## Output Structure

```
data/features/
└── INPUT_NAME/
    ├── CONFIG_NAME_1/
    │   ├── processed.parquet
    │   └── results/
    │       ├── metrics.json
    │       └── dimreduction.png
    ├── CONFIG_NAME_2/
    │   └── ...
    ├── aggregated_results.csv
    └── heatmaps/
        └── parameter_sweep_heatmaps.png
```

## Dependencies

Core requirements:
- polars >= 0.19.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.11.0
- scikit-learn >= 1.3.0
- hydra-core >= 1.3.0
- hydra-joblib-launcher >= 1.2.0
- matplotlib >= 3.0.0
- seaborn >= 0.13.2
- copairs >= 0.5.1
- broad-babel >= 0.1.24
- scib-metrics >= 0.5.7
- umap-learn >= 0.5.9

See `../../pyproject.toml` for complete list.

## Differences from Original

This is a cleaned, minimal version of `src/normalize/` with:
- ✅ Removed experimental/unused code
- ✅ Fixed all NaN/inf handling
- ✅ Added actual variance checking
- ✅ Improved heatmap visualization
- ✅ Clearer module structure
- ✅ Updated to use `norm` namespace

## Notes

- All feature filtering properly handles NaN and inf values
- Variance threshold includes statistical variance checking (not just frequency)
- Drop outliers uses z-score normalization
- Output directory names include all swept parameters for uniqueness
- Heatmaps use consistent color scales within each metric column
