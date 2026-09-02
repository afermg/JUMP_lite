# Pipeline Steps

## Step 1: Image Compression
- **Script:** `src/compress_tif_release.py` (`src/compress_tif.py` is retained unchanged as historical analysis provenance)
- **Input:** Raw unsigned 16-bit TIFF microscopy images named `<source>__<batch>__<plate>__<well>__<site>__<channel>.tif`
- **Output:** One site-major Zarr array `(5, H, W)` per site, in canonical channel order AGP, DNA, ER, Mito, RNA
- **Release codecs:** Zstd plus JPEG XL HQ (distance 1), MQ (distance 3), and D20 (distance 20)
- **Validation:** The CLI rejects malformed names, duplicate or missing channels, non-2-D or non-uint16 TIFFs, unknown codecs, and any site compression failure
- **Args:**
  - `--input <path>` input directory containing TIFF files
  - `--output <path>` output directory for codec Zarr stores
  - `--codec <name>` codec name; run `--help` for the complete exploration-codec set
  - `--overwrite` explicitly replace the selected codec store
  - `--n-jobs <N>` parallel site workers (default: 16)
  - `--no-skip-existing` fail if a site already exists instead of validating and skipping it; this option does not replace data
- **Run:** `uv run python src/compress_tif_release.py --input data/raw --output data/compressed --codec jpegxl_lossy_hq`
- **Bounded release smoke test:** `just dataset-smoke <jump_lite_site_index.parquet>` validates the complete release index and regenerates one five-channel site per release source under all four release codecs. See `REPRODUCE.md`.

## Step 1c: Feature Extraction
- **Script:** `src/extract_features.py`
- **Input:** Feature profiles from aliby_output directory tree (`MODEL/DATASET/COMPRESSION/profiles/*.parquet`)
- **Output:** Well-level aggregated features parquet (`{model}_{dataset}_{compression}_raw_features.parquet`)
- **Deps:** duckdb, polars, trommel
- **Args:**
  - `--input <path>` aliby_output directory
  - `--output <path>` Output directory
  - `--model <name>` Model name filter
  - `--compression <name>` Compression name filter
  - `--dataset <name>` Dataset name filter
  - `--cache-dir <path>` DuckDB cache directory
  - `--filter-border-cells` Exclude cells touching image borders
- **Run:** `nix develop . uv run python src/extract_features.py --input data/aliby_output --output output/ --model cp_measure --compression zstd.zarr`

- **Script:** `src/extract_features_with_size_filter.py`
- **Purpose:** Same as extract_features.py but with additional cell size filtering
- **Additional args:**
  - `--filter-size` Enable size-based filtering
  - `--min-nuclei-diameter <px>` Minimum nuclei diameter
  - `--min-cell-diameter <px>` Minimum cell diameter
- **Run:** `nix develop . uv run python src/extract_features_with_size_filter.py --input data/aliby_output --output output/ --model cp_measure --filter-border-cells --filter-size`

## Step 1d: Reformat Raw CellProfiler Profiles
- **Script:** `src/reformat_raw_cp_profiles.py`
- **Input:** Raw CellProfiler profiles parquet + metadata parquet with wells of interest
- **Output:** Reformatted parquet with standardized `Metadata_*` columns and model/compression tags
- **Args:**
  - `--source <path>` (required) Source profiles parquet
  - `--metadata <path>` (required) Metadata parquet with wells of interest
  - `--output <path>` (required) Output parquet file
  - `--model <name>` Metadata_model value (default: `cellprofiler_raw`)
  - `--dataset <name>` Metadata_dataset value (default: `jump_core_annotated`)
  - `--compression <name>` Metadata_compression value (default: `none`)
- **Run:** `nix develop . uv run python src/reformat_raw_cp_profiles.py --source /path/to/profiles.parquet --metadata metadata/metadata_dataset_filtered_4reps.parquet --output output/reformatted.parquet`

## Step 2: Image Quality Assessment
- **Script:** `analysis/image_quality/compare_codecs.py`
- **Input:** Compressed zarr files (lossy codecs) + zstd reference zarr
- **Output:** `quality_metrics.csv`, violin plots (PSNR, SSIM, LPIPS)
- **Deps:** see `analysis/image_quality/pyproject.toml` (torch, torchmetrics, lpips, zarr, imagecodecs)
- **Args:** `--data-dir <path>` (default: `data/jump_target2_4plate`), `--figures-only` (skip computation, plot from existing CSV)
- **Run:** `cd analysis/image_quality && uv run python compare_codecs.py --data-dir /path/to/zarr/files`

## Auxiliary: Compression Parameter Exploration
- **Script:** `analysis/compression_exploration/explore.py`
- **Purpose:** Auxiliary exploration of JPEG XL compression parameters (distance vs effort grid). Not part of the main pipeline — used for ad-hoc investigation of compression trade-offs.
- **Input:** Raw TIF images + compressed zarr files
- **Output:** Comparison plots, histograms
- **Args:** `--hist-only` (only run histogram + peak comparison)
- **Run:** `nix develop . uv run python analysis/compression_exploration/explore.py`

## Step 3: Segmentation Comparison
- **Script:** `analysis/segmentation/compare_segmentations.py`
- **Input:** Segmentation masks from aliby_output for each codec + ground truth (zstd)
- **Output:** Per-site IoU, Dice, F1, panoptic quality CSVs; boxen/violin plots; sample visualizations
- **Deps:** numpy, scipy, medpy, polars, matplotlib, seaborn, cellpose, PIL, zarr, imagecodecs
- **Args:**
  - `--root <path>` (required) Root directory containing all methods
  - `--ground-truth <name>` (required) Ground truth method name (e.g., `zstd.zarr`)
  - `--methods <name> [<name> ...]` (required) Methods to compare
  - `--output <prefix>` Output file prefix (default: `segmentation_comparison`)
  - `--segment-step <name>` `segment_cell` or `segment_nuclei` (default: `segment_cell`)
  - `--both` Process both cell and nuclei together
  - `--workers <N>` Parallel workers (default: 8)
  - `--fast` Skip expensive metrics (hausdorff, asd) for ~2-3x speedup
  - `--save-mappings` Save instance ID mappings to parquet
  - `--filter-percentile <N>` Filter outlier wells by cell count percentile
  - `--samples <N>` Limit to N samples for quick testing
  - `--visualize-sample` / `--visualize-sample-grid` + `--well <id>` Single sample visualization
- **Run:** `nix develop . uv run python analysis/segmentation/compare_segmentations.py --root data/aliby_output/cp_measure/jump_target2_4plate --ground-truth zstd.zarr --methods jpegxl_lossy_hq.zarr jpegxl_lossy_mq.zarr --both --fast`
- **Utility:** `analysis/segmentation/instance_matching.py` — Instance matching between reference and compressed masks (imported by compare_segmentations.py)

## Step 3b: Segmentation Plotting
- **Script:** `analysis/segmentation/plot_segmentation_iou.py`
- **Input:** CSV outputs from Step 3
- **Output:** Combined IoU/Dice violin and boxen plots
- **Run:** `nix develop . uv run python analysis/segmentation/plot_segmentation_iou.py`

- **Script:** `analysis/segmentation/plot_cell_level_iou.py`
- **Input:** Instance mapping parquets from Step 3 (`--save-mappings`)
- **Output:** Per-cell IoU distribution plots
- **Args:** `--mappings-dir <path>` (required), `--output <prefix>`, `--thresh <float>` (default: 0.5)
- **Run:** `nix develop . uv run python analysis/segmentation/plot_cell_level_iou.py --mappings-dir output/segmentation_comparison_with_mapping/instance_mappings`

## Auxiliary: Segmentation Visualization & Validation
- **Script:** `analysis/segmentation/validate_feature_mask_alignment.py`
- **Purpose:** Verify that extracted features correspond to the correct segmentation masks. Spot-check alignment.
- **Args:** `--base-path <path>`, `--codec <name>`, `--object-type cell|nuclei`, `--n-samples <N>`, `--output <path>`
- **Run:** `nix develop . uv run python analysis/segmentation/validate_feature_mask_alignment.py --base-path data/aliby_output/cp_measure/jump_target2_4plate`

- **Script:** `analysis/segmentation/visualize_cell_compression.py`
- **Purpose:** Visualize how individual cells look across compression levels. Useful for qualitative assessment.
- **Args:** `--mappings-dir <path>` (required), `--zarr-root <path>`, `--masks-root <path>`, `--gt-method <name>`, `--n-samples <N>`, `--seed <int>`
- **Run:** `nix develop . uv run python analysis/segmentation/visualize_cell_compression.py --mappings-dir output/instance_mappings`

- **Script:** `analysis/segmentation/interactive_cell_count_viewer.py`
- **Purpose:** Interactive Panel dashboard for browsing cell count differences with images and masks.
- **Args:** `--csv <path>` (required), `--mask-root <path>` (required), `--zarr-root <path>`, `--port <N>`
- **Run:** `nix develop . uv run python analysis/segmentation/interactive_cell_count_viewer.py --csv output/large_cell_count_diff.csv --mask-root data/aliby_output/cp_measure/jump_target2_4plate`

- **Script:** `analysis/segmentation/segmentation_dashboard.py`
- **Purpose:** Interactive Panel dashboard for exploring segmentation comparison results with instance mappings.
- **Args:** `--mappings-dir <path>`, `--zarr-root <path>`, `--masks-root <path>`, `--port <N>`
- **Run:** `nix develop . uv run python analysis/segmentation/segmentation_dashboard.py --mappings-dir output/instance_mappings`

- **Script:** `analysis/segmentation/test_viewer_simple.py`
- **Purpose:** Simple Panel viewer test for debugging the dashboard setup.

## Step 4: Feature Similarity Analysis
- **Script:** `analysis/feature_similarity/feature_correlation_cp_measure_script.py`
- **Input:** Feature profiles from aliby_output (per compression), compound metadata from `input/JUMP-Target-2_compound_metadata.tsv`
- **Output:** Correlation heatmaps, violin plots, parquet correlation results
- **Config:** Paths to aliby_output workspace and cache directory are set within the script
- **Utility:** `analysis/feature_similarity/utils_cp_measure_name_mapping.py` — Maps between CP measure naming conventions (imported by main script)
- **Run:** `nix develop . uv run python analysis/feature_similarity/feature_correlation_cp_measure_script.py`

- **Script:** `analysis/feature_similarity/correlate_vs_raw_cp.py`
- **Input:** Raw CellProfiler profiles + normalized features from the pipeline (filtered and non-filtered)
- **Output:** Spearman/Pearson correlation scatter plots, violin plots, bar charts per feature category
- **Config:** Hardcoded paths to raw CP profiles, filtered/non-filtered feature directories, and output directory — edit `RAW_CP_PATH`, `FILTERED_DIR`, `NONFILTERED_DIR`, `OUTPUT_DIR` in the script
- **Note:** Imports `scripts/map_cellprofiler_features.py:FeatureMapper` via sys.path manipulation
- **Run:** `nix develop . uv run python analysis/feature_similarity/correlate_vs_raw_cp.py`

- **Script:** `analysis/feature_similarity/compare_codec_features.py`
- **Input:** Instance mapping parquets from Step 3 (`--save-mappings`) + per-cell feature profiles from aliby_output
- **Output:** Per-cell and per-site feature correlation plots/CSVs across codecs, feature ranking
- **Args:**
  - `--mappings-dir <path>` (required) Directory with instance mapping parquet files
  - `--features-base <path>` Base path for feature profiles (default: `data/aliby_output/cp_measure/jump_target2_4plate`)
  - `--gt-codec <name>` Ground truth codec (default: `zstd.zarr`)
  - `--codecs <name> [...]` Codecs to compare (default: jpegxl variants)
  - `--object-type cell|nuclei` (default: `cell`)
  - `--site-level` Also run site-level analysis (median of matched cells per site)
  - `--n-samples <N>` Number of random source_ids to sample (default: 5)
  - `--features <name> [...]` / `--feature-pattern <regex>` Filter specific features
  - `--list-features` List available features and exit
  - `--min-cells <N>` Minimum GT cell count per site (default: 5)
  - `--filter-percentile <N>` Filter outlier sites by cell count percentile
- **Run:** `nix develop . uv run python analysis/feature_similarity/compare_codec_features.py --mappings-dir output/instance_mappings --site-level`

### Input data
- `analysis/feature_similarity/input/JUMP-Target-2_compound_metadata.tsv` — Compound metadata
- `analysis/feature_similarity/input/JUMP-Target-2_compound_platemap.tsv` — Plate map metadata

## Step 5: Normalization Pipeline (GPU)
- **Script:** `src/norm_3/pipeline.py`
- **Purpose:** GPU-accelerated normalization pipeline for Cell Painting morphological profiles. Runs an ordered sequence of configurable steps: clean NaNs, merge metadata, filter features, prune correlated, normalize (RobustMAD/standardize), batch correction (TVN/TVN_EFAAR), spherize, PCA, well position correction, inverse normal transform, aggregate wells, evaluate metrics.
- **Input:** Raw feature parquets from Step 1c (extract_features)
- **Output:** Normalized profiles parquet, `metrics.json` (PA, PC), pipeline config, per-compound/target CSVs
- **Deps:** RAPIDS stack (`cupy`, `cuml`) via `pixi.toml`, plus polars, scipy, copairs, hydra, omegaconf, scikit-learn
- **Config:** Hydra-based — `src/norm_3/conf/pipeline.yaml` (base), `conf/preset/` (per-model/compression), `conf/sweep/` (parameter search)
- **Modules:** `core.py` (GPU transformers), `io.py` (data loading), `linalg.py` (GPU linear algebra), `utils.py` (GPU memory), `config.py` (dataclasses), `metrics.py` (PA/PC evaluation)
- **Single run:**
  ```
  cd src/norm_3 && pixi run python pipeline.py +preset=gpu_base input_override=/path/to/raw_features.parquet
  ```
- **Sweep (Hydra multirun):**
  ```
  cd src/norm_3 && pixi run python pipeline.py --multirun +preset=gpu_base +sweep=focused_cp_v6 input_override=/path/to/raw_features.parquet
  ```

## Step 6: Sweep Results Aggregation
- **Script:** `src/norm_3/gather_sweep_results.py`
- **Input:** Sweep output directory from Step 5 (contains `metrics.json` files across all sweep configurations)
- **Output:** `sweep_results.csv` (combined metrics), optional visualization plots (PA vs PC scatter, per-model bar charts)
- **Args:**
  - `--sweep-dir <path>` (required) Path to sweep output directory
  - `--output <path>` Output CSV path (default: `sweep_results.csv` in sweep-dir)
  - `--plot` Generate visualization plots
  - `--plot-dir <path>` Directory for plots (default: sweep-dir/plots)
  - `--filter-degenerate` Filter out degenerate configs (spherize + no PCA)
  - `--best-metric balanced|nap_balanced` Metric for selecting best config
  - `--exclude-families <name> [...]` Exclude model families from plots
  - `--exclude-codecs <name> [...]` Exclude codecs from plots
- **Run:** `nix develop . uv run python src/norm_3/gather_sweep_results.py --sweep-dir data/features/my_sweep --plot`

## Sweep Runner Scripts
Batch scripts that orchestrate Step 5 across many datasets and codecs, with GPU memory cleanup between runs.

- **Script:** `run_focused_v6_sweep.sh`
- **Purpose:** Latest comprehensive sweep. Two parts: CellProfiler models (15 datasets × 54 configs via `focused_cp_v6`) + Deep Learning models (35 datasets × 12 configs via `focused_dl_v6`). Covers CP raw, CP filtered-border-size, DINOv2, SubCell, MorphEm, OpenPhenom across 7-9 JPEG-XL distance levels.
- **Run:** `bash run_focused_v6_sweep.sh`

- **Script:** `run_variance_first_v5_cl_filtered_sweep.sh`
- **Purpose:** Variance-first pipeline ordering (filter→normalize→prune→PCA→batch-correct). 52 datasets × 20 configs via `simple_cellprofiler_variance_first_v5`. Covers all model families (CP, DINOv2, SubCell, MorphEm, OpenPhenom) across all codecs.
- **Run:** `bash run_variance_first_v5_cl_filtered_sweep.sh`

- **Script:** `run_variance_first_v5_CP1_annotations_sweep.sh`
- **Purpose:** Same as v5 but evaluates using CPJUMP1 primary target annotations (Chandrasekaran et al. 2024) — 130 curated targets with 2 compounds each — instead of the full target list. 43 datasets × 20 configs.
- **Run:** `CUDA_VISIBLE_DEVICES=3 bash run_variance_first_v5_CP1_annotations_sweep.sh`

- **Script:** `sweep_runner_single_loop.sh`
- **Purpose:** Generic sweep orchestrator using norm_2 (CPU pipeline). Loops over models × compressions, runs Optuna sweeps, aggregates results, and generates summary. Template for building new sweep scripts.
- **Run:** `bash sweep_runner_single_loop.sh`

## Step 0: Build Metadata Dataset
- **Script:** `scripts/build_metadata_dataset.py`
- **Purpose:** Unified 6-step pipeline that generates the final filtered metadata dataset. Combines logic from `standardize_annotations.py`, `download_images.py`, `analyze_metadata.py`, `prepare_negative_controls.py`, `compare_metadata_profiles.py`, and `compare_compound_overlap.py`.
- **Input:** MOTIVE annotation databases, JUMP perturbation lists (downloaded), raw CellProfiler profiles, RefChemDB annotations
- **Output:** `metadata_dataset_filtered_4reps.parquet` (wells with ≥4 compound replicates, plus ORF/CRISPR/negcons)
- **Deps:** polars, duckdb, broad_babel, jump_portrait, pooch
- **Args:**
  - `--annotations-db <path>` JUMP metadata DuckDB
  - `--annotations-cc <path>` MOTIVE compound-compound parquet
  - `--annotations-cg <path>` MOTIVE compound-gene parquet
  - `--profiles <path>` Raw CellProfiler profiles parquet
  - `--refchemdb <path>` RefChemDB annotations parquet (optional, for overlap stats)
  - `--output-dir <path>` Output directory
  - `--skip-to <1-6>` Resume from a specific step (requires intermediates)
  - `--save-intermediates` Save per-step parquet outputs
  - `--min-fill-rate <float>` Minimum plate fill rate (default: 0.25)
  - `--min-replicates <int>` Minimum replicates per compound (default: 4)
  - `--seed <int>` Random seed for negative control sampling (default: 42)
- **Run:** `nix develop . uv run python scripts/build_metadata_dataset.py --annotations-db data/annotations/jump_metadata.duckdb --annotations-cc data/annotations/annotations_compound_compound.parquet --annotations-cg data/annotations/annotations_compound_gene.parquet --profiles data/raw/raw_jump_CP_profiles/profiles.parquet --output-dir metadata/ --save-intermediates`

- **Notebook:** `scripts/04_refchemdb_match.ipynb`
- **Purpose:** Generates `refchemdb_conf_jump_matched.parquet` — the RefChemDB target annotation file used as optional input to `build_metadata_dataset.py` Step 6 (via `--refchemdb`). Filters RefChemDB to gene targets with confident interactions, adds CrossModalityTier and WithinModalityTier classifications, and matches compound mode with perturbation modality.
- **Input:** RefChemDB overlap CSV (`ref_chem_overlap.csv`), JUMP perturbation metadata
- **Output:** `refchemdb_conf_jump_matched.parquet`

## Utility: CellProfiler Feature Mapping
- **Script:** `src/utils/map_cellprofiler_features.py`
- **Purpose:** Maps CellProfiler features between traditional naming (`Compartment_Category_FeatureName_Parameters`) and cp_measure naming (`compartment_channel/aggregation/featuretype_FeatureName`). Provides `FeatureMapper` class used by `analysis/feature_similarity/correlate_vs_raw_cp.py`.
- **Deps:** polars, pandas

## Step 7: Phenotypic Activity & Consistency Evaluation
- **Script:** `evaluation/evaluate_phenotypic_activity.py`
- **Purpose:** Evaluates Phenotypic Activity (compound replicate retrieval via copairs mAP) and Phenotypic Consistency (target-based compound retrieval) per compound. Also calls cross-modality retrieval.
- **Input:** Normalized profile parquet from Step 5/6, metadata, RefChemDB annotations
- **Output:** Per-compound PA/PC metrics JSON, evaluation summary, CSVs
- **Deps:** polars, copairs, numpy
- **Args:**
  - `--input <path>` (required) Profiles parquet
  - `--annotations <path>` RefChemDB annotations (default: `metadata/refchemdb_conf_jump_matched.parquet`)
  - `--metadata <path>` Metadata parquet (default: `metadata/metadata_dataset.parquet`)
  - `--output <path>` Output directory
  - `--null-size <N>` Null distribution size (default: 10000)
  - `--p-threshold <float>` Significance threshold (default: 0.05)
  - `--min-compounds-per-target <N>` Min compounds per target for PC (default: 3)
  - `--skip-batch-effects` Skip batch effect calculation
  - `--skip-cross-modality` Skip cross-modality retrieval
  - `--no-annotations` Skip merging annotations (pre-annotated data)
  - `--no-metadata` Skip merging metadata (data already has compound IDs)
  - `--include-genetic-pairs` Include ORF vs CRISPR cross-modality
- **Run:** `nix develop . uv run python evaluation/evaluate_phenotypic_activity.py --input data/features/output.parquet`

- **Script:** `evaluation/evaluate_cross_modality_retrieval.py`
- **Purpose:** Cross-modality retrieval — ranks ORF/CRISPR profiles by cosine similarity to compound profiles based on shared targets. Calculates recall@k%. Called by `evaluate_phenotypic_activity.py`.
- **Deps:** numpy, polars

## Step 8: Batch Effect Analysis
- **Script:** `evaluation/analyze_batch_effects.py`
- **Purpose:** Analyzes well position and plate batch effects across sweep output parquets using copairs-based similarity. Uses random sampling for speed.
- **Input:** Sweep output directory containing `output.parquet` files
- **Output:** `batch_effects.csv` summary + per-config `batch_effects.json`
- **Deps:** polars, numpy, copairs
- **Args:**
  - `--sweep-dir <path>` Sweep output directory (default: `src/norm_3/data/features/unified_batch_sweep`)
  - `--output <path>` Output CSV path
  - `--seed <int>` Random seed (default: 42)
  - `--no-individual` Don't save per-config JSON files
- **Run:** `nix develop . uv run python evaluation/analyze_batch_effects.py --sweep-dir data/features/my_sweep`
