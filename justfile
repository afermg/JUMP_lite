# JUMP_core analysis pipeline
# Run `just --list` to see all recipes
# Run `just <recipe>` to execute a step

# ─── Data Root ─────────────────────────────────────────────────
# All external data lives under DATA_ROOT. Override per-machine:
#   export DATA_ROOT=/my/other/path && just metadata
data_root            := env("DATA_ROOT", "/work/datasets")

# ─── Derived Paths (external data) ────────────────────────────
annotations_dir      := data_root / "jump_core" / "annotations"
annotations_db       := annotations_dir / "jump_metadata.duckdb"
annotations_cc       := annotations_dir / "annotations_compound_compound.parquet"
annotations_cg       := annotations_dir / "annotations_compound_gene.parquet"
refchemdb            := data_root / "annotations" / "refchemdb_conf_jump_matched.parquet"
cp_profiles          := data_root / "jump_core_annotated" / "raw_jump_CP_profiles" / "profiles.parquet"
raw_images           := data_root / "jump_target2_4plate" / "raw"
compressed_dir       := data_root / "jump_target2_4plate"
aliby_output         := data_root / "aliby_output"
aliby_target2_dl     := aliby_output / "plate4_rerun_scale_stdwork" / "datasets" / "aliby_output" / "plate4"

# ─── Derived Paths (repo-relative) ────────────────────────────
features_lite        := "data/features/jump_lite"
features_lite_2        := "data/features/jump_lite"
features_target2     := "data/features/jump_target2_4plate"
features_target2_cl  := "data/features/jump_target2_4plate_cl"
norm3_dir            := "src/norm_3"

# ─── Environment ───────────────────────────────────────────────

# Clone sibling repos (aliby, nahual) if not present, then sync deps
setup:
    #!/usr/bin/env bash
    set -euo pipefail
    for repo in aliby nahual; do
        if [ ! -d "../${repo}" ]; then
            echo "Cloning ${repo} into ../$(basename $(pwd))/../${repo}..."
            git clone "https://github.com/afermg/${repo}.git" "../${repo}"
        else
            echo "${repo} already present at ../${repo}"
        fi
    done
    echo "Syncing dependencies..."
    uv sync --all-groups

# Verify the development environment
check-env:
    python --version
    python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
    pixi --version

# ─── Step 0: Metadata ─────────────────────────────────────────

# Build the unified metadata dataset
metadata output_dir="metadata/":
    uv run python scripts/build_metadata_dataset.py \
        --annotations-db {{ annotations_db }} \
        --annotations-cc {{ annotations_cc }} \
        --annotations-cg {{ annotations_cg }} \
        --profiles {{ cp_profiles }} \
        --refchemdb {{ refchemdb }} \
        --output-dir {{ output_dir }} \
        --save-intermediates

# ─── Step 1: Image Compression ────────────────────────────────

# Batch compress all codecs (edit paths in script first)
compress-batch:
    uv run python src/compress_tif.py

# Compress with a single codec
compress codec="jpegxl_lossy_mq" jobs="16":
    uv run python src/compress_tif_single.py \
        --input {{ raw_images }} \
        --output {{ compressed_dir }} \
        --codec {{ codec }} \
        --n-jobs {{ jobs }}

# ─── Step 2: Image Quality ────────────────────────────────────

# Compute PSNR/SSIM quality metrics (100 sites, skip LPIPS, exclude mq_new and d50)
quality-metrics n_samples="100" exclude="jpegxl_lossy_mq_new jpegxl_lossy_d50":
    cd analysis/image_quality && uv run python compare_codecs.py \
        --data-dir {{ compressed_dir }} \
        --n-samples {{ n_samples }} \
        --skip-lpips \
        --exclude-codecs {{ exclude }}

# Compute sharpness-only metrics (Laplacian variance + Tenengrad, no GPU needed)
quality-sharpness n_samples="100" exclude="jpegxl_lossy_mq_new jpegxl_lossy_d50":
    cd analysis/image_quality && uv run python compare_codecs.py \
        --data-dir {{ compressed_dir }} \
        --n-samples {{ n_samples }} \
        --sharpness-only \
        --exclude-codecs {{ exclude }}

# Regenerate quality violin plots from existing CSV (reads from analysis/image_quality/output/)
quality-figures:
    cd analysis/image_quality && uv run python compare_codecs.py \
        --figures-only

# ─── Step 3: Segmentation Comparison ──────────────────────────

# Compare segmentation masks across codecs — runs cell, nuclei, then combined plots
segmentation-compare methods="jpegxl_lossy_hq.zarr jpegxl_lossy_effort_3.zarr jpegxl_lossy_d2_e8.zarr jpegxl_lossy_mq.zarr jpegxl_lossy_lq.zarr jpegxl_lossy_d10.zarr":
    uv run python analysis/segmentation/compare_segmentations.py \
        --root {{ aliby_output }}/cp_measure/jump_target2_4plate \
        --ground-truth zstd.zarr \
        --methods {{ methods }} \
        --segment-step segment_cell --fast --save-mappings
    uv run python analysis/segmentation/compare_segmentations.py \
        --root {{ aliby_output }}/cp_measure/jump_target2_4plate \
        --ground-truth zstd.zarr \
        --methods {{ methods }} \
        --segment-step segment_nuclei --fast --save-mappings
    uv run python analysis/segmentation/compare_segmentations.py \
        --root {{ aliby_output }}/cp_measure/jump_target2_4plate \
        --ground-truth zstd.zarr \
        --methods {{ methods }} \
        --both

# Compare segmentation with limited samples (faster)
segmentation-compare-sampled samples="1000" methods="jpegxl_lossy_hq.zarr jpegxl_lossy_effort_3.zarr jpegxl_lossy_d2_e8.zarr jpegxl_lossy_mq.zarr jpegxl_lossy_lq.zarr jpegxl_lossy_d10.zarr":
    uv run python analysis/segmentation/compare_segmentations.py \
        --root {{ aliby_output }}/cp_measure/jump_target2_4plate \
        --ground-truth zstd.zarr \
        --methods {{ methods }} \
        --segment-step segment_cell --fast --save-mappings --samples {{ samples }}
    uv run python analysis/segmentation/compare_segmentations.py \
        --root {{ aliby_output }}/cp_measure/jump_target2_4plate \
        --ground-truth zstd.zarr \
        --methods {{ methods }} \
        --segment-step segment_nuclei --fast --save-mappings --samples {{ samples }}
    uv run python analysis/segmentation/compare_segmentations.py \
        --root {{ aliby_output }}/cp_measure/jump_target2_4plate \
        --ground-truth zstd.zarr \
        --methods {{ methods }} \
        --both

# Quick segmentation test (50 samples, cell only)
segmentation-quick:
    uv run python analysis/segmentation/compare_segmentations.py \
        --root {{ aliby_output }}/cp_measure/jump_target2_4plate \
        --ground-truth zstd.zarr \
        --methods jpegxl_lossy_hq.zarr jpegxl_lossy_mq.zarr \
        --segment-step segment_cell --fast --samples 50

# Generate combined IoU vs file-size plot
segmentation-iou-plot:
    uv run python analysis/segmentation/plot_segmentation_iou.py

# Generate per-cell IoU distribution plots
segmentation-cell-iou mappings_dir="analysis/segmentation/output/segmentation_comparison/instance_mappings":
    uv run python analysis/segmentation/plot_cell_level_iou.py \
        --mappings-dir {{ mappings_dir }}

# 3b. Create cell-count baseline features from CP parquets (target2: 7 codecs, jump_lite: 1 file)
create-cell-count-features:
    uv run python scripts/create_cell_count_features.py

# ─── Step 4: Feature Extraction ───────────────────────────────

# 4a. Extract DL features for target2 (morphem, openphenom × all codecs)
extract-dl-target2:
    #!/usr/bin/env bash
    set -euo pipefail
    for model in dinov2 dinov2_random morphem openphenom openphenom_nonclip openphenom_stdscale openphenom_stdscale_false subcell subcell__nonstd subcell_nonstd subcell_wrongchannels; do
        echo "=== Extracting $model for target2 ==="
        uv run python src/extract_features.py \
            --input /work/datasets/aliby_output/plate4_rerun_scale_std \
            --output /work/users/jfredinh/projects/JUMP_core/data/features/jump_target2_4plate_cl_2 \
            --model "$model" \
            --dataset jump_target2_4plate
    done

# 4b. Extract CellProfiler features for target2 (all codecs)
extract-cp-target2:
    uv run python src/extract_features.py \
        --input {{ aliby_output }} \
        --output {{ features_target2 }} \
        --model cp_measure \
        --dataset jump_target2_4plate

# 4c. Extract DL features for jump_lite (all DL models, mq codec)
extract-dl-lite:
    #!/usr/bin/env bash
    set -euo pipefail
    for model in dinov2 morphem openphenom subcell; do
        echo "=== Extracting $model for jump_lite ==="
        uv run python src/extract_features.py \
            --input {{ aliby_output }} \
            --output {{ features_lite }} \
            --model "$model" \
            --dataset jump_lite_updated
    done

# 4d. Reformat raw CellProfiler profiles for jump_lite
reformat-cp:
    uv run python src/reformat_raw_cp_profiles.py \
        --source {{ cp_profiles }} \
        --metadata metadata/metadata_dataset_filtered_4reps.parquet \
        --output {{ features_lite }}/cellprofiler_raw_jump_lite_raw_features.parquet

# Extract single model/codec (manual)
extract-features model codec output="data/features/jump_lite":
    uv run python src/extract_features.py \
        --input {{ aliby_output }} \
        --output {{ output }} \
        --model {{ model }} \
        --compression {{ codec }}

# ─── Step 4-fast: Feature Extraction (Parallel) ──────────────

# 4a-fast. Extract DL features for target2 (parallel, all models at once)
extract-dl-target2-fast jobs="-1":
    uv run python src/extract_features_fast.py \
        --input /work/datasets/aliby_output/plate4_rerun_scale_std \
        --output /work/users/jfredinh/projects/JUMP_core/data/features/jump_target2_4plate_cl_2 \
        --dataset jump_target2_4plate \
        --model morphem \
        --n-jobs {{ jobs }}

# 4b-fast. Extract CellProfiler features for target2 (parallel across codecs)
extract-cp-target2-fast jobs="-1":
    uv run python src/extract_features_fast.py \
        --input {{ aliby_output }} \
        --output {{ features_target2 }} \
        --model cp_measure \
        --dataset jump_target2_4plate \
        --n-jobs {{ jobs }}

# 4c-fast. Extract DL features for jump_lite (parallel, all models at once)
extract-dl-lite-fast jobs="-1":
    uv run python src/extract_features_fast.py \
        --input {{ aliby_output }}/jump_lite_rerun \
        --output {{ features_lite_2 }} \
        --dataset jump_lite_updated \
        --n-jobs {{ jobs }}

# Extract single model/codec (fast, manual)
extract-features-fast model codec output="data/features/jump_lite" jobs="-1":
    uv run python src/extract_features_fast.py \
        --input {{ aliby_output }} \
        --output {{ output }} \
        --model {{ model }} \
        --compression {{ codec }} \
        --n-jobs {{ jobs }}

# Feature similarity: CellProfiler correlation heatmaps
feature-correlation-cp:
    uv run python analysis/feature_similarity/feature_correlation_cp_measure_script.py

# Feature similarity: correlation vs raw CellProfiler
feature-correlation-raw:
    uv run python analysis/feature_similarity/correlate_vs_raw_cp.py

# Feature similarity: per-cell codec comparison
feature-codec-compare mappings_dir="output/instance_mappings" n_samples="500":
    uv run python analysis/feature_similarity/compare_codec_features.py \
        --mappings-dir {{ mappings_dir }} --site-level --n-samples {{ n_samples }}

# Cross-well feature consistency for same-treatment replicates
feature-cross-well metadata="metadata/metadata.parquet":
    uv run python analysis/feature_similarity/compare_cross_well_features.py \
        --features-base {{ aliby_output }}/cp_measure/jump_target2_4plate \
        --metadata {{ metadata }}

# ─── Step 5: Normalization — jump_lite (v9) ───────────────────

# Single normalization run (test)
norm-single input:
    cd {{ norm3_dir }} && pixi run python pipeline.py \
        +preset=gpu_base_variance_first_v9 \
        input.path={{ input }}

# Single sweep config for one model
norm-sweep-one input sweep="focused_dl_v9_none" jobs="4":
    cd {{ norm3_dir }} && pixi run python pipeline.py --multirun \
        +sweep={{ sweep }} \
        input.path={{ input }} \
        hydra/launcher=joblib hydra.launcher.n_jobs={{ jobs }}

# Full v9 sweep: 5 models, 4 GPUs, ~500 configs
sweep-v9:
    bash run_focused_v9_sweep.sh

# Quick v9 test: CellProfiler + MorphEm, 2 GPUs
sweep-v9-test:
    bash run_v9_test_cp_morphem.sh

# Monitor running v9 sweep logs
sweep-v9-monitor:
    tail -f logs/sweep_v9/*.log

# ─── Step 5-target2: Normalization — target2 (v6) ─────────────

# Full target2 sweep: CP (15 datasets x 54) + DL (110 datasets x 336)
sweep-target2:
    bash run_focused_v6_sweep.sh

# Single target2 CP sweep
sweep-target2-cp input jobs="8":
    cd {{ norm3_dir }} && pixi run python -m norm_3.pipeline --multirun \
        +sweep=focused_cp_v6 \
        input.path={{ input }} \
        hydra/launcher=joblib hydra.launcher.n_jobs={{ jobs }}

# Single target2 DL sweep
sweep-target2-dl input jobs="8":
    cd {{ norm3_dir }} && pixi run python -m norm_3.pipeline --multirun \
        +sweep=focused_dl_v6 \
        input.path={{ input }} \
        hydra/launcher=joblib hydra.launcher.n_jobs={{ jobs }}

# ─── Step 5-target2-v10: Normalization — target2 (v10) ─────────

# Full target2 v10 sweep: 4 DL models (10 codecs) + CP (7 codecs), 4 GPUs
sweep-v10:
    bash run_focused_v10_sweep.sh

# Monitor running v10 sweep logs
sweep-v10-monitor:
    tail -f logs/sweep_v10/*.log

# Full target2 v11 sweep: 5 DL models (median agg, cl_3) + CP + cell_count, 3 GPUs
sweep-v11:
    bash run_focused_v11_sweep.sh

# Monitor running v11 sweep logs
sweep-v11-monitor:
    tail -f logs/sweep_v11/*.log

# Full jump_lite v11 lite sweep: 5 DL + CP + cell_count, 3 GPUs
sweep-v11-lite:
    bash run_focused_v11_lite_sweep.sh

# Monitor running v11 lite sweep logs
sweep-v11-lite-monitor:
    tail -f logs/sweep_v11_lite/*.log

# ─── Step 6: Results Aggregation & Figures ─────────────────────

# Aggregate v9 sweep results with plots
results-v9:
    cd {{ norm3_dir }} && pixi run python gather_sweep_results.py \
        --sweep-dir data/features/variance_first_v9 \
        --plot --filter-degenerate \
        --best-metric nap_balanced \
        --exclude-codecs mq_new d20_e2 d50

# Aggregate target2 (v6) sweep results with plots
results-target2:
    cd {{ norm3_dir }} && pixi run python gather_sweep_results.py \
        --sweep-dir data/features/variance_first_v6 \
        --plot --filter-degenerate \
        --best-metric nap_balanced \
        --exclude-codecs mq_new d20_e2 d50

# Aggregate target2 v10 sweep results with plots
results-v10:
    cd {{ norm3_dir }} && pixi run python gather_sweep_results.py \
        --sweep-dir data/features/variance_first_v10 \
        --plot --filter-degenerate \
        --best-metric nap_balanced \
        --exclude-codecs mq_new d20_e2 d50

# Aggregate target2 v11 sweep results with plots
results-v11:
    cd {{ norm3_dir }} && pixi run python gather_sweep_results.py \
        --sweep-dir data/features/variance_first_v11 \
        --plot --filter-degenerate \
        --best-metric nap_balanced \
        --exclude-codecs mq_new d20_e2 d50

# Aggregate jump_lite v11 lite sweep results with plots
results-v11-lite:
    cd {{ norm3_dir }} && pixi run python gather_sweep_results.py \
        --sweep-dir data/features/variance_first_v11_lite \
        --plot --filter-degenerate \
        --best-metric nap_balanced \
        --exclude-codecs mq_new d20_e2 d50

# Aggregate any sweep dir with plots
results sweep_dir:
    cd {{ norm3_dir }} && pixi run python gather_sweep_results.py \
        --sweep-dir {{ sweep_dir }} \
        --plot --filter-degenerate

# Aggregate with custom best-metric selection
results-best sweep_dir metric="nap_balanced":
    cd {{ norm3_dir }} && pixi run python gather_sweep_results.py \
        --sweep-dir {{ sweep_dir }} \
        --plot --filter-degenerate \
        --best-metric {{ metric }}

# Aggregate restricted to 5 families (cp_measure + dinov2 + morphem + openphenom + subcell__clip01)
# and 6 codecs (hq, e3, d2_e8, mq, lq, d10) — i.e. cp_measure's codec lineup minus d15/d30.
# Output goes to <sweep_dir>/plots_5fam_cp_codecs/ to keep separate from the default plots/.
results-5fam-cp-codecs sweep_dir:
    cd {{ norm3_dir }} && pixi run python gather_sweep_results.py \
        --sweep-dir {{ sweep_dir }} \
        --plot-dir {{ sweep_dir }}/plots_5fam_cp_codecs \
        --plot --filter-degenerate \
        --best-metric nap_balanced \
        --exclude-families cell_count dinov2_random \
        --exclude-codecs mq_new d20_e2 d50 d15 d30

# ─── Auxiliary ─────────────────────────────────────────────────

# Compression parameter exploration (JPEG XL distance vs effort)
compression-explore:
    uv run python analysis/compression_exploration/explore.py

# Interactive sphering demo (Marimo)
sphering-demo:
    uv run marimo run scripts/sphering_demo.py

# Gather key figures into main_figures/
gather-figures:
    bash scripts/gather_figures.sh
