# JUMP_core analysis pipeline (v11 / v11_lite)
# Run `just --list` to see all recipes
# Run `just <recipe>` to execute a step

# ═══════════════════════════════════════════════════════════════
# Section 1: Configuration Variables
# ═══════════════════════════════════════════════════════════════

# ─── Data root ────────────────────────────────────────────────
# All external data lives under DATA_ROOT. Override per-machine:
#   export DATA_ROOT=/my/other/path && just metadata
data_root            := env("DATA_ROOT", "./data")

# ─── External data paths ─────────────────────────────────────
annotations_dir      := data_root / "annotations"
annotations_db       := annotations_dir / "jump_metadata.duckdb"
annotations_cc       := annotations_dir / "annotations_compound_compound.parquet"
annotations_cg       := annotations_dir / "annotations_compound_gene.parquet"
refchemdb_dir        := "data/refchemdb"
refchemdb_raw        := refchemdb_dir / "refchemdb_inchikey.parquet"
refchemdb_overlap    := refchemdb_dir / "ref_chem_overlap.parquet"
refchemdb            := refchemdb_dir / "refchemdb_conf_jump_matched.parquet"
cp_profiles          := data_root / "jump_core_annotated" / "raw_jump_CP_profiles" / "profiles.parquet"
raw_images_target2   := data_root / "jump_lite/archive/jump_target2_4plate_bak" / "raw"
compressed_target2   := data_root / "jump_target2_4plate"
raw_images_lite      := data_root / "jump_lite" / "imgs" / "raw"
compressed_lite      := data_root / "jump_lite" / "imgs"
aliby_output         := env("ALIBY_OUTPUT", data_root / "jump_lite" / "aliby_output")
manifest_dir         := data_root / "manifest"

# ─── Dataset-specific paths ──────────────────────────────────
# target2
target2_aliby_dl     := aliby_output / "plate4_rerun_scale_std"
features_target2_cp  := "data/features/jump_target2_4plate"
features_target2_dl  := "data/features/jump_target2_4plate_cl_3"

# jump_lite
lite_aliby_dl        := aliby_output / "jump_lite_rerun"
features_lite_cp     := "data/raw/features/jump_lite"
features_lite_dl     := "data/raw/features/jump_lite_cl_3"

# ─── Normalization & sweep paths ─────────────────────────────
norm3_dir            := "src/norm_3"
sweep_v11_dir        := "data/features/variance_first_v11"
sweep_v11_lite_dir   := "data/intermediate/sweep_v11_lite"

# ─── Output paths (centralised under data/results) ───────────
results_figures      := "data/results/figures"
results_tables       := "data/results/tables"
results_summaries    := "data/results/summaries"
intermediate_dir     := "data/intermediate"

# ─── Model & codec lists ─────────────────────────────────────
# DL models for target2 extraction
dl_models_target2    := "morphem dinov2 dinov2_random openphenom subcell__clip01"

# DL codecs (10): zstd + 9 JPEG XL variants — from cl_3 (median aggregation)
dl_codecs_target2    := "zstd jpegxl_lossy_hq jpegxl_lossy_mq jpegxl_lossy_mq_new jpegxl_lossy_lq jpegxl_lossy_d2_e8 jpegxl_lossy_d10 jpegxl_lossy_d15 jpegxl_lossy_d30 jpegxl_lossy_effort_3"

# CP codecs (7): zstd + 6 JPEG XL variants
cp_codecs_target2    := "zstd jpegxl_lossy_hq jpegxl_lossy_mq jpegxl_lossy_lq jpegxl_lossy_d2_e8 jpegxl_lossy_d10 jpegxl_lossy_effort_3"

# Lite DL codecs
lite_codecs_dl       := "jpegxl_lossy_mq jpegxl_lossy_d20 jpegxl_lossy_hq"

# All compression codecs (superset used for image compression)
all_codecs           := "zstd jpegxl_lossy_hq jpegxl_lossy_mq jpegxl_lossy_mq_new jpegxl_lossy_lq jpegxl_lossy_d2_e8 jpegxl_lossy_d10 jpegxl_lossy_d15 jpegxl_lossy_d30 jpegxl_lossy_effort_3"

# ─── Threading ────────────────────────────────────────────────
omp_threads          := "12"

# ═══════════════════════════════════════════════════════════════
# Section 2: Setup & Environment
# ═══════════════════════════════════════════════════════════════

# Verify the development environment
check-env:
    python --version
    python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
    pixi --version

# Verify key data paths exist before running the pipeline
check-data:
    #!/usr/bin/env bash
    set -euo pipefail
    ok=true
    for p in "{{ raw_images_target2 }}" "{{ compressed_target2 }}" "{{ raw_images_lite }}" "{{ compressed_lite }}" "{{ aliby_output }}" \
             "{{ annotations_db }}" "{{ cp_profiles }}"; do
        if [ -e "$p" ]; then
            echo "  OK  $p"
        else
            echo "  MISSING  $p"
            ok=false
        fi
    done
    if [ "$ok" = false ]; then
        echo ""
        echo "Some paths are missing — check DATA_ROOT (currently {{ data_root }})"
        exit 1
    fi
    echo ""
    echo "All data paths verified."

# ═══════════════════════════════════════════════════════════════
# Section 3: Metadata
# ═══════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════
# Section 4: Image Compression
# ═══════════════════════════════════════════════════════════════

# Compress target2 images with a single codec
compress-target2 codec="jpegxl_lossy_mq" jobs="16":
    uv run python src/compress_tif.py \
        --input {{ raw_images_target2 }} \
        --output {{ compressed_target2 }} \
        --codec {{ codec }} \
        --n-jobs {{ jobs }}

# Compress jump_lite images with a single codec
compress-lite codec="jpegxl_lossy_mq" jobs="16":
    uv run python src/compress_tif.py \
        --input {{ raw_images_lite }} \
        --output {{ compressed_lite }} \
        --codec {{ codec }} \
        --n-jobs {{ jobs }}

# Compress target2 with all codecs
compress-target2-all jobs="16":
    #!/usr/bin/env bash
    set -euo pipefail
    for codec in {{ all_codecs }}; do
        echo "=== Compressing target2 with ${codec} === $(date)"
        uv run python src/compress_tif.py \
            --input {{ raw_images_target2 }} \
            --output {{ compressed_target2 }} \
            --codec "${codec}" \
            --n-jobs {{ jobs }}
    done
    echo "=== All target2 codecs done === $(date)"

# Compress jump_lite with all codecs
compress-lite-all jobs="16":
    #!/usr/bin/env bash
    set -euo pipefail
    for codec in {{ all_codecs }}; do
        echo "=== Compressing jump_lite with ${codec} === $(date)"
        uv run python src/compress_tif.py \
            --input {{ raw_images_lite }} \
            --output {{ compressed_lite }} \
            --codec "${codec}" \
            --n-jobs {{ jobs }}
    done
    echo "=== All jump_lite codecs done === $(date)"

# ═══════════════════════════════════════════════════════════════
# Section 5: Image Quality
# ═══════════════════════════════════════════════════════════════

# Compute PSNR/SSIM quality metrics (skip LPIPS for speed).
# Intermediate quality_metrics.csv + PNGs → data/intermediate/image_quality/
quality-metrics n_samples="100" exclude="jpegxl_lossy_mq_new jpegxl_lossy_d50":
    uv run python analysis/image_quality/compare_codecs.py \
        --data-dir {{ compressed_target2 }} \
        --n-samples {{ n_samples }} \
        --skip-lpips \
        --exclude-codecs {{ exclude }} \
        --output-dir {{ intermediate_dir }}/image_quality

# Compute sharpness-only metrics (Laplacian variance + Tenengrad, no GPU needed)
quality-sharpness n_samples="100" exclude="jpegxl_lossy_mq_new jpegxl_lossy_d50":
    uv run python analysis/image_quality/compare_codecs.py \
        --data-dir {{ compressed_target2 }} \
        --n-samples {{ n_samples }} \
        --sharpness-only \
        --exclude-codecs {{ exclude }} \
        --output-dir {{ intermediate_dir }}/image_quality

# Regenerate quality violin plots from existing quality_metrics.csv.
# Final PNGs (incl. supplementary ssim_violin.png) copied to data/results/figures/image_quality/.
quality-figures:
    uv run python analysis/image_quality/compare_codecs.py \
        --figures-only \
        --output-dir {{ intermediate_dir }}/image_quality
    mkdir -p {{ results_figures }}/image_quality
    cp {{ intermediate_dir }}/image_quality/*.png {{ results_figures }}/image_quality/

# ═══════════════════════════════════════════════════════════════
# Section 6: Segmentation Comparison
# ═══════════════════════════════════════════════════════════════

# Compare segmentation masks across codecs — cell, then nuclei, then combined.
# Writes intermediate CSVs + instance_mappings/*.parquet + detailed_results/*.csv
# under data/intermediate/segmentation_comparison/. These files are required
# inputs to segmentation-cell-iou and segmentation-iou-ablation.
segmentation-compare methods="jpegxl_lossy_hq.zarr jpegxl_lossy_effort_3.zarr jpegxl_lossy_d2_e8.zarr jpegxl_lossy_mq.zarr jpegxl_lossy_lq.zarr jpegxl_lossy_d10.zarr":
    uv run python analysis/segmentation/compare_segmentations.py \
        --root {{ aliby_output }}/cp_measure/jump_target2_4plate \
        --ground-truth zstd.zarr \
        --methods {{ methods }} \
        --output-dir {{ intermediate_dir }} \
        --force-rerun --segment-step segment_cell --fast --save-mappings
    uv run python analysis/segmentation/compare_segmentations.py \
        --root {{ aliby_output }}/cp_measure/jump_target2_4plate \
        --ground-truth zstd.zarr \
        --methods {{ methods }} \
        --output-dir {{ intermediate_dir }} \
        --force-rerun --segment-step segment_nuclei --fast --save-mappings
    uv run python analysis/segmentation/compare_segmentations.py \
        --root {{ aliby_output }}/cp_measure/jump_target2_4plate \
        --ground-truth zstd.zarr \
        --methods {{ methods }} \
        --output-dir {{ intermediate_dir }} \
        --both
    mkdir -p {{ results_figures }}/segmentation_comparison
    cp {{ intermediate_dir }}/segmentation_comparison/*.png {{ results_figures }}/segmentation_comparison/

# Generate per-cell IoU distribution plots
segmentation-cell-iou mappings_dir=(intermediate_dir + "/segmentation_comparison/instance_mappings"):
    uv run python analysis/segmentation/plot_cell_level_iou.py \
        --mappings-dir {{ mappings_dir }} \
        --output-dir {{ results_figures }}/segmentation_comparison

# IoU ablation: show metric consistency across IoU thresholds (appendix figure)
segmentation-iou-ablation:
    uv run python analysis/segmentation/plot_iou_ablation.py \
        --results-dir {{ intermediate_dir }}/segmentation_comparison/detailed_results \
        --output-dir {{ results_figures }}

# Rank stability: Spearman rho of model rankings across compression levels
rank-stability:
    uv run python analysis/rank_stability.py \
        --input {{ intermediate_dir }}/sweep_summaries/sweep_results_v11_lite_full.csv \
        --output-dir {{ results_figures }}/rank_stability

# Saturation analysis: proper (normalize after subsampling, zero leakage)
saturation-proper:
    cd src/norm_3 && pixi run python ../../analysis/saturation_analysis_proper.py
    mkdir -p {{ results_figures }}/saturation_proper
    cp analysis/output/saturation_proper/saturation_proper_PA_mean_nap.png {{ results_figures }}/saturation_proper/

# Saturation analysis: proper - pilot (2 models, 3 configs, 3 seeds)
saturation-proper-pilot:
    cd src/norm_3 && pixi run python ../../analysis/saturation_analysis_proper.py \
        --models morphem cellprofiler --n-configs 3 --n-seeds 3
    mkdir -p {{ results_figures }}/saturation_proper
    cp analysis/output/saturation_proper/saturation_proper_PA_mean_nap.png {{ results_figures }}/saturation_proper/saturation_proper_pilot.png

# Saturation analysis: per-group best-config replot from existing saturation-proper outputs
# Reads analysis/output/saturation_proper/saturation_results_<group>.csv files
# and emits per-group curves keeping only the best config per (model, n, seed),
# with shaded variance across seeds.
saturation-plot-bestconfig:
    cd src/norm_3 && pixi run python ../../analysis/plot_saturation_bestconfig.py \
        --input-dir ../../{{ intermediate_dir }}/analysis/saturation_proper \
        --output-dir ../../{{ results_figures }}/saturation_proper

# ═══════════════════════════════════════════════════════════════
# Section 7: Feature Extraction
# ═══════════════════════════════════════════════════════════════

# Extract DL features for target2 (5 models × all codecs, parallel)
extract-dl-target2 jobs="4":
    #!/usr/bin/env bash
    set -euo pipefail
    for model in {{ dl_models_target2 }}; do
        echo "=== Extracting $model for target2 === $(date)"
        uv run python src/extract_features.py \
            --input {{ target2_aliby_dl }} \
            --output {{ features_target2_dl }} \
            --model "$model" \
            --dataset jump_target2_4plate \
            --n-jobs {{ jobs }}
    done
    echo ""
    echo "Extraction done! $(date)"
    echo "Parquet count: $(ls {{ features_target2_dl }}/*.parquet 2>/dev/null | wc -l)"

# Extract CellProfiler features for target2 (all codecs)
extract-cp-target2 jobs="-1":
    uv run python src/extract_features.py \
        --input {{ aliby_output }} \
        --output {{ features_target2_cp }} \
        --model cp_measure \
        --dataset jump_target2_4plate \
        --n-jobs {{ jobs }}

# Extract DL features for jump_lite (5 models × lite codecs)
extract-dl-lite jobs="1":
    #!/usr/bin/env bash
    set -euo pipefail
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
    export OPENBLAS_NUM_THREADS=8
    for model in {{ dl_models_target2 }}; do
        echo "=== Extracting $model for jump_lite === $(date)"
        uv run python src/extract_features.py \
            --input {{ lite_aliby_dl }} \
            --output {{ features_lite_dl }} \
            --model "$model" \
            --dataset jump_lite_updated \
            --n-jobs {{ jobs }}
    done
    echo ""
    echo "Extraction done! $(date)"
    echo "Parquet count: $(ls {{ features_lite_dl }}/*.parquet 2>/dev/null | wc -l)"

# Reformat raw CellProfiler profiles for jump_lite
extract-cp-lite:
    uv run python src/reformat_raw_cp_profiles.py \
        --source {{ cp_profiles }} \
        --metadata metadata/metadata_dataset_filtered_4reps.parquet \
        --output {{ features_lite_cp }}/cellprofiler_raw_jump_lite_raw_features.parquet

# Create cell-count baseline features from CP parquets
extract-cell-count:
    uv run python scripts/create_cell_count_features.py

# Extract single model/codec (manual)
extract model dataset="jump_target2_4plate" codec="" output="data/features/" jobs="-1":
    uv run python src/extract_features.py \
        --input {{ aliby_output }} \
        --output {{ output }} \
        --model {{ model }} \
        --dataset {{ dataset }} \
        --n-jobs {{ jobs }}

# ═══════════════════════════════════════════════════════════════
# Section 8: Feature Analysis
# ═══════════════════════════════════════════════════════════════

# CellProfiler correlation heatmaps
# Intermediate parquets → data/intermediate/feature_correlation/
# Final PNGs copied to data/results/figures/feature_correlation/
# Greenlist variant requires `just feature-cross-well` to have run first
# (it produces greenlist_features_{cell,nuclei}.csv that this script consumes).
feature-correlation-cp:
    uv run python analysis/feature_similarity/feature_correlation_cp_measure_script.py \
        --workspace-dir {{ aliby_output }}/cp_measure/jump_target2_4plate \
        --output-dir {{ intermediate_dir }}/feature_correlation \
        --raw-features-path {{ features_target2_cp }}/cp_measure_jump_target2_4plate_zstd_raw_features.parquet \
        --greenlist-dir {{ intermediate_dir }}/feature_correlation
    mkdir -p {{ results_figures }}/feature_correlation
    cp {{ intermediate_dir }}/feature_correlation/*.png {{ results_figures }}/feature_correlation/

# Correlation vs raw CellProfiler
# feature-correlation-raw:
#    uv run python analysis/feature_similarity/correlate_vs_raw_cp.py

# Cross-well feature consistency for same-treatment replicates
# Intermediate CSVs → data/intermediate/feature_correlation/
# Final PNGs (incl. supplementary replicate_vs_codec_correlation.png) copied to
# data/results/figures/feature_correlation/.
feature-cross-well metadata="metadata/metadata.parquet" codec_corr_csv="analysis/output/codec_feature_correlation.csv":
    uv run python analysis/feature_similarity/compare_cross_well_features.py \
        --features-base {{ aliby_output }}/cp_measure/jump_target2_4plate \
        --metadata {{ metadata }} \
        --output-dir {{ intermediate_dir }}/feature_correlation \
        --codec-correlation-csv {{ codec_corr_csv }}
    mkdir -p {{ results_figures }}/feature_correlation/cell {{ results_figures }}/feature_correlation/nuclei
    cp {{ intermediate_dir }}/feature_correlation/*.png       {{ results_figures }}/feature_correlation/        2>/dev/null || true
    cp {{ intermediate_dir }}/feature_correlation/cell/*.png  {{ results_figures }}/feature_correlation/cell/   2>/dev/null || true
    cp {{ intermediate_dir }}/feature_correlation/nuclei/*.png {{ results_figures }}/feature_correlation/nuclei/ 2>/dev/null || true

# ═══════════════════════════════════════════════════════════════
# Section 9: Normalization Sweeps
# ═══════════════════════════════════════════════════════════════

# Full v11 target2 sweep: 5 DL models + CP + cell_count across 3 GPUs
sweep-v11:
    #!/usr/bin/env bash
    set -euo pipefail

    LOG_DIR="logs/sweep_v11"
    mkdir -p "$LOG_DIR"

    echo "==================================================="
    echo "Focused Sweep v11 — target2 (5 DL + CP + cell_count, 3 GPUs)"
    echo "  DL features: median aggregation (cl_3)"
    echo "  tvn_efaar only"
    echo "  DL: 48 configs/codec, CP: 48 configs/codec, cell_count: 4 configs/codec"
    echo "==================================================="
    echo ""

    export OMP_NUM_THREADS={{ omp_threads }}
    export MKL_NUM_THREADS={{ omp_threads }}
    export OPENBLAS_NUM_THREADS={{ omp_threads }}

    DL_CODECS="{{ dl_codecs_target2 }}"
    CP_CODECS="{{ cp_codecs_target2 }}"

    DL_CONFIGS="focused_dl_v11_tvn_efaar"
    CP_CONFIGS="focused_cp_v11_tvn_efaar"
    CC_CONFIG="focused_cell_count_v11"

    DL_FEATURE_DIR="../../{{ features_target2_dl }}"
    CP_FEATURE_DIR="../../{{ features_target2_cp }}"

    # Helper: run all codecs × configs for a DL model on a given GPU
    run_dl_model() {
        local model="$1"
        local gpu="$2"
        echo "=== ${model} on GPU ${gpu} === $(date)"
        for codec in $DL_CODECS; do
            feature_file="${DL_FEATURE_DIR}/${model}_jump_target2_4plate_${codec}_raw_features.parquet"
            echo "--- ${model} codec: ${codec} --- $(date)"
            if [ ! -f "${feature_file}" ]; then
                echo "  SKIPPING: ${feature_file} not found"
                continue
            fi
            for cfg in $DL_CONFIGS; do
                echo "  config: ${cfg} $(date)"
                CUDA_VISIBLE_DEVICES="${gpu}" pixi run python -m norm_3.pipeline --multirun \
                    +sweep="${cfg}" \
                    input.path="${feature_file}" \
                    hydra/launcher=joblib hydra.launcher.n_jobs=4 || {
                    echo "  Warning: ${cfg} encountered errors"
                }
            done
        done
        echo "=== ${model} DONE === $(date)"
        echo ""
    }

    echo "  GPU 0: cp_measure (7 codecs x 48 configs) + cell_count (7 codecs x 4 configs)"
    echo "  GPU 1: morphem + subcell__clip01 (10 codecs x 48 configs each)"
    echo "  GPU 2: openphenom + dinov2 + dinov2_random (10 codecs x 48 configs each)"
    echo ""
    echo "Logs: ${LOG_DIR}/"
    echo ""

    # GPU 0: CellProfiler + cell_count (7 codecs each)
    (
        cd {{ norm3_dir }}
        echo "=== cp_measure on GPU 0 === $(date)"
        for codec in $CP_CODECS; do
            feature_file="${CP_FEATURE_DIR}/cp_measure_jump_target2_4plate_${codec}_raw_features.parquet"
            echo "--- cp_measure codec: ${codec} --- $(date)"
            if [ ! -f "${feature_file}" ]; then
                echo "  SKIPPING: ${feature_file} not found"
                continue
            fi
            for cfg in $CP_CONFIGS; do
                echo "  config: ${cfg} $(date)"
                CUDA_VISIBLE_DEVICES=0 pixi run python -m norm_3.pipeline --multirun \
                    +sweep="${cfg}" \
                    input.path="${feature_file}" \
                    hydra/launcher=joblib hydra.launcher.n_jobs=4 || {
                    echo "  Warning: ${cfg} encountered errors"
                }
            done
        done
        echo "=== cp_measure DONE === $(date)"

        echo "=== cell_count baseline on GPU 0 === $(date)"
        for codec in $CP_CODECS; do
            feature_file="${CP_FEATURE_DIR}/cell_count_jump_target2_4plate_${codec}_raw_features.parquet"
            echo "--- cell_count codec: ${codec} --- $(date)"
            if [ ! -f "${feature_file}" ]; then
                echo "  SKIPPING: ${feature_file} not found"
                continue
            fi
            echo "  config: ${CC_CONFIG} $(date)"
            CUDA_VISIBLE_DEVICES=0 pixi run python -m norm_3.pipeline --multirun \
                +sweep="${CC_CONFIG}" \
                input.path="${feature_file}" \
                hydra/launcher=joblib hydra.launcher.n_jobs=4 || {
                echo "  Warning: ${CC_CONFIG} encountered errors"
            }
        done
        echo "=== cell_count baseline DONE === $(date)"
    ) > "${LOG_DIR}/gpu0_cp_cellcount.log" 2>&1 &
    echo "  Started cp_measure + cell_count on GPU 0 (PID $!)"

    # GPU 1: morphem + subcell__clip01
    (
        cd {{ norm3_dir }}
        run_dl_model morphem 1
        run_dl_model subcell__clip01 1
    ) > "${LOG_DIR}/gpu1_morphem_subcell.log" 2>&1 &
    echo "  Started morphem + subcell__clip01 on GPU 1 (PID $!)"

    # GPU 2: openphenom + dinov2 + dinov2_random
    (
        cd {{ norm3_dir }}
        run_dl_model openphenom 2
        run_dl_model dinov2 2
        run_dl_model dinov2_random 2
    ) > "${LOG_DIR}/gpu2_openphenom_dinov2.log" 2>&1 &
    echo "  Started openphenom + dinov2 + dinov2_random on GPU 2 (PID $!)"

    echo ""
    echo "All models launched. Waiting for completion..."
    echo "  Monitor with: tail -f ${LOG_DIR}/*.log"
    echo ""

    wait

    echo ""
    echo "==================================================="
    echo "All Focused v11 Sweeps Complete! $(date)"
    echo "==================================================="

    echo ""
    echo "Final counts:"
    for d in {{ norm3_dir }}/{{ sweep_v11_dir }}/*/; do
        name=$(basename "$d")
        count=$(find "$d" -name metrics.json 2>/dev/null | wc -l)
        echo "  ${name}: ${count} configs"
    done

# Full v11_lite jump_lite sweep: 5 DL + CP + cell_count across 3 GPUs
sweep-v11-lite:
    #!/usr/bin/env bash
    set -euo pipefail

    LOG_DIR="logs/sweep_v11_lite"
    mkdir -p "$LOG_DIR"

    echo "==================================================="
    echo "Focused Sweep v11 lite — jump_lite (5 DL + CP + cell_count, 3 GPUs)"
    echo "  DL features: median aggregation (cl_3)"
    echo "  tvn_efaar only"
    echo "  DL: 48 configs/codec, CP: 48 configs/codec, cell_count: 4 configs"
    echo "==================================================="
    echo ""

    export OMP_NUM_THREADS={{ omp_threads }}
    export MKL_NUM_THREADS={{ omp_threads }}
    export OPENBLAS_NUM_THREADS={{ omp_threads }}

    NJOBS_CP=8
    NJOBS_DL=16
    NJOBS_CELL_COUNT=4

    DL_CONFIGS="focused_dl_v11_lite_tvn_efaar"
    CP_CONFIGS="focused_cp_v11_lite_tvn_efaar"

    DL_FEATURE_DIR="../../{{ features_lite_dl }}"
    CP_FEATURE_DIR="../../{{ features_lite_cp }}"

    LITE_CODECS="{{ lite_codecs_dl }}"

    echo "  GPU 0: cellprofiler (1 dataset x 48 configs) + cell_count (4 configs) + subcell mq (48 configs)"
    echo "  GPU 1: morphem (3 codecs x 48) + subcell_clip01 (3 codecs x 48)"
    echo "  GPU 2: openphenom (3 codecs) + dinov2 (3 codecs) + dinov2_random (mq)"
    echo ""
    echo "Logs: ${LOG_DIR}/"
    echo ""

    # GPU 0: CellProfiler + cell_count + subcell
    (
        cd {{ norm3_dir }}
        echo "=== cellprofiler on GPU 0 === $(date)"
        for cfg in $CP_CONFIGS; do
            echo "--- ${cfg} --- $(date)"
            CUDA_VISIBLE_DEVICES=0 pixi run python -m norm_3.pipeline --multirun \
                +sweep="${cfg}" \
                input.path="${CP_FEATURE_DIR}/cellprofiler_raw_jump_lite_raw_features.parquet" \
                hydra/launcher=joblib hydra.launcher.n_jobs=${NJOBS_CP} || {
                echo "  Warning: ${cfg} encountered errors"
            }
        done
        echo "=== cellprofiler DONE === $(date)"

        echo "=== cell_count baseline on GPU 0 === $(date)"
        CUDA_VISIBLE_DEVICES=0 pixi run python -m norm_3.pipeline --multirun \
            +sweep=focused_cell_count_v11_lite \
            input.path="${CP_FEATURE_DIR}/cell_count_jump_lite_raw_features.parquet" \
            hydra/launcher=joblib hydra.launcher.n_jobs=${NJOBS_CELL_COUNT} || {
            echo "  Warning: cell_count baseline encountered errors"
        }
        echo "=== cell_count baseline DONE === $(date)"

        echo "=== subcell on GPU 0 === $(date)"
        feature_file="${DL_FEATURE_DIR}/subcell_jump_lite_updated_jpegxl_lossy_mq_raw_features.parquet"
        if [ -f "${feature_file}" ]; then
            for cfg in $DL_CONFIGS; do
                echo "--- ${cfg} --- $(date)"
                CUDA_VISIBLE_DEVICES=0 pixi run python -m norm_3.pipeline --multirun \
                    +sweep="${cfg}" \
                    input.path="${feature_file}" \
                    hydra/launcher=joblib hydra.launcher.n_jobs=${NJOBS_DL} || {
                    echo "  Warning: ${cfg} encountered errors"
                }
            done
        else
            echo "  SKIPPING: ${feature_file} not found"
        fi
        echo "=== subcell DONE === $(date)"
    ) > "${LOG_DIR}/gpu0_cp_subcell.log" 2>&1 &
    echo "  Started cellprofiler + subcell on GPU 0 (PID $!)"

    # GPU 1: morphem + subcell_clip01
    (
        cd {{ norm3_dir }}
        for codec in $LITE_CODECS; do
            feature_file="${DL_FEATURE_DIR}/morphem_jump_lite_updated_${codec}_raw_features.parquet"
            echo "--- morphem codec: ${codec} --- $(date)"
            if [ ! -f "${feature_file}" ]; then
                echo "  SKIPPING: ${feature_file} not found"
                continue
            fi
            for cfg in $DL_CONFIGS; do
                echo "  config: ${cfg} $(date)"
                CUDA_VISIBLE_DEVICES=1 pixi run python -m norm_3.pipeline --multirun \
                    +sweep="${cfg}" \
                    input.path="${feature_file}" \
                    hydra/launcher=joblib hydra.launcher.n_jobs=${NJOBS_DL} || {
                    echo "  Warning: ${cfg} encountered errors"
                }
            done
        done
        echo "=== morphem DONE === $(date)"

        for codec in $LITE_CODECS; do
            feature_file="${DL_FEATURE_DIR}/subcell__clip01_jump_lite_updated_${codec}_raw_features.parquet"
            echo "--- subcell_clip01 codec: ${codec} --- $(date)"
            if [ ! -f "${feature_file}" ]; then
                echo "  SKIPPING: ${feature_file} not found"
                continue
            fi
            for cfg in $DL_CONFIGS; do
                echo "  config: ${cfg} $(date)"
                CUDA_VISIBLE_DEVICES=1 pixi run python -m norm_3.pipeline --multirun \
                    +sweep="${cfg}" \
                    input.path="${feature_file}" \
                    hydra/launcher=joblib hydra.launcher.n_jobs=${NJOBS_DL} || {
                    echo "  Warning: ${cfg} encountered errors"
                }
            done
        done
        echo "=== subcell_clip01 DONE === $(date)"
    ) > "${LOG_DIR}/gpu1_morphem_subcell_clip01.log" 2>&1 &
    echo "  Started morphem + subcell_clip01 on GPU 1 (PID $!)"

    # GPU 2: openphenom + dinov2 + dinov2_random
    (
        cd {{ norm3_dir }}
        for codec in $LITE_CODECS; do
            feature_file="${DL_FEATURE_DIR}/openphenom_jump_lite_updated_${codec}_raw_features.parquet"
            echo "--- openphenom codec: ${codec} --- $(date)"
            if [ ! -f "${feature_file}" ]; then
                echo "  SKIPPING: ${feature_file} not found"
                continue
            fi
            for cfg in $DL_CONFIGS; do
                echo "  config: ${cfg} $(date)"
                CUDA_VISIBLE_DEVICES=2 pixi run python -m norm_3.pipeline --multirun \
                    +sweep="${cfg}" \
                    input.path="${feature_file}" \
                    hydra/launcher=joblib hydra.launcher.n_jobs=${NJOBS_DL} || {
                    echo "  Warning: ${cfg} encountered errors"
                }
            done
        done
        echo "=== openphenom DONE === $(date)"

        for codec in $LITE_CODECS; do
            feature_file="${DL_FEATURE_DIR}/dinov2_jump_lite_updated_${codec}_raw_features.parquet"
            echo "--- dinov2 codec: ${codec} --- $(date)"
            if [ ! -f "${feature_file}" ]; then
                echo "  SKIPPING: ${feature_file} not found"
                continue
            fi
            for cfg in $DL_CONFIGS; do
                echo "  config: ${cfg} $(date)"
                CUDA_VISIBLE_DEVICES=2 pixi run python -m norm_3.pipeline --multirun \
                    +sweep="${cfg}" \
                    input.path="${feature_file}" \
                    hydra/launcher=joblib hydra.launcher.n_jobs=${NJOBS_DL} || {
                    echo "  Warning: ${cfg} encountered errors"
                }
            done
        done
        echo "=== dinov2 DONE === $(date)"

        feature_file="${DL_FEATURE_DIR}/dinov2_random_jump_lite_updated_jpegxl_lossy_mq_raw_features.parquet"
        echo "--- dinov2_random codec: jpegxl_lossy_mq --- $(date)"
        if [ -f "${feature_file}" ]; then
            for cfg in $DL_CONFIGS; do
                echo "  config: ${cfg} $(date)"
                CUDA_VISIBLE_DEVICES=2 pixi run python -m norm_3.pipeline --multirun \
                    +sweep="${cfg}" \
                    input.path="${feature_file}" \
                    hydra/launcher=joblib hydra.launcher.n_jobs=${NJOBS_DL} || {
                    echo "  Warning: ${cfg} encountered errors"
                }
            done
        else
            echo "  SKIPPING: ${feature_file} not found"
        fi
        echo "=== dinov2_random DONE === $(date)"
    ) > "${LOG_DIR}/gpu2_openphenom_dinov2.log" 2>&1 &
    echo "  Started openphenom + dinov2 + dinov2_random on GPU 2 (PID $!)"

    echo ""
    echo "All models launched. Waiting for completion..."
    echo "  Monitor with: tail -f ${LOG_DIR}/*.log"
    echo ""

    wait

    echo ""
    echo "==================================================="
    echo "All Focused v11 Lite Sweeps Complete! $(date)"
    echo "==================================================="

    echo ""
    echo "Final counts:"
    for d in {{ norm3_dir }}/{{ sweep_v11_lite_dir }}/*/; do
        name=$(basename "$d")
        count=$(find "$d" -name metrics.json 2>/dev/null | wc -l)
        echo "  ${name}: ${count} configs"
    done

# ═══════════════════════════════════════════════════════════════
# Section 10: Results Aggregation & Figures
# ═══════════════════════════════════════════════════════════════

# Reproduce all final figures + tables from intermediate checkpoints under
# data/intermediate/. Assumes the sweeps (data/intermediate/sweep_v11_lite/,
# data/features/variance_first_v11/) and the MOTIVE eval
# (data/intermediate/motive_eval/large_strict/) are already populated —
# those upstream stages are NOT re-run here.
reproduce: results-v11-lite results-v11 results-v11-lite-best-avg results-v11-best-avg
    just motive-plot       data/intermediate/motive_eval/large_strict {{ results_figures }}/motive_large_strict
    just motive-plot-delta {{ results_figures }}/motive_large_strict
    just motive-table-delta {{ results_figures }}/motive_large_strict
    just motive-plot-cross {{ results_figures }}/motive_large_strict
    just model-task-rank   {{ results_figures }}/motive_large_strict
    just combined-codec-delta-table
    just rank-stability
    just segmentation-cell-iou
    just segmentation-iou-ablation
    just feature-cross-well
    just feature-correlation-cp
    just quality-figures
    just saturation-plot-bestconfig
    @echo
    @echo "DONE. Final outputs under {{ results_figures }}/ and {{ results_tables }}/"

# End-to-end from compressed images → final paper figures.
# PREREQUISITES (run `just <recipe>` from the bootstrap section below if missing):
#   1. Raw JUMP images at {{ raw_images_lite }} (run `just build-jl-index download-raw`)
#   2. Aliby segmentation/cp_measure output under {{ aliby_output }}
#      (run `just aliby-featurize` — requires external aliby + Nahual GPU servers;
#       see prep/README.md)
#   3. CellProfiler profiles.parquet at {{ cp_profiles }}
#      (run `just fetch-cp-profiles` — ~13.5 GB anonymous S3 download)
#   4. Annotation DBs + curated metadata/ (run `just prep-annotations` — all
#      inputs are committed or auto-fetched; no manual downloads required)
# Expect MANY hours of wall time and hundreds of GB of disk for the heavy steps.
produce-paper:
    # Stage 1: Compression
    just compress-lite-all
    just compress-target2-all
    # Stage 2: Feature extraction (requires aliby_output for CP)
    just extract-cell-count
    just extract-cp-lite
    just extract-cp-target2
    just extract-dl-lite
    just extract-dl-target2
    # Stage 3: Normalization sweep (multi-GPU, long-running)
    just sweep-v11-lite
    just sweep-v11
    # Stage 4: Motive curate + eval
    just motive-curate-strict
    just motive-run-top
    # Stage 5: Segmentation comparison (produces instance_mappings + detailed_results
    # under data/intermediate/segmentation_comparison/, required by reproduce's
    # segmentation-cell-iou and segmentation-iou-ablation)
    just segmentation-compare
    # Stage 6: Aggregate, plot, and produce all paper figures
    just reproduce

# Remove everything under data/results/ so `just reproduce` regenerates cleanly.
clean-results:
    rm -rf {{ results_figures }} {{ results_tables }} {{ results_summaries }}
    mkdir -p {{ results_figures }} {{ results_tables }} {{ results_summaries }}

# Aggregate target2 v11 sweep results with plots
results-v11:
    cd {{ norm3_dir }} && pixi run python gather_sweep_results.py \
        --sweep-dir ../../{{ sweep_v11_dir }} \
        --plot-dir ../../{{ results_figures }}/sweep_v11 \
        --plot --filter-degenerate \
        --best-metric nap_balanced \
        --exclude-codecs mq_new d20_e2 d50

# Aggregate jump_lite v11_lite sweep results with plots
results-v11-lite:
    cd {{ norm3_dir }} && pixi run python gather_sweep_results.py \
        --sweep-dir ../../{{ sweep_v11_lite_dir }} \
        --plot-dir ../../{{ results_figures }}/sweep_v11_lite \
        --plot --filter-degenerate \
        --best-metric nap_balanced

# Same as results-v11 but with --best-selection best_avg_codec
# (produces the `*_best_avg_codec_*` suffix variants used in fig5/6/supplementary).
results-v11-best-avg:
    cd {{ norm3_dir }} && pixi run python gather_sweep_results.py \
        --sweep-dir ../../{{ sweep_v11_dir }} \
        --plot-dir ../../{{ results_figures }}/sweep_v11 \
        --plot --filter-degenerate \
        --best-metric nap_balanced \
        --exclude-codecs mq_new d20_e2 d50 \
        --best-selection best_avg_codec

# Same as results-v11-lite but with --best-selection best_avg_codec.
results-v11-lite-best-avg:
    cd {{ norm3_dir }} && pixi run python gather_sweep_results.py \
        --sweep-dir ../../{{ sweep_v11_lite_dir }} \
        --plot-dir ../../{{ results_figures }}/sweep_v11_lite \
        --plot --filter-degenerate \
        --best-metric nap_balanced \
        --best-selection best_avg_codec

# Generic sweep results aggregation
results sweep_dir metric="nap_balanced":
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

# One-shot: curate MOTIVE annotations + map published splits to JCP2022.
# Defaults to the committed metadata/motive_eval_compounds.parquet (the canonicalised
# JCP2022 allowlist; all 26,450 rows labelled "test"). Override to point at a
# different upstream splits file if regenerating from scratch.
motive-curate motive_splits_path="metadata/motive_eval_compounds.parquet":
    uv run python scripts/curate_motive.py \
        --metadata metadata/metadata_dataset_filtered_4reps.parquet \
        --annotations-cc {{ annotations_cc }} \
        --inchikey-map metadata/inchikey_to_jcp2022_mapping_compound_compound.csv \
        --motive-splits-path {{ motive_splits_path }} \
        --output-dir metadata/

# Run MOTIVE evaluation on a single output.parquet (mirrors the PA invocation)
motive-eval input output:
    uv run python src/motive/evaluate_motive.py \
        --input {{ input }} \
        --output {{ output }} \
        --annotations metadata/motive_annotations.parquet \
        --splits metadata/motive_eval_compounds.parquet

# Curate the STRICT variant of MOTIVE annotations (rel_type-filtered).
# Writes metadata/motive_annotations_strict.parquet alongside the existing
# full file. Splits are mode-agnostic — reuse motive_eval_compounds.parquet from the
# full curate run.
motive-curate-strict:
    uv run python scripts/curate_motive.py \
        --mode strict \
        --metadata metadata/metadata_dataset_filtered_4reps.parquet \
        --inchikey-map metadata/inchikey_to_jcp2022_mapping_compound_compound.csv \
        --output-dir metadata/ \
        --skip-splits

# Curate the ULTRA-STRICT variant: completed direct-binding allowlist on CG,
# action-class-aware CC bridge, tightened GG (PPI/binding/PTM only).
# Writes metadata/motive_annotations_ultra_strict.parquet alongside the
# existing files. Splits are mode-agnostic — reuse motive_eval_compounds.parquet.
motive-curate-ultra-strict:
    uv run python scripts/curate_motive.py \
        --mode ultra_strict \
        --metadata metadata/metadata_dataset_filtered_4reps.parquet \
        --inchikey-map metadata/inchikey_to_jcp2022_mapping_compound_compound.csv \
        --output-dir metadata/ \
        --skip-splits

# Filter a sweep_results.csv to the top-N configs per (family, codec) by metric.
# Writes a text file of absolute output.parquet paths usable by motive-eval-list.
motive-filter-top sweep_results sweep_dir top_n="50" metric="PA_mean_nap" out_list="metadata/motive_top_configs.txt":
    uv run python analysis/filter_top_configs.py \
        --sweep-results {{ sweep_results }} \
        --sweep-dir {{ sweep_dir }} \
        --top-n {{ top_n }} \
        --metric {{ metric }} \
        --out {{ out_list }} \
        --force

# Run MOTIVE eval on a precomputed list of output.parquet paths (one per line).
# Mirrors the sweep-dir → output-dir subtree exactly like motive-eval-sweep.
motive-eval-list sweep_dir output_dir list_file jobs="4" annotations="metadata/motive_annotations.parquet" splits="metadata/motive_eval_compounds.parquet":
    #!/usr/bin/env bash
    set -euo pipefail
    SWEEP="{{ sweep_dir }}"
    OUT_BASE="{{ output_dir }}"
    LIST="{{ list_file }}"
    ANN="{{ annotations }}"
    SPLITS="{{ splits }}"
    export SWEEP OUT_BASE ANN SPLITS
    [ -f "$ANN" ]    || { echo "ERROR: annotations file not found: $ANN" >&2; exit 1; }
    [ -f "$SPLITS" ] || { echo "ERROR: splits file not found: $SPLITS" >&2; exit 1; }
    [ -f "$LIST" ]   || { echo "ERROR: list file not found: $LIST" >&2; exit 1; }
    n=$(wc -l < "$LIST")
    echo "[motive-eval-list] processing $n parquets from $LIST"
    cat "$LIST" \
      | xargs -n1 -P{{ jobs }} -I{} bash -c '
          in="$1"
          [ -f "$in" ] || { echo "WARN missing input: $in" >&2; exit 0; }
          in_dir="$(dirname "$in")"
          rel="${in_dir#${SWEEP%/}/}"
          out="$OUT_BASE/$rel/motive"
          mkdir -p "$out"
          uv run python src/motive/evaluate_motive.py \
              --input "$in" --output "$out" \
              --annotations "$ANN" \
              --splits "$SPLITS" \
            || echo "WARN motive-eval failed: $in" >&2
      ' _ {}

# Plot MOTIVE sweep results in the gather_sweep_results style.
# Walks <sweep_dir>/**/motive/metrics.json, writes summary CSV + per-k plots
# to <output_dir> (default: data/results/figures/motive/). One figure set per
# best-selection mode (per_codec / best_avg_codec / best_any_codec /
# zstd_reference); filename suffix matches plot_motive_cross_task.py.
motive-plot sweep_dir output_dir="data/results/figures/motive" selections="per_codec,best_avg_codec,best_any_codec,zstd_reference":
    uv run python analysis/plot_motive_results.py \
        --sweep-dir {{ sweep_dir }} \
        --output-dir {{ output_dir }} \
        --k-pcts 1,5,10 \
        --best-selections {{ selections }}

# Plot MOTIVE codec-delta from the raw/lossless baseline per family.
# Mirrors gather_sweep_results.generate_codec_delta_from_raw_groups_plot
# (absolute deltas only — no percentage row). Reads the summary CSV emitted
# by motive-plot, so run motive-plot first.
motive-plot-delta plot_dir:
    uv run python analysis/plot_motive_codec_delta.py \
        --summary-csv {{ plot_dir }}/motive_sweep_summary.csv \
        --output-dir {{ plot_dir }} \
        --k-pcts 1,5,10

# LaTeX percentage-delta table for MOTIVE results, one .tex file per k%%.
# Mirrors gather_sweep_results' codec_delta_pct_table.tex layout. Reads the
# same summary CSV as motive-plot-delta, so run motive-plot first.
motive-table-delta plot_dir:
    uv run python analysis/generate_motive_delta_table.py \
        --summary-csv {{ plot_dir }}/motive_sweep_summary.csv \
        --output-dir {{ plot_dir }} \
        --k-pcts 1,5,10 \
        --force

# Cross-task scatter (CC vs GG, CG-Divs vs CG-Bioact) — one figure per k%%,
# with a `_noORF` variant alongside. Reads the same summary CSV as
# motive-plot-delta. Pass --show-all-points to also emit a faint background
# cloud of every config. `selection` mirrors gather_sweep_results --best-selection
# (per_codec / best_avg_codec / best_any_codec / zstd_reference); default
# best_avg_codec to pair with sweep_nap_pa_vs_pc_panel_a_best_avg_codec.
motive-plot-cross plot_dir selection="best_avg_codec":
    uv run python analysis/plot_motive_cross_task.py \
        --summary-csv {{ plot_dir }}/motive_sweep_summary.csv \
        --output-dir {{ plot_dir }} \
        --k-pcts 1,5,10 \
        --best-selection {{ selection }} \
        --show-all-points

# Per-codec model-rank heatmap across the 11 tasks from the combined
# RefChem + MOTIVE delta table. One PNG per codec (raw / hq / mq / d20),
# rows = models sorted by mean per-task-normalised score. RefChem CSV uses
# the script's default path (variance_first_v11_lite/sweep_results.csv);
# `selection` mirrors gather_sweep_results --best-selection.
model-task-rank plot_dir selection="best_avg_codec":
    uv run python analysis/plot_model_task_rank.py \
        --motive-csv {{ plot_dir }}/motive_sweep_summary.csv \
        --output-dir {{ plot_dir }} \
        --best-selection {{ selection }} \
        --codecs raw,hq,mq,d20

# Combined RefCam (PA/PC NAP) + MOTIVE (CC/GG/CG -> CRISPR) codec-delta
# table. Reads sweep_results.csv from JUMP_core/.../variance_first_v11_lite
# and motive_sweep_summary.csv from motive_large_strict. Writes both the
# full table and a summary-only variant.
combined-codec-delta-table:
    uv run python analysis/generate_combined_codec_delta_table.py

# Run MOTIVE evaluation across every output.parquet under a sweep dir.
# Idempotent — skips a config if its metrics.json already exists.
#
#   output_dir empty (default) — write next to input: <config_dir>/results/motive/
#   output_dir set             — mirror sweep subtree under it:
#                                   <sweep_dir>/<rel>/output.parquet
#                                   → <output_dir>/<rel>/motive/
#
# Examples:
#   just motive-eval-sweep src/norm_3/data/features/variance_first_v11_lite
#   just motive-eval-sweep src/norm_3/data/features/variance_first_v11_lite /scratch/motive_results jobs=8
motive-eval-sweep sweep_dir output_dir="" jobs="4" annotations="metadata/motive_annotations.parquet" splits="metadata/motive_eval_compounds.parquet":
    #!/usr/bin/env bash
    set -euo pipefail
    SWEEP="{{ sweep_dir }}"
    OUT_BASE="{{ output_dir }}"
    ANN="{{ annotations }}"
    SPLITS="{{ splits }}"
    export SWEEP OUT_BASE ANN SPLITS
    [ -f "$ANN" ]    || { echo "ERROR: annotations file not found: $ANN" >&2; exit 1; }
    [ -f "$SPLITS" ] || { echo "ERROR: splits file not found: $SPLITS" >&2; exit 1; }
    find "$SWEEP" -name output.parquet -print0 \
      | xargs -0 -n1 -P{{ jobs }} -I{} bash -c '
          in="$1"
          in_dir="$(dirname "$in")"
          if [ -n "$OUT_BASE" ]; then
              rel="${in_dir#${SWEEP%/}/}"
              out="$OUT_BASE/$rel/motive"
          else
              out="$in_dir/results/motive"
          fi
          mkdir -p "$out"
          uv run python src/motive/evaluate_motive.py \
              --input "$in" --output "$out" \
              --annotations "$ANN" \
              --splits "$SPLITS" \
            || echo "WARN motive-eval failed: $in" >&2
      ' _ {}

# End-to-end MOTIVE eval on the top-N configs of a norm_3 sweep:
# filter top-N -> eval full ann -> eval strict ann -> plot both. Override
# SWEEP_DIR / TOP_N / METRIC / JOBS / etc. via env vars at invocation
# (see scripts/run_motive_top.sh for the full list).
motive-run-top:
    bash scripts/run_motive_top.sh

# End-to-end MOTIVE eval on EVERY config under a sweep dir (fills in the
# non-top-N remainder; idempotent against an existing motive-run-top run).
motive-run-all:
    bash scripts/run_motive_all.sh

# ═══════════════════════════════════════════════════════════════
# Section 11: Auxiliary
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# Section 9: Bootstrap (run once if starting from zero)
# ═══════════════════════════════════════════════════════════════
# Skip this section if you already have raw TIFFs at {{ raw_images_lite }}
# and aliby outputs at {{ aliby_output }}. See prep/README.md for details
# and DATA_SOURCES.md for the upstream artifacts we cannot fetch.

# Build the URI manifest (jl_index_tidy.parquet) from JUMP-cellpainting GitHub + S3.
build-jl-index:
    mkdir -p {{ manifest_dir }}
    cd {{ manifest_dir }} && pixi run duckdb < {{ justfile_directory() }}/prep/build_jl_index.sql

# Download raw TIFFs from cellpainting-gallery S3 using the manifest.
download-raw n_jobs="16":
    pixi run python prep/download_raw.py \
        --manifest {{ manifest_dir }}/jl_index_tidy.parquet \
        --out-dir {{ raw_images_lite }} \
        --n-jobs {{ n_jobs }}

# Aliby featurization → aliby_output/. REQUIRES external aliby + Nahual GPU servers.
aliby-featurize:
    pixi run python prep/aliby_featurize.py

# Fetch the upstream annotation bundle (Zenodo 18197517 + cpg0042 duckdb) into
# {{ annotations_dir }}. md5-verified; idempotent.
fetch-annotations:
    uv run python prep/fetch_annotations.py --output-dir {{ annotations_dir }}

# Fetch raw CellProfiler profiles.parquet (cpg0016 v1.0c, ~13.5 GB) into
# {{ cp_profiles }}. Anonymous S3; size-verified; idempotent.
fetch-cp-profiles:
    uv run python prep/fetch_cp_profiles.py --output {{ cp_profiles }}

# Regenerate ref_chem_overlap.parquet from raw RefChemDB + JUMP compound table.
build-refchemdb-overlap:
    uv run python prep/build_refchemdb_overlap.py \
        --raw {{ refchemdb_raw }} \
        --jump-duckdb {{ annotations_db }} \
        --output {{ refchemdb_overlap }}

# Regenerate refchemdb_conf_jump_matched.parquet from overlap + JUMP duckdb.
build-refchemdb-matched:
    uv run python prep/build_refchemdb_matched.py \
        --overlap {{ refchemdb_overlap }} \
        --jump-duckdb {{ annotations_db }} \
        --output {{ refchemdb }}

# Full chain: raw RefChemDB → overlap → tier-matched parquet.
build-refchemdb: build-refchemdb-overlap build-refchemdb-matched

# ─── Annotation prep ─────────────────────────────────────────────
# Regenerate metadata/*.parquet from the upstream annotation bundle. Skip if
# you trust the committed metadata/ files.

# Run the full annotation chain: fetch upstream → metadata bundle → motive
# full → motive strict. Defaults to the committed metadata/motive_eval_compounds.parquet;
# override to regenerate from a different upstream MOTIVE splits file.
prep-annotations motive_splits_path="metadata/motive_eval_compounds.parquet":
    just fetch-annotations
    just build-refchemdb
    just metadata
    just motive-curate {{ motive_splits_path }}
    just motive-curate-strict
