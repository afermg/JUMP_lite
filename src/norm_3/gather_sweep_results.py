#!/usr/bin/env python3
"""Gather all metrics.json files from a norm_3 sweep into a single CSV table with plots.

Designed for the norm_3 variance-first pipeline config naming convention:
  {norm}_ctrl__outlier{cutoff}__INT__prune{thresh}[__pca{n}][__batch_method]

Batch methods:
  - none (no suffix)
  - tvn_original_k{k}
  - tvn_efaar_e{epsilon}[_c{n_components}]  (c suffix only when != 128)
  - cascade_tvn_k{k1}_k{k2}
  - ZCA-cor_{fit}_{epsilon}  (spherize)

Usage:
  python src/norm_3/gather_sweep_results.py --sweep-dir src/norm_3/data/features/variance_first_v4 --plot
  python src/norm_3/gather_sweep_results.py --sweep-dir src/norm_3/data/features/variance_first_v4 --plot --filter-degenerate
"""

import argparse
import json
import re
import time
from itertools import groupby as itertools_groupby
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

# Display names for compression codecs
COMPRESSION_DISPLAY = {
    "raw_jump_cp_profiles_reformatted_filtered": "raw",
    "zstd_raw": "raw",
    "zstd_filtered_raw": "raw_f",
    "jpegxl_lossy_hq_raw": "hq",
    "jpegxl_lossy_hq_filtered_raw": "hq_f",
    "jpegxl_lossy_effort_3_raw": "e3",
    "jpegxl_lossy_effort_3_filtered_raw": "e3_f",
    "jpegxl_lossy_mq_raw": "mq",
    "jpegxl_lossy_mq_filtered_raw": "mq_f",
    "jpegxl_lossy_lq_raw": "lq",
    "jpegxl_lossy_lq_filtered_raw": "lq_f",
    "jpegxl_lossy_d2_e8_raw": "d2_e8",
    "jpegxl_lossy_d2_e8_filtered_raw": "d2_e8_f",
    "jpegxl_lossy_d10_raw": "d10",
    "jpegxl_lossy_d10_filtered_raw": "d10_f",
    "jpegxl_lossy_d15_raw": "d15",
    "jpegxl_lossy_d30_raw": "d25",
    # Embedding models: DINOv2-490
    "dinov2_490_zstd_raw": "dv2_490_raw",
    "dinov2_490_jpegxl_lossy_hq_raw": "dv2_490_hq",
    "dinov2_490_jpegxl_lossy_effort_3_raw": "dv2_490_e3",
    "dinov2_490_jpegxl_lossy_mq_raw": "dv2_490_mq",
    "dinov2_490_jpegxl_lossy_lq_raw": "dv2_490_lq",
    # Embedding models: DINOv2-random
    "dinov2_random_zstd_raw": "dv2_rand_raw",
    "dinov2_random_jpegxl_lossy_hq_raw": "dv2_rand_hq",
    "dinov2_random_jpegxl_lossy_effort_3_raw": "dv2_rand_e3",
    "dinov2_random_jpegxl_lossy_mq_raw": "dv2_rand_mq",
    "dinov2_random_jpegxl_lossy_lq_raw": "dv2_rand_lq",
    # Embedding models: MorphEm
    "morphem_zstd_raw": "morphem_raw",
    "morphem_jpegxl_lossy_hq_raw": "morphem_hq",
    "morphem_jpegxl_lossy_effort_3_raw": "morphem_e3",
    "morphem_jpegxl_lossy_mq_raw": "morphem_mq",
    "morphem_jpegxl_lossy_lq_raw": "morphem_lq",
    # Embedding models: SubCell
    "subcell_zstd_raw": "subcell_raw",
    "subcell_jpegxl_lossy_hq_raw": "subcell_hq",
    "subcell_jpegxl_lossy_effort_3_raw": "subcell_e3",
    "subcell_jpegxl_lossy_mq_raw": "subcell_mq",
    "subcell_jpegxl_lossy_lq_raw": "subcell_lq",
    # Cell Count baseline (from cp_measure cell/nuclei counts only)
    "cell_count_zstd_raw": "cc_raw",
    "cell_count_jpegxl_lossy_hq_raw": "cc_hq",
    "cell_count_jpegxl_lossy_effort_3_raw": "cc_e3",
    "cell_count_jpegxl_lossy_mq_raw": "cc_mq",
    "cell_count_jpegxl_lossy_lq_raw": "cc_lq",
    "cell_count_jpegxl_lossy_d2_e8_raw": "cc_d2_e8",
    "cell_count_jpegxl_lossy_d10_raw": "cc_d10",
    "cell_count_lite_raw": "cc_lite_raw",
    # CellProfiler filtered_border_size (cp_measure from jump_target2_4plate_filtered)
    "zstd_filtered_border_size_raw": "cp_fbs_raw",
    "jpegxl_lossy_hq_filtered_border_size_raw": "cp_fbs_hq",
    "jpegxl_lossy_effort_3_filtered_border_size_raw": "cp_fbs_e3",
    "jpegxl_lossy_mq_filtered_border_size_raw": "cp_fbs_mq",
    "jpegxl_lossy_lq_filtered_border_size_raw": "cp_fbs_lq",
    "jpegxl_lossy_d2_e8_filtered_border_size_raw": "cp_fbs_d2_e8",
    "jpegxl_lossy_d10_filtered_border_size_raw": "cp_fbs_d10",
    # Embedding models: DINOv2 (dinov2 without 490/random, from jump_target2_4plate_cl)
    "dinov2_zstd_raw": "dv2_raw",
    "dinov2_jpegxl_lossy_hq_raw": "dv2_hq",
    "dinov2_jpegxl_lossy_effort_3_raw": "dv2_e3",
    "dinov2_jpegxl_lossy_mq_raw": "dv2_mq",
    "dinov2_jpegxl_lossy_lq_raw": "dv2_lq",
    "dinov2_jpegxl_lossy_d2_e8_raw": "dv2_d2_e8",
    "dinov2_jpegxl_lossy_d10_raw": "dv2_d10",
    # Embedding models: DINOv2-random d2_e8/d10 (new codecs from jump_target2_4plate_cl)
    "dinov2_random_jpegxl_lossy_d2_e8_raw": "dv2_rand_d2_e8",
    "dinov2_random_jpegxl_lossy_d10_raw": "dv2_rand_d10",
    # Embedding models: SubCell d2_e8/d10 (new codecs from jump_target2_4plate_cl)
    "subcell_jpegxl_lossy_d2_e8_raw": "subcell_d2_e8",
    "subcell_jpegxl_lossy_d10_raw": "subcell_d10",
    # Embedding models: MorphEm d2_e8/d10 (new codecs from jump_target2_4plate_cl)
    "morphem_jpegxl_lossy_d2_e8_raw": "morphem_d2_e8",
    "morphem_jpegxl_lossy_d10_raw": "morphem_d10",
    "morphem_jpegxl_lossy_d15_raw": "morphem_d15",
    "morphem_jpegxl_lossy_d20_e2_raw": "morphem_d20_e2",
    # Embedding models: OpenPhenom (bare, from plate4_rerun_scale_std)
    "openphenom_zstd_raw": "ophenom_raw",
    "openphenom_jpegxl_lossy_hq_raw": "ophenom_hq",
    "openphenom_jpegxl_lossy_effort_3_raw": "ophenom_e3",
    "openphenom_jpegxl_lossy_mq_raw": "ophenom_mq",
    "openphenom_jpegxl_lossy_mq_new_raw": "ophenom_mq_new",
    "openphenom_jpegxl_lossy_lq_raw": "ophenom_lq",
    "openphenom_jpegxl_lossy_d2_e8_raw": "ophenom_d2_e8",
    "openphenom_jpegxl_lossy_d10_raw": "ophenom_d10",
    "openphenom_jpegxl_lossy_d15_raw": "ophenom_d15",
    "openphenom_jpegxl_lossy_d20_e2_raw": "ophenom_d20_e2",
    "openphenom_jpegxl_lossy_d30_raw": "ophenom_d25",
    "openphenom_jpegxl_lossy_d50_raw": "ophenom_d50",
    # Embedding models: OpenPhenom stdscale (from plate4_rerun_scale_std)
    "openphenom_stdscale_zstd_raw": "ophenom_ss_raw",
    "openphenom_stdscale_jpegxl_lossy_hq_raw": "ophenom_ss_hq",
    "openphenom_stdscale_jpegxl_lossy_effort_3_raw": "ophenom_ss_e3",
    "openphenom_stdscale_jpegxl_lossy_mq_raw": "ophenom_ss_mq",
    "openphenom_stdscale_jpegxl_lossy_lq_raw": "ophenom_ss_lq",
    "openphenom_stdscale_jpegxl_lossy_d2_e8_raw": "ophenom_ss_d2_e8",
    "openphenom_stdscale_jpegxl_lossy_d10_raw": "ophenom_ss_d10",
    "openphenom_stdscale_jpegxl_lossy_d15_raw": "ophenom_ss_d15",
    "openphenom_stdscale_jpegxl_lossy_d20_e2_raw": "ophenom_ss_d20_e2",
    # DINOv2-random d15/d20_e2/d30
    "dinov2_random_jpegxl_lossy_d15_raw": "dv2_rand_d15",
    "dinov2_random_jpegxl_lossy_d20_e2_raw": "dv2_rand_d20_e2",
    "dinov2_random_jpegxl_lossy_d30_raw": "dv2_rand_d25",
    "dinov2_random_jpegxl_lossy_d50_raw": "dv2_rand_d50",
    # SubCell d15/d20_e2/d30/d50
    "subcell_jpegxl_lossy_d15_raw": "subcell_d15",
    "subcell_jpegxl_lossy_d20_e2_raw": "subcell_d20_e2",
    "subcell_jpegxl_lossy_d30_raw": "subcell_d25",
    "subcell_jpegxl_lossy_d50_raw": "subcell_d50",
    # MorphEm d30/d50
    "morphem_jpegxl_lossy_d30_raw": "morphem_d25",
    "morphem_jpegxl_lossy_d50_raw": "morphem_d50",
    # OpenPhenom stdscale d30/d50
    "openphenom_stdscale_jpegxl_lossy_d30_raw": "ophenom_ss_d25",
    "openphenom_stdscale_jpegxl_lossy_d50_raw": "ophenom_ss_d50",
    # DINOv2 d15/d20_e2/d30/d50
    "dinov2_jpegxl_lossy_d15_raw": "dv2_d15",
    "dinov2_jpegxl_lossy_d20_e2_raw": "dv2_d20_e2",
    "dinov2_jpegxl_lossy_d30_raw": "dv2_d25",
    "dinov2_jpegxl_lossy_d50_raw": "dv2_d50",
    "dinov2_jpegxl_lossy_mq_new_raw": "dv2_mq_new",
    # mq_new codec variants
    "dinov2_random_jpegxl_lossy_mq_new_raw": "dv2_rand_mq_new",
    "morphem_jpegxl_lossy_mq_new_raw": "morphem_mq_new",
    "subcell_jpegxl_lossy_mq_new_raw": "subcell_mq_new",
    "openphenom_stdscale_jpegxl_lossy_mq_new_raw": "ophenom_ss_mq_new",
    # SubCell clip01
    "subcell__clip01_zstd_raw": "sc01_raw",
    "subcell__clip01_jpegxl_lossy_hq_raw": "sc01_hq",
    "subcell__clip01_jpegxl_lossy_effort_3_raw": "sc01_e3",
    "subcell__clip01_jpegxl_lossy_d2_e8_raw": "sc01_d2_e8",
    "subcell__clip01_jpegxl_lossy_mq_new_raw": "sc01_mq_new",
    "subcell__clip01_jpegxl_lossy_mq_raw": "sc01_mq",
    "subcell__clip01_jpegxl_lossy_lq_raw": "sc01_lq",
    "subcell__clip01_jpegxl_lossy_d10_raw": "sc01_d10",
    "subcell__clip01_jpegxl_lossy_d15_raw": "sc01_d15",
    "subcell__clip01_jpegxl_lossy_d30_raw": "sc01_d25",
    # === Rerun families (from jump_target2_4plate_cl_rerun/) ===
    # DINOv2 rerun
    "dinov2_cl_zstd_rr_raw": "dv2_rr_raw",
    "dinov2_cl_jpegxl_lossy_hq_rr_raw": "dv2_rr_hq",
    "dinov2_cl_jpegxl_lossy_effort_3_rr_raw": "dv2_rr_e3",
    "dinov2_cl_jpegxl_lossy_mq_rr_raw": "dv2_rr_mq",
    "dinov2_cl_jpegxl_lossy_mq_new_rr_raw": "dv2_rr_mq_new",
    "dinov2_cl_jpegxl_lossy_lq_rr_raw": "dv2_rr_lq",
    "dinov2_cl_jpegxl_lossy_d2_e8_rr_raw": "dv2_rr_d2_e8",
    "dinov2_cl_jpegxl_lossy_d10_rr_raw": "dv2_rr_d10",
    "dinov2_cl_jpegxl_lossy_d15_rr_raw": "dv2_rr_d15",
    "dinov2_cl_jpegxl_lossy_d20_e2_rr_raw": "dv2_rr_d20_e2",
    "dinov2_cl_jpegxl_lossy_d30_rr_raw": "dv2_rr_d25",
    "dinov2_cl_jpegxl_lossy_d50_rr_raw": "dv2_rr_d50",
    # DINOv2-random rerun
    "dinov2_random_zstd_rr_raw": "dv2_rand_rr_raw",
    "dinov2_random_jpegxl_lossy_hq_rr_raw": "dv2_rand_rr_hq",
    "dinov2_random_jpegxl_lossy_effort_3_rr_raw": "dv2_rand_rr_e3",
    "dinov2_random_jpegxl_lossy_mq_rr_raw": "dv2_rand_rr_mq",
    "dinov2_random_jpegxl_lossy_mq_new_rr_raw": "dv2_rand_rr_mq_new",
    "dinov2_random_jpegxl_lossy_lq_rr_raw": "dv2_rand_rr_lq",
    "dinov2_random_jpegxl_lossy_d2_e8_rr_raw": "dv2_rand_rr_d2_e8",
    "dinov2_random_jpegxl_lossy_d10_rr_raw": "dv2_rand_rr_d10",
    "dinov2_random_jpegxl_lossy_d15_rr_raw": "dv2_rand_rr_d15",
    "dinov2_random_jpegxl_lossy_d20_e2_rr_raw": "dv2_rand_rr_d20_e2",
    "dinov2_random_jpegxl_lossy_d30_rr_raw": "dv2_rand_rr_d25",
    "dinov2_random_jpegxl_lossy_d50_rr_raw": "dv2_rand_rr_d50",
    # MorphEm rerun
    "morphem_zstd_rr_raw": "morphem_rr_raw",
    "morphem_jpegxl_lossy_hq_rr_raw": "morphem_rr_hq",
    "morphem_jpegxl_lossy_effort_3_rr_raw": "morphem_rr_e3",
    "morphem_jpegxl_lossy_mq_rr_raw": "morphem_rr_mq",
    "morphem_jpegxl_lossy_mq_new_rr_raw": "morphem_rr_mq_new",
    "morphem_jpegxl_lossy_lq_rr_raw": "morphem_rr_lq",
    "morphem_jpegxl_lossy_d2_e8_rr_raw": "morphem_rr_d2_e8",
    "morphem_jpegxl_lossy_d10_rr_raw": "morphem_rr_d10",
    "morphem_jpegxl_lossy_d15_rr_raw": "morphem_rr_d15",
    "morphem_jpegxl_lossy_d20_e2_rr_raw": "morphem_rr_d20_e2",
    "morphem_jpegxl_lossy_d30_rr_raw": "morphem_rr_d25",
    "morphem_jpegxl_lossy_d50_rr_raw": "morphem_rr_d50",
    # SubCell rerun
    "subcell_zstd_rr_raw": "subcell_rr_raw",
    "subcell_jpegxl_lossy_hq_rr_raw": "subcell_rr_hq",
    "subcell_jpegxl_lossy_effort_3_rr_raw": "subcell_rr_e3",
    "subcell_jpegxl_lossy_mq_rr_raw": "subcell_rr_mq",
    "subcell_jpegxl_lossy_mq_new_rr_raw": "subcell_rr_mq_new",
    "subcell_jpegxl_lossy_lq_rr_raw": "subcell_rr_lq",
    "subcell_jpegxl_lossy_d2_e8_rr_raw": "subcell_rr_d2_e8",
    "subcell_jpegxl_lossy_d10_rr_raw": "subcell_rr_d10",
    "subcell_jpegxl_lossy_d15_rr_raw": "subcell_rr_d15",
    "subcell_jpegxl_lossy_d20_e2_rr_raw": "subcell_rr_d20_e2",
    "subcell_jpegxl_lossy_d30_rr_raw": "subcell_rr_d25",
    "subcell_jpegxl_lossy_d50_rr_raw": "subcell_rr_d50",
    # OpenPhenom rerun
    "openphenom_zstd_rr_raw": "ophenom_rr_raw",
    "openphenom_jpegxl_lossy_hq_rr_raw": "ophenom_rr_hq",
    "openphenom_jpegxl_lossy_effort_3_rr_raw": "ophenom_rr_e3",
    "openphenom_jpegxl_lossy_mq_rr_raw": "ophenom_rr_mq",
    "openphenom_jpegxl_lossy_mq_new_rr_raw": "ophenom_rr_mq_new",
    "openphenom_jpegxl_lossy_lq_rr_raw": "ophenom_rr_lq",
    "openphenom_jpegxl_lossy_d2_e8_rr_raw": "ophenom_rr_d2_e8",
    "openphenom_jpegxl_lossy_d10_rr_raw": "ophenom_rr_d10",
    "openphenom_jpegxl_lossy_d15_rr_raw": "ophenom_rr_d15",
    "openphenom_jpegxl_lossy_d20_e2_rr_raw": "ophenom_rr_d20_e2",
    "openphenom_jpegxl_lossy_d30_rr_raw": "ophenom_rr_d25",
    "openphenom_jpegxl_lossy_d50_rr_raw": "ophenom_rr_d50",
    # Jump-lite models
    "cellprofiler_lite_raw": "cp_lite_raw",
    "dinov2_lite_jpegxl_lossy_raw_raw": "dv2_lite_raw",
    "dinov2_lite_jpegxl_lossy_hq_raw": "dv2_lite_hq",
    "dinov2_lite_jpegxl_lossy_mq_raw": "dv2_lite_mq",
    "dinov2_lite_jpegxl_lossy_d20_raw": "dv2_lite_d20",
    "dinov2_random_lite_jpegxl_lossy_raw_raw": "dv2_rand_lite_raw",
    "dinov2_random_lite_jpegxl_lossy_hq_raw": "dv2_rand_lite_hq",
    "dinov2_random_lite_jpegxl_lossy_mq_raw": "dv2_rand_lite_mq",
    "dinov2_random_lite_jpegxl_lossy_d20_raw": "dv2_rand_lite_d20",
    "morphem_lite_jpegxl_lossy_raw_raw": "morphem_lite_raw",
    "morphem_lite_jpegxl_lossy_hq_raw": "morphem_lite_hq",
    "morphem_lite_jpegxl_lossy_mq_raw": "morphem_lite_mq",
    "morphem_lite_jpegxl_lossy_d20_raw": "morphem_lite_d20",
    "subcell_lite_jpegxl_lossy_mq_raw": "subcell_lite_mq",
    "subcell__clip01_lite_jpegxl_lossy_raw_raw": "sc01_lite_raw",
    "subcell__clip01_lite_jpegxl_lossy_mq_raw": "sc01_lite_mq",
    "subcell__clip01_lite_jpegxl_lossy_d20_raw": "sc01_lite_d20",
    "subcell__clip01_lite_jpegxl_lossy_hq_raw": "sc01_lite_hq",
    "openphenom_lite_jpegxl_lossy_raw_raw": "ophenom_lite_raw",
    "openphenom_lite_jpegxl_lossy_hq_raw": "ophenom_lite_hq",
    "openphenom_lite_jpegxl_lossy_mq_raw": "ophenom_lite_mq",
    "openphenom_lite_jpegxl_lossy_d20_raw": "ophenom_lite_d20",
}

# Order for compression codecs (raw first, then filtered, by quality, then embedding models)
# Canonical codec ordering (lossless → heavy lossy) used to sort codecs within each model family.
# Maps codec substring patterns found in raw model names to a sort rank.
_CODEC_SORT_ORDER = {
    "raw_jump_cp_profiles": 0,  # special: reformatted CP baseline
    "jpegxl_lossy_raw": 0,     # features from raw (uncompressed) images
    "zstd": 1,
    "jpegxl_lossy_hq": 2,
    "jpegxl_lossy_effort_3": 3,
    "jpegxl_lossy_d2_e8": 4,
    "jpegxl_lossy_mq_new": 5,
    "jpegxl_lossy_mq": 6,
    "jpegxl_lossy_lq": 7,
    "jpegxl_lossy_d10": 8,
    "jpegxl_lossy_d15": 9,
    "jpegxl_lossy_d20": 10,
    "jpegxl_lossy_d20_e2": 10,
    "jpegxl_lossy_d30": 11,
    "jpegxl_lossy_d50": 12,
}


def _get_codec_sort_rank(model: str) -> int:
    """Extract codec sort rank from a raw model name.

    Tries each codec pattern (longest first to avoid partial matches like
    'jpegxl_lossy_d10' matching before 'jpegxl_lossy_d2_e8').
    """
    # Sort patterns longest-first so e.g. jpegxl_lossy_d2_e8 matches before jpegxl_lossy_d10
    for codec, rank in sorted(_CODEC_SORT_ORDER.items(), key=lambda kv: -len(kv[0])):
        if codec in model:
            return rank
    return 99  # unknown codec goes last

# Batch method display names
BATCH_DISPLAY = {
    "none": "None",
    "tvn_original": "TVN Original",
    "tvn_efaar": "TVN EFAAR",
    "cascade_tvn": "Cascade TVN",
    "spherize": "Spherize",
}

# Model family groupings for color assignment
# Each family gets a distinct hue; codecs within a family get brightness variations
MODEL_FAMILIES = {
    # CellProfiler reformatted (original JUMP CP profiles)
    "cellprofiler": [
        "raw_jump_cp_profiles_reformatted_filtered",
    ],
    # cp_measure raw
    "cp_measure": [
        "zstd_raw", "jpegxl_lossy_hq_raw", "jpegxl_lossy_effort_3_raw",
        "jpegxl_lossy_d2_e8_raw", "jpegxl_lossy_mq_raw",
        "jpegxl_lossy_lq_raw", "jpegxl_lossy_d10_raw",
        "jpegxl_lossy_d15_raw", "jpegxl_lossy_d30_raw",
    ],
    # Cell Count baseline (cell/nuclei counts from CP parquets)
    "cell_count": [
        "cell_count_zstd_raw", "cell_count_jpegxl_lossy_hq_raw",
        "cell_count_jpegxl_lossy_effort_3_raw", "cell_count_jpegxl_lossy_d2_e8_raw",
        "cell_count_jpegxl_lossy_mq_raw", "cell_count_jpegxl_lossy_lq_raw",
        "cell_count_jpegxl_lossy_d10_raw",
    ],
    "cell_count_lite": [
        "cell_count_lite_raw",
    ],
    # cp_measure filtered
    "cp_measure_filtered": [
        "zstd_filtered_raw", "jpegxl_lossy_hq_filtered_raw",
        "jpegxl_lossy_effort_3_filtered_raw", "jpegxl_lossy_d2_e8_filtered_raw",
        "jpegxl_lossy_mq_filtered_raw", "jpegxl_lossy_lq_filtered_raw",
        "jpegxl_lossy_d10_filtered_raw",
    ],
    # DINOv2-490
    "dinov2_490": [
        "dinov2_490_zstd_raw", "dinov2_490_jpegxl_lossy_hq_raw",
        "dinov2_490_jpegxl_lossy_effort_3_raw", "dinov2_490_jpegxl_lossy_mq_raw",
        "dinov2_490_jpegxl_lossy_lq_raw",
    ],
    # DINOv2-random
    "dinov2_random": [
        "dinov2_random_zstd_raw", "dinov2_random_jpegxl_lossy_hq_raw",
        "dinov2_random_jpegxl_lossy_effort_3_raw", "dinov2_random_jpegxl_lossy_d2_e8_raw",
        "dinov2_random_jpegxl_lossy_mq_new_raw", "dinov2_random_jpegxl_lossy_mq_raw",
        "dinov2_random_jpegxl_lossy_lq_raw",
        "dinov2_random_jpegxl_lossy_d10_raw", "dinov2_random_jpegxl_lossy_d15_raw",
        "dinov2_random_jpegxl_lossy_d20_e2_raw", "dinov2_random_jpegxl_lossy_d30_raw",
        "dinov2_random_jpegxl_lossy_d50_raw",
    ],
    # DINOv2 (non-random, from jump_target2_4plate_cl)
    "dinov2": [
        "dinov2_zstd_raw", "dinov2_jpegxl_lossy_hq_raw",
        "dinov2_jpegxl_lossy_effort_3_raw", "dinov2_jpegxl_lossy_d2_e8_raw",
        "dinov2_jpegxl_lossy_mq_new_raw", "dinov2_jpegxl_lossy_mq_raw",
        "dinov2_jpegxl_lossy_lq_raw",
        "dinov2_jpegxl_lossy_d10_raw", "dinov2_jpegxl_lossy_d15_raw",
        "dinov2_jpegxl_lossy_d20_e2_raw", "dinov2_jpegxl_lossy_d30_raw",
        "dinov2_jpegxl_lossy_d50_raw",
    ],
    # MorphEm
    "morphem": [
        "morphem_zstd_raw", "morphem_jpegxl_lossy_hq_raw",
        "morphem_jpegxl_lossy_effort_3_raw", "morphem_jpegxl_lossy_d2_e8_raw",
        "morphem_jpegxl_lossy_mq_new_raw", "morphem_jpegxl_lossy_mq_raw",
        "morphem_jpegxl_lossy_lq_raw",
        "morphem_jpegxl_lossy_d10_raw", "morphem_jpegxl_lossy_d15_raw",
        "morphem_jpegxl_lossy_d20_e2_raw", "morphem_jpegxl_lossy_d30_raw",
        "morphem_jpegxl_lossy_d50_raw",
    ],
    # SubCell
    "subcell": [
        "subcell_zstd_raw", "subcell_jpegxl_lossy_hq_raw",
        "subcell_jpegxl_lossy_effort_3_raw", "subcell_jpegxl_lossy_d2_e8_raw",
        "subcell_jpegxl_lossy_mq_new_raw", "subcell_jpegxl_lossy_mq_raw",
        "subcell_jpegxl_lossy_lq_raw",
        "subcell_jpegxl_lossy_d10_raw", "subcell_jpegxl_lossy_d15_raw",
        "subcell_jpegxl_lossy_d20_e2_raw", "subcell_jpegxl_lossy_d30_raw",
        "subcell_jpegxl_lossy_d50_raw",
    ],
    # OpenPhenom (bare, from plate4_rerun_scale_std)
    "openphenom": [
        "openphenom_zstd_raw", "openphenom_jpegxl_lossy_hq_raw",
        "openphenom_jpegxl_lossy_effort_3_raw", "openphenom_jpegxl_lossy_d2_e8_raw",
        "openphenom_jpegxl_lossy_mq_new_raw", "openphenom_jpegxl_lossy_mq_raw",
        "openphenom_jpegxl_lossy_lq_raw",
        "openphenom_jpegxl_lossy_d10_raw", "openphenom_jpegxl_lossy_d15_raw",
        "openphenom_jpegxl_lossy_d20_e2_raw", "openphenom_jpegxl_lossy_d30_raw",
        "openphenom_jpegxl_lossy_d50_raw",
    ],
    # OpenPhenom stdscale (from plate4_rerun_scale_std/)
    "openphenom_stdscale": [
        "openphenom_stdscale_zstd_raw", "openphenom_stdscale_jpegxl_lossy_hq_raw",
        "openphenom_stdscale_jpegxl_lossy_effort_3_raw", "openphenom_stdscale_jpegxl_lossy_d2_e8_raw",
        "openphenom_stdscale_jpegxl_lossy_mq_new_raw", "openphenom_stdscale_jpegxl_lossy_mq_raw",
        "openphenom_stdscale_jpegxl_lossy_lq_raw",
        "openphenom_stdscale_jpegxl_lossy_d10_raw", "openphenom_stdscale_jpegxl_lossy_d15_raw",
        "openphenom_stdscale_jpegxl_lossy_d20_e2_raw", "openphenom_stdscale_jpegxl_lossy_d30_raw",
        "openphenom_stdscale_jpegxl_lossy_d50_raw",
    ],
    # OpenPhenom nonclip (from plate4_rerun_scale_std/)
    "openphenom_nonclip": [
        "openphenom_nonclip_zstd_raw", "openphenom_nonclip_jpegxl_lossy_hq_raw",
        "openphenom_nonclip_jpegxl_lossy_effort_3_raw", "openphenom_nonclip_jpegxl_lossy_d2_e8_raw",
        "openphenom_nonclip_jpegxl_lossy_mq_new_raw", "openphenom_nonclip_jpegxl_lossy_mq_raw",
        "openphenom_nonclip_jpegxl_lossy_lq_raw",
        "openphenom_nonclip_jpegxl_lossy_d10_raw", "openphenom_nonclip_jpegxl_lossy_d15_raw",
        "openphenom_nonclip_jpegxl_lossy_d20_e2_raw", "openphenom_nonclip_jpegxl_lossy_d30_raw",
        "openphenom_nonclip_jpegxl_lossy_d50_raw",
    ],
    # OpenPhenom stdscale_false (from plate4_rerun_scale_std/)
    "openphenom_stdscale_false": [
        "openphenom_stdscale_false_zstd_raw", "openphenom_stdscale_false_jpegxl_lossy_hq_raw",
        "openphenom_stdscale_false_jpegxl_lossy_effort_3_raw", "openphenom_stdscale_false_jpegxl_lossy_d2_e8_raw",
        "openphenom_stdscale_false_jpegxl_lossy_mq_new_raw", "openphenom_stdscale_false_jpegxl_lossy_mq_raw",
        "openphenom_stdscale_false_jpegxl_lossy_lq_raw",
        "openphenom_stdscale_false_jpegxl_lossy_d10_raw", "openphenom_stdscale_false_jpegxl_lossy_d15_raw",
        "openphenom_stdscale_false_jpegxl_lossy_d20_e2_raw", "openphenom_stdscale_false_jpegxl_lossy_d30_raw",
        "openphenom_stdscale_false_jpegxl_lossy_d50_raw",
    ],
    # SubCell double-underscore nonstd (from plate4_rerun_scale_std/)
    "subcell__nonstd": [
        "subcell__nonstd_zstd_raw", "subcell__nonstd_jpegxl_lossy_hq_raw",
        "subcell__nonstd_jpegxl_lossy_effort_3_raw", "subcell__nonstd_jpegxl_lossy_d2_e8_raw",
        "subcell__nonstd_jpegxl_lossy_mq_new_raw", "subcell__nonstd_jpegxl_lossy_mq_raw",
        "subcell__nonstd_jpegxl_lossy_lq_raw",
        "subcell__nonstd_jpegxl_lossy_d10_raw", "subcell__nonstd_jpegxl_lossy_d15_raw",
        "subcell__nonstd_jpegxl_lossy_d20_e2_raw", "subcell__nonstd_jpegxl_lossy_d30_raw",
        "subcell__nonstd_jpegxl_lossy_d50_raw",
    ],
    # SubCell nonstd (from plate4_rerun_scale_std/, only mq_new)
    "subcell_nonstd": [
        "subcell_nonstd_jpegxl_lossy_mq_new_raw",
    ],
    # SubCell clip01
    "subcell__clip01": [
        "subcell__clip01_zstd_raw",
        "subcell__clip01_jpegxl_lossy_hq_raw",
        "subcell__clip01_jpegxl_lossy_effort_3_raw",
        "subcell__clip01_jpegxl_lossy_d2_e8_raw",
        "subcell__clip01_jpegxl_lossy_mq_new_raw",
        "subcell__clip01_jpegxl_lossy_mq_raw",
        "subcell__clip01_jpegxl_lossy_lq_raw",
        "subcell__clip01_jpegxl_lossy_d10_raw",
        "subcell__clip01_jpegxl_lossy_d15_raw",
        "subcell__clip01_jpegxl_lossy_d30_raw",
    ],
    # SubCell wrongchannels (from plate4_rerun_scale_std/)
    "subcell_wrongchannels": [
        "subcell_wrongchannels_zstd_raw", "subcell_wrongchannels_jpegxl_lossy_hq_raw",
        "subcell_wrongchannels_jpegxl_lossy_effort_3_raw", "subcell_wrongchannels_jpegxl_lossy_d2_e8_raw",
        "subcell_wrongchannels_jpegxl_lossy_mq_new_raw", "subcell_wrongchannels_jpegxl_lossy_mq_raw",
        "subcell_wrongchannels_jpegxl_lossy_lq_raw",
        "subcell_wrongchannels_jpegxl_lossy_d10_raw", "subcell_wrongchannels_jpegxl_lossy_d15_raw",
        "subcell_wrongchannels_jpegxl_lossy_d20_e2_raw", "subcell_wrongchannels_jpegxl_lossy_d30_raw",
        "subcell_wrongchannels_jpegxl_lossy_d50_raw",
    ],
    # === Rerun families (from jump_target2_4plate_cl_rerun/) ===
    "dinov2_rr": [
        "dinov2_cl_zstd_rr_raw", "dinov2_cl_jpegxl_lossy_hq_rr_raw",
        "dinov2_cl_jpegxl_lossy_effort_3_rr_raw", "dinov2_cl_jpegxl_lossy_mq_rr_raw",
        "dinov2_cl_jpegxl_lossy_mq_new_rr_raw", "dinov2_cl_jpegxl_lossy_lq_rr_raw",
        "dinov2_cl_jpegxl_lossy_d2_e8_rr_raw", "dinov2_cl_jpegxl_lossy_d10_rr_raw",
        "dinov2_cl_jpegxl_lossy_d15_rr_raw", "dinov2_cl_jpegxl_lossy_d20_e2_rr_raw",
        "dinov2_cl_jpegxl_lossy_d30_rr_raw", "dinov2_cl_jpegxl_lossy_d50_rr_raw",
    ],
    "dinov2_random_rr": [
        "dinov2_random_zstd_rr_raw", "dinov2_random_jpegxl_lossy_hq_rr_raw",
        "dinov2_random_jpegxl_lossy_effort_3_rr_raw", "dinov2_random_jpegxl_lossy_mq_rr_raw",
        "dinov2_random_jpegxl_lossy_mq_new_rr_raw", "dinov2_random_jpegxl_lossy_lq_rr_raw",
        "dinov2_random_jpegxl_lossy_d2_e8_rr_raw", "dinov2_random_jpegxl_lossy_d10_rr_raw",
        "dinov2_random_jpegxl_lossy_d15_rr_raw", "dinov2_random_jpegxl_lossy_d20_e2_rr_raw",
        "dinov2_random_jpegxl_lossy_d30_rr_raw", "dinov2_random_jpegxl_lossy_d50_rr_raw",
    ],
    "morphem_rr": [
        "morphem_zstd_rr_raw", "morphem_jpegxl_lossy_hq_rr_raw",
        "morphem_jpegxl_lossy_effort_3_rr_raw", "morphem_jpegxl_lossy_mq_rr_raw",
        "morphem_jpegxl_lossy_mq_new_rr_raw", "morphem_jpegxl_lossy_lq_rr_raw",
        "morphem_jpegxl_lossy_d2_e8_rr_raw", "morphem_jpegxl_lossy_d10_rr_raw",
        "morphem_jpegxl_lossy_d15_rr_raw", "morphem_jpegxl_lossy_d20_e2_rr_raw",
        "morphem_jpegxl_lossy_d30_rr_raw", "morphem_jpegxl_lossy_d50_rr_raw",
    ],
    "subcell_rr": [
        "subcell_zstd_rr_raw", "subcell_jpegxl_lossy_hq_rr_raw",
        "subcell_jpegxl_lossy_effort_3_rr_raw", "subcell_jpegxl_lossy_mq_rr_raw",
        "subcell_jpegxl_lossy_mq_new_rr_raw", "subcell_jpegxl_lossy_lq_rr_raw",
        "subcell_jpegxl_lossy_d2_e8_rr_raw", "subcell_jpegxl_lossy_d10_rr_raw",
        "subcell_jpegxl_lossy_d15_rr_raw", "subcell_jpegxl_lossy_d20_e2_rr_raw",
        "subcell_jpegxl_lossy_d30_rr_raw", "subcell_jpegxl_lossy_d50_rr_raw",
    ],
    "openphenom_rr": [
        "openphenom_zstd_rr_raw", "openphenom_jpegxl_lossy_hq_rr_raw",
        "openphenom_jpegxl_lossy_effort_3_rr_raw", "openphenom_jpegxl_lossy_mq_rr_raw",
        "openphenom_jpegxl_lossy_mq_new_rr_raw", "openphenom_jpegxl_lossy_lq_rr_raw",
        "openphenom_jpegxl_lossy_d2_e8_rr_raw", "openphenom_jpegxl_lossy_d10_rr_raw",
        "openphenom_jpegxl_lossy_d15_rr_raw", "openphenom_jpegxl_lossy_d20_e2_rr_raw",
        "openphenom_jpegxl_lossy_d30_rr_raw", "openphenom_jpegxl_lossy_d50_rr_raw",
    ],
    # Jump-lite families
    "cellprofiler_lite": [
        "cellprofiler_lite_raw",
    ],
    "dinov2_lite": [
        "dinov2_lite_jpegxl_lossy_raw_raw",
        "dinov2_lite_jpegxl_lossy_hq_raw",
        "dinov2_lite_jpegxl_lossy_mq_raw",
        "dinov2_lite_jpegxl_lossy_d20_raw",
    ],
    "dinov2_random_lite": [
        "dinov2_random_lite_jpegxl_lossy_raw_raw",
        "dinov2_random_lite_jpegxl_lossy_hq_raw",
        "dinov2_random_lite_jpegxl_lossy_mq_raw",
        "dinov2_random_lite_jpegxl_lossy_d20_raw",
    ],
    "morphem_lite": [
        "morphem_lite_jpegxl_lossy_raw_raw",
        "morphem_lite_jpegxl_lossy_hq_raw",
        "morphem_lite_jpegxl_lossy_mq_raw",
        "morphem_lite_jpegxl_lossy_d20_raw",
    ],
    "subcell_lite": [
        "subcell_lite_jpegxl_lossy_mq_raw",
    ],
    "subcell__clip01_lite": [
        "subcell__clip01_lite_jpegxl_lossy_raw_raw",
        "subcell__clip01_lite_jpegxl_lossy_hq_raw",
        "subcell__clip01_lite_jpegxl_lossy_mq_raw",
        "subcell__clip01_lite_jpegxl_lossy_d20_raw",
    ],
    "openphenom_lite": [
        "openphenom_lite_jpegxl_lossy_raw_raw",
        "openphenom_lite_jpegxl_lossy_hq_raw",
        "openphenom_lite_jpegxl_lossy_mq_raw",
        "openphenom_lite_jpegxl_lossy_d20_raw",
    ],
    # CellProfiler filtered_border_size (from jump_target2_4plate_filtered)
    "cp_measure_fbs": [
        "zstd_filtered_border_size_raw",
        "jpegxl_lossy_hq_filtered_border_size_raw",
        "jpegxl_lossy_effort_3_filtered_border_size_raw",
        "jpegxl_lossy_d2_e8_filtered_border_size_raw",
        "jpegxl_lossy_mq_filtered_border_size_raw",
        "jpegxl_lossy_lq_filtered_border_size_raw",
        "jpegxl_lossy_d10_filtered_border_size_raw",
    ],
}

# Base hues for each family (HSV hue in [0, 1])
FAMILY_HUES = {
    "cellprofiler": 0.0,       # Red
    "cp_measure": 0.05,        # Orange-red
    "cell_count": 0.90,        # Pink-magenta
    "cell_count_lite": 0.90,   # Same as cell_count
    "cp_measure_fbs": 0.10,    # Yellow-orange
    "cp_measure_filtered": 0.14,  # Orange
    "dinov2": 0.20,            # Yellow-green
    "dinov2_490": 0.23,        # Green (near dinov2)
    "dinov2_random": 0.35,     # Green
    "morphem": 0.50,           # Cyan
    "subcell": 0.95,           # Hot pink / magenta-red
    "openphenom": 0.65,        # Blue
    "openphenom_stdscale": 0.70,  # Blue-violet
    "openphenom_nonclip": 0.73,   # Violet
    "openphenom_stdscale_false": 0.76,  # Purple
    "subcell__nonstd": 0.93,      # Pink (near subcell)
    "subcell_nonstd": 0.94,       # Pink (near subcell)
    "subcell__clip01": 0.80,      # Purple-magenta
    "subcell_wrongchannels": 0.96,  # Pink-red (near subcell)
    # Rerun families (shifted hues to distinguish from originals)
    "dinov2_rr": 0.24,         # Yellow-green (near dinov2)
    "dinov2_random_rr": 0.42,  # Teal (near dinov2_random)
    "morphem_rr": 0.55,        # Cyan (near morphem)
    "subcell_rr": 0.93,        # Pink (near subcell)
    "openphenom_rr": 0.92,     # Pink-magenta (near openphenom)
    # Jump-lite families (same hues as originals)
    "cellprofiler_lite": 0.0,   # Red (same as cellprofiler)
    "dinov2_lite": 0.20,        # Yellow-green (same as dinov2)
    "dinov2_random_lite": 0.35, # Green (same as dinov2_random)
    "morphem_lite": 0.50,       # Cyan (same as morphem)
    "subcell_lite": 0.95,       # Hot pink (same as subcell)
    "subcell__clip01_lite": 0.80,  # Purple-magenta (same as subcell__clip01)
    "openphenom_lite": 0.65,    # Blue (same as openphenom)
}

# Fixed Set2 palette index per family (used by group_nap and combined plots).
# Set2 has 8 colours; families sharing an index get the same colour.
# Swap: ViT-rand families use index 4, SubCell families use index 1.
_SET2_PALETTE = sns.color_palette("Set2", 8)
_CELLCOUNT_PINK = tuple(c * 0.75 for c in _SET2_PALETTE[3])  # darkened pink for visibility
FAMILY_SET2_COLOR: dict[str, tuple] = {
    # CellCount (darkened pink for visibility at low alpha)
    "cell_count": _CELLCOUNT_PINK, "cell_count_lite": _CELLCOUNT_PINK,
    # 1 – CellProfiler
    "cellprofiler": _SET2_PALETTE[1], "cellprofiler_lite": _SET2_PALETTE[1],
    "cp_measure": _SET2_PALETTE[1], "cp_measure_filtered": _SET2_PALETTE[1],
    "cp_measure_fbs": _SET2_PALETTE[1],
    # 2 – SubCell
    "subcell": _SET2_PALETTE[2], "subcell_rr": _SET2_PALETTE[2],
    "subcell_lite": _SET2_PALETTE[2], "subcell__clip01": _SET2_PALETTE[2],
    "subcell__clip01_lite": _SET2_PALETTE[2], "subcell__nonstd": _SET2_PALETTE[2],
    "subcell_nonstd": _SET2_PALETTE[2], "subcell_wrongchannels": _SET2_PALETTE[2],
    # 0 – ViT-rand (muted teal)
    "dinov2_random": _SET2_PALETTE[0], "dinov2_random_rr": _SET2_PALETTE[0],
    "dinov2_random_lite": _SET2_PALETTE[0],
    # 4 – MorphEm  (swapped with ViT-rand)
    "morphem": _SET2_PALETTE[4], "morphem_rr": _SET2_PALETTE[4],
    "morphem_lite": _SET2_PALETTE[4],
    # 5 – OpenPhenom
    "openphenom": _SET2_PALETTE[5], "openphenom_rr": _SET2_PALETTE[5],
    "openphenom_lite": _SET2_PALETTE[5], "openphenom_stdscale": _SET2_PALETTE[5],
    "openphenom_nonclip": _SET2_PALETTE[5], "openphenom_stdscale_false": _SET2_PALETTE[5],
    "openphenom_8clip_std": _SET2_PALETTE[5],
    # 6 – DINOv2
    "dinov2": _SET2_PALETTE[6], "dinov2_rr": _SET2_PALETTE[6],
    "dinov2_lite": _SET2_PALETTE[6], "dinov2_490": _SET2_PALETTE[6],
}


def _infer_family(model_name: str) -> str | None:
    """Infer the model family from a model name by prefix matching.

    Handles jump_lite style names like:
      morphem_jump_lite_updated_jpegxl_lossy_mq_raw_features -> morphem
      cellprofiler_raw_jump_lite_raw_features -> cellprofiler
    """
    # Ordered by specificity (longer prefixes first)
    PREFIX_TO_FAMILY = [
        # Jump-lite families (must come before originals)
        ("cellprofiler_lite", "cellprofiler_lite"),
        ("dinov2_random_lite", "dinov2_random_lite"),
        ("dinov2_lite", "dinov2_lite"),
        ("morphem_lite", "morphem_lite"),
        ("subcell__clip01_lite", "subcell__clip01_lite"),
        ("subcell_lite", "subcell_lite"),
        ("openphenom_lite", "openphenom_lite"),
        # Original families
        ("cellprofiler", "cellprofiler"),
        ("openphenom_stdscale_false", "openphenom_stdscale_false"),
        ("openphenom_stdscale", "openphenom_stdscale"),
        ("openphenom_nonclip", "openphenom_nonclip"),
        ("openphenom_8clip_std", "openphenom_8clip_std"),
        ("openphenom", "openphenom"),
        ("morphem", "morphem"),
        ("subcell_wrongchannels", "subcell_wrongchannels"),
        ("subcell__clip01", "subcell__clip01"),
        ("subcell__nonstd", "subcell__nonstd"),
        ("subcell_nonstd", "subcell_nonstd"),
        ("subcell", "subcell"),
        ("cell_count_lite", "cell_count_lite"),
        ("cell_count", "cell_count"),
        ("dinov2_490", "dinov2_490"),
        ("dinov2_random", "dinov2_random"),
        ("dinov2_cl", "dinov2"),
        ("dinov2", "dinov2"),
    ]
    name_lower = model_name.lower()
    for prefix, family in PREFIX_TO_FAMILY:
        if name_lower.startswith(prefix):
            return family
    return None


def _build_model_colors(models: list[str]) -> dict[str, tuple]:
    """Build a color map giving each model a unique color, grouped by family.

    Models within the same family share a base hue but vary in saturation/value
    so they are visually related yet distinguishable.
    """
    import matplotlib.colors as mcolors

    # Build reverse lookup: model -> (family, index_within_family)
    model_to_family: dict[str, tuple[str, int, int]] = {}
    for family, members in MODEL_FAMILIES.items():
        for idx, m in enumerate(members):
            model_to_family[m] = (family, idx, len(members))

    colors = {}
    for m in models:
        if m in model_to_family:
            family, idx, n = model_to_family[m]
            hue = FAMILY_HUES[family]
            # All codecs within a family share the same color
            colors[m] = mcolors.hsv_to_rgb([hue, 0.85, 0.9])
        else:
            # Try to infer family from model name prefix
            inferred = _infer_family(m)
            if inferred and inferred in FAMILY_HUES:
                hue = FAMILY_HUES[inferred]
                colors[m] = mcolors.hsv_to_rgb([hue, 0.85, 0.85])
            else:
                # Fallback: hash-based gray-ish color
                h = hash(m) % 360 / 360.0
                colors[m] = mcolors.hsv_to_rgb([h, 0.4, 0.7])
    return colors


def parse_model_name(folder_name: str) -> str:
    """Extract a short compression name from the model folder name.

    Examples:
      cp_measure_jump_target2_4plate_zstd_raw_features -> zstd_raw
      cp_measure_jump_target2_4plate_jpegxl_lossy_hq_filtered_raw_features -> jpegxl_lossy_hq_filtered_raw
      cp_measure_jump_target2_4plate_zstd_filtered_border_size_raw_features -> zstd_filtered_border_size_raw
      raw_jump_cp_profiles_reformatted_filtered -> raw_jump_cp_profiles_reformatted_filtered
      dinov2_490_jump_target2_4plate_zstd_raw_features -> dinov2_490_zstd_raw
      dinov2_jump_target2_4plate_jpegxl_lossy_d2_e8_raw_features -> dinov2_jpegxl_lossy_d2_e8_raw
      morphem_jump_target2_4plate_jpegxl_lossy_hq_raw_features -> morphem_jpegxl_lossy_hq_raw
      cell_count_jump_target2_4plate_zstd_raw_features -> cell_count_zstd_raw
    """
    suffix = "_features"
    # Cell Count baseline prefix
    cc_prefix = "cell_count_jump_target2_4plate_"
    if folder_name.startswith(cc_prefix):
        name = folder_name[len(cc_prefix):]
        if name.endswith(suffix):
            name = name[: -len(suffix)]
        return f"cell_count_{name}"

    # CellProfiler prefix
    cp_prefix = "cp_measure_jump_target2_4plate_"
    if folder_name.startswith(cp_prefix):
        name = folder_name[len(cp_prefix):]
        if name.endswith(suffix):
            name = name[: -len(suffix)]
        return name

    # Embedding model prefixes: {model}_jump_target2_4plate_{codec}_raw_features
    # Map from folder prefix -> short model name used in parsed output.
    # Order matters: longer/more-specific prefixes must come before shorter ones
    # (e.g. dinov2_490_ and dinov2_random_ before bare dinov2_).
    embedding_prefixes = [
        ("dinov2_490_jump_target2_4plate_", "dinov2_490"),
        ("dinov2_random_jump_target2_4plate_", "dinov2_random"),
        ("dinov2_jump_target2_4plate_", "dinov2"),  # DINOv2 (must be after dinov2_490_ and dinov2_random_)
        ("morphem_jump_target2_4plate_", "morphem"),
        ("subcell_wrongchannels_jump_target2_4plate_", "subcell_wrongchannels"),  # must be before subcell_nonstd and bare subcell_
        ("subcell__clip01_jump_target2_4plate_", "subcell__clip01"),  # must be before bare subcell_
        ("subcell__nonstd_jump_target2_4plate_", "subcell__nonstd"),  # must be before bare subcell_
        ("subcell_nonstd_jump_target2_4plate_", "subcell_nonstd"),  # must be before bare subcell_
        ("subcell_jump_target2_4plate_", "subcell"),
        ("openphenom_stdscale_false_jump_target2_4plate_", "openphenom_stdscale_false"),  # must be before openphenom_stdscale_
        ("openphenom_stdscale_jump_target2_4plate_", "openphenom_stdscale"),  # must be before bare openphenom_
        ("openphenom_nonclip_jump_target2_4plate_", "openphenom_nonclip"),  # must be before bare openphenom_
        ("openphenom_8clip_std_jump_target2_4plate_", "openphenom_8clip_std"),  # must be before bare openphenom_
        ("openphenom_jump_target2_4plate_", "openphenom"),
    ]
    for emb_prefix, model_name in embedding_prefixes:
        if folder_name.startswith(emb_prefix):
            codec_part = folder_name[len(emb_prefix):]
            if codec_part.endswith(suffix):
                codec_part = codec_part[: -len(suffix)]
            return f"{model_name}_{codec_part}"

    # Jump-lite prefixes: {model}_jump_lite_updated_{codec}_raw_features
    # Also: cellprofiler_raw_jump_lite_raw_features (special case)
    if folder_name.startswith("cellprofiler_raw_jump_lite_"):
        return "cellprofiler_lite_raw"
    if folder_name.startswith("cell_count_jump_lite_"):
        return "cell_count_lite_raw"

    lite_prefixes = [
        ("dinov2_random_jump_lite_updated_", "dinov2_random_lite"),
        ("dinov2_jump_lite_updated_", "dinov2_lite"),
        ("morphem_jump_lite_updated_", "morphem_lite"),
        ("subcell__clip01_jump_lite_updated_", "subcell__clip01_lite"),
        ("subcell_jump_lite_updated_", "subcell_lite"),
        ("openphenom_jump_lite_updated_", "openphenom_lite"),
    ]
    for lite_prefix, model_name in lite_prefixes:
        if folder_name.startswith(lite_prefix):
            codec_part = folder_name[len(lite_prefix):]
            if codec_part.endswith(suffix):
                codec_part = codec_part[: -len(suffix)]
            return f"{model_name}_{codec_part}"

    return folder_name


def get_display_name(model: str) -> str:
    """Get short display name for a model/codec."""
    return COMPRESSION_DISPLAY.get(model, model)


def sort_models(models: list[str]) -> list[str]:
    """Sort models by (family order, codec order within family).

    Family order follows MODEL_FAMILIES dict insertion order.
    Codec order within each family follows _CODEC_SORT_ORDER (lossless → heavy lossy).
    """
    # Build reverse lookup: model -> (family_index, codec_rank)
    _family_index = {fam: i for i, fam in enumerate(MODEL_FAMILIES)}
    _model_to_key: dict[str, tuple[int, int]] = {}
    for family, members in MODEL_FAMILIES.items():
        fi = _family_index[family]
        for m in members:
            _model_to_key[m] = (fi, _get_codec_sort_rank(m))

    n_families = len(MODEL_FAMILIES)

    def _sort_key(m):
        if m in _model_to_key:
            return _model_to_key[m]
        # Try to infer family for unknown models (e.g. jump_lite names)
        inferred = _infer_family(m)
        if inferred and inferred in _family_index:
            return (_family_index[inferred], _get_codec_sort_rank(m))
        return (n_families, _get_codec_sort_rank(m))

    return sorted(models, key=_sort_key)


def parse_config_name(config_name: str) -> dict:
    """Parse a norm_3 config folder name into individual settings.

    Config format: {norm}_ctrl__outlier{cutoff}__INT__prune{thresh}[__pca{n}][__batch]

    Examples:
      robustmad_ctrl__outlier100__INT__prune0.9
      std_ctrl__outlier100__INT__prune0.9__pca64__tvn_efaar_e0.5
      robustmad_ctrl__outlier100__INT__prune0.9__ZCA-cor_all_e0.5
      std_ctrl__outlier100__INT__prune0.9__pca64__cascade_tvn_k128_k32
    """
    parts = config_name.split("__")

    settings = {
        "norm_method": "unknown",
        "outlier_cutoff": None,
        "use_int": False,
        "prune_thresh": None,
        "use_pca": False,
        "pca_components": None,
        "batch_method": "none",
        "spherize_fit": None,
        "spherize_epsilon": None,
        "tvn_epsilon": None,
        "tvn_original_k": None,
        "tvn_efaar_n_components": None,
        "tvn_cascade_k1": None,
        "tvn_cascade_k2": None,
    }

    for part in parts:
        # Normalization method
        if part.startswith("robustmad"):
            settings["norm_method"] = "robustmad"
        elif part.startswith("std"):
            settings["norm_method"] = "standardize"

        # Outlier cutoff
        elif part.startswith("outlier"):
            try:
                settings["outlier_cutoff"] = int(part.replace("outlier", ""))
            except ValueError:
                pass

        # Inverse Normal Transform
        elif part == "INT":
            settings["use_int"] = True

        # Correlation pruning
        elif part.startswith("prune"):
            try:
                settings["prune_thresh"] = float(part.replace("prune", ""))
            except ValueError:
                pass

        # PCA
        elif part.startswith("pca"):
            settings["use_pca"] = True
            try:
                settings["pca_components"] = int(part.replace("pca", ""))
            except ValueError:
                pass

        # Batch correction methods
        elif part.startswith("tvn_original"):
            settings["batch_method"] = "tvn_original"
            # Extract k: tvn_original_k64 -> 64
            k_match = re.search(r"_k(\d+)$", part)
            if k_match:
                settings["tvn_original_k"] = int(k_match.group(1))
        elif part.startswith("tvn_efaar"):
            settings["batch_method"] = "tvn_efaar"
            # Extract epsilon and optional n_components:
            #   tvn_efaar_e0.5       -> epsilon=0.5, n_components=128 (default)
            #   tvn_efaar_e0.5_c256  -> epsilon=0.5, n_components=256
            efaar_match = re.search(r"_e([\d.]+)(?:_c(\d+))?$", part)
            if efaar_match:
                try:
                    settings["tvn_epsilon"] = float(efaar_match.group(1))
                except ValueError:
                    pass
                if efaar_match.group(2):
                    settings["tvn_efaar_n_components"] = int(efaar_match.group(2))
                else:
                    settings["tvn_efaar_n_components"] = 128
        elif part.startswith("cascade_tvn"):
            settings["batch_method"] = "cascade_tvn"
            # Extract k1, k2: cascade_tvn_k128_k32 -> k1=128, k2=32
            k_match = re.search(r"_k(\d+)_k(\d+)$", part)
            if k_match:
                settings["tvn_cascade_k1"] = int(k_match.group(1))
                settings["tvn_cascade_k2"] = int(k_match.group(2))
        elif part.startswith("ZCA-cor_global_"):
            settings["batch_method"] = "spherize_global"
            # Parse: ZCA-cor_global_ctrl_e0.1 or ZCA-cor_global_ctrl_e4 (=1e-4)
            remainder = part[len("ZCA-cor_global_"):]
            if "_e" in remainder:
                fit_part, eps_part = remainder.rsplit("_e", 1)
                settings["spherize_fit"] = fit_part  # "ctrl" or "all"
                try:
                    # e6 means 1e-6, e0.5 means 0.5, e100.0 means 100.0
                    eps_val = float(eps_part)
                    if "." not in eps_part and eps_val >= 2:
                        settings["spherize_epsilon"] = 10 ** (-eps_val)
                    else:
                        settings["spherize_epsilon"] = eps_val
                except ValueError:
                    pass
        elif part.startswith("ZCA-cor_"):
            settings["batch_method"] = "spherize"
            # Parse: ZCA-cor_all_e0.5 or ZCA-cor_ctrl_e6
            remainder = part[len("ZCA-cor_"):]
            if "_e" in remainder:
                fit_part, eps_part = remainder.rsplit("_e", 1)
                settings["spherize_fit"] = fit_part  # "all" or "ctrl"
                try:
                    # e6 means 1e-6, e0.5 means 0.5, e100.0 means 100.0
                    eps_val = float(eps_part)
                    if "." not in eps_part and eps_val >= 2:
                        settings["spherize_epsilon"] = 10 ** (-eps_val)
                    else:
                        settings["spherize_epsilon"] = eps_val
                except ValueError:
                    pass

    return settings


def load_metrics(json_path: Path) -> dict:
    """Load metrics from a metrics.json file and combine with path metadata."""
    with open(json_path) as f:
        data = json.load(f)

    # Path: .../sweep_dir/model_folder/config_folder/results/metrics.json
    config_name = json_path.parent.parent.name
    model_folder = json_path.parent.parent.parent.name
    model_name = parse_model_name(model_folder)

    metrics = {
        "model": model_name,
        "config": config_name,
        "PA": data.get("PA"),
        "PC": data.get("PC"),
        "PA_mean_nap": data.get("PA_mean_nap"),
        "PA_median_nap": data.get("PA_median_nap"),
        "PC_mean_nap": data.get("PC_mean_nap"),
        "PC_median_nap": data.get("PC_median_nap"),
        "n_compounds": data.get("n_compounds"),
        "n_targets_active": data.get("n_targets_active"),
        "n_targets_total": data.get("n_targets_total"),
        "n_features": data.get("n_features"),
        "tvn_ill_conditioned": data.get("tvn_ill_conditioned"),
        "tvn_condition_number": data.get("tvn_max_condition_number"),
        "PC1_variance": data.get("PC1_variance"),
        "PC2_variance": data.get("PC2_variance"),
        "PC_replicable": data.get("PC_replicable"),
        "PC_replicable_n_targets_active": data.get("PC_replicable_n_targets_active"),
        "PC_replicable_n_targets_total": data.get("PC_replicable_n_targets_total"),
        "PC_replicable_mean_nap": data.get("PC_replicable_mean_nap"),
        "PC_replicable_median_nap": data.get("PC_replicable_median_nap"),
        "PC_replicable_n_compounds": data.get("PC_replicable_n_compounds"),
        # Batch effects
        "well_effect_pct": data.get("well_effect_pct"),
        "well_effect_mean_nap": data.get("well_effect_mean_nap"),
        "well_effect_n_active": data.get("well_effect_n_active"),
        "well_effect_n_total": data.get("well_effect_n_total"),
        "plate_effect_pct": data.get("plate_effect_pct"),
        "plate_effect_mean_nap": data.get("plate_effect_mean_nap"),
        "plate_effect_n_active": data.get("plate_effect_n_active"),
        "plate_effect_n_total": data.get("plate_effect_n_total"),
    }

    # Flatten per-group summaries into top-level columns
    # e.g. PA_group_summary.group_orf.pct_active -> PA_group_orf_pct_active
    for summary_key, prefix in [
        ("PA_group_summary", "PA"),
        ("PC_group_summary", "PC"),
        ("PC_replicable_group_summary", "PC_rep"),
        ("well_effect_group_summary", "well_effect"),
        ("plate_effect_group_summary", "plate_effect"),
    ]:
        group_data = data.get(summary_key)
        if isinstance(group_data, dict):
            for group_name, group_stats in group_data.items():
                if isinstance(group_stats, dict):
                    for stat_name, stat_value in group_stats.items():
                        col_name = f"{prefix}_{group_name}_{stat_name}"
                        metrics[col_name] = stat_value

    # Parse config settings
    settings = parse_config_name(config_name)
    metrics.update(settings)

    return metrics


def filter_degenerate(df: pl.DataFrame) -> pl.DataFrame:
    """Filter out degenerate configs: spherize without PCA.

    Spherize (ZCA whitening) on high-dimensional features without prior PCA
    produces isotropic noise where PC1_variance ≈ 1/n_features, artificially
    inflating PA while PC stays low.
    """
    before = len(df)
    df = df.filter(
        ~((pl.col("batch_method") == "spherize") & (pl.col("use_pca") == False))
    )
    after = len(df)
    if before > after:
        print(f"  Filtered {before - after} degenerate configs (spherize + no PCA)")
    return df


def _add_best_column(pdf, best_metric="balanced"):
    """Add the selection column used to pick the best config per model.

    Args:
        pdf: pandas DataFrame (must already have PA, PC columns).
        best_metric: 'balanced' for PA%*PC%/100, 'nap_balanced' for PA_mean_nap*PC_mean_nap,
                     'pc' for PC%, 'pc_nap' for PC_mean_nap.

    Returns:
        (pdf, column_name) where column_name is the column to call idxmax() on.
    """
    if best_metric == "nap_balanced":
        if "PA_mean_nap" in pdf.columns and "PC_mean_nap" in pdf.columns:
            pa_max = pdf["PA_mean_nap"].max()
            pc_max = pdf["PC_mean_nap"].max()
            pa_scaled = pdf["PA_mean_nap"] / pa_max if pa_max > 0 else pdf["PA_mean_nap"]
            pc_scaled = pdf["PC_mean_nap"] / pc_max if pc_max > 0 else pdf["PC_mean_nap"]
            pdf["_best_score"] = pa_scaled * pc_scaled
        else:
            print("Warning: NAP columns not found, falling back to balanced score")
            pdf["_best_score"] = pdf["PA"] * pdf["PC"] / 100
    elif best_metric == "pc":
        pdf["_best_score"] = pdf["PC"]
    elif best_metric == "pc_nap":
        if "PC_mean_nap" in pdf.columns:
            pdf["_best_score"] = pdf["PC_mean_nap"]
        else:
            print("Warning: PC_mean_nap column not found, falling back to PC%")
            pdf["_best_score"] = pdf["PC"]
    else:
        pdf["_best_score"] = pdf["PA"] * pdf["PC"] / 100
    return pdf, "_best_score"


def _find_zstd_best_config_per_family(pdf, best_metric="balanced", best_col=None):
    """For each model family, find the best normalization config of the zstd codec.

    Returns:
        dict[str, str]: family -> config name
    """
    if best_col is None or best_col not in pdf.columns:
        pdf, best_col = _add_best_column(pdf.copy(), best_metric)

    zstd_best = {}
    for model in pdf["model"].unique():
        if _get_codec_sort_rank(model) != 1:  # rank 1 = zstd
            continue
        family = _get_model_family(model)
        if family == "unknown":
            family = _infer_family(model)
        if family is None:
            continue

        mdf = pdf[pdf["model"] == model]
        if len(mdf) > 0 and not mdf[best_col].isna().all():
            bi = mdf[best_col].idxmax()
            zstd_best[family] = pdf.loc[bi, "config"]

    return zstd_best


def _find_best_config_any_codec_per_family(pdf, best_metric="balanced", best_col=None):
    """For each model family, find the best config across ALL codecs.

    Pools all codecs within a family and picks the single config with the
    highest score regardless of which codec produced it.

    Returns:
        dict[str, str]: family -> config name
    """
    if best_col is None or best_col not in pdf.columns:
        pdf, best_col = _add_best_column(pdf.copy(), best_metric)

    # Group models by family
    family_models: dict[str, list[str]] = {}
    for model in pdf["model"].unique():
        family = _get_model_family(model)
        if family == "unknown":
            family = _infer_family(model)
        if family is None:
            continue
        family_models.setdefault(family, []).append(model)

    best_configs = {}
    for family, fam_models in family_models.items():
        fam_df = pdf[pdf["model"].isin(fam_models)]
        if len(fam_df) > 0 and not fam_df[best_col].isna().all():
            bi = fam_df[best_col].idxmax()
            best_configs[family] = pdf.loc[bi, "config"]

    return best_configs


def _find_best_avg_config_per_family(pdf, best_metric="balanced", best_col=None):
    """For each model family, find the config with the highest average score across codecs.

    For each config, computes its mean score across all codecs in the family,
    then picks the config with the highest mean.

    Returns:
        dict[str, str]: family -> config name
    """
    if best_col is None or best_col not in pdf.columns:
        pdf, best_col = _add_best_column(pdf.copy(), best_metric)

    # Group models by family
    family_models: dict[str, list[str]] = {}
    for model in pdf["model"].unique():
        family = _get_model_family(model)
        if family == "unknown":
            family = _infer_family(model)
        if family is None:
            continue
        family_models.setdefault(family, []).append(model)

    best_configs = {}
    for family, fam_models in family_models.items():
        fam_df = pdf[pdf["model"].isin(fam_models)]
        if len(fam_df) == 0 or fam_df[best_col].isna().all():
            continue
        # Average score per config across all codecs in the family
        avg_by_config = fam_df.groupby("config")[best_col].mean()
        best_configs[family] = avg_by_config.idxmax()

    return best_configs


def _get_family_configs(pdf, best_selection, best_metric, best_col=None):
    """Get the family -> config mapping for the given selection mode.

    Returns None for 'per_codec' (no pinned config).
    """
    if best_selection == "per_codec":
        return None
    elif best_selection == "zstd_reference":
        return _find_zstd_best_config_per_family(pdf, best_metric, best_col=best_col)
    elif best_selection == "best_any_codec":
        return _find_best_config_any_codec_per_family(pdf, best_metric, best_col=best_col)
    elif best_selection == "best_avg_codec":
        return _find_best_avg_config_per_family(pdf, best_metric, best_col=best_col)
    return None


def _compute_best_idx(pdf, models, best_col, best_selection="per_codec",
                      best_metric="balanced"):
    """Compute best config index for each model.

    Args:
        pdf: pandas DataFrame with _best_score column already added.
        models: List of model names.
        best_col: Column name to maximize (from _add_best_column).
        best_selection: 'per_codec' (each codec picks own best),
                        'zstd_reference' (use zstd's best config for all codecs
                        in the same model family), 'best_any_codec' (use the
                        best config from any codec in the family), or
                        'best_avg_codec' (use the config with the highest
                        average score across all codecs in the family).
        best_metric: Metric name (passed to family config finders).

    Returns:
        dict[str, int]: model -> pandas index of best row
    """
    if best_selection == "per_codec":
        best_idx = {}
        for model in models:
            mdf = pdf[pdf["model"] == model]
            if len(mdf) > 0 and not mdf[best_col].isna().all():
                best_idx[model] = mdf[best_col].idxmax()
        return best_idx

    # Resolve family -> config mapping
    if best_selection == "zstd_reference":
        family_configs = _find_zstd_best_config_per_family(pdf, best_metric)
    elif best_selection == "best_any_codec":
        family_configs = _find_best_config_any_codec_per_family(pdf, best_metric)
    elif best_selection == "best_avg_codec":
        family_configs = _find_best_avg_config_per_family(pdf, best_metric)
    else:
        raise ValueError(f"Unknown best_selection: {best_selection}")

    # Pin each model to its family's chosen config
    zstd_configs = family_configs

    best_idx = {}
    for model in models:
        family = _get_model_family(model)
        if family == "unknown":
            family = _infer_family(model) or "unknown"

        mdf = pdf[pdf["model"] == model]
        if len(mdf) == 0:
            continue

        if family in zstd_configs:
            config = zstd_configs[family]
            pinned = mdf[mdf["config"] == config]
            if len(pinned) > 0:
                best_idx[model] = pinned.index[0]
                continue

        # Fallback: use per-codec best
        if not mdf[best_col].isna().all():
            best_idx[model] = mdf[best_col].idxmax()

    return best_idx


def generate_all_metrics_plot(pdf, output_dir: Path, model_colors: dict,
                              models: list, best_idx: dict, best_col: str,
                              best_metric: str = "balanced",
                              best_selection: str = "per_codec"):
    """Generate a comprehensive grid plot with all key metrics for every model.

    Shows PA, PC, balanced score, PC1_variance, and n_features as strip plots
    with each model on the x-axis and its own color.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)

    n_models = len(models)
    display_names = [get_display_name(m) for m in models]

    all_metrics = [
        ("PA", "PA (%)", "PA"),
        ("PC", "PC (%)", "PC"),
        ("PC Replicable", "PC_rep (%)", "PC_replicable"),
        ("Balanced Score", "PA * PC / 100", "balanced_score"),
        ("Balanced NAP", "PA_nap * PC_nap", "nap_balanced"),
        ("PA Mean NAP", "NAP", "PA_mean_nap"),
        ("PA Median NAP", "NAP", "PA_median_nap"),
        ("PC Mean NAP", "NAP", "PC_mean_nap"),
        ("PC Median NAP", "NAP", "PC_median_nap"),
        ("PC Rep Mean NAP", "NAP", "PC_replicable_mean_nap"),
        ("PC Rep Median NAP", "NAP", "PC_replicable_median_nap"),
        ("n Compounds", "Count", "n_compounds"),
        ("n Targets Active", "Count", "n_targets_active"),
        ("n Targets Total", "Count", "n_targets_total"),
        ("PC Rep n Targets Active", "Count", "PC_replicable_n_targets_active"),
        ("PC Rep n Targets Total", "Count", "PC_replicable_n_targets_total"),
        ("PC Rep n Compounds", "Count", "PC_replicable_n_compounds"),
        ("n Features", "Count", "n_features"),
        ("PC1 Variance", "Variance", "PC1_variance"),
        ("PC2 Variance", "Variance", "PC2_variance"),
    ]
    available = [(t, y, c) for t, y, c in all_metrics if c in pdf.columns and not pdf[c].isna().all()]

    n_metrics = len(available)
    n_cols = min(4, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    col_width = max(8, n_models * 0.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(col_width * n_cols, 6 * n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (title, ylabel, col) in enumerate(available):
        ax = axes[i]
        for j, model in enumerate(models):
            mdf = pdf[pdf["model"] == model]
            bi = best_idx.get(model)
            other = mdf[mdf.index != bi] if bi is not None else mdf
            vals = other[col].dropna()
            if len(vals) > 0:
                x_jitter = np.random.normal(j, 0.12, len(vals))
                ax.scatter(x_jitter, vals, c=[model_colors[model]], s=30, alpha=0.5,
                           edgecolors="white", linewidths=0.2)
            if bi is not None:
                bv = pdf.loc[bi, col]
                if not np.isnan(bv):
                    ax.scatter(j, bv, c=[model_colors[model]], s=350, alpha=1.0,
                               edgecolors=[model_colors[model]], linewidths=1.5, marker="*", zorder=10)

        ax.set_xticks(range(n_models))
        ax.set_xticklabels(display_names, rotation=60, ha="right", fontsize=6)
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    metric_label = {"balanced": "PA*PC", "nap_balanced": "NAP balanced", "pc": "PC %", "pc_nap": "PC mean NAP"}.get(best_metric, best_metric)
    title_suffix = {"zstd_reference": " (zstd-pinned)", "best_any_codec": " (best-any-codec)", "best_avg_codec": " (best-avg-codec)"}.get(best_selection, "")
    fig.suptitle(f"All Metrics (* = best by {metric_label} per model){title_suffix}", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec", "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")
    fname = f"sweep_all_metrics{_sel_suffix}.png"
    plt.savefig(output_dir / fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / fname}")


def generate_group_nap_plot(pdf, output_dir: Path, model_colors: dict,
                            models: list, best_idx: dict, best_col: str,
                            best_metric: str = "balanced",
                            best_selection: str = "per_codec"):
    """Per-group mean NAP strip plot: PA NAP for 4 groups + PC NAP for 2 groups.

    Layout: 2 rows x 3 cols.
    Row 0: PA CRISPR, PA ORF, PA Compounds (high)
    Row 1: PA Compounds (low), PC high, PC low
    Style matches generate_all_metrics_plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)

    group_metrics = [
        ("PA CRISPR NAP", "NAP", "PA_group_crispr_mean_normalized_average_precision"),
        ("PA ORF NAP", "NAP", "PA_group_orf_mean_normalized_average_precision"),
        ("PA Compounds (high) NAP", "NAP", "PA_group_high_mean_normalized_average_precision"),
        ("PA Compounds (low) NAP", "NAP", "PA_group_low_mean_normalized_average_precision"),
        ("PC high NAP", "NAP", "PC_group_high_mean_normalized_average_precision"),
        ("PC low NAP", "NAP", "PC_group_low_mean_normalized_average_precision"),
        ("Well Position Effect NAP", "NAP", "well_effect_mean_nap"),
        ("Batch Effect NAP", "NAP", "plate_effect_mean_nap"),
    ]
    available = [(t, y, c) for t, y, c in group_metrics if c in pdf.columns and not pdf[c].isna().all()]

    if not available:
        print("Per-group NAP columns not available, skipping group NAP plot.")
        return

    # Group models by family, each family gets exactly 3 slots
    # Preferred display order for families
    # Alphabetical by display name: CellCount, CellProfiler, DINOv2, MorphEm, OpenPhenom, SubCell, ViT-rand
    _FAMILY_PLOT_ORDER = [
        "cell_count", "cell_count_lite",
        "cellprofiler", "cellprofiler_lite", "cp_measure", "cp_measure_filtered", "cp_measure_fbs",
        "dinov2", "dinov2_rr", "dinov2_lite", "dinov2_490",
        "morphem", "morphem_rr", "morphem_lite",
        "openphenom", "openphenom_rr", "openphenom_lite",
        "openphenom_stdscale", "openphenom_nonclip", "openphenom_stdscale_false",
        "openphenom_8clip_std",
        "subcell", "subcell_rr", "subcell_lite", "subcell__clip01", "subcell__clip01_lite",
        "subcell__nonstd", "subcell_nonstd", "subcell_wrongchannels",
        "dinov2_random", "dinov2_random_rr", "dinov2_random_lite",
    ]
    _fam_rank = {f: i for i, f in enumerate(_FAMILY_PLOT_ORDER)}

    MIN_SLOTS_PER_FAMILY = 3
    GAP_BETWEEN_FAMILIES = 1.5  # extra space between family groups
    family_models: dict[str, list[str]] = {}
    for m in models:
        fam = _get_model_family(m)
        if fam not in family_models:
            family_models[fam] = []
        family_models[fam].append(m)

    family_order = sorted(family_models.keys(),
                          key=lambda f: _fam_rank.get(f, len(_FAMILY_PLOT_ORDER)))
    n_families = len(family_order)

    # Build slot positions with gaps between families
    model_xpos: dict[str, float] = {}
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    family_groups: list[tuple[str, float, float]] = []  # (display, x_start, x_end)
    cursor = 0.0
    for fi, fam in enumerate(family_order):
        fam_models = family_models[fam]
        n_in_fam = len(fam_models)
        slots = max(MIN_SLOTS_PER_FAMILY, n_in_fam)
        slot_start = cursor
        slot_end = cursor + slots - 1
        # Center the codecs within the allocated slots
        offset = (slots - n_in_fam) / 2.0
        for mi, m in enumerate(fam_models):
            xpos = cursor + offset + mi
            model_xpos[m] = xpos
            display = get_display_name(m)
            tick_positions.append(xpos)
            tick_labels.append(_get_codec_label(display))
        family_groups.append((FAMILY_DISPLAY.get(fam, fam), slot_start, slot_end))
        cursor += slots + GAP_BETWEEN_FAMILIES
    total_width = cursor - GAP_BETWEEN_FAMILIES

    # Per-family colors from global Set2 mapping
    _model_colors_local = {}
    for fam, mlist in family_models.items():
        c = FAMILY_SET2_COLOR.get(fam, (0.5, 0.5, 0.5))
        for m in mlist:
            _model_colors_local[m] = c

    # Scale visual parameters based on max family width
    max_fam_size = max(len(v) for v in family_models.values()) if family_models else 3
    # density_scale: 1.0 at 3 members, shrinks for larger families
    density_scale = min(1.0, 3.0 / max(max_fam_size, 1))

    n_cols = 4
    n_rows = 2

    # Font sizes matching generate_nap_pa_vs_pc_combined for paper readability
    fs_title = 28
    fs_subtitle = 20
    fs_axis = 18
    fs_tick = 18
    fs_xtick = max(10, int(13 * density_scale))
    fs_family = max(11, int(14 * density_scale))

    # Scatter parameters scaled by density
    jitter_sigma = 0.12 * density_scale
    scatter_size = max(30, int(50 * density_scale))
    star_size = max(300, int(500 * density_scale))
    scatter_lw = 0.3 * density_scale
    star_lw = max(0.8, 1.2 * density_scale)

    col_width = max(8, total_width * 0.45)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(col_width * n_cols, 11 * n_rows))
    axes = axes.flatten()

    _panel_labels = "abcdefghijklmnop"
    for i, (title, ylabel, col) in enumerate(available):
        ax = axes[i]
        for model in models:
            xpos = model_xpos[model]
            mdf = pdf[pdf["model"] == model]
            bi = best_idx.get(model)
            other = mdf[mdf.index != bi] if bi is not None else mdf
            vals = other[col].dropna()
            if len(vals) > 0:
                x_jitter = np.random.normal(xpos, jitter_sigma, len(vals))
                ax.scatter(x_jitter, vals, c=[_model_colors_local[model]], s=scatter_size, alpha=0.45,
                           edgecolors="white", linewidths=scatter_lw)
            if bi is not None:
                bv = pdf.loc[bi, col]
                if not np.isnan(bv):
                    ax.scatter(xpos, bv, c=[_model_colors_local[model]], s=star_size, alpha=1.0,
                               edgecolors="black", linewidths=star_lw, marker="*", zorder=10)

        # Top tier: codec labels (only where data exists)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=fs_xtick)
        ax.set_xlim(-0.5, total_width + 0.5)
        ax.set_ylabel(ylabel, fontsize=fs_axis, fontweight="bold")
        ax.set_title(title, fontsize=fs_subtitle, fontweight="bold")
        ax.tick_params(axis="y", labelsize=fs_tick)
        ax.tick_params(axis="x", length=4, pad=4)
        ax.grid(True, alpha=0.2, axis="y", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Panel label (a, b, c, ...)
        ax.text(-0.02, 1.05, _panel_labels[i], transform=ax.transAxes,
                fontsize=fs_subtitle + 8, fontweight="bold", va="bottom", ha="right")

        # Bottom tier: family labels (one per group, centered, with bracket line)
        trans = ax.get_xaxis_transform()
        for fam_disp, slot_start, slot_end in family_groups:
            mid = (slot_start + slot_end) / 2.0
            ax.text(mid, -0.22, fam_disp, transform=trans,
                    ha="center", va="top", fontsize=fs_family, fontweight="bold",
                    rotation=45, rotation_mode="anchor")
            bracket_pad = 0.3 * density_scale
            ax.plot([slot_start - bracket_pad, slot_end + bracket_pad], [-0.14, -0.14], transform=trans,
                    color="gray", linewidth=max(0.6, 0.8 * density_scale), clip_on=False)

    for i in range(len(available), len(axes)):
        axes[i].set_visible(False)

    metric_label = {"balanced": "PA*PC", "nap_balanced": "NAP balanced", "pc": "PC %", "pc_nap": "PC mean NAP"}.get(best_metric, best_metric)
    title_suffix = {"zstd_reference": " (zstd-pinned)", "best_any_codec": " (best-any-codec)", "best_avg_codec": " (best-avg-codec)"}.get(best_selection, "")
    fig.suptitle(
        f"Per-Group Mean NAP\n(\u2605 = best by {metric_label} per model){title_suffix}",
        fontsize=fs_title, fontweight="bold", y=1.02,
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec", "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")
    fname = f"sweep_group_nap{_sel_suffix}.png"
    plt.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / fname}")


def generate_group_nap_plot_compact(pdf, output_dir: Path, model_colors: dict,
                                    models: list, best_idx: dict, best_col: str,
                                    best_metric: str = "balanced",
                                    best_selection: str = "per_codec"):
    """Compact per-group mean NAP strip plot: PA NAP for 4 groups + PC NAP for 2 groups.

    Layout: 2 rows x 3 cols (no well/batch effect plots).
    Same styling as generate_group_nap_plot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)

    group_metrics = [
        ("PA - CRISPR", "NAP", "PA_group_crispr_mean_normalized_average_precision"),
        ("PA - ORF", "NAP", "PA_group_orf_mean_normalized_average_precision"),
        ("PA - Compound Diversity", "NAP", "PA_group_high_mean_normalized_average_precision"),
        ("PA - Compound Bioactive-library", "NAP", "PA_group_low_mean_normalized_average_precision"),
        ("PC - Compound Diversity", "NAP", "PC_group_high_mean_normalized_average_precision"),
        ("PC - Compound Bioactive-library", "NAP", "PC_group_low_mean_normalized_average_precision"),
    ]
    available = [(t, y, c) for t, y, c in group_metrics if c in pdf.columns and not pdf[c].isna().all()]

    if not available:
        print("Per-group NAP columns not available, skipping compact group NAP plot.")
        return

    # Alphabetical family ordering
    _FAMILY_PLOT_ORDER = [
        "cell_count", "cell_count_lite",
        "cellprofiler", "cellprofiler_lite", "cp_measure", "cp_measure_filtered", "cp_measure_fbs",
        "dinov2", "dinov2_rr", "dinov2_lite", "dinov2_490",
        "morphem", "morphem_rr", "morphem_lite",
        "openphenom", "openphenom_rr", "openphenom_lite",
        "openphenom_stdscale", "openphenom_nonclip", "openphenom_stdscale_false",
        "openphenom_8clip_std",
        "subcell", "subcell_rr", "subcell_lite", "subcell__clip01", "subcell__clip01_lite",
        "subcell__nonstd", "subcell_nonstd", "subcell_wrongchannels",
        "dinov2_random", "dinov2_random_rr", "dinov2_random_lite",
    ]
    _fam_rank = {f: i for i, f in enumerate(_FAMILY_PLOT_ORDER)}

    GAP_BETWEEN_FAMILIES = 0.8
    family_models: dict[str, list[str]] = {}
    for m in models:
        fam = _get_model_family(m)
        if fam not in family_models:
            family_models[fam] = []
        family_models[fam].append(m)

    family_order = sorted(family_models.keys(),
                          key=lambda f: _fam_rank.get(f, len(_FAMILY_PLOT_ORDER)))
    n_families = len(family_order)

    model_xpos: dict[str, float] = {}
    tick_positions: list[float] = []
    tick_labels: list[str] = []
    family_groups: list[tuple[str, float, float]] = []
    cursor = 0.0
    for fi, fam in enumerate(family_order):
        fam_models = family_models[fam]
        n_in_fam = len(fam_models)
        # Allocate slots matching the actual family size (min 1)
        slot_width = max(n_in_fam, 1)
        slot_start = cursor
        slot_end = cursor + slot_width - 1
        offset = (slot_width - n_in_fam) / 2.0
        for mi, m in enumerate(fam_models):
            xpos = cursor + offset + mi
            model_xpos[m] = xpos
            display = get_display_name(m)
            tick_positions.append(xpos)
            tick_labels.append(_get_codec_label(display))
        family_groups.append((FAMILY_DISPLAY.get(fam, fam), slot_start, slot_end))
        cursor += slot_width + GAP_BETWEEN_FAMILIES
    total_width = cursor - GAP_BETWEEN_FAMILIES

    _model_colors_local = {}
    for fam, mlist in family_models.items():
        c = FAMILY_SET2_COLOR.get(fam, (0.5, 0.5, 0.5))
        for m in mlist:
            _model_colors_local[m] = c

    # Scale visual parameters based on max family width
    max_fam_size = max(len(v) for v in family_models.values()) if family_models else 3
    # density_scale: 1.0 at 3 members, shrinks for larger families
    density_scale = min(1.0, 3.0 / max(max_fam_size, 1))

    n_cols = 3
    n_rows = 2

    # Font sizes for A4 paper readability (scaled for density)
    fs_title = 46
    fs_subtitle = 38
    fs_axis = 36
    fs_tick = 32
    fs_xtick = max(22, int(28 * density_scale))
    fs_family = max(24, int(30 * density_scale))
    fs_panel = 50

    # Scatter parameters scaled by density
    jitter_sigma = 0.12 * density_scale
    scatter_size = max(90, int(150 * density_scale))
    star_size = max(500, int(800 * density_scale))
    scatter_lw = max(0.3, 0.4 * density_scale)
    star_lw = max(1.4, 2.0 * density_scale)

    col_width = max(12, total_width * 0.65)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(col_width * n_cols, 14 * n_rows))
    axes = axes.flatten()

    _panel_labels = "abcdefghijklmnop"
    for i, (title, ylabel, col) in enumerate(available):
        ax = axes[i]
        # Determine y-axis label based on metric type
        if col.startswith("PA_"):
            yaxis_label = "PA Mean NAP"
        elif col.startswith("PC_"):
            yaxis_label = "PC Mean NAP"
        else:
            yaxis_label = ylabel

        for model in models:
            xpos = model_xpos[model]
            mdf = pdf[pdf["model"] == model]
            bi = best_idx.get(model)
            other = mdf[mdf.index != bi] if bi is not None else mdf
            vals = other[col].dropna()
            if len(vals) > 0:
                x_jitter = np.random.normal(xpos, jitter_sigma, len(vals))
                ax.scatter(x_jitter, vals, c=[_model_colors_local[model]], s=scatter_size, alpha=0.5,
                           edgecolors="white", linewidths=scatter_lw)
            if bi is not None:
                bv = pdf.loc[bi, col]
                if not np.isnan(bv):
                    ax.scatter(xpos, bv, c=[_model_colors_local[model]], s=star_size, alpha=1.0,
                               edgecolors="black", linewidths=star_lw, marker="*", zorder=10)

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, ha="center", fontsize=fs_xtick)
        ax.set_xlim(-0.5, total_width + 0.5)
        ax.set_ylabel(yaxis_label, fontsize=fs_axis, fontweight="bold")
        ax.set_title(title, fontsize=fs_subtitle, fontweight="bold")
        ax.tick_params(axis="y", labelsize=fs_tick)
        ax.tick_params(axis="x", length=6, pad=6)
        ax.grid(True, alpha=0.2, axis="y", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.text(-0.02, 1.05, _panel_labels[i], transform=ax.transAxes,
                fontsize=fs_panel, fontweight="bold", va="bottom", ha="right")

        trans = ax.get_xaxis_transform()
        for fam_disp, slot_start, slot_end in family_groups:
            mid = (slot_start + slot_end) / 2.0
            ax.text(mid, -0.18, fam_disp, transform=trans,
                    ha="right", va="top", fontsize=fs_xtick, fontweight="bold",
                    rotation=60, rotation_mode="anchor")
            bracket_pad = max(0.2, 0.3 * density_scale)
            ax.plot([slot_start - bracket_pad, slot_end + bracket_pad], [-0.15, -0.15], transform=trans,
                    color="gray", linewidth=max(0.9, 1.2 * density_scale), clip_on=False)

        # Add star legend to first subplot
        if i == 0:
            import matplotlib.lines as mlines
            metric_label = {"balanced": "PA*PC", "nap_balanced": "balanced NAP", "pc": "PC %", "pc_nap": "PC mean NAP"}.get(best_metric, best_metric)
            star_handle = mlines.Line2D([], [], color="black", marker="*", linestyle="None",
                                        markersize=30, label=f"Best by {metric_label}")
            ax.legend(handles=[star_handle], loc="lower left", fontsize=fs_family,
                      framealpha=0.9, edgecolor="gray")

    for i in range(len(available), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout(rect=[0, 0.05, 1, 1.0])
    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec", "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")
    fname = f"sweep_group_nap_compact{_sel_suffix}.png"
    plt.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / fname}")


def generate_group_nap_latex(pdf, output_dir: Path, model_colors: dict,
                             models: list, best_idx: dict, best_col: str,
                             best_metric: str = "balanced",
                             best_selection: str = "per_codec"):
    """Generate a LaTeX table of per-group mean NAP for the best config per model.

    Columns: Model, PA Mean NAP, PA CRISPR, PA ORF, PA High, PA Low,
             PC Mean NAP, PC High, PC Low.
    One row per model (best config by --best-metric).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    col_map = {
        "PA Mean NAP": "PA_mean_nap",
        "PA CRISPR": "PA_group_crispr_mean_normalized_average_precision",
        "PA ORF": "PA_group_orf_mean_normalized_average_precision",
        "PA High": "PA_group_high_mean_normalized_average_precision",
        "PA Low": "PA_group_low_mean_normalized_average_precision",
        "PC Mean NAP": "PC_mean_nap",
        "PC High": "PC_group_high_mean_normalized_average_precision",
        "PC Low": "PC_group_low_mean_normalized_average_precision",
        "EF Well": "well_effect_mean_nap",
        "EF Batch": "plate_effect_mean_nap",
    }

    # Check that at least some columns exist
    available_cols = {k: v for k, v in col_map.items() if v in pdf.columns}
    if len(available_cols) < 2:
        print("Not enough per-group NAP columns for LaTeX table, skipping.")
        return

    rows = []
    for model in models:
        bi = best_idx.get(model)
        if bi is None:
            continue
        row_data = pdf.loc[bi]
        family = _get_model_family(model)
        family_disp = FAMILY_DISPLAY.get(family, family)
        codec = _get_codec_label(get_display_name(model))
        entry = {"Model": family_disp, "Codec": codec}
        for label, col in available_cols.items():
            val = row_data.get(col)
            if val is not None and not np.isnan(val):
                entry[label] = f"{val:.4f}"
            else:
                entry[label] = "--"
        rows.append(entry)

    if not rows:
        print("No data for group NAP LaTeX table.")
        return

    # Build LaTeX
    headers = ["Model", "Codec"] + list(available_cols.keys())
    col_spec = "ll" + "r" * len(available_cols)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-group mean NAP for best configuration per model"
                 f" (selected by {best_metric}, {best_selection})" r"}")
    lines.append(r"\label{tab:group_nap}")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")

    # Header row with multi-column grouping
    pa_cols = [h for h in headers[1:] if h.startswith("PA")]
    pc_cols = [h for h in headers[1:] if h.startswith("PC")]
    ef_cols = [h for h in headers[1:] if h.startswith("EF")]
    # Sub-header: short names without PA/PC/EF prefix
    short_names = {"PA Mean NAP": "Mean", "PA CRISPR": "CRISPR", "PA ORF": "ORF",
                   "PA High": "High", "PA Low": "Low",
                   "PC Mean NAP": "Mean", "PC High": "High", "PC Low": "Low",
                   "EF Well": "Well", "EF Batch": "Batch"}

    # Build grouped header
    group_header_parts = [" ", " "]
    cmidrule_parts = []
    col_pos = 3  # 1-indexed, columns 1-2 are Model and Codec
    for group_name, group_cols in [("PA NAP", pa_cols), ("PC NAP", pc_cols), ("Effects NAP", ef_cols)]:
        if group_cols:
            group_header_parts.append(r"\multicolumn{" + str(len(group_cols)) + r"}{c}{\textbf{" + group_name + r"}}")
            cmidrule_parts.append(r"\cmidrule(lr){" + str(col_pos) + r"-" + str(col_pos + len(group_cols) - 1) + r"}")
            col_pos += len(group_cols)
    lines.append(" & ".join(group_header_parts) + r" \\")
    lines.append(" ".join(cmidrule_parts))
    short_header = " & ".join([r"\textbf{Model}", r"\textbf{Codec}"] + [short_names.get(h, h) for h in headers[2:]])
    lines.append(short_header + r" \\")
    lines.append(r"\midrule")

    # Find best value per column for bolding
    # For effect columns, lower is better (less confound)
    _lower_is_better = {"EF Well", "EF Batch"}
    col_best = {}
    for h in headers[1:]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[h]))
            except (ValueError, KeyError):
                pass
        if vals:
            col_best[h] = min(vals) if h in _lower_is_better else max(vals)

    for entry in rows:
        cells = [entry["Model"].replace("_", r"\_"), entry["Codec"].replace("_", r"\_")]
        for h in headers[2:]:
            val_str = entry.get(h, "--")
            try:
                val = float(val_str)
                if h in col_best and abs(val - col_best[h]) < 1e-6:
                    cells.append(r"\textbf{" + val_str + r"}")
                else:
                    cells.append(val_str)
            except ValueError:
                cells.append(val_str)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)

    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec", "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")
    fname = f"group_nap_table{_sel_suffix}.tex"
    out_path = output_dir / fname
    out_path.write_text(latex_str)
    print(f"Saved LaTeX table: {out_path}")


def generate_group_nap_latex_compact(pdf, output_dir: Path, model_colors: dict,
                                     models: list, best_idx: dict, best_col: str,
                                     best_metric: str = "balanced",
                                     best_selection: str = "per_codec"):
    """Compact LaTeX table: one row per model family (highest quality codec only).

    Drops Codec, EF Well, and EF Batch columns.
    Picks the model with the lowest compression level per family.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    col_map = {
        "PA Mean NAP": "PA_mean_nap",
        "PA CRISPR": "PA_group_crispr_mean_normalized_average_precision",
        "PA ORF": "PA_group_orf_mean_normalized_average_precision",
        "PA High": "PA_group_high_mean_normalized_average_precision",
        "PA Low": "PA_group_low_mean_normalized_average_precision",
        "PC Mean NAP": "PC_mean_nap",
        "PC High": "PC_group_high_mean_normalized_average_precision",
        "PC Low": "PC_group_low_mean_normalized_average_precision",
    }

    available_cols = {k: v for k, v in col_map.items() if v in pdf.columns}
    if len(available_cols) < 2:
        print("Not enough per-group NAP columns for compact LaTeX table, skipping.")
        return

    # Pick the highest quality (lowest compression level) codec per family
    family_best: dict[str, tuple[str, int]] = {}  # family -> (model, comp_level)
    for model in models:
        bi = best_idx.get(model)
        if bi is None:
            continue
        family = _get_model_family(model)
        display = get_display_name(model)
        comp_level = _get_compression_level(display)
        if family not in family_best or comp_level < family_best[family][1]:
            family_best[family] = (model, comp_level)

    rows = []
    for model in models:
        family = _get_model_family(model)
        if family not in family_best or family_best[family][0] != model:
            continue
        bi = best_idx.get(model)
        if bi is None:
            continue
        row_data = pdf.loc[bi]
        family_disp = FAMILY_DISPLAY.get(family, family)
        codec = _get_codec_label(get_display_name(model))
        entry = {"Model": family_disp, "Codec": codec}
        for label, col in available_cols.items():
            val = row_data.get(col)
            if val is not None and not np.isnan(val):
                entry[label] = f"{val:.4f}"
            else:
                entry[label] = "--"
        rows.append(entry)

    if not rows:
        print("No data for compact group NAP LaTeX table.")
        return

    headers = ["Model", "Codec"] + list(available_cols.keys())
    col_spec = "ll" + "r" * len(available_cols)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Per-group mean NAP for best configuration per model (highest quality codec)"
                 f" (selected by {best_metric}, {best_selection})" r"}")
    lines.append(r"\label{tab:group_nap_compact}")
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")

    pa_cols = [h for h in headers[1:] if h.startswith("PA")]
    pc_cols = [h for h in headers[1:] if h.startswith("PC")]
    short_names = {"PA Mean NAP": "Mean", "PA CRISPR": "CRISPR", "PA ORF": "ORF",
                   "PA High": "High", "PA Low": "Low",
                   "PC Mean NAP": "Mean", "PC High": "High", "PC Low": "Low"}

    group_header_parts = [" ", " "]
    cmidrule_parts = []
    col_pos = 3
    for group_name, group_cols in [("PA NAP", pa_cols), ("PC NAP", pc_cols)]:
        if group_cols:
            group_header_parts.append(r"\multicolumn{" + str(len(group_cols)) + r"}{c}{\textbf{" + group_name + r"}}")
            cmidrule_parts.append(r"\cmidrule(lr){" + str(col_pos) + r"-" + str(col_pos + len(group_cols) - 1) + r"}")
            col_pos += len(group_cols)
    lines.append(" & ".join(group_header_parts) + r" \\")
    lines.append(" ".join(cmidrule_parts))
    short_header = " & ".join([r"\textbf{Model}", r"\textbf{Codec}"] + [short_names.get(h, h) for h in headers[2:]])
    lines.append(short_header + r" \\")
    lines.append(r"\midrule")

    col_best = {}
    for h in headers[1:]:
        vals = []
        for r in rows:
            try:
                vals.append(float(r[h]))
            except (ValueError, KeyError):
                pass
        if vals:
            col_best[h] = max(vals)

    for entry in rows:
        cells = [entry["Model"].replace("_", r"\_"), entry["Codec"].replace("_", r"\_")]
        for h in headers[2:]:
            val_str = entry.get(h, "--")
            try:
                val = float(val_str)
                if h in col_best and abs(val - col_best[h]) < 1e-6:
                    cells.append(r"\textbf{" + val_str + r"}")
                else:
                    cells.append(val_str)
            except ValueError:
                cells.append(val_str)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex_str = "\n".join(lines)

    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec", "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")
    fname = f"group_nap_table_compact{_sel_suffix}.tex"
    out_path = output_dir / fname
    out_path.write_text(latex_str)
    print(f"Saved LaTeX table: {out_path}")


def generate_all_metrics_violin(pdf, output_dir: Path, model_colors: dict,
                                models: list, best_idx: dict, best_col: str,
                                best_metric: str = "balanced",
                                best_selection: str = "per_codec"):
    """Violin plot version of the all-metrics grid.

    Shows distribution of each metric per model as violins with the best
    config highlighted as a star marker.
    """
    import pandas as pd
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)

    n_models = len(models)
    display_names = [get_display_name(m) for m in models]

    all_metrics = [
        ("PA", "PA (%)", "PA"),
        ("PC", "PC (%)", "PC"),
        ("PC Replicable", "PC_rep (%)", "PC_replicable"),
        ("Balanced Score", "PA * PC / 100", "balanced_score"),
        ("Balanced NAP", "PA_nap * PC_nap", "nap_balanced"),
        ("PA Mean NAP", "NAP", "PA_mean_nap"),
        ("PA Median NAP", "NAP", "PA_median_nap"),
        ("PC Mean NAP", "NAP", "PC_mean_nap"),
        ("PC Median NAP", "NAP", "PC_median_nap"),
        ("PC Rep Mean NAP", "NAP", "PC_replicable_mean_nap"),
        ("PC Rep Median NAP", "NAP", "PC_replicable_median_nap"),
        ("n Compounds", "Count", "n_compounds"),
        ("n Targets Active", "Count", "n_targets_active"),
        ("n Targets Total", "Count", "n_targets_total"),
        ("PC Rep n Targets Active", "Count", "PC_replicable_n_targets_active"),
        ("PC Rep n Targets Total", "Count", "PC_replicable_n_targets_total"),
        ("PC Rep n Compounds", "Count", "PC_replicable_n_compounds"),
        ("n Features", "Count", "n_features"),
        ("PC1 Variance", "Variance", "PC1_variance"),
        ("PC2 Variance", "Variance", "PC2_variance"),
    ]
    available = [(t, y, c) for t, y, c in all_metrics if c in pdf.columns and not pdf[c].isna().all()]

    n_metrics = len(available)
    n_cols = min(4, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    col_width = max(8, n_models * 0.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(col_width * n_cols, 6 * n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Build a palette list matching model order
    palette = [model_colors[m] for m in models]

    for i, (title, ylabel, col) in enumerate(available):
        ax = axes[i]

        # Prepare data for violins: only rows where col is not NaN
        plot_data = pdf[pdf[col].notna()].copy()
        # Map model to categorical with correct order
        plot_data["_model_cat"] = pd.Categorical(plot_data["model"], categories=models, ordered=True)

        # Draw violins
        sns.violinplot(
            data=plot_data,
            x="_model_cat",
            y=col,
            hue="_model_cat",
            order=models,
            hue_order=models,
            palette=palette,
            inner="box",
            cut=0,
            ax=ax,
            saturation=0.8,
            linewidth=0.5,
            legend=False,
        )

        # Overlay best config stars
        for j, model in enumerate(models):
            bi = best_idx.get(model)
            if bi is not None:
                bv = pdf.loc[bi, col]
                if not np.isnan(bv):
                    ax.scatter(j, bv, c=[model_colors[model]], s=350, alpha=1.0,
                               edgecolors="black", linewidths=1.5, marker="*", zorder=10)

        ax.set_xticks(range(n_models))
        ax.set_xticklabels(display_names, rotation=60, ha="right", fontsize=6)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    metric_label = {"balanced": "PA*PC", "nap_balanced": "NAP balanced", "pc": "PC %", "pc_nap": "PC mean NAP"}.get(best_metric, best_metric)
    title_suffix = {"zstd_reference": " (zstd-pinned)", "best_any_codec": " (best-any-codec)", "best_avg_codec": " (best-avg-codec)"}.get(best_selection, "")
    fig.suptitle(f"All Metrics Violin (* = best by {metric_label} per model){title_suffix}", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec", "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")
    fname = f"sweep_all_metrics_violin{_sel_suffix}.png"
    plt.savefig(output_dir / fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / fname}")


def generate_overview_plot(pdf, output_dir: Path, model_colors: dict,
                           models: list, best_idx: dict, best_col: str,
                           best_metric: str = "balanced",
                           best_selection: str = "per_codec"):
    """Generate overview plots: PA, PC, balanced score + best config table."""
    output_dir.mkdir(parents=True, exist_ok=True)

    n_models = len(models)
    display_names = [get_display_name(m) for m in models]

    overview_metrics = [
        ("PA", "PA (%)", "PA"),
        ("PC", "PC (%)", "PC"),
        ("Balanced Score", "PA * PC / 100", "balanced_score"),
        ("PC1 Variance", "Variance", "PC1_variance"),
        ("n Features", "Count", "n_features"),
    ]
    available = [(t, y, c) for t, y, c in overview_metrics if c in pdf.columns and not pdf[c].isna().all()]

    n_metrics = len(available)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 7 * n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (title, ylabel, col) in enumerate(available):
        ax = axes[i]
        for j, model in enumerate(models):
            mdf = pdf[pdf["model"] == model]
            bi = best_idx.get(model)
            other = mdf[mdf.index != bi] if bi is not None else mdf
            vals = other[col].dropna()
            if len(vals) > 0:
                x_jitter = np.random.normal(j, 0.12, len(vals))
                ax.scatter(x_jitter, vals, c=[model_colors[model]], s=40, alpha=0.5,
                           edgecolors="white", linewidths=0.3)
            if bi is not None:
                bv = pdf.loc[bi, col]
                if not np.isnan(bv):
                    ax.scatter(j, bv, c=[model_colors[model]], s=350, alpha=1.0,
                               edgecolors=[model_colors[model]], linewidths=1.5, marker="*", zorder=10)

        ax.set_xticks(range(n_models))
        ax.set_xticklabels(display_names, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    metric_label = {"balanced": "PA*PC", "nap_balanced": "NAP balanced", "pc": "PC %", "pc_nap": "PC mean NAP"}.get(best_metric, best_metric)
    title_suffix = {"zstd_reference": " (zstd-pinned)", "best_any_codec": " (best-any-codec)", "best_avg_codec": " (best-avg-codec)"}.get(best_selection, "")
    fig.suptitle(f"Overview Metrics (* = best by {metric_label} per model){title_suffix}", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec", "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")
    fname = f"sweep_overview{_sel_suffix}.png"
    plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / fname}")

    # Save best config table
    rows = []
    for model in models:
        bi = best_idx.get(model)
        if bi is not None:
            row = {"model": get_display_name(model)}
            for title, _, col in available:
                val = pdf.loc[bi, col]
                row[title] = f"{val:.2f}" if not np.isnan(val) else "N/A"
            row["config"] = pdf.loc[bi, "config"]
            rows.append(row)

    if rows:
        import pandas as pd
        summary = pd.DataFrame(rows)
        summary.to_csv(output_dir / "sweep_overview_best_configs.csv", index=False)
        print(f"Saved: {output_dir / 'sweep_overview_best_configs.csv'}")

        # LaTeX table
        latex_df = summary.drop(columns=["config"])
        metric_cols = [c for c in latex_df.columns if c != "model"]
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Best configuration per compression codec (by balanced score).}",
            r"\label{tab:sweep_overview}",
            r"\scriptsize",
            r"\begin{tabular}{l" + "r" * len(metric_cols) + "}",
            r"\hline\hline",
            r"\rule{0pt}{2.5ex}Compression & " + " & ".join(metric_cols) + r" \\",
            r"\hline",
        ]
        for _, row in latex_df.iterrows():
            vals = [str(row["model"])] + [str(row[c]) for c in metric_cols]
            lines.append(" & ".join(vals) + r" \\")
        lines += [r"\hline\hline", r"\end{tabular}", r"\end{table}"]
        tex_path = output_dir / "sweep_overview_best_configs.tex"
        with open(tex_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Saved: {tex_path}")


def generate_pa_vs_pc_plot(pdf, output_dir: Path, model_colors: dict,
                           models: list, best_idx: dict, family_configs: dict | None,
                           best_metric: str = "balanced",
                           best_selection: str = "per_codec"):
    """PA vs PC scatter plot colored by compression codec."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    model_keys_plotted = []
    for model in models:
        mdf = pdf[pdf["model"] == model]
        ax.scatter(mdf["PC"], mdf["PA"], c=[model_colors[model]], s=40, alpha=0.5,
                   edgecolors="white", linewidths=0.3,
                   label=get_display_name(model))
        model_keys_plotted.append(model)
        bi = best_idx.get(model)
        if bi is not None:
            ax.scatter(pdf.loc[bi, "PC"], pdf.loc[bi, "PA"],
                       c=[model_colors[model]], s=350, edgecolors=[model_colors[model]],
                       linewidths=2, marker="*", zorder=10)

    ax.set_xlim(0, pdf["PC"].max() * 1.1)
    ax.set_ylim(0, pdf["PA"].max() * 1.1)
    _add_balanced_score_lines(ax)
    ax.set_xlabel("Phenotypic Consistency (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Phenotypic Activity (%)", fontsize=14, fontweight="bold")
    title_suffix = {"zstd_reference": " (zstd-pinned)", "best_any_codec": " (best-any-codec)", "best_avg_codec": " (best-avg-codec)"}.get(best_selection, "")
    ax.set_title(f"PA vs PC{title_suffix}", fontsize=16, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    _add_grouped_legend(ax, handles, labels, model_keys_plotted,
                        family_configs=family_configs)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec", "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")
    fname = f"sweep_pa_vs_pc{_sel_suffix}.png"
    plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / fname}")


def generate_pa_vs_pc_targets_plot(pdf, output_dir: Path, model_colors: dict,
                                    models: list, best_idx: dict, family_configs: dict | None,
                                    best_metric: str = "balanced",
                                    best_selection: str = "per_codec"):
    """PA vs n_targets_active scatter plot colored by compression codec."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    model_keys_plotted = []
    for model in models:
        mdf = pdf[pdf["model"] == model]
        ax.scatter(mdf["n_targets_active"], mdf["PA"], c=[model_colors[model]], s=40, alpha=0.5,
                   edgecolors="white", linewidths=0.3,
                   label=get_display_name(model))
        model_keys_plotted.append(model)
        bi = best_idx.get(model)
        if bi is not None:
            ax.scatter(pdf.loc[bi, "n_targets_active"], pdf.loc[bi, "PA"],
                       c=[model_colors[model]], s=350, edgecolors=[model_colors[model]],
                       linewidths=2, marker="*", zorder=10)

    ax.set_xlim(0, pdf["n_targets_active"].max() * 1.1)
    ax.set_ylim(0, pdf["PA"].max() * 1.1)
    ax.set_xlabel("n Targets Active (PC)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Phenotypic Activity (%)", fontsize=14, fontweight="bold")
    title_suffix = {"zstd_reference": " (zstd-pinned)", "best_any_codec": " (best-any-codec)", "best_avg_codec": " (best-avg-codec)"}.get(best_selection, "")
    ax.set_title(f"PA vs n Targets Active{title_suffix}", fontsize=16, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    _add_grouped_legend(ax, handles, labels, model_keys_plotted,
                        family_configs=family_configs)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec", "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")
    fname = f"sweep_pa_vs_pc_targets{_sel_suffix}.png"
    plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / fname}")


def generate_batch_method_plot(pdf, output_dir: Path):
    """Compare batch correction methods across all compression codecs."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build a readable batch label
    def batch_label(row):
        bm = row["batch_method"]
        if bm == "spherize":
            fit = row.get("spherize_fit", "?")
            eps = row.get("spherize_epsilon")
            eps_str = f"{eps:.0e}" if eps is not None and eps < 0.01 else str(eps)
            return f"Spherize({fit},{eps_str})"
        return BATCH_DISPLAY.get(bm, bm)

    pdf["batch_label"] = pdf.apply(batch_label, axis=1)

    batch_order = ["None", "TVN Original", "TVN EFAAR", "Cascade TVN",
                   "Spherize(all,0.5)", "Spherize(all,1e-06)",
                   "Spherize(ctrl,0.5)", "Spherize(ctrl,1e-06)"]
    existing = [b for b in batch_order if b in pdf["batch_label"].values]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for ax, (metric, title) in zip(axes, [("PA", "PA (%)"), ("PC", "PC (%)"), ("balanced_score", "Balanced Score")]):
        sns.boxenplot(data=pdf, x="batch_label", y=metric, order=existing,
                      palette="Set2", ax=ax, k_depth="tukey", linewidth=1)
        ax.set_xlabel("Batch Method", fontsize=12, fontweight="bold")
        ax.set_ylabel(title, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Performance by Batch Correction Method", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_batch_method_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'sweep_batch_method_comparison.png'}")


def generate_norm_pca_plot(pdf, output_dir: Path):
    """Compare normalization methods and PCA on/off."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf["norm_display"] = pdf["norm_method"].map({"robustmad": "RobustMAD", "standardize": "Standardize"})
    pdf["pca_display"] = pdf["use_pca"].map({True: "PCA", False: "No PCA"})

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    for ax, (metric, title) in zip(axes, [("PA", "PA (%)"), ("PC", "PC (%)"), ("balanced_score", "Balanced Score")]):
        # Norm comparison
        data_long = pdf.melt(
            id_vars=["norm_display", "pca_display"],
            value_vars=[metric],
            var_name="metric_name",
            value_name="value",
        )
        data_long["group"] = data_long["norm_display"] + " / " + data_long["pca_display"]

        group_order = ["RobustMAD / No PCA", "RobustMAD / PCA",
                       "Standardize / No PCA", "Standardize / PCA"]
        existing = [g for g in group_order if g in data_long["group"].values]

        sns.boxenplot(data=data_long, x="group", y="value", order=existing,
                      palette="Set3", ax=ax, k_depth="tukey", linewidth=1)
        ax.set_xlabel("Norm / PCA", fontsize=12, fontweight="bold")
        ax.set_ylabel(title, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Performance by Normalization and PCA", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_norm_pca_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'sweep_norm_pca_comparison.png'}")


def generate_norm_batch_comparison(pdf, output_dir: Path):
    """Compare normalization methods crossed with batch correction method."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf["norm_display"] = pdf["norm_method"].map({"robustmad": "RobustMAD", "standardize": "Standardize"})
    pdf["batch_display"] = pdf["batch_method"].map(BATCH_DISPLAY).fillna(pdf["batch_method"])

    pdf["group"] = pdf["norm_display"] + " / " + pdf["batch_display"]

    group_order = [
        f"{norm} / {batch}"
        for norm in ["RobustMAD", "Standardize"]
        for batch in ["None", "TVN Original", "TVN EFAAR", "Cascade TVN", "Spherize"]
    ]
    existing = [g for g in group_order if g in pdf["group"].values]

    fig, axes = plt.subplots(1, 3, figsize=(28, 8))

    for ax, (metric, title) in zip(axes, [("PA", "PA (%)"), ("PC", "PC (%)"), ("balanced_score", "Balanced Score")]):
        sns.boxenplot(data=pdf, x="group", y=metric, order=existing,
                      palette="Paired", ax=ax, k_depth="tukey", linewidth=1)
        ax.set_xlabel("Norm / Batch Method", fontsize=12, fontweight="bold")
        ax.set_ylabel(title, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Performance by Normalization and Batch Correction", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_norm_batch_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'sweep_norm_batch_comparison.png'}")


# Compression level ordering for dot-size mapping (lossless → heavy lossy)
# Higher rank = more lossy = smaller dot
COMPRESSION_LEVEL = {
    "raw": 0,       # uncompressed baseline
    "zstd": 1,      # lossless
    "hq": 2,        # light lossy
    "effort_3": 3,  # moderate lossy
    "d2_e8": 4,     # distance 2 effort 8
    "mq_new": 5,    # medium quality (new settings)
    "mq": 6,        # medium quality
    "lq": 7,        # low quality (heavy lossy)
    "d10": 8,       # distance 10
    "d15": 9,       # distance 15
    "d20": 10,      # distance 20
    "d20_e2": 10,   # distance 20 effort 2
    "d30": 11,      # distance 30
    "d50": 12,      # distance 50 (most lossy)
}

# Short codec aliases that map to canonical COMPRESSION_LEVEL keys
_CODEC_ALIASES = {
    "e3": "effort_3",
    "effort_3": "effort_3",
    "hq": "hq",
    "mq": "mq",
    "lq": "lq",
    "zstd": "zstd",
    "d10": "d10",
    "d15": "d15",
    "d20": "d20",
    "d20_e2": "d20_e2",
    "d2_e8": "d2_e8",
    "mq_new": "mq_new",
    "d25": "d30",  # d25 is displayed name for d30
    "d30": "d30",
    "d50": "d50",
    "raw": "raw",
}

# Prefixes to strip from display names to get bare codec label
_MODEL_DISPLAY_PREFIXES = [
    "dv2_490_",
    "dv2_rand_rr_", "dv2_rand_lite_", "dv2_rand_",  # rr/lite before non-rr
    "dv2_rr_", "dv2_lite_", "dv2_cl_", "dv2_",      # rr/lite before cl before bare dv2
    "morphem_rr_", "morphem_lite_", "morphem_",
    "subcell_rr_", "subcell_lite_", "subcell_",
    "ophenom_rr_", "ophenom_lite_", "ophenom_ss_", "ophenom_cl_", "ophenom_",
    "cp_lite_",     # CellProfiler lite
    "cp_fbs_",
    "cc_lite_",     # Cell Count lite
    "cc_",          # Cell Count baseline
    "sc01_lite_",   # SubCell clip01 lite
    "sc01_",        # SubCell clip01
]

# Map from display name back to compression level rank
_DISPLAY_TO_LEVEL = {}
# Map from display name to codec-only label (no model prefix)
_DISPLAY_TO_CODEC = {}
for _raw_name, _disp in COMPRESSION_DISPLAY.items():
    # Extract base codec from display name (strip model prefix like dv2_490_, morphem_, etc.)
    for _prefix in _MODEL_DISPLAY_PREFIXES:
        if _disp.startswith(_prefix):
            _codec = _disp[len(_prefix):]
            _DISPLAY_TO_CODEC[_disp] = _codec
            _canonical = _CODEC_ALIASES.get(_codec, _codec)
            if _canonical in COMPRESSION_LEVEL:
                _DISPLAY_TO_LEVEL[_disp] = COMPRESSION_LEVEL[_canonical]
            break
    else:
        # CellProfiler codecs: strip _f suffix for filtered variants
        _DISPLAY_TO_CODEC[_disp] = _disp
        _base = _disp[:-2] if _disp.endswith("_f") else _disp
        _canonical = _CODEC_ALIASES.get(_base, _base)
        if _canonical in COMPRESSION_LEVEL:
            _DISPLAY_TO_LEVEL[_disp] = COMPRESSION_LEVEL[_canonical]


def _get_codec_label(display_name: str) -> str:
    """Strip model prefix from display name, returning only the codec part."""
    return _DISPLAY_TO_CODEC.get(display_name, display_name)


def _get_model_codec_label(model: str) -> str:
    """Return a two-line label 'Family\\nCodec' for x-axis tick labels."""
    family = _get_model_family(model)
    family_disp = FAMILY_DISPLAY.get(family, family)
    display = get_display_name(model)
    codec = _DISPLAY_TO_CODEC.get(display, display)
    return f"{family_disp}\n{codec}"


def _add_balanced_score_lines(ax, is_nap=False):
    """Add iso-balanced-score hyperbolas to a PA-vs-PC scatter plot.

    Balanced score = PA * PC / 100 (percentage axes) or PA * PC (NAP axes).
    Lines are clipped to the current axis limits.

    Args:
        ax: Matplotlib axes (x=PC, y=PA).
        is_nap: If True, axes are NAP values ([0,~0.5]) instead of percentages.
    """
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()

    # Pick score values that fall within the visible range
    if is_nap:
        candidates = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    else:
        candidates = [2, 5, 10, 15, 20, 25]

    # Only keep scores whose hyperbola intersects the visible rectangle
    score_values = []
    for s in candidates:
        if is_nap:
            # PA = s / PC  → at x=x_hi, PA_min = s / x_hi
            pa_at_xhi = s / x_hi if x_hi > 0 else float("inf")
        else:
            # PA = s * 100 / PC  → at x=x_hi, PA_min = s * 100 / x_hi
            pa_at_xhi = s * 100 / x_hi if x_hi > 0 else float("inf")
        if pa_at_xhi < y_hi:
            score_values.append(s)

    if not score_values:
        return

    pc = np.linspace(max(x_lo, 0.01), x_hi, 500)

    for score in score_values:
        if is_nap:
            pa = score / pc
        else:
            pa = score * 100 / pc

        mask = (pa >= y_lo) & (pa <= y_hi)
        if mask.sum() < 2:
            continue
        ax.plot(pc[mask], pa[mask], "--", color="gray", alpha=0.35, linewidth=0.8, zorder=1)

        pass  # lines only, no labels


def _get_compression_level(display_name: str) -> int:
    """Get compression level rank for a display name. Lower = less lossy."""
    return _DISPLAY_TO_LEVEL.get(display_name, 4)  # default to middle


def _level_to_size(level: int, min_size: float = 60, max_size: float = 350) -> float:
    """Convert compression level (0=lossless, 7=heavy lossy) to dot size."""
    max_level = max(COMPRESSION_LEVEL.values())
    # Invert: lossless = big dot, heavy lossy = small dot
    return max_size - (max_size - min_size) * level / max(max_level, 1)


def _get_model_family(model: str) -> str:
    """Get the model family name for a raw model key."""
    for family, members in MODEL_FAMILIES.items():
        if model in members:
            return family
    # Fallback: infer from prefix (handles jump_lite names etc.)
    inferred = _infer_family(model)
    return inferred if inferred else "unknown"


def _add_grouped_legend(ax, handles, labels, model_keys, loc="upper left",
                        fontsize=7, title_fontsize=9, family_configs=None):
    """Add a grouped legend to the right of the plot.

    Families are stacked vertically on the right side, each with a bold
    title followed by its codec entries.

    Args:
        ax: Matplotlib axes.
        handles/labels: From ax.get_legend_handles_labels().
        model_keys: List of raw model keys in the same order as handles/labels.
        family_configs: Optional dict of family -> config name. When provided,
                        the pinned pipeline config is shown under each family title.
    """
    # Group handles/labels by family, preserving family order from MODEL_FAMILIES
    family_order = list(MODEL_FAMILIES.keys())
    family_groups: dict[str, list[tuple]] = {f: [] for f in family_order}

    for model_key, handle, label in zip(model_keys, handles, labels):
        fam = _get_model_family(model_key)
        if fam in family_groups:
            family_groups[fam].append((handle, label))
        else:
            family_groups.setdefault("unknown", []).append((handle, label))

    # Only keep families that have entries
    active_families = [(f, items) for f, items in family_groups.items() if items]
    if not active_families:
        return

    all_handles = []
    all_labels = []

    for fam, items in active_families:
        title = FAMILY_DISPLAY.get(fam, fam)
        title_escaped = title.replace("_", r"\_")
        all_handles.append(ax.scatter([], [], s=0, alpha=0))
        all_labels.append(f"$\\bf{{{title_escaped}}}$")
        if family_configs:
            config = family_configs.get(fam)
            config_short = (config if config and len(config) <= 40
                            else (config[:37] + "..." if config else "(fallback)"))
            all_handles.append(ax.scatter([], [], s=0, alpha=0))
            all_labels.append(f"  [{config_short}]")
        for handle, label in items:
            all_handles.append(handle)
            all_labels.append(label)

    ax.legend(handles=all_handles, labels=all_labels,
              loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=5, title_fontsize=7,
              ncol=1, labelspacing=0.95, handletextpad=1.0,
              frameon=False, borderaxespad=0)


def _build_family_colors() -> dict[str, tuple]:
    """Build one color per model family, matching the first codec in _build_model_colors."""
    import matplotlib.colors as mcolors

    # Use the same HSV formula as _build_model_colors with idx=0
    # (sat = 1.0, val = 1.0 for the first/brightest member)
    return {
        family: mcolors.hsv_to_rgb([hue, 1.0, 1.0])
        for family, hue in FAMILY_HUES.items()
    }


# Display names for model families (used in legends)
FAMILY_DISPLAY = {
    "cellprofiler": "CellProfiler",
    "cp_measure": "cp_measure",
    "cell_count": "CellCount",
    "cell_count_lite": "CellCount",
    "cp_measure_fbs": "cp_measure_fbs",
    "cp_measure_filtered": "cp_measure_filtered",
    "dinov2_490": "DINOv2-490",
    "dinov2": "DINOv2",
    "dinov2_random": "ViT-rand",
    "morphem": "MorphEm",
    "subcell": "SubCell",
    "subcell__nonstd": "SubCell-NonStd",
    "subcell__clip01": "SubCell",
    "openphenom_stdscale": "OpenPhenom-StdScale",
    "openphenom_stdscale_false": "OpenPhenom-StdScale-F",
    "openphenom_nonclip": "OpenPhenom-NoClip",
    "openphenom": "OpenPhenom",
    "dinov2_rr": "DINOv2-RR",
    "dinov2_random_rr": "ViT-rand-RR",
    "morphem_rr": "MorphEm-RR",
    "subcell_rr": "SubCell-RR",
    "openphenom_rr": "OpenPhenom-RR",
    # Jump-lite families
    "cellprofiler_lite": "CellProfiler",
    "dinov2_lite": "DINOv2",
    "dinov2_random_lite": "ViT-rand",
    "morphem_lite": "MorphEm",
    "subcell_lite": "SubCell",
    "subcell__clip01_lite": "SubCell",
    "openphenom_lite": "OpenPhenom",
}


def generate_pa_vs_pc_best_balanced(pdf, output_dir: Path, model_colors: dict,
                                     models: list, best_idx: dict, family_configs: dict | None,
                                     best_metric: str = "balanced",
                                     best_selection: str = "per_codec"):
    """PA% vs PC% scatter showing best config per model-codec.

    Each dot = best pipeline config for one model-codec combination.
    Color = model family (CellProfiler, MorphEm, etc.).
    Size = compression level (larger = less lossy).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    best_rows = [pdf.loc[bi] for bi in best_idx.values()]
    if not best_rows:
        print("No data for PA vs PC best balanced plot.")
        return

    import pandas as pd
    best = pd.DataFrame(best_rows)
    best["comp_level"] = best["display_name"].apply(_get_compression_level)
    best["dot_size"] = best["comp_level"].apply(_level_to_size)

    fig, ax = plt.subplots(figsize=(14, 12))

    model_keys_plotted = []
    for _, row in best.iterrows():
        model = row["model"]
        color = model_colors.get(model, (0.5, 0.5, 0.5))
        ax.scatter(
            row["PC"], row["PA"],
            c=[color],
            s=row["dot_size"],
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=row["display_name"],
        )
        model_keys_plotted.append(model)
        ax.annotate(
            _get_codec_label(row["display_name"]),
            (row["PC"], row["PA"]),
            fontsize=7, alpha=0.8,
            xytext=(4, 4), textcoords="offset points",
        )

    handles, labels = ax.get_legend_handles_labels()
    _add_grouped_legend(ax, handles, labels, model_keys_plotted,
                        family_configs=family_configs)

    ax.set_xlim(0, best["PC"].max() * 1.1)
    ax.set_ylim(0, best["PA"].max() * 1.1)
    _add_balanced_score_lines(ax)
    ax.set_xlabel("Phenotypic Consistency (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Phenotypic Activity (%)", fontsize=14, fontweight="bold")
    metric_label = {"balanced": "PA*PC", "nap_balanced": "NAP balanced", "pc": "PC %", "pc_nap": "PC mean NAP"}.get(best_metric, best_metric)
    title_suffix = {"zstd_reference": " (zstd-pinned)", "best_any_codec": " (best-any-codec)", "best_avg_codec": " (best-avg-codec)"}.get(best_selection, "")
    ax.set_title(f"PA vs PC — Best by {metric_label} per Model-Codec{title_suffix}", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec", "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")
    fname = f"sweep_pa_vs_pc_best_balanced{_sel_suffix}.png"
    plt.savefig(output_dir / fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / fname}")


def generate_nap_pa_vs_pc_best_balanced(pdf, output_dir: Path, model_colors: dict,
                                         models: list, best_idx: dict, family_configs: dict | None,
                                         best_metric: str = "balanced",
                                         best_selection: str = "per_codec"):
    """Mean NAP PA vs Mean NAP PC scatter showing best config per model-codec.

    Same layout as generate_pa_vs_pc_best_balanced but using NAP metrics.
    Color = model family, size = compression level.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if "PA_mean_nap" not in pdf.columns or "PC_mean_nap" not in pdf.columns:
        print("NAP metrics not available, skipping NAP PA vs PC plot.")
        return
    best_rows = [pdf.loc[bi] for bi in best_idx.values()]
    if not best_rows:
        print("No data for NAP PA vs PC best balanced plot.")
        return

    import pandas as pd
    best = pd.DataFrame(best_rows)
    best["comp_level"] = best["display_name"].apply(_get_compression_level)
    best["dot_size"] = best["comp_level"].apply(_level_to_size)

    fig, ax = plt.subplots(figsize=(10, 10))

    model_keys_plotted = []
    for _, row in best.iterrows():
        pa_nap = row.get("PA_mean_nap")
        pc_nap = row.get("PC_mean_nap")
        if pa_nap is None or pc_nap is None or np.isnan(pa_nap) or np.isnan(pc_nap):
            continue
        model = row["model"]
        color = model_colors.get(model, (0.5, 0.5, 0.5))
        ax.scatter(
            pc_nap, pa_nap,
            c=[color],
            s=row["dot_size"],
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=row["display_name"],
        )
        model_keys_plotted.append(model)
        ax.annotate(
            _get_codec_label(row["display_name"]),
            (pc_nap, pa_nap),
            fontsize=12, alpha=0.8,
            xytext=(4, 4), textcoords="offset points",
        )

    handles, labels = ax.get_legend_handles_labels()
    _add_grouped_legend(ax, handles, labels, model_keys_plotted,
                        family_configs=family_configs)

    nap_pa_vals = best["PA_mean_nap"].dropna()
    nap_pc_vals = best["PC_mean_nap"].dropna()
    ax.set_xlim(0, nap_pc_vals.max() * 1.15 if len(nap_pc_vals) > 0 else 0.15)
    ax.set_ylim(0, nap_pa_vals.max() * 1.15 if len(nap_pa_vals) > 0 else 0.5)
    _add_balanced_score_lines(ax, is_nap=True)
    ax.set_xlabel("PC Mean NAP", fontsize=20, fontweight="bold")
    ax.set_ylabel("PA Mean NAP", fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=16)
    metric_label = {"balanced": "PA*PC", "nap_balanced": "NAP balanced", "pc": "PC %", "pc_nap": "PC mean NAP"}.get(best_metric, best_metric)
    title_suffix = {"zstd_reference": " (zstd-pinned)", "best_any_codec": " (best-any-codec)", "best_avg_codec": " (best-avg-codec)"}.get(best_selection, "")
    ax.set_title(f"Mean NAP: PA vs PC — Best by {metric_label} per Model-Codec{title_suffix}", fontsize=22, fontweight="bold")
    plt.tight_layout()
    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec", "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")
    fname = f"sweep_nap_pa_vs_pc_best_balanced{_sel_suffix}.png"
    plt.savefig(output_dir / fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / fname}")


def generate_nap_pa_vs_pc_best_balanced_clean(pdf, output_dir: Path, model_colors: dict,
                                               models: list, best_idx: dict, family_configs: dict | None,
                                               best_metric: str = "balanced",
                                               best_selection: str = "per_codec"):
    """Clean NAP PA vs PC scatter: uniform dot size, one legend entry per family, no annotations.

    Same data as generate_nap_pa_vs_pc_best_balanced but simplified for presentation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if "PA_mean_nap" not in pdf.columns or "PC_mean_nap" not in pdf.columns:
        print("NAP metrics not available, skipping clean NAP PA vs PC plot.")
        return
    best_rows = [pdf.loc[bi] for bi in best_idx.values()]
    if not best_rows:
        print("No data for clean NAP PA vs PC plot.")
        return

    import pandas as pd
    best = pd.DataFrame(best_rows)
    best["family"] = best["model"].apply(_get_model_family)

    fig, ax = plt.subplots(figsize=(10, 10))

    # One legend entry per family
    families_seen = set()
    for _, row in best.iterrows():
        pa_nap = row.get("PA_mean_nap")
        pc_nap = row.get("PC_mean_nap")
        if pa_nap is None or pc_nap is None or np.isnan(pa_nap) or np.isnan(pc_nap):
            continue
        model = row["model"]
        family = row["family"]
        color = model_colors.get(model, (0.5, 0.5, 0.5))
        label = FAMILY_DISPLAY.get(family, family) if family not in families_seen else None
        ax.scatter(
            pc_nap, pa_nap,
            c=[color],
            s=120,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=label,
        )
        families_seen.add(family)

    nap_pa_vals = best["PA_mean_nap"].dropna()
    nap_pc_vals = best["PC_mean_nap"].dropna()
    ax.set_xlim(0, nap_pc_vals.max() * 1.15 if len(nap_pc_vals) > 0 else 0.15)
    ax.set_ylim(0, nap_pa_vals.max() * 1.15 if len(nap_pa_vals) > 0 else 0.5)
    _add_balanced_score_lines(ax, is_nap=True)
    ax.set_xlabel("PC Mean NAP", fontsize=20, fontweight="bold")
    ax.set_ylabel("PA Mean NAP", fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=16)
    metric_label = {"balanced": "PA*PC", "nap_balanced": "NAP balanced", "pc": "PC %", "pc_nap": "PC mean NAP"}.get(best_metric, best_metric)
    title_suffix = {"zstd_reference": " (zstd-pinned)", "best_any_codec": " (best-any-codec)", "best_avg_codec": " (best-avg-codec)"}.get(best_selection, "")
    ax.set_title(f"Mean NAP: PA vs PC — Best by {metric_label}{title_suffix}", fontsize=22, fontweight="bold")
    ax.legend(loc="lower right", fontsize=14, framealpha=0.9, edgecolor="gray")
    plt.tight_layout()
    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec", "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")
    fname = f"sweep_nap_pa_vs_pc_best_balanced_clean{_sel_suffix}.png"
    plt.savefig(output_dir / fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / fname}")


def generate_nap_replicable_pa_vs_pc_best_balanced(pdf, output_dir: Path, model_colors: dict,
                                                     models: list, best_idx: dict, family_configs: dict | None,
                                                     best_metric: str = "balanced",
                                                     best_selection: str = "per_codec"):
    """Mean NAP PA vs Mean NAP PC_replicable scatter showing best config per model-codec.

    Same layout as generate_nap_pa_vs_pc_best_balanced but using PC_replicable_mean_nap
    on the x-axis instead of PC_mean_nap.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if "PA_mean_nap" not in pdf.columns or "PC_replicable_mean_nap" not in pdf.columns:
        print("NAP replicable metrics not available, skipping NAP PA vs PC_replicable plot.")
        return

    best_rows = [pdf.loc[bi] for bi in best_idx.values()]
    if not best_rows:
        print("No data for NAP PA vs PC_replicable best balanced plot.")
        return

    import pandas as pd
    best = pd.DataFrame(best_rows)
    best["comp_level"] = best["display_name"].apply(_get_compression_level)
    best["dot_size"] = best["comp_level"].apply(_level_to_size)

    fig, ax = plt.subplots(figsize=(14, 12))

    model_keys_plotted = []
    for _, row in best.iterrows():
        pa_nap = row.get("PA_mean_nap")
        pc_rep_nap = row.get("PC_replicable_mean_nap")
        if pa_nap is None or pc_rep_nap is None or np.isnan(pa_nap) or np.isnan(pc_rep_nap):
            continue
        model = row["model"]
        color = model_colors.get(model, (0.5, 0.5, 0.5))
        ax.scatter(
            pc_rep_nap, pa_nap,
            c=[color],
            s=row["dot_size"],
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=row["display_name"],
        )
        model_keys_plotted.append(model)
        ax.annotate(
            _get_codec_label(row["display_name"]),
            (pc_rep_nap, pa_nap),
            fontsize=7, alpha=0.8,
            xytext=(4, 4), textcoords="offset points",
        )

    handles, labels = ax.get_legend_handles_labels()
    _add_grouped_legend(ax, handles, labels, model_keys_plotted,
                        family_configs=family_configs)

    nap_pa_vals = best["PA_mean_nap"].dropna()
    nap_pc_vals = best["PC_replicable_mean_nap"].dropna()
    ax.set_xlim(0, nap_pc_vals.max() * 1.15 if len(nap_pc_vals) > 0 else 0.15)
    ax.set_ylim(0, nap_pa_vals.max() * 1.15 if len(nap_pa_vals) > 0 else 0.5)
    _add_balanced_score_lines(ax, is_nap=True)
    ax.set_xlabel("PC Replicable Mean NAP", fontsize=14, fontweight="bold")
    ax.set_ylabel("PA Mean NAP", fontsize=14, fontweight="bold")
    metric_label = {"balanced": "PA*PC", "nap_balanced": "NAP balanced", "pc": "PC %", "pc_nap": "PC mean NAP"}.get(best_metric, best_metric)
    title_suffix = {"zstd_reference": " (zstd-pinned)", "best_any_codec": " (best-any-codec)", "best_avg_codec": " (best-avg-codec)"}.get(best_selection, "")
    ax.set_title(f"Mean NAP: PA vs PC Replicable — Best by {metric_label} per Model-Codec{title_suffix}", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec", "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")
    fname = f"sweep_nap_pa_vs_pc_replicable_best_balanced{_sel_suffix}.png"
    plt.savefig(output_dir / fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / fname}")


def generate_nap_pa_vs_pc_targets_best_balanced(pdf, output_dir: Path, model_colors: dict,
                                                  models: list, best_idx: dict, family_configs: dict | None,
                                                  best_metric: str = "balanced",
                                                  best_selection: str = "per_codec"):
    """Mean NAP PA (y) vs n_targets_active (x) scatter showing best config per model-codec.

    Same layout as generate_nap_pa_vs_pc_best_balanced but with the number of
    active PC targets on the x-axis instead of PC_mean_nap.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if "PA_mean_nap" not in pdf.columns or "n_targets_active" not in pdf.columns:
        print("PA_mean_nap or n_targets_active not available, skipping NAP PA vs targets plot.")
        return
    best_rows = [pdf.loc[bi] for bi in best_idx.values()]
    if not best_rows:
        print("No data for NAP PA vs targets plot.")
        return

    import pandas as pd
    best = pd.DataFrame(best_rows)
    best["comp_level"] = best["display_name"].apply(_get_compression_level)
    best["dot_size"] = best["comp_level"].apply(_level_to_size)

    fig, ax = plt.subplots(figsize=(14, 12))

    model_keys_plotted = []
    for _, row in best.iterrows():
        pa_nap = row.get("PA_mean_nap")
        n_active = row.get("n_targets_active")
        if pa_nap is None or n_active is None or np.isnan(pa_nap) or np.isnan(n_active):
            continue
        model = row["model"]
        color = model_colors.get(model, (0.5, 0.5, 0.5))
        ax.scatter(
            n_active, pa_nap,
            c=[color],
            s=row["dot_size"],
            alpha=0.85,
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label=row["display_name"],
        )
        model_keys_plotted.append(model)
        ax.annotate(
            _get_codec_label(row["display_name"]),
            (n_active, pa_nap),
            fontsize=7, alpha=0.8,
            xytext=(4, 4), textcoords="offset points",
        )

    handles, labels = ax.get_legend_handles_labels()
    _add_grouped_legend(ax, handles, labels, model_keys_plotted,
                        family_configs=family_configs)

    n_active_vals = best["n_targets_active"].dropna()
    nap_pa_vals = best["PA_mean_nap"].dropna()
    ax.set_xlim(0, n_active_vals.max() * 1.15 if len(n_active_vals) > 0 else 50)
    ax.set_ylim(0, nap_pa_vals.max() * 1.15 if len(nap_pa_vals) > 0 else 0.5)
    ax.set_xlabel("n Targets Active (PC)", fontsize=14, fontweight="bold")
    ax.set_ylabel("PA Mean NAP", fontsize=14, fontweight="bold")
    metric_label = {"balanced": "PA*PC", "nap_balanced": "NAP balanced", "pc": "PC %", "pc_nap": "PC mean NAP"}.get(best_metric, best_metric)
    title_suffix = {"zstd_reference": " (zstd-pinned)", "best_any_codec": " (best-any-codec)", "best_avg_codec": " (best-avg-codec)"}.get(best_selection, "")
    ax.set_title(f"Mean NAP PA vs n Targets Active — Best by {metric_label} per Model-Codec{title_suffix}", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec", "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")
    fname = f"sweep_nap_pa_vs_pc_targets_best_balanced{_sel_suffix}.png"
    plt.savefig(output_dir / fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / fname}")


def generate_nap_pa_vs_pc_per_model(pdf, output_dir: Path, model_colors: dict,
                                     models: list, best_idx: dict, family_configs: dict | None,
                                     best_metric: str = "balanced",
                                     best_selection: str = "per_codec"):
    """Per-family NAP PA vs PC subplots, each zoomed to its own data range.

    One subplot per model family.  Within each subplot the best config per
    model-codec is plotted, coloured by model and sized by compression level.
    Axes are auto-ranged to the family's data with a small margin.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if "PA_mean_nap" not in pdf.columns or "PC_mean_nap" not in pdf.columns:
        print("NAP metrics not available, skipping per-model NAP PA vs PC plot.")
        return

    import pandas as pd

    best_rows = [pdf.loc[bi] for bi in best_idx.values()]
    if not best_rows:
        print("No data for per-model NAP PA vs PC plot.")
        return
    best = pd.DataFrame(best_rows)
    best["comp_level"] = best["display_name"].apply(_get_compression_level)
    # Exponential sizing: lossless (level 0) = huge, heavy lossy = small
    _max_level = max(max(COMPRESSION_LEVEL.values()), 1)
    best["dot_size"] = best["comp_level"].apply(
        lambda lvl: 30 + 950 * np.exp(-2.5 * lvl / _max_level)
    )
    best["family"] = best["model"].apply(_get_model_family)

    # Only keep rows with valid NAP values
    best = best.dropna(subset=["PA_mean_nap", "PC_mean_nap"])
    if best.empty:
        print("No valid NAP data for per-model plot.")
        return

    # Determine families present (preserve MODEL_FAMILIES order)
    families_present = [f for f in MODEL_FAMILIES if f in best["family"].values]
    n_families = len(families_present)
    if n_families == 0:
        print("No families found for per-model NAP PA vs PC plot.")
        return

    nrows = 4
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 10 * nrows), squeeze=False)

    # Scale fonts so they match the 10x10 clean figure when both are displayed
    # at the same height (this figure is nrows * 10 tall).
    s = nrows  # scale factor
    fs_title = 22 * s
    fs_subtitle = 20 * s
    fs_axis = 20 * s
    fs_tick = 16 * s
    fs_annot = 12 * s

    for idx, family in enumerate(families_present):
        row_idx, col_idx = idx // ncols, idx % ncols
        ax = axes[row_idx][col_idx]
        fam_data = best[best["family"] == family]
        if fam_data.empty:
            ax.set_visible(False)
            continue

        for _, row in fam_data.iterrows():
            color = model_colors.get(row["model"], (0.5, 0.5, 0.5))
            ax.scatter(
                row["PC_mean_nap"], row["PA_mean_nap"],
                c=[color],
                s=row["dot_size"],
                alpha=0.85,
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
            ax.annotate(
                _get_codec_label(row["display_name"]),
                (row["PC_mean_nap"], row["PA_mean_nap"]),
                fontsize=fs_annot, alpha=0.8,
                xytext=(16, 4), textcoords="offset points",
            )

        # Zoom to this family's range with 15% margin
        pa_vals = fam_data["PA_mean_nap"]
        pc_vals = fam_data["PC_mean_nap"]
        pa_range = pa_vals.max() - pa_vals.min() if len(pa_vals) > 1 else pa_vals.max() * 0.1
        pc_range = pc_vals.max() - pc_vals.min() if len(pc_vals) > 1 else pc_vals.max() * 0.1
        margin_pa = max(pa_range * 0.15, 0.002)
        margin_pc = max(pc_range * 0.15, 0.002)
        ax.set_xlim(pc_vals.min() - margin_pc, pc_vals.max() + margin_pc)
        ax.set_ylim(pa_vals.min() - margin_pa, pa_vals.max() + margin_pa)

        _add_balanced_score_lines(ax, is_nap=True)
        ax.set_title(FAMILY_DISPLAY.get(family, family), fontsize=fs_subtitle, fontweight="bold")
        ax.tick_params(axis="both", labelsize=fs_tick)

    # Hide unused subplots
    for idx in range(n_families, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Shared axis labels via fig.text
    fig.text(0.5, 0.02, "PC Mean NAP", ha="center", fontsize=fs_axis, fontweight="bold")
    fig.text(0.02, 0.5, "PA Mean NAP", va="center", rotation="vertical", fontsize=fs_axis, fontweight="bold")

    metric_label = {"balanced": "PA*PC", "nap_balanced": "NAP balanced", "pc": "PC %", "pc_nap": "PC mean NAP"}.get(best_metric, best_metric)
    title_suffix = {"zstd_reference": " (zstd-pinned)", "best_any_codec": " (best-any-codec)", "best_avg_codec": " (best-avg-codec)"}.get(best_selection, "")
    fig.suptitle(f"NAP PA vs PC per Model Family — Best by {metric_label}{title_suffix}", fontsize=fs_title, fontweight="bold", y=1.01)
    fig.subplots_adjust(left=0.08, bottom=0.06)
    plt.tight_layout(rect=[0.05, 0.04, 1, 0.98])
    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec", "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")
    fname = f"sweep_nap_pa_vs_pc_per_model{_sel_suffix}.png"
    plt.savefig(output_dir / fname, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / fname}")


def generate_nap_pa_vs_pc_combined(pdf, output_dir: Path, model_colors: dict,
                                    models: list, best_idx: dict, family_configs: dict | None,
                                    best_metric: str = "balanced",
                                    best_selection: str = "per_codec",
                                    show_all_points: bool = False):
    """Combined figure: clean overview (left 50%) + 3x2 per-family grid (right 50%).

    Layout uses GridSpec with 3 rows x 4 cols (width_ratios 1:1:1:1).
    Left half (cols 0-1): overview spanning all 3 rows.
    Right half (cols 2-3): 3 rows x 2 cols of per-family subplots (6 slots).
    cell_count and dinov2_random are merged into a single subplot.

    When show_all_points=True, the overview also draws a faint background
    cloud of every sweep result (per-family colored), and the filename
    gets a "_with_all_points" suffix.
    """
    import matplotlib.gridspec as gridspec
    output_dir.mkdir(parents=True, exist_ok=True)

    if "PA_mean_nap" not in pdf.columns or "PC_mean_nap" not in pdf.columns:
        print("NAP metrics not available, skipping combined NAP PA vs PC plot.")
        return

    import pandas as pd
    best_rows = [pdf.loc[bi] for bi in best_idx.values()]
    if not best_rows:
        print("No data for combined NAP PA vs PC plot.")
        return
    best = pd.DataFrame(best_rows)
    best["comp_level"] = best["display_name"].apply(_get_compression_level)
    _max_level = max(max(COMPRESSION_LEVEL.values()), 1)
    best["dot_size"] = best["comp_level"].apply(
        lambda lvl: 30 + 950 * np.exp(-2.5 * lvl / _max_level)
    )
    best["family"] = best["model"].apply(_get_model_family)
    best = best.dropna(subset=["PA_mean_nap", "PC_mean_nap"])
    if best.empty:
        print("No valid NAP data for combined plot.")
        return

    # Families that get merged into one subplot
    _MERGED_FAMILIES = {"cell_count", "dinov2_random", "cell_count_lite", "dinov2_random_lite"}
    _MERGED_TITLE = "CellCount + ViT-rand"

    # Build detail slot list: each entry is (title, list_of_families), sorted alphabetically
    detail_slots: list[tuple[str, list[str]]] = []
    merged_added = False
    for fam in MODEL_FAMILIES:
        if fam not in best["family"].values:
            continue
        if fam in _MERGED_FAMILIES:
            if not merged_added:
                detail_slots.append((_MERGED_TITLE, sorted(_MERGED_FAMILIES & set(best["family"].values))))
                merged_added = True
        else:
            detail_slots.append((FAMILY_DISPLAY.get(fam, fam), [fam]))
    detail_slots.sort(key=lambda x: x[0].lower())

    n_detail_rows = 3
    n_detail_cols = 2
    max_slots = n_detail_rows * n_detail_cols

    # --- Layout: 3 rows x 4 cols, width 50/25/25 ---
    fig_w, fig_h = 24, 16
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        n_detail_rows, 4, figure=fig,
        width_ratios=[1, 1, 1, 1],
        hspace=0.35, wspace=0.35,
    )

    # Font sizes — consistent across all panels
    fs_title = 32
    fs_subtitle = 24
    fs_axis = 24
    fs_tick = 18
    fs_annot = 14
    fs_legend = 17

    # ---- Left half: Clean overview (cols 0-1, all rows) ----
    ax_ov = fig.add_subplot(gs[:, 0:2])

    # Optional background cloud: every sweep result, colored by family
    pdf_all = None
    if show_all_points:
        pdf_all = pdf.dropna(subset=["PA_mean_nap", "PC_mean_nap"]).copy()
        pdf_all["family"] = pdf_all["model"].apply(_get_model_family)
        pdf_all = pdf_all[pdf_all["family"] != "unknown"]
        for fam, fam_df in pdf_all.groupby("family"):
            color = FAMILY_SET2_COLOR.get(fam, (0.5, 0.5, 0.5))
            ax_ov.scatter(
                fam_df["PC_mean_nap"], fam_df["PA_mean_nap"],
                c=[color], s=60, alpha=0.42,
                edgecolors="none", zorder=2,
            )

    families_seen = set()
    for _, row in best.iterrows():
        pa_nap, pc_nap = row["PA_mean_nap"], row["PC_mean_nap"]
        family = row["family"]
        color = FAMILY_SET2_COLOR.get(family, (0.5, 0.5, 0.5))
        label = FAMILY_DISPLAY.get(family, family) if family not in families_seen else None
        ax_ov.scatter(
            pc_nap, pa_nap,
            c=[color], s=200, alpha=0.85,
            edgecolors="black", linewidths=0.8, zorder=5,
            label=label,
        )
        families_seen.add(family)

    nap_pa_vals = best["PA_mean_nap"]
    nap_pc_vals = best["PC_mean_nap"]
    pc_max = nap_pc_vals.max() if len(nap_pc_vals) > 0 else 0
    pa_max = nap_pa_vals.max() if len(nap_pa_vals) > 0 else 0
    if show_all_points and pdf_all is not None and len(pdf_all) > 0:
        pc_max = max(pc_max, pdf_all["PC_mean_nap"].max())
        pa_max = max(pa_max, pdf_all["PA_mean_nap"].max())
    ax_ov.set_xlim(0, pc_max * 1.15 if pc_max > 0 else 0.15)
    ax_ov.set_ylim(0, pa_max * 1.15 if pa_max > 0 else 0.5)
    _add_balanced_score_lines(ax_ov, is_nap=True)
    ax_ov.set_xlabel("PC Mean NAP", fontsize=fs_axis, fontweight="bold")
    ax_ov.set_ylabel("PA Mean NAP", fontsize=fs_axis, fontweight="bold")
    ax_ov.tick_params(axis="both", labelsize=fs_tick)
    metric_label = {"balanced": "PA*PC", "nap_balanced": "NAP balanced", "pc": "PC %", "pc_nap": "PC mean NAP"}.get(best_metric, best_metric)
    title_suffix = {"zstd_reference": " (zstd-pinned)", "best_any_codec": " (best-any-codec)", "best_avg_codec": " (best-avg-codec)"}.get(best_selection, "")
    ax_ov.set_title("Mean NAP: PA vs PC", fontsize=fs_title, fontweight="bold")
    ax_ov.spines["top"].set_visible(False)
    ax_ov.spines["right"].set_visible(False)
    ax_ov.legend(loc="lower right", fontsize=fs_legend, framealpha=0.9, edgecolor="gray")

    # Panel label for overview
    ax_ov.text(-0.02, 1.02, "a", transform=ax_ov.transAxes,
               fontsize=fs_title + 4, fontweight="bold", va="bottom", ha="right")

    # ---- Right half: 3x2 per-family detail subplots (cols 2-3) ----
    _panel_labels = "bcdefghijklmnop"
    for idx, (slot_title, slot_families) in enumerate(detail_slots[:max_slots]):
        r, c = divmod(idx, n_detail_cols)
        ax = fig.add_subplot(gs[r, 2 + c])

        fam_data = best[best["family"].isin(slot_families)]
        if fam_data.empty:
            ax.set_visible(False)
            continue

        for _, row in fam_data.iterrows():
            color = FAMILY_SET2_COLOR.get(row["family"], (0.5, 0.5, 0.5))
            ax.scatter(
                row["PC_mean_nap"], row["PA_mean_nap"],
                c=[color], s=row["dot_size"], alpha=0.85,
                edgecolors="black", linewidths=0.8, zorder=5,
            )
            ax.annotate(
                _get_codec_label(row["display_name"]),
                (row["PC_mean_nap"], row["PA_mean_nap"]),
                fontsize=fs_annot, alpha=0.8,
                xytext=(16, 4), textcoords="offset points",
            )

        pa_vals = fam_data["PA_mean_nap"]
        pc_vals = fam_data["PC_mean_nap"]
        pa_range = pa_vals.max() - pa_vals.min() if len(pa_vals) > 1 else pa_vals.max() * 0.1
        pc_range = pc_vals.max() - pc_vals.min() if len(pc_vals) > 1 else pc_vals.max() * 0.1
        margin_pa = max(pa_range * 0.15, 0.002)
        margin_pc = max(pc_range * 0.15, 0.002)
        ax.set_xlim(pc_vals.min() - margin_pc, pc_vals.max() + margin_pc)
        ax.set_ylim(pa_vals.min() - margin_pa, pa_vals.max() + margin_pa)

        _add_balanced_score_lines(ax, is_nap=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_title(slot_title, fontsize=fs_subtitle, fontweight="bold")
        ax.tick_params(axis="both", labelsize=fs_tick)
        # Panel label
        ax.text(-0.02, 1.05, _panel_labels[idx], transform=ax.transAxes,
                fontsize=fs_subtitle + 8, fontweight="bold", va="bottom", ha="right")

        # Only put axis labels on edge subplots
        if r == n_detail_rows - 1:
            ax.set_xlabel("PC Mean NAP", fontsize=fs_axis, fontweight="bold")
        if c == 0:
            ax.set_ylabel("PA Mean NAP", fontsize=fs_axis, fontweight="bold")

    # Hide unused detail subplots
    for idx in range(len(detail_slots), max_slots):
        r, c = divmod(idx, n_detail_cols)
        fig.add_subplot(gs[r, 2 + c]).set_visible(False)

    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec", "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")
    _all_suffix = "_with_all_points" if show_all_points else ""
    fname = f"sweep_nap_pa_vs_pc_combined{_sel_suffix}{_all_suffix}.png"
    plt.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / fname}")


def generate_nap_pa_vs_pc_panel_a(pdf, output_dir: Path, model_colors: dict,
                                  models: list, best_idx: dict, family_configs: dict | None,
                                  best_metric: str = "balanced",
                                  best_selection: str = "per_codec",
                                  show_all_points: bool = False,
                                  codec_filter: list[str] | None = None,
                                  filename_suffix: str = ""):
    """Panel A of sweep_nap_pa_vs_pc_combined, rendered standalone.

    Uses the identical figure size + GridSpec as generate_nap_pa_vs_pc_combined
    and only adds the left-half subplot (gs[:, 0:2]). bbox_inches='tight' crops
    away the unused right half on save, preserving panel A's exact geometry.

    ``codec_filter`` (if set) restricts both the best-per-codec dots and the
    optional background cloud to the listed codec labels (e.g. ``["raw", "mq"]``).
    ``filename_suffix`` is appended to the saved PNG name to distinguish variants.
    """
    import matplotlib.gridspec as gridspec
    output_dir.mkdir(parents=True, exist_ok=True)

    if "PA_mean_nap" not in pdf.columns or "PC_mean_nap" not in pdf.columns:
        print("NAP metrics not available, skipping panel A.")
        return

    import pandas as pd
    best_rows = [pdf.loc[bi] for bi in best_idx.values()]
    if not best_rows:
        print("No data for panel A.")
        return
    best = pd.DataFrame(best_rows)
    best["comp_level"] = best["display_name"].apply(_get_compression_level)
    _max_level = max(max(COMPRESSION_LEVEL.values()), 1)
    best["dot_size"] = best["comp_level"].apply(
        lambda lvl: 30 + 950 * np.exp(-2.5 * lvl / _max_level)
    )
    best["family"] = best["model"].apply(_get_model_family)
    best["codec_label"] = best["display_name"].apply(_get_codec_label)
    best = best.dropna(subset=["PA_mean_nap", "PC_mean_nap"])
    if codec_filter is not None:
        keep = set(codec_filter)
        before = len(best)
        best = best[best["codec_label"].isin(keep)]
        print(f"Panel A codec filter {sorted(keep)}: {before} -> {len(best)} rows")
    if best.empty:
        print("No valid NAP data for panel A.")
        return

    n_detail_rows = 3
    fig_w, fig_h = 24, 16
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = gridspec.GridSpec(
        n_detail_rows, 4, figure=fig,
        width_ratios=[1, 1, 1, 1],
        hspace=0.35, wspace=0.35,
    )

    fs_title = 32
    fs_axis = 24
    fs_tick = 18
    fs_legend = 17

    ax_ov = fig.add_subplot(gs[:, 0:2])

    pdf_all = None
    if show_all_points:
        pdf_all = pdf.dropna(subset=["PA_mean_nap", "PC_mean_nap"]).copy()
        pdf_all["family"] = pdf_all["model"].apply(_get_model_family)
        pdf_all = pdf_all[pdf_all["family"] != "unknown"]
        if codec_filter is not None:
            pdf_all["codec_label"] = pdf_all["display_name"].apply(_get_codec_label)
            pdf_all = pdf_all[pdf_all["codec_label"].isin(set(codec_filter))]
        for fam, fam_df in pdf_all.groupby("family"):
            color = FAMILY_SET2_COLOR.get(fam, (0.5, 0.5, 0.5))
            ax_ov.scatter(
                fam_df["PC_mean_nap"], fam_df["PA_mean_nap"],
                c=[color], s=60, alpha=0.42,
                edgecolors="none", zorder=2,
            )

    families_seen: set = set()
    # codec_label -> comp_level for codecs actually drawn (drives the size
    # legend). Dedupe by short codec label (e.g. "raw", "hq", "d20") rather
    # than the full per-family display_name so each codec appears once.
    codecs_seen: dict[str, int] = {}
    for _, row in best.iterrows():
        pa_nap, pc_nap = row["PA_mean_nap"], row["PC_mean_nap"]
        family = row["family"]
        color = FAMILY_SET2_COLOR.get(family, (0.5, 0.5, 0.5))
        ax_ov.scatter(
            pc_nap, pa_nap,
            c=[color], s=row["dot_size"], alpha=0.85,
            edgecolors="black", linewidths=0.8, zorder=5,
        )
        families_seen.add(family)
        codec_label = _get_codec_label(row["display_name"])
        if codec_label and codec_label not in codecs_seen:
            codecs_seen[codec_label] = int(row["comp_level"])

    nap_pa_vals = best["PA_mean_nap"]
    nap_pc_vals = best["PC_mean_nap"]
    pc_max = nap_pc_vals.max() if len(nap_pc_vals) > 0 else 0
    pa_max = nap_pa_vals.max() if len(nap_pa_vals) > 0 else 0
    if show_all_points and pdf_all is not None and len(pdf_all) > 0:
        pc_max = max(pc_max, pdf_all["PC_mean_nap"].max())
        pa_max = max(pa_max, pdf_all["PA_mean_nap"].max())
    ax_ov.set_xlim(0, pc_max * 1.15 if pc_max > 0 else 0.15)
    ax_ov.set_ylim(0, pa_max * 1.15 if pa_max > 0 else 0.5)
    _add_balanced_score_lines(ax_ov, is_nap=True)
    ax_ov.set_xlabel("PC Mean NAP", fontsize=fs_axis, fontweight="bold")
    ax_ov.set_ylabel("PA Mean NAP", fontsize=fs_axis, fontweight="bold")
    ax_ov.tick_params(axis="both", labelsize=fs_tick)
    ax_ov.set_title("Mean NAP: PA vs PC", fontsize=fs_title, fontweight="bold")
    ax_ov.spines["top"].set_visible(False)
    ax_ov.spines["right"].set_visible(False)
    # Two side-by-side legends at lower right: family color key + codec→size
    # key. Both use empty-scatter handles so the legend markers inherit the
    # exact styling/sizes of the actual scatter dots.
    _fam_canonical_order = list(MODEL_FAMILIES.keys())
    _fam_order = sorted(
        families_seen,
        key=lambda f: _fam_canonical_order.index(f) if f in _fam_canonical_order else 999,
    )
    _fam_handles = [
        ax_ov.scatter([], [], c=[FAMILY_SET2_COLOR.get(f, (0.5, 0.5, 0.5))],
                      s=200, alpha=0.85, edgecolors="black", linewidths=0.8)
        for f in _fam_order
    ]
    _fam_labels = [FAMILY_DISPLAY.get(f, f) for f in _fam_order]
    fam_leg = ax_ov.legend(handles=_fam_handles, labels=_fam_labels,
                           loc="lower right",
                           bbox_to_anchor=(0.98, 0.02),
                           bbox_transform=ax_ov.transAxes,
                           fontsize=fs_legend,
                           framealpha=0.9, edgecolor="gray",
                           title="family", title_fontsize=fs_legend,
                           labelspacing=1.0, borderpad=0.8)
    fam_leg.get_title().set_fontweight("bold")
    ax_ov.add_artist(fam_leg)

    if codecs_seen:
        # Codec legend handles via empty scatters with s=dot_size, so legend
        # marker size matches the actual scatter dot size for each codec.
        _max_lvl = max(max(COMPRESSION_LEVEL.values()), 1)
        _codec_items = sorted(codecs_seen.items(), key=lambda kv: kv[1])
        _codec_handles = []
        _codec_labels = []
        for _label, _lvl in _codec_items:
            _dot_s = 30 + 950 * float(np.exp(-2.5 * _lvl / _max_lvl))
            _codec_handles.append(ax_ov.scatter(
                [], [], c=["lightgray"], s=_dot_s, alpha=0.85,
                edgecolors="black", linewidths=0.8,
            ))
            _codec_labels.append(_label)
        codec_leg = ax_ov.legend(handles=_codec_handles, labels=_codec_labels,
                                 loc="lower right",
                                 bbox_to_anchor=(0.62, 0.02),
                                 bbox_transform=ax_ov.transAxes,
                                 fontsize=fs_legend,
                                 framealpha=0.9, edgecolor="gray",
                                 title="codec", title_fontsize=fs_legend,
                                 labelspacing=1.8, borderpad=0.8)
        codec_leg.get_title().set_fontweight("bold")

    ax_ov.text(-0.02, 1.02, "a", transform=ax_ov.transAxes,
               fontsize=fs_title + 4, fontweight="bold", va="bottom", ha="right")

    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec",
                   "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")
    _all_suffix = "_with_all_points" if show_all_points else ""
    fname = f"sweep_nap_pa_vs_pc_panel_a{_sel_suffix}{filename_suffix}{_all_suffix}.png"
    plt.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / fname}")


def generate_per_group_plot(pdf, output_dir: Path, model_colors: dict,
                            models: list, best_idx: dict,
                            best_metric: str = "balanced",
                            best_selection: str = "per_codec"):
    """Per-group PA and PC breakdown for the best config of each model.

    Shows bar charts of pct_active for each group (orf, crispr, high, low)
    for both PA and PC metrics, plus a scatter of per-group PA vs PC.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    best_rows = [pdf.loc[bi] for bi in best_idx.values()]
    if not best_rows:
        print("No data for per-group plot.")
        return

    import pandas as pd
    best = pd.DataFrame(best_rows)

    # Discover which groups are available
    pa_groups = sorted({
        col.replace("PA_", "").replace("_pct_active", "")
        for col in best.columns
        if col.startswith("PA_group_") and col.endswith("_pct_active")
    })
    pc_groups = sorted({
        col.replace("PC_", "").replace("_pct_active", "")
        for col in best.columns
        if col.startswith("PC_group_") and col.endswith("_pct_active")
        and not col.startswith("PC_rep_")
    })
    pc_rep_groups = sorted({
        col.replace("PC_rep_", "").replace("_pct_active", "")
        for col in best.columns
        if col.startswith("PC_rep_") and col.endswith("_pct_active")
    })

    all_groups = sorted(set(pa_groups) | set(pc_groups) | set(pc_rep_groups))
    if not all_groups:
        print("No per-group metrics available, skipping per-group plot.")
        return

    GROUP_DISPLAY = {
        "group_crispr": "CRISPR",
        "group_orf": "ORF",
        "group_high": "Compounds (high)",
        "group_low": "Compounds (low)",
    }

    # --- Figure 1: Grouped bar chart of PA pct_active per group ---
    n_models = len(best)
    n_groups = len(pa_groups)
    if n_groups > 0 and n_models > 0:
        fig, ax = plt.subplots(figsize=(max(12, n_models * 1.5), 8))
        x = np.arange(n_models)
        width = 0.8 / n_groups
        group_colors = plt.cm.Set2(np.linspace(0, 1, max(n_groups, 3)))

        for i, grp in enumerate(pa_groups):
            col = f"PA_{grp}_pct_active"
            vals = best[col].fillna(0).values if col in best.columns else np.zeros(n_models)
            ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                   label=GROUP_DISPLAY.get(grp, grp), color=group_colors[i],
                   edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([get_display_name(m) for m in best["model"]], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("PA pct_active (%)", fontsize=12, fontweight="bold")
        ax.set_title("Phenotypic Activity by Group (best config per model)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(output_dir / "sweep_pa_per_group.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_dir / 'sweep_pa_per_group.png'}")

    # --- Figure 2: Grouped bar chart of PC pct_active per group ---
    if len(pc_groups) > 0 and n_models > 0:
        fig, ax = plt.subplots(figsize=(max(12, n_models * 1.5), 8))
        x = np.arange(n_models)
        n_pc = len(pc_groups)
        width = 0.8 / n_pc
        group_colors = plt.cm.Set2(np.linspace(0, 1, max(n_pc, 3)))

        for i, grp in enumerate(pc_groups):
            col = f"PC_{grp}_pct_active"
            vals = best[col].fillna(0).values if col in best.columns else np.zeros(n_models)
            ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                   label=GROUP_DISPLAY.get(grp, grp), color=group_colors[i],
                   edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([get_display_name(m) for m in best["model"]], rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("PC pct_active (%)", fontsize=12, fontweight="bold")
        ax.set_title("Phenotypic Consistency by Group (best config per model)", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(output_dir / "sweep_pc_per_group.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_dir / 'sweep_pc_per_group.png'}")

    # --- Figure 3: Per-group PA vs PC scatter (one point per model×group) ---
    scatter_data = []
    for _, row in best.iterrows():
        model = row["model"]
        for grp in all_groups:
            pa_col = f"PA_{grp}_pct_active"
            pc_col = f"PC_{grp}_pct_active"
            pa_val = row.get(pa_col)
            pc_val = row.get(pc_col)
            if pa_val is not None and pc_val is not None and not (np.isnan(pa_val) or np.isnan(pc_val)):
                scatter_data.append({
                    "model": model,
                    "group": GROUP_DISPLAY.get(grp, grp),
                    "PA": pa_val,
                    "PC": pc_val,
                })

    if scatter_data:
        sdf = pd.DataFrame(scatter_data)
        fig, ax = plt.subplots(figsize=(14, 12))
        group_markers = {"CRISPR": "^", "ORF": "s", "Compounds (high)": "D", "Compounds (low)": "o"}

        for _, row in sdf.iterrows():
            model = row["model"]
            color = model_colors.get(model, (0.5, 0.5, 0.5))
            marker = group_markers.get(row["group"], "o")
            ax.scatter(row["PC"], row["PA"], c=[color], s=120, alpha=0.8,
                       edgecolors="black", linewidths=0.5, marker=marker, zorder=5)

        # Build legend: model colors + group markers
        from matplotlib.lines import Line2D
        model_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=model_colors.get(m, (0.5, 0.5, 0.5)),
                   markersize=10, label=get_display_name(m))
            for m in models if m in sdf["model"].values
        ]
        group_handles = [
            Line2D([0], [0], marker=group_markers.get(g, "o"), color="w", markerfacecolor="gray",
                   markersize=10, label=g, markeredgecolor="black", markeredgewidth=0.5)
            for g in sdf["group"].unique()
        ]
        ax.legend(handles=model_handles + group_handles, fontsize=9, ncol=2,
                  loc="upper left", framealpha=0.9)

        ax.set_xlabel("PC pct_active (%)", fontsize=14, fontweight="bold")
        ax.set_ylabel("PA pct_active (%)", fontsize=14, fontweight="bold")
        ax.set_title("PA vs PC by Group (best config per model)", fontsize=16, fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "sweep_pa_vs_pc_per_group.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_dir / 'sweep_pa_vs_pc_per_group.png'}")


def generate_best_per_model_plot(pdf, output_dir: Path, model_colors: dict, models: list):
    """Bar chart of best PA and PC per compression codec."""
    output_dir.mkdir(parents=True, exist_ok=True)
    display_names = [get_display_name(m) for m in models]
    colors = [model_colors[m] for m in models]

    best_pa = []
    best_pc = []
    for model in models:
        mdf = pdf[pdf["model"] == model]
        best_pa.append(mdf["PA"].max() if len(mdf) > 0 else 0)
        best_pc.append(mdf["PC"].max() if len(mdf) > 0 else 0)

    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(max(20, len(models) * 0.6), 8))

    axes[0].bar(x, best_pa, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(display_names, rotation=60, ha="right", fontsize=8)
    axes[0].set_ylabel("Best PA (%)", fontsize=12, fontweight="bold")
    axes[0].set_title("Best Phenotypic Activity", fontsize=14, fontweight="bold")
    axes[0].grid(True, alpha=0.3, axis="y")

    axes[1].bar(x, best_pc, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(display_names, rotation=60, ha="right", fontsize=8)
    axes[1].set_ylabel("Best PC (%)", fontsize=12, fontweight="bold")
    axes[1].set_title("Best Phenotypic Consistency", fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "sweep_best_per_model.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'sweep_best_per_model.png'}")


def generate_best_mean_nap_plot(pdf, output_dir: Path, model_colors: dict, models: list,
                                best_metric: str = "balanced",
                                best_selection: str = "per_codec"):
    """3-panel bar chart of NAP PA, NAP PC, and NAP balanced for best config per model."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if "PA_mean_nap" not in pdf.columns or "PC_mean_nap" not in pdf.columns:
        print("Warning: PA_mean_nap / PC_mean_nap columns not found, skipping best_mean_nap plot")
        return

    # This plot always uses nap_balanced for selection, which may differ from main best_metric
    pdf_nap = pdf.copy()
    pdf_nap, best_col = _add_best_column(pdf_nap, "nap_balanced")

    display_names = [get_display_name(m) for m in models]
    colors = [model_colors[m] for m in models]

    best_idx_map = _compute_best_idx(pdf_nap, models, best_col, best_selection, best_metric)

    nap_pa_vals, nap_pc_vals, nap_bal_vals = [], [], []
    for model in models:
        bi = best_idx_map.get(model)
        if bi is None:
            nap_pa_vals.append(0)
            nap_pc_vals.append(0)
            nap_bal_vals.append(0)
            continue
        pa = pdf_nap.loc[bi, "PA_mean_nap"]
        pc = pdf_nap.loc[bi, "PC_mean_nap"]
        nap_pa_vals.append(pa)
        nap_pc_vals.append(pc)
        nap_bal_vals.append(pa * pc)

    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 3, figsize=(max(30, len(models) * 0.9), 8))

    title_suffix = {"zstd_reference": " (zstd-pinned)", "best_any_codec": " (best-any-codec)", "best_avg_codec": " (best-avg-codec)"}.get(best_selection, "")
    for ax, vals, ylabel, title in [
        (axes[0], nap_pa_vals, "NAP PA", f"Best NAP Phenotypic Activity{title_suffix}"),
        (axes[1], nap_pc_vals, "NAP PC", f"Best NAP Phenotypic Consistency{title_suffix}"),
        (axes[2], nap_bal_vals, "NAP PA * NAP PC", f"Best NAP Balanced{title_suffix}"),
    ]:
        ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(display_names, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec", "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")
    fname = f"sweep_best_mean_nap{_sel_suffix}.png"
    plt.savefig(output_dir / fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / fname}")


def generate_filtered_vs_raw_plot(pdf, output_dir: Path):
    """Compare filtered vs raw (unfiltered) features for each compression codec."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Classify as filtered or raw
    pdf["is_filtered"] = pdf["model"].str.contains("filtered")

    # Extract base codec (without raw/filtered suffix)
    def base_codec(m):
        return m.replace("_filtered_raw", "_RAW").replace("_raw", "").replace("_RAW", "_raw")

    pdf["base_codec"] = pdf["model"].apply(base_codec)

    codecs = sorted(pdf["base_codec"].unique())
    # Only keep codecs that have both filtered and raw
    paired = [c for c in codecs if
              (pdf[(pdf["base_codec"] == c) & (pdf["is_filtered"])].shape[0] > 0 and
               pdf[(pdf["base_codec"] == c) & (~pdf["is_filtered"])].shape[0] > 0)]

    if not paired:
        print("No paired filtered/raw codecs found, skipping comparison plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for ax, (metric, title) in zip(axes, [("PA", "PA (%)"), ("PC", "PC (%)"), ("balanced_score", "Balanced Score")]):
        plot_data = pdf[pdf["base_codec"].isin(paired)].copy()
        plot_data["filter_label"] = plot_data["is_filtered"].map({True: "Filtered", False: "Raw"})

        sns.boxenplot(data=plot_data, x="base_codec", y=metric, hue="filter_label",
                      palette={"Raw": "#4daf4a", "Filtered": "#377eb8"},
                      ax=ax, k_depth="tukey", linewidth=1)
        ax.set_xlabel("Codec", fontsize=12, fontweight="bold")
        ax.set_ylabel(title, fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=10)

    plt.suptitle("Filtered vs Raw Features", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "sweep_filtered_vs_raw.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'sweep_filtered_vs_raw.png'}")


def generate_degenerate_report(pdf_unfiltered, output_dir: Path):
    """Flag potentially degenerate configs (spherize+noPCA, high PA + low PC)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf = pdf_unfiltered

    # Flag degenerate: spherize + no PCA, or PA > 90 and PC < 5, or PC1 variance > 0.1
    degenerate = pdf[
        ((pdf["batch_method"] == "spherize") & (pdf["use_pca"] == False))
        | ((pdf["PA"] > 90) & (pdf["PC"] < 5))
        | (pdf["PC1_variance"] > 0.1)
    ].copy()

    if len(degenerate) == 0:
        print("No degenerate configs found.")
        return

    degenerate = degenerate.sort_values("PA", ascending=False)
    cols = ["model", "config", "PA", "PC", "PC1_variance", "n_features",
            "batch_method", "use_pca", "spherize_fit", "spherize_epsilon"]
    cols = [c for c in cols if c in degenerate.columns]
    degenerate[cols].to_csv(output_dir / "degenerate_configs.csv", index=False)
    print(f"\nWARNING: {len(degenerate)} potentially degenerate configs detected!")
    print(f"Saved: {output_dir / 'degenerate_configs.csv'}")
    print(degenerate[cols].head(20).to_string(index=False))


def generate_codec_delta_from_raw_plot(pdf, output_dir: Path, model_colors: dict,
                                       models: list, best_idx: dict, best_col: str,
                                       best_metric: str = "balanced",
                                       best_selection: str = "per_codec"):
    """Performance delta from baseline (raw/zstd) codec per normalization pipeline.

    For each model family that contains a 'raw' or 'zstd' codec, compute the
    per-config delta (codec_value − baseline_value) for PA mean NAP, PC mean
    NAP, and NAP balanced.  Show individual config deltas as dots and mean
    delta as a bar.
    """
    import pandas as pd
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)

    # Identify families that have a raw codec
    family_models: dict[str, list[str]] = {}
    for m in models:
        fam = _get_model_family(m)
        if fam not in family_models:
            family_models[fam] = []
        family_models[fam].append(m)

    # Find families with a "raw" or "zstd" baseline codec entry
    families_with_raw: dict[str, tuple[str, list[str]]] = {}  # fam -> (baseline_model, other_models)
    for fam, mlist in family_models.items():
        baseline_model = None
        others = []
        for m in mlist:
            codec = _get_codec_label(get_display_name(m))
            if codec in ("raw", "zstd") and baseline_model is None:
                baseline_model = m
            else:
                others.append(m)
        if baseline_model is not None and others:
            families_with_raw[fam] = (baseline_model, others)

    if not families_with_raw:
        print("No families with a 'raw' or 'zstd' baseline found, skipping codec delta plot.")
        return

    # Metrics to show
    metrics = [
        ("PA Mean NAP", "PA_mean_nap"),
        ("PC Mean NAP", "PC_mean_nap"),
        ("NAP Balanced", "nap_balanced"),
    ]
    metrics = [(t, c) for t, c in metrics if c in pdf.columns and not pdf[c].isna().all()]
    if not metrics:
        print("NAP columns not available, skipping codec delta plot.")
        return

    n_metrics = len(metrics)
    n_families = len(families_with_raw)

    # Family ordering
    _FAMILY_PLOT_ORDER = [
        "cell_count", "cell_count_lite",
        "cellprofiler", "cellprofiler_lite", "cp_measure", "cp_measure_filtered", "cp_measure_fbs",
        "dinov2", "dinov2_rr", "dinov2_lite", "dinov2_490",
        "morphem", "morphem_rr", "morphem_lite",
        "openphenom", "openphenom_rr", "openphenom_lite",
        "openphenom_stdscale", "openphenom_nonclip", "openphenom_stdscale_false",
        "openphenom_8clip_std",
        "subcell", "subcell_rr", "subcell_lite", "subcell__clip01", "subcell__clip01_lite",
        "subcell__nonstd", "subcell_nonstd", "subcell_wrongchannels",
        "dinov2_random", "dinov2_random_rr", "dinov2_random_lite",
    ]
    _fam_rank = {f: i for i, f in enumerate(_FAMILY_PLOT_ORDER)}
    fam_order = sorted(families_with_raw.keys(),
                       key=lambda f: _fam_rank.get(f, len(_FAMILY_PLOT_ORDER)))

    # Compute deltas: for each (family, config, metric) → delta per non-raw codec
    delta_records = []
    for fam in fam_order:
        raw_model, others = families_with_raw[fam]
        raw_df = pdf[pdf["model"] == raw_model].copy()
        if raw_df.empty:
            continue
        raw_by_config = raw_df.set_index("config")

        for other_model in others:
            other_df = pdf[pdf["model"] == other_model].copy()
            if other_df.empty:
                continue
            other_by_config = other_df.set_index("config")

            # Intersection of configs
            shared_configs = raw_by_config.index.intersection(other_by_config.index)
            if len(shared_configs) == 0:
                continue

            codec_label = _get_codec_label(get_display_name(other_model))
            for cfg in shared_configs:
                for metric_title, metric_col in metrics:
                    raw_val = raw_by_config.loc[cfg, metric_col]
                    other_val = other_by_config.loc[cfg, metric_col]
                    # Handle duplicate configs (take first if multiple)
                    if hasattr(raw_val, '__len__'):
                        raw_val = raw_val.iloc[0]
                    if hasattr(other_val, '__len__'):
                        other_val = other_val.iloc[0]
                    if np.isnan(raw_val) or np.isnan(other_val):
                        continue
                    # Cap negative metric values to zero
                    raw_val = max(0.0, raw_val)
                    other_val = max(0.0, other_val)
                    delta_records.append({
                        "family": fam,
                        "codec": codec_label,
                        "model": other_model,
                        "config": cfg,
                        "metric": metric_title,
                        "metric_col": metric_col,
                        "delta": other_val - raw_val,
                        "raw_val": raw_val,
                    })

    if not delta_records:
        print("No delta records computed, skipping codec delta plot.")
        return

    delta_df = pd.DataFrame(delta_records)

    # Add percentage delta: (codec - raw) / |raw| * 100
    delta_df["delta_pct"] = np.where(
        delta_df["raw_val"].abs() > 1e-12,
        delta_df["delta"] / delta_df["raw_val"].abs() * 100,
        np.nan,
    )

    # --- Layout ---
    # Build codec order within each family (by _CODEC_SORT_ORDER rank)
    codec_entries = []  # (family, codec_label, model, sort_rank)
    for fam in fam_order:
        _, others = families_with_raw[fam]
        for m in others:
            cl = _get_codec_label(get_display_name(m))
            rank = _get_codec_sort_rank(m)
            codec_entries.append((fam, cl, m, rank))
    # Sort by family order then codec rank
    codec_entries.sort(key=lambda x: (_fam_rank.get(x[0], 999), x[3]))

    # Build y-positions with gaps between families
    GAP = 0.6
    y_positions = {}
    y_ticks = []
    y_tick_labels = []
    family_spans = []  # (fam_name, y_start, y_end)
    cursor = 0.0
    prev_fam = None
    fam_start = 0.0
    for fam, cl, m, _ in codec_entries:
        if prev_fam is not None and fam != prev_fam:
            # Close the previous family span
            family_spans.append((prev_fam, fam_start, cursor - 1.0))
            cursor += GAP
            fam_start = cursor
        y_positions[m] = cursor
        y_ticks.append(cursor)
        y_tick_labels.append(cl)
        prev_fam = fam
        cursor += 1.0
    if prev_fam is not None:
        family_spans.append((prev_fam, fam_start, cursor - 1.0))
    total_height = cursor

    # Per-family colors
    fam_colors = {}
    for fam in fam_order:
        fam_colors[fam] = FAMILY_SET2_COLOR.get(fam, (0.5, 0.5, 0.5))

    # Font sizes
    fs_title = 16
    fs_axis = 13
    fs_tick = 11
    fs_family = 12

    # --- Plot: render delta rows (abs / pct) x n_metrics cols ---
    # After axis swap: codecs on x-axis, delta on y-axis.
    cat_extent_in = max(6, 1.4 * n_families)   # inches per subplot for the codec (x) axis
    delta_extent_in = 6                         # inches per subplot for the delta (y) axis

    _ROW_ABS = (0, "delta", "")
    _ROW_PCT = (1, "delta_pct", " %")
    _ALL_ROWS = [_ROW_ABS, _ROW_PCT]

    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D
    import math

    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec", "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")

    def _split_metric_title(title):
        """Split 'PA CRISPR' \u2192 ('PA', 'CRISPR'); 'NAP Balanced' \u2192 ('', 'NAP Balanced')."""
        if title.startswith("PA "):
            return "PA", title[3:]
        if title.startswith("PC "):
            return "PC", title[3:]
        return "", title

    def _render(rows_to_render, fname_suffix, metric_grid=None):
        """Render the delta figure.

        metric_grid: optional (n_metric_rows, n_metric_cols) tuple. Only valid
        when len(rows_to_render) == 1 (the abs-only / pct-only variants).
        """
        n_rows = len(rows_to_render)

        if metric_grid is not None:
            assert n_rows == 1, "metric_grid only valid for single delta-row variants"
            mr, mc = metric_grid
            assert mr * mc >= n_metrics, f"metric_grid {metric_grid} too small for {n_metrics} metrics"
            fig, all_axes = plt.subplots(mr, mc,
                                         figsize=(cat_extent_in * mc, delta_extent_in * mr),
                                         squeeze=False)
            flat_axes = [all_axes[r, c] for r in range(mr) for c in range(mc)]
        else:
            fig, all_axes = plt.subplots(n_rows, n_metrics,
                                         figsize=(cat_extent_in * n_metrics, delta_extent_in * n_rows),
                                         squeeze=False)

        _panel_labels = "abcdefghijklmnop"
        panel_idx = 0

        for grid_row, (_orig_row_idx, value_col, pct_sfx) in enumerate(rows_to_render):
            is_pct_row = value_col == "delta_pct"
            for col_idx, (metric_title, metric_col) in enumerate(metrics):
                if metric_grid is not None:
                    ax = flat_axes[col_idx]
                else:
                    ax = all_axes[grid_row, col_idx]
                metric_df = delta_df[delta_df["metric"] == metric_title]
                prefix, subset = _split_metric_title(metric_title)

                for fam, cl, m, _ in codec_entries:
                    xpos = y_positions[m]
                    mdf = metric_df[metric_df["model"] == m]
                    vals = mdf[value_col].dropna().values
                    color = fam_colors.get(fam, (0.5, 0.5, 0.5))

                    if len(vals) > 0:
                        x_jitter = np.random.normal(xpos, 0.12, len(vals))
                        ax.scatter(x_jitter, vals, c=[color], s=20, alpha=0.4,
                                   edgecolors="white", linewidths=0.2, zorder=3)
                        mean_val = np.mean(vals)
                        ax.scatter(xpos, mean_val, c=[color], s=120, alpha=1.0,
                                   edgecolors="black", linewidths=0.8, marker="D", zorder=5)
                        ax.plot([xpos, xpos], [0, mean_val], color=color, linewidth=1.5,
                                alpha=0.7, zorder=2)
                        if is_pct_row:
                            sign = "+" if mean_val >= 0 else ""
                            ax.text(xpos, mean_val, f"{sign}{mean_val:.1f}%",
                                    fontsize=8, ha="center",
                                    va="bottom" if mean_val >= 0 else "top",
                                    color="black", fontweight="bold", zorder=6)

                ax.axhline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.6, zorder=1)

                ax.set_xticks(y_ticks)
                ax.set_xticklabels(y_tick_labels, fontsize=fs_tick, rotation=45, ha="right")
                ax.set_xlim(-0.5, total_height - 0.5)
                # Y-label: "\u0394 PA (codec \u2212 baseline)" / "\u0394 PA % (codec \u2212 baseline)"
                # When the metric has no PA/PC prefix, fall back to the full name.
                ylabel_metric = prefix if prefix else subset
                ax.set_ylabel(f"\u0394 {ylabel_metric}{pct_sfx}",
                              fontsize=fs_axis + 3, fontweight="bold")
                # Subplot title: just the data subset (PA/PC moved to y-label).
                # Break multi-word subsets onto two lines for compact panel headers.
                subset_display = subset.replace(" ", "\n", 1) if " " in subset else subset
                ax.set_title(subset_display, fontsize=fs_title, fontweight="bold")
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
                ax.tick_params(axis="y", labelsize=fs_tick + 3)
                ax.grid(True, alpha=0.15, axis="y", linewidth=0.5)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                ax.text(-0.02, 1.05, _panel_labels[panel_idx], transform=ax.transAxes,
                        fontsize=fs_title + 4, fontweight="bold", va="bottom", ha="right")
                panel_idx += 1

        # Hide unused subplots when metrics don't fill the grid
        if metric_grid is not None:
            for i in range(n_metrics, mr * mc):
                flat_axes[i].set_visible(False)

        # Family color legend (replaces the per-subplot family labels)
        legend_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=fam_colors.get(fam, (0.5, 0.5, 0.5)),
                   markeredgecolor="black", markeredgewidth=0.8, markersize=18,
                   label=FAMILY_DISPLAY.get(fam, fam))
            for fam in fam_order
        ]
        # Single-row horizontal legend below the figure (one column per family).
        fig.legend(handles=legend_handles, loc="upper center",
                   bbox_to_anchor=(0.5, 0.0), fontsize=fs_family + 6,
                   frameon=False, ncol=max(1, len(fam_order)),
                   title="Model family", title_fontsize=fs_family + 8,
                   handletextpad=0.5, columnspacing=1.5)

        plt.tight_layout()
        fname = f"codec_delta_from_raw{fname_suffix}{_sel_suffix}.png"
        plt.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_dir / fname}")

    _render(_ALL_ROWS, "")
    # Single-delta-row variants: arrange metrics in 2 rows
    _two_row_grid = (2, math.ceil(n_metrics / 2))
    _render([_ROW_ABS], "_abs_only", metric_grid=_two_row_grid)
    _render([_ROW_PCT], "_pct_only", metric_grid=_two_row_grid)


def generate_codec_delta_balanced_violin(pdf, output_dir: Path, model_colors: dict,
                                          models: list, best_idx: dict, best_col: str,
                                          best_metric: str = "balanced",
                                          best_selection: str = "per_codec"):
    """Violin/boxen plots of NAP Balanced delta vs raw/zstd baseline, per codec.

    Re-renders the last subfigure of `codec_delta_from_raw_abs_only.png`
    (NAP Balanced) in the style of `plot_cell_level_iou.py` /
    `compare_segmentations.py`: codec on the x-axis (canonical sort rank),
    delta on the y-axis, codecs colored by a viridis quality gradient
    (`hue='codec'`, `palette='viridis'`, `legend=False`, matching
    `compare_segmentations.py:1675`).

    Emits violin / violin+p5p95 / boxen variants for both absolute Δ and
    percentage drop Δ%, plus one figure per model family in a subfolder.
    """
    import pandas as pd
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_col = "nap_balanced"
    if metric_col not in pdf.columns or pdf[metric_col].isna().all():
        print(f"{metric_col} not available, skipping balanced delta violin plot.")
        return

    # Identify families with a 'raw' or 'zstd' baseline codec
    family_models: dict[str, list[str]] = {}
    for m in models:
        fam = _get_model_family(m)
        family_models.setdefault(fam, []).append(m)

    families_with_raw: dict[str, tuple[str, list[str]]] = {}
    for fam, mlist in family_models.items():
        baseline_model = None
        others = []
        for m in mlist:
            codec = _get_codec_label(get_display_name(m))
            if codec in ("raw", "zstd") and baseline_model is None:
                baseline_model = m
            else:
                others.append(m)
        if baseline_model is not None and others:
            families_with_raw[fam] = (baseline_model, others)

    if not families_with_raw:
        print("No families with a 'raw' or 'zstd' baseline found, skipping balanced delta violin.")
        return

    # Compute deltas (codec_value - baseline_value) per shared config
    delta_records = []
    for fam, (raw_model, others) in families_with_raw.items():
        raw_df = pdf[pdf["model"] == raw_model]
        if raw_df.empty:
            continue
        raw_by_config = raw_df.set_index("config")
        for other_model in others:
            other_df = pdf[pdf["model"] == other_model]
            if other_df.empty:
                continue
            other_by_config = other_df.set_index("config")
            shared_configs = raw_by_config.index.intersection(other_by_config.index)
            if len(shared_configs) == 0:
                continue
            codec_label = _get_codec_label(get_display_name(other_model))
            for cfg in shared_configs:
                raw_val = raw_by_config.loc[cfg, metric_col]
                other_val = other_by_config.loc[cfg, metric_col]
                if hasattr(raw_val, "__len__"):
                    raw_val = raw_val.iloc[0]
                if hasattr(other_val, "__len__"):
                    other_val = other_val.iloc[0]
                if np.isnan(raw_val) or np.isnan(other_val):
                    continue
                # Cap negative metric values to zero (matches abs-only panel)
                raw_val = max(0.0, raw_val)
                other_val = max(0.0, other_val)
                delta_records.append({
                    "family": fam,
                    "codec": codec_label,
                    "model": other_model,
                    "delta": other_val - raw_val,
                    "raw_val": raw_val,
                })

    if not delta_records:
        print("No delta records computed, skipping balanced delta violin plot.")
        return

    df_plot = pd.DataFrame(delta_records)
    # Signed percent change vs raw (matches the _pct_only panel of
    # codec_delta_from_raw): negative = performance drop, positive = gain.
    df_plot["delta_pct"] = np.where(
        df_plot["raw_val"].abs() > 1e-12,
        df_plot["delta"] / df_plot["raw_val"].abs() * 100,
        np.nan,
    )
    # Codec performance as a percentage of raw baseline. 100% = equal,
    # <100% = degraded, >100% = improved. Reference line drawn at 100.
    df_plot["pct_of_raw"] = np.where(
        df_plot["raw_val"].abs() > 1e-12,
        (df_plot["raw_val"] + df_plot["delta"]) / df_plot["raw_val"] * 100,
        np.nan,
    )

    # Codec order: canonical sort rank (lossless first, heavier-lossy later)
    codec_rank: dict[str, int] = {}
    for m in models:
        cl = _get_codec_label(get_display_name(m))
        rank = _get_codec_sort_rank(m)
        if cl not in codec_rank or rank < codec_rank[cl]:
            codec_rank[cl] = rank
    codec_order = sorted(df_plot["codec"].unique(), key=lambda c: codec_rank.get(c, 9999))

    # Family ordering for the per-model loop (canonical order used elsewhere)
    _FAMILY_PLOT_ORDER = [
        "cell_count", "cell_count_lite",
        "cellprofiler", "cellprofiler_lite", "cp_measure", "cp_measure_filtered", "cp_measure_fbs",
        "dinov2", "dinov2_rr", "dinov2_lite", "dinov2_490",
        "morphem", "morphem_rr", "morphem_lite",
        "openphenom", "openphenom_rr", "openphenom_lite",
        "openphenom_stdscale", "openphenom_nonclip", "openphenom_stdscale_false",
        "openphenom_8clip_std",
        "subcell", "subcell_rr", "subcell_lite", "subcell__clip01", "subcell__clip01_lite",
        "subcell__nonstd", "subcell_nonstd", "subcell_wrongchannels",
        "dinov2_random", "dinov2_random_rr", "dinov2_random_lite",
    ]
    _fam_rank = {f: i for i, f in enumerate(_FAMILY_PLOT_ORDER)}
    fam_order_keys = sorted(df_plot["family"].unique(),
                             key=lambda f: _fam_rank.get(f, len(_FAMILY_PLOT_ORDER)))

    # Display labels for codecs (Title Case where we have nice names)
    _known_labels = {
        "lq": "Low", "mq": "Medium", "e3": "Mid-High", "effort_3": "Mid-High", "hq": "High",
        "d2_e8": "D2 E8", "d10": "D10", "d15": "D15",
        "d20_e2": "D20 E2", "d25": "D25", "d30": "D30",
    }
    codec_labels = {c: _known_labels.get(c, c.upper()) for c in codec_order}

    _sel_suffix = {
        "zstd_reference": "_zstd_pinned",
        "best_any_codec": "_best_any_codec",
        "best_avg_codec": "_best_avg_codec",
    }.get(best_selection, "")

    # Match the canvas used by analysis/segmentation/plot_cell_level_iou.py
    # (fixed 7x7 inches for every violin/boxen panel).
    fig_w = 7
    fig_h = 7

    # Three value variants:
    #   - absolute delta (codec − raw), reference at 0
    #   - signed percent change ((codec − raw) / |raw| · 100), reference at 0
    #   - codec performance as % of raw (codec / raw · 100), reference at 100,
    #     y-limited to [0, 115] so the 100% baseline sits near the top.
    VARIANTS = [
        ("delta",      "Δ NAP Balanced",            "",            "{:+.3f}",  0.0,   None),
        ("delta_pct",  "% Change in NAP Balanced",  "_pct_change", "{:+.1f}%", 0.0,   None),
        ("pct_of_raw", "NAP Balanced (% of Raw)",   "_pct_of_raw", "{:.1f}%",  100.0, (0, 160)),
    ]

    per_model_dir = output_dir / "codec_delta_balanced_per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)

    def _safe_name(s: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in s)

    def _violin(ax, data, order):
        sns.violinplot(
            data=data, x="codec", y=value_col,
            hue="codec", order=order, hue_order=order,
            palette="viridis", inner="box", cut=0,
            legend=False, ax=ax,
        )

    def _boxen(ax, data, order):
        sns.boxenplot(
            data=data, x="codec", y=value_col,
            hue="codec", order=order, hue_order=order,
            palette="viridis", legend=False, ax=ax,
        )

    def _decorate(ax, order, title):
        ax.axhline(ref_y, color="black", linewidth=1.0, linestyle="-", alpha=0.6, zorder=1)
        ax.set_xlabel("Compression Quality", fontsize=24, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=24, fontweight="bold")
        ax.set_title(title, fontsize=26, fontweight="bold")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([codec_labels[c] for c in order], fontsize=20,
                           rotation=45, ha="right")
        ax.tick_params(axis="both", labelsize=20)
        if ylim is not None:
            ax.set_ylim(*ylim)

    def _annotate_p5_p95(ax, data, order):
        for i, codec in enumerate(order):
            vals = data.loc[data["codec"] == codec, value_col].dropna().values
            if len(vals) == 0:
                continue
            p5 = np.percentile(vals, 5)
            p95 = np.percentile(vals, 95)
            ax.scatter([i], [p95], marker="_", s=300, linewidths=3,
                       color="black", zorder=10)
            ax.annotate(annot_fmt.format(p95), (i, p95), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=10, fontweight="bold",
                        color="black")
            ax.scatter([i], [p5], marker="_", s=300, linewidths=3,
                       color="black", zorder=10)
            ax.annotate(annot_fmt.format(p5), (i, p5), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=10, fontweight="bold",
                        color="black")

    for value_col, ylabel, file_sfx, annot_fmt, ref_y, ylim in VARIANTS:
        # --- Combined: Violin ---
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        _violin(ax, df_plot, codec_order)
        _decorate(ax, codec_order, ylabel)
        plt.tight_layout()
        out = output_dir / f"codec_delta_balanced_violin{file_sfx}{_sel_suffix}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved combined {ylabel} violin to: {out}")

        # --- Combined: Violin with p5/p95 markers ---
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        _violin(ax, df_plot, codec_order)
        _decorate(ax, codec_order, f"{ylabel} (5th & 95th percentile)")
        _annotate_p5_p95(ax, df_plot, codec_order)
        plt.tight_layout()
        out = output_dir / f"codec_delta_balanced_violin_p5_p95{file_sfx}{_sel_suffix}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved combined {ylabel} violin (p5/p95) to: {out}")

        # --- Combined: Boxen ---
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        _boxen(ax, df_plot, codec_order)
        _decorate(ax, codec_order, ylabel)
        plt.tight_layout()
        out = output_dir / f"codec_delta_balanced_boxen{file_sfx}{_sel_suffix}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved combined {ylabel} boxen to: {out}")

        # --- Per-model figures (one per family, same fig size) ---
        for fam_key in fam_order_keys:
            fam_disp = FAMILY_DISPLAY.get(fam_key, fam_key)
            df_fam = df_plot[df_plot["family"] == fam_key]
            if df_fam.empty:
                continue
            fam_codec_order = [c for c in codec_order if c in set(df_fam["codec"].unique())]
            if not fam_codec_order:
                continue
            slug = _safe_name(fam_key)

            # Violin
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            _violin(ax, df_fam, fam_codec_order)
            _decorate(ax, fam_codec_order, ylabel)
            plt.tight_layout()
            plt.savefig(per_model_dir / f"codec_delta_balanced_violin{file_sfx}__{slug}{_sel_suffix}.png",
                        dpi=150, bbox_inches="tight")
            plt.close()

            # Violin + p5/p95
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            _violin(ax, df_fam, fam_codec_order)
            _decorate(ax, fam_codec_order, f"{ylabel} (5th & 95th percentile)")
            _annotate_p5_p95(ax, df_fam, fam_codec_order)
            plt.tight_layout()
            plt.savefig(per_model_dir / f"codec_delta_balanced_violin_p5_p95{file_sfx}__{slug}{_sel_suffix}.png",
                        dpi=150, bbox_inches="tight")
            plt.close()

            # Boxen
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            _boxen(ax, df_fam, fam_codec_order)
            _decorate(ax, fam_codec_order, ylabel)
            plt.tight_layout()
            plt.savefig(per_model_dir / f"codec_delta_balanced_boxen{file_sfx}__{slug}{_sel_suffix}.png",
                        dpi=150, bbox_inches="tight")
            plt.close()

        print(f"Saved per-model {ylabel} plots → {per_model_dir}/")


def generate_codec_delta_from_raw_groups_plot(pdf, output_dir: Path, model_colors: dict,
                                              models: list, best_idx: dict, best_col: str,
                                              best_metric: str = "balanced",
                                              best_selection: str = "per_codec"):
    """Per-group performance delta from baseline (raw/zstd) codec.

    Same as generate_codec_delta_from_raw_plot but with 6 per-group NAP metrics
    (PA CRISPR, PA ORF, PA Compounds High/Low, PC Compounds High/Low) in a
    2x6 layout (top: absolute delta, bottom: percentage delta).
    """
    import pandas as pd
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)

    # Identify families that have a raw codec
    family_models: dict[str, list[str]] = {}
    for m in models:
        fam = _get_model_family(m)
        if fam not in family_models:
            family_models[fam] = []
        family_models[fam].append(m)

    families_with_raw: dict[str, tuple[str, list[str]]] = {}
    for fam, mlist in family_models.items():
        baseline_model = None
        others = []
        for m in mlist:
            codec = _get_codec_label(get_display_name(m))
            if codec in ("raw", "zstd") and baseline_model is None:
                baseline_model = m
            else:
                others.append(m)
        if baseline_model is not None and others:
            families_with_raw[fam] = (baseline_model, others)

    if not families_with_raw:
        print("No families with a 'raw' or 'zstd' baseline found, skipping codec delta groups plot.")
        return

    # Per-group NAP metrics (6 columns)
    metrics = [
        ("PA CRISPR", "PA_group_crispr_mean_normalized_average_precision"),
        ("PA ORF", "PA_group_orf_mean_normalized_average_precision"),
        ("PA Compound Diversity", "PA_group_high_mean_normalized_average_precision"),
        ("PA Compound Bioactive-library", "PA_group_low_mean_normalized_average_precision"),
        ("PC Compound Diversity", "PC_group_high_mean_normalized_average_precision"),
        ("PC Compound Bioactive-library", "PC_group_low_mean_normalized_average_precision"),
    ]
    metrics = [(t, c) for t, c in metrics if c in pdf.columns and not pdf[c].isna().all()]

    if not metrics:
        print("Per-group NAP columns not available, skipping codec delta groups plot.")
        return

    n_metrics = len(metrics)
    n_families = len(families_with_raw)

    # Family ordering
    _FAMILY_PLOT_ORDER = [
        "cell_count", "cell_count_lite",
        "cellprofiler", "cellprofiler_lite", "cp_measure", "cp_measure_filtered", "cp_measure_fbs",
        "dinov2", "dinov2_rr", "dinov2_lite", "dinov2_490",
        "morphem", "morphem_rr", "morphem_lite",
        "openphenom", "openphenom_rr", "openphenom_lite",
        "openphenom_stdscale", "openphenom_nonclip", "openphenom_stdscale_false",
        "openphenom_8clip_std",
        "subcell", "subcell_rr", "subcell_lite", "subcell__clip01", "subcell__clip01_lite",
        "subcell__nonstd", "subcell_nonstd", "subcell_wrongchannels",
        "dinov2_random", "dinov2_random_rr", "dinov2_random_lite",
    ]
    _fam_rank = {f: i for i, f in enumerate(_FAMILY_PLOT_ORDER)}
    fam_order = sorted(families_with_raw.keys(),
                       key=lambda f: _fam_rank.get(f, len(_FAMILY_PLOT_ORDER)))

    # Compute deltas
    delta_records = []
    for fam in fam_order:
        raw_model, others = families_with_raw[fam]
        raw_df = pdf[pdf["model"] == raw_model].copy()
        if raw_df.empty:
            continue
        raw_by_config = raw_df.set_index("config")

        for other_model in others:
            other_df = pdf[pdf["model"] == other_model].copy()
            if other_df.empty:
                continue
            other_by_config = other_df.set_index("config")

            shared_configs = raw_by_config.index.intersection(other_by_config.index)
            if len(shared_configs) == 0:
                continue

            codec_label = _get_codec_label(get_display_name(other_model))
            for cfg in shared_configs:
                for metric_title, metric_col in metrics:
                    raw_val = raw_by_config.loc[cfg, metric_col]
                    other_val = other_by_config.loc[cfg, metric_col]
                    if hasattr(raw_val, '__len__'):
                        raw_val = raw_val.iloc[0]
                    if hasattr(other_val, '__len__'):
                        other_val = other_val.iloc[0]
                    if np.isnan(raw_val) or np.isnan(other_val):
                        continue
                    # Cap negative metric values to zero
                    raw_val = max(0.0, raw_val)
                    other_val = max(0.0, other_val)
                    delta_records.append({
                        "family": fam,
                        "codec": codec_label,
                        "model": other_model,
                        "config": cfg,
                        "metric": metric_title,
                        "metric_col": metric_col,
                        "delta": other_val - raw_val,
                        "raw_val": raw_val,
                    })

    if not delta_records:
        print("No delta records computed, skipping codec delta groups plot.")
        return

    delta_df = pd.DataFrame(delta_records)
    delta_df["delta_pct"] = np.where(
        delta_df["raw_val"].abs() > 1e-12,
        delta_df["delta"] / delta_df["raw_val"].abs() * 100,
        np.nan,
    )

    # Build codec order within each family
    codec_entries = []
    for fam in fam_order:
        _, others = families_with_raw[fam]
        for m in others:
            cl = _get_codec_label(get_display_name(m))
            rank = _get_codec_sort_rank(m)
            codec_entries.append((fam, cl, m, rank))
    codec_entries.sort(key=lambda x: (_fam_rank.get(x[0], 999), x[3]))

    # Build y-positions with gaps between families
    GAP = 0.6
    y_positions = {}
    y_ticks = []
    y_tick_labels = []
    family_spans = []
    cursor = 0.0
    prev_fam = None
    fam_start = 0.0
    for fam, cl, m, _ in codec_entries:
        if prev_fam is not None and fam != prev_fam:
            family_spans.append((prev_fam, fam_start, cursor - 1.0))
            cursor += GAP
            fam_start = cursor
        y_positions[m] = cursor
        y_ticks.append(cursor)
        y_tick_labels.append(cl)
        prev_fam = fam
        cursor += 1.0
    if prev_fam is not None:
        family_spans.append((prev_fam, fam_start, cursor - 1.0))
    total_height = cursor

    fam_colors = {}
    for fam in fam_order:
        fam_colors[fam] = FAMILY_SET2_COLOR.get(fam, (0.5, 0.5, 0.5))

    # Font sizes (slightly smaller for 6-col layout)
    fs_title = 14
    fs_axis = 11
    fs_tick = 12
    fs_family = 10

    # --- Plot: render delta rows (abs / pct) x n_metrics cols ---
    # After axis swap: codecs on x-axis, delta on y-axis.
    cat_extent_in = max(5, 1.2 * n_families)   # inches per subplot for the codec (x) axis
    delta_extent_in = 4                         # inches per subplot for the delta (y) axis

    _ROW_ABS = (0, "delta", "")
    _ROW_PCT = (1, "delta_pct", " %")
    _ALL_ROWS = [_ROW_ABS, _ROW_PCT]

    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D
    import math

    _sel_suffix = {"zstd_reference": "_zstd_pinned", "best_any_codec": "_best_any_codec", "best_avg_codec": "_best_avg_codec"}.get(best_selection, "")

    def _split_metric_title(title):
        """Split 'PA CRISPR' \u2192 ('PA', 'CRISPR'); 'NAP Balanced' \u2192 ('', 'NAP Balanced')."""
        if title.startswith("PA "):
            return "PA", title[3:]
        if title.startswith("PC "):
            return "PC", title[3:]
        return "", title

    def _render(rows_to_render, fname_suffix, metric_grid=None):
        """Render the per-group delta figure.

        metric_grid: optional (n_metric_rows, n_metric_cols) tuple to wrap the
        n_metrics columns into a grid. Only valid when len(rows_to_render) == 1
        (i.e. abs-only or pct-only variants).
        """
        n_rows = len(rows_to_render)

        if metric_grid is not None:
            assert n_rows == 1, "metric_grid only valid for single delta-row variants"
            mr, mc = metric_grid
            assert mr * mc >= n_metrics, f"metric_grid {metric_grid} too small for {n_metrics} metrics"
            fig, all_axes = plt.subplots(mr, mc,
                                         figsize=(cat_extent_in * mc, delta_extent_in * mr),
                                         squeeze=False)
            flat_axes = [all_axes[r, c] for r in range(mr) for c in range(mc)]
        else:
            fig, all_axes = plt.subplots(n_rows, n_metrics,
                                         figsize=(cat_extent_in * n_metrics, delta_extent_in * n_rows),
                                         squeeze=False)

        _panel_labels = "abcdefghijklmnop"
        panel_idx = 0

        for grid_row, (_orig_row_idx, value_col, pct_sfx) in enumerate(rows_to_render):
            is_pct_row = value_col == "delta_pct"
            for col_idx, (metric_title, metric_col) in enumerate(metrics):
                if metric_grid is not None:
                    ax = flat_axes[col_idx]
                else:
                    ax = all_axes[grid_row, col_idx]
                metric_df = delta_df[delta_df["metric"] == metric_title]
                prefix, subset = _split_metric_title(metric_title)

                for fam, cl, m, _ in codec_entries:
                    xpos = y_positions[m]
                    mdf = metric_df[metric_df["model"] == m]
                    vals = mdf[value_col].dropna().values
                    color = fam_colors.get(fam, (0.5, 0.5, 0.5))

                    if len(vals) > 0:
                        x_jitter = np.random.normal(xpos, 0.12, len(vals))
                        ax.scatter(x_jitter, vals, c=[color], s=15, alpha=0.35,
                                   edgecolors="white", linewidths=0.15, zorder=3)
                        mean_val = np.mean(vals)
                        ax.scatter(xpos, mean_val, c=[color], s=90, alpha=1.0,
                                   edgecolors="black", linewidths=0.6, marker="D", zorder=5)
                        ax.plot([xpos, xpos], [0, mean_val], color=color, linewidth=1.2,
                                alpha=0.7, zorder=2)
                        if is_pct_row:
                            sign = "+" if mean_val >= 0 else ""
                            ax.text(xpos, mean_val, f"{sign}{mean_val:.1f}%",
                                    fontsize=7, ha="center",
                                    va="bottom" if mean_val >= 0 else "top",
                                    color="black", fontweight="bold", zorder=6)

                ax.axhline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.6, zorder=1)

                ax.set_xticks(y_ticks)
                ax.set_xticklabels(y_tick_labels, fontsize=fs_tick, rotation=45, ha="right")
                ax.set_xlim(-0.5, total_height - 0.5)
                # Y-label: "\u0394 PA (codec \u2212 baseline)" / "\u0394 PA % (codec \u2212 baseline)"
                ylabel_metric = prefix if prefix else subset
                ax.set_ylabel(f"\u0394 {ylabel_metric}{pct_sfx}",
                              fontsize=fs_axis + 3, fontweight="bold")
                # Subplot title: just the data subset (PA/PC moved to y-label).
                # Break multi-word subsets onto two lines for compact panel headers.
                subset_display = subset.replace(" ", "\n", 1) if " " in subset else subset
                ax.set_title(subset_display, fontsize=fs_title, fontweight="bold")
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
                ax.tick_params(axis="y", labelsize=fs_tick + 3)
                ax.grid(True, alpha=0.15, axis="y", linewidth=0.5)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                ax.text(-0.02, 1.05, _panel_labels[panel_idx], transform=ax.transAxes,
                        fontsize=fs_title + 4, fontweight="bold", va="bottom", ha="right")
                panel_idx += 1

        # Hide unused subplots when metrics don't fill the grid
        if metric_grid is not None:
            for i in range(n_metrics, mr * mc):
                flat_axes[i].set_visible(False)

        # Family color legend (replaces the per-subplot family labels)
        legend_handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=fam_colors.get(fam, (0.5, 0.5, 0.5)),
                   markeredgecolor="black", markeredgewidth=0.8, markersize=18,
                   label=FAMILY_DISPLAY.get(fam, fam))
            for fam in fam_order
        ]
        # Single-row horizontal legend below the figure (one column per family).
        fig.legend(handles=legend_handles, loc="upper center",
                   bbox_to_anchor=(0.5, 0.0), fontsize=fs_family + 6,
                   frameon=False, ncol=max(1, len(fam_order)),
                   title="Model family", title_fontsize=fs_family + 8,
                   handletextpad=0.5, columnspacing=1.5)

        plt.tight_layout()
        fname = f"codec_delta_from_raw_groups{fname_suffix}{_sel_suffix}.png"
        plt.savefig(output_dir / fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_dir / fname}")

    _render(_ALL_ROWS, "")
    # Single-delta-row variants: arrange metrics in 2 rows
    _two_row_grid = (2, math.ceil(n_metrics / 2))
    _render([_ROW_ABS], "_abs_only", metric_grid=_two_row_grid)
    _render([_ROW_PCT], "_pct_only", metric_grid=_two_row_grid)
    # Portrait (3x2) variant of the abs-only figure — same content, transposed grid.
    _two_col_grid = (math.ceil(n_metrics / 2), 2)
    _render([_ROW_ABS], "_abs_only_3x2", metric_grid=_two_col_grid)

    # --- LaTeX table: mean ± std of percentage delta per codec ---
    # Metric groups: PA (4 sub-groups + mean) and PC (2 sub-groups + mean)
    pa_metrics = [
        ("CRISPR", "PA CRISPR"),
        ("ORF", "PA ORF"),
        ("Diversity", "PA Compound Diversity"),
        ("Bioactive", "PA Compound Bioactive-library"),
    ]
    pc_metrics = [
        ("Diversity", "PC Compound Diversity"),
        ("Bioactive", "PC Compound Bioactive-library"),
    ]
    # Filter to metrics that were actually plotted
    plotted_titles = {t for t, _ in metrics}
    pa_metrics = [(s, t) for s, t in pa_metrics if t in plotted_titles]
    pc_metrics = [(s, t) for s, t in pc_metrics if t in plotted_titles]

    # Aggregate: mean and std of delta_pct per (family, model, metric)
    pct_df = delta_df.dropna(subset=["delta_pct"])
    agg = pct_df.groupby(["family", "model", "metric"])["delta_pct"].agg(["mean", "std"]).reset_index()

    def _fmt_cell(mean_v, std_v):
        """Format a cell as mean ± std."""
        if np.isnan(mean_v):
            return "--"
        if np.isnan(std_v) or std_v == 0:
            return f"{mean_v:+.1f}"
        return f"{mean_v:+.1f} $\\pm$ {std_v:.1f}"

    def _get_agg(model, metric_title):
        """Get (mean, std) for a model/metric pair."""
        row = agg[(agg["model"] == model) & (agg["metric"] == metric_title)]
        if row.empty:
            return np.nan, np.nan
        return row["mean"].iloc[0], row["std"].iloc[0]

    def _compute_row_cells(m):
        """Compute data cells for a single model entry. Returns list of cell strings."""
        cells = []
        if pa_metrics:
            pa_means = []
            for _, metric_title in pa_metrics:
                mv, sv = _get_agg(m, metric_title)
                pa_means.append(mv)
                cells.append(_fmt_cell(mv, sv))
            valid_pa = [v for v in pa_means if not np.isnan(v)]
            pa_mean_avg = np.mean(valid_pa) if valid_pa else np.nan
            cells.insert(0, f"{pa_mean_avg:+.1f}" if not np.isnan(pa_mean_avg) else "--")
        if pc_metrics:
            pc_means = []
            pc_cells = []
            for _, metric_title in pc_metrics:
                mv, sv = _get_agg(m, metric_title)
                pc_means.append(mv)
                pc_cells.append(_fmt_cell(mv, sv))
            valid_pc = [v for v in pc_means if not np.isnan(v)]
            pc_mean_avg = np.mean(valid_pc) if valid_pc else np.nan
            cells.append(f"{pc_mean_avg:+.1f}" if not np.isnan(pc_mean_avg) else "--")
            cells.extend(pc_cells)
        return cells

    def _compute_row_means(m):
        """Compute numeric means per column for a single model. Returns list of floats (nan for missing)."""
        vals = []
        if pa_metrics:
            pa_means = []
            for _, metric_title in pa_metrics:
                mv, _ = _get_agg(m, metric_title)
                pa_means.append(mv)
                vals.append(mv)
            valid_pa = [v for v in pa_means if not np.isnan(v)]
            vals.insert(0, np.mean(valid_pa) if valid_pa else np.nan)
        if pc_metrics:
            pc_means = []
            for _, metric_title in pc_metrics:
                mv, _ = _get_agg(m, metric_title)
                pc_means.append(mv)
                vals.append(mv)
            valid_pc = [v for v in pc_means if not np.isnan(v)]
            vals.insert(len(vals) - len(pc_means), np.mean(valid_pc) if valid_pc else np.nan)
        return vals

    # Sort codec_entries by compression level first, then model family
    codec_entries_by_level = sorted(
        codec_entries,
        key=lambda x: (x[3], _fam_rank.get(x[0], 999)),
    )

    # Group entries by compression level (sort_rank)
    level_groups = []
    for level_rank, group in itertools_groupby(codec_entries_by_level, key=lambda x: x[3]):
        level_groups.append((level_rank, list(group)))

    # Build table
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    sel_escaped = best_selection.replace("_", r"\_")
    lines.append(
        r"\caption{Mean percentage performance change from baseline (raw/zstd) codec "
        r"($\pm$ std across normalization configs)"
        + (f", {sel_escaped}" if best_selection != "per_codec" else "")
        + r"}"
    )
    lines.append(r"\label{tab:codec_delta_pct}")

    # Columns: Codec, Model, [PA Mean, PA sub-groups...], [PC Mean, PC sub-groups...]
    pa_col_count = (1 + len(pa_metrics)) if pa_metrics else 0  # Mean + sub-groups
    pc_col_count = (1 + len(pc_metrics)) if pc_metrics else 0
    n_data_cols = pa_col_count + pc_col_count
    col_spec = "ll" + "c" * n_data_cols
    lines.append(r"\begin{tabular}{" + col_spec + r"}")
    lines.append(r"\toprule")

    # Group header row
    group_header_parts = [" ", " "]  # empty cells for Codec & Model
    cmidrule_parts = []
    col_pos = 3  # 1-indexed, columns 1-2 are Codec and Model
    if pa_col_count:
        group_header_parts.append(
            r"\multicolumn{" + str(pa_col_count) + r"}{c}{\textbf{PA NAP \% $\Delta$}}")
        cmidrule_parts.append(
            r"\cmidrule(lr){" + str(col_pos) + "-" + str(col_pos + pa_col_count - 1) + "}")
        col_pos += pa_col_count
    if pc_col_count:
        group_header_parts.append(
            r"\multicolumn{" + str(pc_col_count) + r"}{c}{\textbf{PC NAP \% $\Delta$}}")
        cmidrule_parts.append(
            r"\cmidrule(lr){" + str(col_pos) + "-" + str(col_pos + pc_col_count - 1) + "}")
    lines.append(" & ".join(group_header_parts) + r" \\")
    lines.append(" ".join(cmidrule_parts))

    # Sub-header: column names
    sub_names = []
    if pa_metrics:
        sub_names.extend(["Mean"] + [s for s, _ in pa_metrics])
    if pc_metrics:
        sub_names.extend(["Mean"] + [s for s, _ in pc_metrics])
    lines.append(r"\textbf{Codec} & \textbf{Model} & " + " & ".join(sub_names) + r" \\")
    lines.append(r"\midrule")

    # Data rows grouped by compression level
    for lvl_idx, (level_rank, entries) in enumerate(level_groups):
        # Collect numeric values for the mean row
        all_row_vals = []
        codec_label = entries[0][1]
        # Total rows: model entries + mean row (if >1 entry)
        n_group_rows = len(entries) + (1 if len(entries) > 1 else 0)
        mid_row = n_group_rows // 2
        for row_idx, (fam, cl, m, _) in enumerate(entries):
            fam_disp = FAMILY_DISPLAY.get(fam, fam)
            cells = _compute_row_cells(m)
            # Show codec label only on the middle row of the group
            cl_cell = codec_label if row_idx == mid_row else ""
            lines.append(f"{cl_cell} & {fam_disp} & " + " & ".join(cells) + r" \\")
            all_row_vals.append(_compute_row_means(m))

        # Mean row across models for this compression level
        if len(all_row_vals) > 1:
            arr = np.array(all_row_vals, dtype=float)
            mean_cells = []
            for col_i in range(arr.shape[1]):
                col_vals = arr[:, col_i]
                valid = col_vals[~np.isnan(col_vals)]
                if len(valid) > 0:
                    mean_cells.append(f"\\textbf{{{np.mean(valid):+.1f}}}")
                else:
                    mean_cells.append("--")
            lines.append(
                r" & \textit{Mean} & " + " & ".join(mean_cells) + r" \\"
            )

        # Add separator between compression level groups (but not after last)
        if lvl_idx < len(level_groups) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_fname = f"codec_delta_pct_table{_sel_suffix}.tex"
    tex_path = output_dir / tex_fname
    tex_path.write_text("\n".join(lines) + "\n")
    print(f"Saved: {tex_path}")


def main():
    parser = argparse.ArgumentParser(description="Gather norm_3 sweep results into CSV")
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("src/norm_3/data/features/variance_first_v4"),
        help="Path to sweep output directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: sweep_results.csv in sweep-dir)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="Directory for plots (default: sweep-dir/plots)",
    )
    parser.add_argument(
        "--filter-degenerate",
        action="store_true",
        help="Filter out degenerate configs (spherize + no PCA) from plots and summary",
    )
    parser.add_argument(
        "--best-metric",
        choices=["balanced", "nap_balanced", "pc", "pc_nap"],
        default="balanced",
        help="Metric for selecting best config: 'balanced' (PA%%*PC%%/100), 'nap_balanced' (PA_mean_nap*PC_mean_nap), 'pc' (PC%%), or 'pc_nap' (PC_mean_nap)",
    )
    parser.add_argument(
        "--best-selection",
        choices=["per_codec", "zstd_reference", "best_any_codec", "best_avg_codec"],
        default="per_codec",
        help="How to select best config: 'per_codec' (each codec picks own best), "
             "'zstd_reference' (use best zstd config for all codecs in same model family), "
             "'best_any_codec' (use the best config from any codec in the family), "
             "'best_avg_codec' (use the config with highest average score across codecs)",
    )
    parser.add_argument(
        "--exclude-families",
        nargs="+",
        default=[],
        metavar="FAMILY",
        help="Exclude model families from plots/summaries (e.g. cellprofiler cp_measure_filtered)",
    )
    parser.add_argument(
        "--exclude-codecs",
        nargs="+",
        default=[],
        metavar="CODEC",
        help="Exclude compression codecs from plots/summaries (e.g. d10 lq d2_e8)",
    )
    parser.add_argument(
        "--only-families",
        nargs="+",
        default=[],
        metavar="FAMILY",
        help="Only include these model families (e.g. dinov2_rr openphenom_rr)",
    )
    parser.add_argument(
        "--pa-vs-pc",
        action="store_true",
        help="Generate NAP PA vs PC scatter plots (sweep_nap_pa_vs_pc_best_balanced, replicable, targets, and per-family zoomed)",
    )
    args = parser.parse_args()

    if not args.sweep_dir.exists():
        print(f"Error: sweep directory {args.sweep_dir} does not exist")
        return 1

    if args.output is None:
        args.output = args.sweep_dir / "sweep_results.csv"

    # Find all metrics.json files
    json_files = list(args.sweep_dir.rglob("metrics.json"))
    print(f"Found {len(json_files)} metrics.json files")

    if not json_files:
        print("No results found!")
        return 1

    # Load all metrics
    all_metrics = []
    errors = 0
    for json_path in json_files:
        try:
            metrics = load_metrics(json_path)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
            errors += 1

    if errors:
        print(f"WARNING: {errors} files failed to load")

    df = pl.DataFrame(all_metrics, infer_schema_length=None)

    # Sort by model, then by PA descending
    df = df.sort(["model", "PA"], descending=[False, True], nulls_last=True)

    # Save full CSV (always unfiltered)
    df.write_csv(args.output)
    print(f"Saved {len(df)} results to {args.output}")

    # Apply degenerate filter for summaries and plots
    df_plot = filter_degenerate(df) if args.filter_degenerate else df

    # Include only specified families if requested
    if args.only_families:
        include_models = set()
        for fam in args.only_families:
            if fam in MODEL_FAMILIES:
                include_models.update(MODEL_FAMILIES[fam])
            else:
                print(f"Warning: unknown family '{fam}', available: {list(MODEL_FAMILIES.keys())}")
        if include_models:
            before = len(df_plot)
            df_plot = df_plot.filter(pl.col("model").is_in(list(include_models)))
            print(f"Only families {args.only_families}: {before} -> {len(df_plot)} rows")

    # Exclude model families if requested
    if args.exclude_families:
        exclude_models = set()
        for fam in args.exclude_families:
            if fam in MODEL_FAMILIES:
                exclude_models.update(MODEL_FAMILIES[fam])
            else:
                print(f"Warning: unknown family '{fam}', available: {list(MODEL_FAMILIES.keys())}")
        if exclude_models:
            before = len(df_plot)
            df_plot = df_plot.filter(~pl.col("model").is_in(list(exclude_models)))
            print(f"Excluded families {args.exclude_families}: {before} -> {len(df_plot)} rows")

    # Exclude codecs if requested (matches against display name codec suffix)
    if args.exclude_codecs:
        # Map codec names to canonical keys
        codec_keys = set()
        for c in args.exclude_codecs:
            canonical = _CODEC_ALIASES.get(c, c)
            if canonical in COMPRESSION_LEVEL:
                codec_keys.add(canonical)
            else:
                print(f"Warning: unknown codec '{c}', available: {list(COMPRESSION_LEVEL.keys())}")
        if codec_keys:
            # Build set of display names that match excluded codecs
            excluded_displays = set()
            for raw_name, disp in COMPRESSION_DISPLAY.items():
                level = _DISPLAY_TO_LEVEL.get(disp)
                if level is not None and any(
                    COMPRESSION_LEVEL.get(ck) == level for ck in codec_keys
                ):
                    excluded_displays.add(raw_name)
            if excluded_displays:
                before = len(df_plot)
                df_plot = df_plot.filter(~pl.col("model").is_in(list(excluded_displays)))
                print(f"Excluded codecs {args.exclude_codecs}: {before} -> {len(df_plot)} rows")

    # Print summary
    print("\n=== Summary by Compression Codec ===")
    summary = (
        df_plot.group_by("model")
        .agg(
            pl.len().alias("n_configs"),
            pl.col("PA").max().alias("best_PA"),
            pl.col("PC").max().alias("best_PC"),
            pl.col("PA").mean().alias("mean_PA"),
            pl.col("PC").mean().alias("mean_PC"),
        )
        .sort("model")
    )
    print(summary)

    print("\n=== Summary by Batch Method ===")
    batch_summary = (
        df_plot.group_by("batch_method")
        .agg(
            pl.len().alias("n_configs"),
            pl.col("PA").mean().alias("mean_PA"),
            pl.col("PC").mean().alias("mean_PC"),
            pl.col("PA").max().alias("best_PA"),
            pl.col("PC").max().alias("best_PC"),
        )
        .sort("batch_method")
    )
    print(batch_summary)

    # --- Convert to pandas ONCE and pre-compute shared state ---
    t0_prep = time.monotonic()
    pdf_plot = df_plot.to_pandas()
    pdf_plot["balanced_score"] = pdf_plot["PA"] * pdf_plot["PC"] / 100
    if "PA_mean_nap" in pdf_plot.columns and "PC_mean_nap" in pdf_plot.columns:
        pdf_plot["nap_balanced"] = pdf_plot["PA_mean_nap"] * pdf_plot["PC_mean_nap"]
    pdf_plot["display_name"] = pdf_plot["model"].map(lambda m: get_display_name(m))
    pdf_plot, best_col = _add_best_column(pdf_plot, args.best_metric)
    all_models = sort_models(pdf_plot["model"].unique().tolist())
    best_idx = _compute_best_idx(pdf_plot, all_models, best_col, args.best_selection, args.best_metric)
    family_configs = _get_family_configs(pdf_plot, args.best_selection, args.best_metric, best_col=best_col)
    model_colors = _build_model_colors(all_models)
    print(f"Pandas conversion + pre-compute: {time.monotonic() - t0_prep:.2f}s")

    _metric_labels = {"balanced": "PA * PC", "nap_balanced": "NAP balanced", "pc": "PC %", "pc_nap": "PC mean NAP"}
    metric_label = _metric_labels.get(args.best_metric, args.best_metric)
    print(f"\n=== Best Config per Codec (by {metric_label}, selection={args.best_selection}) ===")
    for model in all_models:
        bi = best_idx.get(model)
        if bi is None:
            continue
        row = pdf_plot.loc[bi]
        print(f"  {get_display_name(model):16s}  PA={row['PA']:.1f}%  PC={row['PC']:.1f}%  "
              f"score={row[best_col]:.3f}  config={row['config']}")

    if args.best_selection == "zstd_reference":
        zstd_configs = _find_zstd_best_config_per_family(pdf_plot, args.best_metric, best_col=best_col)
        print("\n=== zstd Reference Configs per Family ===")
        for family, config in sorted(zstd_configs.items()):
            print(f"  {family:24s}  {config}")
    elif args.best_selection == "best_any_codec":
        any_configs = _find_best_config_any_codec_per_family(pdf_plot, args.best_metric, best_col=best_col)
        print("\n=== Best Config (any codec) per Family ===")
        for family, config in sorted(any_configs.items()):
            print(f"  {family:24s}  {config}")
    elif args.best_selection == "best_avg_codec":
        avg_configs = _find_best_avg_config_per_family(pdf_plot, args.best_metric, best_col=best_col)
        print("\n=== Best Config (avg across codecs) per Family ===")
        for family, config in sorted(avg_configs.items()):
            print(f"  {family:24s}  {config}")

    # Generate plots
    if args.plot:
        plot_dir = args.plot_dir or args.sweep_dir / "plots"
        print(f"\nGenerating plots in {plot_dir}...")

        # Shared kwargs for functions that need best_idx
        bkw = dict(models=all_models, best_idx=best_idx, best_col=best_col,
                    best_metric=args.best_metric, best_selection=args.best_selection)
        # Shared kwargs for scatter plots that also need family_configs
        fkw = dict(models=all_models, best_idx=best_idx, family_configs=family_configs,
                    best_metric=args.best_metric, best_selection=args.best_selection)

        # Unfiltered pandas for degenerate report
        pdf_unfiltered = df.to_pandas()

        plot_timings = []
        def _timed(name, fn, *a, **kw):
            t0 = time.monotonic()
            fn(*a, **kw)
            elapsed = time.monotonic() - t0
            plot_timings.append((name, elapsed))
            print(f"  [{elapsed:.2f}s] {name}")

        # Always generated plots
        _timed("group_nap", generate_group_nap_plot, pdf_plot, plot_dir, model_colors, **bkw)
        _timed("group_nap_compact", generate_group_nap_plot_compact, pdf_plot, plot_dir, model_colors, **bkw)
        _timed("group_nap_latex", generate_group_nap_latex, pdf_plot, plot_dir, model_colors, **bkw)
        _timed("group_nap_latex_compact", generate_group_nap_latex_compact, pdf_plot, plot_dir, model_colors, **bkw)
        _timed("nap_pa_vs_pc_best_balanced", generate_nap_pa_vs_pc_best_balanced, pdf_plot, plot_dir, model_colors, **fkw)
        _timed("nap_pa_vs_pc_best_balanced_clean", generate_nap_pa_vs_pc_best_balanced_clean, pdf_plot, plot_dir, model_colors, **fkw)
        _timed("nap_replicable_pa_vs_pc", generate_nap_replicable_pa_vs_pc_best_balanced, pdf_plot, plot_dir, model_colors, **fkw)
        _timed("nap_pa_vs_pc_targets", generate_nap_pa_vs_pc_targets_best_balanced, pdf_plot, plot_dir, model_colors, **fkw)
        _timed("nap_pa_vs_pc_per_model", generate_nap_pa_vs_pc_per_model, pdf_plot, plot_dir, model_colors, **fkw)
        _timed("nap_pa_vs_pc_combined", generate_nap_pa_vs_pc_combined, pdf_plot, plot_dir, model_colors, **fkw)
        _timed("nap_pa_vs_pc_combined_all_points", generate_nap_pa_vs_pc_combined, pdf_plot, plot_dir, model_colors, **fkw, show_all_points=True)
        _timed("nap_pa_vs_pc_panel_a", generate_nap_pa_vs_pc_panel_a, pdf_plot, plot_dir, model_colors, **fkw)
        _timed("nap_pa_vs_pc_panel_a_all_points", generate_nap_pa_vs_pc_panel_a, pdf_plot, plot_dir, model_colors, **fkw, show_all_points=True)
        # Raw + medium-quality only — strips out hq/d20/etc. for a compact comparison.
        _timed("nap_pa_vs_pc_panel_a_raw_mq", generate_nap_pa_vs_pc_panel_a, pdf_plot, plot_dir, model_colors,
               **fkw, codec_filter=["raw", "mq"], filename_suffix="_raw_mq")
        _timed("nap_pa_vs_pc_panel_a_raw_mq_all_points", generate_nap_pa_vs_pc_panel_a, pdf_plot, plot_dir, model_colors,
               **fkw, show_all_points=True, codec_filter=["raw", "mq"], filename_suffix="_raw_mq")
        _timed("codec_delta_from_raw", generate_codec_delta_from_raw_plot, pdf_plot, plot_dir, model_colors, **bkw)
        _timed("codec_delta_from_raw_groups", generate_codec_delta_from_raw_groups_plot, pdf_plot, plot_dir, model_colors, **bkw)
        _timed("codec_delta_balanced_violin", generate_codec_delta_balanced_violin, pdf_plot, plot_dir, model_colors, **bkw)

        if not args.pa_vs_pc:
            _timed("all_metrics", generate_all_metrics_plot, pdf_plot, plot_dir, model_colors, **bkw)
            # Disabled: takes ~73s due to seaborn KDE on 60 models x 20 metrics
            # _timed("all_metrics_violin", generate_all_metrics_violin, pdf_plot, plot_dir, model_colors, **bkw)
            _timed("overview", generate_overview_plot, pdf_plot, plot_dir, model_colors, **bkw)
            _timed("pa_vs_pc", generate_pa_vs_pc_plot, pdf_plot, plot_dir, model_colors, **fkw)
            _timed("pa_vs_pc_targets", generate_pa_vs_pc_targets_plot, pdf_plot, plot_dir, model_colors, **fkw)
            _timed("pa_vs_pc_best_balanced", generate_pa_vs_pc_best_balanced, pdf_plot, plot_dir, model_colors, **fkw)
            _timed("batch_method", generate_batch_method_plot, pdf_plot, plot_dir)
            _timed("norm_pca", generate_norm_pca_plot, pdf_plot, plot_dir)
            _timed("norm_batch", generate_norm_batch_comparison, pdf_plot, plot_dir)
            _timed("best_per_model", generate_best_per_model_plot, pdf_plot, plot_dir, model_colors, all_models)
            _timed("best_mean_nap", generate_best_mean_nap_plot, pdf_plot, plot_dir, model_colors, all_models,
                   best_metric=args.best_metric, best_selection=args.best_selection)
            _timed("per_group", generate_per_group_plot, pdf_plot, plot_dir, model_colors, all_models, best_idx,
                   best_metric=args.best_metric, best_selection=args.best_selection)
            _timed("filtered_vs_raw", generate_filtered_vs_raw_plot, pdf_plot, plot_dir)
            _timed("degenerate_report", generate_degenerate_report, pdf_unfiltered, plot_dir)

        total = sum(t for _, t in plot_timings)
        print(f"\nAll plots generated! Total plot time: {total:.2f}s")
        print("\n=== Plot Timing Summary (sorted slowest first) ===")
        for name, elapsed in sorted(plot_timings, key=lambda x: -x[1]):
            print(f"  {name:35s} {elapsed:6.2f}s  ({elapsed/total*100:4.1f}%)")

    return 0


if __name__ == "__main__":
    exit(main())
