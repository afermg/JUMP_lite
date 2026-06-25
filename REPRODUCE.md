# Reproducing JUMP_lite results

This document covers how to reproduce the figures and tables from a clean
checkout. Two paths are supported:

- **Post-sweep** (~55 min, ~350 MB written): reuse an existing sweep output and
  regenerate motive evaluation + figures + tables.
- **Full sweep** (hours, hundreds of GB): re-run the v11_lite sweep from raw
  features, then continue through the post-sweep path.

## 1. Environment

The repo pins everything via Nix + pixi. After cloning:

```bash
nix develop                # drops you in a shell with pixi + system libs
pixi run python --version  # verify
```

All `just` recipes call `nix develop . pixi run python ...` internally; you do
not need to activate anything else.

## 2. Data layout

Everything reproducible lives under `data/`:

```
data/
  raw/features/         # raw feature parquets (symlinks into data/features/)
  features/             # canonical raw feature dirs (jump_lite, jump_lite_cl_3, ...)
  intermediate/         # sweep outputs + motive evaluation outputs
    sweep_v11_lite/
    motive_eval/
  results/
    figures/            # PNGs
    tables/             # .tex
    summaries/          # CSV / JSON
```

Recipes write only to `data/intermediate/<new_subdir>/` and `data/results/`,
never in place — safe to point heavy inputs at a shared canonical copy.

### Sharing heavy data across checkouts

The `data/features/` and `data/intermediate/sweep_*` dirs are large (~22 GB
features, ~50 GB per sweep). To share them across checkouts, symlink:

From a fresh checkout, link the heavy raw inputs:

```bash
mkdir -p data/raw/features data/intermediate
ln -s /abs/path/to/canonical/data/features  data/features
ln -s ../../features/jump_lite              data/raw/features/jump_lite
ln -s ../../features/jump_lite_cl_3         data/raw/features/jump_lite_cl_3
```

`just reproduce` expects **two sweep dirs** and **one motive eval dir** at
fixed paths under `data/intermediate/` and `data/features/`. The canonical
copies live under non-default names, so symlink them in explicitly:

```bash
# lite sweep (~976 GB at canonical path)
ln -s /work/users/jfredinh/projects/JUMP_core/src/norm_3/data/features/variance_first_v11_lite \
      data/intermediate/sweep_v11_lite

# target2 sweep (~5.4 GB)
ln -s /work/users/jfredinh/projects/JUMP_core/src/norm_3/data/features/MAIN_RESULTS__figure_4_variance_first_v11 \
      data/features/variance_first_v11

# motive eval (~13 MB) — needed for motive figures + tables
cp -a /path/to/motive_eval data/intermediate/motive_eval
# (or symlink if you have a stable canonical location)

# segmentation comparison data (~458 MB, already in this repo at analysis/...)
ln -s ../../analysis/segmentation/output/segmentation_comparison \
      data/intermediate/segmentation_comparison

# rank-stability input CSV (~11 MB, already in src/norm_3/data/features/)
mkdir -p data/intermediate/sweep_summaries
cp src/norm_3/data/features/sweep_results_v11_lite_full.csv \
   data/intermediate/sweep_summaries/
```

Verify all five are present:

```bash
ls data/intermediate/sweep_v11_lite/sweep_results.csv
ls data/features/variance_first_v11/ | head
ls data/intermediate/motive_eval/large_strict/ | head
ls data/intermediate/segmentation_comparison/detailed_results/ | head
ls data/intermediate/sweep_summaries/sweep_results_v11_lite_full.csv
```

If motive eval isn't available, generate it from the lite sweep:

```bash
just motive-run-top      # writes data/intermediate/motive_eval/large_{full,strict}/
```

Recipes only write under `data/intermediate/<new>` and `data/results/` — they
do **not** mutate the symlinked inputs.

## 3. Post-sweep reproduction

Assumes `data/intermediate/sweep_v11_lite/` is populated (real dir or
symlink). End-to-end:

```bash
just clean-results       # wipe data/results/{figures,tables,summaries}
just reproduce           # runs gather + motive plots + tables + rank + segmentation
```

`just reproduce` chains:

1. `results-v11-lite` — `gather_sweep_results.py` over the lite sweep → CSV + plots under `data/results/figures/sweep_v11_lite/`
2. `results-v11` — same over the target2 sweep → CSV + plots under `data/results/figures/sweep_v11/`
3. `motive-plot` / `motive-plot-delta` / `motive-table-delta` / `motive-plot-cross` — motive figures from `data/intermediate/motive_eval/large_strict/` (lite only; no target2 motive eval exists)
4. `model-task-rank` + `combined-codec-delta-table`
5. `rank-stability`
6. `segmentation-iou-ablation`
7. `saturation-plot-bestconfig`

Outputs land in `data/results/figures/`, `tables/`, `summaries/`.

## 4. Sweep regeneration (heavy path)

Only needed if you don't have a sweep output to reuse.

```bash
just sweep-v11-lite      # full v11_lite sweep — hours, hundreds of GB
```

Granularity options for validation before committing to the full run:

```bash
just sweep-validate-slice-cellcount 0 4         # cell-count baseline, 1 GPU, 4 jobs
just sweep-validate-slice-cp        0.05,0.1 0 3
just sweep-validate-slice           morphem jpegxl_lossy_mq 0.05,0.1 0 3
```

Each slice writes under `data/intermediate/sweep_v11_lite_validate/` so it
doesn't collide with the real sweep.

### Smoke variant (4 configs/(model, codec))

For an end-to-end pipeline test that exercises the sweep without committing
hours of GPU time, run the smoke variant. It collapses the per-(model, codec)
grid from 48 → 4 by overriding `hydra.sweeper.params` on the CLI, and writes
to a separate `_smoke` output dir so the full sweep is not clobbered.

```bash
just sweep-v11-lite-smoke 0 4   # GPU 0, 4 joblib workers
just sweep-v11-smoke      0 4   # target2 (only needed if also regenerating results-v11)
```

Wall time: ~20–40 min for lite, ~1–2 h for target2. To feed the smoke output
into `just reproduce`:

```bash
ln -sf ../../src/norm_3/data/features/variance_first_v11_lite_smoke \
       data/intermediate/sweep_v11_lite
ln -sf src/norm_3/data/features/variance_first_v11_smoke \
       data/features/variance_first_v11
just reproduce
```

Downstream figures (saturation plots, rank stability, etc.) will be
qualitatively different from the paper because the underlying sweep is far
smaller — this is a smoke test, not a result.

## 5. MOTIVE evaluation

If you're regenerating motive outputs from a sweep (rather than reusing
`data/intermediate/motive_eval/`):

```bash
just motive-run-top       # top-50 per (family, codec) — ~55 min, production path
just motive-run-all       # exhaustive — ~5 hours
```

`motive-run-top` is what the validated end-to-end test uses.

## 6. Known caveats

- **Metadata drift.** The metadata files in this repo (`metadata/`) differ from
  the JUMP_core upstream by ~1.4% of rows for CP and cell_count features.
  Numerical results match within float tolerance for `morphem` features and
  for the motive pipeline; CP/cell_count slices will diverge in row counts.
- **Single-cell slices.** Four recipes exit 1 when run on a single-cell sweep
  (insufficient input): `motive-plot-delta`, `motive-table-delta`,
  `model-task-rank`, `combined-codec-delta-table`. Not a bug for production
  runs; relevant only if you do tiny validation sweeps.
- **Symlink traversal.** `gather_sweep_results.py` uses
  `os.walk(followlinks=True)` to traverse symlinked sweep dirs — required
  because `Path.rglob` on Python < 3.13 silently skips symlinked
  subdirectories.

## 7. What was validated

The post-sweep path was end-to-end validated on the full v11_lite sweep:

- 298 PNG + 12 .tex + 8 CSV produced under `data/results/`
- 6/6 comparable PNGs (iou_ablation_accuracy + 5 rank_stability_*) byte-identical to tracked references in `aux_figures/`
- Wall time ~55 min, ~350 MB written (sweep stayed symlinked)
- MOTIVE numeric output bit-identical to reference (74/74 keys match)

Target2 (`results-v11`) was wired into `reproduce` and verified to produce
44 figures + summaries under `data/results/figures/sweep_v11/` against the
symlinked canonical sweep. Cross-checking those outputs against the paper's
final target2 figures has **not** been done in this branch.
