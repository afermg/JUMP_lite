# Post-processing validation/test rebuttal analysis

## Strict split-before-fit analysis

`strict_split_fit.py` is the primary leakage-free runner for reviewer point A3.
It constructs the treatment split before reading any feature matrix. Every
feature-selection rule and fitted numerical transform then excludes held-out
test treatments. Shared negative controls remain fit and retrieval references.
The runner reads raw features and the resolved sweep YAML files, but it never
reads canonical normalized `output.parquet` files or archived PA maps for a
strict score.

The fixed seed `20260811` assigns non-control `Metadata_JCP2022` identities to a
20% selection partition and an 80% held-out test partition within complete group
membership strata. Each candidate is fit on Raw selection treatments plus shared
controls. Both PA and PC are recomputed on that selection partition. One recipe
per family is selected by the existing within-family
`min-max(PA) * min-max(PC)` rule. The winning recipe is frozen across codecs.
Its numerical state is fit separately within each codec using only that codec's
selection treatments plus shared controls, then applied to held-out treatments.
No fallback may fit a test row.

The learned-family archive contains 350 directory aliases per family but only
175 effective resolved pipelines. The two aliases differ only in the unused
`use_prune_correlated` top-level flag because the resolved learned pipeline has
no correlation-pruning step. The strict runner evaluates each effective recipe
once and records both aliases. The complete inventory is 1,160 effective
recipes represented by 2,035 aliases: 175/350 for each of DINOv2, MorphEM,
OpenPhenom, SubCell, and ViT-rand; 280/280 for CellProfiler; and 5/5 for Cell
Count.

Strict candidate fit state includes missingness and low-variance feature
retention, per-plate standardization or RobustMAD, outlier removal,
out-of-sample empirical INT, correlation pruning where the resolved pipeline
enables it, and all TVN EFAAR state. Steps fit on controls use shared controls
only. Other fitted steps use selection treatments plus controls. OpenPhenom
feature prefixes are mapped to one numbered canonical axis before fitting.
Missing fit strata, unseen transform strata, nonfinite values, schema drift,
missing controls, and incomplete codec coverage fail closed.

Run the bounded one-GPU smoke without changing production services:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/run/opengl-driver/lib:$NIX_LD_LIBRARY_PATH \
PYTHONPATH=. \
/work/users/amunoz/projects/JUMP_lite/src/norm_3/.pixi/envs/default/bin/python \
  rebuttal/postprocessing_validation/strict_split_fit.py \
  --mode smoke --families dinov2 --max-recipes 1 \
  --output-dir /work/scratch/amunoz/jump-lite-strict-postprocessing-smoke-v1
```

The recommended create-only production command is:

```bash
CUDA_VISIBLE_DEVICES=0 \
LD_LIBRARY_PATH=/run/opengl-driver/lib:$NIX_LD_LIBRARY_PATH \
PYTHONPATH=. \
/work/users/amunoz/projects/JUMP_lite/src/norm_3/.pixi/envs/default/bin/python \
  rebuttal/postprocessing_validation/strict_split_fit.py \
  --mode production \
  --output-dir /work/scratch/amunoz/jump-lite-strict-postprocessing-final-v1
```

A failed invocation may resume only against its exact protocol and code identity
by adding `--resume`. Candidate cache keys bind the raw input hash, split hash,
exact fit-ID hash, code hash, family, codec, and effective recipe. Winners are
refit deterministically. Candidate transform states are not retained.

Final-code DINOv2 candidate smoke v4 completed in 31.23 seconds with 3.36 GiB
peak process RSS. A one-candidate CellProfiler smoke completed in 268.59 seconds.
GPU sampling observed 7,502 MiB total use for DINOv2 and 24,022 MiB for
CellProfiler, from a 4,090 MiB production baseline. The one-recipe DINOv2
production smoke fit Raw, HQ, MQ, and D20 separately and completed in
3 minutes 46 seconds with 10.58 GiB peak RSS. Its 20-file hash closure passed,
and every codec scored the same 134,645-well test-plus-control intersection.
Evidence roots are `/work/scratch/amunoz/jump-lite-strict-postprocessing-smoke-v4`,
`/work/scratch/amunoz/jump-lite-strict-postprocessing-cellprofiler-smoke-v1`,
and `/work/scratch/amunoz/jump-lite-strict-postprocessing-production-smoke-v3`.
Sequential extrapolation is about 29 hours. Allow 28 to 36 hours on one shared
H100 because the 280 CellProfiler candidates dominate runtime and general DL
production remains active.

The older `run_analysis.py` analysis below remains preserved as historical
selection-only evidence. It reads transforms fitted before the split and must
not be used as the final response to A3 or relabeled as strict held-out evidence.

## Historical selection-only analysis

This directory also contains the earlier read-only, resumable analysis of
post-processing configuration selection. It uses the frozen normalized profiles
under `/work/datasets/JUMP-lite-wacv`; it does not regenerate or modify the
canonical sweep.

## Protocol

- Deterministic 20% validation / 80% test split of non-control
  `Metadata_JCP2022` IDs, using SHA-256 with seed `20260811`.
- IDs are stratified by their complete `Metadata_Group` membership, so a compound
  present in both compound groups cannot cross partitions.
- Exactly one configuration is selected per family using **Raw validation data
  only**. PA mean NAP is subset from the archived per-treatment PA map; PC mean
  NAP is recomputed on validation treatments. The intended within-family
  min-max-scaled PA and PC values are multiplied.
- The selected configuration must exist for every intended codec. Missing or
  zero-byte profiles fail the run; there is no per-codec fallback.
- Final PA and PC are recomputed on held-out test treatments using one frozen
  `Metadata_id` intersection across every selected family/codec output. Shared
  negative-control wells remain retrieval references.

The split isolates configuration selection, not transform fitting. Archived
normalization transforms were fitted before the split, so this is not a strict
inductive preprocessing holdout.

## Environment

Use the existing `norm_3` pixi environment, which provides Polars, copairs,
pandas, and PyYAML.

## Smoke test

The following bounded run evaluates two CellCount candidates and its Raw-only
held-out result. Always use a dedicated smoke output directory:

```bash
cd src/norm_3
pixi run python ../../rebuttal/postprocessing_validation/run_analysis.py \
  --families cell_count \
  --max-selection-configs 2 \
  --output-dir ../../rebuttal/postprocessing_validation/smoke
```

Rerunning the identical command resumes completed checkpoints. A different
protocol must use a different output directory.

## Full production run

```bash
cd src/norm_3
pixi run python ../../rebuttal/postprocessing_validation/run_analysis.py \
  --workers 4 \
  --output-dir ../../rebuttal/postprocessing_validation/results
```

The four workers evaluate independent candidates and final profiles concurrently.
The exact copairs PA/PC implementation is CPU-bound; the normalized profiles are
already materialized, so this stage does not rerun GPU normalization.

The full run evaluates 2,035 Raw candidates in the current archive (350 for
each of five learned families, 280 CellProfiler configurations, and five
CellCount configurations), then evaluates 22 pinned family/codec profiles. It
is CPU- and I/O-intensive and is expected to take hours.

## Paired uncertainty

Run the conditional paired cluster bootstrap from the frozen per-unit outputs:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 \
  src/norm_3/.pixi/envs/default/bin/python \
  rebuttal/postprocessing_validation/bootstrap_uncertainty.py
```

The production protocol uses 50,000 resamples and seed `20260812`. The base
analysis report generator includes the downstream uncertainty and figure
sections only when the complete uncertainty bundle is present and its point
estimates match the held-out score table; incomplete or stale bundles fail
closed. PA treatment IDs are resampled within the frozen composite split strata,
retaining every
evaluable group row for a sampled treatment. PC targets are resampled within
complete observed group-membership strata. Cluster weights are shared across
all models/codecs, so contrasts are paired. PA and PC margins are resampled
independently because the saved target tables do not contain the treatment--
target/query decomposition required for an end-to-end joint resample.

The pointwise intervals therefore quantify uncertainty over the observed
held-out PA treatment and PC target distributions under a conditional working
product-of-margins model. Independent PA/PC streams omit their unknown
covariance, so intervals and centered-bootstrap p-values may be too narrow or
too wide; they are not end-to-end or unconditional intervals. Product tests are
adjusted in two separate Holm families: all 15 codec-vs-Raw contrasts and all 51
same-codec model contrasts. See `results/uncertainty/REPORT.md` for results and
limitations.

## Figure

Regenerate the paper figure from the frozen held-out score and uncertainty
tables with:

```bash
.venv/bin/python rebuttal/postprocessing_validation/plot_heldout_codec_performance.py
```

This writes `results/heldout_codec_performance.pdf` and `.png`. Panel (a) shows
the absolute unscaled PA--PC product; panel (b) expresses each learned model's
point estimate relative to its own Raw result. Whiskers are pointwise,
conditional 95% paired cluster-bootstrap percentile intervals from the
50,000-replicate run under the working product-of-margins model.

## Outputs

Each output directory contains:

- `protocol.json` and `provenance.json`
- `treatment_split.csv` and `split_summary.csv`
- `validation_config_scores.csv` and `selected_configs.csv`
- `selected_codec_coverage.csv` and `common_wells.csv`
- `heldout_test_scores.csv`
- `heldout_codec_performance.pdf` and `.png`
- `per_unit/*` treatment-level PA and target-level PC tables
- `uncertainty/*` paired interval, contrast, rank, diagnostic, and provenance tables
- `checkpoints/*` and suppressed copairs logs
- `REPORT.md`

`FAILURE.json` is written on fail-closed termination. Existing full-data,
cross-codec best-average plots remain a separate sensitivity analysis and are
not read or relabeled by this runner.

## Tests

```bash
src/norm_3/.pixi/envs/default/bin/python \
  rebuttal/postprocessing_validation/test_run_analysis.py
src/norm_3/.pixi/envs/default/bin/python \
  rebuttal/postprocessing_validation/test_bootstrap_uncertainty.py
```
