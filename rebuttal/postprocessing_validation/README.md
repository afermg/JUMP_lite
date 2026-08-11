# Post-processing validation/test rebuttal analysis

This directory contains a read-only, resumable analysis of post-processing
configuration selection. It uses the frozen normalized profiles under
`/work/datasets/JUMP-lite-wacv`; it does not regenerate or modify the canonical
sweep.

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

## Outputs

Each output directory contains:

- `protocol.json` and `provenance.json`
- `treatment_split.csv` and `split_summary.csv`
- `validation_config_scores.csv` and `selected_configs.csv`
- `selected_codec_coverage.csv` and `common_wells.csv`
- `heldout_test_scores.csv`
- `per_unit/*` treatment-level PA and target-level PC tables
- `checkpoints/*` and suppressed copairs logs
- `REPORT.md`

`FAILURE.json` is written on fail-closed termination. Existing full-data,
cross-codec best-average plots remain a separate sensitivity analysis and are
not read or relabeled by this runner.

## Tests

```bash
src/norm_3/.pixi/envs/default/bin/python \
  rebuttal/postprocessing_validation/test_run_analysis.py
```
