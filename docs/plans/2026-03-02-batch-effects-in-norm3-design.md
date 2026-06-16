# Design: Integrate Batch Effects into norm_3 Metrics Pipeline

## Goal

Add well position effect and plate batch effect computation to `src/norm_3/metrics.py`'s `evaluate_all()`, so sweep results include batch effect metrics alongside PA, PC, and PC_replicable.

## Approach

### New function: `calculate_batch_effects()` in `metrics.py`

Consolidates logic from `evaluation/analyze_batch_effects.py` (fast, random-sampling) and `scripts/evaluate_phenotypic_activity.py` (exact copairs).

- Default mode: `"fast"` — uses random index sampling (`_random_index`, `_random_index_pos`, `_random_index_neg`) for speed during sweeps
- Fallback mode: `"exact"` — full copairs computation without random sampling
- Supports per-group statistics (consistent with PA/PC)
- Two sub-metrics:
  1. **Well Position Effect** (treatments only): Do wells at the same position cluster across plates?
  2. **Plate Batch Effect** (treatments only): Are plates distinguishable?

### Integration into `evaluate_all()`

New parameters (all backward-compatible defaults):
- `skip_batch_effects: bool = True` — skipped by default so existing sweeps are unaffected
- `batch_effects_mode: str = "fast"` — `"fast"` or `"exact"`
- `well_col: str = "Metadata_Well"` — well position column

Output keys added to metrics dict:
- `well_effect_pct`, `well_effect_mean_nap`, `well_effect_n_active`, `well_effect_n_total`
- `plate_effect_pct`, `plate_effect_mean_nap`, `plate_effect_n_active`, `plate_effect_n_total`
- `batch_effects_group_summary` (per-group breakdown)

### Integration into `pipeline.py`

`evaluate_metrics()` passes new config params to `evaluate_all()`:
- `skip_batch_effects` from config (default `True`)
- `batch_effects_mode` from config (default `"fast"`)
- `well_col` from config (default `"Metadata_Well"`)

### Config changes

**v9 preset** (`gpu_base_variance_first_v9.yaml`): No changes needed — `skip_batch_effects` defaults to `True`.

**v11_lite sweep configs**: Add `skip_batch_effects: false` to evaluate_metrics params so batch effects run by default for these sweeps.

## Files changed

1. `src/norm_3/metrics.py` — Add `calculate_batch_effects()`, integrate into `evaluate_all()`
2. `src/norm_3/pipeline.py` — Pass new config params to `evaluate_all()`
3. `src/norm_3/conf/sweep/focused_dl_v11_lite_tvn_efaar.yaml` — Add `skip_batch_effects: false`
4. `src/norm_3/conf/sweep/focused_cp_v11_lite_tvn_efaar.yaml` — Add `skip_batch_effects: false`

## Backward compatibility

- Existing sweeps (v6, v9, v10, v11 non-lite): `skip_batch_effects` defaults to `True` — no behavior change
- v11_lite: explicitly sets `skip_batch_effects: false`
