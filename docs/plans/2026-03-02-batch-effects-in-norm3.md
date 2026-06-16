# Batch Effects in norm_3 Metrics Pipeline — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add well position effect and plate batch effect computation to the norm_3 sweep pipeline so metrics.json includes batch effect scores.

**Architecture:** Add `calculate_batch_effects()` to `src/norm_3/metrics.py` supporting fast (random-sampling) and exact modes. Integrate into `evaluate_all()` with `skip_batch_effects=True` default for backward compatibility. Enable by default only in v11_lite sweep configs.

**Tech Stack:** copairs, polars, numpy, hydra (yaml configs)

---

### Task 1: Add `calculate_batch_effects()` to metrics.py

**Files:**
- Modify: `src/norm_3/metrics.py` (add new function after line 411, before `evaluate_all`)

**Step 1: Add the `calculate_batch_effects` function**

Insert before `evaluate_all` (line 414). This consolidates logic from `evaluation/analyze_batch_effects.py` (fast mode) and `scripts/evaluate_phenotypic_activity.py` (exact mode):

```python
def calculate_batch_effects(
    df: pl.DataFrame,
    features: list[str],
    null_size: int = 10_000,
    p_threshold: float = 0.05,
    seed: int = 0,
    mode: str = "fast",
    n_random_groups: int = 15,
    compound_col: str = "Metadata_pert_iname",
    negcon_col: str = "Metadata_negcon",
    batch_col: str = "Metadata_Plate",
    well_col: str = "Metadata_Well",
    group_col: str = "Metadata_Group",
) -> dict[str, Any]:
    """Calculate batch effect metrics: well position effect and plate batch effect.

    Two metrics are computed:
    1. Well Position Effect (treatments only): Do wells at the same position
       cluster across plates? High % = well position bias exists.
    2. Plate Batch Effect (treatments only): Are plates distinguishable?
       High % = plate-level batch effects exist.

    Args:
        df: Profiles with metadata
        features: Feature column names
        null_size: Size of null distribution
        p_threshold: Significance threshold
        seed: Random seed
        mode: "fast" (random sampling for speed) or "exact" (full copairs)
        n_random_groups: Number of random groups for fast mode
        compound_col: Column containing compound identifier
        negcon_col: Column containing negative control flag
        batch_col: Column containing plate identifier
        well_col: Column containing well identifier
        group_col: Column containing group identifier

    Returns:
        Dictionary with well_position_effect and plate_batch_effect results
    """
    from copairs import map as copairs_map

    results = {}
    rng = np.random.default_rng(seed)
    df_pd = df.to_pandas()

    if well_col not in df_pd.columns:
        print(f"  Warning: {well_col} not found, skipping batch effect analysis")
        return {"well_position_effect": None, "plate_batch_effect": None}

    has_groups = group_col in df_pd.columns

    # Add random index columns for fast mode
    if mode == "fast":
        df_pd["_random_index"] = rng.integers(1, n_random_groups + 1, size=len(df_pd))
        df_pd["_random_index_pos"] = rng.integers(1, 11, size=len(df_pd))
        df_pd["_random_index_neg"] = rng.integers(1, 251, size=len(df_pd))

    # Filter out negative controls for both analyses
    if negcon_col in df_pd.columns:
        df_treatments = df_pd[df_pd[negcon_col] == False].copy()
    else:
        df_treatments = df_pd.copy()

    if len(df_treatments) < 10:
        print(f"    Not enough treatment samples ({len(df_treatments)})")
        _empty = {"pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0}
        return {"well_position_effect": _empty, "plate_batch_effect": _empty}

    # =========================================
    # 1. Well Position Effect
    # =========================================
    print("  Calculating Well Position Effect...")
    try:
        if mode == "fast":
            if has_groups:
                pos_sameby_well = [group_col, well_col]
                neg_sameby_well = [group_col, batch_col, "_random_index"]
            else:
                pos_sameby_well = [well_col]
                neg_sameby_well = [batch_col, "_random_index"]
            pos_diffby_well = [batch_col, compound_col]
            neg_diffby_well = [well_col, compound_col]
        else:
            # Exact mode
            if has_groups:
                pos_sameby_well = [group_col, well_col]
                neg_sameby_well = [group_col, batch_col]
            else:
                pos_sameby_well = [well_col]
                neg_sameby_well = [batch_col]
            pos_diffby_well = [compound_col]
            neg_diffby_well = [well_col, compound_col]

        # Filter: need wells appearing on multiple plates (fast) or with multiple compounds (exact)
        if mode == "fast":
            if has_groups:
                well_plate_counts = df_treatments.groupby([group_col, well_col])[batch_col].nunique()
                valid_combinations = well_plate_counts[well_plate_counts >= 2].index.tolist()
                if len(valid_combinations) < 2:
                    print("    Not enough valid (group, well) combinations across plates")
                    results["well_position_effect"] = {
                        "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0
                    }
                else:
                    df_well = df_treatments[
                        df_treatments.apply(lambda r: (r[group_col], r[well_col]) in valid_combinations, axis=1)
                    ].copy()
            else:
                well_plate_counts = df_treatments.groupby(well_col)[batch_col].nunique()
                valid_well_locs = well_plate_counts[well_plate_counts >= 2].index.tolist()
                if len(valid_well_locs) < 2:
                    results["well_position_effect"] = {
                        "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0
                    }
                    df_well = None
                else:
                    df_well = df_treatments[df_treatments[well_col].isin(valid_well_locs)].copy()
        else:
            # Exact mode: need at least 2 different compounds per well location
            if has_groups:
                well_loc_counts = df_treatments.groupby([group_col, well_col])[compound_col].nunique()
                valid_combinations = well_loc_counts[well_loc_counts >= 2].index.tolist()
                if len(valid_combinations) < 2:
                    results["well_position_effect"] = {
                        "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0
                    }
                    df_well = None
                else:
                    df_well = df_treatments[
                        df_treatments.apply(lambda r: (r[group_col], r[well_col]) in valid_combinations, axis=1)
                    ].copy()
            else:
                well_loc_counts = df_treatments.groupby(well_col)[compound_col].nunique()
                valid_well_locs = well_loc_counts[well_loc_counts >= 2].index.tolist()
                if len(valid_well_locs) < 2:
                    results["well_position_effect"] = {
                        "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0
                    }
                    df_well = None
                else:
                    df_well = df_treatments[df_treatments[well_col].isin(valid_well_locs)].copy()

        if "well_position_effect" not in results and df_well is not None and len(df_well) > 0:
            metadata_well = df_well.filter(regex="^Metadata|^_random_index")
            profiles_well = df_well[features].values

            well_ap = copairs_map.average_precision(
                metadata_well, profiles_well,
                pos_sameby_well, pos_diffby_well,
                neg_sameby_well, neg_diffby_well
            )

            if len(well_ap) > 0:
                well_map = copairs_map.mean_average_precision(
                    well_ap, pos_sameby_well,
                    null_size=null_size, threshold=p_threshold, seed=seed
                )
                well_map["below_corrected_p"] = well_map["corrected_p_value"] < p_threshold

                pct_active = (well_map["below_corrected_p"].sum() / len(well_map)) * 100 if len(well_map) > 0 else 0
                mean_map_val = well_map["mean_average_precision"].mean() if len(well_map) > 0 else 0
                mean_nap_val = well_map["mean_normalized_average_precision"].mean() if len(well_map) > 0 else 0

                result_dict = {
                    "pct_active": float(pct_active),
                    "n_active": int(well_map["below_corrected_p"].sum()),
                    "n_total": int(len(well_map)),
                    "mean_map": float(mean_map_val),
                    "mean_nap": float(mean_nap_val),
                    "n_samples_used": int(len(df_well)),
                }

                if has_groups:
                    per_group_stats = {}
                    for grp in well_map[group_col].unique():
                        grp_data = well_map[well_map[group_col] == grp]
                        grp_active = grp_data["below_corrected_p"].sum()
                        grp_total = len(grp_data)
                        grp_pct = (grp_active / grp_total * 100) if grp_total > 0 else 0
                        per_group_stats[grp] = {
                            "pct_active": float(grp_pct),
                            "n_active": int(grp_active),
                            "n_total": int(grp_total),
                            "mean_map": float(grp_data["mean_average_precision"].mean()),
                            "mean_nap": float(grp_data["mean_normalized_average_precision"].mean()),
                        }
                    result_dict["per_group"] = per_group_stats

                results["well_position_effect"] = result_dict
                print(f"    Well Position Effect: {pct_active:.2f}%")
            else:
                results["well_position_effect"] = {
                    "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0
                }

    except Exception as e:
        print(f"    Warning: Well position effect failed: {e}")
        results["well_position_effect"] = {
            "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0, "error": str(e)
        }

    # =========================================
    # 2. Plate Batch Effect
    # =========================================
    print("  Calculating Plate Batch Effect...")
    try:
        n_plates = df_treatments[batch_col].nunique()

        if n_plates < 2:
            print(f"    Not enough plates ({n_plates})")
            results["plate_batch_effect"] = {
                "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0
            }
        else:
            metadata_treat = df_treatments.filter(regex="^Metadata|^_random_index")
            profiles_treat = df_treatments[features].values

            if mode == "fast":
                if has_groups:
                    pos_sameby_plate = [group_col, batch_col, "_random_index_pos"]
                    neg_sameby_plate = [group_col, "_random_index_neg"]
                else:
                    pos_sameby_plate = [batch_col, "_random_index_pos"]
                    neg_sameby_plate = ["_random_index_neg"]
                pos_diffby_plate = [compound_col]
                neg_diffby_plate = [batch_col, well_col, compound_col]
            else:
                # Exact mode: use negcons only
                df_negcon = df_pd[df_pd[negcon_col] == True].copy() if negcon_col in df_pd.columns else None
                if df_negcon is not None and len(df_negcon) >= 10:
                    metadata_treat = df_negcon.filter(regex="^Metadata")
                    profiles_treat = df_negcon[features].values
                else:
                    print(f"    Not enough negative controls for exact plate batch analysis")
                    results["plate_batch_effect"] = {
                        "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0
                    }

                if has_groups:
                    pos_sameby_plate = [batch_col]
                    neg_sameby_plate = [well_col]
                else:
                    pos_sameby_plate = [batch_col]
                    neg_sameby_plate = [well_col]
                pos_diffby_plate = []
                neg_diffby_plate = [batch_col]

            if "plate_batch_effect" not in results:
                plate_ap = copairs_map.average_precision(
                    metadata_treat, profiles_treat,
                    pos_sameby_plate, pos_diffby_plate,
                    neg_sameby_plate, neg_diffby_plate
                )

                if len(plate_ap) > 0:
                    map_groupby = [group_col, batch_col] if has_groups and mode == "fast" else [batch_col]
                    plate_map = copairs_map.mean_average_precision(
                        plate_ap, map_groupby,
                        null_size=null_size, threshold=p_threshold, seed=seed
                    )
                    plate_map["below_corrected_p"] = plate_map["corrected_p_value"] < p_threshold

                    pct_active = (plate_map["below_corrected_p"].sum() / len(plate_map)) * 100 if len(plate_map) > 0 else 0
                    mean_map_val = plate_map["mean_average_precision"].mean() if len(plate_map) > 0 else 0
                    mean_nap_val = plate_map["mean_normalized_average_precision"].mean() if len(plate_map) > 0 else 0

                    result_dict = {
                        "pct_active": float(pct_active),
                        "n_active": int(plate_map["below_corrected_p"].sum()),
                        "n_total": int(len(plate_map)),
                        "mean_map": float(mean_map_val),
                        "mean_nap": float(mean_nap_val),
                        "n_plates": int(n_plates),
                        "n_samples": int(len(metadata_treat)),
                    }

                    if has_groups and group_col in plate_map.columns:
                        per_group_stats = {}
                        for grp in plate_map[group_col].unique():
                            grp_data = plate_map[plate_map[group_col] == grp]
                            grp_active = grp_data["below_corrected_p"].sum()
                            grp_total = len(grp_data)
                            grp_pct = (grp_active / grp_total * 100) if grp_total > 0 else 0
                            per_group_stats[grp] = {
                                "pct_active": float(grp_pct),
                                "n_active": int(grp_active),
                                "n_total": int(grp_total),
                                "mean_map": float(grp_data["mean_average_precision"].mean()),
                                "mean_nap": float(grp_data["mean_normalized_average_precision"].mean()),
                            }
                        result_dict["per_group"] = per_group_stats

                    results["plate_batch_effect"] = result_dict
                    print(f"    Plate Batch Effect: {pct_active:.2f}% ({n_plates} plates)")
                else:
                    results["plate_batch_effect"] = {
                        "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0
                    }

    except Exception as e:
        error_msg = str(e)
        if "No data left" in error_msg:
            print("    Warning: Plate batch effect failed: not enough valid pairs")
        else:
            print(f"    Warning: Plate batch effect failed: {e}")
        results["plate_batch_effect"] = {
            "pct_active": 0.0, "n_active": 0, "n_total": 0, "mean_map": 0.0, "mean_nap": 0.0, "error": error_msg
        }

    return results
```

**Step 2: Verify no syntax errors**

Run: `nix develop . --command uv run python -c "from norm_3.metrics import calculate_batch_effects; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/norm_3/metrics.py
git commit -m "feat: add calculate_batch_effects() to norm_3 metrics"
```

---

### Task 2: Integrate batch effects into `evaluate_all()`

**Files:**
- Modify: `src/norm_3/metrics.py:414-637` (the `evaluate_all` function)

**Step 1: Add new parameters to `evaluate_all` signature**

Add after `pc_groups` parameter (line 427):
```python
    skip_batch_effects: bool = True,
    batch_effects_mode: str = "fast",
    well_col: str = "Metadata_Well",
```

Update the docstring Args section to include:
```python
        skip_batch_effects: Skip batch effect calculation (default True for backward compat)
        batch_effects_mode: "fast" (random sampling) or "exact" (full copairs)
        well_col: Column for well position identifier
```

**Step 2: Add batch effects block inside `evaluate_all`**

Insert after the PCA variance block (after line 626) and before "Add feature space size" (line 628):

```python
    # Batch effects (well position + plate)
    _batch_defaults = {
        "well_effect_pct": None,
        "well_effect_mean_nap": None,
        "well_effect_n_active": None,
        "well_effect_n_total": None,
        "plate_effect_pct": None,
        "plate_effect_mean_nap": None,
        "plate_effect_n_active": None,
        "plate_effect_n_total": None,
    }
    if skip_batch_effects:
        print("  Batch effects: skipped")
        results.update(_batch_defaults)
    else:
        try:
            batch_results = calculate_batch_effects(
                df, features,
                null_size=10_000,
                p_threshold=0.05,
                seed=0,
                mode=batch_effects_mode,
                compound_col=compound_col,
                negcon_col=negcon_col,
                batch_col=batch_col,
                well_col=well_col,
                group_col=group_col,
            )

            well_effect = batch_results.get("well_position_effect") or {}
            plate_effect = batch_results.get("plate_batch_effect") or {}

            results["well_effect_pct"] = well_effect.get("pct_active")
            results["well_effect_mean_nap"] = well_effect.get("mean_nap")
            results["well_effect_n_active"] = well_effect.get("n_active")
            results["well_effect_n_total"] = well_effect.get("n_total")

            results["plate_effect_pct"] = plate_effect.get("pct_active")
            results["plate_effect_mean_nap"] = plate_effect.get("mean_nap")
            results["plate_effect_n_active"] = plate_effect.get("n_active")
            results["plate_effect_n_total"] = plate_effect.get("n_total")

            # Flatten per-group batch effects
            for effect_name, effect_data in [("well_effect", well_effect), ("plate_effect", plate_effect)]:
                per_group = effect_data.get("per_group", {})
                if per_group:
                    results[f"{effect_name}_group_summary"] = per_group

            if output_dir is not None:
                import json as json_mod
                batch_path = output_dir / "batch_effects.json"
                with open(batch_path, "w") as f:
                    json_mod.dump(batch_results, f, indent=2)

        except Exception as e:
            print(f"  Batch effects ERROR: {e}")
            results.update(_batch_defaults)
```

**Step 3: Verify import works**

Run: `nix develop . --command uv run python -c "from norm_3.metrics import evaluate_all; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/norm_3/metrics.py
git commit -m "feat: integrate batch effects into evaluate_all()"
```

---

### Task 3: Update `pipeline.py` to pass batch effects config

**Files:**
- Modify: `src/norm_3/pipeline.py:1201-1227` (the `evaluate_metrics` function)

**Step 1: Add new config params to `evaluate_all()` call**

In `evaluate_metrics()` at line 1212-1225, add three new keyword arguments to the `evaluate_all()` call:

```python
    evaluate_all(
        df,
        features,
        output_dir=output_dir,
        skip_visualization=config.get("skip_visualization", False),
        skip_umap=config.get("skip_umap", False),
        n_top_compounds=config.get("n_top_compounds", 20),
        min_compounds_per_target=config.get("min_compounds_per_target", 3),
        compound_col=config.get("compound_col", "Metadata_pert_iname"),
        target_col=config.get("target_col", "Metadata_target_list"),
        negcon_col=config.get("negcon_col", "Metadata_negcon"),
        batch_col=config.get("batch_col", "Metadata_Plate"),
        pc_groups=config.get("pc_groups", None),
        skip_batch_effects=config.get("skip_batch_effects", True),
        batch_effects_mode=config.get("batch_effects_mode", "fast"),
        well_col=config.get("well_col", "Metadata_Well"),
    )
```

**Step 2: Verify pipeline still loads**

Run: `nix develop . --command uv run python -c "from norm_3.pipeline import evaluate_metrics; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add src/norm_3/pipeline.py
git commit -m "feat: pass batch effects config from pipeline to evaluate_all"
```

---

### Task 4: Enable batch effects in v11_lite sweep configs

**Files:**
- Modify: `src/norm_3/conf/sweep/focused_dl_v11_lite_tvn_efaar.yaml`
- Modify: `src/norm_3/conf/sweep/focused_cp_v11_lite_tvn_efaar.yaml`

**Step 1: Add `skip_batch_effects: false` to DL v11 lite**

In `focused_dl_v11_lite_tvn_efaar.yaml`, the sweep config currently has `skip_visualization: true` at line 57. Add `skip_batch_effects: false` right after it:

```yaml
# === FIXED PARAMETERS ===
skip_visualization: true
skip_batch_effects: false
```

**Step 2: Add `skip_batch_effects: false` to CP v11 lite**

In `focused_cp_v11_lite_tvn_efaar.yaml`, same location (line 61). Add:

```yaml
# === FIXED PARAMETERS ===
skip_visualization: true
skip_batch_effects: false
```

**Step 3: Verify YAML is valid**

Run: `nix develop . --command uv run python -c "import yaml; yaml.safe_load(open('src/norm_3/conf/sweep/focused_dl_v11_lite_tvn_efaar.yaml')); print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add src/norm_3/conf/sweep/focused_dl_v11_lite_tvn_efaar.yaml src/norm_3/conf/sweep/focused_cp_v11_lite_tvn_efaar.yaml
git commit -m "feat: enable batch effects by default in v11_lite sweep configs"
```

---

### Task 5: Update `gather_sweep_results.py` to read batch effect metrics

**Files:**
- Modify: `src/norm_3/gather_sweep_results.py:817-860` (the `load_metrics` function)

**Step 1: Add batch effect keys to the metrics dict**

After line 839 (`"PC_replicable_n_compounds": data.get("PC_replicable_n_compounds"),`), add:

```python
        # Batch effects
        "well_effect_pct": data.get("well_effect_pct"),
        "well_effect_mean_nap": data.get("well_effect_mean_nap"),
        "well_effect_n_active": data.get("well_effect_n_active"),
        "well_effect_n_total": data.get("well_effect_n_total"),
        "plate_effect_pct": data.get("plate_effect_pct"),
        "plate_effect_mean_nap": data.get("plate_effect_mean_nap"),
        "plate_effect_n_active": data.get("plate_effect_n_active"),
        "plate_effect_n_total": data.get("plate_effect_n_total"),
```

**Step 2: Add per-group batch effect flattening**

In the per-group flattening loop (lines 844-855), add batch effect group summaries. After the existing loop, add:

```python
    # Flatten per-group batch effects
    for summary_key, prefix in [
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
```

**Step 3: Verify gather script loads**

Run: `nix develop . --command uv run python -c "from norm_3.gather_sweep_results import load_metrics; print('OK')"`
Expected: May fail if `load_metrics` isn't importable directly — verify manually by checking the function parses correctly.

**Step 4: Commit**

```bash
git add src/norm_3/gather_sweep_results.py
git commit -m "feat: read batch effect metrics in gather_sweep_results"
```

---

### Task 6: Verify backward compatibility

**Step 1: Dry-run import of the full pipeline**

Run: `nix develop . --command uv run python -c "from norm_3.pipeline import STEPS; print(list(STEPS.keys()))"`
Expected: prints list of step names including `evaluate_metrics`

**Step 2: Verify existing v9 preset doesn't run batch effects**

The v9 preset has no `skip_batch_effects` key in evaluate_metrics params. Confirm it defaults to `True`:

Run: `nix develop . --command uv run python -c "
config = {}
skip = config.get('skip_batch_effects', True)
print(f'skip_batch_effects={skip}')
assert skip == True, 'Default should be True'
print('OK: backward compatible')
"`
Expected: `skip_batch_effects=True` then `OK: backward compatible`

**Step 3: Verify v11_lite config sets skip_batch_effects=false**

Run: `nix develop . --command uv run python -c "
import yaml
with open('src/norm_3/conf/sweep/focused_dl_v11_lite_tvn_efaar.yaml') as f:
    config = yaml.safe_load(f)
assert config.get('skip_batch_effects') == False, 'v11_lite should enable batch effects'
print('OK: v11_lite enables batch effects')
"`
Expected: `OK: v11_lite enables batch effects`

**Step 4: Commit (no changes expected — verification only)**

No commit needed for this task.
