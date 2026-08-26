from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from rebuttal.postprocessing_validation.run_analysis import (
    AnalysisError,
    minmax_score_candidates,
)
from norm_3.core import RobustMAD_CPU, StandardScaler_CPU
from rebuttal.postprocessing_validation.strict_split_fit import (
    EffectiveRecipe,
    _apply_plate_scalers,
    _checkpoint,
    _fit_plate_scalers,
    _gpu,
    _load_checkpoint,
    _variance_keep,
    apply_fitted_recipe,
    build_annotation_table,
    candidate_cache_key,
    canonicalize_feature_schema,
    checkpoint_identity,
    complete_recipe_from_prefix,
    contiguous_prefix_groups,
    discover_effective_recipes,
    effective_recipe_signature,
    ensure_create_only,
    exact_fit_ids_hash,
    fit_empirical_int,
    fit_recipe_prefix,
    fit_transform_recipe,
    prefix_cache_signature,
    recipe_prefix_behavior_sha256,
    require_canonical_schema,
    score_partition,
    transform_empirical_int,
    validate_projected_target_labels,
)


def _recipe(method: str) -> EffectiveRecipe:
    step_name = f"normalize_{method}"
    config = {
        "steps": [
            {"name": "clean_nans", "enabled": True, "params": {"na_cutoff": 0.3}},
            {
                "name": step_name,
                "enabled": True,
                "params": {"batch_col": "Metadata_Plate", "fit_on_controls": False},
            },
            {"name": "evaluate_metrics", "enabled": True, "params": {}},
        ]
    }
    return EffectiveRecipe(
        family="cell_count",
        canonical_name=method,
        aliases=(method,),
        signature=effective_recipe_signature(config),
        config=config,
        config_paths=(Path(f"{method}.yaml"),),
    )


def _frame() -> tuple[pl.DataFrame, dict[str, str]]:
    rows = []
    split: dict[str, str] = {}
    index = 0
    for plate_index, plate in enumerate(("P1", "P2")):
        for control_index in range(3):
            rows.append(
                {
                    "Metadata_id": f"w{index}",
                    "Metadata_Plate": plate,
                    "Metadata_Batch": f"B{plate_index}",
                    "Metadata_JCP2022": "neg",
                    "Metadata_Group": "group_high",
                    "Metadata_negcon": True,
                    "Metadata_RefChemDB_target": "unknown",
                    "f0": float(control_index + plate_index),
                    "f1": float(2 * control_index - plate_index),
                }
            )
            index += 1
        for role, treatment, base in (
            ("validation", f"v{plate_index}", 5.0),
            ("test", f"t{plate_index}", 20.0),
        ):
            split[treatment] = role
            for replicate in range(4):
                rows.append(
                    {
                        "Metadata_id": f"w{index}",
                        "Metadata_Plate": plate,
                        "Metadata_Batch": f"B{plate_index}",
                        "Metadata_JCP2022": treatment,
                        "Metadata_Group": "group_high",
                        "Metadata_negcon": False,
                        "Metadata_RefChemDB_target": "T1",
                        "f0": base + replicate,
                        "f1": base - replicate,
                    }
                )
                index += 1
    return pl.DataFrame(rows), split


def _int_recipe() -> EffectiveRecipe:
    config = {
        "steps": [
            {"name": "clean_nans", "enabled": True, "params": {"na_cutoff": 0.3}},
            {
                "name": "filter_features",
                "enabled": True,
                "params": {
                    "filters": [
                        {
                            "name": "variance_threshold",
                            "freq_cut": 0.05,
                            "unique_cut": 0.01,
                        }
                    ]
                },
            },
            {"name": "inverse_normal_transform", "enabled": True, "params": {}},
            {"name": "evaluate_metrics", "enabled": True, "params": {}},
        ]
    }
    return EffectiveRecipe(
        family="cell_count",
        canonical_name="variance_int",
        aliases=("variance_int",),
        signature=effective_recipe_signature(config),
        config=config,
        config_paths=(Path("variance_int.yaml"),),
    )


def _int_frame() -> tuple[pl.DataFrame, dict[str, str]]:
    rows = []
    split: dict[str, str] = {}
    index = 0
    for plate_index, plate in enumerate(("P1", "P2")):
        for control_index in range(5):
            rows.append(
                {
                    "Metadata_id": f"w{index}",
                    "Metadata_Plate": plate,
                    "Metadata_Batch": f"B{plate_index}",
                    "Metadata_JCP2022": "neg",
                    "Metadata_Group": "group_high",
                    "Metadata_negcon": True,
                    "Metadata_RefChemDB_target": "unknown",
                    "f0": float(control_index + plate_index * 0.1),
                    "f1": float(2 * control_index - plate_index * 0.2),
                    "f2": float(control_index % 2),
                }
            )
            index += 1
        for treatment_index in range(6):
            treatment = f"v{treatment_index}"
            split[treatment] = "validation"
            for replicate in range(4):
                rows.append(
                    {
                        "Metadata_id": f"w{index}",
                        "Metadata_Plate": plate,
                        "Metadata_Batch": f"B{plate_index}",
                        "Metadata_JCP2022": treatment,
                        "Metadata_Group": "group_high",
                        "Metadata_negcon": False,
                        "Metadata_RefChemDB_target": f"T{treatment_index // 3}",
                        "f0": float(
                            10 * treatment_index + replicate + plate_index * 0.1
                        ),
                        "f1": float(treatment_index - replicate - plate_index * 0.2),
                        "f2": float((treatment_index + replicate) % 2),
                    }
                )
                index += 1
        treatment = f"t{plate_index}"
        split[treatment] = "test"
        for replicate in range(4):
            rows.append(
                {
                    "Metadata_id": f"w{index}",
                    "Metadata_Plate": plate,
                    "Metadata_Batch": f"B{plate_index}",
                    "Metadata_JCP2022": treatment,
                    "Metadata_Group": "group_high",
                    "Metadata_negcon": False,
                    "Metadata_RefChemDB_target": "T2",
                    "f0": float(100 + replicate),
                    "f1": float(50 - replicate),
                    "f2": float(replicate % 2),
                }
            )
            index += 1
    return pl.DataFrame(rows), split


def _cached_recipe(epsilon: float) -> EffectiveRecipe:
    base = _int_recipe()
    config = deepcopy(base.config)
    config["steps"].insert(
        -1,
        {
            "name": "normalize_tvn_efaar",
            "enabled": True,
            "params": {
                "n_components": 2,
                "dim_ratio_threshold": 2.5,
                "epsilon": epsilon,
            },
        },
    )
    name = f"variance_int_tvn_{epsilon}"
    return EffectiveRecipe(
        family="cell_count",
        canonical_name=name,
        aliases=(name,),
        signature=effective_recipe_signature(config),
        config=config,
        config_paths=(Path(f"{name}.yaml"),),
    )


def _selection_proxy(transformed: pl.DataFrame, split: dict[str, str]) -> float:
    validation = [key for key, value in split.items() if value == "validation"]
    return float(
        transformed.filter(pl.col("Metadata_JCP2022").is_in(validation))["f0"].mean()
    )


class StrictSplitFitTests(unittest.TestCase):
    def test_leakage_sentinel_test_values_do_not_change_fit_or_selection(self) -> None:
        frame, split = _frame()
        perturbed = frame.with_columns(
            pl.when(pl.col("Metadata_JCP2022").str.starts_with("t"))
            .then(pl.col("f0") * 1000 + 777)
            .otherwise(pl.col("f0"))
            .alias("f0"),
            pl.when(pl.col("Metadata_JCP2022").str.starts_with("t"))
            .then(pl.col("f1") * -900 - 333)
            .otherwise(pl.col("f1"))
            .alias("f1"),
        )
        rows_a = []
        rows_b = []
        heldout_changed = []
        for recipe in (_recipe("standardize"), _recipe("robustmad")):
            out_a, state_a = fit_transform_recipe(
                frame, ["f0", "f1"], recipe, split, "Raw"
            )
            out_b, state_b = fit_transform_recipe(
                perturbed, ["f0", "f1"], recipe, split, "Raw"
            )
            self.assertEqual(state_a.digest(), state_b.digest())
            validation_ids = [
                key for key, value in split.items() if value == "validation"
            ]
            np.testing.assert_allclose(
                out_a.filter(pl.col("Metadata_JCP2022").is_in(validation_ids))
                .select("f0", "f1")
                .to_numpy(),
                out_b.filter(pl.col("Metadata_JCP2022").is_in(validation_ids))
                .select("f0", "f1")
                .to_numpy(),
            )
            rows_a.append(
                {
                    "config": recipe.canonical_name,
                    "validation_pa_mean_nap": _selection_proxy(out_a, split),
                    "validation_pc_mean_nap": _selection_proxy(out_a, split) ** 2,
                }
            )
            rows_b.append(
                {
                    "config": recipe.canonical_name,
                    "validation_pa_mean_nap": _selection_proxy(out_b, split),
                    "validation_pc_mean_nap": _selection_proxy(out_b, split) ** 2,
                }
            )
            test_ids = [key for key, value in split.items() if value == "test"]
            heldout_changed.append(
                not np.allclose(
                    out_a.filter(pl.col("Metadata_JCP2022").is_in(test_ids))
                    .select("f0", "f1")
                    .to_numpy(),
                    out_b.filter(pl.col("Metadata_JCP2022").is_in(test_ids))
                    .select("f0", "f1")
                    .to_numpy(),
                )
            )
        self.assertEqual(rows_a, rows_b)
        self.assertEqual(
            minmax_score_candidates(rows_a)[0][0]["config"],
            minmax_score_candidates(rows_b)[0][0]["config"],
        )
        self.assertTrue(all(heldout_changed))

    def test_out_of_sample_int_ties_and_tails_are_finite_and_monotone(self) -> None:
        reference = [np.array([0.0, 1.0, 1.0, 2.0])]
        values = np.array([[-100.0], [0.0], [1.0], [1.0], [2.0], [100.0]])
        transformed = transform_empirical_int(values, reference)[:, 0]
        self.assertTrue(np.isfinite(transformed).all())
        self.assertTrue(np.all(np.diff(transformed) >= 0))
        self.assertEqual(transformed[2], transformed[3])
        self.assertEqual(transformed[0], transformed[1])
        self.assertEqual(transformed[-1], transformed[-2])

    def test_cpu_worker_feature_kernels_have_exact_parity(self) -> None:
        rng = np.random.default_rng(20260826)
        X_fit = rng.integers(-4, 5, size=(47, 19)).astype(np.float64)
        X_transform = rng.integers(-8, 9, size=(31, 19)).astype(np.float64)
        fitted_1 = fit_empirical_int(X_fit, cpu_workers=1)
        fitted_8 = fit_empirical_int(X_fit, cpu_workers=8)
        self.assertEqual(len(fitted_1), len(fitted_8))
        for serial, parallel in zip(fitted_1, fitted_8, strict=True):
            np.testing.assert_array_equal(serial, parallel)
        np.testing.assert_array_equal(
            transform_empirical_int(X_transform, fitted_1, cpu_workers=1),
            transform_empirical_int(X_transform, fitted_8, cpu_workers=8),
        )

        cp = _gpu()
        X_gpu = cp.asarray(X_fit)
        fit_mask = np.arange(len(X_fit)) % 3 != 0
        np.testing.assert_array_equal(
            _variance_keep(X_gpu, fit_mask, 0.05, 0.01, cp, cpu_workers=1),
            _variance_keep(X_gpu, fit_mask, 0.05, 0.01, cp, cpu_workers=8),
        )
        with self.assertRaisesRegex(AnalysisError, "at least 1"):
            fit_empirical_int(X_fit, cpu_workers=0)

    def test_cpu_worker_recipe_metrics_state_and_leakage_have_exact_parity(
        self,
    ) -> None:
        frame, split = _int_frame()
        features = ["f0", "f1", "f2"]
        recipe = _int_recipe()
        out_1, state_1 = fit_transform_recipe(
            frame, features, recipe, split, "Raw", cpu_workers=1
        )
        out_8, state_8 = fit_transform_recipe(
            frame, features, recipe, split, "Raw", cpu_workers=8
        )
        self.assertEqual(state_1.retained_features, state_8.retained_features)
        self.assertEqual(state_1.digest(), state_8.digest())
        np.testing.assert_array_equal(
            out_1.select(state_1.retained_features).to_numpy(),
            out_8.select(state_8.retained_features).to_numpy(),
        )
        applied_1 = apply_fitted_recipe(frame, features, state_1, cpu_workers=1)
        applied_8 = apply_fitted_recipe(frame, features, state_1, cpu_workers=8)
        np.testing.assert_array_equal(
            applied_1.select(state_1.retained_features).to_numpy(),
            applied_8.select(state_1.retained_features).to_numpy(),
        )

        validation = {key for key, value in split.items() if value == "validation"}
        metrics_1, pa_1, pc_1 = score_partition(
            out_1, state_1.retained_features, validation
        )
        metrics_8, pa_8, pc_8 = score_partition(
            out_8, state_8.retained_features, validation
        )
        self.assertEqual(metrics_1, metrics_8)
        self.assertTrue(pa_1.equals(pa_8))
        self.assertTrue(pc_1.equals(pc_8))

        perturbed = frame.with_columns(
            pl.when(pl.col("Metadata_JCP2022").str.starts_with("t"))
            .then(pl.col("f0") * 1000 + 777)
            .otherwise(pl.col("f0"))
            .alias("f0")
        )
        perturbed_out, perturbed_state = fit_transform_recipe(
            perturbed, features, recipe, split, "Raw", cpu_workers=8
        )
        self.assertEqual(state_1.digest(), perturbed_state.digest())
        validation_ids = sorted(validation)
        np.testing.assert_array_equal(
            out_1.filter(pl.col("Metadata_JCP2022").is_in(validation_ids))
            .select(state_1.retained_features)
            .to_numpy(),
            perturbed_out.filter(pl.col("Metadata_JCP2022").is_in(validation_ids))
            .select(perturbed_state.retained_features)
            .to_numpy(),
        )

    def test_prefix_grouping_and_bound_signature_preserve_order(self) -> None:
        recipes = [_cached_recipe(0.05), _cached_recipe(0.1), _int_recipe()]
        groups = contiguous_prefix_groups(recipes)
        self.assertEqual(
            [[r.canonical_name for r in group] for group in groups],
            [
                [recipes[0].canonical_name, recipes[1].canonical_name],
                [recipes[2].canonical_name],
            ],
        )
        self.assertEqual(
            recipe_prefix_behavior_sha256(recipes[0]),
            recipe_prefix_behavior_sha256(recipes[1]),
        )
        self.assertIsNone(recipe_prefix_behavior_sha256(recipes[2]))
        fields = {
            "prefix_behavior_sha256": str(recipe_prefix_behavior_sha256(recipes[0])),
            "family": "cell_count",
            "codec": "Raw",
            "input_sha256": "input",
            "source_schema_sha256": "source",
            "canonical_schema_sha256": "canonical",
            "split_sha256": "split",
            "fit_ids_sha256": "fit",
            "code_sha256": "code",
            "protocol_sha256": "protocol",
        }
        reference = prefix_cache_signature(**fields)
        for key in fields:
            changed = dict(fields)
            changed[key] += "-changed"
            self.assertNotEqual(prefix_cache_signature(**changed), reference)

    def test_cached_prefix_matches_uncached_state_outputs_metrics_and_leakage(
        self,
    ) -> None:
        frame, split = _int_frame()
        features = ["f0", "f1", "f2"]
        recipe = _cached_recipe(0.05)
        uncached, uncached_state = fit_transform_recipe(
            frame, features, recipe, split, "Raw", cpu_workers=1
        )
        prefix_frame, prefix_state = fit_recipe_prefix(
            frame, features, recipe, split, "Raw", cpu_workers=1
        )
        cached, cached_state = complete_recipe_from_prefix(
            prefix_frame, prefix_state, recipe, split, cpu_workers=1
        )
        self.assertEqual(
            uncached_state.retained_features, cached_state.retained_features
        )
        self.assertEqual(uncached_state.digest(), cached_state.digest())
        np.testing.assert_array_equal(
            uncached.select(uncached_state.retained_features).to_numpy(),
            cached.select(cached_state.retained_features).to_numpy(),
        )
        validation = {key for key, value in split.items() if value == "validation"}
        uncached_metrics, uncached_pa, uncached_pc = score_partition(
            uncached, uncached_state.retained_features, validation
        )
        cached_metrics, cached_pa, cached_pc = score_partition(
            cached, cached_state.retained_features, validation
        )
        self.assertEqual(uncached_metrics, cached_metrics)
        self.assertTrue(uncached_pa.equals(cached_pa))
        self.assertTrue(uncached_pc.equals(cached_pc))

        second_recipe = _cached_recipe(0.1)
        second_uncached, second_uncached_state = fit_transform_recipe(
            frame, features, second_recipe, split, "Raw", cpu_workers=1
        )
        second_cached, second_cached_state = complete_recipe_from_prefix(
            prefix_frame, prefix_state, second_recipe, split, cpu_workers=1
        )
        self.assertEqual(second_uncached_state.digest(), second_cached_state.digest())
        np.testing.assert_array_equal(
            second_uncached.select(second_uncached_state.retained_features).to_numpy(),
            second_cached.select(second_cached_state.retained_features).to_numpy(),
        )
        second_uncached_metrics, _, _ = score_partition(
            second_uncached, second_uncached_state.retained_features, validation
        )
        second_cached_metrics, _, _ = score_partition(
            second_cached, second_cached_state.retained_features, validation
        )
        self.assertEqual(second_uncached_metrics, second_cached_metrics)
        uncached_rows = [
            {
                "config": recipe.canonical_name,
                "validation_pa_mean_nap": uncached_metrics["pa_mean_nap"],
                "validation_pc_mean_nap": uncached_metrics["pc_mean_nap"],
            },
            {
                "config": second_recipe.canonical_name,
                "validation_pa_mean_nap": second_uncached_metrics["pa_mean_nap"],
                "validation_pc_mean_nap": second_uncached_metrics["pc_mean_nap"],
            },
        ]
        cached_rows = deepcopy(uncached_rows)
        cached_rows[0]["validation_pa_mean_nap"] = cached_metrics["pa_mean_nap"]
        cached_rows[0]["validation_pc_mean_nap"] = cached_metrics["pc_mean_nap"]
        cached_rows[1]["validation_pa_mean_nap"] = second_cached_metrics["pa_mean_nap"]
        cached_rows[1]["validation_pc_mean_nap"] = second_cached_metrics["pc_mean_nap"]
        self.assertEqual(
            minmax_score_candidates(uncached_rows)[0][0]["config"],
            minmax_score_candidates(cached_rows)[0][0]["config"],
        )

        perturbed = frame.with_columns(
            pl.when(pl.col("Metadata_JCP2022").str.starts_with("t"))
            .then(pl.col("f0") * 1000 + 777)
            .otherwise(pl.col("f0"))
            .alias("f0")
        )
        perturbed_prefix, perturbed_prefix_state = fit_recipe_prefix(
            perturbed, features, recipe, split, "Raw", cpu_workers=1
        )
        _, perturbed_state = complete_recipe_from_prefix(
            perturbed_prefix,
            perturbed_prefix_state,
            recipe,
            split,
            cpu_workers=1,
        )
        self.assertEqual(cached_state.digest(), perturbed_state.digest())

    def test_prefix_checkpoint_resume_rejects_signature_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.json"
            identity = {"config": "a", "prefix_signature": "prefix-a"}
            _checkpoint(path, "protocol", {"result": identity}, identity)
            self.assertIsNotNone(_load_checkpoint(path, "protocol", identity))
            drifted = {**identity, "prefix_signature": "prefix-b"}
            with self.assertRaisesRegex(AnalysisError, "identity mismatch"):
                _load_checkpoint(path, "protocol", drifted)

    def test_codec_schema_guards_and_openphenom_mapping(self) -> None:
        canonical, source = canonicalize_feature_schema(
            "openphenom",
            ["openphenom_nahualX_10", "openphenom_nahualX_2", "openphenom_nahualX_1"],
        )
        self.assertEqual(canonical, ["nahualX_1", "nahualX_2", "nahualX_10"])
        self.assertEqual(source["nahualX_2"], "openphenom_nahualX_2")
        raw, raw_source = canonicalize_feature_schema(
            "openphenom", ["nahualX_10", "nahualX_2", "nahualX_1"]
        )
        self.assertEqual(raw, canonical)
        self.assertEqual(set(raw_source), set(source))
        # Physical order may vary because canonicalization explicitly reorders it.
        reordered, _ = canonicalize_feature_schema(
            "openphenom",
            ["openphenom_nahualX_2", "openphenom_nahualX_1", "openphenom_nahualX_10"],
        )
        require_canonical_schema("openphenom", "HQ", canonical, reordered)
        for observed, message in (
            (canonical[:-1], "missing"),
            ([*canonical, "nahualX_11"], "extra"),
            (list(reversed(canonical)), "reordered=True"),
        ):
            with self.assertRaisesRegex(AnalysisError, message):
                require_canonical_schema("openphenom", "HQ", canonical, observed)

    def test_plate_scalers_match_norm3_cpu_semantics(self) -> None:
        cp = _gpu()
        # normal, exactly zero scale, and nonzero scale below 1e-18
        X = np.array(
            [
                [1.0, 4.0, 0.0],
                [2.0, 4.0, 2e-20],
                [5.0, 4.0, 4e-20],
            ],
            dtype=np.float64,
        )
        plates = np.array(["P1"] * len(X))
        fit_mask = np.ones(len(X), dtype=bool)
        for method, epsilon, transformer in (
            ("robustmad", 1e-20, RobustMAD_CPU(epsilon=1e-20)),
            ("standardize", 123.0, StandardScaler_CPU()),
        ):
            state = _fit_plate_scalers(
                cp.asarray(X), plates, fit_mask, method, epsilon, cp
            )
            observed = cp.asnumpy(
                _apply_plate_scalers(cp.asarray(X), plates, state, cp)
            )
            expected = transformer.fit_transform(X)
            np.testing.assert_allclose(observed, expected, rtol=1e-13, atol=0.0)

    def test_projected_metadata_targets_match_archived_labels(self) -> None:
        parity = validate_projected_target_labels(build_annotation_table())
        self.assertEqual(parity["rows"], 163_776)
        self.assertEqual(parity["mismatches"], 0)
        self.assertEqual(
            parity["columns_read"],
            ["Metadata_id", "Metadata_RefChemDB_target"],
        )

    def test_alias_deduplication_inventory_matches_archive(self) -> None:
        for family in ("dinov2", "morphem", "openphenom", "subcell", "dinov2_random"):
            recipes = discover_effective_recipes(family)
            self.assertEqual(len(recipes), 175)
            self.assertEqual(sum(len(recipe.aliases) for recipe in recipes), 350)
            self.assertEqual({len(recipe.aliases) for recipe in recipes}, {2})
        cellprofiler = discover_effective_recipes("cellprofiler")
        self.assertEqual(len(cellprofiler), 280)
        self.assertEqual(len(contiguous_prefix_groups(cellprofiler)), 8)
        self.assertEqual(len(discover_effective_recipes("cell_count")), 5)

    def test_missing_controls_and_unseen_plate_fail_closed(self) -> None:
        frame, split = _frame()
        no_controls = frame.filter(~pl.col("Metadata_negcon"))
        with self.assertRaisesRegex(AnalysisError, "split roles"):
            fit_transform_recipe(
                no_controls, ["f0", "f1"], _recipe("standardize"), split, "Raw"
            )
        _, state = fit_transform_recipe(
            frame, ["f0", "f1"], _recipe("standardize"), split, "Raw"
        )
        unseen = frame.with_columns(pl.lit("P-new").alias("Metadata_Plate"))
        with self.assertRaisesRegex(AnalysisError, "unseen transform plate"):
            apply_fitted_recipe(unseen, ["f0", "f1"], state)

    def test_exact_fit_id_hash_ignores_test_nan_but_binds_fit_nan(self) -> None:
        frame, split = _frame()
        recipe = _recipe("standardize")
        original = exact_fit_ids_hash(frame, ["f0", "f1"], recipe, split)
        test_nan = frame.with_columns(
            pl.when(pl.col("Metadata_JCP2022") == "t0")
            .then(pl.lit(float("nan")))
            .otherwise(pl.col("f0"))
            .alias("f0")
        )
        self.assertEqual(
            original,
            exact_fit_ids_hash(test_nan, ["f0", "f1"], recipe, split),
        )
        validation_nan = frame.with_columns(
            pl.when(pl.col("Metadata_JCP2022") == "v0")
            .then(pl.lit(float("nan")))
            .otherwise(pl.col("f0"))
            .alias("f0")
        )
        self.assertNotEqual(
            original,
            exact_fit_ids_hash(validation_nan, ["f0", "f1"], recipe, split),
        )

    def test_candidate_cache_key_binds_all_required_identities(self) -> None:
        base = [
            "input",
            "split",
            "fit",
            "code",
            "family",
            "Raw",
            "config",
            "recipe",
            "source-schema",
            "canonical-schema",
        ]
        reference = candidate_cache_key(*base)
        self.assertNotEqual(candidate_cache_key(*base, "prefix"), reference)
        for index in range(len(base)):
            changed = deepcopy(base)
            changed[index] += "-changed"
            self.assertNotEqual(candidate_cache_key(*changed), reference)

    def test_create_only_and_checkpoint_protocol_guards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "new"
            ensure_create_only(root)
            self.assertTrue(root.is_dir())
            with self.assertRaises(FileExistsError):
                ensure_create_only(root)
            checkpoint = root / "checkpoint.json"
            _checkpoint(checkpoint, "protocol-a", {"result": {"value": 3}})
            self.assertEqual(
                _load_checkpoint(checkpoint, "protocol-a")["result"]["value"], 3
            )
            with self.assertRaisesRegex(AnalysisError, "protocol mismatch"):
                _load_checkpoint(checkpoint, "protocol-b")

    def test_final_checkpoint_rejects_winner_config_and_input_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "final.json"
            base = checkpoint_identity(
                code_sha256="code",
                family="dinov2",
                codec="HQ",
                config="winner-a",
                effective_signature="signature-a",
                input_sha256="input-a",
                fit_ids_sha256="fit",
                split_sha256="split",
                source_schema_sha256="source",
                canonical_schema_sha256="canonical",
            )
            _checkpoint(path, "protocol", {"result": {}}, base)
            self.assertIsNotNone(_load_checkpoint(path, "protocol", base))
            for field in ("config", "effective_signature", "input_sha256"):
                drifted = dict(base)
                drifted[field] += "-drift"
                with self.assertRaisesRegex(AnalysisError, "identity mismatch"):
                    _load_checkpoint(path, "protocol", drifted)


if __name__ == "__main__":
    unittest.main()
