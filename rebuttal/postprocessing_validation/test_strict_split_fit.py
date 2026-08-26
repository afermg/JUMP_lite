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
    apply_fitted_recipe,
    build_annotation_table,
    candidate_cache_key,
    canonicalize_feature_schema,
    checkpoint_identity,
    discover_effective_recipes,
    effective_recipe_signature,
    ensure_create_only,
    exact_fit_ids_hash,
    fit_transform_recipe,
    require_canonical_schema,
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
        self.assertEqual(len(discover_effective_recipes("cellprofiler")), 280)
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
