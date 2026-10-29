# /// script
# requires-python = "==3.12.12"
# dependencies = ["numpy==1.26.4", "polars==1.35.2", "scipy==1.16.3", "pyarrow==22.0.0",
#   "scikit-learn==1.5.2", "pyyaml==6.0.3", "certifi==2025.10.5"]
# ///
"""Offline, synthetic protocol tests. No biological or GPU parity claims."""

import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "benchmark", ROOT / "examples/uv_jump_benchmark.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    rows = []
    for plate in ["p1", "p2"]:
        for i in range(30):
            control = i < 10
            rows.append(
                dict(
                    zip(
                        m.META,
                        [
                            f"{plate}-{i:02}",
                            "DMSO" if control else f"drug-{i:02}",
                            "group_high",
                            plate,
                            plate,
                            control,
                            "unknown" if control else f"target-{i % 3}",
                        ],
                    )
                )
            )
    meta = m.validate_metadata(pl.DataFrame(rows))
    x = np.random.default_rng(1234).normal(size=(len(meta), 8))
    return x, meta


def recipe(tvn=True, scaler="normalize_standardize", controls=False):
    steps = [
        {"name": "clean_nans", "enabled": True, "params": {"na_cutoff": 0.3}},
        {
            "name": "filter_features",
            "enabled": True,
            "params": {
                "filters": [
                    {"name": "variance_threshold", "freq_cut": 0.05, "unique_cut": 0.01}
                ]
            },
        },
        {
            "name": scaler,
            "enabled": True,
            "params": {"fit_on_controls": controls, "batch_col": m.PLATE},
        },
    ]
    if tvn:
        steps.append(
            {
                "name": "normalize_tvn_efaar",
                "enabled": True,
                "params": {"batch_col": m.BATCH, "epsilon": 0.5, "n_components": 4},
            }
        )
    return {"steps": steps}


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.x, self.meta = fixture()
        self.split = m.split_treatments(self.meta)

    def test_split_matches_historical_implementation(self):
        # Compile only the repository's original split helpers, not its runtime.
        tree = ast.parse(
            (ROOT / "rebuttal/postprocessing_validation/run_analysis.py").read_text()
        )
        names = {"stable_assignment_key", "make_treatment_split"}
        nodes = [
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        from typing import Mapping, Iterable, Any
        import hashlib
        import math

        ns = dict(
            Mapping=Mapping,
            Iterable=Iterable,
            Any=Any,
            hashlib=hashlib,
            math=math,
            AnalysisError=ValueError,
        )
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), "original-split", "exec"),
            ns,
        )
        memberships = {
            i: {"group_high"} for i in self.meta.filter(~pl.col(m.CONTROL))[m.PERT]
        }
        expected = ns["make_treatment_split"](memberships)
        for row in expected:
            got = self.split[row["treatment_id"]]
            self.assertEqual(got["assignment_hash"], row["assignment_hash"])
            self.assertEqual(
                got["role"], "selection" if row["split"] == "validation" else "test"
            )

    def test_split_order_independent_and_cross_group_disjoint(self):
        extra = self.meta.filter(~pl.col(m.CONTROL)).with_columns(
            (pl.col(m.ID) + "-other").alias(m.ID), pl.lit("group_low").alias(m.GROUP)
        )
        combined = pl.concat([self.meta, extra])
        split = m.split_treatments(combined)
        self.assertEqual(split, m.split_treatments(combined.reverse()))
        self.assertTrue(
            all(r["stratum"] == "group_high|group_low" for r in split.values())
        )

    def test_strict_state_excludes_heldout_values(self):
        changed = self.x.copy()
        heldout = np.array(
            [self.split.get(i, {}).get("role") == "test" for i in self.meta[m.PERT]]
        )
        changed[heldout] = changed[heldout] * 1000 + 99
        a, _, sa, audit = m.fit_recipe(
            self.x, self.meta, recipe(), "strict", self.split
        )
        b, _, sb, _ = m.fit_recipe(changed, self.meta, recipe(), "strict", self.split)
        self.assertEqual(m.state_digest(sa), m.state_digest(sb))
        np.testing.assert_allclose(a[~heldout], b[~heldout], rtol=0, atol=0)
        self.assertGreater(audit["test_rows_excluded_from_fit"], 0)
        self.assertFalse(
            set(audit["fit_ids"]) & set(self.meta.filter(pl.Series(heldout))[m.ID])
        )

    def test_paper_fits_declared_all_rows(self):
        changed = self.x.copy()
        changed[-1] += 100
        _, _, a, audit = m.fit_recipe(self.x, self.meta, recipe(), "paper-v11", {})
        _, _, b, _ = m.fit_recipe(changed, self.meta, recipe(), "paper-v11", {})
        self.assertNotEqual(m.state_digest(a), m.state_digest(b))
        self.assertEqual(audit["fit_rows"], len(self.meta))
        self.assertEqual(audit["test_rows_excluded_from_fit"], 0)

    def test_transform_reuses_fitted_state(self):
        for target in m.TARGETS:
            x, meta, states, _ = m.fit_recipe(
                self.x, self.meta, recipe(), target, self.split
            )
            applied, matched = m.apply_recipe(self.x, self.meta, states)
            np.testing.assert_allclose(x, applied, rtol=1e-12, atol=1e-12)
            self.assertTrue(meta.equals(matched))

    def test_cleaning_fit_only(self):
        x = self.x.copy()
        test = np.array(
            [self.split.get(i, {}).get("role") == "test" for i in self.meta[m.PERT]]
        )
        selected = np.flatnonzero(test)[:2]
        x[selected, 0] = np.nan
        _, _, states, audit = m.fit_recipe(
            x, self.meta, recipe(False), "strict", self.split
        )
        self.assertTrue(states[0]["keep"][0])
        self.assertEqual(
            set(audit["excluded_nonfinite_ids"]),
            {self.meta[m.ID][int(i)] for i in selected},
        )

    def test_cleaning_cannot_empty_test_role(self):
        x = self.x.copy()
        test = np.array(
            [self.split.get(i, {}).get("role") == "test" for i in self.meta[m.PERT]]
        )
        x[test, :] = np.nan
        with self.assertRaisesRegex(ValueError, "test role"):
            m.fit_recipe(x, self.meta, recipe(), "strict", self.split)

    def test_controls_only_scaling_ignores_selection_values(self):
        conf = recipe(False, controls=True)
        _, _, a, _ = m.fit_recipe(self.x, self.meta, conf, "strict", self.split)
        changed = self.x.copy()
        changed[~self.meta[m.CONTROL].to_numpy()] += 1000
        _, _, b, _ = m.fit_recipe(changed, self.meta, conf, "strict", self.split)
        self.assertEqual(m.state_digest(a[-1]), m.state_digest(b[-1]))

    def test_robustmad_matches_formula(self):
        conf = recipe(False, scaler="normalize_robustmad", controls=True)
        output, meta, _, _ = m.fit_recipe(self.x, self.meta, conf, "paper-v11", {})
        for plate in ["p1", "p2"]:
            mask = meta[m.PLATE].to_numpy() == plate
            fit = self.x[mask & meta[m.CONTROL].to_numpy()]
            center = np.median(fit, axis=0)
            expected = (self.x[mask] - center) / (
                np.median(abs(fit - center), axis=0) + 1e-18
            )
            np.testing.assert_array_equal(output[mask], expected)

    def test_unknown_step_not_silently_disabled(self):
        conf = recipe()
        conf["steps"].append({"name": "invented", "enabled": True})
        with self.assertRaisesRegex(ValueError, "Unsupported enabled"):
            m.enabled_steps(conf)

    def test_disabled_steps_do_not_change_effective_signature(self):
        conf = recipe()
        other = copy.deepcopy(conf)
        other["steps"].append({"name": "invented", "enabled": False})
        other["unused_top_level_flag"] = True
        self.assertEqual(
            m.digest(m.enabled_steps(conf)), m.digest(m.enabled_steps(other))
        )

    def test_unresolved_recipe_rejected(self):
        conf = recipe()
        conf["steps"][0]["params"]["na_cutoff"] = "${missing}"
        with self.assertRaisesRegex(ValueError, "resolved"):
            m.enabled_steps(conf)

    def test_int_and_correlation_replay(self):
        conf = recipe(False)
        conf["steps"] += [
            {"name": "inverse_normal_transform", "params": {}},
            {"name": "prune_correlated", "params": {"threshold": 0.8}},
        ]
        x, _, state, _ = m.fit_recipe(self.x, self.meta, conf, "strict", self.split)
        result, _ = m.apply_recipe(self.x, self.meta, state)
        np.testing.assert_array_equal(x, result)

    def test_tvn_dimension_clamp(self):
        controls = np.zeros(len(self.meta), dtype=bool)
        controls[[0, 1, 2, 30, 31, 32]] = True
        output, state = m.fit_tvn(
            self.x,
            controls,
            self.meta[m.BATCH].to_numpy(),
            {"n_components": 100, "dim_ratio_threshold": 1.0},
        )
        self.assertEqual(output.shape[1], 2)
        self.assertEqual(state["control_counts"], {"p1": 3, "p2": 3})

    def test_unseen_plate_rejected(self):
        _, _, states, _ = m.fit_recipe(
            self.x, self.meta, recipe(), "strict", self.split
        )
        changed = self.meta.with_columns(pl.lit("unknown").alias(m.PLATE))
        with self.assertRaisesRegex(ValueError, "Unseen transform plate"):
            m.apply_recipe(self.x, changed, states)

    def test_missing_batch_controls_rejected(self):
        mask = self.meta[m.CONTROL].to_numpy() & (self.meta[m.BATCH].to_numpy() == "p1")
        with self.assertRaisesRegex(ValueError, "Every batch"):
            m.fit_tvn(self.x, mask, self.meta[m.BATCH].to_numpy(), {})

    def test_bad_metadata_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            m.validate_metadata(pl.concat([self.meta, self.meta.head(1)]))
        with self.assertRaisesRegex(ValueError, "typed booleans"):
            m.validate_metadata(
                self.meta.with_columns(pl.col(m.CONTROL).cast(pl.String))
            )

    def test_winner_uses_product_and_lexical_ties(self):
        values = [
            {"recipe": name, "pa_mean_nap": a, "pc_mean_nap": c}
            for name, a, c in [("b", 1.0, 1.0), ("a", 1.0, 1.0), ("c", 0.0, 2.0)]
        ]
        ranked = m.choose_winner(values)
        self.assertEqual(ranked[0]["recipe"], "a")
        self.assertEqual(m.choose_winner(values[:2])[0]["selection_score"], 1.0)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.identity = {"target": "strict", "cohort_sha256": "fixed"}
        self.scores = [
            {
                "codec": "Raw",
                "recipe": "a",
                "pa_mean_nap": 0.2,
                "pc_mean_nap": 0.3,
                "pa_pc_product": 0.06,
                "wells": 10,
                "controls": 3,
                "features": 5,
                "pa_units": 4,
                "pc_targets": 2,
            }
        ]
        self.reference = {
            "protocol": self.identity,
            "scores": copy.deepcopy(self.scores),
            "reference_kind": "local_cpu_replay_not_published",
        }

    def test_match(self):
        self.assertEqual(
            m.validate_reference(self.reference, self.identity, self.scores)["status"],
            "match",
        )

    def test_cross_target_refused(self):
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            m.validate_reference(
                self.reference, self.identity | {"target": "paper-v11"}, self.scores
            )

    def test_mismatched_metrics_reported(self):
        self.reference["scores"][0]["pa_mean_nap"] = 0.4
        result = m.validate_reference(self.reference, self.identity, self.scores)
        self.assertEqual(result["status"], "mismatch")
        self.assertEqual(len(result["differences"]), 1)

    def test_population_mismatch_rejected(self):
        self.reference["scores"][0]["wells"] = 9
        with self.assertRaisesRegex(ValueError, "population"):
            m.validate_reference(self.reference, self.identity, self.scores)

    def test_nonfinite_reference_rejected(self):
        self.reference["scores"][0]["pa_mean_nap"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            m.validate_reference(self.reference, self.identity, self.scores)

    def test_input_hash_and_create_only(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "input.json"
            p.write_text("{}")
            record = {"path": p.name, "sha256": m.sha(p)}
            self.assertEqual(m.artifact(record, Path(folder), 100), p)
            p.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "checksum"):
                m.artifact(record, Path(folder), 100)
            with self.assertRaisesRegex(ValueError, "Output"):
                m.safe_output(Path(folder))

    def test_fit_artifacts_use_nonpickle_arrays(self):
        x, meta = fixture()
        transformed, meta, states, audit = m.fit_recipe(
            x, meta, recipe(), "strict", m.split_treatments(meta)
        )
        with tempfile.TemporaryDirectory() as root:
            out = Path(root) / "fit"
            m.save_fit(out, transformed, meta, states, audit)
            with np.load(out / "state.npz", allow_pickle=False) as arrays:
                self.assertTrue(arrays.files)
                self.assertTrue(all(arrays[k].dtype != object for k in arrays.files))
            self.assertEqual(
                json.loads((out / "fit_audit.json").read_text())["state_sha256"],
                m.state_digest(states),
            )


if __name__ == "__main__":
    unittest.main()
