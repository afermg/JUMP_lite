#!/usr/bin/env python3
"""Focused tests for fixed-recipe MQ-versus-D2-E8 bootstrap."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("fixed_recipe_bootstrap", HERE / "analyze.py")
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class FixedRecipeTests(unittest.TestCase):
    def test_holm_known_values(self) -> None:
        observed = analysis.holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(observed, [0.03, 0.06, 0.06])

    def test_centered_pvalue_is_finite(self) -> None:
        samples = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
        self.assertEqual(analysis.centered_pvalue(samples, 1.0), 1 / 6)

    def test_lexical_tie_resolution(self) -> None:
        rows = []
        for family in analysis.FAMILIES:
            for i in range(48):
                rows.append(
                    {
                        "model": analysis.MODEL[family],
                        "config": "a" if i == 0 else ("b" if i == 1 else f"z{i:02d}"),
                        "PA": 10.0 if i < 2 else 1.0,
                        "PC": 10.0 if i < 2 else 1.0,
                    }
                )
        winners = analysis.select_recipes(pd.DataFrame(rows))
        self.assertEqual(winners, {family: "a" for family in analysis.FAMILIES})

    def test_align_rejects_key_drift(self) -> None:
        left = pd.DataFrame({"id": ["a", "b"], "x": [1.0, 2.0]})
        right = pd.DataFrame({"id": ["a", "c"], "y": [1.0, 2.0]})
        with self.assertRaises(analysis.AnalysisError):
            analysis.align_tables([left, right], "id", 2)

    def test_common_pa_queries_intersect_and_preserve_mapping(self) -> None:
        left = pd.DataFrame(
            {
                "Metadata_id": ["a", "b", "c"],
                "Metadata_broad_sample": ["x", "y", "z"],
                "normalized_average_precision": [0.1, 0.2, 0.3],
            }
        )
        right = pd.DataFrame(
            {
                "Metadata_id": ["b", "c", "d"],
                "Metadata_broad_sample": ["y", "z", "q"],
                "normalized_average_precision": [0.4, 0.5, 0.6],
            }
        )
        result = analysis.restrict_to_common_pa_queries({"left": left, "right": right}, 2)
        self.assertEqual(result["left"].Metadata_id.tolist(), ["b", "c"])
        self.assertEqual(result["right"].Metadata_id.tolist(), ["b", "c"])

    def test_common_pa_queries_reject_mapping_drift(self) -> None:
        left = pd.DataFrame(
            {"Metadata_id": ["a"], "Metadata_broad_sample": ["x"], "normalized_average_precision": [0.1]}
        )
        right = pd.DataFrame(
            {"Metadata_id": ["a"], "Metadata_broad_sample": ["wrong"], "normalized_average_precision": [0.2]}
        )
        with self.assertRaises(analysis.AnalysisError):
            analysis.restrict_to_common_pa_queries({"left": left, "right": right}, 1)

    def test_bootstrap_deterministic_and_paired(self) -> None:
        pa = pd.DataFrame({"Metadata_broad_sample": ["a", "b", "c"]})
        pc = pd.DataFrame({"Metadata_target": ["x", "y"]})
        for family in analysis.FAMILIES:
            for codec, offset in (("Zstd", 0.0), ("D2-E8", 0.0), ("MQ", 0.1)):
                pa[f"{family}__{codec}"] = np.array([0.1, 0.2, 0.3]) + offset
                pc[f"{family}__{codec}"] = np.array([0.2, 0.4])
        scores1, contrasts1 = analysis.bootstrap(pa, pc, 200, 7)
        scores2, contrasts2 = analysis.bootstrap(pa, pc, 200, 7)
        pd.testing.assert_frame_equal(scores1, scores2)
        pd.testing.assert_frame_equal(contrasts1, contrasts2)
        np.testing.assert_allclose(contrasts1.pa_delta_mq_minus_d2e8, 0.1)
        np.testing.assert_allclose(contrasts1.pc_delta_mq_minus_d2e8, 0.0)

    def test_clean_staging_preserves_existing_release_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            output = Path(name) / "release"
            output.mkdir()
            (output / "sentinel").write_text("old")

            def fail(staged: Path) -> None:
                staged.mkdir()
                (staged / "partial").write_text("new")
                raise analysis.AnalysisError("expected")

            with self.assertRaises(analysis.AnalysisError):
                analysis.publish_release(output, fail)
            self.assertEqual((output / "sentinel").read_text(), "old")
            self.assertFalse((output / "partial").exists())

    def test_release_verifier_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "provenance.json").write_text("{}")
            (root / "artifact_checksums.json").write_text('{"artifacts": [{"path": "x", "size_bytes": 1, "sha256": "bad"}]}')
            (root / "x").write_text("x")
            with self.assertRaises(analysis.AnalysisError):
                analysis.verify_release(root)

    def test_production_inputs_validate(self) -> None:
        sweep = analysis.validate_sweep()
        records, pa, pc = analysis.validate_inputs(sweep)
        self.assertEqual(len(records), 15)
        self.assertEqual(pa.shape, (306, 16))
        self.assertEqual(pc.shape, (201, 16))
        self.assertEqual({row["family"] for row in records}, set(analysis.FAMILIES))
        cp = [row for row in records if row["family"] == "cp_measure"]
        self.assertEqual({row["pa_query_rows_common"] for row in cp}, {1263})
        self.assertEqual(
            {row["codec"]: row["pa_query_rows_dropped"] for row in cp},
            {"Zstd": 3, "D2-E8": 0, "MQ": 1},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
