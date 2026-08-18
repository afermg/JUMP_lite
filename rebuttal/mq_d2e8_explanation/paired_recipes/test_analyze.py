#!/usr/bin/env python3
"""Focused tests for the paired MQ-versus-D2-E8 recipe audit."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

MODULE = Path(__file__).with_name("analyze.py")
spec = importlib.util.spec_from_file_location("paired_recipes_analyze", MODULE)
assert spec and spec.loader
analysis = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analysis
spec.loader.exec_module(analysis)


def synthetic_grid() -> pd.DataFrame:
    rows = []
    for family, codecs in analysis.FAMILY_MODELS.items():
        family_shift = list(analysis.FAMILY_MODELS).index(family) * 0.001
        for codec, model in codecs.items():
            for index in range(analysis.EXPECTED_RECIPES):
                pa = 0.2 + index / 10_000
                pc = 0.1 + family_shift
                if codec == "MQ":
                    pc += 0.001 if family == "SubCell" else -0.001
                rows.append(
                    {
                        "model": model,
                        "config": f"recipe_{index:02d}",
                        "PA_mean_nap": pa,
                        "PC_mean_nap": pc,
                    }
                )
    return pd.DataFrame(rows)


class GridValidationTests(unittest.TestCase):
    def test_complete_grid_pairs_exact_recipes(self) -> None:
        selected = analysis.select_and_validate_grid(synthetic_grid())
        pairs = analysis.build_pairs(selected)
        self.assertEqual(len(selected), 480)
        self.assertEqual(len(pairs), 240)
        self.assertEqual(pairs.groupby("family")["config"].nunique().to_dict(), {
            family: 48 for family in analysis.FAMILY_MODELS
        })

    def test_duplicate_model_config_fails_closed(self) -> None:
        source = synthetic_grid()
        source = pd.concat([source, source.iloc[[0]]], ignore_index=True)
        with self.assertRaises(analysis.AnalysisError):
            analysis.select_and_validate_grid(source)

    def test_missing_recipe_fails_closed(self) -> None:
        source = synthetic_grid().iloc[1:].copy()
        with self.assertRaises(analysis.AnalysisError):
            analysis.select_and_validate_grid(source)

    def test_nonfinite_metric_fails_closed(self) -> None:
        source = synthetic_grid()
        source.loc[0, "PA_mean_nap"] = np.nan
        with self.assertRaises(analysis.AnalysisError):
            analysis.select_and_validate_grid(source)


class SummaryTests(unittest.TestCase):
    def test_delta_direction_and_pa_pc_decomposition(self) -> None:
        grid = analysis.select_and_validate_grid(synthetic_grid())
        pairs = analysis.build_pairs(grid)
        family, pooled = analysis.summarize(grid, pairs)
        subcell = family.set_index("family").loc["SubCell"]
        morphem = family.set_index("family").loc["MorphEM"]
        self.assertGreater(subcell["paired_nap_product_mean_delta"], 0)
        self.assertLess(morphem["paired_nap_product_mean_delta"], 0)
        self.assertTrue(np.allclose(pairs["delta_PA_mean_nap"], 0))
        self.assertEqual(int(pooled.iloc[0]["n_paired_rows"]), 240)

    def test_production_input_identity_and_expected_result(self) -> None:
        identity = analysis.validate_production_input(analysis.DEFAULT_INPUT)
        self.assertEqual(identity["sha256"], analysis.EXPECTED_INPUT_SHA256)
        source = pd.read_csv(analysis.DEFAULT_INPUT)
        grid = analysis.select_and_validate_grid(source)
        pairs = analysis.build_pairs(grid)
        family, pooled = analysis.summarize(grid, pairs)
        row = pooled.iloc[0]
        self.assertEqual(len(grid), 480)
        self.assertEqual(len(pairs), 240)
        self.assertAlmostEqual(row["d2e8_marginal_product_median"], 0.02250418913194207)
        self.assertAlmostEqual(row["mq_marginal_product_median"], 0.024059920431485162)
        self.assertAlmostEqual(row["paired_product_mean_delta"], -0.0013226519441399238)
        self.assertAlmostEqual(row["paired_product_median_delta"], -0.0007706092520828774)
        self.assertAlmostEqual(row["paired_product_mq_greater_fraction"], 0.4)
        self.assertTrue(bool(row["pooled_median_inversion"]))
        self.assertEqual(family.loc[family.family == "MorphEM", "paired_nap_product_mq_greater_fraction"].item(), 0)


class ChecksumTests(unittest.TestCase):
    def test_checksum_inventory_rejects_drift_and_extra(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("stable")
            analysis.write_checksums(root)
            analysis.verify_checksums(root)
            (root / "a.txt").write_text("drift")
            with self.assertRaises(analysis.AnalysisError):
                analysis.verify_checksums(root)

    def test_checksum_inventory_rejects_unsafe_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "artifact_checksums.json").write_text(
                json.dumps({"artifacts": [{"path": "../escape", "size_bytes": 0, "sha256": "x"}]})
            )
            with self.assertRaises(analysis.AnalysisError):
                analysis.verify_checksums(root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
