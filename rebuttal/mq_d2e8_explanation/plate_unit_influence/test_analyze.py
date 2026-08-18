#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import analyze as a


class PlateUnitInfluenceTests(unittest.TestCase):
    def synthetic_sweep(self) -> pd.DataFrame:
        rows = []
        for family, spec in a.FAMILIES.items():
            for i in range(a.EXPECTED_CONFIGS):
                rows.append({
                    "model": spec["sweep_model"], "config": f"cfg_{i:02d}",
                    "PA": 10.0 + (i == 7), "PC": 20.0 + (i == 7),
                    "PA_mean_nap": .1, "PC_mean_nap": .2,
                })
        return pd.DataFrame(rows)

    def test_zstd_selection_covers_five_families(self):
        got = a.select_recipes(self.synthetic_sweep())
        self.assertEqual(set(got.family), set(a.FAMILIES))
        self.assertTrue((got.config == "cfg_07").all())

    def test_zstd_selection_uses_lexical_tie_break(self):
        data = self.synthetic_sweep()
        for model in [s["sweep_model"] for s in a.FAMILIES.values()]:
            data.loc[(data.model == model) & data.config.isin(["cfg_07", "cfg_08"]), ["PA", "PC"]] = [11.0, 21.0]
        got = a.select_recipes(data)
        self.assertTrue((got.config == "cfg_07").all())
        self.assertTrue((got.n_tied == 2).all())

    def test_duplicate_recipe_fails_closed(self):
        data = pd.concat([self.synthetic_sweep(), self.synthetic_sweep().iloc[[0]]], ignore_index=True)
        with self.assertRaises(a.AnalysisError):
            a.select_recipes(data)

    def test_aligned_units_reject_key_drift(self):
        pa1 = pd.DataFrame({"Metadata_broad_sample": range(a.EXPECTED_PA), "mean_normalized_average_precision": 0.1})
        pa2 = pa1.copy(); pa2.loc[0, "Metadata_broad_sample"] = 9999
        pc = pd.DataFrame({"Metadata_target": range(a.EXPECTED_PC), "mean_normalized_average_precision": 0.2})
        with self.assertRaises(a.AnalysisError):
            a.aligned_units("dinov2", {"D2-E8": (pa1, pc), "MQ": (pa2, pc)})

    def test_symmetric_unit_contributions_sum_to_product_delta(self):
        pa = pd.DataFrame({"Metadata_broad_sample": range(a.EXPECTED_PA), "d2_value": np.linspace(-.1, .5, a.EXPECTED_PA), "mq_value": np.linspace(-.08, .47, a.EXPECTED_PA)})
        pc = pd.DataFrame({"Metadata_target": range(a.EXPECTED_PC), "d2_value": np.linspace(-.05, .2, a.EXPECTED_PC), "mq_value": np.linspace(-.03, .18, a.EXPECTED_PC)})
        rows, summary = a.unit_contributions("dinov2", pa, pc)
        expected = pa.mq_value.mean() * pc.mq_value.mean() - pa.d2_value.mean() * pc.d2_value.mean()
        self.assertAlmostEqual(rows.symmetric_product_contribution.sum(), expected, places=14)
        self.assertAlmostEqual(summary["product_delta_mq_minus_d2"], expected, places=14)
        self.assertGreaterEqual(summary["top_10_absolute_share"], summary["top_5_absolute_share"])
        self.assertGreaterEqual(summary["top_20_absolute_share"], summary["top_10_absolute_share"])

    def test_verify_detects_artifact_drift(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / "x.txt").write_text("ok")
            a.write_json(root / "artifact_checksums.json", {"artifacts": [{"path": "x.txt", "size_bytes": 2, "sha256": a.sha256_file(root / "x.txt")}]})
            a.verify(root)
            (root / "x.txt").write_text("bad")
            with self.assertRaises(a.AnalysisError):
                a.verify(root)


if __name__ == "__main__":
    unittest.main()
