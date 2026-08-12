#!/usr/bin/env python3
"""Focused tests for bootstrap_uncertainty.py."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).with_name("bootstrap_uncertainty.py")
spec = importlib.util.spec_from_file_location("bootstrap_uncertainty", MODULE_PATH)
assert spec and spec.loader
bootstrap = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bootstrap
spec.loader.exec_module(bootstrap)

PROJECT_ROOT = Path("/work/users/amunoz/projects/JUMP_lite")
RESULTS_DIR = PROJECT_ROOT / "rebuttal/postprocessing_validation/results"


def hash_csvs(directory: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.glob("*.csv"))
    }


class HolmTests(unittest.TestCase):
    def test_holm_adjustment(self) -> None:
        adjusted = bootstrap.holm_adjust([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_centered_bootstrap_pvalue(self) -> None:
        samples = np.asarray([1.8, 1.9, 2.0, 2.1, 2.2])
        self.assertEqual(bootstrap.centered_bootstrap_pvalue(samples, 2.0), 1 / 6)


class AlignmentTests(unittest.TestCase):
    def make_frame(self, family: str, codec: str, keys: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "family": family,
                "codec": codec,
                "config": "cfg",
                "Metadata_JCP2022": keys,
                "Metadata_Group": ["group_low"] * len(keys),
                "mean_normalized_average_precision": np.arange(len(keys), dtype=float),
            }
        )

    def test_alignment_uses_keys_not_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            variants = (("model_a", "Raw"), ("model_b", "Raw"))
            paths = {}
            for variant, keys in zip(variants, (["a", "b"], ["b", "a"]), strict=True):
                path = root / f"{variant[0]}__{variant[1]}__pa_treatments.csv"
                self.make_frame(*variant, list(keys)).to_csv(path, index=False)
                paths[variant] = path
            aligned = bootstrap.align_metric_tables("pa_treatments", paths, variants)
            self.assertEqual(aligned.keys["Metadata_JCP2022"].tolist(), ["a", "b"])
            np.testing.assert_array_equal(aligned.values[:, 0], [0.0, 1.0])
            np.testing.assert_array_equal(aligned.values[:, 1], [1.0, 0.0])

    def test_alignment_fails_closed_on_key_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            variants = (("model_a", "Raw"), ("model_b", "Raw"))
            paths = {}
            for variant, keys in zip(variants, (["a", "b"], ["a", "c"]), strict=True):
                path = root / f"{variant[0]}__{variant[1]}__pa_treatments.csv"
                self.make_frame(*variant, list(keys)).to_csv(path, index=False)
                paths[variant] = path
            with self.assertRaises(bootstrap.BootstrapError):
                bootstrap.align_metric_tables("pa_treatments", paths, variants)


class ClusterTests(unittest.TestCase):
    def test_paired_cluster_weights_preserve_identical_difference(self) -> None:
        metric = bootstrap.ClusterMetric(
            metric="synthetic",
            cluster_ids=("a", "b", "c"),
            strata=np.asarray(["x", "x", "y"], dtype=object),
            row_counts=np.asarray([1, 2, 1]),
            value_sums=np.asarray([[1.0, 2.0], [4.0, 8.0], [2.0, 4.0]]),
            n_rows=4,
            observed_memberships=("x", "x", "y"),
        )
        rng = np.random.Generator(np.random.PCG64DXSM(12))
        means = bootstrap.bootstrap_component(metric, 200, rng, 25)
        # Column two is exactly twice column one for every cluster and therefore
        # in every paired bootstrap replicate, including variable row counts.
        np.testing.assert_allclose(means[:, 1], 2 * means[:, 0])


class ProductionInputIntegrationTests(unittest.TestCase):
    def make_args(self, output: Path) -> argparse.Namespace:
        return argparse.Namespace(
            results_dir=RESULTS_DIR,
            output_dir=output,
            replicates=200,
            seed=20_260_812,
            batch_size=50,
            diagnostic_prefixes=[50, 100],
        )

    def test_production_inputs_and_reproducibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first"
            second = Path(tmp) / "second"
            result = bootstrap.run(self.make_args(first))
            bootstrap.run(self.make_args(second))

            self.assertEqual(len(result["summary"]), 22)
            self.assertEqual(len(result["codec"]), 15)
            self.assertEqual(len(result["pairwise"]), 51)
            self.assertEqual(len(result["ranks"]), 22)
            self.assertLess(
                max(result["provenance"]["point_reproduction_max_abs_error"].values()),
                1e-12,
            )
            dimensions = result["provenance"]["dimensions"]
            self.assertEqual(dimensions["pa_units"], 22_068)
            self.assertEqual(dimensions["pa_clusters"], 21_414)
            self.assertEqual(dimensions["pc_units"], 669)
            self.assertEqual(dimensions["pc_clusters"], 443)
            self.assertEqual(dimensions["pa_partial_membership_clusters"], 6)
            codec_row = result["codec"].loc[
                (result["codec"]["family"] == "dinov2")
                & (result["codec"]["codec"] == "HQ")
            ].iloc[0]
            diagnostic_row = result["diagnostics"].loc[
                result["diagnostics"]["estimand"] == "codec:dinov2:HQ-Raw"
            ].iloc[0]
            self.assertEqual(
                diagnostic_row["point"], codec_row["product_delta_point"]
            )
            self.assertEqual(hash_csvs(first), hash_csvs(second))


if __name__ == "__main__":
    unittest.main()
