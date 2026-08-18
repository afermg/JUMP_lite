#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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

    def make_release(self, root: Path) -> Path:
        source = root / "input.txt"
        source.write_text("frozen")
        points, loo, influence, coverage = [], [], [], []
        for i, family in enumerate(a.FAMILIES):
            d2, mq = 0.01 + i / 100, 0.011 + i / 100
            for codec, product in (("D2-E8", d2), ("MQ", mq)):
                points.append({"family": family, "codec": codec, "population": "common_Metadata_id", "product": product})
                coverage.append({"family": family, "codec": codec, "common_rows": 1519})
            delta = mq - d2
            influence.append({"family": family, "product_delta_mq_minus_d2": delta})
            for j in range(a.EXPECTED_PLATES + 1):
                loo.append({"family": family, "omitted_label": "Full" if j == 0 else f"P{j}", "product_delta_mq_minus_d2": delta})
        pd.DataFrame(points).to_csv(root / "fixed_recipe_points.csv", index=False)
        pd.DataFrame(loo).to_csv(root / "leave_one_plate_out.csv", index=False)
        pd.DataFrame(influence).to_csv(root / "influence_summary.csv", index=False)
        pd.DataFrame(coverage).to_csv(root / "coverage_manifest.csv", index=False)
        a.write_json(root / "provenance.json", {
            "analysis": "plate_unit_influence", "protocol_version": 3,
            "inputs": [a.record(source) | {"role": "sweep"}],
        })
        artifacts = []
        for path in sorted(root.iterdir()):
            if path.name in {"artifact_checksums.json", "input.txt"}: continue
            artifacts.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": a.sha256_file(path)})
        a.write_json(root / "artifact_checksums.json", {"artifacts": artifacts})
        return source

    def move_release(self, root: Path) -> Path:
        release = root / "release"
        release.mkdir()
        for name in ("fixed_recipe_points.csv", "leave_one_plate_out.csv", "influence_summary.csv", "coverage_manifest.csv", "provenance.json", "artifact_checksums.json"):
            (root / name).replace(release / name)
        return release

    def test_verify_detects_artifact_and_input_drift(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = self.make_release(root)
            release = self.move_release(root)
            a.verify(release)
            source.write_text("drift")
            with self.assertRaisesRegex(a.AnalysisError, "input drift"):
                a.verify(release)

    def test_repo_source_identity_survives_relocation(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = root / "checkout-a"
            second = root / "checkout-b"
            relative = Path("src/norm_3/metrics.py")
            for checkout in (first, second):
                source = checkout / relative
                source.parent.mkdir(parents=True)
                source.write_text("same scoring source\n")
            with mock.patch.object(a, "REPO", first):
                row = a.record_repo_source(first / relative) | {"role": "scoring_source"}
            self.assertEqual(row["path"], relative.as_posix())
            self.assertEqual(row["path_scope"], "repo_relative")
            self.assertNotIn(str(first), row["path"])
            with mock.patch.object(a, "REPO", second):
                a.verify_input_records([row])

    def test_repo_relative_paths_fail_closed_on_unsafe_spellings_and_escape(self):
        base = {"path_scope": "repo_relative", "role": "scoring_source", "size_bytes": 1, "sha256": "0" * 64}
        for path in ("../outside.py", "src/../outside.py", "/absolute.py", "src//metrics.py", "./src/metrics.py", "src\\metrics.py"):
            with self.subTest(path=path), self.assertRaisesRegex(a.AnalysisError, "unsafe repo-relative"):
                a.resolve_input_path(base | {"path": path})
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            repo = root / "repo"; repo.mkdir()
            outside = root / "outside.py"; outside.write_text("x")
            (repo / "link.py").symlink_to(outside)
            with mock.patch.object(a, "REPO", repo), self.assertRaisesRegex(a.AnalysisError, "escapes repository"):
                a.resolve_input_path(base | {"path": "link.py"})

    def test_input_scope_contract_rejects_misclassified_paths(self):
        with tempfile.TemporaryDirectory() as td:
            external = Path(td) / "external.txt"
            external.write_text("x")
            absolute = a.record(external)
            with self.assertRaisesRegex(a.AnalysisError, "scoring source must be repo-relative"):
                a.resolve_input_path(absolute | {"role": "scoring_source"})
            with mock.patch.object(a, "REPO", Path(td)):
                with self.assertRaisesRegex(a.AnalysisError, "repository input must be repo-relative"):
                    a.resolve_input_path(absolute | {"role": "sweep"})

    def test_verify_rejects_extra_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            self.make_release(root)
            release = self.move_release(root)
            (release / "unexpected.txt").write_text("bad")
            with self.assertRaisesRegex(a.AnalysisError, "output inventory drift"):
                a.verify(release)


if __name__ == "__main__":
    unittest.main()
