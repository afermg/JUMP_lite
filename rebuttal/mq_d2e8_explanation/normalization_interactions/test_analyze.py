#!/usr/bin/env python3
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import analyze


def synthetic_frame(duplicate: bool = False) -> tuple[pd.DataFrame, dict[str, tuple[str, str]]]:
    pairs = {"FamilyA": ("a_d2", "a_mq"), "FamilyB": ("b_d2", "b_mq")}
    rows = []
    for family_index, (_family, (d2_model, mq_model)) in enumerate(pairs.items()):
        for norm in ("robustmad", "standardize"):
            config = f"{norm}_all__prune0.9__tvn_efaar_e0.05"
            for codec_index, model in enumerate((d2_model, mq_model)):
                pa = 0.2 + family_index * 0.01 + codec_index * 0.001
                pc = 0.1 + (0.001 if norm == "standardize" else 0.0)
                rows.append(
                    {
                        "model": model,
                        "config": config,
                        "PA_mean_nap": pa,
                        "PC_mean_nap": pc,
                        "norm_method": norm,
                        "outlier_cutoff": np.nan,
                        "use_int": False,
                        "prune_thresh": 0.9 if norm == "robustmad" else np.nan,
                        "use_pca": False,
                        "pca_components": np.nan,
                        "batch_method": "tvn_efaar",
                        "tvn_epsilon": 0.05,
                        "tvn_efaar_n_components": 128,
                    }
                )
    frame = pd.DataFrame(rows)
    if duplicate:
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    return frame, pairs


class AnalysisTests(unittest.TestCase):
    def test_build_pairs_aligns_structural_recipes(self) -> None:
        frame, pairs = synthetic_frame()
        paired = analyze.build_paired_table(frame, model_pairs=pairs, expected_configs=2)
        self.assertEqual(len(paired), 4)
        self.assertEqual(paired["recipe_signature"].nunique(), 2)
        self.assertTrue((paired["delta_mq_minus_d2e8"] > 0).all())
        self.assertEqual(set(paired["prune_relative"]), {"lower", "higher"})

    def test_duplicate_model_config_fails_closed(self) -> None:
        frame, pairs = synthetic_frame(duplicate=True)
        with self.assertRaisesRegex(ValueError, "duplicate model/config"):
            analyze.build_paired_table(frame, model_pairs=pairs, expected_configs=2)

    def test_two_way_decomposition_closes(self) -> None:
        frame, pairs = synthetic_frame()
        paired = analyze.build_paired_table(frame, model_pairs=pairs, expected_configs=2)
        result = analyze.two_way_decomposition(paired)
        self.assertAlmostEqual(result["fraction_of_total_variation"].sum(), 1.0)
        self.assertTrue((result["sum_squares"] >= 0).all())

    def test_verify_release_rejects_extra_inventory(self) -> None:
        release = Path(analyze.__file__).resolve().parent / "outputs" / "release_v1"
        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "release"
            shutil.copytree(release, copy)
            analyze.verify_release(copy)
            (copy / "extra.txt").write_text("not inventoried\n")
            with self.assertRaisesRegex(ValueError, "release inventory drift"):
                analyze.verify_release(copy)

    def test_verify_release_rejects_artifact_drift(self) -> None:
        release = Path(analyze.__file__).resolve().parent / "outputs" / "release_v1"
        with tempfile.TemporaryDirectory() as temp:
            copy = Path(temp) / "release"
            shutil.copytree(release, copy)
            with (copy / "REPORT.md").open("a") as stream:
                stream.write("drift\n")
            with self.assertRaisesRegex(ValueError, "artifact size drift|artifact hash drift"):
                analyze.verify_release(copy)

    def test_verify_release_binds_current_source(self) -> None:
        release = Path(analyze.__file__).resolve().parent / "outputs" / "release_v1"
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "analyze.py"
            source.write_text(Path(analyze.__file__).read_text() + "# drift\n")
            with self.assertRaisesRegex(ValueError, "source identity drift"):
                analyze.verify_release(release, source_path=source)

    def test_publish_staged_atomically_replaces_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "release"
            staged = root / "staged"
            output.mkdir()
            staged.mkdir()
            (output / "old.txt").write_text("old\n")
            (staged / "new.txt").write_text("new\n")
            analyze._publish_staged(staged, output)
            self.assertFalse(staged.exists())
            self.assertFalse((output / "old.txt").exists())
            self.assertEqual((output / "new.txt").read_text(), "new\n")
            self.assertEqual(list(root.glob(".release.backup-*")), [])

    def test_generation_failure_preserves_existing_release(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = root / "release"
            output.mkdir()
            (output / "sentinel.txt").write_text("original\n")

            def fail_after_partial(_input: Path, staged: Path, _source: Path) -> None:
                (staged / "partial.txt").write_text("partial\n")
                raise RuntimeError("injected failure")

            with mock.patch.object(analyze, "_generate_release", side_effect=fail_after_partial):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    analyze.run(analyze.INPUT, output)
            self.assertEqual((output / "sentinel.txt").read_text(), "original\n")
            self.assertEqual(list(root.glob(".release.staging-*")), [])


if __name__ == "__main__":
    unittest.main()
