#!/usr/bin/env python3
"""Focused unit and production-input tests for compression-order robustness."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
import shutil
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import polars as pl

MODULE = Path(__file__).with_name("analyze.py")
spec = importlib.util.spec_from_file_location("compression_order_analyze", MODULE)
assert spec and spec.loader
analysis = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analysis
spec.loader.exec_module(analysis)


class QuotaTests(unittest.TestCase):
    def test_largest_remainder_is_deterministic_and_lexical_on_tie(self) -> None:
        self.assertEqual(
            analysis.largest_remainder_quotas({"b": 10, "a": 10, "c": 20}, 7),
            {"b": 2, "a": 2, "c": 3},
        )
        first = analysis.largest_remainder_quotas({"high": 942, "low": 2069}, 306)
        second = analysis.largest_remainder_quotas({"low": 2069, "high": 942}, 306)
        self.assertEqual(first, second)
        self.assertEqual(sum(first.values()), 306)

    def test_quota_fails_when_sample_exceeds_population(self) -> None:
        with self.assertRaises(analysis.AnalysisError):
            analysis.largest_remainder_quotas({"a": 2}, 3)


class TargetRecipeTests(unittest.TestCase):
    def test_zstd_only_selection_and_lexical_tie(self) -> None:
        rows = []
        for _, prefix in analysis.TARGET_FOLDERS.items():
            rows.extend([
                {"model": f"{prefix}_zstd_raw", "config": "zeta", "PA": 50, "PC": 20,
                 "PA_mean_nap": .2, "PC_mean_nap": .1},
                {"model": f"{prefix}_zstd_raw", "config": "alpha", "PA": 50, "PC": 20,
                 "PA_mean_nap": .2, "PC_mean_nap": .1},
                {"model": f"{prefix}_jpegxl_lossy_d10_raw", "config": "other", "PA": 99,
                 "PC": 99, "PA_mean_nap": .9, "PC_mean_nap": .9},
            ])
        self.assertEqual(set(analysis.select_target_recipes(pd.DataFrame(rows)).values()), {"alpha"})

    def test_selection_fails_on_missing_family(self) -> None:
        rows = pd.DataFrame([{"model": "dinov2_zstd_raw", "config": "a", "PA": 1,
                              "PC": 1, "PA_mean_nap": .1, "PC_mean_nap": .1}])
        with self.assertRaises(analysis.AnalysisError):
            analysis.select_target_recipes(rows)

    def test_alignment_rejects_key_mismatch_and_duplicates(self) -> None:
        left = pd.DataFrame({"key": ["a", "b"], "x": [1, 2]})
        with self.assertRaises(analysis.AnalysisError):
            analysis.align_target_tables([left, pd.DataFrame({"key": ["a", "c"], "y": [3, 4]})], "key")
        with self.assertRaises(analysis.AnalysisError):
            analysis.align_target_tables([left, pd.DataFrame({"key": ["a", "a"], "y": [3, 4]})], "key")


class PairingTests(unittest.TestCase):
    def test_paired_weights_preserve_constant_difference(self) -> None:
        values = np.asarray([[1., 3.], [2., 4.], [5., 7.]])
        rng = np.random.Generator(np.random.PCG64DXSM(7))
        weights = rng.multinomial(3, np.full(3, 1 / 3), size=100)
        means = weights @ values / 3
        np.testing.assert_allclose(means[:, 1] - means[:, 0], 2.0)

    def test_holm_adjustment(self) -> None:
        np.testing.assert_allclose(analysis.holm_adjust([.01, .04, .03]), [.03, .06, .06])

    def test_target_bootstrap_is_deterministic_and_finite(self) -> None:
        rng = np.random.default_rng(5)
        columns = [f"{family}__{codec}" for family in analysis.FAMILIES for codec in ("Zstd", "D10", "D15")]
        pa = pd.DataFrame(rng.normal(.4, .03, (20, 12)), columns=columns)
        pc = pd.DataFrame(rng.normal(.05, .01, (15, 12)), columns=columns)
        pa.insert(0, "Metadata_broad_sample", [f"p{i}" for i in range(20)])
        pc.insert(0, "Metadata_target", [f"t{i}" for i in range(15)])
        records = [{"family": family, "config": "fixed"} for family in analysis.FAMILIES]
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = analysis.target_bootstrap(Path(a), records, pa, pc, replicates=500, seed=9)
            second = analysis.target_bootstrap(Path(b), records, pa, pc, replicates=500, seed=9)
            pd.testing.assert_frame_equal(first["contrasts"], second["contrasts"])
            self.assertTrue(np.isfinite(first["contrasts"].select_dtypes("number")).all().all())


class CheckpointTests(unittest.TestCase):
    def identity(self) -> dict[str, object]:
        return {"protocol_id": "abc", "family": "dinov2", "codec": "Raw", "config": "fixed",
                "output": "/frozen/output.parquet", "output_sha256": "input-hash",
                "output_size_bytes": 123, "config_sha256": "config-hash",
                "runner_sha256": "runner-hash", "n_samples": 2}

    def test_identity_resume_and_reject_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "dinov2__Raw.csv"
            identity = self.identity()
            self.assertEqual(analysis.load_checkpoint(checkpoint, identity), [])
            self.assertTrue(checkpoint.with_suffix(".protocol.json").is_file())
            pd.DataFrame([{"sample_id": 0, "family": "dinov2", "codec": "Raw", "config": "fixed",
                           "pa": .1, "pc": .2, "product": .02}]).to_csv(checkpoint, index=False)
            self.assertEqual(len(analysis.load_checkpoint(checkpoint, identity)), 1)
            for drifted in (
                identity | {"n_samples": 3},
                identity | {"runner_sha256": "new-runner"},
                identity | {"output_sha256": "new-input"},
                identity | {"config_sha256": "new-config"},
            ):
                with self.assertRaises(analysis.AnalysisError):
                    analysis.load_checkpoint(checkpoint, drifted)

    def test_rejects_unidentified_and_nonfinite_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "dinov2__Raw.csv"
            row = {"sample_id": 0, "family": "dinov2", "codec": "Raw", "config": "fixed",
                   "pa": np.nan, "pc": .2, "product": .02}
            pd.DataFrame([row]).to_csv(checkpoint, index=False)
            with self.assertRaises(analysis.AnalysisError):
                analysis.load_checkpoint(checkpoint, self.identity())
            analysis.atomic_json(checkpoint.with_suffix(".protocol.json"), self.identity())
            with self.assertRaises(analysis.AnalysisError):
                analysis.load_checkpoint(checkpoint, self.identity())


class PcEligibilityTests(unittest.TestCase):
    def test_excludes_exact_no_negative_query_only(self) -> None:
        rows = []
        targets = {"a": "T1|T2", "b": "T1", "c": "T1", "d": "T2", "e": "T2"}
        for compound, target in targets.items():
            for well in range(3):
                rows.append({analysis.COMPOUND: compound, analysis.TARGET: target,
                             analysis.NEGCON: False, analysis.GROUP: "group_high",
                             analysis.ID: f"{compound}-{well}"})
        frame = pl.DataFrame(rows)
        exclusions = analysis.pc_undefined_query_exclusions(frame)
        self.assertEqual(exclusions, [{analysis.COMPOUND: "a", analysis.GROUP: "group_high",
                                       "eligible_targets": "T1|T2", "reason": "no_negative",
                                       "positive_candidate_count": 4, "negative_candidate_count": 0}])
        self.assertEqual(analysis.pc_removal_records(exclusions), exclusions)
        filtered = analysis.apply_pc_exclusions(frame, exclusions)
        self.assertEqual(set(filtered[analysis.COMPOUND]), {"b", "c", "d", "e"})
        self.assertEqual(filtered.filter(pl.col(analysis.COMPOUND) != "a").height,
                         frame.filter(pl.col(analysis.COMPOUND) != "a").height)

    def test_no_positive_and_multilabel_reasons(self) -> None:
        rows = []
        # Overlapping multilabel query D has positives through Y/Z and negatives through X.
        for compound, target in (("a", "X"), ("b", "X"), ("c", "X"),
                                 ("d", "Y|Z"), ("e", "Y"), ("f", "Y"),
                                 ("g", "Z"), ("h", "Z")):
            rows.append({analysis.COMPOUND: compound, analysis.TARGET: target,
                         analysis.NEGCON: False, analysis.GROUP: "group_high"})
        excluded = analysis.pc_undefined_query_exclusions(pl.DataFrame(rows))
        self.assertFalse(any(row[analysis.COMPOUND] == "d" for row in excluded))
        # A group with wholly distinct eligible target sets has no positives.
        separate = []
        for index, target in enumerate(("A", "B", "C")):
            separate.append({analysis.COMPOUND: f"high-{index}", analysis.TARGET: target,
                             analysis.NEGCON: False, analysis.GROUP: "group_high"})
            for suffix in range(2):
                separate.append({analysis.COMPOUND: f"low-{index}-{suffix}", analysis.TARGET: target,
                                 analysis.NEGCON: False, analysis.GROUP: "group_low"})
        separate_frame = pl.DataFrame(separate)
        result = analysis.pc_undefined_query_exclusions(separate_frame)
        high = [row for row in result if row[analysis.GROUP] == "group_high"]
        self.assertEqual(len(high), 3)
        self.assertEqual({row["reason"] for row in high}, {"no_positive"})
        removals = analysis.pc_removal_records(result)
        self.assertEqual({row[analysis.GROUP] for row in removals}, {"group_high"})
        filtered = analysis.apply_pc_exclusions(separate_frame, result)
        self.assertEqual(set(filtered[analysis.GROUP]), {"group_low"})

    def test_individual_no_positive_is_record_only(self) -> None:
        rows = []
        # T3 is globally valid (three compounds) but g is its only query in low.
        for group, members in (
            ("group_low", (("a", "T1"), ("b", "T1"), ("c", "T1"),
                           ("d", "T2"), ("e", "T2"), ("f", "T2"), ("g", "T3"))),
            ("group_high", (("h", "T3"), ("i", "T3"), ("j", "T4"),
                            ("k", "T4"), ("l", "T4"))),
        ):
            for compound, target in members:
                rows.append({analysis.COMPOUND: compound, analysis.TARGET: target,
                             analysis.NEGCON: False, analysis.GROUP: group})
        frame = pl.DataFrame(rows)
        exclusions = analysis.pc_undefined_query_exclusions(frame)
        g_records = [row for row in exclusions if row[analysis.COMPOUND] == "g"]
        self.assertEqual(len(g_records), 1)
        self.assertEqual(g_records[0]["reason"], "no_positive")
        removals = analysis.pc_removal_records(exclusions)
        self.assertFalse(any(row[analysis.COMPOUND] == "g" for row in removals))
        filtered = analysis.apply_pc_exclusions(frame, exclusions)
        self.assertIn("g", set(filtered[analysis.COMPOUND]))

    def test_ordinary_disjoint_queries_unchanged(self) -> None:
        rows = []
        for compound, target in (("a", "T1"), ("b", "T1"), ("c", "T1"),
                                 ("d", "T2"), ("e", "T2"), ("f", "T2")):
            rows.append({analysis.COMPOUND: compound, analysis.TARGET: target,
                         analysis.NEGCON: False, analysis.GROUP: "group_low"})
        frame = pl.DataFrame(rows)
        self.assertEqual(analysis.pc_undefined_query_exclusions(frame), [])
        self.assertTrue(analysis.apply_pc_exclusions(frame, []).equals(frame))


class ChecksumTests(unittest.TestCase):
    def test_inventory_is_relocatable_and_detects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "original"
            (root / "results").mkdir(parents=True)
            (root / "checkpoints").mkdir()
            (root / "results/value.csv").write_text("a,b\n1,2\n")
            (root / "checkpoints/state.csv").write_text("sample_id\n0\n")
            analysis.write_checksums(root)
            payload = json.loads((root / "artifact_checksums.json").read_text())
            records = payload["final_release_artifacts"] + payload["retained_checkpoint_state"]
            self.assertTrue(records and all(not Path(row["path"]).is_absolute() for row in records))
            relocated = Path(tmp) / "relocated"
            shutil.copytree(root, relocated)
            analysis.verify_checksums(relocated)
            (relocated / "results/value.csv").write_text("corrupt\n")
            with self.assertRaises(analysis.AnalysisError):
                analysis.verify_checksums(relocated)


class FrozenInputGuardTests(unittest.TestCase):
    def matched_full_records(self) -> list[dict[str, str]]:
        return [
            {
                "family": family, "codec": codec,
                "pa_path": f"/{family}/{codec}/pa.csv", "pa_sha256": "pa-hash",
                "pc_path": f"/{family}/{codec}/pc.csv", "pc_sha256": "pc-hash",
            }
            for family in analysis.FAMILIES for codec in analysis.CODECS
        ]

    def test_target_hash_drift_is_rejected(self) -> None:
        base = [{"family": "dinov2", "codec": "D10", "metrics_sha256": "a",
                 "pa_sha256": "b", "pc_sha256": "c", "config_sha256": "d",
                 "output_sha256": "e"}]
        analysis.verify_frozen_target_records(base, [dict(base[0])])
        drift = [dict(base[0], pc_sha256="changed")]
        with self.assertRaises(analysis.AnalysisError):
            analysis.verify_frozen_target_records(drift, base)

    def test_matched_full_hash_drift_is_rejected(self) -> None:
        frozen = self.matched_full_records()
        analysis.verify_frozen_matched_full_records(frozen, list(reversed(frozen)))
        drift = [dict(row) for row in frozen]
        drift[0]["pa_sha256"] = "changed"
        with self.assertRaisesRegex(analysis.AnalysisError, "matched-full frozen input drift"):
            analysis.verify_frozen_matched_full_records(drift, frozen)

    def test_summarize_only_matched_full_drift_writes_nothing(self) -> None:
        frozen = self.matched_full_records()
        current = [dict(row) for row in frozen]
        current[0]["pc_sha256"] = "changed"
        baseline_frame = pd.DataFrame(current)
        baseline = {
            family: {codec: 0.1 for codec in analysis.CODECS}
            for family in analysis.FAMILIES
        }
        old = {
            "full_variants": [], "target2_inputs": [],
            "matched_full_inputs": frozen,
        }
        args = mock.Mock(
            output_dir=Path("/tmp/not-written"), verify_only=False, summarize_only=True,
            smoke=False, variant=None, workers=1,
        )
        writes: list[str] = []
        with (
            mock.patch.object(analysis, "verify_checksums"),
            mock.patch.object(analysis, "load_variants", return_value=([], None)),
            mock.patch.object(analysis, "protocol_identity", return_value="protocol"),
            mock.patch.object(analysis.Path, "read_text", return_value=json.dumps(old)),
            mock.patch.object(analysis, "validate_target_inputs", return_value=([], pd.DataFrame(), pd.DataFrame())),
            mock.patch.object(analysis, "verify_frozen_target_records"),
            mock.patch.object(analysis, "matched_full_baseline", return_value=(baseline, baseline_frame)),
            mock.patch.object(analysis, "atomic_csv", side_effect=lambda *a, **k: writes.append("csv")),
            mock.patch.object(analysis, "summarize_full", side_effect=lambda *a, **k: writes.append("summary")),
        ):
            with self.assertRaisesRegex(analysis.AnalysisError, "matched-full frozen input drift"):
                analysis.run(args)
        self.assertEqual(writes, [])


class ProductionInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.variants, _ = analysis.load_variants()

    def test_frozen_full_variants_are_exact_and_complete(self) -> None:
        import itertools
        self.assertEqual(len(self.variants), 16)
        self.assertEqual({(v.family, v.codec) for v in self.variants},
                         set(itertools.product(analysis.FAMILIES, analysis.CODECS)))
        self.assertTrue(all(v.output.is_file() and v.sha256 for v in self.variants))

    def test_production_manifest_construction_and_invariants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            info = analysis.build_manifests(root, self.variants, n_samples=2)
            treatments = pl.read_parquet(root / "manifests/sample_treatments.parquet")
            wells = pl.read_parquet(root / "manifests/sample_treatment_wells.parquet")
            self.assertEqual(treatments.group_by("sample_id").len()["len"].to_list(), [306, 306])
            units = wells.group_by(["sample_id", analysis.COMPOUND, analysis.GROUP]).agg(
                pl.len().alias("n"), pl.col(analysis.ID).n_unique().alias("u"))
            self.assertEqual(units.select(pl.col("n").min()).item(), 4)
            self.assertEqual(units.select(pl.col("u").min()).item(), 4)
            self.assertEqual(sum(info["quotas"].values()), 306)

    def test_matched_full_baseline_is_group_restricted_and_key_aligned(self) -> None:
        baseline, rows = analysis.matched_full_baseline()
        self.assertEqual(len(rows), 16)
        self.assertEqual(set(rows.groups), {"group_high|group_low"})
        self.assertEqual(set(rows.pa_units), {3654})
        self.assertEqual(set(rows.pc_targets), {669})
        self.assertEqual(set(baseline), set(analysis.FAMILIES))
        self.assertTrue(np.isfinite(rows[["pa", "pc", "product"]]).all().all())

    def test_target_inputs_are_complete_aligned_and_reproduce_points(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records, pa, pc = analysis.validate_target_inputs(Path(tmp))
            self.assertEqual(len(records), 12)
            self.assertEqual((len(pa), len(pc)), (306, 201))
            self.assertEqual((pa.shape[1], pc.shape[1]), (13, 13))
            for row in records:
                column = f"{row['family']}__{row['codec']}"
                self.assertAlmostEqual(pa[column].mean(), row["pa_point"], places=12)
                self.assertAlmostEqual(pc[column].mean(), row["pc_point"], places=12)


class SmokeArtifactTests(unittest.TestCase):
    """Exercise the actual spawned worker output produced by documented smoke mode."""

    SMOKE = Path("/tmp/compression_order_smoke_v3")

    def test_actual_worker_path_and_summary_regeneration(self) -> None:
        if not self.SMOKE.is_dir():
            self.skipTest("run documented --smoke command before this integration test")
        metrics = pl.read_parquet(self.SMOKE / "results/full_subsample_metrics.parquet")
        self.assertEqual(metrics.shape[0], 16)
        self.assertEqual(metrics.select(["family", "codec"]).unique().height, 16)
        self.assertTrue(metrics.select(pl.all_horizontal(pl.col(["pa", "pc", "product"]).is_finite()).all()).item())
        payload = json.loads((self.SMOKE / "artifact_checksums.json").read_text())
        if "path_semantics" in payload:
            analysis.verify_checksums(self.SMOKE)
        names = ("results/full_reversal_summary.csv", "results/full_pairwise_discordance.csv")
        analysis.summarize_full(self.SMOKE, n_samples=1)
        first = {name: (self.SMOKE / name).read_bytes() for name in names}
        analysis.summarize_full(self.SMOKE, n_samples=1)
        second = {name: (self.SMOKE / name).read_bytes() for name in names}
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
