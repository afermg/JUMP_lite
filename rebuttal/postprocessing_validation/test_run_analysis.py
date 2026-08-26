from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from plot_heldout_codec_performance import (  # noqa: E402
    DISPLAY_NAMES,
    FAMILIES,
    load_codec_comparisons,
    load_scores,
    load_uncertainty,
    make_figure,
)
from run_analysis import (  # noqa: E402
    AnalysisError,
    CoverageError,
    FamilySpec,
    assert_fixed_config_coverage,
    atomic_write_json,
    load_checkpoint,
    make_treatment_split,
    minmax_score_candidates,
    render_report,
    write_checkpoint,
)


class AnalysisUnitTests(unittest.TestCase):
    def test_split_is_deterministic_stratified_and_leakage_free(self) -> None:
        memberships = {
            **{f"a{i}": {"group_a"} for i in range(10)},
            **{f"b{i}": {"group_b"} for i in range(10)},
            **{f"both{i}": {"group_a", "group_b"} for i in range(10)},
        }
        first = make_treatment_split(memberships, validation_fraction=0.2, seed=7)
        second = make_treatment_split(dict(reversed(list(memberships.items()))), 0.2, 7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(memberships))
        self.assertEqual(len({row["treatment_id"] for row in first}), len(first))
        by_stratum: dict[str, list[dict]] = {}
        for row in first:
            by_stratum.setdefault(row["stratum"], []).append(row)
        self.assertEqual(set(by_stratum), {"group_a", "group_b", "group_a|group_b"})
        self.assertTrue(
            all(sum(row["split"] == "validation" for row in rows) == 2 for rows in by_stratum.values())
        )

    def test_minmax_selection_and_lexical_tie_break(self) -> None:
        rows = [
            {"config": "zeta", "validation_pa_mean_nap": 1.0, "validation_pc_mean_nap": 1.0},
            {"config": "beta", "validation_pa_mean_nap": 2.0, "validation_pc_mean_nap": 2.0},
            {"config": "alpha", "validation_pa_mean_nap": 2.0, "validation_pc_mean_nap": 2.0},
            {"config": "tradeoff", "validation_pa_mean_nap": 3.0, "validation_pc_mean_nap": 1.0},
        ]
        scored, ranges = minmax_score_candidates(rows)
        self.assertEqual([row["config"] for row in scored[:2]], ["alpha", "beta"])
        self.assertAlmostEqual(scored[0]["selection_score"], 0.5)
        self.assertEqual(
            ranges,
            {"pa_min": 1.0, "pa_max": 3.0, "pc_min": 1.0, "pc_max": 2.0},
        )
        constant, _ = minmax_score_candidates(
            [
                {"config": "b", "validation_pa_mean_nap": 1.0, "validation_pc_mean_nap": 1.0},
                {"config": "a", "validation_pa_mean_nap": 1.0, "validation_pc_mean_nap": 1.0},
            ]
        )
        self.assertEqual([row["config"] for row in constant], ["a", "b"])
        self.assertEqual(constant[0]["selection_score"], 1.0)

    @staticmethod
    def _write_valid_config(path: Path) -> None:
        path.write_text(
            """steps:
  - name: evaluate_metrics
    enabled: true
    params:
      compound_col: Metadata_JCP2022
      target_col: Metadata_RefChemDB_target
      negcon_col: Metadata_negcon
      pc_groups: [group_high, group_low]
"""
        )

    def test_fixed_config_coverage_fails_without_exact_codec(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec = FamilySpec("demo", "Demo", {"Raw": "demo_raw", "MQ": "demo_mq"})
            raw = root / "demo_raw" / "fixed"
            raw.mkdir(parents=True)
            (raw / "output.parquet").write_bytes(b"not-empty")
            self._write_valid_config(raw / "pipeline_config.yaml")
            with self.assertRaisesRegex(CoverageError, "MQ"):
                assert_fixed_config_coverage(root, spec, "fixed")
            mq = root / "demo_mq" / "fixed"
            mq.mkdir(parents=True)
            (mq / "output.parquet").write_bytes(b"not-empty")
            self._write_valid_config(mq / "pipeline_config.yaml")
            refs = assert_fixed_config_coverage(root, spec, "fixed")
            self.assertEqual(set(refs), {"Raw", "MQ"})

    def test_atomic_checkpoint_resume_and_protocol_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested" / "checkpoint.json"
            atomic_write_json(target, {"version": 1})
            self.assertEqual(json.loads(target.read_text()), {"version": 1})
            atomic_write_json(target, {"version": 2})
            self.assertEqual(json.loads(target.read_text()), {"version": 2})
            self.assertFalse(list(target.parent.glob("*.tmp")))

            write_checkpoint(target, "protocol-a", {"result": {"status": "ok", "value": 3}})
            resumed = load_checkpoint(target, "protocol-a")
            self.assertIsNotNone(resumed)
            self.assertEqual(resumed["result"]["value"], 3)
            with self.assertRaisesRegex(AnalysisError, "protocol mismatch"):
                load_checkpoint(target, "protocol-b")

    def test_heldout_figure_includes_vit_random_baseline(self) -> None:
        source_results = HERE / "results"
        scores = load_scores(source_results / "heldout_test_scores.csv")
        uncertainty = load_uncertainty(
            source_results / "uncertainty" / "heldout_uncertainty.csv", scores
        )
        comparisons = load_codec_comparisons(
            source_results / "uncertainty" / "codec_vs_raw_paired.csv", scores
        )

        self.assertIn("dinov2_random", FAMILIES)
        self.assertEqual(len(scores), len(FAMILIES) * 4)
        self.assertEqual(
            len(comparisons.loc[comparisons["family"] == "dinov2_random"]), 3
        )
        figure = make_figure(scores, uncertainty, comparisons)
        try:
            _, labels = figure.axes[0].get_legend_handles_labels()
            self.assertEqual(labels, [DISPLAY_NAMES[family] for family in FAMILIES])
            self.assertIn("ViT-rand", labels)
        finally:
            plt.close(figure)

    def test_report_requires_a_bound_finite_uncertainty_bundle(self) -> None:
        source_results = HERE / "results"
        protocol = json.loads((source_results / "protocol.json").read_text())
        split_rows = pd.read_csv(source_results / "treatment_split.csv").to_dict("records")
        selected_rows = pd.read_csv(source_results / "selected_configs.csv").to_dict("records")
        final_rows = pd.read_csv(source_results / "heldout_test_scores.csv").to_dict("records")
        coverage_rows = pd.read_csv(source_results / "selected_codec_coverage.csv").to_dict("records")
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "results"
            shutil.copytree(source_results / "uncertainty", output_dir / "uncertainty")
            for filename in (
                "artifact_checksums.json",
                "heldout_codec_performance.pdf",
                "heldout_codec_performance.png",
            ):
                shutil.copy2(source_results / filename, output_dir / filename)
            render_report(
                output_dir,
                protocol,
                split_rows,
                selected_rows,
                final_rows,
                coverage_rows,
                [],
            )
            report = (output_dir / "REPORT.md").read_text()
            self.assertIn("50,000-replicate paired", report)
            self.assertIn("all 12 learned-model pairwise", report)
            self.assertIn("## Figure", report)
            self.assertIn("`uncertainty/*`", report)

            summary_path = output_dir / "uncertainty/heldout_uncertainty.csv"
            original_summary = summary_path.read_bytes()
            summary = pd.read_csv(summary_path)
            summary.loc[0, "product_point"] = float("nan")
            summary.to_csv(summary_path, index=False)
            with self.assertRaisesRegex(AnalysisError, "non-finite"):
                render_report(
                    output_dir, protocol, split_rows, selected_rows,
                    final_rows, coverage_rows, [],
                )

            summary_path.write_bytes(original_summary)
            summary = pd.read_csv(summary_path)
            summary.loc[0, "product_point"] += 0.01
            summary.to_csv(summary_path, index=False)
            with self.assertRaisesRegex(AnalysisError, "stale"):
                render_report(
                    output_dir, protocol, split_rows, selected_rows,
                    final_rows, coverage_rows, [],
                )

            summary_path.write_bytes(original_summary)
            diagnostics = output_dir / "uncertainty/bootstrap_diagnostics.csv"
            diagnostics.write_text("ready\n")
            with self.assertRaisesRegex(AnalysisError, "missing columns"):
                render_report(
                    output_dir, protocol, split_rows, selected_rows,
                    final_rows, coverage_rows, [],
                )


if __name__ == "__main__":
    unittest.main()
