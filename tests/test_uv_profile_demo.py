# /// script
# requires-python = "==3.12.12"
# dependencies = ["numpy==2.2.6"]
# ///
"""Focused tests, runnable without the repository environment using uv run."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

# Optional path lets this test file and demo run from an unrelated directory.
if "--demo-script" in sys.argv:
    position = sys.argv.index("--demo-script")
    SCRIPT = Path(sys.argv[position + 1]).resolve()
    del sys.argv[position:position + 2]
else:
    SCRIPT = Path(__file__).resolve().parents[1] / "examples/uv_profile_demo.py"
spec = importlib.util.spec_from_file_location("uv_profile_demo", SCRIPT)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
np = demo.np


class ProfileDemoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.profiles, self.annotations = demo.synthetic_inputs()
        self.columns, self.rows = demo.parse_csv(self.profiles)
        self.ann_columns, self.edges = demo.parse_csv(self.annotations)

    def run_demo(self, rows=None, edges=None, features=None, name="out"):
        return demo.run(
            demo.csv_bytes(self.columns, self.rows if rows is None else rows),
            demo.csv_bytes(self.ann_columns, self.edges if edges is None else edges),
            demo.FEATURES if features is None else features, self.root / name,
            "SYNTHETIC_NOT_SCIENTIFIC_EVIDENCE",
        )

    def test_deterministic_outputs_and_provenance_hashes(self):
        self.run_demo(name="a")
        self.run_demo(name="b")
        for file in sorted((self.root / "a").iterdir()):
            self.assertEqual(file.read_bytes(), (self.root / "b" / file.name).read_bytes())
        report = json.loads((self.root / "a/provenance.json").read_text())
        self.assertFalse(report["scientific_evidence"])
        for name, digest in report["output_sha256"].items():
            self.assertEqual(demo.sha256((self.root / "a" / name).read_bytes()), digest)
        metrics = json.loads((self.root / "a/metrics.json").read_text())
        self.assertEqual(metrics["metrics"], {"recall@1%": 1., "recall@5%": 1., "recall@10%": 1.})
        self.assertEqual(metrics["coverage"]["matched_directed_edges"], 6)
        self.assertEqual(metrics["coverage"]["evaluation_wells"], 12)

    def test_row_permutation_keeps_numeric_results(self):
        self.run_demo(name="a")
        self.run_demo(rows=list(reversed(self.rows)), edges=list(reversed(self.edges)), name="b")
        for name in ("normalized_profiles.csv", "metrics.json", "configuration.json", "per_query.csv"):
            self.assertEqual((self.root / "a" / name).read_bytes(), (self.root / "b" / name).read_bytes())

    def test_raw_mad_formula_and_training_only_fit(self):
        x = demo.validate_profiles(self.columns, self.rows, demo.FEATURES)
        normalized, fit = demo.normalize(self.rows, x)
        first = fit[0]
        np.testing.assert_array_equal(first["median"], [10, 20, 30, 40])
        np.testing.assert_array_equal(first["raw_mad"], [1, 2, 3, 4])
        np.testing.assert_allclose(normalized[:5], np.repeat(np.arange(-2., 3.)[:, None], 4, axis=1))
        changed = x.copy()
        for i, row in enumerate(self.rows):
            if row["Metadata_Split"] == "eval":
                changed[i] += 10000
        self.assertEqual(fit, demo.normalize(self.rows, changed)[1])
        # Even an evaluation control is not used to fit the transform.
        extra = dict(self.rows[0], Metadata_Sample="heldout-control", Metadata_Split="eval")
        self.assertEqual(fit, demo.normalize(self.rows + [extra], np.vstack([x, x[0] + 10000]))[1])

    def test_metadata_never_features(self):
        with self.assertRaisesRegex(ValueError, "metadata is forbidden"):
            self.run_demo(features=["Metadata_Plate"])
        for row in self.rows:
            row["Metadata_Numeric"] = "999999"
        self.columns.append("Metadata_Numeric")
        self.run_demo()
        config = json.loads((self.root / "out/configuration.json").read_text())
        self.assertEqual(config["features"], demo.FEATURES)

    def test_invalid_features_fail_before_output(self):
        for invalid in ("nan", "inf", "-inf", "not-a-number", "1e309"):
            with self.subTest(invalid=invalid):
                self.rows[0]["Feature_1"] = invalid
                with self.assertRaises(ValueError):
                    self.run_demo()
                self.assertFalse((self.root / "out").exists())

    def test_csv_and_schema_failures(self):
        for data in (b"a,a\n1,2\n", b"a,b\n1\n", b"a,b\n1,2,3\n", b"a,b\n", b"a,b\n1, \n"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                demo.parse_csv(data)
        with self.assertRaisesRegex(ValueError, "Duplicate sample"):
            self.run_demo(rows=self.rows + [self.rows[0]])
        with self.assertRaisesRegex(ValueError, "Missing metadata/feature"):
            self.run_demo(features=["Feature_missing"])
        with self.assertRaisesRegex(ValueError, "unique"):
            self.run_demo(features=["Feature_1", "Feature_1"])

    def test_control_coverage_and_roles(self):
        with self.assertRaisesRegex(ValueError, "at least 3 training controls"):
            self.run_demo(rows=self.rows[3:])
        self.rows[0]["Metadata_negcon"] = "false"
        with self.assertRaisesRegex(ValueError, "Only declared negative controls"):
            self.run_demo()
        self.rows[0]["Metadata_negcon"] = "True"
        with self.assertRaisesRegex(ValueError, "true or false"):
            self.run_demo()

    def test_conflicting_identity_and_split(self):
        self.rows[5]["Metadata_JCP2022"] = "synthetic-control"
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            self.run_demo()
        self.rows[5]["Metadata_JCP2022"] = "synthetic-drug-0"
        self.rows[5]["Metadata_Split"] = "test"
        with self.assertRaisesRegex(ValueError, "Split"):
            self.run_demo()

    def test_degenerate_features_fail(self):
        for row in self.rows:
            row["Feature_1"] = "1"
        with self.assertRaisesRegex(ValueError, "MAD"):
            self.run_demo()
        with self.assertRaisesRegex(ValueError, "Zero/nonfinite cosine"):
            demo.evaluate(self.rows, np.zeros((len(self.rows), 4)), self.ann_columns, self.edges)

    def test_annotation_alignment_duplicates_and_self_edges(self):
        for field, value, match in (("query_id", "absent", "Unmatched"),
                                    ("reference_id", "synthetic-control", "Unmatched"),
                                    ("query_group", "other", "mismatch"),
                                    ("task", "CG", "CC only"),
                                    ("reference_id", "synthetic-drug-0", "Self-edges")):
            changed = [dict(edge) for edge in self.edges]
            changed[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, match):
                self.run_demo(edges=changed)
        with self.assertRaisesRegex(ValueError, "Duplicate annotation"):
            self.run_demo(edges=self.edges + [self.edges[0]])
        with self.assertRaisesRegex(ValueError, "Empty CSV"):
            self.run_demo(edges=[])

    def test_coverage_excludes_unannotated_queries_not_references(self):
        self.run_demo(edges=self.edges[:1])
        coverage = json.loads((self.root / "out/metrics.json").read_text())["coverage"]
        self.assertEqual(coverage["queries_with_positives"], 1)
        self.assertEqual(len(coverage["queries_without_positives"]), 5)
        _, records = demo.parse_csv((self.root / "out/per_query.csv").read_bytes())
        self.assertEqual(records[0]["n_candidates"], "5")

    def test_self_removal_and_deterministic_ties(self):
        # All identical cosine vectors: self must not win, lexical tie breaking must.
        normalized = np.ones((len(self.rows), 4))
        _, per_query, _ = demo.evaluate(self.rows, normalized, self.ann_columns, self.edges)
        self.assertEqual(per_query[0]["top_reference"], "synthetic-drug-1")
        self.assertEqual(per_query[1]["top_reference"], "synthetic-drug-0")
        self.assertTrue(all(row["query_id"] != row["top_reference"] for row in per_query))

    def test_recall_counts_positive_fraction_not_hit_rate(self):
        edge = dict(self.edges[0], reference_id="synthetic-drug-2")
        _, per_query, _ = demo.evaluate(self.rows, np.ones((len(self.rows), 4)), self.ann_columns,
                                        [self.edges[0], edge])
        self.assertEqual(per_query[0]["recall_1pct"], 0.5)

    def test_bounds_and_failed_backend_no_completion(self):
        with patch.object(demo, "MAX_PAIRS", 1), self.assertRaisesRegex(ValueError, "pair limit"):
            self.run_demo()
        with patch.object(demo, "evaluate", side_effect=RuntimeError("backend failed")):
            with self.assertRaisesRegex(RuntimeError, "backend failed"):
                self.run_demo()
        self.assertFalse((self.root / "out").exists())
        path = self.root / "oversized.csv"
        path.write_bytes(b"12345")
        with patch.object(demo, "MAX_BYTES", 4), self.assertRaisesRegex(ValueError, "limit"):
            demo.read_bounded(path)

    def test_existing_output_never_overwritten(self):
        self.run_demo()
        before = (self.root / "out/provenance.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "Refusing existing output"):
            self.run_demo()
        self.assertEqual(before, (self.root / "out/provenance.json").read_bytes())

    def test_cli_custom_inputs_and_failure_exit(self):
        (self.root / "profiles.csv").write_bytes(self.profiles)
        (self.root / "annotations.csv").write_bytes(self.annotations)
        command = [sys.executable, str(SCRIPT), "--profiles", "profiles.csv", "--annotations",
                   "annotations.csv", "--features", ",".join(demo.FEATURES), "--output", "custom"]
        completed = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("USER_SUPPLIED", completed.stdout)
        failed = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(failed.returncode, 2)
        self.assertIn("Refusing existing output", failed.stderr)
        missing = subprocess.run([sys.executable, str(SCRIPT), "--profiles", "missing.csv"],
                                 cwd=self.root, capture_output=True, text=True)
        self.assertEqual(missing.returncode, 2)
        self.assertFalse((self.root / "uv-profile-demo-output").exists())


if __name__ == "__main__":
    unittest.main()
