#!/usr/bin/env python3
"""Focused contracts for the compact current-release cluster-control exporter."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "export_cluster_selection_control.py"
SPEC = importlib.util.spec_from_file_location("cluster_control_exporter", RUNNER)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load exporter: {RUNNER}")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ClusterControlExportTests(unittest.TestCase):
    def test_deterministic_byte_identical_export_and_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left"
            right = Path(tmp) / "right"
            exporter.export_package(left)
            exporter.export_package(right)
            self.assertEqual({path.name for path in left.iterdir()}, exporter.PACKAGE_FILES)
            self.assertEqual({path.name for path in right.iterdir()}, exporter.PACKAGE_FILES)
            for name in sorted(exporter.PACKAGE_FILES):
                self.assertEqual((left / name).read_bytes(), (right / name).read_bytes(), name)

    def test_existing_and_partial_output_roots_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(RuntimeError, "must be absent"):
                exporter.export_package(existing)
            partial = Path(tmp) / "partial"
            partial.mkdir()
            (partial / "partial.txt").write_text("partial\n")
            with self.assertRaisesRegex(RuntimeError, "must be absent"):
                exporter.export_package(partial)
            self.assertEqual((partial / "partial.txt").read_text(), "partial\n")

    def test_canonical_input_hash_drift_fails_closed_and_leaves_no_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            canonical = exporter.SOURCE_SPECS["cluster_selection_summary_table.csv"]
            specs = dict(exporter.SOURCE_SPECS)
            specs["cluster_selection_summary_table.csv"] = (
                canonical[0],
                canonical[1],
                "0" * 64,
            )
            output = tmp_path / "output"
            with self.assertRaisesRegex(RuntimeError, "Canonical source drift"):
                exporter.export_package(output, source_specs=specs)
            self.assertFalse(output.exists())

    def test_relative_provenance_and_sha256sums(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "export"
            exporter.export_package(output)
            provenance = json.loads((output / "RESULTS_PROVENANCE.json").read_text())
            self.assertEqual(provenance["version"], exporter.EXPORTER_VERSION)
            self.assertEqual(provenance["canonical_source_commit"], exporter.CANONICAL_SOURCE_COMMIT)
            records = [provenance["exporter"], *provenance["sources"].values(),
                       *provenance["package_files_excluding_provenance_and_checksums"]]
            for record in records:
                path = Path(record["path"])
                self.assertFalse(path.is_absolute())
                self.assertNotIn("..", path.parts)
                self.assertIn(record["path_scope"], {"repository_root", "package_root"})
                base = exporter.REPOSITORY_ROOT if record["path_scope"] == "repository_root" else output
                artifact = base / path
                self.assertTrue(artifact.is_file())
                self.assertEqual(artifact.stat().st_size, record["bytes"])
                self.assertEqual(sha256_file(artifact), record["sha256"])
            result = subprocess.run(
                ["sha256sum", "-c", "SHA256SUMS"], cwd=output,
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(len(result.stdout.strip().splitlines()), 9)

    def test_scientific_contract_and_current_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "export"
            exporter.export_package(output)
            summary = pd.read_csv(output / "cluster_selection_summary_table.csv").set_index("metric")["value"]
            expected = {
                "total_compounds": 115_721,
                "fit_eligible_compounds": 95_426,
                "selected_compounds": 3_775,
                "occupied_clusters": 120,
                "total_clusters": 128,
                "eligible_mass_in_occupied_clusters": 0.962368746463228,
                "total_variation_selected_vs_eligible": 0.474989347870955,
                "average_precision_acquisition_structure": 0.513536652643861,
                "average_precision_cluster_only": 0.273104769075815,
                "average_precision_structure_plus_cluster": 0.573313325285965,
                "combined_to_structure_ap_ratio": 1.11640195949861,
                "conditional_permutation_p": 0.000499750124937531,
                "mean_seed_adjusted_rand_index": 0.162414221608182,
            }
            for metric, value in expected.items():
                self.assertAlmostEqual(float(summary[metric]), value, places=13)
            selected = pd.read_csv(output / "release_selected_compounds.csv")
            self.assertEqual(list(selected.columns), ["Metadata_JCP2022"])
            self.assertEqual(len(selected), 3_775)
            self.assertEqual(selected.Metadata_JCP2022.nunique(), 3_775)
            self.assertEqual(selected.Metadata_JCP2022.tolist(), sorted(selected.Metadata_JCP2022))
            self.assertNotIn("JCP2022_033924", set(selected.Metadata_JCP2022))
            table = pd.read_csv(output / "cluster_selection_table.csv")
            self.assertEqual(len(table), 128)
            self.assertEqual(int(table.n_selected.sum()), 3_775)
            self.assertEqual(int((table.n_selected > 0).sum()), 120)
            readme = (output / "README.md").read_text()
            self.assertIn(exporter.CENTRAL_CLAIM, readme)
            normalized_readme = " ".join(readme.split())
            for qualification in (
                "proportional or random sampling", "stable biological classes",
                "coverage of every phenotype", "other feature models",
                "genetic-perturbation representativeness",
            ):
                self.assertIn(qualification, normalized_readme)

    def test_corrected_figure_identity_and_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "export"
            exporter.export_package(output)
            pdf = output / "cluster_selection_compound_map.pdf"
            png = output / "cluster_selection_compound_map.png"
            self.assertEqual(
                sha256_file(pdf),
                "59c639a30940bcce5fbbd94669049998ba9f3181164888c839748b35c8b14b3c",
            )
            self.assertEqual(
                sha256_file(png),
                "a436610f578c770913c5e7e14ea25d504ebfc264c3e94bfc278ea1bf4c0103ea",
            )
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF-1.4"))
            self.assertTrue(pdf.read_bytes().rstrip().endswith(b"%%EOF"))
            with Image.open(png) as image:
                self.assertEqual(image.size, (4560, 1575))
            provenance = json.loads((output / "RESULTS_PROVENANCE.json").read_text())
            source = provenance["sources"]["cluster_selection_compound_map.pdf"]
            self.assertEqual(source["sha256"], sha256_file(pdf))
            self.assertEqual(source["bytes"], pdf.stat().st_size)
            self.assertTrue(provenance["pipeline"]["export_only_no_fit_or_rescore"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
