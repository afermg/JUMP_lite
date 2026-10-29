# /// script
# requires-python = "==3.12.12"
# dependencies = ["numpy==1.26.4", "polars==1.35.2", "certifi==2025.10.5", "scipy==1.14.1"]
# ///
"""Offline unit tests; fixture data are synthetic, never biological evidence."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl
from scipy.stats import median_abs_deviation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("uv_jump_lite", ROOT / "examples/uv_jump_lite.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture():
    wells, sites = [], []
    for i in range(11):
        key = f"source__batch__plate__A{i:02}"
        row = dict(zip(m.KEYS, ["source", "batch", "plate", f"A{i:02}"]))
        wells.append({**row, "Metadata_Well_Key": key,
                      "Metadata_JCP2022": "control" if i < 5 else f"drug{i-5}",
                      "Metadata_pert_type": "negcon" if i < 5 else "trt", "Metadata_Group": "group_high"})
        for site in (0, 1):
            sites.append({**row, "Metadata_Site_Key": f"{key}__{site}",
                          **{f"URL_Orig{c}": f"s3://cellpainting-gallery/test/{key}-{site}-{c}.tif" for c in m.CHANNELS}})
    annotations = pl.DataFrame({"Metadata_JCP2022": ["drug0", "drug1", "drug1", "drug2"],
                               "target": ["target", "target", "target", "other"],
                               "support": [5, 6, 6, 1], "activity_class": ["active"] * 4})
    return pl.DataFrame(sites), pl.DataFrame(wells), annotations


class CohortTests(unittest.TestCase):
    def test_fixed_cohort_and_deduplicated_annotations(self):
        rows, edges = m.select_cohort(*fixture())
        self.assertEqual(len(rows), 11)
        self.assertEqual(sum(r["control"] for r in rows), 5)
        self.assertTrue(all(r["site_key"].endswith("__0") for r in rows))
        self.assertEqual([(e["query"], e["reference"]) for e in edges], [("drug0", "drug1"), ("drug1", "drug0")])
        self.assertTrue(all(url.startswith(m.BASE) for r in rows for url in r["channels"].values()))

    def test_order_independent(self):
        tables = fixture()
        self.assertEqual(m.select_cohort(*tables), m.select_cohort(*(d.reverse() for d in tables)))

    def test_missing_control_refused(self):
        s, w, a = fixture()
        with self.assertRaisesRegex(ValueError, "No plate"):
            m.select_cohort(s, w.slice(1), a)

    def test_missing_annotations_refused(self):
        s, w, a = fixture()
        with self.assertRaisesRegex(ValueError, "No plate"):
            m.select_cohort(s, w, a.head(0))

    def test_duplicate_well_refused(self):
        s, w, a = fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate release well"):
            m.select_cohort(s, pl.concat([w, w.head(1)]), a)

    def test_duplicate_site_refused(self):
        s, w, a = fixture()
        with self.assertRaisesRegex(ValueError, "Duplicate release site"):
            m.select_cohort(pl.concat([s, s.head(1)]), w, a)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.url = m.BASE + "test/file"
        self.payload = b"real transport mocked by an explicitly synthetic test fixture"
        self.sha = hashlib.sha256(self.payload).hexdigest()

    def tearDown(self):
        self.temp.cleanup()

    def response(self):
        response = io.BytesIO(self.payload)
        response.url = self.url
        response.headers = {"Content-Length": str(len(self.payload))}
        return response

    def test_pinned_download_and_offline_cache_reuse(self):
        with patch.object(m, "urlopen", return_value=self.response()) as network:
            p, receipt = m.fetch(self.url, self.root, self.sha, len(self.payload))
            self.assertEqual(p.read_bytes(), self.payload)
            self.assertEqual(receipt["sha256"], self.sha)
            m.fetch(self.url, self.root, self.sha, len(self.payload))
            self.assertEqual(network.call_count, 1)

    def test_corrupted_cache_refused(self):
        with patch.object(m, "urlopen", return_value=self.response()):
            p, _ = m.fetch(self.url, self.root, self.sha)
        p.write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "drift"):
            m.fetch(self.url, self.root, self.sha)

    def test_wrong_pin_not_published(self):
        with patch.object(m, "urlopen", return_value=self.response()):
            with self.assertRaisesRegex(ValueError, "SHA256"):
                m.fetch(self.url, self.root, "0" * 64)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_wrong_size_not_published(self):
        with patch.object(m, "urlopen", return_value=self.response()):
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                m.fetch(self.url, self.root, self.sha, 1000)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_header_bound(self):
        with patch.object(m, "urlopen", return_value=self.response()):
            with self.assertRaisesRegex(ValueError, "exceeds bound"):
                m.fetch(self.url, self.root, maximum=1)

    def test_stream_bound_without_header(self):
        response = self.response()
        response.headers = {}
        with patch.object(m, "urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "byte bound"):
                m.fetch(self.url, self.root, maximum=1)

    def test_unapproved_urls(self):
        for url in ["file:///etc/passwd", "http://localhost/", "https://example.com/a", "s3://private/a",
                    "https://user:pass@huggingface.co/a", self.url + "?token=abc"]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                m.checked_url(url)

    def test_incomplete_cache_refused(self):
        key = hashlib.sha256(self.url.encode()).hexdigest()
        (self.root / key).write_bytes(self.payload)
        with self.assertRaisesRegex(ValueError, "Incomplete cache"):
            m.fetch(self.url, self.root)

    def test_symlinked_cache_refused(self):
        link = self.root / "link"
        link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            m.fetch(self.url, link)


class ComputationTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.edges = m.select_cohort(*fixture())
        self.x = np.arange(33, dtype=float).reshape(11, 3) + np.array([1, 4, 7])

    def test_exact_cpu_production_primitive(self):
        # Compile ONLY the unchanged production CPU class, avoiding unrelated
        # module-level CuPy/cuML imports. This is not full-pipeline parity.
        source = ast.parse((ROOT / "src/norm_3/core.py").read_text())
        node = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "RobustMAD_CPU")
        namespace = {"np": np, "scipy_mad": median_abs_deviation}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "production-core-cpu", "exec"), namespace)
        expected = namespace["RobustMAD_CPU"]().fit(self.x[:5]).transform(self.x)
        got, _ = m.normalize(self.rows, self.x)
        np.testing.assert_array_equal(got, expected)

    def test_treatments_do_not_affect_fit(self):
        _, fit = m.normalize(self.rows, self.x)
        changed = self.x.copy()
        changed[5:] *= 1000
        _, fit2 = m.normalize(self.rows, changed)
        self.assertEqual(fit, fit2)

    def test_constant_controls_fail(self):
        self.x[:5] = 1
        with self.assertRaisesRegex(ValueError, "MAD"):
            m.normalize(self.rows, self.x)

    def test_nonfinite_fail(self):
        self.x[6, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "feature matrix"):
            m.normalize(self.rows, self.x)

    def test_tie_policy_and_self_exclusion(self):
        result = m.evaluate(self.rows, np.ones_like(self.x), self.edges)
        queries = result["per_query"]
        self.assertEqual(queries[0]["top_reference"], "drug1")
        self.assertEqual(queries[1]["top_reference"], "drug0")
        self.assertEqual(queries[0]["candidates"], 5)
        self.assertEqual(result["metrics"]["recall@1%"], 1.)
        self.assertEqual(result["queries_without_positives"], ["drug2", "drug3", "drug4", "drug5"])

    def test_invalid_edges(self):
        for bad in [[], [self.edges[0], self.edges[0]], [{"query": "drug0", "reference": "drug0"}],
                    [{"query": "drug0", "reference": "absent"}]]:
            with self.subTest(edges=bad), self.assertRaises(ValueError):
                m.evaluate(self.rows, self.x, bad)

    def test_zero_cosine_norm(self):
        with self.assertRaisesRegex(ValueError, "cosine norm"):
            m.evaluate(self.rows, np.zeros_like(self.x), self.edges)


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.rows, _ = m.select_cohort(*fixture())
        tables = []
        contract = {"contract_version": "synthetic-test-only", "channels": m.CHANNELS,
                    "codec": "lossless", "schema_sha256": "a" * 64, "parity_mode": "fixture",
                    "legacy_intensity_semantics": False, "cp_measure": {"version": "fixture"}}
        for i, row in enumerate(self.rows):
            path = self.root / f"{i}.parquet"
            pl.DataFrame({"Metadata_Site_Key": [row["site_key"]] * 3, "object_id": [1, 2, 3],
                          "metadata_object": ["cell"] * 3, "area": [10., 20., 30.]}).write_parquet(path)
            receipt = {**{k: v for k, v in contract.items() if k != "cp_measure"},
                       "dependency_contract": {"cp_measure": contract["cp_measure"]},
                       "status": "complete", "site_key": row["site_key"], "output_sha256": m.digest(path),
                       "output_bytes": path.stat().st_size, "object_counts": {"cell": 3}}
            receipt_path = self.root / f"{i}.receipt.json"
            receipt_path.write_text(json.dumps(receipt))
            tables.append({"site_key": row["site_key"], "path": path.name,
                           "sha256": m.digest(path), "object_id": "object_id",
                           "receipt": {"path": receipt_path.name, "sha256": m.digest(receipt_path)}})
        self.spec = {"schema_version": 1, "measurement_contract": contract, "features": ["area"],
                     "object_type": {"column": "metadata_object", "value": "cell"},
                     "units": {"area": "pixels^2"}, "tables": tables}
        self.path = self.root / "manifest.json"

    def refresh_first_receipt(self):
        path = self.root / "0.parquet"
        self.spec["tables"][0]["sha256"] = m.digest(path)
        receipt_path = self.root / "0.receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt.update(output_sha256=m.digest(path), output_bytes=path.stat().st_size,
                       object_counts={"cell": len(pl.read_parquet(path))})
        receipt_path.write_text(json.dumps(receipt))
        self.spec["tables"][0]["receipt"]["sha256"] = m.digest(receipt_path)

    def tearDown(self):
        self.temp.cleanup()

    def run_manifest(self):
        self.path.write_text(json.dumps(self.spec))
        return m.measurement_features(self.path, self.rows)

    def test_median_and_provenance(self):
        x, evidence = self.run_manifest()
        np.testing.assert_array_equal(x, np.full((11, 1), 20.))
        self.assertEqual(evidence["units"], {"area": "pixels^2"})
        self.assertEqual(evidence["tables"][0]["objects"], 3)

    def test_missing_site(self):
        self.spec["tables"].pop()
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.run_manifest()

    def test_checksum_drift(self):
        self.spec["tables"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum"):
            self.run_manifest()

    def test_no_assumed_units(self):
        self.spec["units"] = {}
        with self.assertRaisesRegex(ValueError, "units"):
            self.run_manifest()

    def test_no_object_id_feature(self):
        self.spec["features"] = ["object_id"]
        self.spec["units"] = {"object_id": "none"}
        with self.assertRaisesRegex(ValueError, "not a feature"):
            self.run_manifest()

    def test_wrong_site_in_table(self):
        p = self.root / "0.parquet"
        pl.DataFrame({"Metadata_Site_Key": ["wrong"], "object_id": [1], "metadata_object": ["cell"], "area": [2.]}).write_parquet(p)
        self.refresh_first_receipt()
        with self.assertRaisesRegex(ValueError, "site mismatch"):
            self.run_manifest()

    def test_duplicate_objects(self):
        p = self.root / "0.parquet"
        pl.DataFrame({"Metadata_Site_Key": [self.rows[0]["site_key"]] * 2,
                      "object_id": [1, 1], "metadata_object": ["cell", "cell"], "area": [2., 3.]}).write_parquet(p)
        self.refresh_first_receipt()
        with self.assertRaisesRegex(ValueError, "cell identity"):
            self.run_manifest()


    def test_mixed_contract_refused(self):
        self.spec["measurement_contract"]["codec"] = "mq"
        with self.assertRaisesRegex(ValueError, "measurement contracts"):
            self.run_manifest()

    def test_receipt_drift_refused(self):
        self.spec["tables"][0]["receipt"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "receipt checksum"):
            self.run_manifest()


class CliTests(unittest.TestCase):
    def test_full_reports_blocked_without_network_or_success_marker(self):
        with tempfile.TemporaryDirectory() as root, patch.object(m, "urlopen", side_effect=AssertionError("network")):
            output = Path(root) / "out"
            self.assertEqual(m.main(["--mode", "full", "--output", str(output)]), 3)
            self.assertFalse((output / "provenance.json").exists())
            self.assertFalse(json.loads((output / "full_readiness.json").read_text())["benchmark_executed"])

    def test_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(m.main(["--output", root]), 2)
            self.assertEqual(list(Path(root).iterdir()), [])

    def test_prepare_only_never_calls_model(self):
        with tempfile.TemporaryDirectory() as root, patch.object(m, "fetch", return_value=(Path("unused"), {})), \
                patch.object(m.pl, "read_parquet", side_effect=fixture()), \
                patch.object(m, "image_features", side_effect=AssertionError("model")):
            output = Path(root) / "out"
            self.assertEqual(m.main(["--prepare-only", "--output", str(output)]), 0)
            self.assertTrue((output / "prepared.json").exists())
            self.assertFalse((output / "provenance.json").exists())


if __name__ == "__main__":
    unittest.main()
