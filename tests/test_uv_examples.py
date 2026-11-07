# /// script
# requires-python = "==3.12.12"
# dependencies = ["numpy==1.26.4", "polars==1.35.2", "scipy==1.16.3", "pyarrow==22.0.0",
#   "scikit-learn==1.5.2", "pyyaml==6.0.3", "certifi==2025.10.5"]
# ///
"""Minimal offline smoke tests; synthetic fixtures are not benchmark evidence."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def load(name):
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = load("uv_jump_benchmark")
public = load("uv_jump_lite")
demo = load("uv_profile_demo")
np, pl = benchmark.np, benchmark.pl


class ExampleTests(unittest.TestCase):
    def test_demo_repeatability_and_no_overwrite(self):
        profiles, annotations = demo.synthetic_inputs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("a", "b"):
                demo.run(profiles, annotations, demo.FEATURES, root / name,
                         "SYNTHETIC_NOT_SCIENTIFIC_EVIDENCE")
            for path in (root / "a").iterdir():
                self.assertEqual(path.read_bytes(), (root / "b" / path.name).read_bytes())
            self.assertFalse(json.loads((root / "a/provenance.json").read_text())["scientific_evidence"])
            with self.assertRaisesRegex(ValueError, "Refusing existing output"):
                demo.run(profiles, annotations, demo.FEATURES, root / "a", "synthetic")

    def test_download_checks_pin_and_cached_bytes(self):
        url, payload = public.BASE + "test/file", b"synthetic fixture"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response = io.BytesIO(payload)
            response.url, response.headers = url, {"Content-Length": str(len(payload))}
            with patch.object(public, "urlopen", return_value=response):
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    public.fetch(url, root, "0" * 64)
            self.assertEqual(list(root.iterdir()), [])
            response = io.BytesIO(payload)
            response.url, response.headers = url, {}
            pin = benchmark.hashlib.sha256(payload).hexdigest()
            with patch.object(public, "urlopen", return_value=response) as network:
                path, _ = public.fetch(url, root, pin)
                public.fetch(url, root, pin)
                self.assertEqual(network.call_count, 1)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "drift"):
                public.fetch(url, root, pin)

    def test_public_normalization_fits_only_controls(self):
        rows = [{"control": i < 5, "plate": "p1", "sample_id": str(i)} for i in range(11)]
        x = np.arange(33, dtype=float).reshape(11, 3) + [1, 4, 7]
        _, fit = public.normalize(rows, x)
        x[5:] *= 1000
        self.assertEqual(fit, public.normalize(rows, x)[1])
        x[:5] = 1
        with self.assertRaisesRegex(ValueError, "MAD"):
            public.normalize(rows, x)

    def test_full_readiness_never_claims_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = public.full_readiness(root)
            self.assertEqual(report["status"], "blocked")
            self.assertFalse(report["benchmark_executed"])
            self.assertFalse(report["published_score_parity"])
            self.assertFalse((root / "provenance.json").exists())

    def test_strict_isolation_state_replay_and_paper_difference(self):
        rows = []
        for plate in ("p1", "p2"):
            for i in range(30):
                rows.append(dict(zip(benchmark.META, [
                    f"{plate}-{i:02}", "DMSO" if i < 10 else f"drug-{i:02}",
                    "group_high", plate, plate, i < 10, "unknown"])))
        meta = benchmark.validate_metadata(pl.DataFrame(rows))
        split = benchmark.split_treatments(meta)
        self.assertEqual(split, benchmark.split_treatments(meta.reverse()))
        self.assertEqual(sum(v["role"] == "selection" for v in split.values()), 4)
        x = np.random.default_rng(1234).normal(size=(len(meta), 8))
        test = np.array([split.get(i, {}).get("role") == "test" for i in meta[benchmark.PERT]])
        changed = x.copy()
        changed[test] = changed[test] * 1000 + 99
        config = {"steps": [
            {"name": "clean_nans", "params": {"na_cutoff": .3}},
            {"name": "filter_features", "params": {"filters": [{"name": "variance_threshold"}]}},
            {"name": "normalize_standardize", "params": {}},
            {"name": "normalize_tvn_efaar", "params": {"n_components": 4}},
        ]}
        a, _, sa, audit = benchmark.fit_recipe(x, meta, config, "strict", split)
        b, _, sb, _ = benchmark.fit_recipe(changed, meta, config, "strict", split)
        self.assertEqual(benchmark.state_digest(sa), benchmark.state_digest(sb))
        np.testing.assert_array_equal(a[~test], b[~test])
        self.assertFalse(set(audit["fit_ids"]) & set(meta.filter(pl.Series(test))[benchmark.ID]))
        applied, matched = benchmark.apply_recipe(x, meta, sa)
        np.testing.assert_allclose(a, applied, rtol=1e-12, atol=1e-12)
        self.assertTrue(meta.equals(matched))
        _, _, sa, _ = benchmark.fit_recipe(x, meta, config, "paper-v11", {})
        _, _, sb, _ = benchmark.fit_recipe(changed, meta, config, "paper-v11", {})
        self.assertNotEqual(benchmark.state_digest(sa), benchmark.state_digest(sb))

    def test_reference_binding_and_mismatch(self):
        identity = {"target": "strict", "cohort_sha256": "fixed"}
        score = {"codec": "Raw", "recipe": "a", "pa_mean_nap": .2, "pc_mean_nap": .3,
                 "pa_pc_product": .06, "wells": 10, "controls": 3, "features": 5,
                 "pa_units": 4, "pc_targets": 2}
        reference = {"protocol": identity, "scores": [dict(score)],
                     "reference_kind": "local_cpu_replay_not_published"}
        self.assertEqual(benchmark.validate_reference(reference, identity, [score])["status"], "match")
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            benchmark.validate_reference(reference, {"target": "paper-v11"}, [score])
        reference["scores"][0]["pa_mean_nap"] = .4
        self.assertEqual(benchmark.validate_reference(reference, identity, [score])["status"], "mismatch")


if __name__ == "__main__":
    unittest.main()
