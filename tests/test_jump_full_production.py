from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import polars as pl

from jump_full_compression.model import (
    COMPRESSION_CPUS,
    ProductionConfig,
    sha256_file,
)
from jump_full_compression.pipeline import STOP
from jump_full_compression.production import (
    _validate_identity,
    acknowledge_production_errors,
    bootstrap_production,
    finalize_validation,
    run_production,
    verify_tranche,
)


class ProductionRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        STOP.clear()
        self.images = []
        for channel in range(5):
            path = self.root / f"channel-{channel}.tif"
            Image.fromarray(
                np.full(
                    (8, 9), channel * 10 + np.arange(72).reshape(8, 9), dtype=np.uint16
                )
            ).save(path)
            self.images.append(path.as_uri())

    def tearDown(self):
        STOP.clear()
        self.tmp.cleanup()

    def _config(self, sites: int) -> ProductionConfig:
        rows = []
        for site in range(sites):
            row = {
                "Metadata_Source": "source_1",
                "Metadata_Batch": "Batch1",
                "Metadata_Plate": "P0001",
                "Metadata_Well": f"A{site // 1000 + 1:02d}",
                "Metadata_Site": site,
            }
            for channel, uri in zip(("AGP", "DNA", "ER", "Mito", "RNA"), self.images):
                row[f"URL_Orig{channel}"] = uri
            rows.append(row)
        manifest = self.root / "manifest.parquet"
        pl.DataFrame(rows).write_parquet(manifest)
        artifacts = []
        for name in ("audit", "build", "policy", "objects", "sites", "qc"):
            path = self.root / f"{name}.json"
            path.write_text("{}\n")
            artifacts.append(path)
        return ProductionConfig(
            "test-production",
            manifest,
            artifacts[0],
            artifacts[1],
            artifacts[2],
            artifacts[3],
            artifacts[4],
            artifacts[5],
            self.root / "output",
            self.root / "state",
            "a" * 64,
            sha256_file(manifest),
            manifest.stat().st_size,
            sha256_file(artifacts[0]),
            sha256_file(artifacts[1]),
            sha256_file(artifacts[2]),
            sha256_file(artifacts[3]),
            sha256_file(artifacts[4]),
            sha256_file(artifacts[5]),
            sites,
            test_mode=True,
        )

    def _identity_patch(self):
        return mock.patch(
            "jump_full_compression.production._validate_identity",
            return_value={"inventory_digest": "a" * 64},
        )

    def _bootstrap(self, config: ProductionConfig) -> None:
        with (
            self._identity_patch(),
            mock.patch(
                "jump_full_compression.production.software_identity",
                return_value={"git_commit": "1" * 40, "tracked_tree_clean": True},
            ),
        ):
            bootstrap_production(config, True)
        self._unpause(config)

    def _unpause(self, config: ProductionConfig, acknowledged: int = 0) -> None:
        control = {
            "format_version": "full-jump-compression-control-v2",
            "candidate_id": config.production_id,
            "config_sha256": config.digest,
            "paused": False,
            "desired_workers": 4,
            "max_workers": 16,
            "consecutive_healthy_windows": 0,
            "compression_cpus": list(COMPRESSION_CPUS),
            "acknowledged_error_count": acknowledged,
            "acknowledgement_source": "explicit-cli" if acknowledged else "zero",
            "reasons": ["test"],
            "observed_at_unix": time.time(),
            "feature_processes_mutated": False,
        }
        (config.state_root / "control.json").write_text(json.dumps(control))

    def test_513_sites_commit_exact_three_compact_tranches_and_tamper_fails(self):
        config = self._config(513)
        self._bootstrap(config)
        with self._identity_patch():
            result = run_production(config, 3, True)
        self.assertEqual(result["status"], "complete")
        checkpoint_path = config.state_root / "checkpoint.json"
        checkpoint = json.loads(checkpoint_path.read_text())
        self.assertEqual(checkpoint["next_index"], 513)
        self.assertEqual(checkpoint["completed_tranches"], 3)
        self.assertLess(checkpoint_path.stat().st_size, 2048)
        self.assertEqual(
            sorted(path.name for path in (config.output_root / "tranches").iterdir()),
            ["00000000.json", "00000001.json", "00000002.json"],
        )
        with self._identity_patch():
            self.assertEqual(verify_tranche(config, 2)["sites"], 1)
            self.assertEqual(finalize_validation(config)["sites"], 513)

        receipt = next((config.output_root / "receipts/00000000").glob("*.json"))
        original = receipt.read_bytes()
        receipt.write_text("{}\n")
        with self._identity_patch(), self.assertRaises(RuntimeError):
            verify_tranche(config, 0)
        receipt.write_bytes(original)

        chunk = next(
            (config.output_root / "codecs/jpegxl_lossy_hq.zarr").glob("*/0.0.0")
        )
        original = chunk.read_bytes()
        chunk.write_bytes(original + b"tamper")
        with self._identity_patch(), self.assertRaises(Exception):
            verify_tranche(config, 0)
        chunk.write_bytes(original)

        record = config.output_root / "tranches/00000000.json"
        original = record.read_bytes()
        value = json.loads(original)
        value["previous_tranche_digest"] = "f" * 64
        record.write_text(json.dumps(value))
        with self._identity_patch(), self.assertRaises(RuntimeError):
            run_production(config, 1, True)
        record.write_bytes(original)

        checkpoint_original = checkpoint_path.read_bytes()
        value = json.loads(checkpoint_original)
        value["chain_head"] = "e" * 64
        checkpoint_path.write_text(json.dumps(value))
        with self._identity_patch(), self.assertRaises(RuntimeError):
            run_production(config, 1, True)
        checkpoint_path.write_bytes(checkpoint_original)

        producer = config.output_root / "producer.json"
        original = producer.read_bytes()
        producer.write_text("{}\n")
        with self._identity_patch(), self.assertRaises(RuntimeError):
            verify_tranche(config, 0)
        producer.write_bytes(original)

        manifest_original = config.manifest.read_bytes()
        config.manifest.write_bytes(manifest_original + b"tamper")
        with self.assertRaises(RuntimeError):
            _validate_identity(config)
        config.manifest.write_bytes(manifest_original)

    def test_orphan_site_and_site_receipt_crashes_are_recovered_uncommitted(self):
        for fault in ("before_site_receipt", "after_site_receipt"):
            with self.subTest(fault=fault):
                config = self._config(1)
                # Give each subtest disjoint roots and manifest artifacts.
                suffix = fault.replace("_", "-")
                config = ProductionConfig(
                    **{
                        **config.__dict__,
                        "output_root": self.root / f"out-{suffix}",
                        "state_root": self.root / f"state-{suffix}",
                    }
                )
                self._bootstrap(config)
                fired = False

                def inject(point):
                    nonlocal fired
                    if point == fault and not fired:
                        fired = True
                        raise RuntimeError(fault)

                with (
                    self._identity_patch(),
                    mock.patch("jump_full_compression.production.FAULT_HOOK", inject),
                    self.assertRaises(RuntimeError),
                ):
                    run_production(config, 1, True)
                checkpoint = json.loads(
                    (config.state_root / "checkpoint.json").read_text()
                )
                self.assertEqual(checkpoint["next_index"], 0)
                count = checkpoint["cumulative_errors"]
                with self._identity_patch():
                    acknowledge_production_errors(config, count, True)
                self._unpause(config, count)
                with self._identity_patch():
                    self.assertEqual(
                        run_production(config, 1, True)["status"], "complete"
                    )

    def test_stop_after_worker_subbatch_does_not_commit_partial_tranche(self):
        config = self._config(5)
        self._bootstrap(config)

        def stop_after_receipt(point):
            if point == "after_site_receipt":
                STOP.set()

        with (
            self._identity_patch(),
            mock.patch(
                "jump_full_compression.production.FAULT_HOOK", stop_after_receipt
            ),
        ):
            result = run_production(config, 1, True)
        self.assertEqual(result["status"], "stopped-partial")
        checkpoint = json.loads((config.state_root / "checkpoint.json").read_text())
        self.assertEqual(checkpoint["next_index"], 0)
        self.assertFalse((config.output_root / "tranches/00000000.json").exists())
        STOP.clear()
        with self._identity_patch():
            self.assertEqual(run_production(config, 1, True)["status"], "complete")

    def test_after_record_crash_is_adopted_only_after_full_validation(self):
        config = self._config(3)
        self._bootstrap(config)
        fired = False

        def inject(point):
            nonlocal fired
            if point == "after_tranche_record" and not fired:
                fired = True
                raise RuntimeError(point)

        with (
            self._identity_patch(),
            mock.patch("jump_full_compression.production.FAULT_HOOK", inject),
            self.assertRaises(RuntimeError),
        ):
            run_production(config, 1, True)
        checkpoint = json.loads((config.state_root / "checkpoint.json").read_text())
        self.assertEqual(checkpoint["next_index"], 0)
        count = checkpoint["cumulative_errors"]
        with self._identity_patch():
            acknowledge_production_errors(config, count, True)
        self._unpause(config, count)
        with self._identity_patch():
            result = run_production(config, 1, True)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["next_index"], 3)

    def test_positive_max_tranches_and_no_continuous_mode(self):
        config = self._config(1)
        with self._identity_patch(), self.assertRaises(ValueError):
            run_production(config, 0, False)
        self.assertFalse(hasattr(config, "continuous"))


if __name__ == "__main__":
    unittest.main()
