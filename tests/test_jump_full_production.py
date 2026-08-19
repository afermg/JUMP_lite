from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import types
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
    ManifestSnapshot,
    _load_producer,
    _rows_slice,
    _source_observation_valid,
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
        def validate(_config, snapshot):
            snapshot.load_metadata()
            return {"inventory_digest": "a" * 64}

        return mock.patch(
            "jump_full_compression.production._validate_identity",
            side_effect=validate,
        )

    def _bootstrap(self, config: ProductionConfig) -> None:
        with (
            self._identity_patch(),
            mock.patch(
                "jump_full_compression.production.software_identity",
                return_value={
                    "git_commit": "1" * 40,
                    "tracked_tree_clean": True,
                    "python_executable": "/frozen/python",
                    "python_executable_sha256": "2" * 64,
                },
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
            first = run_production(config, 1, True)
            second = run_production(config, 1, True)
            result = run_production(config, 1, True)
        self.assertEqual(first["next_index"], 256)
        self.assertEqual(second["next_index"], 512)
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
        rogue = config.output_root / "codecs/jpegxl_lossy_mq.zarr/rogue-partial"
        rogue.mkdir()
        (rogue / ".zarray").write_text("{}")
        with self._identity_patch(), self.assertRaises(RuntimeError):
            finalize_validation(config)
        shutil.rmtree(rogue)

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
        with (
            ManifestSnapshot(config.manifest) as snapshot,
            self.assertRaises(RuntimeError),
        ):
            _validate_identity(config, snapshot)
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

    def test_one_pool_reuses_persistent_native_threads_across_subbatches(self):
        config = self._config(13)
        self._bootstrap(config)
        control_path = config.state_root / "control.json"

        class PersistentNativePool:
            instances = 0
            exits = 0
            native_threads = 14
            batch_sizes = []

            def __init__(self, max_workers):
                self.max_workers = max_workers
                self.started = False
                type(self).instances += 1

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                type(self).exits += 1
                return False

            def map(self, function, rows):
                rows = list(rows)
                if not self.started:
                    type(self).native_threads += self.max_workers
                    self.started = True
                type(self).batch_sizes.append(len(rows))
                results = [function(row) for row in rows]
                if len(type(self).batch_sizes) == 1:
                    control = json.loads(control_path.read_text())
                    control["desired_workers"] = 8
                    control_path.write_text(json.dumps(control))
                return results

        checks = []

        def check(_config, additional):
            checks.append(additional)
            if PersistentNativePool.native_threads + additional > 24:
                raise RuntimeError("runtime task ceiling exceeded")

        with (
            self._identity_patch(),
            mock.patch(
                "jump_full_compression.production.ThreadPoolExecutor",
                PersistentNativePool,
            ),
            mock.patch(
                "jump_full_compression.production._production_task_check",
                side_effect=check,
            ),
        ):
            result = run_production(config, 1, True)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(PersistentNativePool.instances, 1)
        self.assertEqual(PersistentNativePool.exits, 1)
        self.assertEqual(PersistentNativePool.batch_sizes, [4, 4, 4, 1])
        self.assertEqual(checks, [4, 0, 0, 0, 0])
        self.assertEqual(PersistentNativePool.native_threads, 18)

    def test_task_ceiling_failure_prevents_pool_and_tranche(self):
        config = self._config(1)
        self._bootstrap(config)
        with (
            self._identity_patch(),
            mock.patch(
                "jump_full_compression.production._production_task_check",
                side_effect=RuntimeError(
                    "runtime task ceiling exceeded: observed=22 additional=4 ceiling=24"
                ),
            ),
            mock.patch(
                "jump_full_compression.production.ThreadPoolExecutor"
            ) as executor,
            self.assertRaisesRegex(RuntimeError, "observed=22 additional=4"),
        ):
            run_production(config, 1, True)
        executor.assert_not_called()
        checkpoint = json.loads((config.state_root / "checkpoint.json").read_text())
        self.assertEqual(checkpoint["next_index"], 0)
        self.assertEqual(checkpoint["cumulative_errors"], 1)
        self.assertFalse((config.output_root / "tranches/00000000.json").exists())

    def test_after_record_crash_adoption_consumes_restart_tranche_allowance(self):
        config = self._config(513)
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

        # This invocation may adopt tranche 0, but must not also build tranche 1.
        with self._identity_patch():
            adopted = run_production(config, 1, True)
        self.assertEqual(adopted["status"], "session-complete")
        self.assertEqual(adopted["committed_tranches"], 1)
        self.assertEqual(adopted["next_index"], 256)
        self.assertFalse((config.output_root / "tranches/00000001.json").exists())

        # A later, separate invocation retains permission to build tranche 1.
        with self._identity_patch():
            second = run_production(config, 1, True)
        self.assertEqual(second["status"], "session-complete")
        self.assertEqual(second["next_index"], 512)
        self.assertTrue((config.output_root / "tranches/00000001.json").is_file())

    def test_corrupt_after_write_blocks_tranche_commit(self):
        config = self._config(4)
        self._bootstrap(config)
        fired = False

        def corrupt(point):
            nonlocal fired
            if point == "before_tranche_validation" and not fired:
                fired = True
                chunk = next(
                    (config.output_root / "codecs/jpegxl_lossy_hq.zarr").glob("*/0.0.0")
                )
                chunk.write_bytes(chunk.read_bytes() + b"corrupt")

        with (
            self._identity_patch(),
            mock.patch("jump_full_compression.production.FAULT_HOOK", corrupt),
            self.assertRaises(Exception),
        ):
            run_production(config, 1, True)
        self.assertFalse((config.output_root / "tranches/00000000.json").exists())
        checkpoint = json.loads((config.state_root / "checkpoint.json").read_text())
        self.assertEqual(checkpoint["next_index"], 0)

    def test_live_source_observation_requires_version_and_last_modified(self):
        uri = "s3://cellpainting-gallery/object.tif"
        legacy = {"uri": uri, "size": 1, "etag": "abc"}
        self.assertTrue(_source_observation_valid(legacy, uri, True))
        self.assertFalse(_source_observation_valid(legacy, uri, False))
        self.assertFalse(
            _source_observation_valid({**legacy, "version_id": "v1"}, uri, False)
        )
        self.assertTrue(
            _source_observation_valid(
                {**legacy, "version_id": "v1", "last_modified": "now"},
                uri,
                False,
            )
        )

    def test_empty_source_observation_rejected_before_write(self):
        config = self._config(1)
        self._bootstrap(config)
        stack = np.zeros((5, 8, 9), dtype=np.uint16)
        with (
            self._identity_patch(),
            mock.patch(
                "jump_full_compression.production.decode_stack",
                return_value=(stack, [{} for _ in range(5)]),
            ),
            self.assertRaises(RuntimeError),
        ):
            run_production(config, 1, True)
        for codec in ("jpegxl_lossy_hq", "jpegxl_lossy_mq"):
            self.assertEqual(
                list((config.output_root / "codecs" / f"{codec}.zarr").iterdir()),
                [],
            )

    def test_manifest_snapshot_survives_path_replacement_without_materializing(self):
        config = self._config(513)
        original_binding = {
            "bytes": config.manifest_size,
            "sha256": config.manifest_sha256,
        }
        with ManifestSnapshot(config.manifest) as snapshot:
            self.assertEqual(snapshot.binding, original_binding)
            self.assertNotIn("bytes", snapshot.__dict__)
            snapshot.load_metadata()
            replacement = self.root / "replacement.parquet"
            row = (
                pl.read_parquet(config.manifest)
                .head(1)
                .with_columns(pl.lit(9999).alias("Metadata_Site"))
            )
            row.write_parquet(replacement)
            os.replace(replacement, config.manifest)
            observed = _rows_slice(snapshot, 500, 1)
            self.assertEqual(observed[0]["Metadata_Site_Key"].rsplit("__", 1)[1], "500")
            self.assertGreaterEqual(len(snapshot.row_groups), 1)

    def test_live_software_must_equal_stored_producer(self):
        config = self._config(1)
        self._bootstrap(config)
        stored = json.loads((config.output_root / "producer.json").read_text())
        live_config = types.SimpleNamespace(
            output_root=config.output_root,
            production_id=config.production_id,
            digest=config.digest,
            inventory_digest=config.inventory_digest,
            test_mode=False,
        )
        for field, changed_value in (
            ("python_executable", "/different/python"),
            ("python_executable_sha256", "3" * 64),
        ):
            changed = dict(stored["software"])
            changed[field] = changed_value
            with (
                self.subTest(field=field),
                mock.patch(
                    "jump_full_compression.production.software_identity",
                    return_value=changed,
                ),
                self.assertRaisesRegex(RuntimeError, "checkout/interpreter/dependency"),
            ):
                _load_producer(live_config)
        with mock.patch(
            "jump_full_compression.production.software_identity",
            return_value=stored["software"],
        ):
            _, digest = _load_producer(live_config)
        self.assertEqual(digest, sha256_file(config.output_root / "producer.json"))

    def test_one_tranche_gate_and_no_continuous_mode(self):
        config = self._config(1)
        for value in (0, -1, 2, 100):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_production(config, value, False)
        self.assertFalse(hasattr(config, "continuous"))


if __name__ == "__main__":
    unittest.main()
