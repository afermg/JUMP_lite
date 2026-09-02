from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
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
    _verify_receipt_signature,
    acknowledge_production_errors,
    authorize_continuous,
    bootstrap_production,
    finalize_validation,
    migrate_producer,
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
                    "zarr_runtime_limits": {
                        "threading_max_workers": 4,
                        "async_concurrency": 4,
                    },
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

    def _acceptance_artifacts(self, config: ProductionConfig, digest: str):
        predecessor = json.loads((config.output_root / "producer.json").read_text())
        predecessor_sha = sha256_file(config.output_root / "producer.json")
        verification_path = self.root / f"verification-{config.site_count}.json"
        verification_path.write_text(
            json.dumps(
                {
                    "status": "valid",
                    "tranche": 0,
                    "sites": 256,
                    "tranche_digest": digest,
                }
            )
        )
        before_path = self.root / f"governor-before-{config.site_count}.json"
        post_path = self.root / f"governor-post-{config.site_count}.json"
        progress_before = {
            "MQ": {"receipt_backed_masks": 10, "canonical_profiles": 20},
            "lossless": {"receipt_backed_masks": 30, "canonical_profiles": 40},
        }
        progress_post = {
            "MQ": {"receipt_backed_masks": 11, "canonical_profiles": 22},
            "lossless": {"receipt_backed_masks": 33, "canonical_profiles": 44},
        }
        before_path.write_text(
            json.dumps(
                {
                    "metrics": {
                        "authoritative_progress": progress_before,
                        "io_pressure_avg10": 0.0,
                    }
                }
            )
        )
        post_path.write_text(
            json.dumps(
                {
                    "metrics": {
                        "authoritative_progress": progress_post,
                        "io_pressure_avg10": 0.0,
                    }
                }
            )
        )
        accepted_at = "2026-08-19T04:00:00+00:00"
        acceptance = self.root / f"acceptance-{config.site_count}.json"
        acceptance_value = {
            "format_version": "full-jump-one-tranche-acceptance-v1",
            "decision": "GO",
            "production_id": config.production_id,
            "config_sha256": config.digest,
            "inventory_digest": config.inventory_digest,
            "frozen_manifest": {
                "sha256": config.manifest_sha256,
                "bytes": config.manifest_size,
                "site_count": config.site_count,
            },
            "checkpoint": {
                "sha256": sha256_file(config.state_root / "checkpoint.json"),
                "next_index": 256,
                "completed_tranches": 1,
                "cumulative_errors": 0,
                "chain_head": digest,
            },
            "tranche0": {
                "record_sha256": sha256_file(
                    config.output_root / "tranches/00000000.json"
                ),
                "tranche_digest": digest,
                "site_count": 256,
            },
            "verification": {
                "artifact": {
                    "path": str(verification_path.resolve()),
                    "sha256": sha256_file(verification_path),
                },
                "status": "valid",
                "tranche": 0,
                "sites": 256,
                "tranche_digest": digest,
            },
            "governor": {
                "before": {
                    "path": str(before_path.resolve()),
                    "sha256": sha256_file(before_path),
                },
                "post": {
                    "path": str(post_path.resolve()),
                    "sha256": sha256_file(post_path),
                },
                "feature_deltas": {
                    "MQ": {"receipt_backed_masks": 1, "canonical_profiles": 2},
                    "lossless": {
                        "receipt_backed_masks": 3,
                        "canonical_profiles": 4,
                    },
                },
                "io_pressure": {
                    "before_some_avg10": 0,
                    "after_some_avg10": 0,
                    "max_some_avg10": 0,
                },
            },
            "predecessor_producer": {
                "sha256": predecessor_sha,
                "git_commit": predecessor["software"]["git_commit"],
            },
            "reviews": {
                name: {
                    "identifier": f"{name}-review-{config.site_count}",
                    "reviewed_at": accepted_at,
                }
                for name in ("code", "science", "ops")
            },
            "accepted_at": accepted_at,
        }
        acceptance.write_text(
            json.dumps(acceptance_value, sort_keys=True, indent=2) + "\n"
        )
        successor = {
            **predecessor["software"],
            "git_commit": "9" * 40,
            "source_tree_sha256": "8" * 64,
        }
        migration = self.root / f"migration-{config.site_count}.json"
        migration_value = {
            "format_version": "producer-migration-acceptance-v1",
            "decision": "GO",
            "production_id": config.production_id,
            "config_sha256": config.digest,
            "inventory_digest": config.inventory_digest,
            "checkpoint_sha256": acceptance_value["checkpoint"]["sha256"],
            "tranche0_record_sha256": acceptance_value["tranche0"]["record_sha256"],
            "tranche0_digest": digest,
            "one_tranche_acceptance": {
                "path": str(acceptance.resolve()),
                "sha256": sha256_file(acceptance),
            },
            "predecessor": {
                "producer_sha256": predecessor_sha,
                "software": predecessor["software"],
            },
            "successor": {"software": successor},
            "review": {"identifier": "migration-review", "reviewed_at": accepted_at},
            "approved_at": accepted_at,
        }
        migration.write_text(
            json.dumps(migration_value, sort_keys=True, indent=2) + "\n"
        )
        return acceptance, migration, successor

    def _migrate_and_authorize(self, config: ProductionConfig, digest: str):
        acceptance, migration, successor = self._acceptance_artifacts(config, digest)
        with (
            self._identity_patch(),
            mock.patch(
                "jump_full_compression.production.software_identity",
                return_value=successor,
            ),
        ):
            migrate_producer(
                config,
                acceptance,
                sha256_file(acceptance),
                migration,
                sha256_file(migration),
                True,
            )
        return acceptance, migration

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
        self.assertEqual(result["status"], "stopped")
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
        telemetry = json.loads((config.state_root / "compression.json").read_text())
        self.assertEqual(telemetry["state"], "error")
        self.assertGreaterEqual(telemetry["peak_tasks"], telemetry["current_tasks"])

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
            (
                "zarr_runtime_limits",
                {"threading_max_workers": 4, "async_concurrency": 5},
            ),
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

    def test_continuous_systemd_unit_is_separate_and_marker_gated(self):
        base = Path("ops/systemd")
        bounded = (base / "jump-full-production-compress.service").read_text()
        continuous = (base / "jump-full-production-continuous.service").read_text()
        self.assertIn("--max-tranches 1", bounded)
        self.assertNotIn("--continuous", bounded)
        self.assertIn("--continuous", continuous)
        self.assertNotIn("--max-tranches", continuous)
        self.assertIn("continuous-authorization.json", continuous)
        self.assertIn("Restart=no", continuous)
        for contract in (
            "CPUAffinity=64-80",
            "Nice=19",
            "CPUWeight=1",
            "IOWeight=1",
            "TasksMax=256",
            "ProtectSystem=strict",
        ):
            self.assertIn(contract, bounded)
            self.assertIn(contract, continuous)

    def test_one_tranche_gate_requires_exactly_one(self):
        config = self._config(1)
        for value in (0, -1, 2, 100):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_production(config, value, False)
        with self.assertRaises(ValueError):
            run_production(config, 1, False, continuous=True)

    def test_continuous_authorization_and_multi_tranche_terminal_telemetry(self):
        config = self._config(513)
        self._bootstrap(config)
        with self._identity_patch():
            first = run_production(config, 1, True)
        self.assertEqual(first["status"], "session-complete")
        state = json.loads((config.state_root / "compression.json").read_text())
        self.assertEqual(state["state"], "session-complete")
        self.assertEqual(state["processed"], 256)
        for field in (
            "current_tasks",
            "peak_tasks",
            "rss_bytes",
            "max_rss_bytes",
            "affinity",
        ):
            self.assertIn(field, state)
        digest = json.loads(
            (config.output_root / "tranches/00000000.json").read_text()
        )["tranche_digest"]
        with mock.patch(
            "jump_full_compression.production.AUTHORIZED_FIRST_TRANCHE_DIGEST", digest
        ):
            acceptance, _ = self._migrate_and_authorize(config, digest)
        with (
            self._identity_patch(),
            mock.patch(
                "jump_full_compression.production.AUTHORIZED_FIRST_TRANCHE_DIGEST",
                digest,
            ),
            self.assertRaisesRegex(RuntimeError, "bounded production is forbidden"),
        ):
            run_production(config, 1, True)
        with (
            self._identity_patch(),
            mock.patch(
                "jump_full_compression.production.AUTHORIZED_FIRST_TRANCHE_DIGEST",
                digest,
            ),
        ):
            preview = authorize_continuous(
                config, acceptance, sha256_file(acceptance), False
            )
            self.assertEqual(preview["status"], "would-authorize-continuous")
            authorize_continuous(config, acceptance, sha256_file(acceptance), True)
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                authorize_continuous(config, acceptance, sha256_file(acceptance), True)
            self._unpause(config)
            result = run_production(config, None, True, continuous=True)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["completed_tranches"], 3)
        with (
            self._identity_patch(),
            mock.patch(
                "jump_full_compression.production.AUTHORIZED_FIRST_TRANCHE_DIGEST",
                digest,
            ),
        ):
            restart = run_production(config, None, False, continuous=True)
        self.assertEqual(restart["status"], "would-run-continuous")
        self.assertEqual(restart["next_index"], 513)
        state = json.loads((config.state_root / "compression.json").read_text())
        self.assertEqual(state["state"], "complete")
        self.assertEqual(state["processed"], 513)

    def test_continuous_missing_marker_and_tampered_acceptance_fail_closed(self):
        config = self._config(257)
        self._bootstrap(config)
        with self._identity_patch():
            run_production(config, 1, True)
            with self.assertRaisesRegex(RuntimeError, "authorization missing"):
                run_production(config, None, False, continuous=True)
        digest = json.loads(
            (config.output_root / "tranches/00000000.json").read_text()
        )["tranche_digest"]
        with mock.patch(
            "jump_full_compression.production.AUTHORIZED_FIRST_TRANCHE_DIGEST", digest
        ):
            acceptance, _ = self._migrate_and_authorize(config, digest)
        with (
            self._identity_patch(),
            mock.patch(
                "jump_full_compression.production.AUTHORIZED_FIRST_TRANCHE_DIGEST",
                digest,
            ),
        ):
            authorize_continuous(config, acceptance, sha256_file(acceptance), True)
            value = json.loads(acceptance.read_text())
            value["decision"] = "NO"
            acceptance.write_text(json.dumps(value))
            self._unpause(config)
            with self.assertRaisesRegex(
                RuntimeError, "acceptance receipt binding drift"
            ):
                run_production(config, None, True, continuous=True)

    def test_continuous_pause_between_tranches_is_terminal(self):
        config = self._config(513)
        self._bootstrap(config)
        with self._identity_patch():
            run_production(config, 1, True)
        digest = json.loads(
            (config.output_root / "tranches/00000000.json").read_text()
        )["tranche_digest"]
        with mock.patch(
            "jump_full_compression.production.AUTHORIZED_FIRST_TRANCHE_DIGEST", digest
        ):
            acceptance, _ = self._migrate_and_authorize(config, digest)

        def pause_after_record(point):
            if point == "after_tranche_record":
                control_path = config.state_root / "control.json"
                control = json.loads(control_path.read_text())
                control["paused"] = True
                control_path.write_text(json.dumps(control))

        with (
            self._identity_patch(),
            mock.patch(
                "jump_full_compression.production.AUTHORIZED_FIRST_TRANCHE_DIGEST",
                digest,
            ),
        ):
            authorize_continuous(config, acceptance, sha256_file(acceptance), True)
            self._unpause(config)
            with mock.patch(
                "jump_full_compression.production.FAULT_HOOK", pause_after_record
            ):
                result = run_production(config, None, True, continuous=True)
        self.assertEqual(result["status"], "paused")
        state = json.loads((config.state_root / "compression.json").read_text())
        self.assertEqual(state["state"], "paused")
        self.assertEqual(state["processed"], state["next_index"])

    def test_production_approval_rejects_unsigned_arbitrary_and_tampered_signatures(
        self,
    ):
        receipt = self.root / "signed-review.json"
        receipt.write_text('{"decision":"GO"}\n')
        key = self.root / "untrusted-review-key"
        subprocess.run(
            ["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)],
            check=True,
        )
        subprocess.run(
            [
                "ssh-keygen",
                "-Y",
                "sign",
                "-f",
                str(key),
                "-n",
                "jump-full-production-v1",
                str(receipt),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        signature = receipt.with_suffix(".json.sig")
        with self.assertRaisesRegex(RuntimeError, "required"):
            _verify_receipt_signature(
                receipt.read_bytes(), None, test_mode=False, label="test receipt"
            )
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            _verify_receipt_signature(
                receipt.read_bytes(), signature, test_mode=False, label="test receipt"
            )
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            _verify_receipt_signature(
                receipt.read_bytes() + b"tamper",
                signature,
                test_mode=False,
                label="test receipt",
            )
        completed = types.SimpleNamespace(
            returncode=0, stdout=b"Good signature", stderr=b""
        )
        with mock.patch(
            "jump_full_compression.production.subprocess.run", return_value=completed
        ):
            observed = _verify_receipt_signature(
                receipt.read_bytes(), signature, test_mode=False, label="test receipt"
            )
        self.assertEqual(observed, sha256_file(signature))

    def test_strict_acceptance_rejects_arbitrary_no_and_bad_evidence(self):
        config = self._config(257)
        self._bootstrap(config)
        with self._identity_patch():
            run_production(config, 1, True)
        digest = json.loads(
            (config.output_root / "tranches/00000000.json").read_text()
        )["tranche_digest"]
        arbitrary = self.root / "arbitrary.txt"
        arbitrary.write_text("reviewed and accepted")
        with (
            self._identity_patch(),
            mock.patch(
                "jump_full_compression.production.AUTHORIZED_FIRST_TRANCHE_DIGEST",
                digest,
            ),
            self.assertRaisesRegex(RuntimeError, "malformed"),
        ):
            authorize_continuous(config, arbitrary, sha256_file(arbitrary), False)
        acceptance, migration, successor = self._acceptance_artifacts(config, digest)
        for mutation in (
            lambda value: value.update(decision="NO"),
            lambda value: value["reviews"]["ops"].update(identifier=""),
            lambda value: value["governor"]["feature_deltas"]["MQ"].update(
                receipt_backed_masks=0
            ),
            lambda value: value["governor"]["io_pressure"].update(max_some_avg10=0.1),
        ):
            value = json.loads(acceptance.read_text())
            mutation(value)
            candidate = self.root / f"bad-{len(list(self.root.glob('bad-*')))}.json"
            candidate.write_text(json.dumps(value))
            migration_value = json.loads(migration.read_text())
            migration_value["one_tranche_acceptance"] = {
                "path": str(candidate.resolve()),
                "sha256": sha256_file(candidate),
            }
            bad_migration = candidate.with_suffix(".migration.json")
            bad_migration.write_text(json.dumps(migration_value))
            with (
                self._identity_patch(),
                mock.patch(
                    "jump_full_compression.production.AUTHORIZED_FIRST_TRANCHE_DIGEST",
                    digest,
                ),
                mock.patch(
                    "jump_full_compression.production.software_identity",
                    return_value=successor,
                ),
                self.assertRaises(RuntimeError),
            ):
                migrate_producer(
                    config,
                    candidate,
                    sha256_file(candidate),
                    bad_migration,
                    sha256_file(bad_migration),
                    False,
                )
        acceptance, migration, successor = self._acceptance_artifacts(config, digest)
        value = json.loads(acceptance.read_text())
        before = Path(value["governor"]["before"]["path"])
        before_value = json.loads(before.read_text())
        before_value["metrics"]["io_pressure_avg10"] = 99.0
        before.write_text(json.dumps(before_value))
        value["governor"]["before"]["sha256"] = sha256_file(before)
        acceptance.write_text(json.dumps(value))
        migration_value = json.loads(migration.read_text())
        migration_value["one_tranche_acceptance"]["sha256"] = sha256_file(acceptance)
        migration.write_text(json.dumps(migration_value))
        with (
            self._identity_patch(),
            mock.patch(
                "jump_full_compression.production.AUTHORIZED_FIRST_TRANCHE_DIGEST",
                digest,
            ),
            mock.patch(
                "jump_full_compression.production.software_identity",
                return_value=successor,
            ),
            self.assertRaisesRegex(RuntimeError, "I/O pressure artifacts"),
        ):
            migrate_producer(
                config,
                acceptance,
                sha256_file(acceptance),
                migration,
                sha256_file(migration),
                False,
            )

    def test_migration_preserves_tranche_zero_and_converges_after_each_fault(self):
        class SimulatedKill(BaseException):
            pass

        for index, fault in enumerate(
            (
                "after_migration_pause",
                "after_migration_history",
                "after_migration_transition",
                "after_migration_current_producer",
                "after_migration_checkpoint",
            )
        ):
            with self.subTest(fault=fault):
                config = self._config(257)
                config = ProductionConfig(
                    **{
                        **config.__dict__,
                        "output_root": self.root / f"migration-output-{index}",
                        "state_root": self.root / f"migration-state-{index}",
                    }
                )
                self._bootstrap(config)
                with self._identity_patch():
                    run_production(config, 1, True)
                digest = json.loads(
                    (config.output_root / "tranches/00000000.json").read_text()
                )["tranche_digest"]
                immutable = {
                    str(path.relative_to(config.output_root)): sha256_file(path)
                    for root in (
                        config.output_root / "receipts/00000000",
                        config.output_root / "codecs/jpegxl_lossy_hq.zarr",
                        config.output_root / "codecs/jpegxl_lossy_mq.zarr",
                        config.output_root / "tranches",
                    )
                    for path in root.rglob("*")
                    if path.is_file()
                }
                acceptance, migration, successor = self._acceptance_artifacts(
                    config, digest
                )
                fired = False

                def inject(point):
                    nonlocal fired
                    if point == fault and not fired:
                        fired = True
                        raise SimulatedKill(point)

                common = dict(
                    config=config,
                    one_tranche_acceptance=acceptance,
                    one_tranche_acceptance_sha256=sha256_file(acceptance),
                    migration_acceptance=migration,
                    migration_acceptance_sha256=sha256_file(migration),
                    apply=True,
                )
                if index == 0:
                    accepted_config_digest = config.digest
                    with (
                        self._identity_patch(),
                        mock.patch(
                            "jump_full_compression.production.AUTHORIZED_FIRST_TRANCHE_DIGEST",
                            digest,
                        ),
                        mock.patch(
                            "jump_full_compression.production.software_identity",
                            return_value=successor,
                        ),
                    ):
                        preview = migrate_producer(**{**common, "apply": False})
                    self.assertEqual(preview["status"], "would-migrate-producer")
                    self.assertEqual(config.digest, accepted_config_digest)
                    self.assertFalse((config.output_root / "producers").exists())
                    self.assertFalse((config.output_root / "transitions").exists())
                with (
                    self._identity_patch(),
                    mock.patch(
                        "jump_full_compression.production.AUTHORIZED_FIRST_TRANCHE_DIGEST",
                        digest,
                    ),
                    mock.patch(
                        "jump_full_compression.production.software_identity",
                        return_value=successor,
                    ),
                    mock.patch("jump_full_compression.production.FAULT_HOOK", inject),
                    self.assertRaisesRegex(SimulatedKill, fault),
                ):
                    migrate_producer(**common)
                control_after_fault = json.loads(
                    (config.state_root / "control.json").read_text()
                )
                telemetry_after_fault = json.loads(
                    (config.state_root / "compression.json").read_text()
                )
                self.assertTrue(control_after_fault["paused"])
                self.assertEqual(telemetry_after_fault["state"], "session-complete")
                self.assertEqual(telemetry_after_fault["processed"], 256)
                with (
                    self._identity_patch(),
                    self.assertRaises(RuntimeError),
                ):
                    run_production(config, None, False, continuous=True)
                with (
                    self._identity_patch(),
                    mock.patch(
                        "jump_full_compression.production.AUTHORIZED_FIRST_TRANCHE_DIGEST",
                        digest,
                    ),
                    mock.patch(
                        "jump_full_compression.production.software_identity",
                        return_value=successor,
                    ),
                ):
                    result = migrate_producer(**common)
                    self.assertEqual(result["status"], "producer-migrated")
                    self.assertEqual(verify_tranche(config, 0)["status"], "valid")
                observed = {
                    str(path.relative_to(config.output_root)): sha256_file(path)
                    for relative in immutable
                    for path in [config.output_root / relative]
                }
                self.assertEqual(observed, immutable)
                checkpoint = json.loads(
                    (config.state_root / "checkpoint.json").read_text()
                )
                current_sha = sha256_file(config.output_root / "producer.json")
                self.assertEqual(checkpoint["producer_sha256"], current_sha)
                state = json.loads((config.state_root / "compression.json").read_text())
                control = json.loads((config.state_root / "control.json").read_text())
                self.assertEqual(state["state"], "session-complete")
                self.assertEqual(state["processed"], 256)
                self.assertTrue(control["paused"])

    def test_continuous_stop_between_tranches_is_terminal(self):
        config = self._config(513)
        self._bootstrap(config)
        with self._identity_patch():
            run_production(config, 1, True)
        digest = json.loads(
            (config.output_root / "tranches/00000000.json").read_text()
        )["tranche_digest"]
        with mock.patch(
            "jump_full_compression.production.AUTHORIZED_FIRST_TRANCHE_DIGEST", digest
        ):
            acceptance, _ = self._migrate_and_authorize(config, digest)

        def stop_after_record(point):
            if point == "after_tranche_record":
                STOP.set()

        with (
            self._identity_patch(),
            mock.patch(
                "jump_full_compression.production.AUTHORIZED_FIRST_TRANCHE_DIGEST",
                digest,
            ),
        ):
            authorize_continuous(config, acceptance, sha256_file(acceptance), True)
            self._unpause(config)
            with mock.patch(
                "jump_full_compression.production.FAULT_HOOK", stop_after_record
            ):
                result = run_production(config, None, True, continuous=True)
        self.assertEqual(result["status"], "stopped")
        state = json.loads((config.state_root / "compression.json").read_text())
        self.assertEqual(state["state"], "stopped")
        self.assertEqual(state["processed"], state["next_index"])


if __name__ == "__main__":
    unittest.main()
