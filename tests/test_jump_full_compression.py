from __future__ import annotations

import fcntl
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import polars as pl

from jump_full_compression.governor import GovernorPaths, decide, run_governor
from jump_full_compression.inventory import (
    audit_inventory,
    inventory_digest_from_report,
    load_audit,
)
from jump_full_compression.model import (
    CODECS,
    COMPRESSION_CPUS,
    CandidateConfig,
    THREAD_ENV,
)
from jump_full_compression.pipeline import (
    STOP,
    _control,
    acknowledge_errors,
    bootstrap_candidate,
    read_source,
    run_candidate,
    sha256_file,
    validate_adoption_seam,
)


class FullJumpCompressionTests(unittest.TestCase):
    def setUp(self):
        STOP.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        STOP.clear()
        self.temp.cleanup()

    def _tiff(self, name: str, value: int) -> str:
        path = self.root / name
        Image.fromarray(np.full((12, 16), value, dtype=np.uint16)).save(path)
        return path.as_uri()

    def _row(self, source="source_1", site=1, **extra):
        result = {
            "Metadata_Source": source,
            "Metadata_Batch": "batch",
            "Metadata_Plate": "plate",
            "Metadata_Well": "A01",
            "Metadata_Site": site,
            "Selection_Note": "bounded-test",
        }
        for index, channel in enumerate(("AGP", "DNA", "ER", "Mito", "RNA")):
            result[f"URL_Orig{channel}"] = (
                None
                if source == "source_15" and channel == "RNA"
                else self._tiff(f"{source}-{site}-{channel}.tif", index + 1)
            )
        result.update(extra)
        return result

    def _manifest(self, rows, name="manifest.parquet"):
        path = self.root / name
        overrides = {
            f"URL_Orig{x}": pl.String for x in ("AGP", "DNA", "ER", "Mito", "RNA")
        }
        pl.DataFrame(rows, schema_overrides=overrides).write_parquet(path)
        return path

    def _config(self, rows=None, candidate="test-candidate"):
        manifest = self._manifest(rows or [self._row(), self._row("source_15", 2)])
        report = self.root / f"{candidate}-audit.json"
        audit = audit_inventory(manifest, report, kind="candidate")
        config = CandidateConfig(
            candidate,
            manifest,
            report,
            self.root / f"{candidate}-out",
            self.root / f"{candidate}-state",
            audit["inventory_digest"],
            audit["manifest"]["sha256"],
            audit["manifest"]["bytes"],
            test_mode=True,
        )
        return config, audit

    def _control(self, config, **changes):
        config.state_root.mkdir(parents=True, exist_ok=True)
        value = {
            "format_version": "full-jump-compression-control-v2",
            "candidate_id": config.candidate_id,
            "config_sha256": config.digest,
            "paused": False,
            "desired_workers": 4,
            "max_workers": 16,
            "consecutive_healthy_windows": 0,
            "compression_cpus": list(COMPRESSION_CPUS),
            "acknowledged_error_count": 0,
            "observed_at_unix": time.time(),
            "reasons": ["test"],
            "feature_processes_mutated": False,
        }
        value.update(changes)
        (config.state_root / "control.json").write_text(json.dumps(value))
        return value

    def test_audit_binds_all_columns_and_manifest_identity(self):
        config, audit = self._config()
        self.assertEqual(audit["site_count"], 2)
        self.assertEqual(audit["audit_kind"], "candidate")
        changed = [self._row(Selection_Note="different")]
        other = self._manifest(changed, "other.parquet")
        second = audit_inventory(other, kind="candidate")
        self.assertNotEqual(audit["full_row_sha256"], second["full_row_sha256"])
        self.assertEqual(len(audit["manifest"]["sha256"]), 64)
        # Replacement after audit is rejected even when its schema remains valid.
        self._manifest([self._row(site=9)], config.manifest.name)
        with self.assertRaises(RuntimeError):
            run_candidate(config, apply=False)

    def test_inventory_identity_is_portable_and_report_tamper_fails(self):
        config, audit = self._config(rows=[self._row()])
        relocated = self.root / "relocated.parquet"
        shutil.copyfile(config.manifest, relocated)
        loaded = load_audit(
            config.audit_report, relocated, audit["inventory_digest"], kind="candidate"
        )
        self.assertEqual(loaded["inventory_digest"], audit["inventory_digest"])
        second = audit_inventory(relocated, kind="candidate")
        self.assertEqual(second["inventory_digest"], audit["inventory_digest"])
        payload = json.loads(config.audit_report.read_text())
        payload["source_counts"]["source_1"] = 99
        self.assertNotEqual(
            inventory_digest_from_report(payload), payload["inventory_digest"]
        )
        config.audit_report.write_text(json.dumps(payload))
        with self.assertRaises(RuntimeError):
            load_audit(
                config.audit_report,
                config.manifest,
                audit["inventory_digest"],
                kind="candidate",
            )

    def test_audit_rejects_duplicate_ambiguous_missing_extra_and_malformed(self):
        row = self._row()
        for rows in (
            [row, dict(row)],
            [{**self._row(site=2), "URL_OrigER": None}],
            [{**self._row(site=3), "URL_OrigFoo": self._tiff("foo.tif", 1)}],
            [{**self._row(site=4), "Metadata_Well": "A__01"}],
        ):
            with self.assertRaises(RuntimeError):
                audit_inventory(self._manifest(rows), kind="candidate")
        ambiguous = dict(row)
        ambiguous["URL_OrigAGP"] = self._tiff("alternate.tif", 7)
        with self.assertRaises(RuntimeError):
            audit_inventory(self._manifest([row, ambiguous]), kind="candidate")

    def test_candidate_ceiling_blocks_raw_scale(self):
        rows = [self._row(site=index) for index in range(257)]
        with self.assertRaises(RuntimeError):
            audit_inventory(self._manifest(rows), kind="candidate")

    def test_dual_codec_receipts_thread_caps_and_complete_checkpoint_restart(self):
        config, _ = self._config()
        self._control(config)
        result = run_candidate(config, apply=True)
        self.assertEqual(result["created"], 2)
        self.assertTrue(all(os.environ[name] == "1" for name in THREAD_ENV))
        checkpoint = json.loads((config.state_root / "checkpoint.json").read_text())
        self.assertTrue(checkpoint["complete"])
        first = "source_1__batch__plate__A01__1"
        (config.output_root / "receipts" / f"{first}.json").unlink()
        rerun = run_candidate(config, apply=True)
        self.assertEqual(rerun["next_index"], 2)
        self.assertEqual(rerun["created"], 2)
        receipt = json.loads(
            (config.output_root / "receipts" / f"{first}.json").read_text()
        )
        self.assertEqual(receipt["shape"], [5, 12, 16])
        self.assertIn("package_source_sha256", receipt["software"])
        self.assertIn("libjxl 0.", receipt["software"]["libjxl"])
        for codec in CODECS:
            root = config.output_root / "codecs" / f"{codec}.zarr"
            self.assertFalse((root / ".zgroup").exists())
            metadata = json.loads((root / first / ".zarray").read_text())
            self.assertEqual(metadata["compressor"]["numthreads"], 1)

    def test_bootstrap_and_restart_after_explicit_error_acknowledgement(self):
        config, _ = self._config(rows=[self._row()])
        dry = bootstrap_candidate(config, False)
        self.assertEqual(dry["status"], "bootstrap-dry-run")
        boot = bootstrap_candidate(config, True)
        self.assertEqual(boot["status"], "bootstrapped-paused")
        control = json.loads((config.state_root / "control.json").read_text())
        telemetry = json.loads((config.state_root / "compression.json").read_text())
        self.assertTrue(control["paused"])
        self.assertTrue(control["governor_evaluation_required"])
        self.assertTrue(_control(config, cumulative_errors=0)["paused"])
        self.assertEqual(telemetry["cumulative_errors"], 0)
        with self.assertRaises(RuntimeError):
            bootstrap_candidate(config, True)

        self._control(config)
        with mock.patch(
            "jump_full_compression.pipeline.build_site",
            side_effect=RuntimeError("synthetic failure"),
        ):
            with self.assertRaises(RuntimeError):
                run_candidate(config, True)
        state = json.loads((config.state_root / "compression.json").read_text())
        self.assertEqual(state["cumulative_errors"], 1)
        STOP.clear()
        with mock.patch.object(STOP, "wait", side_effect=lambda _: STOP.set()):
            refused = run_candidate(config, True)
        self.assertEqual(refused["next_index"], 0)
        self.assertIn(
            "acknowledgement", _control(config, cumulative_errors=1)["reason"]
        )
        STOP.clear()
        with self.assertRaises(RuntimeError):
            acknowledge_errors(config, 2, True)
        with self.assertRaises(ValueError):
            acknowledge_errors(config, 1_000_001, True)
        preview = acknowledge_errors(config, 1, False)
        self.assertEqual(preview["status"], "acknowledgement-dry-run")
        acknowledged = acknowledge_errors(config, 1, True)
        self.assertEqual(acknowledged["acknowledged_error_count"], 1)
        paused = json.loads((config.state_root / "control.json").read_text())
        self.assertTrue(paused["paused"])
        self.assertTrue(paused["governor_evaluation_required"])
        self._control(config, acknowledged_error_count=1)
        completed = run_candidate(config, True)
        self.assertEqual(completed["status"], "complete")
        self.assertEqual(completed["cumulative_errors"], 1)

    def test_corrupt_chunk_with_complete_checkpoint_rolls_back(self):
        config, _ = self._config()
        self._control(config)
        run_candidate(config, True)
        site = "source_1__batch__plate__A01__1"
        chunk = config.output_root / "codecs/jpegxl_lossy_mq.zarr" / site / "0.0.0"
        chunk.write_bytes(b"corrupt")
        result = run_candidate(config, True)
        self.assertEqual(result["created"], 2)
        receipt = json.loads(
            (config.output_root / "receipts" / f"{site}.json").read_text()
        )
        self.assertEqual(
            sha256_file(chunk), receipt["outputs"]["jpegxl_lossy_mq"]["0.0.0"]["sha256"]
        )

    def test_adoption_audits_actual_frozen_identity_outputs_and_producer(self):
        config, _ = self._config()
        self._control(config)
        run_candidate(config, True)
        frozen_audit_path = self.root / "frozen.json"
        frozen = audit_inventory(config.manifest, frozen_audit_path, kind="frozen")
        result = validate_adoption_seam(
            config, config.manifest, frozen_audit_path, frozen["inventory_digest"]
        )
        self.assertEqual(result["validated_receipts"], 2)
        self.assertFalse(result["promotion_performed"])
        extra = config.output_root / "receipts" / "unexpected.json"
        extra.write_text("{}")
        with self.assertRaises(RuntimeError):
            validate_adoption_seam(
                config, config.manifest, frozen_audit_path, frozen["inventory_digest"]
            )
        extra.unlink()
        with self.assertRaises(RuntimeError):
            validate_adoption_seam(config, config.manifest, frozen_audit_path, "0" * 64)
        site = "source_1__batch__plate__A01__1"
        receipt_path = config.output_root / "receipts" / f"{site}.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["sources"][0]["etag"] = "0" * 32
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaises(RuntimeError):
            validate_adoption_seam(
                config, config.manifest, frozen_audit_path, frozen["inventory_digest"]
            )
        run_candidate(config, True)
        hq_root = config.output_root / "codecs/jpegxl_lossy_hq.zarr"
        extra_site = hq_root / "unexpected-site"
        extra_site.mkdir()
        with self.assertRaises(RuntimeError):
            validate_adoption_seam(
                config, config.manifest, frozen_audit_path, frozen["inventory_digest"]
            )
        extra_site.rmdir()
        shutil.rmtree(hq_root / site)
        with self.assertRaises(RuntimeError):
            validate_adoption_seam(
                config, config.manifest, frozen_audit_path, frozen["inventory_digest"]
            )
        run_candidate(config, True)
        (hq_root / site / ".zattrs").unlink()
        with self.assertRaises(Exception):
            validate_adoption_seam(
                config, config.manifest, frozen_audit_path, frozen["inventory_digest"]
            )

    def test_symlink_escape_and_live_literal_path_rejected(self):
        config, audit = self._config(rows=[self._row()])
        target = self.root / "real"
        target.mkdir()
        link = self.root / "linked"
        link.symlink_to(target, target_is_directory=True)
        bad = CandidateConfig(
            "test-link",
            config.manifest,
            config.audit_report,
            link,
            self.root / "state",
            audit["inventory_digest"],
            audit["manifest"]["sha256"],
            audit["manifest"]["bytes"],
            test_mode=True,
        )
        with self.assertRaises(ValueError):
            bad.validate()
        live = CandidateConfig(
            "live",
            config.manifest,
            config.audit_report,
            self.root / "jump_full",
            self.root / "state",
            audit["inventory_digest"],
            audit["manifest"]["sha256"],
            audit["manifest"]["bytes"],
        )
        with self.assertRaises(ValueError):
            live.validate()

    def test_codec_symlink_escape_rejected(self):
        config, _ = self._config(rows=[self._row()])
        self._control(config)
        outside = self.root / "outside"
        outside.mkdir()
        codec_parent = config.output_root / "codecs"
        codec_parent.mkdir(parents=True)
        (codec_parent / "jpegxl_lossy_hq.zarr").symlink_to(
            outside, target_is_directory=True
        )
        with self.assertRaises(ValueError):
            run_candidate(config, True)

    def test_internal_receipt_and_state_symlinks_rejected(self):
        config, _ = self._config(rows=[self._row()])
        self._control(config)
        config.output_root.mkdir()
        outside = self.root / "outside-internal"
        outside.mkdir()
        (config.output_root / "receipts").symlink_to(outside, target_is_directory=True)
        with self.assertRaises((ValueError, RuntimeError)):
            run_candidate(config, True)
        (config.output_root / "receipts").unlink()
        (config.state_root / "checkpoint.json").symlink_to(outside / "checkpoint")
        with self.assertRaises((ValueError, RuntimeError)):
            run_candidate(config, True)

    def test_graceful_stop_at_boundary(self):
        config, _ = self._config(rows=[self._row()])
        self._control(config)
        STOP.set()
        result = run_candidate(config, True)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["next_index"], 0)

    def test_control_missing_stale_and_drift_fail_closed(self):
        config, _ = self._config(rows=[self._row()])
        self.assertTrue(_control(config)["paused"])
        self._control(config, observed_at_unix=time.time() - 20000)
        self.assertTrue(_control(config)["paused"])
        self._control(config, compression_cpus=[1])
        self.assertTrue(_control(config)["paused"])
        self._control(config)
        self.assertFalse(_control(config)["paused"])

    def test_exclusive_controller_lock(self):
        config, _ = self._config(rows=[self._row()])
        self._control(config)
        config.state_root.mkdir(parents=True, exist_ok=True)
        with (config.state_root / "controller.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(RuntimeError):
                run_candidate(config, True)

    def test_s3_allowlist_truncation_and_retry(self):
        with self.assertRaises(ValueError):
            read_source("s3://evil/key", attempts=1)
        good = {"Body": io.BytesIO(b"abc"), "ContentLength": 3, "ETag": '"etag"'}
        client = mock.Mock()
        client.get_object.side_effect = [OSError("transient"), good]
        with mock.patch(
            "jump_full_compression.pipeline._s3_client", return_value=client
        ):
            payload, observed = read_source(
                "s3://cellpainting-gallery/key", attempts=2, sleep=lambda _: None
            )
        self.assertEqual(payload, b"abc")
        self.assertEqual(observed["etag"], "etag")
        truncated = {"Body": io.BytesIO(b"a"), "ContentLength": 2}
        with mock.patch(
            "jump_full_compression.pipeline._s3_client",
            return_value=mock.Mock(get_object=mock.Mock(return_value=truncated)),
        ):
            with self.assertRaises(OSError):
                read_source(
                    "s3://cellpainting-gallery/key", attempts=2, sleep=lambda _: None
                )

    def _metrics(self, processed=10, masks=50, profiles=20, errors=0):
        now = 10000.0
        states = {
            codec: {
                "heartbeat_unix": now - 10,
                "expected_sites": 100,
                "receipt_backed_masks": masks,
                "canonical_profiles": profiles,
                "worker_pids": [1],
            }
            for codec in ("MQ", "lossless")
        }
        return {
            "observed_at_unix": now,
            "candidate_id": "candidate",
            "config_sha256": "a" * 64,
            "feature_states": states,
            "profile_worker_affinities": {
                codec: {"1": list(range(288, 300))} for codec in states
            },
            "feature_services": [
                {
                    "name": "segment-MQ",
                    "active": "active",
                    "pid": 2,
                    "process_affinities": {"2": list(range(0, 32))},
                },
                {
                    "name": "segment-lossless",
                    "active": "active",
                    "pid": 3,
                    "process_affinities": {"3": list(range(32, 64))},
                },
            ],
            "compression": {
                "format_version": "full-jump-compression-state-v2",
                "candidate_id": "candidate",
                "config_sha256": "a" * 64,
                "heartbeat_unix": now - 10,
                "cumulative_errors": errors,
                "processed": processed,
                "state": "compressing",
            },
            "load1": 100,
            "memory_available_bytes": 256 * 1024**3,
            "storage_available_bytes": 10 * 1024**4,
            "io_pressure_avg10": 1,
        }

    def test_governor_multiwindow_independent_progress_and_error_ack(self):
        m1 = self._metrics()
        c1 = decide(m1)
        self.assertEqual(c1["desired_workers"], 4)
        m2 = self._metrics(processed=11, masks=51, profiles=21)
        c2 = decide(m2, c1, m1)
        self.assertEqual(c2["desired_workers"], 8)
        self.assertEqual(c2["consecutive_healthy_windows"], 0)
        m3 = self._metrics(processed=12, masks=52, profiles=22)
        c3 = decide(m3, c2, m2)
        m4 = self._metrics(processed=13, masks=53, profiles=23)
        c4 = decide(m4, c3, m3)
        self.assertEqual(c4["desired_workers"], 12)
        stalled = self._metrics(processed=14, masks=54, profiles=23)
        self.assertTrue(decide(stalled, c4, m4)["paused"])
        errored = self._metrics(errors=1)
        self.assertTrue(decide(errored)["paused"])
        acknowledged = {**c4, "acknowledged_error_count": 1}
        self.assertFalse(decide(errored, acknowledged)["paused"])

    def test_active_segmentation_without_mainpid_or_cgroup_fails_closed(self):
        metrics = self._metrics()
        metrics["feature_services"][0]["pid"] = 0
        metrics["feature_services"][0]["process_affinities"] = {}
        decision = decide(metrics)
        self.assertTrue(decision["paused"])
        self.assertTrue(
            any(
                "segmentation cgroup affinity" in reason
                for reason in decision["reasons"]
            )
        )

    def test_governor_dry_run_and_missing_telemetry_fail_closed(self):
        paths = GovernorPaths(
            "candidate",
            "a" * 64,
            self.root / "mq",
            self.root / "lossless",
            self.root / "compression",
            self.root / "control",
            self.root / "snapshots",
            self.root,
        )
        with mock.patch(
            "jump_full_compression.governor.collect_metrics",
            return_value=self._metrics(),
        ):
            self.assertTrue(run_governor(paths, True)["dry_run"])
        self.assertFalse(paths.control.exists())
        applied = run_governor(paths, False)
        self.assertTrue(applied["decision"]["paused"])
        self.assertTrue(paths.control.exists())

    def test_systemd_argument_timer_and_path_contract(self):
        base = Path("ops/systemd")
        compressor = (base / "jump-full-candidate-compress.service").read_text()
        governor = (base / "jump-full-compression-governor.service").read_text()
        timer = (base / "jump-full-compression-governor.timer").read_text()
        self.assertNotIn("CANDIDATE_ARGS", compressor)
        for field in (
            "${CANDIDATE_ID}",
            "${MANIFEST}",
            "${AUDIT_REPORT}",
            "${STATE_ROOT}",
            "${DEPLOY_ROOT}",
        ):
            self.assertIn(field, compressor)
        self.assertIn(
            "/work/users/amunoz/projects/JUMP_lite/.venv/bin/python",
            compressor + governor,
        )
        self.assertIn("OnCalendar=*-*-* 00/3:00:00", timer)
        self.assertIn("Persistent=true", timer)
        docs = Path("docs/full_jump_candidate_compression.md").read_text()
        for command in (
            'bootstrap "${ARGS[@]}" --apply',
            "governor --apply",
            'acknowledge-errors "${ARGS[@]}" --expected-count N --apply',
        ):
            self.assertIn(command, docs)
        for field in (
            "CONFIG_SHA256",
            "FEATURE_ROOT",
            "INVENTORY_DIGEST",
            "MANIFEST_SHA256",
        ):
            self.assertIn(field, docs)


if __name__ == "__main__":
    unittest.main()
