from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
from PIL import Image
import polars as pl
import pyarrow.parquet as pq
import zarr

from jump_full_compression import inventory as inventory_module
from jump_full_compression.governor import (
    GovernorPaths,
    collect_metrics,
    decide,
    run_governor,
)
from jump_full_compression.inventory import (
    audit_inventory,
    inventory_digest_from_report,
    load_audit,
)
from jump_full_compression.model import (
    CODECS,
    COMPRESSION_CPUS,
    CandidateConfig,
    LIVE_CANDIDATE_PARENT,
    LIVE_STATE_PARENT,
    THREAD_ENV,
)
from jump_full_compression.pipeline import (
    STOP,
    _control,
    acknowledge_errors,
    bootstrap_candidate,
    read_source,
    run_candidate,
    software_identity,
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

    def _frozen_policy_artifacts(
        self, *, resolved: bool, action="exclude_red_include_gray"
    ):
        metadata = Path(__file__).parents[1] / "metadata/full_jump_compression"
        policy = json.loads(
            (metadata / "production_exclusion_policy_v1.json").read_text()
        )
        policy["red_gray_release_policy"].update(
            action=action if resolved else None,
            status="resolved" if resolved else "unresolved",
            release_identity_blocked=not resolved,
        )
        policy_path = self.root / (
            "resolved-policy.json" if resolved else "policy.json"
        )
        policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n")
        return (
            policy_path,
            metadata / "known_damaged_objects_v1.json",
            metadata / "known_damaged_sites_v1.json",
            metadata / "qc_plate_classification_v1.json",
        )

    def _manifest_build_report(self, manifest, policy, objects, sites, qc_plates):
        def binding(path):
            content = path.read_bytes()
            return {
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }

        parquet = pq.ParquetFile(manifest)
        report = {
            "format_version": "full-jump-production-manifest-build-v1",
            "build_success": True,
            "release_identity_frozen": False,
            "policy_action": json.loads(policy.read_text())["red_gray_release_policy"][
                "action"
            ],
            "inputs": {
                "exclusion_policy": binding(policy),
                "damaged_objects": binding(objects),
                "damaged_sites": binding(sites),
                "qc_plates": binding(qc_plates),
            },
            "counts": {"final_rows": parquet.metadata.num_rows},
            "output": {
                **binding(manifest),
                "rows": parquet.metadata.num_rows,
                "columns": parquet.schema_arrow.names,
                "schema": str(parquet.schema_arrow),
                "strict_identity_order": True,
                "unique_identities": True,
            },
        }
        report["build_digest"] = hashlib.sha256(
            json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        path = self.root / "manifest-build-report.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return path

    def _frozen_audit(
        self,
        manifest,
        *,
        resolved=True,
        action="exclude_red_include_gray",
        **changes,
    ):
        policy, objects, sites, qc_plates = self._frozen_policy_artifacts(
            resolved=resolved, action=action
        )
        build_report = self._manifest_build_report(
            manifest, policy, objects, sites, qc_plates
        )
        arguments = {
            "kind": "frozen",
            "exclusion_policy": policy,
            "damaged_objects": objects,
            "damaged_sites": sites,
            "qc_plates": qc_plates,
            "build_report": build_report,
        }
        arguments.update(changes)
        return audit_inventory(manifest, self.root / "frozen-audit.json", **arguments)

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
        if value["acknowledged_error_count"] > 0:
            value["acknowledgement_source"] = "explicit-cli"
        (config.state_root / "control.json").write_text(json.dumps(value))
        return value

    def test_candidate_source_15_remains_allowed(self):
        manifest = self._manifest([self._row("source_15")])
        result = audit_inventory(manifest, kind="candidate")
        self.assertEqual(result["source_counts"], {"source_15": 1})
        self.assertNotIn("frozen_exclusion_policy", result)

    def test_frozen_audit_requires_policy_artifacts_and_resolved_qc(self):
        manifest = self._manifest([self._row()])
        with self.assertRaisesRegex(RuntimeError, "requires explicit"):
            audit_inventory(manifest, kind="frozen")
        with self.assertRaisesRegex(RuntimeError, "red/gray release policy unresolved"):
            self._frozen_audit(manifest, resolved=False)

    def test_frozen_audit_rejects_source_15_and_known_damaged_sites(self):
        source_15 = self._manifest([self._row("source_15")], "source15.parquet")
        with self.assertRaisesRegex(RuntimeError, "inventory audit failed"):
            self._frozen_audit(source_15)
        report_path = self.root / "frozen-audit.json"
        report = json.loads(report_path.read_text())
        self.assertEqual(report["frozen_exclusion_policy"]["source_15_rows_present"], 1)
        self.assertFalse(report["audit_success"])
        self.assertFalse(report["release_identity_frozen"])
        with self.assertRaisesRegex(RuntimeError, "successful freeze"):
            load_audit(
                report_path,
                source_15,
                report["inventory_digest"],
                kind="frozen",
            )

        damaged = self._row("source_7", 2)
        damaged.update(
            Metadata_Batch="20210727_Run3",
            Metadata_Plate="CP3-SC1-18",
            Metadata_Well="I22",
        )
        damaged_manifest = self._manifest([damaged], "damaged.parquet")
        with self.assertRaisesRegex(RuntimeError, "inventory audit failed"):
            self._frozen_audit(damaged_manifest)
        report = json.loads((self.root / "frozen-audit.json").read_text())
        self.assertEqual(
            report["frozen_exclusion_policy"]["known_damaged_site_rows_present"], 1
        )

    def test_frozen_audit_accepts_resolved_policy_and_healthy_five_channel_row(self):
        manifest = self._manifest([self._row("source_1")])
        result = self._frozen_audit(manifest)
        frozen = result["frozen_exclusion_policy"]
        self.assertTrue(result["release_identity_frozen"])
        self.assertEqual(frozen["source_15_rows_present"], 0)
        self.assertEqual(frozen["known_damaged_site_rows_present"], 0)
        self.assertEqual(frozen["red_gray_action"], "exclude_red_include_gray")
        self.assertEqual(frozen["red_plate_rows_present"], 0)
        self.assertEqual(frozen["gray_plate_rows_present"], 0)
        self.assertEqual(len(frozen["policy"]["sha256"]), 64)
        self.assertEqual(len(frozen["damaged_objects"]["sha256"]), 64)
        self.assertEqual(len(frozen["qc_plates"]["sha256"]), 64)

        report_path = self.root / "frozen-audit.json"
        legacy_frozen = json.loads(report_path.read_text())
        legacy_frozen.pop("audit_success")
        legacy_frozen["inventory_digest"] = inventory_digest_from_report(legacy_frozen)
        report_path.write_text(json.dumps(legacy_frozen, sort_keys=True))
        with self.assertRaisesRegex(RuntimeError, "successful freeze"):
            load_audit(
                report_path,
                manifest,
                legacy_frozen["inventory_digest"],
                kind="frozen",
            )

    def test_frozen_audit_requires_matching_builder_completion_marker(self):
        manifest = self._manifest([self._row()])
        policy, objects, sites, qc_plates = self._frozen_policy_artifacts(resolved=True)
        arguments = {
            "kind": "frozen",
            "exclusion_policy": policy,
            "damaged_objects": objects,
            "damaged_sites": sites,
            "qc_plates": qc_plates,
        }
        with self.assertRaisesRegex(RuntimeError, "manifest build report"):
            audit_inventory(manifest, **arguments)
        build_report = self._manifest_build_report(
            manifest, policy, objects, sites, qc_plates
        )
        payload = json.loads(build_report.read_text())
        payload["output"]["sha256"] = "0" * 64
        payload["build_digest"] = hashlib.sha256(
            json.dumps(
                {k: v for k, v in payload.items() if k != "build_digest"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        build_report.write_text(json.dumps(payload))
        with self.assertRaisesRegex(RuntimeError, "identity/completion drift"):
            audit_inventory(manifest, build_report=build_report, **arguments)
        build_report.unlink()
        with self.assertRaisesRegex(RuntimeError, "regular non-symlink"):
            audit_inventory(manifest, build_report=build_report, **arguments)

    def test_frozen_audit_manifest_replacement_cannot_mix_bytes_and_binding(self):
        manifest = self._manifest([self._row()], "replace-target.parquet")
        red = self._row("source_3", Metadata_Batch="CP59", Metadata_Plate="BR5867a3")
        replacement = self._manifest([red], "red-replacement.parquet")
        original_validate = inventory_module._validate_frozen_policy

        def replace_after_capture(*args, **kwargs):
            result = original_validate(*args, **kwargs)
            os.replace(replacement, manifest)
            return result

        with mock.patch(
            "jump_full_compression.inventory._validate_frozen_policy",
            side_effect=replace_after_capture,
        ):
            with self.assertRaisesRegex(RuntimeError, "inventory audit failed"):
                self._frozen_audit(manifest)
        report = json.loads((self.root / "frozen-audit.json").read_text())
        self.assertFalse(report["audit_success"])
        self.assertFalse(report["manifest_path_stable_through_audit"])
        self.assertEqual(report["frozen_exclusion_policy"]["red_plate_rows_present"], 0)
        with self.assertRaisesRegex(RuntimeError, "identity drift"):
            load_audit(
                self.root / "frozen-audit.json",
                manifest,
                report["inventory_digest"],
                kind="frozen",
            )

    def test_frozen_audit_enforces_pinned_red_gray_action(self):
        metadata = Path(__file__).parents[1] / "metadata/full_jump_compression"
        classification = json.loads(
            (metadata / "qc_plate_classification_v1.json").read_text()
        )
        red = next(
            item
            for item in classification["red_plates"]
            if item["source"] != "source_15"
        )
        red_manifest = self._manifest(
            [
                self._row(
                    red["source"],
                    Metadata_Batch=red["batch"],
                    Metadata_Plate=red["plate"],
                )
            ],
            "red.parquet",
        )
        with self.assertRaisesRegex(RuntimeError, "inventory audit failed"):
            self._frozen_audit(red_manifest)
        failed = json.loads((self.root / "frozen-audit.json").read_text())
        self.assertEqual(failed["frozen_exclusion_policy"]["red_plate_rows_present"], 1)

        gray = classification["gray_plates"][0]
        gray_manifest = self._manifest(
            [
                self._row(
                    gray["source"],
                    Metadata_Batch=gray["batch"],
                    Metadata_Plate=gray["plate"],
                )
            ],
            "gray.parquet",
        )
        included = self._frozen_audit(gray_manifest)
        self.assertTrue(included["audit_success"])
        self.assertEqual(
            included["frozen_exclusion_policy"]["gray_plate_rows_present"], 1
        )
        with self.assertRaisesRegex(RuntimeError, "inventory audit failed"):
            self._frozen_audit(gray_manifest, action="exclude_red_and_gray")
        failed = json.loads((self.root / "frozen-audit.json").read_text())
        self.assertEqual(
            failed["frozen_exclusion_policy"]["gray_plate_rows_present"], 1
        )

    def test_frozen_audit_rejects_coordinated_policy_and_ledger_drift(self):
        manifest = self._manifest([self._row()])
        policy, objects, sites, qc_plates = self._frozen_policy_artifacts(resolved=True)
        changed_objects = self.root / "changed-objects.json"
        object_payload = json.loads(objects.read_text())
        object_payload["scope"] = "coherently_changed_scope"
        object_payload["objects"][0]["evidence"][0] = "coherently changed evidence"
        changed_objects.write_text(
            json.dumps(object_payload, indent=2, sort_keys=True) + "\n"
        )

        changed_sites = self.root / "changed-sites.json"
        site_payload = json.loads(sites.read_text())
        site_payload["derived_from_sha256"] = sha256_file(changed_objects)
        changed_sites.write_text(
            json.dumps(site_payload, indent=2, sort_keys=True) + "\n"
        )

        changed_policy = self.root / "changed-policy.json"
        policy_payload = json.loads(policy.read_text())
        policy_payload["known_damaged_objects"]["object_ledger"] = {
            "path": "coherently/changed-objects.json",
            "bytes": changed_objects.stat().st_size,
            "sha256": sha256_file(changed_objects),
        }
        policy_payload["known_damaged_objects"]["site_ledger"] = {
            "path": "coherently/changed-sites.json",
            "bytes": changed_sites.stat().st_size,
            "sha256": sha256_file(changed_sites),
        }
        changed_policy.write_text(
            json.dumps(policy_payload, indent=2, sort_keys=True) + "\n"
        )
        with self.assertRaisesRegex(
            RuntimeError, "canonical damaged-ledger binding drift"
        ):
            audit_inventory(
                manifest,
                kind="frozen",
                exclusion_policy=changed_policy,
                damaged_objects=changed_objects,
                damaged_sites=changed_sites,
                qc_plates=qc_plates,
            )

        changed_qc = self.root / "changed-qc.json"
        qc_payload = json.loads(qc_plates.read_text())
        qc_payload["rules"]["gray"] = "coherently changed gray rule"
        changed_qc.write_text(json.dumps(qc_payload, indent=2, sort_keys=True) + "\n")
        qc_policy_payload = json.loads(policy.read_text())
        qc_policy_payload["red_gray_release_policy"]["classification_ledger"] = {
            "path": "coherently/changed-qc.json",
            "bytes": changed_qc.stat().st_size,
            "sha256": sha256_file(changed_qc),
        }
        changed_qc_policy = self.root / "changed-qc-policy.json"
        changed_qc_policy.write_text(
            json.dumps(qc_policy_payload, indent=2, sort_keys=True) + "\n"
        )
        with self.assertRaisesRegex(
            RuntimeError, "canonical QC plate-ledger binding drift"
        ):
            audit_inventory(
                manifest,
                kind="frozen",
                exclusion_policy=changed_qc_policy,
                damaged_objects=objects,
                damaged_sites=sites,
                qc_plates=changed_qc,
            )

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

    def test_legacy_v2_candidate_audit_loads_without_byte_or_digest_change(self):
        config, _ = self._config(rows=[self._row()])
        payload = json.loads(config.audit_report.read_text())
        self.assertEqual(payload["format_version"], "full-jump-inventory-audit-v2")
        payload.pop("audit_success")
        payload["inventory_digest"] = inventory_digest_from_report(payload)
        config.audit_report.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        before_bytes = config.audit_report.read_bytes()
        before_sha256 = sha256_file(config.audit_report)
        loaded = load_audit(
            config.audit_report,
            config.manifest,
            payload["inventory_digest"],
            kind="candidate",
        )
        self.assertEqual(loaded["inventory_digest"], payload["inventory_digest"])
        self.assertEqual(config.audit_report.read_bytes(), before_bytes)
        self.assertEqual(sha256_file(config.audit_report), before_sha256)

    def test_audit_rejects_physically_unsorted_manifest(self):
        manifest = self._manifest([self._row(site=2), self._row(site=1)])
        with self.assertRaisesRegex(RuntimeError, "physical canonical identity order"):
            audit_inventory(manifest, kind="candidate")

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
        config, _ = self._config(rows=[self._row(), self._row(site=2)])
        self._control(config)
        run_candidate(config, True)
        frozen_audit_path = self.root / "frozen.json"
        policy, objects, sites, qc_plates = self._frozen_policy_artifacts(resolved=True)
        build_report = self._manifest_build_report(
            config.manifest, policy, objects, sites, qc_plates
        )
        frozen = audit_inventory(
            config.manifest,
            frozen_audit_path,
            kind="frozen",
            exclusion_policy=policy,
            damaged_objects=objects,
            damaged_sites=sites,
            qc_plates=qc_plates,
            build_report=build_report,
        )

        def adopt(digest=frozen["inventory_digest"]):
            return validate_adoption_seam(
                config,
                config.manifest,
                frozen_audit_path,
                digest,
                policy,
                objects,
                sites,
                qc_plates,
                build_report,
            )

        result = adopt()
        self.assertEqual(result["validated_receipts"], 2)
        self.assertFalse(result["promotion_performed"])
        extra = config.output_root / "receipts" / "unexpected.json"
        extra.write_text("{}")
        with self.assertRaises(RuntimeError):
            adopt()
        extra.unlink()
        with self.assertRaises(RuntimeError):
            adopt("0" * 64)
        site = "source_1__batch__plate__A01__1"
        receipt_path = config.output_root / "receipts" / f"{site}.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["sources"][0]["etag"] = "0" * 32
        receipt_path.write_text(json.dumps(receipt))
        with self.assertRaises(RuntimeError):
            adopt()
        run_candidate(config, True)
        hq_root = config.output_root / "codecs/jpegxl_lossy_hq.zarr"
        extra_site = hq_root / "unexpected-site"
        extra_site.mkdir()
        with self.assertRaises(RuntimeError):
            adopt()
        extra_site.rmdir()
        shutil.rmtree(hq_root / site)
        with self.assertRaises(RuntimeError):
            adopt()
        run_candidate(config, True)
        (hq_root / site / ".zattrs").unlink()
        with self.assertRaises(Exception):
            adopt()

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

    def test_subprocess_audit_and_prepare_stay_within_task_ceiling(self):
        config, _ = self._config(rows=[self._row()])
        report = self.root / "subprocess-audit.json"
        script = r"""
import json, os, sys
from pathlib import Path
from jump_full_compression.cli import main
from jump_full_compression.model import CandidateConfig, runtime_task_count
from jump_full_compression.pipeline import run_candidate
manifest, report, output, state = map(Path, sys.argv[1:])
main(['audit', '--input', str(manifest), '--report', str(report), '--kind', 'candidate'])
audit = json.loads(report.read_text())
config = CandidateConfig('test-subprocess', manifest, report, output, state,
    audit['inventory_digest'], audit['manifest']['sha256'], audit['manifest']['bytes'],
    test_mode=True)
result = run_candidate(config, False)
print(json.dumps({'tasks': runtime_task_count(), 'status': result['status'],
    'duckdb_loaded': 'duckdb' in sys.modules}))
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path("src").absolute())
        for name in THREAD_ENV:
            environment.pop(name, None)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(config.manifest),
                str(report),
                str(self.root / "subprocess-output"),
                str(self.root / "subprocess-state"),
            ],
            cwd=Path.cwd(),
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["status"], "dry-run")
        self.assertLessEqual(payload["tasks"], 24)
        self.assertFalse(payload["duckdb_loaded"])

    def test_live_prepare_rejects_excess_preexisting_tasks(self):
        config, audit = self._config(rows=[self._row()])
        live = CandidateConfig(
            "live-task-gate",
            config.manifest,
            config.audit_report,
            LIVE_CANDIDATE_PARENT / "live-task-gate",
            LIVE_STATE_PARENT / "live-task-gate",
            audit["inventory_digest"],
            audit["manifest"]["sha256"],
            audit["manifest"]["bytes"],
        )
        with mock.patch(
            "jump_full_compression.pipeline.assert_runtime_task_ceiling",
            side_effect=RuntimeError("runtime task ceiling exceeded: observed=25"),
        ):
            with self.assertRaisesRegex(RuntimeError, "observed=25"):
                run_candidate(live, False)

    def test_software_identity_binds_interpreter_and_zarr_runtime_limits(self):
        identity = software_identity(require_clean=False)
        interpreter = Path(sys.executable).resolve(strict=True)
        self.assertEqual(identity["python_executable"], str(interpreter))
        self.assertEqual(identity["python_executable_sha256"], sha256_file(interpreter))
        self.assertEqual(
            identity["zarr_runtime_limits"],
            {"threading_max_workers": 4, "async_concurrency": 4},
        )
        with (
            zarr.config.set({"async.concurrency": 5}),
            self.assertRaisesRegex(RuntimeError, "Zarr runtime limits drifted"),
        ):
            software_identity(require_clean=False)

    def test_repeated_concurrent_zarr_io_stays_within_task_ceiling(self):
        script = r"""
import json
from pathlib import Path
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import zarr
from jump_full_compression.model import runtime_task_count
from jump_full_compression.pipeline import _write_staged, validate_site

root = Path(sys.argv[1])
samples = [runtime_task_count()]
stop = threading.Event()

def sample_tasks():
    while not stop.is_set():
        samples.append(runtime_task_count())
        time.sleep(0.001)

def write_and_read(index):
    stack = np.arange(5 * 8 * 9, dtype=np.uint16).reshape(5, 8, 9) + index
    staging = root / f"site-{index}"
    site = f"site-{index}"
    for codec in ("jpegxl_lossy_hq", "jpegxl_lossy_mq"):
        path = _write_staged(stack, staging, site, codec)
        validate_site(path, stack.shape, codec)
    return runtime_task_count()

sampler = threading.Thread(target=sample_tasks, daemon=True)
sampler.start()
with ThreadPoolExecutor(max_workers=4) as pool:
    for batch in range(6):
        samples.extend(pool.map(write_and_read, range(batch * 4, batch * 4 + 4)))
        samples.append(runtime_task_count())
stop.set()
sampler.join()
print(json.dumps({
    "max_tasks": max(samples),
    "samples": len(samples),
    "zarr_runtime_limits": {
        "threading_max_workers": zarr.config.get("threading.max_workers"),
        "async_concurrency": zarr.config.get("async.concurrency"),
    },
}))
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path("src").absolute())
        for name in THREAD_ENV:
            environment[name] = "1"
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.root / "zarr-task-regression")],
            cwd=Path.cwd(),
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertGreater(payload["samples"], 24)
        self.assertLessEqual(payload["max_tasks"], 24)
        self.assertEqual(
            payload["zarr_runtime_limits"],
            {"threading_max_workers": 4, "async_concurrency": 4},
        )

    def test_software_identity_rejects_non_file_interpreter(self):
        with (
            mock.patch(
                "jump_full_compression.pipeline.sys.executable", str(Path.cwd())
            ),
            self.assertRaisesRegex(RuntimeError, "regular file"),
        ):
            software_identity(require_clean=False)

    def test_live_clean_producer_rejects_untracked_importable_source(self):
        shadow = Path("src/boto3.py")
        self.assertFalse(shadow.exists())
        shadow.write_text("raise RuntimeError('must never be importable')\n")
        calls = []

        def git_output(command, text=True):
            calls.append(command)
            if "rev-parse" in command:
                return "a" * 40 + "\n"
            if "status" in command:
                return "?? src/boto3.py\n"
            if "ls-files" in command:
                return "\n".join(
                    str(path) for path in Path("src/jump_full_compression").glob("*.py")
                )
            raise AssertionError(command)

        try:
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "GIT_EXECUTABLE": str(
                            Path(shutil.which("git") or "git").resolve()
                        )
                    },
                ),
                mock.patch(
                    "jump_full_compression.pipeline.subprocess.check_output",
                    side_effect=git_output,
                ),
            ):
                with self.assertRaises(RuntimeError):
                    software_identity(require_clean=True)
        finally:
            shadow.unlink(missing_ok=True)
        status_call = next(command for command in calls if "status" in command)
        self.assertIn("--untracked-files=all", status_call)
        self.assertFalse(shadow.exists())

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
            "authoritative_progress": {
                codec: {
                    "receipt_backed_masks": masks,
                    "canonical_profiles": profiles,
                }
                for codec in states
            },
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

    def _governor_control(self, metrics, acknowledged=0):
        return {
            "format_version": "full-jump-compression-control-v2",
            "candidate_id": metrics["candidate_id"],
            "config_sha256": metrics["config_sha256"],
            "paused": True,
            "desired_workers": 4,
            "max_workers": 16,
            "consecutive_healthy_windows": 0,
            "compression_cpus": list(COMPRESSION_CPUS),
            "acknowledged_error_count": acknowledged,
            "acknowledgement_source": "explicit-cli" if acknowledged else "zero",
            "observed_at_unix": metrics["observed_at_unix"],
            "feature_processes_mutated": False,
        }

    def test_governor_stale_authenticated_idle_accepted_but_running_rejected(self):
        idle = self._metrics()
        idle["compression"].update(
            state="session-complete",
            heartbeat_unix=idle["observed_at_unix"] - 3600,
            processed=256,
            next_index=256,
            sites=1000,
        )
        accepted = decide(idle, self._governor_control(idle))
        self.assertFalse(accepted["paused"])
        running = json.loads(json.dumps(idle))
        running["compression"]["state"] = "running"
        rejected = decide(running, self._governor_control(running))
        self.assertTrue(rejected["paused"])
        self.assertIn("compression heartbeat stale", rejected["reasons"])
        incoherent = json.loads(json.dumps(idle))
        incoherent["compression"]["processed"] = 255
        rejected = decide(incoherent, self._governor_control(incoherent))
        self.assertTrue(rejected["paused"])
        self.assertIn("compression idle terminal state incoherent", rejected["reasons"])

    def test_governor_multiwindow_independent_progress_and_error_ack(self):
        m1 = self._metrics()
        c1 = decide(m1, self._governor_control(m1))
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
        acknowledged = {
            **c4,
            "acknowledged_error_count": 1,
            "acknowledgement_source": "explicit-cli",
        }
        self.assertFalse(decide(errored, acknowledged)["paused"])

    def test_invalid_prior_controls_reset_acknowledgement_to_zero(self):
        metrics = self._metrics(errors=1)
        valid = {
            "format_version": "full-jump-compression-control-v2",
            "candidate_id": "candidate",
            "config_sha256": "a" * 64,
            "paused": True,
            "desired_workers": 4,
            "max_workers": 16,
            "consecutive_healthy_windows": 0,
            "compression_cpus": list(COMPRESSION_CPUS),
            "acknowledged_error_count": 1,
            "acknowledgement_source": "explicit-cli",
            "observed_at_unix": metrics["observed_at_unix"],
            "feature_processes_mutated": False,
        }
        self.assertFalse(decide(metrics, valid)["paused"])
        invalid = (
            {**valid, "candidate_id": "foreign"},
            {**valid, "desired_workers": "malformed"},
            {**valid, "observed_at_unix": metrics["observed_at_unix"] - 20_000},
            {**valid, "acknowledged_error_count": 2},
            {**valid, "observed_at_unix": metrics["observed_at_unix"] + 1},
            {**valid, "acknowledgement_source": "forged"},
        )
        for prior in invalid:
            decision = decide(metrics, prior)
            self.assertTrue(decision["paused"])
            self.assertEqual(decision["acknowledged_error_count"], 0)

    def test_foreign_control_cannot_bypass_acknowledgement_in_two_windows(self):
        first_metrics = self._metrics(errors=1)
        foreign = {
            "format_version": "full-jump-compression-control-v2",
            "candidate_id": "foreign",
            "config_sha256": "b" * 64,
            "paused": False,
            "desired_workers": 16,
            "max_workers": 16,
            "consecutive_healthy_windows": 1,
            "compression_cpus": list(COMPRESSION_CPUS),
            "acknowledged_error_count": 1,
            "acknowledgement_source": "explicit-cli",
            "observed_at_unix": first_metrics["observed_at_unix"],
            "feature_processes_mutated": False,
        }
        first = decide(first_metrics, foreign)
        self.assertTrue(first["paused"])
        self.assertEqual(first["acknowledged_error_count"], 0)
        second_metrics = self._metrics(processed=11, masks=51, profiles=21, errors=1)
        second = decide(second_metrics, first, first_metrics)
        self.assertTrue(second["paused"])
        self.assertEqual(second["acknowledged_error_count"], 0)

    def test_authoritative_progress_overrides_frozen_state_and_marks_completion(self):
        old = self._metrics(masks=40, profiles=20)
        current = self._metrics(processed=11, masks=40, profiles=20)
        current["authoritative_progress"] = {
            codec: {"receipt_backed_masks": 41, "canonical_profiles": 21}
            for codec in ("MQ", "lossless")
        }
        decision = decide(current, decide(old, self._governor_control(old)), old)
        self.assertFalse(decision["paused"])
        complete = self._metrics(masks=40, profiles=20)
        complete["authoritative_progress"] = {
            codec: {"receipt_backed_masks": 100, "canonical_profiles": 100}
            for codec in ("MQ", "lossless")
        }
        for service in complete["feature_services"]:
            service.update(active="inactive", pid=0, process_affinities={})
        self.assertFalse(decide(complete, self._governor_control(complete))["paused"])

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
            self.root / "candidate",
            self.root / "mq",
            self.root / "lossless",
            self.root / "mq-masks",
            self.root / "lossless-masks",
            self.root / "mq-profiles",
            self.root / "lossless-profiles",
            self.root,
            True,
        )
        with mock.patch(
            "jump_full_compression.governor.collect_metrics",
            return_value=self._metrics(),
        ):
            self.assertTrue(run_governor(paths, True)["dry_run"])
        self.assertFalse(paths.control.exists())
        paths.state_root.mkdir()
        paths.control.write_text(
            json.dumps({"acknowledged_error_count": 999, "candidate_id": "foreign"})
        )
        applied = run_governor(paths, False)
        self.assertTrue(applied["decision"]["paused"])
        self.assertEqual(applied["decision"]["acknowledged_error_count"], 0)
        self.assertTrue(paths.control.exists())

    def test_collect_metrics_scans_authoritative_progress_roots(self):
        state_root = self.root / "candidate"
        state_root.mkdir()
        canonical = self.root / "canonical"
        control = self.root / "feature-control"
        control.mkdir()
        now_unix = time.time()
        for codec, masks, profiles in (("MQ", 2, 1), ("lossless", 3, 2)):
            (canonical / f"masks/{codec}/receipts").mkdir(parents=True)
            (canonical / f"profiles/{codec}").mkdir(parents=True)
            for index in range(masks):
                (canonical / f"masks/{codec}/receipts/{index}.json").write_text("{}")
            for index in range(profiles):
                (canonical / f"profiles/{codec}/{index}.parquet").write_bytes(b"x")
            (control / f"{codec}_state.json").write_text(
                json.dumps(
                    {
                        "heartbeat_unix": now_unix,
                        "expected_sites": 3,
                        "receipt_backed_masks": 0,
                        "canonical_profiles": 0,
                        "completed_this_process": profiles,
                        "worker_pids": [os.getpid()],
                    }
                )
            )
        (state_root / "compression.json").write_text(
            json.dumps(
                {
                    "format_version": "full-jump-compression-state-v2",
                    "candidate_id": "candidate",
                    "config_sha256": "a" * 64,
                    "heartbeat_unix": now_unix,
                    "cumulative_errors": 0,
                    "processed": 0,
                    "state": "paused",
                }
            )
        )
        paths = GovernorPaths(
            "candidate",
            "a" * 64,
            state_root,
            control / "MQ_state.json",
            control / "lossless_state.json",
            canonical / "masks/MQ/receipts",
            canonical / "masks/lossless/receipts",
            canonical / "profiles/MQ",
            canonical / "profiles/lossless",
            self.root,
            True,
        )
        service = {
            "name": "segment",
            "active": "active",
            "pid": 1,
            "process_affinities": {"1": [1]},
        }
        with (
            mock.patch("jump_full_compression.governor._affinity", return_value=[288]),
            mock.patch("jump_full_compression.governor._service", return_value=service),
            mock.patch(
                "jump_full_compression.governor._memory_available",
                return_value=256 * 1024**3,
            ),
            mock.patch("jump_full_compression.governor._io_pressure", return_value=0),
        ):
            metrics = collect_metrics(paths, now_unix)
        self.assertEqual(
            metrics["authoritative_progress"]["MQ"]["receipt_backed_masks"], 2
        )
        self.assertEqual(
            metrics["authoritative_progress"]["lossless"]["canonical_profiles"], 2
        )
        self.assertEqual(metrics["feature_states"]["MQ"]["receipt_backed_masks"], 0)

    def test_governor_state_boundary_and_snapshot_symlinks_rejected(self):
        canonical = self.root / "canonical"
        paths = GovernorPaths(
            "candidate",
            "a" * 64,
            self.root / "candidate",
            self.root / "mq",
            self.root / "lossless",
            canonical / "masks/MQ/receipts",
            canonical / "masks/lossless/receipts",
            canonical / "profiles/MQ",
            canonical / "profiles/lossless",
            self.root,
            True,
        )
        outside = self.root / "outside-governor"
        outside.mkdir()
        paths.state_root.mkdir()
        paths.snapshots.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            run_governor(paths, True)
        paths.snapshots.unlink()
        redirected_parent = self.root / "redirected-parent"
        redirected_parent.symlink_to(outside, target_is_directory=True)
        redirected = GovernorPaths(
            "candidate",
            "a" * 64,
            redirected_parent / "candidate",
            self.root / "mq",
            self.root / "lossless",
            canonical / "masks/MQ/receipts",
            canonical / "masks/lossless/receipts",
            canonical / "profiles/MQ",
            canonical / "profiles/lossless",
            self.root,
            True,
        )
        with self.assertRaises(ValueError):
            redirected.validate()
        live_wrong = GovernorPaths(
            "candidate",
            "a" * 64,
            self.root / "candidate",
            self.root / "mq",
            self.root / "lossless",
            canonical / "masks/MQ/receipts",
            canonical / "masks/lossless/receipts",
            canonical / "profiles/MQ",
            canonical / "profiles/lossless",
        )
        with self.assertRaises(ValueError):
            live_wrong.validate()

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
        self.assertIn("${CANONICAL_ROOT}", governor)
        self.assertIn("ARROW_NUM_THREADS=1", compressor + governor)
        self.assertIn("POLARS_MAX_THREADS=1", compressor + governor)
        self.assertIn("TasksMax=256", compressor)
        package_text = "\n".join(
            path.read_text() for path in Path("src/jump_full_compression").glob("*.py")
        )
        self.assertNotIn("import duckdb", package_text)
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
            "CANONICAL_ROOT",
            "INVENTORY_DIGEST",
            "MANIFEST_SHA256",
        ):
            self.assertIn(field, docs)


if __name__ == "__main__":
    unittest.main()
