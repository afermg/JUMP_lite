from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq

from prep import build_full_jump_manifest as builder


class FullJumpManifestBuilderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.repo = Path(__file__).parents[1].resolve()
        metadata = self.repo / "metadata/full_jump_compression"
        self.policy = metadata / "production_exclusion_policy_v1.json"
        self.objects = metadata / "known_damaged_objects_v1.json"
        self.sites = metadata / "known_damaged_sites_v1.json"
        self.qc = metadata / "qc_plate_classification_v1.json"
        qc = json.loads(self.qc.read_text())
        self.red = {
            (item["source"], item["batch"], item["plate"]) for item in qc["red_plates"]
        }
        self.gray = [
            (item["source"], item["batch"], item["plate"]) for item in qc["gray_plates"]
        ]
        self.healthy = ("source_1", "batch", "plate")
        self.source15 = ("source_15", "batch", "plate15")
        self.damaged_plate = ("source_7", "20210727_Run3", "CP3-SC1-18")
        self.checkout = self._checkout()
        self.preliminary = self.root / "preliminary.parquet"
        self.rows = [
            self._row(*self.healthy, "A01", 1),
            self._row(*self.source15, "A01", 1, source15=True),
            self._row(*self.damaged_plate, "I22", 2),
            self._row(*self.damaged_plate, "I22", 3),
            self._row(*self.damaged_plate, "I22", 4),
        ]
        self._write_parquet(self.preliminary, self.rows)
        self.gray_receipt = self._gray_inputs()
        self.patch = mock.patch.multiple(
            builder,
            UPSTREAM_COMMIT=self.commit,
            PLATE_CSV_SHA256=self.plate_sha,
            UPSTREAM_PLATE_COUNT=len(self.red) + len(self.gray) + 3,
            PRELIMINARY_SHA256=self._sha(self.preliminary),
            PRELIMINARY_BYTES=self.preliminary.stat().st_size,
        )
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _row(
        self,
        source: str,
        batch: str,
        plate: str,
        well: str,
        site: int,
        *,
        source15: bool = False,
    ) -> dict[str, object]:
        row: dict[str, object] = {
            "Metadata_Source": source,
            "Metadata_Batch": batch,
            "Metadata_Plate": plate,
            "Metadata_Well": well,
            "Metadata_Site": site,
        }
        for channel in builder.CHANNELS:
            row[f"URL_Orig{channel}"] = (
                None
                if source15 and channel == "RNA"
                else f"s3://example/{source}/{batch}/{plate}/{well}/{site}/{channel}.tif"
            )
        return row

    def _write_parquet(self, path: Path, rows: list[dict[str, object]]) -> None:
        schema = pa.schema(
            [
                ("Metadata_Source", pa.string()),
                ("Metadata_Batch", pa.string()),
                ("Metadata_Plate", pa.string()),
                ("Metadata_Well", pa.string()),
                ("Metadata_Site", pa.int64()),
                *((f"URL_Orig{x}", pa.string()) for x in builder.CHANNELS),
            ]
        )
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)

    def _checkout(self) -> Path:
        checkout = self.root / "datasets"
        (checkout / "metadata").mkdir(parents=True)
        rows = [
            {
                "Metadata_Source": source,
                "Metadata_Batch": batch,
                "Metadata_Plate": plate,
                "Metadata_PlateType": "test",
            }
            for source, batch, plate in sorted(
                self.red
                | set(self.gray)
                | {self.healthy, self.source15, self.damaged_plate}
            )
        ]
        plate = checkout / "metadata/plate.csv.gz"
        with gzip.GzipFile(filename=str(plate), mode="wb", mtime=0) as raw:
            text = "Metadata_Source,Metadata_Batch,Metadata_Plate,Metadata_PlateType\n"
            text += "".join(
                f"{x['Metadata_Source']},{x['Metadata_Batch']},{x['Metadata_Plate']},test\n"
                for x in rows
            )
            raw.write(text.encode())
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        subprocess.run(
            ["git", "-C", str(checkout), "add", "metadata/plate.csv.gz"], check=True
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        self.commit = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        self.plate_sha = self._sha(plate)
        return checkout

    def _gray_inputs(self) -> Path:
        entries = []
        for source, batch, plate in self.gray:
            path = self.root / f"{batch}--{plate}.csv"
            row = self._row(source, batch, plate, "A01", 1)
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row) + ["Extra_Illum"])
                writer.writeheader()
                writer.writerow({**row, "Extra_Illum": "allowed non-URL column"})
            entries.append(
                {
                    "source": source,
                    "batch": batch,
                    "plate": plate,
                    "public_uri": (
                        "s3://cellpainting-gallery/cpg0016-jump/"
                        f"{source}/workspace/load_data_csv/{batch}/{plate}/"
                        "load_data_with_illum.csv"
                    ),
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": self._sha(path),
                }
            )
        receipt = self.root / "gray-receipt.json"
        receipt.write_text(
            json.dumps(
                {"format_version": builder.GRAY_RECEIPT_FORMAT, "entries": entries},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return receipt

    def _args(self, name: str = "manifest") -> argparse.Namespace:
        return argparse.Namespace(
            preliminary=self.preliminary,
            preliminary_sha256=self._sha(self.preliminary),
            preliminary_bytes=self.preliminary.stat().st_size,
            datasets_checkout=self.checkout,
            exclusion_policy=self.policy,
            damaged_objects=self.objects,
            damaged_sites=self.sites,
            qc_plates=self.qc,
            gray_receipt=self.gray_receipt,
            output=self.root / f"{name}.parquet",
            report=self.root / f"{name}.json",
            _test_memory_available_bytes=128 * 1024**3,
        )

    def test_build_excludes_source15_and_damaged_sites_and_includes_gray(self):
        args = self._args()
        report = builder.build_manifest(args)
        table = pq.read_table(args.output)
        keys = list(builder._row_batches(table))
        self.assertEqual(table.num_rows, 8)
        self.assertFalse(any(key[0] == "source_15" for key in keys))
        self.assertFalse(builder.DAMAGED_KEYS & set(keys))
        self.assertTrue(set(self.gray) <= {key[:3] for key in keys})
        self.assertEqual(report["counts"]["excluded"]["source_15_rows"], 1)
        self.assertEqual(report["counts"]["excluded"]["known_damaged_site_rows"], 2)
        self.assertEqual(
            report["memory_preflight"]["effective_available_source"],
            "injected-test-only",
        )
        self.assertGreaterEqual(
            report["memory_preflight"]["effective_available_bytes"],
            report["memory_preflight"]["required_headroom_bytes"],
        )
        self.assertFalse(report["release_identity_frozen"])
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(len(keys), len(set(keys)))

    def test_rejects_red_row_and_plate_universe_drift(self):
        red = next(iter(self.red))
        self._write_parquet(self.preliminary, self.rows + [self._row(*red, "A01", 1)])
        with (
            mock.patch.object(
                builder, "PRELIMINARY_SHA256", self._sha(self.preliminary)
            ),
            mock.patch.object(
                builder, "PRELIMINARY_BYTES", self.preliminary.stat().st_size
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "plate-universe drift"):
                builder.build_manifest(self._args("red"))
        self._write_parquet(self.preliminary, self.rows[1:])
        with (
            mock.patch.object(
                builder, "PRELIMINARY_SHA256", self._sha(self.preliminary)
            ),
            mock.patch.object(
                builder, "PRELIMINARY_BYTES", self.preliminary.stat().st_size
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "plate-universe drift"):
                builder.build_manifest(self._args("missing"))

    def test_rejects_gray_receipt_hash_and_identity_drift(self):
        receipt = json.loads(self.gray_receipt.read_text())
        receipt["entries"][0]["sha256"] = "0" * 64
        self.gray_receipt.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(RuntimeError, "binding drift"):
            builder.build_manifest(self._args("hash"))
        self.gray_receipt = self._gray_inputs()
        receipt = json.loads(self.gray_receipt.read_text())
        receipt["entries"][0]["plate"] = "wrong"
        self.gray_receipt.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(RuntimeError, "public URI drift|identities differ"):
            builder.build_manifest(self._args("identity"))

    def test_rejects_missing_channel_and_duplicate_identity(self):
        missing = [dict(row) for row in self.rows]
        missing[0]["URL_OrigRNA"] = None
        self._write_parquet(self.preliminary, missing)
        with (
            mock.patch.object(
                builder, "PRELIMINARY_SHA256", self._sha(self.preliminary)
            ),
            mock.patch.object(
                builder, "PRELIMINARY_BYTES", self.preliminary.stat().st_size
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "missing/empty"):
                builder.build_manifest(self._args("channel"))
        duplicate = self.rows + [dict(self.rows[0])]
        self._write_parquet(self.preliminary, duplicate)
        with (
            mock.patch.object(
                builder, "PRELIMINARY_SHA256", self._sha(self.preliminary)
            ),
            mock.patch.object(
                builder, "PRELIMINARY_BYTES", self.preliminary.stat().st_size
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                builder.build_manifest(self._args("duplicate"))

    def test_consumes_captured_bytes_despite_same_length_path_replacement(self):
        original_capture = builder.capture_regular_file
        original_sha = self._sha(self.preliminary)
        replaced = False

        def adversarial_capture(path, label):
            nonlocal replaced
            payload, observed = original_capture(path, label)
            if label == "preliminary inventory" and not replaced:
                replacement = path.with_suffix(".replacement")
                replacement.write_bytes(b"X" * len(payload))
                os.replace(replacement, path)
                replaced = True
            return payload, observed

        with mock.patch.object(
            builder, "capture_regular_file", side_effect=adversarial_capture
        ):
            report = builder.build_manifest(self._args("captured"))
        self.assertTrue(replaced)
        self.assertEqual(report["inputs"]["preliminary"]["sha256"], original_sha)
        self.assertNotEqual(self._sha(self.preliminary), original_sha)

    def test_gray_csv_parsing_uses_the_captured_bound_bytes(self):
        original_capture = builder.capture_regular_file
        first_entry = json.loads(self.gray_receipt.read_text())["entries"][0]
        target = Path(first_entry["path"])
        replaced = False

        def adversarial_capture(path, label):
            nonlocal replaced
            payload, observed = original_capture(path, label)
            if label == "staged gray CSV" and path == target and not replaced:
                replacement = path.with_suffix(".replacement")
                replacement.write_bytes(b"Y" * len(payload))
                os.replace(replacement, path)
                replaced = True
            return payload, observed

        with mock.patch.object(
            builder, "capture_regular_file", side_effect=adversarial_capture
        ):
            report = builder.build_manifest(self._args("gray-captured"))
        self.assertTrue(replaced)
        reported = next(
            item
            for item in report["inputs"]["gray_entries"]
            if (item["source"], item["batch"], item["plate"])
            == (first_entry["source"], first_entry["batch"], first_entry["plate"])
        )
        self.assertEqual(reported["sha256"], first_entry["sha256"])
        self.assertNotEqual(self._sha(target), first_entry["sha256"])

    def test_memory_preflight_fails_before_table_materialization(self):
        args = self._args("memory")
        args._test_memory_available_bytes = 1
        with mock.patch.object(builder.pq, "read_table") as read_table:
            with self.assertRaisesRegex(RuntimeError, "insufficient memory headroom"):
                builder.build_manifest(args)
        read_table.assert_not_called()
        self.assertFalse(args.output.exists())
        self.assertFalse(args.report.exists())

    def test_cgroup_memory_uses_most_restrictive_finite_ancestor(self):
        cgroup_root = self.root / "cgroup"
        leaf = cgroup_root / "user.slice" / "job.scope"
        leaf.mkdir(parents=True)
        proc = self.root / "self.cgroup"
        proc.write_text("0::/user.slice/job.scope\n")
        for path, maximum, current in (
            (cgroup_root, "max", "1"),
            (cgroup_root / "user.slice", "1000", "100"),
            (leaf, "5000", "200"),
        ):
            (path / "memory.max").write_text(maximum)
            (path / "memory.current").write_text(current)
        self.assertEqual(builder._cgroup_available_memory(proc, cgroup_root), 900)

    def test_first_pyarrow_import_sees_overridden_thread_environment(self):
        names = list(builder.THREAD_ENV)
        code = (
            """
import builtins, json, os
names = %r
original = builtins.__import__
def intercept(name, *args, **kwargs):
    if name == 'pyarrow':
        builtins.__import__ = original
        print(json.dumps({key: os.environ.get(key) for key in names}, sort_keys=True))
    return original(name, *args, **kwargs)
builtins.__import__ = intercept
import prep.build_full_jump_manifest
"""
            % names
        )
        environment = os.environ.copy()
        environment.update({name: "999" for name in names})
        environment["PYTHONPATH"] = f"{self.repo / 'src'}:{self.repo}"
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=self.repo,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        observed = json.loads(result.stdout.strip().splitlines()[0])
        self.assertEqual(observed, {name: "1" for name in names})

    def test_atomic_refusal_cleanup_and_determinism(self):
        first = self._args("first")
        second = self._args("second")
        report1 = builder.build_manifest(first)
        report2 = builder.build_manifest(second)
        self.assertEqual(self._sha(first.output), self._sha(second.output))
        self.assertEqual(report1, report2)
        with self.assertRaises(FileExistsError):
            builder.build_manifest(first)
        orphan = self._args("orphan")
        orphan.output.write_bytes(b"fail-closed orphan")
        with self.assertRaises(FileExistsError):
            builder.build_manifest(orphan)
        self.assertFalse(orphan.report.exists())
        bad = self._args("bad")
        receipt = json.loads(self.gray_receipt.read_text())
        receipt["entries"][0]["sha256"] = "f" * 64
        self.gray_receipt.write_text(json.dumps(receipt))
        with self.assertRaises(RuntimeError):
            builder.build_manifest(bad)
        self.assertFalse(bad.output.exists())
        self.assertFalse(bad.report.exists())
        self.gray_receipt = self._gray_inputs()
        publication = self._args("publication")
        real_link = os.link
        calls = 0

        def fail_report_link(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated report publication failure")
            return real_link(source, destination)

        with mock.patch.object(builder.os, "link", side_effect=fail_report_link):
            with self.assertRaisesRegex(OSError, "report publication failure"):
                builder.build_manifest(publication)
        self.assertFalse(publication.output.exists())
        self.assertFalse(publication.report.exists())
        self.assertEqual(list(self.root.glob(".*.building-*")), [])

    def test_foreign_report_race_removes_only_builder_output_inode(self):
        publication = self._args("foreign-publication")
        real_link = os.link
        calls = 0

        def foreign_report_then_fail(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                Path(destination).write_text('{"foreign": true}\n')
                raise FileExistsError("concurrent foreign report")
            return real_link(source, destination)

        with mock.patch.object(
            builder.os, "link", side_effect=foreign_report_then_fail
        ):
            with self.assertRaisesRegex(FileExistsError, "foreign report"):
                builder.build_manifest(publication)
        self.assertFalse(publication.output.exists())
        self.assertEqual(json.loads(publication.report.read_text()), {"foreign": True})
        self.assertEqual(list(self.root.glob(".*.building-*")), [])


if __name__ == "__main__":
    unittest.main()
