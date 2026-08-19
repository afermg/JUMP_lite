from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
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

    def test_atomic_refusal_cleanup_and_determinism(self):
        first = self._args("first")
        second = self._args("second")
        report1 = builder.build_manifest(first)
        report2 = builder.build_manifest(second)
        self.assertEqual(self._sha(first.output), self._sha(second.output))
        self.assertEqual(report1, report2)
        with self.assertRaises(FileExistsError):
            builder.build_manifest(first)
        bad = self._args("bad")
        receipt = json.loads(self.gray_receipt.read_text())
        receipt["entries"][0]["sha256"] = "f" * 64
        self.gray_receipt.write_text(json.dumps(receipt))
        with self.assertRaises(RuntimeError):
            builder.build_manifest(bad)
        self.assertFalse(bad.output.exists())
        self.assertFalse(bad.report.exists())
        self.assertEqual(list(self.root.glob(".*.building-*")), [])


if __name__ == "__main__":
    unittest.main()
