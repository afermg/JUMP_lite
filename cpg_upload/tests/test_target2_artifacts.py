from __future__ import annotations

import sys
import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CPG_UPLOAD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CPG_UPLOAD))

import build_target2_artifacts as build  # noqa: E402
import upload_target2_artifacts_to_staging as upload  # noqa: E402


class Target2ArtifactTests(unittest.TestCase):
    def test_parse_site_key_and_destination(self) -> None:
        identity = build.parse_site_key(
            "source_4__2021_04_26_Batch1__BR00121438__A03__3"
        )
        self.assertEqual(identity.source, "source_4")
        self.assertEqual(identity.site, "3")
        expected = (
            "segmentation/objects/zstd/source_4/2021_04_26_Batch1/"
            "BR00121438/A03-3/cell_mask.npz"
        )
        self.assertEqual(build.mask_relative_key("zstd", identity, "cell"), expected)
        with self.assertRaisesRegex(ValueError, "object type"):
            build.mask_relative_key("zstd", identity, "cytoplasm")
        with self.assertRaisesRegex(ValueError, "malformed"):
            build.parse_site_key("source_4__plate__well")
        self.assertEqual(
            build.object_feature_relative_key("zstd", identity),
            "object_features/cp_measure/zstd/source_4/2021_04_26_Batch1/"
            "BR00121438/A03-3.parquet",
        )
        with self.assertRaisesRegex(ValueError, "non-numeric"):
            build.parse_site_key("source_4__batch__plate__A01__x")

    def test_profile_parsing_and_destinations(self) -> None:
        cp = build.parse_profile_filename(
            "cp_measure_jump_target2_4plate_jpegxl_lossy_hq_raw_features.parquet"
        )
        self.assertEqual(cp.family, "cp_measure")
        self.assertEqual(cp.expected_columns, 2561)
        self.assertEqual(
            build.profile_relative_key(cp),
            "profiles/cp_measure/jpegxl_lossy_hq/profiles.parquet",
        )
        dl = build.parse_profile_filename(
            "subcell__clip01_jump_target2_4plate_zstd_raw_features.parquet"
        )
        self.assertEqual(dl.model, "subcell__clip01")
        self.assertEqual(dl.expected_columns, 1545)
        self.assertEqual(
            build.profile_relative_key(dl),
            "profiles/deep_learning/subcell__clip01/zstd/profiles.parquet",
        )
        with self.assertRaisesRegex(ValueError, "unexpected"):
            build.parse_profile_filename(
                "cp_measure_jump_target2_4plate_not_a_codec_raw_features.parquet"
            )

    def _write_mask_pair(
        self,
        root: Path,
        include_nuclei: bool = True,
        *,
        codec: str = "zstd",
        site_key: str = "source_3__batch__plate__A01__1",
    ) -> int:
        site = root / f"{codec}.zarr/steps/{site_key}"
        array = np.zeros((1080, 1080), dtype=np.uint16)
        array[10:20, 10:20] = 1
        cell = site / "segment_cell/0000.npz"
        cell.parent.mkdir(parents=True)
        np.savez_compressed(cell, array)
        paths = [cell]
        if include_nuclei:
            nuclei = site / "segment_nuclei/0000.npz"
            nuclei.parent.mkdir(parents=True)
            np.savez_compressed(nuclei, array)
            paths.append(nuclei)
        return sum(path.stat().st_size for path in paths)

    def test_mask_inventory_is_paired_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            size = self._write_mask_pair(root)
            rows = build.build_mask_inventory(
                root,
                workers=2,
                expected_site_counts={"zstd": 1},
                expected_file_count=2,
                expected_total_bytes=size,
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["object_type"] for row in rows}, {"cell", "nuclei"})
            self.assertTrue(all(row["array_dtype"] == "uint16" for row in rows))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_mask_pair(root, include_nuclei=False)
            with self.assertRaisesRegex(RuntimeError, "missing paired nuclei mask"):
                build.build_mask_inventory(
                    root,
                    workers=1,
                    expected_site_counts={"zstd": 1},
                    expected_file_count=None,
                    expected_total_bytes=None,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_mask_pair(root)
            self._write_mask_pair(
                root,
                codec="jpegxl_lossy_hq",
                site_key="source_3__batch__plate__A02__1",
            )
            with self.assertRaisesRegex(RuntimeError, "non-Target-2 keys"):
                build.build_mask_inventory(
                    root,
                    workers=1,
                    expected_site_counts={"zstd": 1, "jpegxl_lossy_hq": 1},
                    expected_file_count=None,
                    expected_total_bytes=None,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_mask_pair(root)
            nuclei = (
                root / "zstd.zarr/steps/source_3__batch__plate__A01__1/"
                "segment_nuclei/0000.npz"
            )
            array = np.zeros((1, 1080, 1080), dtype=np.uint16)
            array[:, 10:20, 10:20] = 1
            np.savez_compressed(nuclei, array)
            with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
                build.build_mask_inventory(
                    root,
                    workers=1,
                    expected_site_counts={"zstd": 1},
                    expected_file_count=None,
                    expected_total_bytes=None,
                )

    def test_object_feature_record_preserves_empty_tables_and_schema(self) -> None:
        schema = pa.schema(
            [
                ("tile", pa.int64()),
                ("label", pa.int64()),
                ("branch", pa.string()),
                ("metric", pa.string()),
                ("value", pa.float64()),
                ("object", pa.string()),
                ("tp", pa.uint8()),
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            identity = build.SiteIdentity("source_3", "batch", "plate", "A01", "1")
            path = root / "nonempty.parquet"
            arrays = [
                pa.array([0] * 1_275, type=field.type)
                if pa.types.is_integer(field.type)
                else pa.array(
                    (["x"] * 1_275 if pa.types.is_string(field.type) else [1.0] * 1_275),
                    type=field.type,
                )
                for field in schema
            ]
            pq.write_table(pa.Table.from_arrays(arrays, schema=schema), path)
            digest = hashlib.sha256(
                pq.ParquetFile(path).schema_arrow.serialize().to_pybytes()
            ).hexdigest()
            with mock.patch.dict(
                build.EXPECTED_OBJECT_FEATURE_SCHEMA_SHA256_BY_EMPTY,
                {False: digest},
            ):
                row = build._object_feature_record(
                    (path, root, "zstd", "complete", identity)
                )
            self.assertFalse(row["empty"])
            self.assertEqual(row["row_count"], 1_275)

            empty_schema = schema.set(6, pa.field("tp", pa.int64()))
            empty_path = root / "empty.parquet"
            pq.write_table(
                pa.Table.from_arrays(
                    [pa.array([], type=field.type) for field in empty_schema],
                    schema=empty_schema,
                ),
                empty_path,
            )
            empty_digest = hashlib.sha256(
                pq.ParquetFile(empty_path).schema_arrow.serialize().to_pybytes()
            ).hexdigest()
            with mock.patch.dict(
                build.EXPECTED_OBJECT_FEATURE_SCHEMA_SHA256_BY_EMPTY,
                {True: empty_digest},
            ):
                empty_row = build._object_feature_record(
                    (empty_path, root, "zstd", "complete", identity)
                )
            self.assertTrue(empty_row["empty"])
            self.assertEqual(empty_row["row_count"], 0)

    def test_profile_record_freezes_arrow_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "profile.parquet"
            table = pa.table(
                {
                    "Metadata_id": ["source_3__batch__plate__A01"],
                    "feature": pa.array([1.0], type=pa.float32()),
                }
            )
            pq.write_table(table, path)
            schema_digest = hashlib.sha256(
                pq.ParquetFile(path).schema_arrow.serialize().to_pybytes()
            ).hexdigest()
            identity = build.ProfileIdentity(
                "cp_measure", "", "zstd", "profiles.parquet", 1, 2
            )
            with mock.patch.dict(
                build.EXPECTED_PROFILE_SCHEMA_SHA256,
                {("cp_measure", ""): schema_digest},
            ):
                row = build._profile_record((path, root, "compact_cp", identity))
            self.assertEqual(row["schema_sha256"], schema_digest)

            pq.write_table(
                pa.table(
                    {
                        "Metadata_id": ["source_3__batch__plate__A01"],
                        "feature": pa.array([1], type=pa.int64()),
                    }
                ),
                path,
            )
            with mock.patch.dict(
                build.EXPECTED_PROFILE_SCHEMA_SHA256,
                {("cp_measure", ""): schema_digest},
            ):
                with self.assertRaisesRegex(RuntimeError, "schema digest"):
                    build._profile_record((path, root, "compact_cp", identity))

    def test_source_hash_rejects_concurrent_stat_change(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.bin"
            path.write_bytes(b"before")
            initial = path.stat()

            def mutate(_path):
                path.write_bytes(b"after-change")
                return "a" * 64

            with mock.patch.object(build, "sha256_file", side_effect=mutate):
                with self.assertRaisesRegex(RuntimeError, "changed while"):
                    build.sha256_source_file(path, initial)

    def test_checkpoint_rejects_destination_or_inventory_drift(self) -> None:
        entry = upload.UploadEntry(
            source=Path("source"),
            relative_key="profiles/cp_measure/zstd/profiles.parquet",
            size_bytes=10,
            sha256="a" * 64,
            content_type="application/vnd.apache.parquet",
        )
        digest = upload.inventory_digest([entry])
        good = {
            "destination_prefix": build.DESTINATION_ROOT,
            "inventory_sha256": digest,
            "next_index": 1,
            "file_count": 1,
            "last_relative_key": entry.relative_key,
            "uploaded_bytes": 10,
        }
        self.assertEqual(upload.validate_checkpoint(good, [entry], digest), (1, 10))
        wrong_destination = dict(good, destination_prefix="wrong")
        with self.assertRaisesRegex(RuntimeError, "destination mismatch"):
            upload.validate_checkpoint(wrong_destination, [entry], digest)
        wrong_inventory = dict(good, inventory_sha256="b" * 64)
        with self.assertRaisesRegex(RuntimeError, "inventory digest mismatch"):
            upload.validate_checkpoint(wrong_inventory, [entry], digest)
        with self.assertRaisesRegex(RuntimeError, "file_count"):
            upload.validate_checkpoint(dict(good, file_count=2), [entry], digest)
        with self.assertRaisesRegex(RuntimeError, "uploaded_bytes"):
            upload.validate_checkpoint(dict(good, uploaded_bytes=9), [entry], digest)

    def test_adopt_existing_verified_data_rebinds_extended_checkpoint(self) -> None:
        existing = upload.UploadEntry(
            source=Path("existing"),
            relative_key="profiles/cp_measure/zstd/profiles.parquet",
            size_bytes=10,
            sha256="a" * 64,
            content_type="application/vnd.apache.parquet",
        )
        object_feature = upload.UploadEntry(
            source=Path("object"),
            relative_key=(
                "object_features/cp_measure/zstd/source_3/batch/plate/A01-1.parquet"
            ),
            size_bytes=20,
            sha256="b" * 64,
            content_type="application/vnd.apache.parquet",
        )

        class Manager:
            def client(self):
                return object()

        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.json"
            upload.atomic_write_json(
                checkpoint,
                {
                    "destination_prefix": build.DESTINATION_ROOT,
                    "inventory_sha256": upload.inventory_digest([existing]),
                    "complete": True,
                    "metadata_published": True,
                },
            )
            with mock.patch.object(upload, "remote_inventory", return_value={}), mock.patch.object(
                upload,
                "verify_remote_entries",
                return_value={"object_count": 1},
            ):
                upload.adopt_existing_verified_data(
                    manager=Manager(),
                    entries=[existing, object_feature],
                    checkpoint_path=checkpoint,
                    workers=1,
                )
            payload = json.loads(checkpoint.read_text())
            self.assertEqual(payload["next_index"], 1)
            self.assertEqual(payload["uploaded_bytes"], 10)
            self.assertFalse(payload["complete"])
            self.assertFalse(payload["metadata_published"])
            self.assertEqual(
                payload["inventory_sha256"],
                upload.inventory_digest([existing, object_feature]),
            )

    def test_metadata_upload_uses_verified_in_memory_bytes(self) -> None:
        class Client:
            def __init__(self):
                self.body = None

            def put_object(self, **kwargs):
                self.body = kwargs["Body"]

        class Manager:
            def __init__(self):
                self._client = Client()

            def client(self, **_kwargs):
                return self._client

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "metadata.json"
            source.write_bytes(b'{"ok": true}\n')
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            entry = upload.UploadEntry(
                source=source,
                relative_key="manifests/provenance.json",
                size_bytes=source.stat().st_size,
                sha256=digest,
                content_type="application/json",
            )
            manager = Manager()
            upload.upload_entry(manager, entry, in_memory=True)
            self.assertIsInstance(manager._client.body, bytes)
            self.assertEqual(manager._client.body, source.read_bytes())

            source.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "metadata source changed"):
                upload.upload_entry(manager, entry, in_memory=True)
            source.write_bytes(b'{"no": true}\n')
            self.assertEqual(source.stat().st_size, entry.size_bytes)
            with self.assertRaisesRegex(RuntimeError, "metadata source changed"):
                upload.upload_entry(manager, entry, in_memory=True)

    def test_metadata_preflight_rejects_unknown_root_objects(self) -> None:
        data = upload.UploadEntry(
            source=Path("data"),
            relative_key="profiles/cp_measure/zstd/profiles.parquet",
            size_bytes=10,
            sha256="a" * 64,
            content_type="application/vnd.apache.parquet",
        )
        metadata = upload.UploadEntry(
            source=Path("readme"),
            relative_key="README.md",
            size_bytes=20,
            sha256="b" * 64,
            content_type="text/markdown; charset=utf-8",
        )

        class Manager:
            def client(self):
                return object()

        old_readme = {metadata.destination_key: 1}
        with mock.patch.object(
            upload,
            "remote_inventory",
            return_value={data.destination_key: 10, **old_readme},
        ):
            upload.verify_root_ready_for_metadata(
                Manager(), data_entries=[data], metadata_entries=[metadata]
            )
        with mock.patch.object(
            upload,
            "remote_inventory",
            return_value={
                data.destination_key: 10,
                **old_readme,
                f"{build.DESTINATION_ROOT}/unexpected": 0,
            },
        ):
            with self.assertRaisesRegex(RuntimeError, "unexpected=1"):
                upload.verify_root_ready_for_metadata(
                    Manager(), data_entries=[data], metadata_entries=[metadata]
                )

    def test_remote_verification_checks_exact_keys_sizes_and_sha256(self) -> None:
        entry = upload.UploadEntry(
            source=Path("unused"),
            relative_key="profiles/cp_measure/zstd/profiles.parquet",
            size_bytes=10,
            sha256="a" * 64,
            content_type="application/vnd.apache.parquet",
        )
        checksum = base64.b64encode(bytes.fromhex(entry.sha256)).decode("ascii")

        class Paginator:
            def __init__(self, rows):
                self.rows = rows

            def paginate(self, **_kwargs):
                return [{"Contents": self.rows}]

        class Client:
            def __init__(self, rows, checksum_value=checksum):
                self.rows = rows
                self.checksum_value = checksum_value

            def get_paginator(self, _name):
                return Paginator(self.rows)

            def head_object(self, **_kwargs):
                return {"ChecksumSHA256": self.checksum_value}

        class Manager:
            def __init__(self, client):
                self._client = client

            def client(self, **_kwargs):
                return self._client

        expected_row = {"Key": entry.destination_key, "Size": entry.size_bytes}
        result = upload.verify_remote_entries(
            Manager(Client([expected_row])),
            entries=[entry],
            prefix=f"{build.DESTINATION_ROOT}/profiles/",
            workers=1,
        )
        self.assertEqual(result["sha256_checksums_verified"], 1)

        extra = {"Key": f"{build.DESTINATION_ROOT}/profiles/extra", "Size": 0}
        with self.assertRaisesRegex(RuntimeError, "inventory mismatch"):
            upload.verify_remote_entries(
                Manager(Client([expected_row, extra])),
                entries=[entry],
                prefix=f"{build.DESTINATION_ROOT}/profiles/",
                workers=1,
            )
        with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
            upload.verify_remote_entries(
                Manager(Client([expected_row], checksum_value="wrong")),
                entries=[entry],
                prefix=f"{build.DESTINATION_ROOT}/profiles/",
                workers=1,
            )

    def test_manifest_bytes_are_deterministic(self) -> None:
        rows = [
            {
                "artifact_type": "raw_feature_profile",
                "family": "cp_measure",
                "model": "",
                "codec": "zstd",
                "coverage_status": "complete",
                "source_group": "compact_cp",
                "source_relative_path": "input.parquet",
                "relative_key": "profiles/cp_measure/zstd/profiles.parquet",
                "size_bytes": 123,
                "mtime_ns": 456,
                "sha256": "a" * 64,
                "row_count": 1536,
                "column_count": 2561,
                "well_id_sha256": "b" * 64,
                "schema_sha256": "c" * 64,
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.parquet"
            second = root / "second.parquet"
            build.write_manifest(first, rows, build.PROFILE_MANIFEST_COLUMNS)
            build.write_manifest(second, rows, build.PROFILE_MANIFEST_COLUMNS)
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
