from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

CPG_UPLOAD = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CPG_UPLOAD))

import upload_profiles_to_staging as uploader  # noqa: E402


class ReleaseSiteManifestTests(unittest.TestCase):
    @staticmethod
    def _write_manifest(path: Path, keys: list[str]) -> None:
        pq.write_table(pa.table({"Metadata_Site_Key": keys}), path)

    def test_release_site_keys_accepts_exact_count_and_digest(self) -> None:
        keys = ["source__batch__plate__A01__1", "source__batch__plate__A01__2"]
        digest = hashlib.sha256("\n".join(sorted(keys)).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sites.parquet"
            self._write_manifest(path, keys)
            with (
                patch.object(uploader, "EXPECTED_SITE_COUNT", len(keys)),
                patch.object(uploader, "CANONICAL_DIGEST", digest),
            ):
                self.assertEqual(uploader.release_site_keys(path), set(keys))

    def test_release_site_keys_rejects_wrong_same_size_inventory(self) -> None:
        expected = ["source__batch__plate__A01__1", "source__batch__plate__A01__2"]
        observed = ["source__batch__plate__A01__1", "source__batch__plate__A01__3"]
        digest = hashlib.sha256("\n".join(sorted(expected)).encode()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sites.parquet"
            self._write_manifest(path, observed)
            with (
                patch.object(uploader, "EXPECTED_SITE_COUNT", len(expected)),
                patch.object(uploader, "CANONICAL_DIGEST", digest),
            ):
                with self.assertRaisesRegex(RuntimeError, "wrong canonical"):
                    uploader.release_site_keys(path)


if __name__ == "__main__":
    unittest.main()
