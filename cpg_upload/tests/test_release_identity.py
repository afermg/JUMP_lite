import tempfile
import unittest
from pathlib import Path

import polars as pl

from cpg_upload.build_cpg_metadata import canonical_key_table


IDENTITY_COLUMNS = [
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_Well",
]


class FrozenReleaseIdentityTests(unittest.TestCase):
    def write_inputs(self, root: Path, *, site_key: str, well: str = "A01") -> tuple[Path, Path, Path]:
        zarr = root / "canonical.zarr"
        zarr.mkdir()
        (zarr / site_key).mkdir()
        site_manifest = root / "sites.parquet"
        well_manifest = root / "wells.parquet"
        coordinates = {
            "Metadata_Source": "source_2",
            "Metadata_Batch": "batch",
            "Metadata_Plate": "plate",
            "Metadata_Well": "A01",
        }
        pl.DataFrame(
            {
                "Metadata_Site_Key": [site_key],
                **{column: [value] for column, value in coordinates.items()},
                "Metadata_Site": [1],
            }
        ).write_parquet(site_manifest)
        pl.DataFrame(
            {
                **{column: [value] for column, value in coordinates.items()},
                "Metadata_Well": [well],
            }
        ).select(IDENTITY_COLUMNS).write_parquet(well_manifest)
        return zarr, site_manifest, well_manifest

    def test_accepts_matching_site_and_well_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            key = "source_2__batch__plate__A01__1"
            zarr, sites, wells = self.write_inputs(root, site_key=key)
            table, keys = canonical_key_table(zarr, sites, wells)
            self.assertEqual(keys, [key])
            self.assertEqual(table.height, 1)

    def test_rejects_site_key_coordinate_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wrong_key = "source_2__batch__plate__B02__1"
            zarr, sites, wells = self.write_inputs(root, site_key=wrong_key)
            with self.assertRaisesRegex(RuntimeError, "disagree with coordinate"):
                canonical_key_table(zarr, sites, wells)

    def test_rejects_well_manifest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            key = "source_2__batch__plate__A01__1"
            zarr, sites, wells = self.write_inputs(root, site_key=key, well="B02")
            with self.assertRaisesRegex(RuntimeError, "does not exactly match"):
                canonical_key_table(zarr, sites, wells)


if __name__ == "__main__":
    unittest.main()
