from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import duckdb
import numpy as np
import pytest
import zarr
from PIL import Image

REPO = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


compress = load_module("compress_tif_release", "src/compress_tif_release.py")
smoke = load_module("reproduce_dataset_sample", "scripts/reproduce_dataset_sample.py")


def write_site(root: Path, site: str, *, offset: int = 0) -> list[Path]:
    paths = []
    for index, channel in enumerate(compress.CHANNEL_ORDER):
        path = root / f"{site}__{channel}.tif"
        values = np.arange(30, dtype=np.uint16).reshape(5, 6) + offset + index
        Image.fromarray(values).save(path)
        paths.append(path)
    return paths


def test_group_tiffs_uses_site_not_channel_and_canonical_order(tmp_path: Path) -> None:
    first = "source_2__batch__plate__A01__1"
    second = "source_4__batch__plate__B02__2"
    paths = write_site(tmp_path, first) + write_site(tmp_path, second, offset=100)
    groups = compress.group_tiffs(reversed(paths))
    assert set(groups) == {
        tuple(first.split("__")),
        tuple(second.split("__")),
    }
    assert [
        path.stem.rsplit("__", 1)[1] for path in groups[tuple(first.split("__"))]
    ] == list(compress.CHANNEL_ORDER)


def test_group_tiffs_rejects_incomplete_sites(tmp_path: Path) -> None:
    paths = write_site(tmp_path, "source_2__batch__plate__A01__1")
    paths[-1].unlink()
    with pytest.raises(ValueError, match="invalid channel inventory"):
        compress.group_tiffs(paths[:-1])


def test_zstd_site_round_trip_and_release_site_key(tmp_path: Path) -> None:
    site = "source_2__batch__plate__A01__1"
    paths = write_site(tmp_path, site)
    groups = compress.group_tiffs(paths)
    output = tmp_path / "output"
    output.mkdir()
    result = compress.compress_tif(
        "zstd",
        compress.available_compressors()["zstd"],
        output,
        groups,
        n_jobs_inner=1,
    )
    assert result["success"] == 1
    assert result["errors"] == 0
    generated = output / "zstd.zarr" / site
    assert generated.is_dir()
    decoded = zarr.open_array(generated, mode="r")[:]
    np.testing.assert_array_equal(decoded, compress.read_stack(paths))
    assert decoded.dtype == np.uint16

    original = decoded.copy()
    rewritten = write_site(tmp_path, site, offset=500)
    refused = compress.compress_tif(
        "zstd",
        compress.available_compressors()["zstd"],
        output,
        compress.group_tiffs(rewritten),
        n_jobs_inner=1,
        skip_existing=False,
    )
    assert refused["success"] == 0
    assert refused["errors"] == 1
    np.testing.assert_array_equal(zarr.open_array(generated, mode="r")[:], original)


def test_failed_site_is_not_published_or_skipped_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = "source_2__batch__plate__A01__1"
    groups = compress.group_tiffs(write_site(tmp_path, site))
    output = tmp_path / "output"
    output.mkdir()
    real_create_array = compress.zarr.create_array

    class FailingArray:
        def __setitem__(self, _key, _value) -> None:
            raise OSError("simulated chunk-write failure")

    def fail_after_metadata(*args, **kwargs):
        real_create_array(*args, **kwargs)
        return FailingArray()

    monkeypatch.setattr(compress.zarr, "create_array", fail_after_metadata)
    failed = compress.compress_tif(
        "zstd",
        compress.available_compressors()["zstd"],
        output,
        groups,
        n_jobs_inner=1,
    )
    assert failed["errors"] == 1
    assert not (output / "zstd.zarr" / site).exists()

    monkeypatch.setattr(compress.zarr, "create_array", real_create_array)
    retried = compress.compress_tif(
        "zstd",
        compress.available_compressors()["zstd"],
        output,
        groups,
        n_jobs_inner=1,
    )
    assert retried["success"] == 1
    assert retried["skipped"] == 0
    compress.validate_site_array(output / "zstd.zarr" / site)


def write_fake_site_index(path: Path) -> tuple[dict[str, int], str]:
    rows = []
    for source, count in (("source_2", 3), ("source_4", 2)):
        for index in range(count):
            site = f"{source}__batch__plate__A01__{index}"
            urls = [
                f"s3://cellpainting-gallery/bucket/{site}-{channel}.tif"
                for channel in smoke.CHANNEL_ORDER
            ]
            rows.append((site, source, "batch", "plate", "A01", index, *urls))
    connection = duckdb.connect()
    connection.execute(
        """
        CREATE TABLE sites (
          Metadata_Site_Key VARCHAR,
          Metadata_Source VARCHAR,
          Metadata_Batch VARCHAR,
          Metadata_Plate VARCHAR,
          Metadata_Well VARCHAR,
          Metadata_Site BIGINT,
          URL_OrigAGP VARCHAR,
          URL_OrigDNA VARCHAR,
          URL_OrigER VARCHAR,
          URL_OrigMito VARCHAR,
          URL_OrigRNA VARCHAR
        )
        """
    )
    connection.executemany(
        "INSERT INTO sites VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
    )
    connection.execute(f"COPY sites TO '{path}' (FORMAT PARQUET)")
    keys = sorted(row[0] for row in rows)
    digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    return {"source_2": 3, "source_4": 2}, digest


def test_site_index_validation_selects_middle_site_per_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = tmp_path / "site_index.parquet"
    counts, digest = write_fake_site_index(index)
    monkeypatch.setattr(smoke, "EXPECTED_SITE_COUNT", 5)
    monkeypatch.setattr(smoke, "EXPECTED_SITE_INDEX_BYTES", index.stat().st_size)
    monkeypatch.setattr(smoke, "EXPECTED_SITE_INDEX_SHA256", smoke.sha256_file(index))
    monkeypatch.setattr(smoke, "EXPECTED_SOURCE_COUNTS", counts)
    monkeypatch.setattr(smoke, "EXPECTED_SITE_DIGEST", digest)
    selected, observed_digest = smoke.validate_and_select(index)
    assert observed_digest == digest
    assert selected["source_2"]["Metadata_Site_Key"].endswith("__1")
    assert selected["source_4"]["Metadata_Site_Key"].endswith("__0")


def test_dataset_smoke_output_rejects_traversal_and_symlink_parent() -> None:
    with pytest.raises(ValueError, match="may not traverse"):
        smoke.safe_output_root(Path("data/generated/dataset-smoke/../escape"))

    root = REPO / "data/generated/dataset-smoke/test-symlink-contract"
    try:
        real = root / "real"
        real.mkdir(parents=True)
        linked = root / "linked"
        linked.symlink_to(real, target_is_directory=True)
        with pytest.raises(ValueError, match="symlinked output parent"):
            smoke.safe_output_root(linked / "run")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_release_d20_codec_is_exposed() -> None:
    if not compress.IMAGECODECS_AVAILABLE:
        pytest.skip("imagecodecs is unavailable")
    assert "jpegxl_lossy_d20" in compress.available_compressors()


def test_recorded_dataset_smoke_attestation_matches_implementation() -> None:
    path = REPO / "reproducibility/validation/dataset-smoke-20260901.json"
    attestation = json.loads(path.read_text())
    assert attestation["summary"] == {
        "generated_site_arrays": 24,
        "exact_attested_array_trees": 24,
        "lossless_roundtrips": 6,
        "canonical_arrays_compared": 24,
        "exact_canonical_array_trees": 24,
    }
    assert len(attestation["sampling"]["sites"]) == 6
    assert len(attestation["source_tiffs"]) == 30
    assert len(attestation["outputs"]) == 24
    assert all(row["exact_reference_match"] for row in attestation["outputs"])
    for relative in (
        "src/compress_tif_release.py",
        "scripts/reproduce_dataset_sample.py",
        "justfile",
        "uv.lock",
    ):
        expected = attestation["implementation"][f"{relative}_sha256"]
        assert smoke.sha256_file(REPO / relative) == expected
