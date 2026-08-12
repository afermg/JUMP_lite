#!/usr/bin/env python3
"""Build deterministic, source-stat-bound manifests for Target-2 artifacts.

This builder is read-only with respect to the canonical masks and profiles. It
validates the frozen inventories, hashes every published input, and writes only
small release metadata beneath ``--release-root``. It never regenerates masks
or features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.lib import format as npformat

DEFAULT_MASK_ROOT = Path(
    "/work/datasets/jump_lite/aliby_output/cp_measure/jump_target2_4plate"
)
DEFAULT_CP_PROFILE_ROOT = Path(
    "/work/datasets/JUMP-lite-wacv/raw_features/jump_target2_4plate"
)
DEFAULT_DL_PROFILE_ROOT = Path(
    "/work/datasets/JUMP-lite-wacv/raw_features/jump_target2_4plate_cl_3"
)
DEFAULT_RELEASE_ROOT = Path("/work/datasets/jump_lite/cpg_release/target_2/v1.0")

DESTINATION_ROOT = (
    "cpg0016-jump/source_all/workspace/publication_data/2026_jump_lite/target_2/v1.0"
)

EXPECTED_MASK_SITE_COUNTS = {
    "jpegxl_lossy_d10": 9_216,
    "jpegxl_lossy_d15": 4_171,
    "jpegxl_lossy_d20_e2": 9_214,
    "jpegxl_lossy_d2_e8": 3_998,
    "jpegxl_lossy_d30": 9_214,
    "jpegxl_lossy_effort_3": 9_216,
    "jpegxl_lossy_hq": 9_216,
    "jpegxl_lossy_lq": 9_216,
    "jpegxl_lossy_mq": 9_216,
    "zstd": 9_216,
}
EXPECTED_MASK_FILE_COUNT = 163_786
EXPECTED_MASK_BYTES = 4_303_626_240
EXPECTED_TARGET2_SITES = 9_216
EXPECTED_SPATIAL_SHAPE_BY_SOURCE = {
    "source_3": (1080, 1080),
    "source_4": (1080, 1080),
    "source_5": (998, 998),
    "source_6": (970, 970),
}
ALLOWED_MASK_NDIMS = {2, 3}

CP_PROFILE_CODECS = {
    "jpegxl_lossy_d10",
    "jpegxl_lossy_d15",
    "jpegxl_lossy_d2_e8",
    "jpegxl_lossy_d30",
    "jpegxl_lossy_effort_3",
    "jpegxl_lossy_hq",
    "jpegxl_lossy_lq",
    "jpegxl_lossy_mq",
    "zstd",
}
CELL_COUNT_CODECS = {
    "jpegxl_lossy_d10",
    "jpegxl_lossy_d2_e8",
    "jpegxl_lossy_effort_3",
    "jpegxl_lossy_hq",
    "jpegxl_lossy_lq",
    "jpegxl_lossy_mq",
    "zstd",
}
DL_MODELS = {
    "dinov2": 393,
    "dinov2_random": 393,
    "morphem": 1_929,
    "openphenom": 393,
    "subcell__clip01": 1_545,
}
DL_PROFILE_CODECS = {
    "jpegxl_lossy_d10",
    "jpegxl_lossy_d15",
    "jpegxl_lossy_d2_e8",
    "jpegxl_lossy_d30",
    "jpegxl_lossy_effort_3",
    "jpegxl_lossy_hq",
    "jpegxl_lossy_lq",
    "jpegxl_lossy_mq",
    "jpegxl_lossy_mq_new",
    "zstd",
}
EXPECTED_CP_ROOT_FILE_COUNT = 16
EXPECTED_CP_ROOT_BYTES = 236_729_382
EXPECTED_DL_ROOT_FILE_COUNT = 50
EXPECTED_DL_ROOT_BYTES = 269_861_585
EXPECTED_OBJECT_FEATURE_SITE_COUNTS = {
    "jpegxl_lossy_d10": 9_174,
    "jpegxl_lossy_d15": 4_133,
    "jpegxl_lossy_d20_e2": 9_196,
    "jpegxl_lossy_d2_e8": 9_215,
    "jpegxl_lossy_d30": 9_191,
    "jpegxl_lossy_effort_3": 9_216,
    "jpegxl_lossy_hq": 9_216,
    "jpegxl_lossy_lq": 9_216,
    "jpegxl_lossy_mq": 9_216,
    "zstd": 9_216,
}
EXPECTED_OBJECT_FEATURE_BYTES_BY_CODEC = {
    "jpegxl_lossy_d10": 26_560_780_307,
    "jpegxl_lossy_d15": 11_691_230_881,
    "jpegxl_lossy_d20_e2": 26_242_936_504,
    "jpegxl_lossy_d2_e8": 27_320_429_200,
    "jpegxl_lossy_d30": 25_236_780_643,
    "jpegxl_lossy_effort_3": 27_557_951_793,
    "jpegxl_lossy_hq": 27_430_957_592,
    "jpegxl_lossy_lq": 27_129_487_934,
    "jpegxl_lossy_mq": 27_257_647_879,
    "zstd": 27_561_869_897,
}
EXPECTED_OBJECT_FEATURE_FILE_COUNT = 86_989
EXPECTED_OBJECT_FEATURE_BYTES = 253_990_072_630
EXPECTED_OBJECT_FEATURE_EMPTY_COUNTS = {
    "jpegxl_lossy_d10": 0,
    "jpegxl_lossy_d15": 0,
    "jpegxl_lossy_d20_e2": 0,
    "jpegxl_lossy_d2_e8": 0,
    "jpegxl_lossy_d30": 0,
    "jpegxl_lossy_effort_3": 19,
    "jpegxl_lossy_hq": 11,
    "jpegxl_lossy_lq": 25,
    "jpegxl_lossy_mq": 3,
    "zstd": 12,
}
EXPECTED_OBJECT_FEATURE_ROWS_BY_CODEC = {
    "jpegxl_lossy_d10": 4_331_998_650,
    "jpegxl_lossy_d15": 1_903_910_325,
    "jpegxl_lossy_d20_e2": 4_298_054_325,
    "jpegxl_lossy_d2_e8": 4_465_585_500,
    "jpegxl_lossy_d30": 4_099_596_750,
    "jpegxl_lossy_effort_3": 4_516_095_900,
    "jpegxl_lossy_hq": 4_482_625_875,
    "jpegxl_lossy_lq": 4_433_069_175,
    "jpegxl_lossy_mq": 4_455_250_350,
    "zstd": 4_501_309_725,
}
EXPECTED_OBJECT_FEATURE_SITE_SET_SHA256 = {
    "jpegxl_lossy_d10": "bd9dab9e442ffb13c02b9eb89cc8a10cbaee65a616a26450bc64862552a551f6",
    "jpegxl_lossy_d15": "38c32dbd0179afcbec894daef175d943e4488313eb45a00df8203b3affd6e989",
    "jpegxl_lossy_d20_e2": "d009bf6d118497dc1a4c643b6a603fa0c558649fc873cdfd66cfe1c1a723ca59",
    "jpegxl_lossy_d2_e8": "9adbf603b5038a0e89be93f53717ed38784e3af7b96b684b9815d7169ab76a4c",
    "jpegxl_lossy_d30": "d21f82a182791d53166c587392efd83d2bb1d3511413719149443ef8962600ba",
    "jpegxl_lossy_effort_3": "15be2a339964f07dc32c0b908fdb0ca45940f60484a2cf2563cacdf2c1dc23e0",
    "jpegxl_lossy_hq": "15be2a339964f07dc32c0b908fdb0ca45940f60484a2cf2563cacdf2c1dc23e0",
    "jpegxl_lossy_lq": "15be2a339964f07dc32c0b908fdb0ca45940f60484a2cf2563cacdf2c1dc23e0",
    "jpegxl_lossy_mq": "15be2a339964f07dc32c0b908fdb0ca45940f60484a2cf2563cacdf2c1dc23e0",
    "zstd": "15be2a339964f07dc32c0b908fdb0ca45940f60484a2cf2563cacdf2c1dc23e0",
}
EXPECTED_OBJECT_FEATURE_SCHEMA_SHA256_BY_EMPTY = {
    False: "72276ec3f5490b3df061cda22377ccd0506639c172a540809eb1616f4a290b66",
    True: "29c3162e7b88284622b7661fc02261c969a6051a24e9175fab6041c40dc960c8",
}
EXPECTED_PROFILE_SCHEMA_SHA256 = {
    ("cp_measure", ""): "94834359687f329a426c03159d978d0fc67d38e4b60f153b7fb22dece99c376a",
    ("cell_count", ""): "d8efe5b08315a15603e44a49d9cea1c0c01d2ea93ce24db7024837553e7ffc9b",
    ("deep_learning", "dinov2"): "d7f183944b4697134aab90d7732ff8b53fa6e968e8402b7c4ba8aa2cb21c1561",
    ("deep_learning", "dinov2_random"): "15cc97e92b3e5a56754ced6cfadd950cf279591a7390cc68c1b6538d62a33870",
    ("deep_learning", "morphem"): "2e2962fe222c74a22c87801c4a3602b7c44d1f5489abe14da7edcaabcd5e601a",
    ("deep_learning", "openphenom"): "5fa52043b15d84128c07750e832588be6f38ba6aef5021bc6efc506e897ca387",
    ("deep_learning", "subcell__clip01"): "4266ccded944ce9f7c0d35c0e2b5752e97819f74d7c8f2ec4c2fcba095a6901b",
}
PROFILE_PRODUCER_PATHS = (
    "src/extract_features.py",
    "prep/aliby_featurize.py",
    "justfile",
    "pyproject.toml",
    "uv.lock",
)

MASK_MANIFEST_COLUMNS = (
    "artifact_type",
    "codec",
    "coverage_status",
    "source",
    "batch",
    "plate",
    "well",
    "site",
    "site_key",
    "object_type",
    "source_relative_path",
    "relative_key",
    "size_bytes",
    "mtime_ns",
    "sha256",
    "array_key",
    "array_dtype",
    "array_ndim",
    "array_height",
    "array_width",
)
OBJECT_FEATURE_MANIFEST_COLUMNS = (
    "artifact_type",
    "family",
    "codec",
    "coverage_status",
    "source",
    "batch",
    "plate",
    "well",
    "site",
    "site_key",
    "retained_mask_pair_present",
    "source_relative_path",
    "relative_key",
    "size_bytes",
    "mtime_ns",
    "sha256",
    "row_count",
    "column_count",
    "empty",
    "schema_sha256",
)
PROFILE_MANIFEST_COLUMNS = (
    "artifact_type",
    "family",
    "model",
    "codec",
    "coverage_status",
    "source_group",
    "source_relative_path",
    "relative_key",
    "size_bytes",
    "mtime_ns",
    "sha256",
    "row_count",
    "column_count",
    "well_id_sha256",
    "schema_sha256",
)


@dataclass(frozen=True)
class SiteIdentity:
    source: str
    batch: str
    plate: str
    well: str
    site: str

    @property
    def key(self) -> str:
        return "__".join(asdict(self).values())


@dataclass(frozen=True)
class ProfileIdentity:
    family: str
    model: str
    codec: str
    destination_name: str
    expected_rows: int
    expected_columns: int


def parse_site_key(value: str) -> SiteIdentity:
    fields = value.split("__")
    if len(fields) != 5 or any(not field for field in fields):
        raise ValueError(f"malformed Target-2 site key: {value!r}")
    source, batch, plate, well, site = fields
    if not source.startswith("source_"):
        raise ValueError(f"site key has invalid source: {value!r}")
    if not site.isdigit():
        raise ValueError(f"site key has non-numeric site: {value!r}")
    return SiteIdentity(source, batch, plate, well, site)


def parse_profile_filename(filename: str) -> ProfileIdentity:
    suffix = "_raw_features.parquet"
    if not filename.endswith(suffix):
        raise ValueError(f"not a canonical raw-feature Parquet: {filename}")
    stem = filename[: -len(suffix)]
    cp_prefix = "cp_measure_jump_target2_4plate_"
    count_prefix = "cell_count_jump_target2_4plate_"
    if stem.startswith(cp_prefix):
        codec = stem.removeprefix(cp_prefix)
        if codec not in CP_PROFILE_CODECS:
            raise ValueError(f"unexpected cp_measure codec in {filename}")
        rows = 1_490 if codec == "jpegxl_lossy_d15" else 1_536
        return ProfileIdentity("cp_measure", "", codec, "profiles.parquet", rows, 2_561)
    if stem.startswith(count_prefix):
        codec = stem.removeprefix(count_prefix)
        if codec not in CELL_COUNT_CODECS:
            raise ValueError(f"unexpected cell-count codec in {filename}")
        return ProfileIdentity(
            "cell_count", "", codec, "cell_counts.parquet", 1_536, 11
        )

    marker = "_jump_target2_4plate_"
    if marker not in stem:
        raise ValueError(f"malformed Target-2 DL profile filename: {filename}")
    model, codec = stem.split(marker, 1)
    if model not in DL_MODELS or codec not in DL_PROFILE_CODECS:
        raise ValueError(f"unexpected Target-2 DL model/codec in {filename}")
    return ProfileIdentity(
        "deep_learning", model, codec, "profiles.parquet", 1_536, DL_MODELS[model]
    )


def profile_relative_key(identity: ProfileIdentity) -> str:
    if identity.family == "deep_learning":
        return (
            f"profiles/deep_learning/{identity.model}/{identity.codec}/"
            f"{identity.destination_name}"
        )
    return f"profiles/{identity.family}/{identity.codec}/{identity.destination_name}"


def object_feature_relative_key(codec: str, identity: SiteIdentity) -> str:
    return (
        f"object_features/cp_measure/{codec}/{identity.source}/{identity.batch}/"
        f"{identity.plate}/{identity.well}-{identity.site}.parquet"
    )


def mask_relative_key(codec: str, identity: SiteIdentity, object_type: str) -> str:
    if object_type not in {"cell", "nuclei"}:
        raise ValueError(f"unexpected mask object type: {object_type!r}")
    return (
        f"segmentation/objects/{codec}/{identity.source}/{identity.batch}/"
        f"{identity.plate}/{identity.well}-{identity.site}/{object_type}_mask.npz"
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int, int, int]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def sha256_source_file(path: Path, initial_stat: os.stat_result) -> str:
    digest = sha256_file(path)
    final_stat = path.stat()
    if _stat_identity(final_stat) != _stat_identity(initial_stat):
        raise RuntimeError(f"source artifact changed while it was inventoried: {path}")
    return digest


def npz_header(path: Path) -> tuple[str, str, int, int, int]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if names != ["arr_0.npy"]:
            raise RuntimeError(
                f"{path} has NPZ members {names!r}; expected ['arr_0.npy']"
            )
        with archive.open("arr_0.npy") as stream:
            version = npformat.read_magic(stream)
            if version == (1, 0):
                shape, fortran_order, dtype = npformat.read_array_header_1_0(stream)
            elif version in {(2, 0), (3, 0)}:
                shape, fortran_order, dtype = npformat.read_array_header_2_0(stream)
            else:
                raise RuntimeError(f"{path} uses unsupported NPY version {version}")
    if fortran_order:
        raise RuntimeError(f"{path} is unexpectedly Fortran-ordered")
    if np.dtype(dtype) != np.dtype("uint16"):
        raise RuntimeError(f"{path} has dtype {dtype}; expected uint16")
    if len(shape) not in ALLOWED_MASK_NDIMS:
        raise RuntimeError(f"{path} has unsupported mask shape {shape}")
    if len(shape) == 3 and shape[0] != 1:
        raise RuntimeError(
            f"{path} has unsupported non-singleton leading axis: {shape}"
        )
    height, width = shape[-2:]
    return "arr_0", "uint16", len(shape), int(height), int(width)


def validate_instance_semantics(path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        array = archive["arr_0"]
    if array.dtype != np.uint16 or array.ndim not in ALLOWED_MASK_NDIMS:
        raise RuntimeError(f"representative mask changed schema: {path}")
    if array.ndim == 3 and array.shape[0] != 1:
        raise RuntimeError(f"representative mask has unsupported leading axis: {path}")
    if not np.any(array == 0):
        raise RuntimeError(f"representative mask has no background label 0: {path}")
    if int(array.max()) < 1:
        raise RuntimeError(
            f"representative mask has no positive instance labels: {path}"
        )


def _mask_record(
    task: tuple[Path, Path, str, str, SiteIdentity, str],
) -> dict[str, Any]:
    path, mask_root, codec, coverage, identity, object_type = task
    stat = path.stat()
    array_key, dtype, ndim, height, width = npz_header(path)
    expected_spatial_shape = EXPECTED_SPATIAL_SHAPE_BY_SOURCE.get(identity.source)
    if expected_spatial_shape is None:
        raise RuntimeError(f"unexpected Target-2 mask source: {identity.source}")
    if (height, width) != expected_spatial_shape:
        raise RuntimeError(
            f"{path} has spatial shape {(height, width)}; expected "
            f"{expected_spatial_shape} for {identity.source}"
        )
    relative_key = mask_relative_key(codec, identity, object_type)
    return {
        "artifact_type": "instance_mask",
        "codec": codec,
        "coverage_status": coverage,
        **asdict(identity),
        "site_key": identity.key,
        "object_type": object_type,
        "source_relative_path": path.relative_to(mask_root).as_posix(),
        "relative_key": relative_key,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_source_file(path, stat),
        "array_key": array_key,
        "array_dtype": dtype,
        "array_ndim": ndim,
        "array_height": height,
        "array_width": width,
    }


def build_mask_inventory(
    mask_root: Path,
    *,
    workers: int,
    expected_site_counts: Mapping[str, int] = EXPECTED_MASK_SITE_COUNTS,
    expected_file_count: int | None = EXPECTED_MASK_FILE_COUNT,
    expected_total_bytes: int | None = EXPECTED_MASK_BYTES,
) -> list[dict[str, Any]]:
    tasks: list[tuple[Path, Path, str, str, SiteIdentity, str]] = []
    seen_relative_keys: set[str] = set()
    site_keys_by_codec: dict[str, set[str]] = {}
    actual_codec_dirs = {
        path.name.removesuffix(".zarr") for path in mask_root.glob("*.zarr")
    }
    if actual_codec_dirs != set(expected_site_counts):
        raise RuntimeError(
            "mask codec inventory mismatch: "
            f"expected={sorted(expected_site_counts)}, actual={sorted(actual_codec_dirs)}"
        )

    for codec, expected_sites in sorted(expected_site_counts.items()):
        steps = mask_root / f"{codec}.zarr" / "steps"
        if not steps.is_dir():
            raise RuntimeError(f"missing mask steps directory: {steps}")
        site_dirs = sorted(path for path in steps.iterdir() if path.is_dir())
        if len(site_dirs) != expected_sites:
            raise RuntimeError(
                f"{codec} has {len(site_dirs):,} site directories; expected {expected_sites:,}"
            )
        coverage = "complete" if expected_sites == EXPECTED_TARGET2_SITES else "partial"
        representative: dict[str, Path] = {}
        codec_site_keys: set[str] = set()
        for site_dir in site_dirs:
            identity = parse_site_key(site_dir.name)
            if identity.key in codec_site_keys:
                raise RuntimeError(f"duplicate Target-2 site key for {codec}: {identity.key}")
            codec_site_keys.add(identity.key)
            for segment_name, object_type in (
                ("segment_cell", "cell"),
                ("segment_nuclei", "nuclei"),
            ):
                path = site_dir / segment_name / "0000.npz"
                if not path.is_file():
                    raise RuntimeError(f"missing paired {object_type} mask: {path}")
                extras = sorted((site_dir / segment_name).glob("*.npz"))
                if extras != [path]:
                    raise RuntimeError(
                        f"unexpected NPZ inventory under {path.parent}: {extras}"
                    )
                relative_key = mask_relative_key(codec, identity, object_type)
                if relative_key in seen_relative_keys:
                    raise RuntimeError(f"duplicate mask destination: {relative_key}")
                seen_relative_keys.add(relative_key)
                representative.setdefault(object_type, path)
                tasks.append((path, mask_root, codec, coverage, identity, object_type))
        site_keys_by_codec[codec] = codec_site_keys
        expected_paths = {task[0] for task in tasks if task[2] == codec}
        actual_npz_paths = set((mask_root / f"{codec}.zarr").rglob("*.npz"))
        if actual_npz_paths != expected_paths:
            raise RuntimeError(
                f"unexpected NPZ inventory for {codec}: "
                f"missing={len(expected_paths - actual_npz_paths)}, "
                f"extra={len(actual_npz_paths - expected_paths)}"
            )
        for path in representative.values():
            validate_instance_semantics(path)

    canonical_site_keys = site_keys_by_codec.get("zstd")
    canonical_expected_count = expected_site_counts.get("zstd")
    if (
        canonical_site_keys is None
        or canonical_expected_count is None
        or len(canonical_site_keys) != canonical_expected_count
    ):
        raise RuntimeError("Zstd does not define the expected canonical Target-2 site set")
    for codec, site_keys in sorted(site_keys_by_codec.items()):
        expected_sites = expected_site_counts[codec]
        if expected_sites == EXPECTED_TARGET2_SITES and site_keys != canonical_site_keys:
            raise RuntimeError(
                f"complete mask site set differs from Zstd for {codec}: "
                f"missing={len(canonical_site_keys - site_keys)}, "
                f"extra={len(site_keys - canonical_site_keys)}"
            )
        if not site_keys.issubset(canonical_site_keys):
            raise RuntimeError(
                f"mask site set for {codec} contains "
                f"{len(site_keys - canonical_site_keys)} non-Target-2 keys"
            )

    if workers < 1:
        raise ValueError("workers must be positive")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(_mask_record, tasks))
    records.sort(key=lambda row: row["relative_key"])
    paired_shapes: dict[tuple[str, str], dict[str, tuple[int, int, int]]] = {}
    for row in records:
        pair = paired_shapes.setdefault((row["codec"], row["site_key"]), {})
        pair[row["object_type"]] = (
            int(row["array_ndim"]),
            int(row["array_height"]),
            int(row["array_width"]),
        )
    mismatches = [
        key
        for key, shapes in paired_shapes.items()
        if shapes.get("cell") != shapes.get("nuclei")
    ]
    if mismatches:
        raise RuntimeError(
            f"cell/nuclei mask shape mismatch for {len(mismatches)} site-codec pairs; "
            f"first={mismatches[0]}"
        )
    total_bytes = sum(int(row["size_bytes"]) for row in records)
    if expected_file_count is not None and len(records) != expected_file_count:
        raise RuntimeError(
            f"mask inventory has {len(records):,} files; expected {expected_file_count:,}"
        )
    if expected_total_bytes is not None and total_bytes != expected_total_bytes:
        raise RuntimeError(
            f"mask inventory has {total_bytes:,} bytes; expected {expected_total_bytes:,}"
        )
    return records


def _object_feature_record(
    task: tuple[Path, Path, str, str, SiteIdentity],
) -> dict[str, Any]:
    path, root, codec, coverage, identity = task
    stat = path.stat()
    parquet = pq.ParquetFile(path)
    rows = parquet.metadata.num_rows
    columns = parquet.metadata.num_columns
    if columns != 7:
        raise RuntimeError(
            f"{path} has {columns} columns; expected a seven-column "
            "object-feature table"
        )
    empty = rows == 0
    if not empty and rows % 1_275:
        raise RuntimeError(
            f"{path} has {rows:,} rows; expected a multiple of 1,275"
        )
    schema_sha256 = hashlib.sha256(
        parquet.schema_arrow.serialize().to_pybytes()
    ).hexdigest()
    expected_schema_sha256 = EXPECTED_OBJECT_FEATURE_SCHEMA_SHA256_BY_EMPTY[empty]
    if schema_sha256 != expected_schema_sha256:
        raise RuntimeError(
            f"{path} has unexpected Arrow schema digest {schema_sha256}; "
            f"expected {expected_schema_sha256} for empty={empty}"
        )
    return {
        "artifact_type": "per_site_object_features",
        "family": "cp_measure",
        "codec": codec,
        "coverage_status": coverage,
        **asdict(identity),
        "site_key": identity.key,
        "source_relative_path": path.relative_to(root).as_posix(),
        "relative_key": object_feature_relative_key(codec, identity),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_source_file(path, stat),
        "row_count": rows,
        "column_count": columns,
        "empty": empty,
        "schema_sha256": schema_sha256,
    }


def build_object_feature_inventory(
    root: Path,
    *,
    workers: int,
    expected_site_counts: Mapping[str, int] = EXPECTED_OBJECT_FEATURE_SITE_COUNTS,
    expected_bytes_by_codec: Mapping[
        str, int
    ] = EXPECTED_OBJECT_FEATURE_BYTES_BY_CODEC,
    expected_file_count: int | None = EXPECTED_OBJECT_FEATURE_FILE_COUNT,
    expected_total_bytes: int | None = EXPECTED_OBJECT_FEATURE_BYTES,
) -> list[dict[str, Any]]:
    if set(expected_site_counts) != set(expected_bytes_by_codec):
        raise RuntimeError("object-feature count and byte expectations cover different codecs")
    actual_codec_dirs = {
        path.name.removesuffix(".zarr") for path in root.glob("*.zarr")
    }
    if actual_codec_dirs != set(expected_site_counts):
        raise RuntimeError(
            "object-feature codec inventory mismatch: "
            f"expected={sorted(expected_site_counts)}, actual={sorted(actual_codec_dirs)}"
        )

    tasks: list[tuple[Path, Path, str, str, SiteIdentity]] = []
    site_keys_by_codec: dict[str, set[str]] = {}
    seen_relative_keys: set[str] = set()
    for codec, expected_sites in sorted(expected_site_counts.items()):
        profiles = root / f"{codec}.zarr" / "profiles"
        if not profiles.is_dir():
            raise RuntimeError(f"missing per-site object-feature directory: {profiles}")
        children = sorted(profiles.iterdir())
        paths = [path for path in children if path.is_file() and path.suffix == ".parquet"]
        extras = [path for path in children if path not in paths]
        if extras:
            raise RuntimeError(
                f"unexpected per-site object-feature inventory under {profiles}: "
                f"{extras[:3]}"
            )
        if len(paths) != expected_sites:
            raise RuntimeError(
                f"{codec} has {len(paths):,} object-feature Parquets; "
                f"expected {expected_sites:,}"
            )
        total_bytes = sum(path.stat().st_size for path in paths)
        expected_codec_bytes = expected_bytes_by_codec[codec]
        if total_bytes != expected_codec_bytes:
            raise RuntimeError(
                f"{codec} object features have {total_bytes:,} bytes; "
                f"expected {expected_codec_bytes:,}"
            )
        coverage = "complete" if expected_sites == EXPECTED_TARGET2_SITES else "partial"
        site_keys: set[str] = set()
        for path in paths:
            identity = parse_site_key(path.stem)
            if identity.key in site_keys:
                raise RuntimeError(
                    f"duplicate per-site object-feature identity for {codec}: "
                    f"{identity.key}"
                )
            site_keys.add(identity.key)
            relative_key = object_feature_relative_key(codec, identity)
            if relative_key in seen_relative_keys:
                raise RuntimeError(f"duplicate object-feature destination: {relative_key}")
            seen_relative_keys.add(relative_key)
            tasks.append((path, root, codec, coverage, identity))
        site_set_sha256 = sha256_json(sorted(site_keys))
        expected_site_set_sha256 = EXPECTED_OBJECT_FEATURE_SITE_SET_SHA256[codec]
        if site_set_sha256 != expected_site_set_sha256:
            raise RuntimeError(
                f"{codec} object-feature site-set digest is {site_set_sha256}; "
                f"expected {expected_site_set_sha256}"
            )
        site_keys_by_codec[codec] = site_keys

    canonical = site_keys_by_codec.get("zstd")
    if canonical is None or len(canonical) != EXPECTED_TARGET2_SITES:
        raise RuntimeError(
            "Zstd object features do not define the complete Target-2 site set"
        )
    for codec, site_keys in sorted(site_keys_by_codec.items()):
        expected_sites = expected_site_counts[codec]
        if expected_sites == EXPECTED_TARGET2_SITES and site_keys != canonical:
            raise RuntimeError(
                f"complete object-feature site set differs from Zstd for {codec}: "
                f"missing={len(canonical - site_keys)}, extra={len(site_keys - canonical)}"
            )
        if not site_keys.issubset(canonical):
            raise RuntimeError(
                f"object-feature site set for {codec} contains "
                f"{len(site_keys - canonical)} non-Target-2 keys"
            )

    if workers < 1:
        raise ValueError("workers must be positive")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(_object_feature_record, tasks))
    records.sort(key=lambda row: row["relative_key"])
    row_counts_by_codec: dict[str, int] = {}
    empty_counts_by_codec: dict[str, int] = {}
    for row in records:
        codec = row["codec"]
        row_counts_by_codec[codec] = row_counts_by_codec.get(codec, 0) + int(
            row["row_count"]
        )
        empty_counts_by_codec[codec] = empty_counts_by_codec.get(codec, 0) + int(
            row["empty"]
        )
    if row_counts_by_codec != EXPECTED_OBJECT_FEATURE_ROWS_BY_CODEC:
        raise RuntimeError(
            "object-feature row totals changed: "
            f"expected={EXPECTED_OBJECT_FEATURE_ROWS_BY_CODEC}, "
            f"actual={row_counts_by_codec}"
        )
    if empty_counts_by_codec != EXPECTED_OBJECT_FEATURE_EMPTY_COUNTS:
        raise RuntimeError(
            "object-feature empty-file counts changed: "
            f"expected={EXPECTED_OBJECT_FEATURE_EMPTY_COUNTS}, "
            f"actual={empty_counts_by_codec}"
        )
    total_bytes = sum(int(row["size_bytes"]) for row in records)
    if expected_file_count is not None and len(records) != expected_file_count:
        raise RuntimeError(
            f"object-feature inventory has {len(records):,} files; "
            f"expected {expected_file_count:,}"
        )
    if expected_total_bytes is not None and total_bytes != expected_total_bytes:
        raise RuntimeError(
            f"object-feature inventory has {total_bytes:,} bytes; "
            f"expected {expected_total_bytes:,}"
        )
    return records


def _profile_record(task: tuple[Path, Path, str, ProfileIdentity]) -> dict[str, Any]:
    path, root, source_group, identity = task
    stat = path.stat()
    parquet = pq.ParquetFile(path)
    rows = parquet.metadata.num_rows
    columns = parquet.metadata.num_columns
    if (rows, columns) != (identity.expected_rows, identity.expected_columns):
        raise RuntimeError(
            f"{path} has ({rows:,} rows, {columns:,} columns); expected "
            f"({identity.expected_rows:,}, {identity.expected_columns:,})"
        )
    if not parquet.schema.names or parquet.schema.names[0] != "Metadata_id":
        raise RuntimeError(f"{path} does not start with the required Metadata_id column")
    schema_sha256 = hashlib.sha256(
        parquet.schema_arrow.serialize().to_pybytes()
    ).hexdigest()
    expected_schema_sha256 = EXPECTED_PROFILE_SCHEMA_SHA256.get(
        (identity.family, identity.model)
    )
    if expected_schema_sha256 is None or schema_sha256 != expected_schema_sha256:
        raise RuntimeError(
            f"{path} has unexpected Arrow schema digest {schema_sha256}; "
            f"expected {expected_schema_sha256}"
        )
    metadata_ids = pq.read_table(path, columns=["Metadata_id"])[
        "Metadata_id"
    ].to_pylist()
    if any(value is None or not isinstance(value, str) for value in metadata_ids):
        raise RuntimeError(f"{path} contains null or non-string Metadata_id values")
    well_ids = set(metadata_ids)
    if len(well_ids) != rows:
        raise RuntimeError(
            f"{path} has {rows - len(well_ids):,} duplicate Metadata_id rows"
        )
    malformed = [value for value in well_ids if len(value.split("__")) != 4]
    if malformed:
        raise RuntimeError(f"{path} has malformed well Metadata_id: {malformed[0]!r}")
    return {
        "artifact_type": "raw_feature_profile",
        "family": identity.family,
        "model": identity.model,
        "codec": identity.codec,
        "coverage_status": "partial" if rows != 1_536 else "complete",
        "source_group": source_group,
        "source_relative_path": path.relative_to(root).as_posix(),
        "relative_key": profile_relative_key(identity),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256_source_file(path, stat),
        "row_count": rows,
        "column_count": columns,
        "well_id_sha256": sha256_json(sorted(well_ids)),
        "schema_sha256": schema_sha256,
        "_well_ids": well_ids,
    }


def build_profile_inventory(
    cp_root: Path,
    dl_root: Path,
    *,
    workers: int,
    expected_cp_files: int = EXPECTED_CP_ROOT_FILE_COUNT,
    expected_cp_bytes: int | None = EXPECTED_CP_ROOT_BYTES,
    expected_dl_files: int = EXPECTED_DL_ROOT_FILE_COUNT,
    expected_dl_bytes: int | None = EXPECTED_DL_ROOT_BYTES,
) -> list[dict[str, Any]]:
    cp_paths = sorted(cp_root.glob("*.parquet"))
    dl_paths = sorted(dl_root.glob("*.parquet"))
    if len(cp_paths) != expected_cp_files:
        raise RuntimeError(
            f"compact CP root has {len(cp_paths)} Parquets; expected {expected_cp_files}"
        )
    if len(dl_paths) != expected_dl_files:
        raise RuntimeError(
            f"compact DL root has {len(dl_paths)} Parquets; expected {expected_dl_files}"
        )
    cp_bytes = sum(path.stat().st_size for path in cp_paths)
    dl_bytes = sum(path.stat().st_size for path in dl_paths)
    if expected_cp_bytes is not None and cp_bytes != expected_cp_bytes:
        raise RuntimeError(
            f"compact CP root has {cp_bytes:,} bytes; expected {expected_cp_bytes:,}"
        )
    if expected_dl_bytes is not None and dl_bytes != expected_dl_bytes:
        raise RuntimeError(
            f"compact DL root has {dl_bytes:,} bytes; expected {expected_dl_bytes:,}"
        )

    identities = [
        (path, cp_root, "compact_cp", parse_profile_filename(path.name))
        for path in cp_paths
    ]
    identities += [
        (path, dl_root, "compact_dl", parse_profile_filename(path.name))
        for path in dl_paths
    ]
    expected_combinations = {
        *(("cp_measure", "", codec) for codec in CP_PROFILE_CODECS),
        *(("cell_count", "", codec) for codec in CELL_COUNT_CODECS),
        *(
            ("deep_learning", model, codec)
            for model in DL_MODELS
            for codec in DL_PROFILE_CODECS
        ),
    }
    actual_combinations = {
        (identity.family, identity.model, identity.codec) for *_, identity in identities
    }
    if actual_combinations != expected_combinations:
        raise RuntimeError(
            "compact profile inventory mismatch: "
            f"missing={sorted(expected_combinations - actual_combinations)}, "
            f"extra={sorted(actual_combinations - expected_combinations)}"
        )
    if workers < 1:
        raise ValueError("workers must be positive")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(_profile_record, identities))
    canonical = next(
        (
            row["_well_ids"]
            for row in records
            if row["family"] == "cp_measure" and row["codec"] == "zstd"
        ),
        None,
    )
    if canonical is None or len(canonical) != 1_536:
        raise RuntimeError("the compact Zstd cp_measure profile is not a complete well set")
    for row in records:
        well_ids = row["_well_ids"]
        if row["coverage_status"] == "complete" and well_ids != canonical:
            raise RuntimeError(
                f"complete profile well set differs from Zstd for {row['relative_key']}: "
                f"missing={len(canonical - well_ids)}, extra={len(well_ids - canonical)}"
            )
        if not well_ids.issubset(canonical):
            raise RuntimeError(
                f"profile {row['relative_key']} contains "
                f"{len(well_ids - canonical)} non-Target-2 well IDs"
            )
        del row["_well_ids"]
    records.sort(key=lambda row: row["relative_key"])
    keys = [row["relative_key"] for row in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate compact profile destinations")
    return records


def _table_for_rows(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> pa.Table:
    normalized = [{column: row[column] for column in columns} for row in rows]
    return pa.Table.from_pylist(normalized)


def write_manifest(
    path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = _table_for_rows(rows, columns)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        use_dictionary=True,
        write_statistics=True,
        version="2.6",
        data_page_version="1.0",
    )


def pipeline_script_inventory(mask_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(mask_root.glob("*.zarr/*_script.py")):
        initial_stat = path.stat()
        rows.append(
            {
                "relative_path": path.relative_to(mask_root).as_posix(),
                "size_bytes": initial_stat.st_size,
                "sha256": sha256_source_file(path, initial_stat),
            }
        )
    if not rows:
        raise RuntimeError(
            f"no recorded Target-2 pipeline scripts found under {mask_root}"
        )
    return rows


def profile_producer_inventory(project_root: Path) -> list[dict[str, Any]]:
    rows = []
    for relative_path in PROFILE_PRODUCER_PATHS:
        path = project_root / relative_path
        if not path.is_file():
            raise RuntimeError(f"missing Target-2 profile producer input: {path}")
        initial_stat = path.stat()
        rows.append(
            {
                "relative_path": relative_path,
                "size_bytes": initial_stat.st_size,
                "sha256": sha256_source_file(path, initial_stat),
            }
        )
    return rows


def current_commit(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        source_commit = project_root / ".source-commit"
        return source_commit.read_text().strip() if source_commit.exists() else None


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def build_release(
    *,
    mask_root: Path,
    cp_profile_root: Path,
    dl_profile_root: Path,
    release_root: Path,
    readme_source: Path,
    workers: int,
) -> dict[str, Any]:
    mask_rows = build_mask_inventory(mask_root, workers=workers)
    object_feature_rows = build_object_feature_inventory(mask_root, workers=workers)
    profile_rows = build_profile_inventory(
        cp_profile_root, dl_profile_root, workers=workers
    )
    mask_site_keys_by_codec: dict[str, set[str]] = {}
    for row in mask_rows:
        mask_site_keys_by_codec.setdefault(row["codec"], set()).add(row["site_key"])
    for row in object_feature_rows:
        row["retained_mask_pair_present"] = (
            row["site_key"] in mask_site_keys_by_codec[row["codec"]]
        )

    release_root.mkdir(parents=True, exist_ok=True)
    mask_manifest = release_root / "manifests/masks.parquet"
    object_feature_manifest = release_root / "manifests/object_features.parquet"
    profile_manifest = release_root / "manifests/profiles.parquet"
    write_manifest(mask_manifest, mask_rows, MASK_MANIFEST_COLUMNS)
    write_manifest(
        object_feature_manifest,
        object_feature_rows,
        OBJECT_FEATURE_MANIFEST_COLUMNS,
    )
    write_manifest(profile_manifest, profile_rows, PROFILE_MANIFEST_COLUMNS)
    shutil.copyfile(readme_source, release_root / "README.md")

    scripts = pipeline_script_inventory(mask_root)
    project_root = Path(__file__).resolve().parents[1]
    profile_producers = profile_producer_inventory(project_root)
    shape_counts: dict[str, int] = {}
    shape_counts_by_codec_object: dict[str, int] = {}
    for row in mask_rows:
        shape = (
            f"{row['array_height']}x{row['array_width']}"
            if row["array_ndim"] == 2
            else f"1x{row['array_height']}x{row['array_width']}"
        )
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
        detail_key = f"{row['codec']}|{row['object_type']}|{shape}"
        shape_counts_by_codec_object[detail_key] = (
            shape_counts_by_codec_object.get(detail_key, 0) + 1
        )

    object_site_keys_by_codec: dict[str, set[str]] = {}
    for row in object_feature_rows:
        object_site_keys_by_codec.setdefault(row["codec"], set()).add(row["site_key"])
    mask_relationship_by_codec = {
        codec: {
            "shared_sites": len(
                object_site_keys_by_codec[codec] & mask_site_keys_by_codec[codec]
            ),
            "object_features_without_retained_masks": len(
                object_site_keys_by_codec[codec] - mask_site_keys_by_codec[codec]
            ),
            "retained_masks_without_object_features": len(
                mask_site_keys_by_codec[codec] - object_site_keys_by_codec[codec]
            ),
        }
        for codec in sorted(EXPECTED_OBJECT_FEATURE_SITE_COUNTS)
    }

    provenance = {
        "schema_version": 2,
        "destination_root": DESTINATION_ROOT,
        "target2_expected_sites": EXPECTED_TARGET2_SITES,
        "mask_inventory": {
            "file_count": len(mask_rows),
            "bytes": sum(int(row["size_bytes"]) for row in mask_rows),
            "site_counts_by_codec": dict(sorted(EXPECTED_MASK_SITE_COUNTS.items())),
            "partial_codecs": sorted(
                codec
                for codec, count in EXPECTED_MASK_SITE_COUNTS.items()
                if count != EXPECTED_TARGET2_SITES
            ),
            "spatial_shape_by_source": {
                source: list(shape)
                for source, shape in sorted(EXPECTED_SPATIAL_SHAPE_BY_SOURCE.items())
            },
            "shape_counts": dict(sorted(shape_counts.items())),
            "shape_counts_by_codec_object": dict(
                sorted(shape_counts_by_codec_object.items())
            ),
            "manifest_sha256": sha256_file(mask_manifest),
        },
        "object_feature_inventory": {
            "file_count": len(object_feature_rows),
            "bytes": sum(int(row["size_bytes"]) for row in object_feature_rows),
            "site_counts_by_codec": dict(
                sorted(EXPECTED_OBJECT_FEATURE_SITE_COUNTS.items())
            ),
            "bytes_by_codec": dict(
                sorted(EXPECTED_OBJECT_FEATURE_BYTES_BY_CODEC.items())
            ),
            "partial_codecs": sorted(
                codec
                for codec, count in EXPECTED_OBJECT_FEATURE_SITE_COUNTS.items()
                if count != EXPECTED_TARGET2_SITES
            ),
            "row_counts_by_codec": dict(
                sorted(EXPECTED_OBJECT_FEATURE_ROWS_BY_CODEC.items())
            ),
            "empty_file_counts_by_codec": dict(
                sorted(EXPECTED_OBJECT_FEATURE_EMPTY_COUNTS.items())
            ),
            "site_set_sha256_by_codec": dict(
                sorted(EXPECTED_OBJECT_FEATURE_SITE_SET_SHA256.items())
            ),
            "arrow_schema_sha256_by_empty": {
                str(empty).lower(): digest
                for empty, digest in EXPECTED_OBJECT_FEATURE_SCHEMA_SHA256_BY_EMPTY.items()
            },
            "mask_relationship_by_codec": mask_relationship_by_codec,
            "manifest_sha256": sha256_file(object_feature_manifest),
        },
        "profile_inventory": {
            "file_count": len(profile_rows),
            "bytes": sum(int(row["size_bytes"]) for row in profile_rows),
            "families": {
                family: sum(row["family"] == family for row in profile_rows)
                for family in ("cp_measure", "cell_count", "deep_learning")
            },
            "schema_sha256_by_family_model": {
                f"{family}|{model}": digest
                for (family, model), digest in sorted(
                    EXPECTED_PROFILE_SCHEMA_SHA256.items()
                )
            },
            "manifest_sha256": sha256_file(profile_manifest),
        },
        "package_metadata": {
            "readme_sha256": sha256_file(release_root / "README.md"),
            "builder_sha256": sha256_file(Path(__file__)),
            "repository_commit": current_commit(project_root),
        },
        "recorded_mask_pipeline_scripts": scripts,
        "recorded_mask_pipeline_script_inventory_sha256": sha256_json(scripts),
        "recorded_profile_producer_files": profile_producers,
        "recorded_profile_producer_inventory_sha256": sha256_json(
            profile_producers
        ),
        "source_roots": {
            "masks": str(mask_root),
            "compact_cp_profiles": str(cp_profile_root),
            "compact_dl_profiles": str(dl_profile_root),
            "per_site_cp_measure_object_features": str(mask_root),
        },
    }
    atomic_write_json(release_root / "manifests/provenance.json", provenance)
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-root", type=Path, default=DEFAULT_MASK_ROOT)
    parser.add_argument("--cp-profile-root", type=Path, default=DEFAULT_CP_PROFILE_ROOT)
    parser.add_argument("--dl-profile-root", type=Path, default=DEFAULT_DL_PROFILE_ROOT)
    parser.add_argument("--release-root", type=Path, default=DEFAULT_RELEASE_ROOT)
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path(__file__).with_name("TARGET2_ARTIFACTS_README.md"),
    )
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    provenance = build_release(
        mask_root=args.mask_root,
        cp_profile_root=args.cp_profile_root,
        dl_profile_root=args.dl_profile_root,
        release_root=args.release_root,
        readme_source=args.readme,
        workers=args.workers,
    )
    print(
        "Target-2 artifact package ready: "
        f"masks={provenance['mask_inventory']['file_count']:,}, "
        f"profiles={provenance['profile_inventory']['file_count']:,}, "
        f"object_features={provenance['object_feature_inventory']['file_count']:,}, "
        f"root={args.release_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
