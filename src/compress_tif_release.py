"""Compress five-channel JUMP TIFF sites into one Zarr store per codec.

Input filenames must follow the release convention::

    <source>__<batch>__<plate>__<well>__<site>__<channel>.tif

The five channels are stored in the canonical order AGP, DNA, ER, Mito, RNA.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from shutil import rmtree
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
import zarr
from joblib import Parallel, delayed
from PIL import Image
from tqdm import tqdm

try:
    import numcodecs
    from imagecodecs.numcodecs import Brotli, Jpegxl

    numcodecs.register_codec(Brotli)
    numcodecs.register_codec(Jpegxl)
    IMAGECODECS_AVAILABLE = True
except (ImportError, AttributeError):
    IMAGECODECS_AVAILABLE = False
    Jpegxl = None

from zarr.codecs import BloscCodec

CHANNEL_ORDER = ("AGP", "DNA", "ER", "Mito", "RNA")
JXL_SETTINGS: dict[str, dict[str, Any]] = {
    "jpegxl_lossy_hq": {"lossless": False, "distance": 1.0},
    "jpegxl_lossy_mq": {"lossless": False, "distance": 3.0},
    "jpegxl_lossy_mq_new": {"lossless": False, "distance": 3.0},
    "jpegxl_lossy_lq": {"lossless": False, "distance": 5.0},
    "jpegxl_lossy_d2_e8": {"lossless": False, "distance": 2.0, "effort": 8},
    "jpegxl_lossy_d10": {"lossless": False, "distance": 10.0},
    "jpegxl_lossy_d15": {"lossless": False, "distance": 15.0},
    "jpegxl_lossy_d20": {"lossless": False, "distance": 20.0},
    "jpegxl_lossy_d20_e2": {"lossless": False, "distance": 20.0, "effort": 2},
    "jpegxl_lossy_d30": {"lossless": False, "distance": 30.0},
    "jpegxl_lossy_effort_3": {"lossless": False, "distance": 1.0, "effort": 3},
}


@dataclass(frozen=True)
class ParsedTiff:
    path: Path
    site: tuple[str, str, str, str, str]
    channel: str


def parse_tiff(path: Path) -> ParsedTiff:
    parts = path.stem.split("__")
    if len(parts) != 6 or not all(parts):
        raise ValueError(f"invalid JUMP TIFF name: {path.name}")
    channel = parts[5]
    if channel not in CHANNEL_ORDER:
        raise ValueError(f"unknown channel in {path.name}: {channel}")
    return ParsedTiff(path=path, site=tuple(parts[:5]), channel=channel)


def group_tiffs(paths: Iterable[Path]) -> dict[tuple[str, ...], list[Path]]:
    grouped: dict[tuple[str, ...], dict[str, Path]] = {}
    for path in sorted(paths):
        parsed = parse_tiff(path)
        channels = grouped.setdefault(parsed.site, {})
        if parsed.channel in channels:
            raise ValueError(
                f"duplicate {parsed.channel} TIFF for {'__'.join(parsed.site)}"
            )
        channels[parsed.channel] = parsed.path

    result: dict[tuple[str, ...], list[Path]] = {}
    for site, channels in sorted(grouped.items()):
        missing = sorted(set(CHANNEL_ORDER) - set(channels))
        extra = sorted(set(channels) - set(CHANNEL_ORDER))
        if missing or extra:
            raise ValueError(
                f"invalid channel inventory for {'__'.join(site)}: "
                f"missing={missing}, extra={extra}"
            )
        result[site] = [channels[channel] for channel in CHANNEL_ORDER]
    if not result:
        raise ValueError("no TIFF sites found")
    return result


def available_compressors() -> dict[str, Any]:
    compressors: dict[str, Any] = {
        "zstd": BloscCodec(cname="zstd", clevel=9, shuffle="bitshuffle")
    }
    if IMAGECODECS_AVAILABLE:
        assert Jpegxl is not None
        compressors.update(
            {name: Jpegxl(**settings) for name, settings in JXL_SETTINGS.items()}
        )
    return compressors


def read_stack(items: list[Path]) -> np.ndarray:
    images: list[np.ndarray] = []
    for path in items:
        with Image.open(path) as image:
            array = np.asarray(image)
        if array.ndim != 2 or array.dtype != np.uint16:
            raise ValueError(
                f"source must be a 2-D uint16 TIFF: {path} {array.shape} {array.dtype}"
            )
        images.append(array)
    if len({image.shape for image in images}) != 1:
        raise ValueError("channel shapes differ")
    return np.stack(images, axis=0)


def validate_site_array(
    path: Path, expected_shape: tuple[int, ...] | None = None
) -> None:
    """Require one complete, readable, full-site uint16 Zarr array."""
    array = zarr.open_array(path, mode="r")
    shape = tuple(array.shape)
    if (
        len(shape) != 3
        or shape[0] != len(CHANNEL_ORDER)
        or array.dtype != np.dtype("uint16")
        or tuple(array.chunks) != shape
        or (expected_shape is not None and shape != expected_shape)
    ):
        raise ValueError(f"invalid existing site array contract: {path}")
    decoded = array[:]
    if decoded.shape != shape or decoded.dtype != np.uint16:
        raise ValueError(f"existing site chunk is incomplete or unreadable: {path}")


def compress_single_group(
    key: tuple[str, ...],
    items: list[Path],
    store_name: Path,
    compressor: Any,
    zarr_format: int,
    skip_existing: bool = True,
) -> tuple[str, str, str | None]:
    """Compress one five-channel site and publish it by atomic rename."""
    site_name = "__".join(key)
    site_path = store_name / site_name
    if site_path.exists() or site_path.is_symlink():
        if not skip_existing:
            return (
                site_name,
                "error",
                "site already exists; use a new output or explicit --overwrite",
            )
        try:
            validate_site_array(site_path)
            return (site_name, "skipped", None)
        except Exception as error:
            return (site_name, "error", f"invalid existing site: {error}")

    staging_parent = store_name.parent / f".{store_name.name}.staging"
    staging_parent.mkdir(exist_ok=True)
    staging = staging_parent / f"{site_name}.{uuid4().hex}"
    try:
        stack = read_stack(items)
        array = zarr.create_array(
            store=zarr.storage.LocalStore(staging),
            shape=stack.shape,
            chunks=stack.shape,
            dtype=stack.dtype,
            compressors=compressor,
            zarr_format=zarr_format,
        )
        array[:] = stack
        validate_site_array(staging, stack.shape)
        staging.rename(site_path)
        return (site_name, "success", None)
    except Exception as error:  # job boundary: preserve every site failure
        rmtree(staging, ignore_errors=True)
        return (site_name, "error", str(error))


def compress_tif(
    name: str,
    compressor: Any,
    output_dir: Path,
    groups: dict[tuple[str, ...], list[Path]],
    *,
    overwrite: bool = False,
    n_jobs_inner: int = 16,
    skip_existing: bool = True,
) -> dict[str, Any]:
    """Compress grouped sites with bounded thread parallelism."""
    store_name = output_dir / f"{name}.zarr"
    if store_name.exists() and overwrite:
        rmtree(store_name)

    zarr_format = 3 if isinstance(compressor, BloscCodec) else 2
    zarr.open_group(
        store=zarr.storage.LocalStore(store_name), mode="a", zarr_format=zarr_format
    )
    start = perf_counter()
    results = Parallel(n_jobs=n_jobs_inner, prefer="threads")(
        delayed(compress_single_group)(
            key, items, store_name, compressor, zarr_format, skip_existing
        )
        for key, items in tqdm(
            list(groups.items()), total=len(groups), desc=name, leave=False
        )
    )
    staging_parent = store_name.parent / f".{store_name.name}.staging"
    try:
        staging_parent.rmdir()
    except OSError:
        pass
    errors = [(site, message) for site, status, message in results if status == "error"]
    return {
        "name": name,
        "time": perf_counter() - start,
        "success": sum(status == "success" for _, status, _ in results),
        "skipped": sum(status == "skipped" for _, status, _ in results),
        "errors": len(errors),
        "error_list": errors,
        "successful_sites": [
            site for site, status, _ in results if status == "success"
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/jump_target2_4plate/raw"),
        help="input directory containing release-named TIFF files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/jump_target2_4plate"),
        help="output directory for one Zarr store per codec",
    )
    parser.add_argument("--codec", default="zstd", help="compression codec name")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace the complete codec store"
    )
    parser.add_argument("--n-jobs", type=int, default=16)
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="treat an existing site as an error instead of validating and skipping it",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be positive")
    compressors = available_compressors()
    if args.codec not in compressors:
        available = ", ".join(sorted(compressors))
        raise ValueError(f"unknown codec {args.codec!r}; available: {available}")

    groups = group_tiffs(args.input.glob("*.tif"))
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Input dir: {args.input}")
    print(f"Output dir: {args.output}")
    print(f"Using codec: {args.codec}")
    print(f"Found {len(groups)} complete five-channel sites")

    result = compress_tif(
        args.codec,
        compressors[args.codec],
        args.output,
        groups,
        overwrite=args.overwrite,
        n_jobs_inner=args.n_jobs,
        skip_existing=not args.no_skip_existing,
    )
    print("\nCompression complete:")
    pprint({key: value for key, value in result.items() if key != "successful_sites"})
    if result["errors"]:
        return 1

    store_name = args.output / f"{args.codec}.zarr"
    file_size = sum(
        path.stat().st_size for path in store_name.rglob("*") if path.is_file()
    )
    print(f"Output size: {file_size / 1e6:.2f} MB")

    check_site = next(iter(result["successful_sites"]), None)
    if check_site is not None:
        array = zarr.open_array(store_name / check_site, mode="r")
        decoded = array[:]
        print(f"Decompression check: {check_site} {decoded.shape} {decoded.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
