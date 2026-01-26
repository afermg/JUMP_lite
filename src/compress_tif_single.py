"""
Script to compress pulled tifs with a single codec, grouping them by their site.

Usage:
    python compress_tif_single.py --input /path/to/raw --output /path/to/output --codec jpegxl_lossy_hq
"""

import argparse
import lzma
from itertools import groupby
from pathlib import Path
from pprint import pprint
from shutil import rmtree
from time import perf_counter

import numpy
import zarr
from joblib import Parallel, delayed
from tqdm import tqdm

try:
    import numcodecs
    from imagecodecs.numcodecs import Brotli, Jpegxl

    # Register the codecs manually
    numcodecs.register_codec(Brotli)
    numcodecs.register_codec(Jpegxl)
    IMAGECODECS_AVAILABLE = True
except (ImportError, AttributeError):
    IMAGECODECS_AVAILABLE = False
    print("Warning: imagecodecs.numcodecs not available, skipping JpegXL and Brotli")
from PIL import Image
from zarr.codecs import BloscCodec


# =============================================================================
# Command line arguments
# =============================================================================

parser = argparse.ArgumentParser(description="Compress TIF images to zarr with a single codec")
parser.add_argument("--input", type=str, required=True,
                    help="Input directory containing .tif files")
parser.add_argument("--output", type=str, required=True,
                    help="Output directory for compressed zarr files")
parser.add_argument("--codec", type=str, required=True,
                    help="Codec to use: zstd, jpegxl_lossy_hq, jpegxl_lossy_mq, jpegxl_lossy_lq, jpegxl_lossy_effort_3")
parser.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing zarr files")
parser.add_argument("--n-jobs", type=int, default=16,
                    help="Number of parallel workers for group compression (default: 16)")
parser.add_argument("--no-skip-existing", action="store_true",
                    help="Don't skip already compressed groups (recompress everything)")
args = parser.parse_args()


# =============================================================================
# Compression functions
# =============================================================================

def compress_single_group(key, items, store_name, compressor, zarr_format, skip_existing=True):
    """
    Compress a single image group (site).

    Returns:
        tuple: (site_name, success, error_message)
    """
    site_name = "__".join(key)

    # Check if already compressed
    if skip_existing:
        store = zarr.storage.LocalStore(store_name)
        try:
            root = zarr.open_group(store, mode="r")
            if site_name in root:
                return (site_name, "skipped", None)
        except Exception:
            pass  # Store doesn't exist yet, continue with compression

    try:
        nchannels = len(items)
        example_arr = numpy.array(Image.open(items[0]))
        shape = example_arr.shape
        dtype = example_arr.dtype

        store = zarr.storage.LocalStore(store_name)

        arr = zarr.create_array(
            store=store,
            name=site_name,
            shape=(nchannels, *shape),
            chunks=(nchannels, *shape),
            dtype=dtype,
            compressors=compressor,
            zarr_format=zarr_format,
        )
        tmp_arr = numpy.zeros((nchannels, *shape))
        for i, img_path in enumerate(items):
            tmp_arr[i] = numpy.array(Image.open(img_path))

        arr[:] = tmp_arr
        return (site_name, "success", None)

    except Exception as e:
        return (site_name, "error", str(e))


def compress_tif(name, compressor, output_dir, groups, overwrite=False, n_jobs_inner=16, skip_existing=True):
    """
    Uses parallel processing for group compression.

    Args:
        n_jobs_inner: Number of parallel jobs for group compression within each codec.
                      Default 16 to avoid over-parallelization when running multiple codecs.
        skip_existing: Skip groups that have already been compressed (default: True)
    """
    store_name = Path(output_dir) / f"{name}.zarr"
    if store_name.exists() and overwrite:
        rmtree(store_name)

    t_start = perf_counter()

    # The API for codecs changed with Zarr 3
    # https://github.com/cgohlke/imagecodecs/issues/123
    zarr_format = 3
    if not isinstance(compressor, zarr.codecs.blosc.BloscCodec):
        zarr_format = 2

    subset = list(groups.items())

    # Compress groups in parallel with limited parallelism to avoid thrashing
    results = Parallel(n_jobs=n_jobs_inner, prefer="threads")(
        delayed(compress_single_group)(key, items, store_name, compressor, zarr_format, skip_existing)
        for key, items in tqdm(subset, total=len(subset), desc=name, leave=False)
    )

    # Count results
    success_count = sum(1 for r in results if r[1] == "success")
    skipped_count = sum(1 for r in results if r[1] == "skipped")
    error_count = sum(1 for r in results if r[1] == "error")

    # Report errors
    errors = [(r[0], r[2]) for r in results if r[1] == "error"]
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for site_name, error_msg in errors[:10]:
            print(f"    {site_name}: {error_msg[:80]}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more errors")

    return {
        "name": name,
        "time": perf_counter() - t_start,
        "success": success_count,
        "skipped": skipped_count,
        "errors": error_count,
        "error_list": errors
    }


# =============================================================================
# Main
# =============================================================================

input_dir = Path(args.input)
output_dir = Path(args.output)
overwrite = args.overwrite

print("Input dir:", input_dir)
print("Output dir:", output_dir)

output_dir.mkdir(parents=True, exist_ok=True)

# Define available codecs
compressing_algs = {
    "zstd": {"clevel": 9}
}
compressors_blosc = {
    k: BloscCodec(cname=k, shuffle="bitshuffle", **v)
    for k, v in compressing_algs.items()
}

compressors = {
    **compressors_blosc,
}

# Add imagecodecs compressors if available
if IMAGECODECS_AVAILABLE:
    compressors.update({
        "jpegxl_lossy_hq": Jpegxl(lossless=False, distance=1.0),
        "jpegxl_lossy_mq": Jpegxl(lossless=False, distance=3.0),
        "jpegxl_lossy_lq": Jpegxl(lossless=False, distance=5.0),
        "jpegxl_lossy_effort_3": Jpegxl(lossless=False, distance=1.0, effort=3)
    })

# Select the specified codec
if args.codec not in compressors:
    print(f"Error: Unknown codec '{args.codec}'")
    print(f"Available codecs: {list(compressors.keys())}")
    exit(1)

name = args.codec
compressor = compressors[name]
print(f"Using codec: {name}")

# Group files based on their name
key_fn = lambda x: (*(x.name.split("__"))[:4], (x.name.split("__"))[5])

groups = {
    k: sorted(g)
    for k, g in groupby(sorted(input_dir.glob("*.tif"), key=key_fn), key=key_fn)
}

print(f"Found {len(groups)} groups to compress")

skip_existing = not args.no_skip_existing
if skip_existing:
    print("Skipping already compressed groups (use --no-skip-existing to recompress)")

# Run compression for single codec
result = compress_tif(name, compressor, output_dir, groups, overwrite, args.n_jobs, skip_existing)

print(f"\nCompression complete:")
print(f"  Time: {result['time']:.1f} seconds")
print(f"  Success: {result['success']}")
print(f"  Skipped: {result['skipped']}")
print(f"  Errors: {result['errors']}")

# Measure file size
store_name = output_dir / f"{name}.zarr"
if store_name.exists():
    filesize = sum(file.stat().st_size for file in store_name.rglob("*"))
    print(f"Output size: {filesize / 1e9:.2f} GB")

# Decompression test
if store_name.exists():
    print("\nTesting decompression...")
    store = zarr.storage.LocalStore(store_name)
    t_start = perf_counter()
    root = zarr.group(store)
    keys = list(root.keys())
    if keys:
        for k in keys:
            tmp = root[k][:]
        decompression_time = perf_counter() - t_start
        print(f"Decompression time: {decompression_time * 1000:.1f} ms")
    else:
        print("No data to decompress")
