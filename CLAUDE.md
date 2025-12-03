# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JUMP_core is a Python project for downloading and compressing biological imaging data from the JUMP Cell Painting dataset. It provides utilities for:

1. **Image Download**: Fetching cell painting images for specific perturbations (CRISPR, ORF, compounds)
2. **Image Compression**: Benchmarking compression algorithms on TIFF images using Zarr storage format
3. **Quality Evaluation**: Computing SSIM (Structural Similarity Index) to assess lossy compression quality with visualizations

## Development Environment

This project uses:
- **Nix Flake**: Complete development environment setup
- **uv**: Python package manager for dependency management
- **Python 3.13**: Required minimum version

### Environment Setup

```bash
# Enter the Nix development shell (recommended)
nix develop

# Alternative: manual setup with uv
uv sync --all-groups
source .venv/bin/activate
```

The Nix flake automatically:
- Sets up Python 3.13 environment
- Installs uv and dependencies via `uv sync --all-groups`
- Configures necessary system libraries (libz, libGL, glib)
- Activates the virtual environment

## Core Components

### src/download_images.py
Downloads cell painting images from the JUMP dataset:
- Fetches metadata for CRISPR, ORF, and compound perturbations
- Downloads specific channels (DNA, ER, Mito) and sites
- Saves images as TIFF files in `./images/raw/`
- Uses `jump-portrait` library for data access
- Supports parallel processing via joblib

Key configuration:
- Sample size: configurable (default 10 per perturbation type)
- Channels: DNA, ER, Mito
- Sites: 1-6 (currently limited to site 1)
- Output format: TIFF with structured naming `source__batch__plate__well__channel__site.tif`

### src/compress_tif.py
Benchmarks compression algorithms on downloaded images:
- Groups images by site and metadata (source, batch, plate, well)
- Tests multiple compression codecs:
  - **Lossless**: Blosc variants (lz4hc, zstd, zlib)
  - **Lossy**: JpegXL with various distance/effort/decodingspeed settings
- Stores compressed data in Zarr format (v2 for imagecodecs, v3 for Blosc)
- Measures compression time, decompression time, and file size
- Computes SSIM (Structural Similarity Index) for quality assessment
- Generates comparison visualizations (original, compressed, difference) using matplotlib

**Architecture notes**:
- Uses parallel compression via joblib for performance
- Configurable via `input_dir`, `output_dir` paths
- JpegXL codecs support quality/speed tradeoffs via `distance`, `effort`, and `decodingspeed` parameters
- Compression results stored as `{codec_name}.zarr/` directories

### src/evaluate_quality.py
GPU-accelerated quality metrics evaluation for compressed images:
- Computes three standardized metrics: **PSNR, SSIM, and LPIPS**
- Compares all compressed codecs against original images
- Outputs results in three formats:
  - **Terminal table**: Rich-formatted comparison table with quality ratings
  - **Matplotlib visualizations**: Per-site comparison plots with metrics overlaid
  - **CSV/JSON exports**: Detailed and summary statistics for further analysis

**Metrics explained**:
- **PSNR** (Peak Signal-to-Noise Ratio): Pixel-level accuracy in dB. Higher = better. 30+ good, 35+ excellent.
- **SSIM** (Structural Similarity Index): Structural similarity on 0-1 scale. Higher = better. 0.9+ good.
- **LPIPS** (Learned Perceptual Image Patch Similarity): Perceptual quality using neural networks. Lower = better. <0.1 good.

**GPU acceleration**:
- Uses PyTorch with CUDA for fast metric computation
- Automatically falls back to CPU if GPU not available
- Target: 50+ 1080p images per minute on GPU
- LPIPS uses VGG backbone for better perceptual accuracy (can switch to 'alex' for speed)

**Architecture notes**:
- Automatically uses same paths as `compress_tif.py` (`/work/datasets/jump_toy/raw` and parent)
- Auto-discovers all compression codecs by scanning for `.zarr` directories
- Handles multi-channel microscopy images (LPIPS computed per channel and averaged)
- Normalizes uint16 images to [0,1] float32 for metric computation
- Results saved to `{output_dir}/results/` directory

### src/merge_metrics.py
Utility script to combine compression and quality metrics:
- Merges `compression_metrics.csv` (from compress_tif.py) with `metrics_summary.csv` (from evaluate_quality.py)
- Creates unified `metrics_combined.csv` and `metrics_combined.json`
- Provides comprehensive codec comparison with all metrics in one table
- Useful for analysis, plotting, and decision-making about codec selection

**Output columns**:
- codec: Codec name
- filesize_ratio: Compressed size / raw size (lower = better)
- compression_ratio: Raw size / compressed size (higher = better)
- compression_time_sec: Time to compress all images
- decompression_time_sec: Time to decompress all images
- filesize_bytes: Absolute compressed size
- psnr_mean/std/min/max: PSNR statistics
- ssim_mean/std/min/max: SSIM statistics
- lpips_mean/std/min/max: LPIPS statistics

### metadata/repurposed_compounds.tsv
Contains compound metadata with repurposing information:
- JCP2022 identifiers
- Compound names and mechanisms of action
- Target proteins

## Key Dependencies

- `jump-portrait>=0.0.29`: JUMP dataset access
- `zarr>=3.1.3`: Compressed array storage (supports v2 and v3 formats)
- `imagecodecs>=2025.8.2`: Image compression codecs (JpegXL, Brotli)
- `pooch>=1.8.2`: Data downloading utilities
- `polars`: DataFrame operations for metadata handling
- `scikit-image>=0.25.2`: SSIM computation and image quality metrics
- `torch>=2.0.0`: PyTorch for GPU-accelerated metrics
- `torchvision>=0.15.0`: Image processing utilities for PyTorch
- `torchmetrics>=1.0.0`: GPU-accelerated PSNR and SSIM implementations
- `lpips>=0.1.4`: Learned Perceptual Image Patch Similarity metric
- `pandas>=2.0.0`: Data handling and CSV export
- `rich>=13.0.0`: Terminal table formatting
- Development: `jupyter>=1.1.1`, `matplotlib` (for visualizations)

## Running Scripts

```bash
# 1. Download sample images (creates ./images/raw/*.tif)
python src/download_images.py

# 2. Benchmark compression algorithms (reads from input_dir, writes to output_dir)
# Note: Edit input_dir/output_dir paths in the script as needed
python src/compress_tif.py

# 3. Evaluate compression quality with GPU-accelerated metrics
python src/evaluate_quality.py

# 4. Merge compression and quality metrics (optional)
python src/merge_metrics.py
```

**evaluate_quality.py** features:
- Automatically uses the same paths as `compress_tif.py` (no configuration needed)
- Auto-discovers all compression codecs by scanning for `.zarr` directories
- Only command-line option: `--no-plots` to skip matplotlib visualizations

The evaluation script outputs:
1. **Terminal table**: Formatted comparison of all codecs with quality ratings (Excellent/Good/Fair/Poor)
2. **Detailed CSV**: `{output_dir}/results/metrics_detailed.csv` with per-site metrics
3. **Summary CSV**: `{output_dir}/results/metrics_summary.csv` with mean/std/min/max statistics
4. **JSON exports**: `{output_dir}/results/metrics_detailed.json` and `metrics_summary.json`
5. **Visualizations**: `{output_dir}/results/comparison_{site}.png` for first 5 sites (unless --no-plots)

The compression script (compress_tif.py) outputs:
1. Compression times (parallel execution)
2. Decompression times (milliseconds, sorted by speed)
3. File size ratios (fraction of raw)
4. **Saved metrics**: `results/compression_metrics.csv` and `.json` with compression/decompression times and file sizes
5. SSIM quality metrics for each codec (CPU-based, legacy)
6. Visual comparison PNGs showing original/compressed/difference

**merge_metrics.py** combines data from both scripts:
- Merges `compression_metrics.csv` with `metrics_summary.csv`
- Creates `metrics_combined.csv` and `.json` with all metrics in one table
- Columns: codec, filesize_ratio, compression_time, decompression_time, psnr_mean, ssim_mean, lpips_mean, etc.
- Useful for comprehensive codec comparison and analysis

## Key Configuration Points

### download_images.py
- `sample`: Number of perturbations per type (default: 10)
- `seed`: Random seed for reproducible sampling
- `channels`: Image channels to download (DNA, ER, Mito)
- `sites`: Imaging sites to include (1-6, currently limited to 1)
- `correction`: Image correction type ("Orig")
- `out_path`: Output directory for raw TIFFs

### compress_tif.py
- `input_dir`: Path to raw TIFF images
- `output_dir`: Path for compressed `.zarr` outputs
- `overwrite`: Whether to overwrite existing compressed data
- `compressing_algs`: Dictionary of Blosc codec configurations
- `compressors`: Combined dict of all codecs to test (Blosc + imagecodecs)
- JpegXL parameters:
  - `distance`: Quality setting (1.0=high quality, 5.0=low quality)
  - `effort`: Compression effort (1=fast, 9=slow, higher compression)
  - `decodingspeed`: Decoding speed tier (0-4, higher=faster decode)

## Data Flow

### Complete Workflow
1. **Metadata Retrieval**: Fetch JCP2022 IDs for perturbations from JUMP dataset CSVs
2. **Parallel Address Resolution**: Map IDs to storage locations (source/batch/plate/well) using joblib
3. **Image Download**: Retrieve TIFF images for specified channels/sites via `jump-portrait`
4. **Image Grouping**: Group TIFFs by site and metadata (source/batch/plate/well/site)
5. **Parallel Compression**: Apply codecs to image groups, store as Zarr arrays
6. **Performance Benchmarking**: Measure compression/decompression time and file sizes
7. **Quality Evaluation**: GPU-accelerated computation of PSNR, SSIM, and LPIPS metrics
8. **Results Export**: Generate terminal tables, visualizations, and CSV/JSON outputs

### Quality Evaluation Workflow (evaluate_quality.py)
1. **GPU Detection**: Check for CUDA availability, fallback to CPU
2. **Load Originals**: Read all TIFF images from raw directory, group by site
3. **Codec Iteration**: For each .zarr directory (codec):
   - Load compressed images for all sites
   - Compute PSNR, SSIM, LPIPS on GPU (batched)
   - Store results in DataFrame
4. **Aggregate Results**: Calculate mean/std/min/max per codec
5. **Output Generation**:
   - Print rich-formatted terminal table
   - Save detailed and summary CSV files
   - Export JSON with full statistics
   - Generate matplotlib comparison plots (first 5 sites)

## Important Implementation Details

### Image Naming Convention
Downloaded TIFFs follow the pattern: `{source}__{batch}__{plate}__{well}__{channel}__{site}.tif`
This structured naming enables grouping by site for compression.

### Zarr Format Selection
- **Zarr v3**: Used for Blosc codecs (BloscCodec)
- **Zarr v2**: Used for imagecodecs (JpegXL, Brotli) due to API compatibility

### Parallel Processing
- Download: joblib with `n_jobs=-1` for metadata fetching
- Compression: joblib with `prefer="threads"` for parallel codec testing
- Both leverage all available CPU cores

### SSIM Evaluation
The script computes Structural Similarity Index to quantify lossy compression quality:
- Range: -1 to 1 (1 = identical, 0 = no structural similarity)
- Uses `multichannel=True` and `channel_axis=0` for multi-channel images
- Visualizations show original, compressed, and pixel-wise difference for each codec

### GPU-Accelerated Quality Metrics (evaluate_quality.py)
Three complementary metrics provide comprehensive quality assessment:

**PSNR (Peak Signal-to-Noise Ratio)**:
- Measures pixel-level accuracy in decibels
- Computed on GPU via `torchmetrics.PeakSignalNoiseRatio`
- Higher values = better quality (30+ good, 35+ excellent)
- Fast to compute but doesn't capture perceptual quality

**SSIM (Structural Similarity Index)**:
- Measures structural similarity on 0-1 scale
- Computed on GPU via `torchmetrics.StructuralSimilarityIndexMeasure`
- Higher values = better (0.9+ good, 0.95+ excellent)
- Better correlation with human perception than PSNR

**LPIPS (Learned Perceptual Image Patch Similarity)**:
- Neural network-based perceptual quality metric
- Uses AlexNet backbone (faster than VGG) on GPU
- Lower values = better quality (<0.1 good, <0.05 excellent)
- Best correlation with human perception
- For multi-channel microscopy: computed per channel, then averaged
- Handles multi-channel by converting each channel to pseudo-RGB (repeat 3x)

**Image Normalization for Metrics**:
- uint16 images: divide by 65535 to get [0, 1] range
- uint8 images: divide by 255 to get [0, 1] range
- Convert to float32 PyTorch tensors with batch dimension (1, C, H, W)
- All metrics computed in float32 precision on GPU