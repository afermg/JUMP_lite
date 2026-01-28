#!/usr/bin/env python3
"""
Compare segmentation masks from different compression methods against ground truth.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import seaborn as sns

# Try to import medpy, fallback to scipy
try:
    from medpy.metric.binary import dc, jc, hd95, asd, precision, recall
    USE_MEDPY = True
except ImportError:
    USE_MEDPY = False
    from scipy.spatial.distance import directed_hausdorff
    warnings.warn("medpy not available, using scipy fallback for Hausdorff distance")


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Dice coefficient."""
    if USE_MEDPY:
        return dc(pred, gt)
    intersection = np.sum(pred & gt)
    return 2.0 * intersection / (np.sum(pred) + np.sum(gt)) if (np.sum(pred) + np.sum(gt)) > 0 else 0.0


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute Intersection over Union (Jaccard)."""
    if USE_MEDPY:
        return jc(pred, gt)
    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    return intersection / union if union > 0 else 0.0


def compute_hausdorff_95(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute 95th percentile Hausdorff distance."""
    if USE_MEDPY:
        try:
            return hd95(pred, gt)
        except:
            return np.nan

    # Scipy fallback - compute full Hausdorff then approximate
    pred_points = np.argwhere(pred)
    gt_points = np.argwhere(gt)

    if len(pred_points) == 0 or len(gt_points) == 0:
        return np.nan

    try:
        d1 = directed_hausdorff(pred_points, gt_points)[0]
        d2 = directed_hausdorff(gt_points, pred_points)[0]
        return max(d1, d2)
    except:
        return np.nan


def compute_asd(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute average surface distance."""
    if USE_MEDPY:
        try:
            return asd(pred, gt)
        except:
            return np.nan
    return np.nan


def compute_precision(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute precision."""
    if USE_MEDPY:
        return precision(pred, gt)
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def compute_recall(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute recall."""
    if USE_MEDPY:
        return recall(pred, gt)
    tp = np.sum(pred & gt)
    fn = np.sum(~pred & gt)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def load_mask(npz_path: Path) -> np.ndarray:
    """Load segmentation mask from .npz file."""
    data = np.load(npz_path)
    # Assume the mask is stored under a key, try common names
    for key in ['mask', 'segmentation', 'arr_0']:
        if key in data:
            # Squeeze out singleton dimensions to ensure consistent shape
            return np.squeeze(data[key]).astype(bool)
    # If none found, take the first array
    return np.squeeze(data[list(data.keys())[0]]).astype(bool)


def compare_masks(gt_path: Path, pred_path: Path, method: str) -> Dict:
    """Compare two segmentation masks and compute metrics."""
    try:
        gt_mask = load_mask(gt_path)
        pred_mask = load_mask(pred_path)

        # Ensure masks have same shape
        if gt_mask.shape != pred_mask.shape:
            warnings.warn(f"Shape mismatch: {gt_path.name} - GT: {gt_mask.shape}, Pred: {pred_mask.shape}")
            return None

        metrics = {
            'method': method,
            'file': gt_path.name,
            'source_id': gt_path.parent.parent.name,
            'dice': compute_dice(pred_mask, gt_mask),
            'iou': compute_iou(pred_mask, gt_mask),
            'hausdorff_95': compute_hausdorff_95(pred_mask, gt_mask),
            'asd': compute_asd(pred_mask, gt_mask),
            'precision': compute_precision(pred_mask, gt_mask),
            'recall': compute_recall(pred_mask, gt_mask)
        }
        return metrics
    except Exception as e:
        warnings.warn(f"Error processing {gt_path.name}: {str(e)}")
        return None


def find_mask_files(root: Path, method: str, segment_step: str = "segment_cell") -> List[Path]:
    """Find all .npz mask files for a given method."""
    method_path = root / method / "steps"
    if not method_path.exists():
        return []

    mask_files = []
    for source_dir in method_path.iterdir():
        if source_dir.is_dir():
            segment_dir = source_dir / segment_step
            if segment_dir.exists():
                mask_files.extend(segment_dir.glob("*.npz"))

    return sorted(mask_files)


def match_files(gt_files: List[Path], method_files: List[Path]) -> List[Tuple[Path, Path]]:
    """Match ground truth and method files by name and source_id."""
    method_dict = {}
    for mf in method_files:
        source_id = mf.parent.parent.name
        key = (source_id, mf.name)
        method_dict[key] = mf

    matches = []
    for gf in gt_files:
        source_id = gf.parent.parent.name
        key = (source_id, gf.name)
        if key in method_dict:
            matches.append((gf, method_dict[key]))

    return matches


def process_file_pair(args):
    """Process a single file pair (for parallel execution)."""
    gt_path, pred_path, method = args
    return compare_masks(gt_path, pred_path, method)


def load_original_image(root: Path, method: str, source_id: str, mask_file: str, channels: List[int] = [0, 1, 2]) -> np.ndarray:
    """
    Load the original image corresponding to a mask file.
    Returns a 3-channel RGB representation using specified channels.
    """
    # Try to load from the source Zarr archive first
    zarr_store_path = Path("/work/datasets/jump_target2_4plate/zstd.zarr")
    result = load_zarr_image(zarr_store_path, source_id, channels)
    if result is not None:
        return result

    # Fallback: Try to find the original image in pipeline output directories
    # Get the base name without extension
    base_name = Path(mask_file).stem

    # Search in common locations
    possible_dirs = [
        root / method / "steps" / source_id / "load",
        root / method / "steps" / source_id,
        root / method / "steps" / source_id / "images"
    ]

    for search_dir in possible_dirs:
        if not search_dir.exists():
            continue

        # Try various extensions
        for ext in ['.npy', '.npz', '.tif', '.tiff', '.png']:
            img_path = search_dir / f"{base_name}{ext}"
            if img_path.exists():
                try:
                    if ext == '.npy':
                        img = np.load(img_path)
                    elif ext == '.npz':
                        data = np.load(img_path)
                        # Try common keys
                        for key in ['image', 'data', 'arr_0']:
                            if key in data:
                                img = data[key]
                                break
                        else:
                            img = data[list(data.keys())[0]]
                    else:
                        # Use matplotlib for image formats
                        img = plt.imread(img_path)

                    # Handle different image shapes
                    if img.ndim == 2:
                        # Grayscale - replicate to 3 channels
                        return np.stack([img, img, img], axis=-1)
                    elif img.ndim == 3:
                        # Multi-channel image
                        if img.shape[-1] <= 3:
                            # Already RGB or fewer channels
                            if img.shape[-1] == 3:
                                return img
                            else:
                                # Pad to 3 channels
                                padded = np.zeros((*img.shape[:2], 3))
                                padded[..., :img.shape[-1]] = img
                                return padded
                        else:
                            # More than 3 channels - select specified channels
                            selected = np.stack([img[..., ch] if ch < img.shape[-1] else np.zeros(img.shape[:2])
                                               for ch in channels], axis=-1)
                            # Normalize to 0-1 range
                            for i in range(3):
                                ch_data = selected[..., i]
                                ch_min, ch_max = ch_data.min(), ch_data.max()
                                if ch_max > ch_min:
                                    selected[..., i] = (ch_data - ch_min) / (ch_max - ch_min)
                            return selected
                    elif img.ndim == 4:
                        # Batch dimension - take first image
                        img = img[0]
                        if img.shape[-1] >= 3:
                            selected = np.stack([img[..., ch] if ch < img.shape[-1] else np.zeros(img.shape[:2])
                                               for ch in channels], axis=-1)
                            for i in range(3):
                                ch_data = selected[..., i]
                                ch_min, ch_max = ch_data.min(), ch_data.max()
                                if ch_max > ch_min:
                                    selected[..., i] = (ch_data - ch_min) / (ch_max - ch_min)
                            return selected
                except Exception as e:
                    warnings.warn(f"Failed to load image from {img_path}: {e}")
                    continue

    # If no image found, return a placeholder
    warnings.warn(f"Could not find original image for {mask_file} in {source_id}")
    return None


def visualize_samples(df: pd.DataFrame, root: Path, gt_method: str, methods: List[str], output_prefix: str, segment_step: str = "segment_cell"):
    """
    Find the best, median, and worst samples and create a visualization comparing all methods.
    Each row shows: original image (3 channels as RGB), ground truth, and method comparisons.
    """
    # Group by file and source_id, compute mean IoU across all methods
    file_metrics = df.groupby(['source_id', 'file'])['iou'].agg(['mean', 'std']).reset_index()
    file_metrics = file_metrics.sort_values('mean')

    if len(file_metrics) == 0:
        print("No files to visualize")
        return

    # Get best, median, and worst samples
    n_samples = len(file_metrics)
    worst_idx = 0
    median_idx = n_samples // 2
    best_idx = n_samples - 1

    samples = {
        'worst': file_metrics.iloc[worst_idx],
        'median': file_metrics.iloc[median_idx],
        'best': file_metrics.iloc[best_idx]
    }

    print(f"\n{'='*80}")
    print("Sample statistics:")
    for sample_name, sample_data in samples.items():
        print(f"  {sample_name.capitalize()}: Source={sample_data['source_id']}, "
              f"File={sample_data['file']}, Mean IoU={sample_data['mean']:.4f}")
    print(f"{'='*80}\n")

    # Create figure with 3 rows (worst, median, best) and columns for: original, GT, methods
    n_methods = len(methods)
    n_cols = 2 + n_methods  # original + GT + methods
    fig, axes = plt.subplots(3, n_cols, figsize=(4 * n_cols, 12))

    # Process each sample type
    for row_idx, (sample_name, sample_data) in enumerate([('worst', samples['worst']),
                                                           ('median', samples['median']),
                                                           ('best', samples['best'])]):
        source_id = sample_data['source_id']
        file_name = sample_data['file']
        mean_iou = sample_data['mean']

        # Load ground truth mask
        gt_path = root / gt_method / "steps" / source_id / segment_step / file_name
        if not gt_path.exists():
            print(f"Ground truth file not found: {gt_path}")
            continue

        gt_mask = load_mask(gt_path)

        # Load original image
        original_img = load_original_image(root, gt_method, source_id, file_name)

        # Column 0: Original image
        if original_img is not None:
            axes[row_idx, 0].imshow(original_img)
            axes[row_idx, 0].set_title('Original Image', fontsize=10, fontweight='bold')
        else:
            axes[row_idx, 0].text(0.5, 0.5, 'Image not found', ha='center', va='center')
            axes[row_idx, 0].set_title('Original Image', fontsize=10, fontweight='bold')
        axes[row_idx, 0].axis('off')

        # Column 1: Ground truth
        axes[row_idx, 1].imshow(gt_mask, cmap='gray')
        axes[row_idx, 1].set_title(f'Ground Truth\n{gt_method}', fontsize=10, fontweight='bold')
        axes[row_idx, 1].axis('off')

        # Remaining columns: Method comparisons
        for method_idx, method in enumerate(methods):
            col_idx = 2 + method_idx
            method_path = root / method / "steps" / source_id / segment_step / file_name

            if method_path.exists():
                method_mask = load_mask(method_path)

                # Get IoU for this specific file
                iou_row = df[(df['method'] == method) &
                           (df['source_id'] == source_id) &
                           (df['file'] == file_name)]
                iou_score = iou_row['iou'].values[0] if len(iou_row) > 0 else np.nan

                # Create RGB overlay showing agreement/disagreement
                overlay = np.zeros((*gt_mask.shape, 3))

                # Green: True positives (both GT and method agree on foreground)
                overlay[gt_mask & method_mask] = [0, 1, 0]

                # Red: False positives (method says foreground, GT says background)
                overlay[~gt_mask & method_mask] = [1, 0, 0]

                # Blue: False negatives (GT says foreground, method says background)
                overlay[gt_mask & ~method_mask] = [0, 0, 1]

                axes[row_idx, col_idx].imshow(overlay)
                method_display = method.replace('.zarr', '').replace('jpegxl_lossy_', '')
                axes[row_idx, col_idx].set_title(f'{method_display}\nIoU: {iou_score:.4f}', fontsize=10)
                axes[row_idx, col_idx].axis('off')
            else:
                axes[row_idx, col_idx].text(0.5, 0.5, 'Not found', ha='center', va='center')
                axes[row_idx, col_idx].axis('off')

        # Add row label
        axes[row_idx, 0].text(-0.1, 0.5, f'{sample_name.upper()}\nMean IoU: {mean_iou:.4f}',
                              transform=axes[row_idx, 0].transAxes,
                              fontsize=12, fontweight='bold', va='center', ha='right',
                              rotation=90)

    # Add legend
    legend_elements = [
        mpatches.Patch(color='green', label='True Positive'),
        mpatches.Patch(color='red', label='False Positive'),
        mpatches.Patch(color='blue', label='False Negative')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=11)

    plt.suptitle('Segmentation Comparison: Best, Median, and Worst Samples',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save figure
    output_path = f"{output_prefix}_samples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Sample visualization saved to: {output_path}")
    plt.close()


def visualize_single_sample(root: Path, gt_method: str, method: str, source_id: str,
                            output_prefix: str, segment_step: str = "segment_cell",
                            file_name: str = None):
    """
    Visualize a single well+compression comparison: original image and segmentation overlay
    showing true positives (green), false positives (red), and false negatives (blue).

    Args:
        root: Root directory containing all methods
        gt_method: Ground truth method name (e.g., 'zstd.zarr')
        method: Compression method to compare (e.g., 'jpegxl_lossy_lq.zarr')
        source_id: Well/source identifier to visualize
        output_prefix: Output file prefix for saving the plot
        segment_step: Segmentation step name (default: 'segment_cell')
        file_name: Specific mask file name. If None, uses the first file found.
    """
    gt_dir = root / gt_method / "steps" / source_id / segment_step
    method_dir = root / method / "steps" / source_id / segment_step

    if not gt_dir.exists():
        print(f"Ground truth directory not found: {gt_dir}")
        return
    if not method_dir.exists():
        print(f"Method directory not found: {method_dir}")
        return

    # Find mask file
    if file_name is None:
        gt_files = sorted(gt_dir.glob("*.npz"))
        if len(gt_files) == 0:
            print(f"No mask files found in {gt_dir}")
            return
        file_name = gt_files[0].name
        print(f"Using mask file: {file_name}")

    gt_path = gt_dir / file_name
    method_path = method_dir / file_name

    if not gt_path.exists():
        print(f"Ground truth mask not found: {gt_path}")
        return
    if not method_path.exists():
        print(f"Method mask not found: {method_path}")
        return

    gt_mask = load_mask(gt_path)
    method_mask = load_mask(method_path)

    if gt_mask.shape != method_mask.shape:
        print(f"Shape mismatch: GT {gt_mask.shape} vs Method {method_mask.shape}")
        return

    # Compute IoU
    intersection = np.logical_and(gt_mask, method_mask).sum()
    union = np.logical_or(gt_mask, method_mask).sum()
    iou = intersection / union if union > 0 else 0.0

    # Load original image
    original_img = load_original_image(root, gt_method, source_id, file_name)

    # Build overlay: green=TP, red=FP, blue=FN
    overlay = np.zeros((*gt_mask.shape, 3))
    overlay[gt_mask & method_mask] = [0, 1, 0]       # True positive
    overlay[~gt_mask & method_mask] = [1, 0, 0]      # False positive
    overlay[gt_mask & ~method_mask] = [0, 0, 1]       # False negative

    method_display = method.replace('.zarr', '').replace('jpegxl_lossy_', 'jxl_')

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5),
                             gridspec_kw={'wspace': 0.05})

    # Panel 1: Original image
    if original_img is not None:
        axes[0].imshow(original_img)
    else:
        axes[0].text(0.5, 0.5, 'Image not found', ha='center', va='center',
                     transform=axes[0].transAxes, fontsize=12)
    axes[0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Segmentation comparison overlay
    axes[1].imshow(overlay)
    axes[1].set_title(f'{method_display} vs {gt_method.replace(".zarr", "")}  IoU: {iou:.4f}',
                      fontsize=11, fontweight='bold')
    axes[1].axis('off')

    # Legend
    legend_elements = [
        mpatches.Patch(color='green', label='True Positive'),
        mpatches.Patch(color='red', label='False Positive'),
        mpatches.Patch(color='blue', label='False Negative')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)

    output_path = f"{output_prefix}_{source_id}_{method_display}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved single sample visualization to: {output_path}")
    plt.close()


def _register_jpegxl_codec():
    """Register the JpegXL numcodecs codec if available."""
    try:
        import numcodecs
        from imagecodecs.numcodecs import Jpegxl
        numcodecs.register_codec(Jpegxl)
    except (ImportError, AttributeError, ValueError):
        pass

_register_jpegxl_codec()


def load_zarr_image(zarr_store_path: Path, source_id: str, channels: List[int] = [0, 1, 2]) -> np.ndarray:
    """Load an image from a zarr group and return a 3-channel float64 array.

    Args:
        zarr_store_path: Path to the zarr store (e.g., /data/jpegxl_lossy_hq.zarr)
        source_id: Key within the zarr group
        channels: Channel indices to use for RGB
    """
    if not zarr_store_path.exists():
        return None
    try:
        import zarr
        store = zarr.open(str(zarr_store_path), mode='r')
        if source_id not in store:
            warnings.warn(f"Source {source_id} not found in {zarr_store_path}")
            return None
        img = store[source_id][:]

        if img.ndim == 3 and img.shape[0] < img.shape[1]:
            selected = np.zeros((*img.shape[1:], 3), dtype=np.float64)
            for i, ch in enumerate(channels):
                if ch < img.shape[0]:
                    ch_data = img[ch].astype(np.float64)
                    ch_min = np.percentile(ch_data, 1)
                    ch_max = np.percentile(ch_data, 99)
                    if ch_max > ch_min:
                        ch_clipped = np.clip(ch_data, ch_min, ch_max)
                        selected[..., i] = (ch_clipped - ch_min) / (ch_max - ch_min)
            return selected
    except Exception as e:
        warnings.warn(f"Failed to load zarr image from {zarr_store_path}/{source_id}: {e}")
    return None


def visualize_compression_grid(root: Path, zarr_root: Path, gt_method: str, methods: List[str],
                               source_id: str, output_prefix: str, file_name: str = None):
    """
    Create a 2x5 grid showing a well across compression levels.
    Row 1: Original images (zstd + 4 lossy codecs).
    Row 2: Combined segmentation masks (cell in green, nuclei in blue).

    Args:
        root: Root directory containing CellProfiler output (segmentation masks)
        zarr_root: Root directory containing zarr source images (e.g., /work/datasets/jump_target2_4plate)
        gt_method: Ground truth method name (e.g., 'zstd.zarr')
        methods: List of compression methods to compare
        source_id: Well/source identifier to visualize
        output_prefix: Output file prefix for saving the plot
        file_name: Specific mask file name. If None, uses the first file found.
    """
    # Fixed order: best quality to worst
    method_order = [
        gt_method,
        "jpegxl_lossy_hq.zarr",
        "jpegxl_lossy_effort_3.zarr",
        "jpegxl_lossy_mq.zarr",
        "jpegxl_lossy_lq.zarr",
    ]
    all_methods = [m for m in method_order if m == gt_method or m in methods]

    # Find mask file name if not specified
    if file_name is None:
        cell_dir = root / gt_method / "steps" / source_id / "segment_cell"
        if cell_dir.exists():
            mask_files = sorted(cell_dir.glob("*.npz"))
            if len(mask_files) > 0:
                file_name = mask_files[0].name
                print(f"Using mask file: {file_name}")
            else:
                print(f"No mask files found in {cell_dir}")
                return
        else:
            print(f"Cell segmentation directory not found: {cell_dir}")
            return

    # Load all images once
    images = {}
    display_names = {}
    for method in all_methods:
        display = method.replace('.zarr', '').replace('jpegxl_lossy_', 'jxl_')
        display_names[method] = display
        zarr_store_path = zarr_root / method
        images[method] = load_zarr_image(zarr_store_path, source_id)

    # --- 2×N grid: images + segmentation ---
    n_cols = len(all_methods)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 7),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.15})

    for col, method in enumerate(all_methods):
        img = images[method]
        if img is not None:
            axes[0, col].imshow(img)
        else:
            axes[0, col].text(0.5, 0.5, 'Not found', ha='center', va='center',
                              transform=axes[0, col].transAxes, fontsize=10)
        axes[0, col].set_title(display_names[method], fontsize=10, fontweight='bold')
        axes[0, col].axis('off')

        # Row 2: Combined cell + nuclei segmentation
        cell_path = root / method / "steps" / source_id / "segment_cell" / file_name
        nuclei_path = root / method / "steps" / source_id / "segment_nuclei" / file_name

        cell_mask = load_mask(cell_path) if cell_path.exists() else None
        nuclei_mask = load_mask(nuclei_path) if nuclei_path.exists() else None

        if cell_mask is not None or nuclei_mask is not None:
            shape = cell_mask.shape if cell_mask is not None else nuclei_mask.shape
            overlay = np.zeros((*shape, 3))
            if cell_mask is not None:
                overlay[cell_mask, 1] = 1.0   # Green for cell
            if nuclei_mask is not None:
                overlay[nuclei_mask, 2] = 1.0  # Blue for nuclei
            axes[1, col].imshow(overlay)
        else:
            axes[1, col].text(0.5, 0.5, 'Not found', ha='center', va='center',
                              transform=axes[1, col].transAxes, fontsize=10)
        axes[1, col].axis('off')

    axes[0, 0].text(-0.05, 0.5, 'Image', transform=axes[0, 0].transAxes,
                    fontsize=12, fontweight='bold', va='center', ha='right', rotation=90)
    axes[1, 0].text(-0.05, 0.5, 'Segmentation', transform=axes[1, 0].transAxes,
                    fontsize=12, fontweight='bold', va='center', ha='right', rotation=90)

    legend_elements = [
        mpatches.Patch(color='green', label='Cell'),
        mpatches.Patch(color='blue', label='Nuclei'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, -0.02), fontsize=10)

    output_path = f"{output_prefix}_{source_id}_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved compression grid to: {output_path}")
    plt.close()

    # --- NxN difference grid ---
    visualize_diff_grid(images, all_methods, display_names, output_prefix, source_id)


def visualize_diff_grid(images: Dict, all_methods: List[str], display_names: Dict,
                        output_prefix: str, source_id: str):
    """
    Create an NxN grid showing pairwise image differences.
    Cell (row, col) shows image[row] - image[col].

    Args:
        images: Dict mapping method name to loaded image array (or None)
        all_methods: Ordered list of method names
        display_names: Dict mapping method name to display string
        output_prefix: Output file prefix
        source_id: Well/source identifier
    """
    # Diverging colormap: blue -> black -> red
    diff_cmap = mcolors.LinearSegmentedColormap.from_list(
        'BkRdBu', ['#2166ac', 'black', '#b2182b'])

    n = len(all_methods)
    fig, axes = plt.subplots(n, n, figsize=(3.5 * n, 3.5 * n),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.05})

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            img_row = images[all_methods[row]]
            img_col = images[all_methods[col]]

            if row == col:
                # Diagonal: show original image
                if img_row is not None:
                    ax.imshow(img_row)
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                            transform=ax.transAxes, fontsize=10)
            elif img_row is not None and img_col is not None:
                # Off-diagonal: grayscale difference with diverging colormap
                # Use per-channel diff to preserve structure, then average
                diff = np.mean(img_row - img_col, axis=2)
                # Scale to 99.5th percentile for contrast (clip outliers)
                p = np.percentile(np.abs(diff), 99.5)
                vbound = max(p, 1e-10)
                ax.imshow(np.clip(diff, -vbound, vbound), cmap=diff_cmap,
                          vmin=-vbound, vmax=vbound)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10)

            ax.axis('off')

            # Column headers on top row
            if row == 0:
                ax.set_title(display_names[all_methods[col]], fontsize=10, fontweight='bold')
            # Row labels on left column
            if col == 0:
                ax.text(-0.05, 0.5, display_names[all_methods[row]],
                        transform=ax.transAxes, fontsize=10, fontweight='bold',
                        va='center', ha='right', rotation=90)

    # Add a shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=diff_cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    fig.colorbar(sm, cax=cbar_ax, label='Normalized difference')

    output_path = f"{output_prefix}_{source_id}_diff.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved difference grid to: {output_path}")
    plt.close()


def plot_iou_boxplot(df, output_prefix, segment_step="segment_cell"):
    """
    Create boxplot showing IoU distribution across compression methods.

    Args:
        df: DataFrame with detailed segmentation results (must have 'method' and 'iou' columns)
        output_prefix: Output file prefix for saving the plot
        segment_step: Segmentation step name for the title
    """
    # Clean up method names for display
    df_plot = df.copy()
    df_plot['codec'] = df_plot['method'].str.replace('.zarr', '').str.replace('jpegxl_lossy_', 'jxl_')

    # Define codec order
    codec_order = ['jxl_lq', 'jxl_mq', 'jxl_effort_3', 'jxl_hq']
    df_plot = df_plot[df_plot['codec'].isin(codec_order)]

    # Calculate stats for labels
    codec_stats = df_plot.groupby('codec').agg(
        n_total=('iou', 'count'),
        median_iou=('iou', 'median')
    )

    codec_labels = {
        codec: f"{codec}\nn={codec_stats.loc[codec, 'n_total']}"
        for codec in codec_order if codec in codec_stats.index
    }
    label_order = [c for c in codec_order if c in codec_stats.index]

    fig, ax = plt.subplots(figsize=(6, 6))

    sns.violinplot(
        data=df_plot,
        x='codec',
        y='iou',
        order=label_order,
        palette='viridis',
        inner='box',
        cut=0,
        ax=ax
    )

    # sns.stripplot(
    #     data=df_plot,
    #     x='codec',
    #     y='iou',
    #     order=label_order,
    #     color='black',
    #     alpha=0.1,
    #     size=3,
    #     jitter=True,
    #     ax=ax
    # )

    ax.set_xlabel('Codec', fontsize=12, fontweight='bold')
    ax.set_ylabel('IoU', fontsize=12, fontweight='bold')
    step_nice = 'Cell' if 'cell' in segment_step else 'Nuclei'
    ax.set_title(f'{step_nice} Segmentation IoU Across Compression Methods',
                 fontsize=14, fontweight='bold')

    # Update x-axis labels to include sample counts
    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels([codec_labels[c] for c in label_order])

    # Add horizontal line at 1.0 for reference
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect overlap')

    plt.tight_layout()
    output_path = f"{output_prefix}_iou_boxplot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved IoU boxplot to: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare segmentation masks from different compression methods")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing all methods")
    parser.add_argument("--ground-truth", type=str, required=True, help="Ground truth method name (e.g., zstd.zarr)")
    parser.add_argument("--methods", nargs='+', required=True, help="List of methods to compare against ground truth")
    parser.add_argument("--output", type=str, default="segmentation_comparison", help="Output file prefix")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--segment-step", type=str, default="segment_cell", help="Segmentation step name (e.g., segment_cell, segment_nuclei)")
    parser.add_argument("--visualize-sample", action="store_true", help="Only visualize a single well+method comparison (requires --well and one --methods entry)")
    parser.add_argument("--well", type=str, default=None, help="Source/well ID for single sample visualization")
    parser.add_argument("--file", type=str, default=None, help="Specific mask file name for single sample visualization (default: first file found)")
    parser.add_argument("--zarr-root", type=str, default="/work/datasets/jump_target2_4plate", help="Root directory containing zarr source images")

    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise ValueError(f"Root directory does not exist: {root}")

    # Embed segment step in output prefix so cell/nuclei results don't overwrite each other
    args.output = f"{args.output}_{args.segment_step}"

    # Single sample visualization mode
    if args.visualize_sample:
        if args.well is None:
            raise ValueError("--well is required when using --visualize-sample")
        if len(args.methods) == 0:
            raise ValueError("At least one --methods entry is required")
        for method in args.methods:
            visualize_single_sample(
                root=root,
                gt_method=args.ground_truth,
                method=method,
                source_id=args.well,
                output_prefix=args.output,
                segment_step=args.segment_step,
                file_name=args.file,
            )
        visualize_compression_grid(
            root=root,
            zarr_root=Path(args.zarr_root),
            gt_method=args.ground_truth,
            methods=args.methods,
            source_id=args.well,
            output_prefix=args.output,
            file_name=args.file,
        )
        return

    # Check if output already exists
    detailed_output = f"{args.output}_detailed.csv"
    rerun = True
    if Path(detailed_output).exists():
        rerun_input = input(f"{detailed_output} exists. Rerun analysis? (y/n): ")
        if rerun_input.lower() != 'y':
            df = pd.read_csv(detailed_output)
            print(f"Loaded existing results from {detailed_output}")
            rerun = False

    if rerun:
        print(f"Loading ground truth from: {args.ground_truth}")
        print(f"Segmentation step: {args.segment_step}")
        gt_files = find_mask_files(root, args.ground_truth, args.segment_step)
        print(f"Found {len(gt_files)} ground truth mask files")

        if len(gt_files) == 0:
            raise ValueError(f"No ground truth files found in {root / args.ground_truth}")

        all_results = []

        for method in args.methods:
            print(f"\nProcessing method: {method}")
            method_files = find_mask_files(root, method, args.segment_step)
            print(f"Found {len(method_files)} mask files for {method}")

            matches = match_files(gt_files, method_files)
            print(f"Matched {len(matches)} file pairs")

            if len(matches) == 0:
                print(f"Warning: No matching files found for {method}")
                continue

            # Prepare arguments for parallel processing
            task_args = [(gt, pred, method) for gt, pred in matches][:]

            # Process in parallel
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(process_file_pair, task_arg) for task_arg in task_args]

                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Comparing {method}"):
                    result = future.result()
                    if result is not None:
                        all_results.append(result)

        if len(all_results) == 0:
            print("No results generated. Check your data paths.")
            return

        # Create detailed results DataFrame
        df = pd.DataFrame(all_results)

        # Save detailed results
        df.to_csv(detailed_output, index=False)
        print(f"\nDetailed results saved to: {detailed_output}")

    # Plot IoU boxplot
    plot_iou_boxplot(df, args.output, args.segment_step)

    # Compute summary statistics
    summary_data = []
    for method in args.methods:
        method_df = df[df['method'] == method]
        if len(method_df) > 0:
            summary_data.append({
                'method': method,
                'n_files': len(method_df),
                'dice_mean': method_df['dice'].mean(),
                'dice_std': method_df['dice'].std(),
                'dice_median': method_df['dice'].median(),
                'iou_mean': method_df['iou'].mean(),
                'iou_std': method_df['iou'].std(),
                'iou_median': method_df['iou'].median(),
                'hausdorff_95_mean': method_df['hausdorff_95'].mean(),
                'hausdorff_95_std': method_df['hausdorff_95'].std(),
                'hausdorff_95_median': method_df['hausdorff_95'].median(),
                'asd_mean': method_df['asd'].mean(),
                'asd_std': method_df['asd'].std(),
                'asd_median': method_df['asd'].median(),
                'precision_mean': method_df['precision'].mean(),
                'precision_std': method_df['precision'].std(),
                'precision_median': method_df['precision'].median(),
                'recall_mean': method_df['recall'].mean(),
                'recall_std': method_df['recall'].std(),
                'recall_median': method_df['recall'].median(),
            })

    summary_df = pd.DataFrame(summary_data)

    # Save summary results
    summary_output = f"{args.output}_summary.csv"
    summary_df.to_csv(summary_output, index=False)
    print(f"Summary results saved to: {summary_output}")

    # Print formatted summary table
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"\n{'Method':<30} {'N Files':<10} {'Dice':<15} {'IoU':<15}")
    print("-"*80)
    for _, row in summary_df.iterrows():
        print(f"{row['method']:<30} {row['n_files']:<10} "
              f"{row['dice_mean']:.4f}±{row['dice_std']:.4f}  "
              f"{row['iou_mean']:.4f}±{row['iou_std']:.4f}")

    print(f"\n{'Method':<30} {'Precision':<15} {'Recall':<15}")
    print("-"*80)
    for _, row in summary_df.iterrows():
        print(f"{row['method']:<30} "
              f"{row['precision_mean']:.4f}±{row['precision_std']:.4f}  "
              f"{row['recall_mean']:.4f}±{row['recall_std']:.4f}")

    print("="*80)

    # Visualize the best, median, and worst performing samples
    visualize_samples(df, root, args.ground_truth, args.methods, args.output, args.segment_step)


if __name__ == "__main__":
    main()
