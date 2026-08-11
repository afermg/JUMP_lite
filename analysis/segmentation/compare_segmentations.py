#!/usr/bin/env python3
"""
Compare segmentation masks from different compression methods against ground truth.
"""

import argparse
import os
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

# Import instance matching (adapted from StarDist)
# Source: https://github.com/stardist/stardist/blob/master/stardist/matching.py
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent))
from instance_matching import matching as stardist_matching
USE_STARDIST = True


SEGMENTATION_CODEC_DISPLAY = {
    'raw': 'Raw',
    'zstd': 'Raw',
    'jxl_raw': 'Raw',
    'jxl_hq': 'JXL-HQ',
    'jxl_effort_3': 'JXL-E3',
    'jxl_d2_e8': 'JXL-D2-E8',
    'jxl_mq': 'JXL-MQ',
    'jxl_lq': 'JXL-LQ',
    'jxl_d10': 'JXL-D10',
    'jxl_d15': 'JXL-D15',
    'jxl_d20': 'JXL-D20',
    'jxl_d20_e2': 'JXL-D20',
    'jxl_d25': 'JXL-D25',
    'jxl_d30': 'JXL-D25',
}


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


def load_instance_mask(npz_path: Path) -> np.ndarray:
    """Load segmentation mask keeping integer labels (0=background, 1,2,3...=cells)."""
    data = np.load(npz_path)
    for key in ['mask', 'segmentation', 'arr_0']:
        if key in data:
            return np.squeeze(data[key]).astype(np.int32)
    return np.squeeze(data[list(data.keys())[0]]).astype(np.int32)


def compare_masks(gt_path: Path, pred_path: Path, method: str, save_mappings: bool = False, fast: bool = False) -> Dict:
    """Compare two segmentation masks and compute metrics.

    Args:
        gt_path: Path to ground truth mask
        pred_path: Path to predicted mask
        method: Name of the compression method
        save_mappings: If True, include instance ID mappings in the result
        fast: If True, skip expensive metrics (hausdorff, asd) for ~2-3x speedup

    Returns:
        Dict with metrics and optionally 'mappings' key containing instance mappings
    """
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
            'precision': compute_precision(pred_mask, gt_mask),
            'recall': compute_recall(pred_mask, gt_mask)
        }

        # Add expensive metrics only if not in fast mode
        if not fast:
            metrics['hausdorff_95'] = compute_hausdorff_95(pred_mask, gt_mask)
            metrics['asd'] = compute_asd(pred_mask, gt_mask)

        mappings_list = []

        # Instance-level matching using StarDist
        if USE_STARDIST:
            gt_inst = load_instance_mask(gt_path)
            pred_inst = load_instance_mask(pred_path)

            # Cell counts
            gt_ids = set(np.unique(gt_inst)) - {0}  # exclude background
            pred_ids = set(np.unique(pred_inst)) - {0}
            metrics['n_true'] = len(gt_ids)
            metrics['n_pred'] = len(pred_ids)

            # Instance matching at multiple IoU thresholds
            for thresh in [0.5, 0.7, 0.8, 0.9]:
                try:
                    m = stardist_matching(gt_inst, pred_inst, thresh=thresh, report_matches=save_mappings)
                    suffix = f'_{int(thresh*100)}'
                    metrics[f'inst_tp{suffix}'] = m.tp
                    metrics[f'inst_fp{suffix}'] = m.fp
                    metrics[f'inst_fn{suffix}'] = m.fn
                    metrics[f'inst_precision{suffix}'] = m.precision
                    metrics[f'inst_recall{suffix}'] = m.recall
                    metrics[f'inst_f1{suffix}'] = m.f1
                    metrics[f'inst_accuracy{suffix}'] = m.accuracy
                    metrics[f'inst_panoptic_quality{suffix}'] = m.panoptic_quality

                    # Extract instance mappings if requested
                    if save_mappings and hasattr(m, 'matched_pairs'):
                        matched_gt_ids = set()
                        matched_pred_ids = set()

                        for idx, (gt_id, pred_id) in enumerate(m.matched_pairs):
                            is_tp = idx in m.matched_tps
                            iou_score = m.matched_scores[idx] if idx < len(m.matched_scores) else 0.0
                            mappings_list.append({
                                'source_id': gt_path.parent.parent.name,
                                'file': gt_path.name,
                                'method': method,
                                'thresh': thresh,
                                'gt_id': int(gt_id),
                                'pred_id': int(pred_id),
                                'iou_score': float(iou_score),
                                'match_type': 'TP' if is_tp else 'BELOW_THRESH'
                            })
                            matched_gt_ids.add(gt_id)
                            matched_pred_ids.add(pred_id)

                        # Add FN entries (GT instances with no match)
                        for gt_id in gt_ids - matched_gt_ids:
                            mappings_list.append({
                                'source_id': gt_path.parent.parent.name,
                                'file': gt_path.name,
                                'method': method,
                                'thresh': thresh,
                                'gt_id': int(gt_id),
                                'pred_id': None,
                                'iou_score': 0.0,
                                'match_type': 'FN'
                            })

                        # Add FP entries (Pred instances with no match)
                        for pred_id in pred_ids - matched_pred_ids:
                            mappings_list.append({
                                'source_id': gt_path.parent.parent.name,
                                'file': gt_path.name,
                                'method': method,
                                'thresh': thresh,
                                'gt_id': None,
                                'pred_id': int(pred_id),
                                'iou_score': 0.0,
                                'match_type': 'FP'
                            })

                except Exception as e:
                    warnings.warn(f"Instance matching failed for {gt_path.name} at thresh={thresh}: {e}")

        if save_mappings:
            metrics['mappings'] = mappings_list

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
    gt_path, pred_path, method, save_mappings, fast = args
    return compare_masks(gt_path, pred_path, method, save_mappings=save_mappings, fast=fast)


def load_original_image(root: Path, method: str, source_id: str, mask_file: str, channels: List[int] = [0, 1, 2]) -> np.ndarray:
    """
    Load the original image corresponding to a mask file.
    Returns a 3-channel RGB representation using specified channels.
    """
    # Try to load from the source Zarr archive first
    zarr_store_path = Path(os.environ.get("ZARR_ROOT", "data/jump_target2_4plate")) / "zstd.zarr"
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

    # Nice display names for compression quality
    codec_nice_names = {
        'jxl_lq': 'Low',
        'jxl_mq': 'Medium',
        'jxl_effort_3': 'Mid-High',
        'jxl_hq': 'High'
    }
    method_display = method.replace('.zarr', '').replace('jpegxl_lossy_', 'jxl_')
    method_nice = codec_nice_names.get(method_display, method_display)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             gridspec_kw={'wspace': 0.05})

    # Panel 1: Original image
    if original_img is not None:
        axes[0].imshow(original_img)
    else:
        axes[0].text(0.5, 0.5, 'Image not found', ha='center', va='center',
                     transform=axes[0].transAxes, fontsize=20)
    axes[0].set_title('Original', fontsize=26, fontweight='bold')
    axes[0].axis('off')

    # Panel 2: Segmentation comparison overlay
    axes[1].imshow(overlay)
    axes[1].set_title(f'{method_nice} Quality - IoU: {iou:.4f}',
                      fontsize=26, fontweight='bold')
    axes[1].axis('off')

    # Legend
    legend_elements = [
        mpatches.Patch(color='green', label='True Positive'),
        mpatches.Patch(color='red', label='False Positive'),
        mpatches.Patch(color='blue', label='False Negative')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=18)

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
        zarr_root: Root directory containing zarr source images (e.g., data/jump_target2_4plate)
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


def plot_iou_boxplot_combined(df_cell, df_nuclei, output_prefix):
    """
    Create violin and boxen plots showing IoU and Dice distributions for both cell and nuclei segmentation.

    Args:
        df_cell: DataFrame with cell segmentation results
        df_nuclei: DataFrame with nuclei segmentation results
        output_prefix: Output file prefix for saving the plot
    """
    # Add segmentation type column and combine
    df_cell = df_cell.copy()
    df_nuclei = df_nuclei.copy()
    df_cell['segmentation'] = 'Cell'
    df_nuclei['segmentation'] = 'Nuclei'
    df_combined = pd.concat([df_cell, df_nuclei], ignore_index=True)

    # Clean up method names for display
    df_combined['codec'] = df_combined['method'].str.replace('.zarr', '').str.replace('jpegxl_lossy_', 'jxl_')

    # Use all codecs, ordered by mean IoU (lowest to highest quality)
    codec_mean_iou = df_combined.groupby('codec')['iou'].mean().sort_values(ascending=False)
    label_order = list(codec_mean_iou.index)
    df_plot = df_combined

    # Nice display names
    codec_labels = {
        c: SEGMENTATION_CODEC_DISPLAY.get(c, c.replace('jxl_', '').upper())
        for c in label_order
    }

    # --- IoU Violin Plot ---
    fig, ax = plt.subplots(figsize=(7, 7))

    sns.violinplot(
        data=df_plot,
        x='codec',
        y='iou',
        hue='segmentation',
        order=label_order,
        hue_order=['Cell', 'Nuclei'],
        palette=['tab:green', 'tab:blue'],
        inner='box',
        cut=0,
        ax=ax
    )

    ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
    ax.set_ylabel('IoU', fontsize=24, fontweight='bold')
    ax.set_title('IoU - Segmentation Similarity', fontsize=26, fontweight='bold')

    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20, rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=20)

    ax.legend(title='Segmentation', fontsize=18, title_fontsize=18, loc='lower left')

    plt.tight_layout()
    output_path = f"{output_prefix}_iou_violinplot_combined.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined IoU violinplot to: {output_path}")
    plt.close()

    # --- IoU Violin Plot with 95th percentile highlighted ---
    fig, ax = plt.subplots(figsize=(7, 7))

    sns.violinplot(
        data=df_plot,
        x='codec',
        y='iou',
        hue='segmentation',
        order=label_order,
        hue_order=['Cell', 'Nuclei'],
        palette=['tab:green', 'tab:blue'],
        inner='box',
        cut=0,
        ax=ax
    )

    ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
    ax.set_ylabel('IoU', fontsize=24, fontweight='bold')
    ax.set_title('IoU - Segmentation Similarity (95th percentile)', fontsize=26, fontweight='bold')

    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20, rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=20)

    ax.legend(title='Segmentation', fontsize=18, title_fontsize=18, loc='lower left')

    # Add 5th and 95th percentile markers
    hue_order = ['Cell', 'Nuclei']
    colors = {'Cell': 'darkgreen', 'Nuclei': 'darkblue'}
    offset = 0.2  # offset for side-by-side violins

    for i, codec in enumerate(label_order):
        for j, seg in enumerate(hue_order):
            subset = df_plot[(df_plot['codec'] == codec) & (df_plot['segmentation'] == seg)]
            if len(subset) > 0:
                p5 = np.percentile(subset['iou'], 5)
                p95 = np.percentile(subset['iou'], 95)
                x_pos = i + (j - 0.5) * offset * 2
                # 95th percentile marker
                ax.scatter([x_pos], [p95], marker='_', s=300, linewidths=3,
                          color=colors[seg], zorder=10)
                ax.annotate(f'{p95:.3f}', (x_pos, p95), textcoords='offset points',
                           xytext=(0, 8), ha='center', fontsize=10, fontweight='bold',
                           color=colors[seg])
                # 5th percentile marker
                ax.scatter([x_pos], [p5], marker='_', s=300, linewidths=3,
                          color=colors[seg], zorder=10)
                ax.annotate(f'{p5:.3f}', (x_pos, p5), textcoords='offset points',
                           xytext=(0, -14), ha='center', fontsize=10, fontweight='bold',
                           color=colors[seg])

    plt.tight_layout()
    output_path = f"{output_prefix}_iou_violinplot_combined_p5_p95.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined IoU violinplot with 5th and 95th percentile to: {output_path}")
    plt.close()

    # --- IoU Boxen Plot ---
    fig, ax = plt.subplots(figsize=(7, 7))

    sns.boxenplot(
        data=df_plot,
        x='codec',
        y='iou',
        hue='segmentation',
        order=label_order,
        hue_order=['Cell', 'Nuclei'],
        palette=['tab:green', 'tab:blue'],
        ax=ax
    )

    ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
    ax.set_ylabel('IoU', fontsize=24, fontweight='bold')
    ax.set_title('IoU - Segmentation Similarity', fontsize=26, fontweight='bold')

    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20, rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=20)

    ax.legend(title='Segmentation', fontsize=18, title_fontsize=18, loc='lower left')

    plt.tight_layout()
    output_path = f"{output_prefix}_iou_boxenplot_combined.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined IoU boxenplot to: {output_path}")
    plt.close()

    # --- Dice Violin Plot ---
    fig, ax = plt.subplots(figsize=(7, 7))

    sns.violinplot(
        data=df_plot,
        x='codec',
        y='dice',
        hue='segmentation',
        order=label_order,
        hue_order=['Cell', 'Nuclei'],
        palette=['tab:green', 'tab:blue'],
        inner='box',
        cut=0,
        ax=ax
    )

    ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
    ax.set_ylabel('Dice Score', fontsize=24, fontweight='bold')
    ax.set_title('Dice - Segmentation Similarity', fontsize=26, fontweight='bold')

    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20, rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=20)

    ax.legend(title='Segmentation', fontsize=18, title_fontsize=18, loc='lower left')

    plt.tight_layout()
    output_path = f"{output_prefix}_dice_violinplot_combined.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined Dice violinplot to: {output_path}")
    plt.close()

    # --- Dice Violin Plot with 5th and 95th percentile highlighted ---
    fig, ax = plt.subplots(figsize=(7, 7))

    sns.violinplot(
        data=df_plot,
        x='codec',
        y='dice',
        hue='segmentation',
        order=label_order,
        hue_order=['Cell', 'Nuclei'],
        palette=['tab:green', 'tab:blue'],
        inner='box',
        cut=0,
        ax=ax
    )

    ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
    ax.set_ylabel('Dice Score', fontsize=24, fontweight='bold')
    ax.set_title('Dice - Segmentation Similarity (5th & 95th percentile)', fontsize=26, fontweight='bold')

    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20, rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=20)

    ax.legend(title='Segmentation', fontsize=18, title_fontsize=18, loc='lower left')

    # Add 5th and 95th percentile markers
    for i, codec in enumerate(label_order):
        for j, seg in enumerate(hue_order):
            subset = df_plot[(df_plot['codec'] == codec) & (df_plot['segmentation'] == seg)]
            if len(subset) > 0:
                p5 = np.percentile(subset['dice'], 5)
                p95 = np.percentile(subset['dice'], 95)
                x_pos = i + (j - 0.5) * offset * 2
                # 95th percentile marker
                ax.scatter([x_pos], [p95], marker='_', s=300, linewidths=3,
                          color=colors[seg], zorder=10)
                ax.annotate(f'{p95:.3f}', (x_pos, p95), textcoords='offset points',
                           xytext=(0, 8), ha='center', fontsize=10, fontweight='bold',
                           color=colors[seg])
                # 5th percentile marker
                ax.scatter([x_pos], [p5], marker='_', s=300, linewidths=3,
                          color=colors[seg], zorder=10)
                ax.annotate(f'{p5:.3f}', (x_pos, p5), textcoords='offset points',
                           xytext=(0, -14), ha='center', fontsize=10, fontweight='bold',
                           color=colors[seg])

    plt.tight_layout()
    output_path = f"{output_prefix}_dice_violinplot_combined_p5_p95.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined Dice violinplot with 5th and 95th percentile to: {output_path}")
    plt.close()

    # --- Dice Boxen Plot ---
    fig, ax = plt.subplots(figsize=(7, 7))

    sns.boxenplot(
        data=df_plot,
        x='codec',
        y='dice',
        hue='segmentation',
        order=label_order,
        hue_order=['Cell', 'Nuclei'],
        palette=['tab:green', 'tab:blue'],
        ax=ax
    )

    ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
    ax.set_ylabel('Dice Score', fontsize=24, fontweight='bold')
    ax.set_title('Dice - Segmentation Similarity', fontsize=26, fontweight='bold')

    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20, rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=20)

    ax.legend(title='Segmentation', fontsize=18, title_fontsize=18, loc='lower left')

    plt.tight_layout()
    output_path = f"{output_prefix}_dice_boxenplot_combined.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined Dice boxenplot to: {output_path}")
    plt.close()

    # --- Instance F1 plots (if stardist metrics available) ---
    if 'inst_f1_50' in df_plot.columns:
        # Instance F1 at IoU=0.5 violin plot
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.violinplot(
            data=df_plot,
            x='codec',
            y='inst_f1_50',
            hue='segmentation',
            order=label_order,
            hue_order=['Cell', 'Nuclei'],
            palette=['tab:green', 'tab:blue'],
            inner='box',
            cut=0,
            ax=ax
        )
        ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
        ax.set_ylabel('Instance F1 (IoU≥0.5)', fontsize=24, fontweight='bold')
        ax.set_title('Instance Matching F1 Score', fontsize=26, fontweight='bold')
        ax.set_xticks(range(len(label_order)))
        ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20, rotation=45, ha='right')
        ax.tick_params(axis='both', labelsize=20)
        ax.legend(title='Segmentation', fontsize=18, title_fontsize=18, loc='lower left')
        plt.tight_layout()
        output_path = f"{output_prefix}_inst_f1_50_violinplot_combined.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved combined Instance F1 (IoU≥0.5) violinplot to: {output_path}")
        plt.close()

        # Instance F1 at IoU=0.7 violin plot
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.violinplot(
            data=df_plot,
            x='codec',
            y='inst_f1_70',
            hue='segmentation',
            order=label_order,
            hue_order=['Cell', 'Nuclei'],
            palette=['tab:green', 'tab:blue'],
            inner='box',
            cut=0,
            ax=ax
        )
        ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
        ax.set_ylabel('Instance F1 (IoU≥0.7)', fontsize=24, fontweight='bold')
        ax.set_title('Instance Matching F1 Score', fontsize=26, fontweight='bold')
        ax.set_xticks(range(len(label_order)))
        ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20, rotation=45, ha='right')
        ax.tick_params(axis='both', labelsize=20)
        ax.legend(title='Segmentation', fontsize=18, title_fontsize=18, loc='lower left')
        plt.tight_layout()
        output_path = f"{output_prefix}_inst_f1_70_violinplot_combined.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved combined Instance F1 (IoU≥0.7) violinplot to: {output_path}")
        plt.close()

        # Average Precision across IoU thresholds (COCO-style AP)
        prec_cols = [c for c in df_plot.columns if c.startswith('inst_precision_')]
        if len(prec_cols) >= 2:
            df_plot['inst_ap'] = df_plot[prec_cols].mean(axis=1)
            thresholds_str = ', '.join(c.replace('inst_precision_', '0.') for c in sorted(prec_cols))

            fig, ax = plt.subplots(figsize=(7, 7))
            sns.violinplot(
                data=df_plot,
                x='codec',
                y='inst_ap',
                hue='segmentation',
                order=label_order,
                hue_order=['Cell', 'Nuclei'],
                palette=['tab:green', 'tab:blue'],
                inner='box',
                cut=0,
                ax=ax
            )
            ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
            ax.set_ylabel('Average Precision', fontsize=24, fontweight='bold')
            ax.set_title('Instance Matching AP', fontsize=26, fontweight='bold')
            ax.set_xticks(range(len(label_order)))
            ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20, rotation=45, ha='right')
            ax.tick_params(axis='both', labelsize=20)
            ax.legend(title='Segmentation', fontsize=18, title_fontsize=18, loc='lower left')
            plt.tight_layout()
            output_path = f"{output_prefix}_inst_ap_violinplot_combined.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved combined Instance AP ({thresholds_str}) violinplot to: {output_path}")
            plt.close()

        # Instance AP@0.5 — TP/(TP+FP+FN) at IoU=0.5
        if 'inst_accuracy_50' in df_plot.columns:
            fig, ax = plt.subplots(figsize=(7, 7))
            sns.boxenplot(
                data=df_plot,
                x='codec',
                y='inst_accuracy_50',
                hue='segmentation',
                order=label_order,
                hue_order=['Cell', 'Nuclei'],
                palette=['tab:green', 'tab:blue'],
                ax=ax
            )
            ax.axhline(y=0.743, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
            ax.set_xlabel('', fontsize=24, fontweight='bold')
            ax.set_ylabel('AP @ IoU=0.5', fontsize=24, fontweight='bold')
            ax.set_title('', fontsize=24, fontweight='bold')
            ax.set_xticks(range(len(label_order)))
            ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20, rotation=45, ha='right')
            ax.tick_params(axis='both', labelsize=20)
            ax.legend(title='Segmentation', fontsize=18, title_fontsize=18, loc='lower left')
            plt.tight_layout()
            output_path = f"{output_prefix}_inst_ap_iou50_boxenplot_combined.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved Instance AP@0.5 boxenplot to: {output_path}")
            plt.close()

        # Cell count comparison (pred - true)
        df_plot['cell_count_diff'] = df_plot['n_pred'] - df_plot['n_true']
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.boxplot(
            data=df_plot,
            x='codec',
            y='cell_count_diff',
            hue='segmentation',
            order=label_order,
            hue_order=['Cell', 'Nuclei'],
            palette=['tab:green', 'tab:blue'],
            ax=ax
        )
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
        ax.set_ylabel('Cell Count Difference (pred - GT)', fontsize=24, fontweight='bold')
        ax.set_title('Cell Count Comparison', fontsize=26, fontweight='bold')
        ax.set_ylim(-50, 50)
        ax.set_xticks(range(len(label_order)))
        ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20, rotation=45, ha='right')
        ax.tick_params(axis='both', labelsize=20)
        ax.legend(title='Segmentation', fontsize=18, title_fontsize=18, loc='lower left')
        plt.tight_layout()
        output_path = f"{output_prefix}_cell_count_diff_boxplot_combined.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved combined cell count difference boxplot to: {output_path}")
        plt.close()

        # Save samples with large cell count differences (>10 or <-10)
        df_large_diff = df_plot[abs(df_plot['cell_count_diff']) > 10].copy()
        if len(df_large_diff) > 0:
            # Sort by absolute difference (largest first)
            df_large_diff['abs_diff'] = abs(df_large_diff['cell_count_diff'])
            df_large_diff = df_large_diff.sort_values('abs_diff', ascending=False)

            csv_output = f"{output_prefix}_large_cell_count_diff.csv"
            df_large_diff.to_csv(csv_output, index=False)
            print(f"Saved {len(df_large_diff)} samples with |cell_count_diff| > 10 to: {csv_output}")
            print(f"  Breakdown by segmentation type:")
            for seg_type in ['Cell', 'Nuclei']:
                n = len(df_large_diff[df_large_diff['segmentation'] == seg_type])
                print(f"    {seg_type}: {n}")
        else:
            print("No samples found with |cell_count_diff| > 10")

        # Mean cell count per group (box plot)
        # Reshape data to have 'count_type' (GT vs Pred) and 'count' columns
        df_gt = df_plot[['codec', 'segmentation', 'n_true']].copy()
        df_gt['count_type'] = 'Ground Truth'
        df_gt = df_gt.rename(columns={'n_true': 'count'})
        df_pred = df_plot[['codec', 'segmentation', 'n_pred']].copy()
        df_pred['count_type'] = 'Predicted'
        df_pred = df_pred.rename(columns={'n_pred': 'count'})
        df_counts = pd.concat([df_gt, df_pred], ignore_index=True)

        fig, ax = plt.subplots(figsize=(10, 7))
        sns.boxplot(
            data=df_counts,
            x='codec',
            y='count',
            hue='count_type',
            order=label_order,
            hue_order=['Ground Truth', 'Predicted'],
            palette=['tab:gray', 'tab:orange'],
            ax=ax
        )
        ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
        ax.set_ylabel('Cell Count', fontsize=24, fontweight='bold')
        ax.set_title('Mean Cell Count per Compression Quality', fontsize=26, fontweight='bold')
        ax.set_xticks(range(len(label_order)))
        ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20, rotation=45, ha='right')
        ax.tick_params(axis='both', labelsize=20)
        ax.legend(title='Count Type', fontsize=18, title_fontsize=18, loc='upper right')
        plt.tight_layout()
        output_path = f"{output_prefix}_cell_count_boxplot_combined.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved combined cell count boxplot to: {output_path}")
        plt.close()

        # Panoptic Quality violin plot (IoU=0.5)
        fig, ax = plt.subplots(figsize=(7, 7))
        sns.violinplot(
            data=df_plot,
            x='codec',
            y='inst_panoptic_quality_50',
            hue='segmentation',
            order=label_order,
            hue_order=['Cell', 'Nuclei'],
            palette=['tab:green', 'tab:blue'],
            inner='box',
            cut=0,
            ax=ax
        )
        ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
        ax.set_ylabel('Panoptic Quality (IoU≥0.5)', fontsize=24, fontweight='bold')
        ax.set_title('Panoptic Quality Score', fontsize=26, fontweight='bold')
        ax.set_xticks(range(len(label_order)))
        ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20, rotation=45, ha='right')
        ax.tick_params(axis='both', labelsize=20)
        ax.legend(title='Segmentation', fontsize=18, title_fontsize=18, loc='lower left')
        plt.tight_layout()
        output_path = f"{output_prefix}_panoptic_quality_50_violinplot_combined.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved combined Panoptic Quality (IoU≥0.5) violinplot to: {output_path}")
        plt.close()


def plot_cell_level_iou_combined(df_cell: pd.DataFrame, df_nuclei: pd.DataFrame,
                                  output_base: str, mappings_dir: Path, thresh: float = 0.5):
    """
    Create violin plots showing cell-level (instance-level) IoU distributions.
    Only includes wells that are present in the filtered df_cell and df_nuclei dataframes.

    Args:
        df_cell: Filtered DataFrame with cell segmentation well-level results
        df_nuclei: Filtered DataFrame with nuclei segmentation well-level results
        output_base: Output file prefix
        mappings_dir: Directory containing instance mapping parquet files
        thresh: IoU threshold to use for filtering (default: 0.5)
    """
    if not mappings_dir.exists():
        print(f"Warning: Mappings directory not found: {mappings_dir}")
        print("Skipping cell-level IoU plots")
        return

    print("\n" + "="*80)
    print("Loading instance mappings for cell-level IoU plots...")
    print("="*80)

    # Get the set of wells to include (from filtered well-level results)
    cell_wells = set(zip(df_cell['source_id'], df_cell['file']))
    nuclei_wells = set(zip(df_nuclei['source_id'], df_nuclei['file']))

    # Cache paths for combined instance mappings
    cache_cell = mappings_dir / "combined_cell_instances.parquet"
    cache_nuclei = mappings_dir / "combined_nuclei_instances.parquet"

    # Load and filter instance mappings
    def load_and_filter_mappings(segment_step: str, valid_wells: set, cache_path: Path) -> pd.DataFrame:
        # Try loading from cache first
        if cache_path.exists():
            print(f"  Loading cached {segment_step} from {cache_path.name}")
            combined = pd.read_parquet(cache_path)
            # Filter to valid wells
            combined['well_key'] = list(zip(combined['source_id'], combined['file']))
            combined = combined[combined['well_key'].isin(valid_wells)]
            combined = combined.drop(columns=['well_key'])
            print(f"  Loaded {len(combined):,} cell instances from cache for {segment_step}")
            return combined

        parquet_files = list(mappings_dir.glob(f"{segment_step}_*.parquet"))
        if len(parquet_files) == 0:
            print(f"  No parquet files found for {segment_step}")
            return None

        dfs = []
        for pq_file in parquet_files:
            method = pq_file.stem.replace(f"{segment_step}_", "")
            df = pd.read_parquet(pq_file)
            df['method'] = f"{method}.zarr"
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)

        # Save combined cache before filtering
        combined.to_parquet(cache_path, index=False)
        print(f"  Saved combined cache to {cache_path.name}")

        # Filter to only wells in the valid set
        combined['well_key'] = list(zip(combined['source_id'], combined['file']))
        combined = combined[combined['well_key'].isin(valid_wells)]
        combined = combined.drop(columns=['well_key'])

        print(f"  Loaded {len(combined):,} cell instances from {len(parquet_files)} files for {segment_step}")
        return combined

    df_cell_inst = load_and_filter_mappings("segment_cell", cell_wells, cache_cell)
    df_nuclei_inst = load_and_filter_mappings("segment_nuclei", nuclei_wells, cache_nuclei)

    if df_cell_inst is None or df_nuclei_inst is None:
        print("Skipping cell-level IoU plots")
        return

    # Prepare data for both plot types — derive codec order from all instance data
    _all_codecs = pd.concat([df_cell_inst, df_nuclei_inst])
    _all_codecs['codec'] = _all_codecs['method'].str.replace('.zarr', '').str.replace('jpegxl_lossy_', 'jxl_')
    _codec_mean = _all_codecs.groupby('codec')['iou_score'].mean().sort_values(ascending=False)
    codec_order = list(_codec_mean.index)
    codec_labels = {
        c: SEGMENTATION_CODEC_DISPLAY.get(c, c.replace('jxl_', '').upper())
        for c in codec_order
    }

    # Create two versions: TP only and All matched (TP + BELOW_THRESH)
    plot_configs = [
        {
            'filter': ['TP'],
            'title_suffix': '',
            'file_suffix': 'tp_only',
            'description': 'TP cells only (IoU >= 0.5)'
        },
        {
            'filter': ['TP', 'BELOW_THRESH'],
            'title_suffix': 'All Matched Cells',
            'file_suffix': 'all_matched',
            'description': 'All matched cells (TP + BELOW_THRESH)'
        }
    ]

    for config in plot_configs:
        # Filter to specified match types at the specified threshold
        df_cell_filt = df_cell_inst[(df_cell_inst['thresh'] == thresh) &
                                      (df_cell_inst['match_type'].isin(config['filter']))].copy()
        df_nuclei_filt = df_nuclei_inst[(df_nuclei_inst['thresh'] == thresh) &
                                          (df_nuclei_inst['match_type'].isin(config['filter']))].copy()

        # Add segmentation type column
        df_cell_filt['segmentation'] = 'Cell'
        df_nuclei_filt['segmentation'] = 'Nuclei'
        df_combined = pd.concat([df_cell_filt, df_nuclei_filt], ignore_index=True)

        # Compute Dice score from IoU: Dice = 2*IoU / (1 + IoU)
        df_combined['dice_score'] = 2 * df_combined['iou_score'] / (1 + df_combined['iou_score'])

        # Clean up method names
        df_combined['codec'] = df_combined['method'].str.replace('.zarr', '').str.replace('jpegxl_lossy_', 'jxl_')

        df_plot = df_combined
        label_order = [c for c in codec_order if c in df_plot['codec'].unique()]

        print(f"\nCell-level statistics (threshold={thresh}, {config['description']}):")
        for seg in ['Cell', 'Nuclei']:
            print(f"\n{seg} Segmentation:")
            for codec in label_order:
                subset = df_plot[(df_plot['codec'] == codec) & (df_plot['segmentation'] == seg)]
                if len(subset) > 0:
                    print(f"  {codec_labels[codec]}: n={len(subset):,}")
                    print(f"    IoU:  mean={subset['iou_score'].mean():.4f}, "
                          f"median={subset['iou_score'].median():.4f}, "
                          f"p5={np.percentile(subset['iou_score'], 5):.4f}, "
                          f"p95={np.percentile(subset['iou_score'], 95):.4f}")
                    print(f"    Dice: mean={subset['dice_score'].mean():.4f}, "
                          f"median={subset['dice_score'].median():.4f}, "
                          f"p5={np.percentile(subset['dice_score'], 5):.4f}, "
                          f"p95={np.percentile(subset['dice_score'], 95):.4f}")

        # --- Cell-level IoU Violin Plot with Percentiles ---
        fig, ax = plt.subplots(figsize=(7, 7))

        sns.violinplot(
            data=df_plot,
            x='codec',
            y='iou_score',
            hue='segmentation',
            order=label_order,
            hue_order=['Cell', 'Nuclei'],
            palette=['tab:green', 'tab:blue'],
            inner='box',
            cut=0,
            ax=ax
        )

        ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
        ax.set_ylabel('Cell-level IoU', fontsize=24, fontweight='bold')
        _suffix = f' - {config["title_suffix"]}' if config["title_suffix"] else ''
        ax.set_title(f'Cell-level IoU{_suffix}', fontsize=24, fontweight='bold')

        ax.set_xticks(range(len(label_order)))
        ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20, rotation=45, ha='right')
        ax.tick_params(axis='both', labelsize=20)

        ax.legend(title='Segmentation', fontsize=18, title_fontsize=18, loc='lower left')

        # Add 5th and 95th percentile markers
        hue_order = ['Cell', 'Nuclei']
        colors = {'Cell': 'darkgreen', 'Nuclei': 'darkblue'}
        offset = 0.2

        for i, codec in enumerate(label_order):
            for j, seg in enumerate(hue_order):
                subset = df_plot[(df_plot['codec'] == codec) & (df_plot['segmentation'] == seg)]
                if len(subset) > 0:
                    p5 = np.percentile(subset['iou_score'], 5)
                    p95 = np.percentile(subset['iou_score'], 95)
                    x_pos = i + (j - 0.5) * offset * 2
                    ax.scatter([x_pos], [p95], marker='_', s=300, linewidths=3,
                              color=colors[seg], zorder=10)
                    ax.annotate(f'{p95:.3f}', (x_pos, p95), textcoords='offset points',
                               xytext=(0, 8), ha='center', fontsize=10, fontweight='bold',
                               color=colors[seg])
                    ax.scatter([x_pos], [p5], marker='_', s=300, linewidths=3,
                              color=colors[seg], zorder=10)
                    ax.annotate(f'{p5:.3f}', (x_pos, p5), textcoords='offset points',
                               xytext=(0, -14), ha='center', fontsize=10, fontweight='bold',
                               color=colors[seg])

        plt.tight_layout()
        output_path = f"{output_base}_cell_level_iou_violinplot_{config['file_suffix']}_p5_p95.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved cell-level IoU violinplot ({config['description']}) to: {output_path}")
        plt.close()

        # --- Cell-level IoU Boxen Plot ---
        fig, ax = plt.subplots(figsize=(7, 7))

        sns.boxenplot(
            data=df_plot,
            x='codec',
            y='iou_score',
            hue='segmentation',
            order=label_order,
            hue_order=['Cell', 'Nuclei'],
            palette=['tab:green', 'tab:blue'],
            ax=ax
        )

        ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
        ax.set_ylabel('Cell-level IoU', fontsize=24, fontweight='bold')
        _suffix = f' - {config["title_suffix"]}' if config["title_suffix"] else ''
        ax.set_title(f'Cell-level IoU{_suffix}', fontsize=24, fontweight='bold')

        ax.set_xticks(range(len(label_order)))
        ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20, rotation=45, ha='right')
        ax.tick_params(axis='both', labelsize=20)

        ax.legend(title='Segmentation', fontsize=18, title_fontsize=18, loc='lower left')

        plt.tight_layout()
        output_path = f"{output_base}_cell_level_iou_boxenplot_{config['file_suffix']}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved cell-level IoU boxenplot ({config['description']}) to: {output_path}")
        plt.close()

        # --- Cell-level Dice Violin Plot with Percentiles ---
        fig, ax = plt.subplots(figsize=(7, 7))

        sns.violinplot(
            data=df_plot,
            x='codec',
            y='dice_score',
            hue='segmentation',
            order=label_order,
            hue_order=['Cell', 'Nuclei'],
            palette=['tab:green', 'tab:blue'],
            inner='box',
            cut=0,
            ax=ax
        )

        ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
        ax.set_ylabel('Cell-level Dice', fontsize=24, fontweight='bold')
        _suffix = f' - {config["title_suffix"]}' if config["title_suffix"] else ''
        ax.set_title(f'Cell-level Dice{_suffix}', fontsize=24, fontweight='bold')

        ax.set_xticks(range(len(label_order)))
        ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20, rotation=45, ha='right')
        ax.tick_params(axis='both', labelsize=20)

        ax.legend(title='Segmentation', fontsize=18, title_fontsize=18, loc='lower left')

        # Add 5th and 95th percentile markers
        for i, codec in enumerate(label_order):
            for j, seg in enumerate(hue_order):
                subset = df_plot[(df_plot['codec'] == codec) & (df_plot['segmentation'] == seg)]
                if len(subset) > 0:
                    p5 = np.percentile(subset['dice_score'], 5)
                    p95 = np.percentile(subset['dice_score'], 95)
                    x_pos = i + (j - 0.5) * offset * 2
                    ax.scatter([x_pos], [p95], marker='_', s=300, linewidths=3,
                              color=colors[seg], zorder=10)
                    ax.annotate(f'{p95:.3f}', (x_pos, p95), textcoords='offset points',
                               xytext=(0, 8), ha='center', fontsize=10, fontweight='bold',
                               color=colors[seg])
                    ax.scatter([x_pos], [p5], marker='_', s=300, linewidths=3,
                              color=colors[seg], zorder=10)
                    ax.annotate(f'{p5:.3f}', (x_pos, p5), textcoords='offset points',
                               xytext=(0, -14), ha='center', fontsize=10, fontweight='bold',
                               color=colors[seg])

        plt.tight_layout()
        output_path = f"{output_base}_cell_level_dice_violinplot_{config['file_suffix']}_p5_p95.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved cell-level Dice violinplot ({config['description']}) to: {output_path}")
        plt.close()

        # --- Cell-level Dice Boxen Plot ---
        fig, ax = plt.subplots(figsize=(7, 7))

        sns.boxenplot(
            data=df_plot,
            x='codec',
            y='dice_score',
            hue='segmentation',
            order=label_order,
            hue_order=['Cell', 'Nuclei'],
            palette=['tab:green', 'tab:blue'],
            ax=ax
        )

        ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
        ax.set_ylabel('Cell-level Dice', fontsize=24, fontweight='bold')
        _suffix = f' - {config["title_suffix"]}' if config["title_suffix"] else ''
        ax.set_title(f'Cell-level Dice{_suffix}', fontsize=24, fontweight='bold')

        ax.set_xticks(range(len(label_order)))
        ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=20, rotation=45, ha='right')
        ax.tick_params(axis='both', labelsize=20)

        ax.legend(title='Segmentation', fontsize=18, title_fontsize=18, loc='lower left')

        plt.tight_layout()
        output_path = f"{output_base}_cell_level_dice_boxenplot_{config['file_suffix']}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved cell-level Dice boxenplot ({config['description']}) to: {output_path}")
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

    # Order codecs by mean IoU (lowest to highest quality)
    codec_mean_iou = df_plot.groupby('codec')['iou'].mean().sort_values(ascending=False)
    label_order = list(codec_mean_iou.index)
    _known_labels = {
        'jxl_lq': 'Low', 'jxl_mq': 'Medium', 'jxl_effort_3': 'Mid-High',
        'jxl_hq': 'High', 'jxl_d2_e8': 'D2 E8', 'jxl_d10': 'D10',
        'jxl_d15': 'D15', 'jxl_d20_e2': 'D20 E2', 'jxl_d30': 'D30',
    }
    codec_labels = {c: _known_labels.get(c, c.replace('jxl_', '').upper()) for c in label_order}

    n_codecs = len(label_order)
    fig, ax = plt.subplots(figsize=(max(14, n_codecs * 2), 7))

    sns.violinplot(
        data=df_plot,
        x='codec',
        y='iou',
        hue='codec',
        order=label_order,
        hue_order=label_order,
        palette='viridis',
        inner='box',
        cut=0,
        legend=False,
        ax=ax
    )

    ax.set_xlabel('Compression Quality', fontsize=20, fontweight='bold')
    ax.set_ylabel('IoU', fontsize=20, fontweight='bold')
    step_nice = 'Cell' if 'cell' in segment_step else 'Nuclei'
    ax.set_title(f'IoU - {step_nice} Segmentation Similarity', fontsize=22, fontweight='bold')

    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels([codec_labels[c] for c in label_order], fontsize=18, rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=18)

    plt.tight_layout()
    output_path = f"{output_prefix}_iou_boxplot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved IoU boxplot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare segmentation masks from different compression methods")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing all methods")
    parser.add_argument("--ground-truth", type=str, required=True, help="Ground truth method name (e.g., zstd.zarr)")
    parser.add_argument("--methods", nargs='+', required=True, help="List of methods to compare against ground truth")
    parser.add_argument("--output", type=str, default="segmentation_comparison", help="Output file prefix")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory parent (default: analysis/segmentation/output/)")
    parser.add_argument("--force-rerun", action="store_true", help="Always regenerate the detailed CSV even if it exists (non-interactive)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers (default: 8)")
    parser.add_argument("--segment-step", type=str, default="segment_cell", help="Segmentation step name (e.g., segment_cell, segment_nuclei)")
    parser.add_argument("--both", action="store_true", help="Process both cell and nuclei segmentation together")
    parser.add_argument("--visualize-sample", action="store_true", help="Only visualize a single well+method comparison (requires --well and one --methods entry)")
    parser.add_argument("--visualize-sample-grid", action="store_true", help="Visualize a grid of samples for different compressions and the difference between them (requires --well, --visualize-sample and one --methods entry)")
    parser.add_argument("--well", type=str, default=None, help="Source/well ID for single sample visualization")
    parser.add_argument("--file", type=str, default=None, help="Specific mask file name for single sample visualization (default: first file found)")
    parser.add_argument("--zarr-root", type=str, default=os.environ.get("ZARR_ROOT", "data/jump_target2_4plate"), help="Root directory containing zarr source images")
    parser.add_argument("--samples", type=int, default=None, help="Limit to N samples for quick testing (default: all)")
    parser.add_argument("--filter-percentile", type=float, default=None, help="Filter out wells in bottom and top N percentile of cell count based on ground truth (e.g., 5 filters bottom 5%% and top 5%%)")
    parser.add_argument("--save-mappings", action="store_true", help="Save instance ID mappings between GT and predictions to a parquet file")
    parser.add_argument("--fast", action="store_true", help="Fast mode: skip expensive metrics (hausdorff, asd) for ~2-3x speedup")

    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise ValueError(f"Root directory does not exist: {root}")

    # Create output directory structure: <parent>/<output_name>/
    parent_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "output"
    output_dir = parent_dir / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Set output prefix to include the directory
    output_base = str(output_dir / args.output)

    # Handle --both flag: load both cell and nuclei segmentation from existing CSV files
    if args.both:
        cell_csv = f"{output_base}_segment_cell_detailed.csv"
        nuclei_csv = f"{output_base}_segment_nuclei_detailed.csv"

        if not Path(cell_csv).exists():
            raise ValueError(f"Cell segmentation results not found: {cell_csv}")
        if not Path(nuclei_csv).exists():
            raise ValueError(f"Nuclei segmentation results not found: {nuclei_csv}")

        df_cell = pd.read_csv(cell_csv)
        df_nuclei = pd.read_csv(nuclei_csv)
        print(f"Loaded cell results from: {cell_csv}")
        print(f"Loaded nuclei results from: {nuclei_csv}")

        # Filter by cell count percentile if specified
        if args.filter_percentile is not None and 'n_true' in df_cell.columns:
            pct = args.filter_percentile
            # Compute percentile thresholds from ground truth cell counts
            lower_thresh_cell = np.percentile(df_cell['n_true'], pct)
            upper_thresh_cell = np.percentile(df_cell['n_true'], 100 - pct)
            lower_thresh_nuclei = np.percentile(df_nuclei['n_true'], pct)
            upper_thresh_nuclei = np.percentile(df_nuclei['n_true'], 100 - pct)

            n_cell_before = len(df_cell)
            n_nuclei_before = len(df_nuclei)

            df_cell = df_cell[(df_cell['n_true'] >= lower_thresh_cell) & (df_cell['n_true'] <= upper_thresh_cell)]
            df_nuclei = df_nuclei[(df_nuclei['n_true'] >= lower_thresh_nuclei) & (df_nuclei['n_true'] <= upper_thresh_nuclei)]

            print(f"Filtered cell data: {n_cell_before} -> {len(df_cell)} (removed bottom/top {pct}% by GT cell count)")
            print(f"  Cell count range: {lower_thresh_cell:.0f} - {upper_thresh_cell:.0f}")
            print(f"Filtered nuclei data: {n_nuclei_before} -> {len(df_nuclei)} (removed bottom/top {pct}% by GT cell count)")
            print(f"  Nuclei count range: {lower_thresh_nuclei:.0f} - {upper_thresh_nuclei:.0f}")

            # Check overlap between filtered cell and nuclei dataframes
            cell_wells = set(zip(df_cell['source_id'], df_cell['file']))
            nuclei_wells = set(zip(df_nuclei['source_id'], df_nuclei['file']))
            overlap_wells = cell_wells & nuclei_wells
            cell_only = cell_wells - nuclei_wells
            nuclei_only = nuclei_wells - cell_wells

            print(f"Well overlap after filtering:")
            print(f"  Unique wells in cell data: {len(cell_wells)}")
            print(f"  Unique wells in nuclei data: {len(nuclei_wells)}")
            print(f"  Wells in both: {len(overlap_wells)}")
            print(f"  Wells only in cell: {len(cell_only)}")
            print(f"  Wells only in nuclei: {len(nuclei_only)}")

        # Create combined well-level IoU plots
        plot_iou_boxplot_combined(df_cell, df_nuclei, output_base)

        # Create cell-level IoU plots if instance mappings are available
        # Automatically look for mappings in output_dir/instance_mappings/
        mappings_dir = output_dir / "instance_mappings"
        if mappings_dir.exists():
            print(f"\nFound instance mappings directory: {mappings_dir}")
            plot_cell_level_iou_combined(df_cell, df_nuclei, output_base, mappings_dir, thresh=0.5)
        else:
            print(f"\nInstance mappings directory not found: {mappings_dir}")
            print("Skipping cell-level IoU plots. To generate these plots:")
            print("  1. Run segmentation analysis with --save-mappings flag")
            print("  2. Run --both command again to include cell-level plots")

        return

    # Embed segment step in output prefix so cell/nuclei results don't overwrite each other
    args.output = f"{output_base}_{args.segment_step}"

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
        if args.visualize_sample_grid:
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
    if Path(detailed_output).exists() and not args.force_rerun:
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
        total_mappings_saved = 0

        # Create subdirectories upfront
        if args.save_mappings:
            mappings_dir = output_dir / "instance_mappings"
            mappings_dir.mkdir(parents=True, exist_ok=True)

        detailed_dir = output_dir / "detailed_results"
        detailed_dir.mkdir(parents=True, exist_ok=True)

        for method in args.methods:
            print(f"\nProcessing method: {method}")
            method_files = find_mask_files(root, method, args.segment_step)
            print(f"Found {len(method_files)} mask files for {method}")

            matches = match_files(gt_files, method_files)
            print(f"Matched {len(matches)} file pairs")

            if len(matches) == 0:
                print(f"Warning: No matching files found for {method}")
                continue

            # Limit samples if requested
            if args.samples is not None and args.samples < len(matches):
                matches = matches[:args.samples]
                print(f"Limited to {args.samples} samples")

            # Prepare arguments for parallel processing
            task_args = [(gt, pred, method, args.save_mappings, args.fast) for gt, pred in matches]

            # Process in parallel
            method_results = []
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(process_file_pair, task_arg) for task_arg in task_args]

                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Comparing {method}"):
                    result = future.result()
                    if result is not None:
                        method_results.append(result)

            if len(method_results) == 0:
                print(f"  No results for {method}")
                continue

            # Save mappings for this codec immediately after processing
            if args.save_mappings:
                method_mappings = []
                for result in method_results:
                    if 'mappings' in result:
                        method_mappings.extend(result.pop('mappings'))

                if len(method_mappings) > 0:
                    mappings_df = pd.DataFrame(method_mappings)
                    # Remove method column since it's redundant (encoded in filename)
                    mappings_df = mappings_df.drop(columns=['method'])

                    # Clean up method name for filename
                    method_clean = method.replace('.zarr', '')
                    mappings_output = mappings_dir / f"{args.segment_step}_{method_clean}.parquet"
                    mappings_df.to_parquet(mappings_output, index=False)
                    print(f"  Saved {len(mappings_df)} mappings to: {mappings_output}")
                    total_mappings_saved += len(mappings_df)

            # Save detailed results for this codec immediately
            method_clean = method.replace('.zarr', '')
            method_detailed_output = detailed_dir / f"{args.segment_step}_{method_clean}.csv"
            method_df = pd.DataFrame(method_results)
            method_df.to_csv(method_detailed_output, index=False)
            print(f"  Saved {len(method_df)} detailed results to: {method_detailed_output}")

            all_results.extend(method_results)

        if len(all_results) == 0:
            print("No results generated. Check your data paths.")
            return

        if args.save_mappings and total_mappings_saved > 0:
            print(f"\nTotal instance mappings saved: {total_mappings_saved}")

        # Combine all detailed results and save combined file
        df = pd.DataFrame(all_results)
        df.to_csv(detailed_output, index=False)
        print(f"\nCombined detailed results saved to: {detailed_output}")

    # Plot IoU boxplot
    plot_iou_boxplot(df, args.output, args.segment_step)

    # Compute summary statistics
    summary_data = []
    for method in args.methods:
        method_df = df[df['method'] == method]
        if len(method_df) > 0:
            stats = {
                'method': method,
                'n_files': len(method_df),
                'dice_mean': method_df['dice'].mean(),
                'dice_std': method_df['dice'].std(),
                'dice_median': method_df['dice'].median(),
                'iou_mean': method_df['iou'].mean(),
                'iou_std': method_df['iou'].std(),
                'iou_median': method_df['iou'].median(),
                'precision_mean': method_df['precision'].mean(),
                'precision_std': method_df['precision'].std(),
                'precision_median': method_df['precision'].median(),
                'recall_mean': method_df['recall'].mean(),
                'recall_std': method_df['recall'].std(),
                'recall_median': method_df['recall'].median(),
            }
            # Add expensive metrics only if available (not in fast mode)
            if 'hausdorff_95' in method_df.columns:
                stats.update({
                    'hausdorff_95_mean': method_df['hausdorff_95'].mean(),
                    'hausdorff_95_std': method_df['hausdorff_95'].std(),
                    'hausdorff_95_median': method_df['hausdorff_95'].median(),
                })
            if 'asd' in method_df.columns:
                stats.update({
                    'asd_mean': method_df['asd'].mean(),
                    'asd_std': method_df['asd'].std(),
                    'asd_median': method_df['asd'].median(),
                })
            # Add instance matching metrics if available
            if 'n_true' in method_df.columns:
                stats.update({
                    'n_true_mean': method_df['n_true'].mean(),
                    'n_pred_mean': method_df['n_pred'].mean(),
                    'cell_count_diff_mean': (method_df['n_pred'] - method_df['n_true']).mean(),
                    'cell_count_diff_std': (method_df['n_pred'] - method_df['n_true']).std(),
                })
            if 'inst_f1_50' in method_df.columns:
                stats.update({
                    'inst_f1_50_mean': method_df['inst_f1_50'].mean(),
                    'inst_f1_50_std': method_df['inst_f1_50'].std(),
                    'inst_f1_70_mean': method_df['inst_f1_70'].mean(),
                    'inst_f1_70_std': method_df['inst_f1_70'].std(),
                    'inst_precision_50_mean': method_df['inst_precision_50'].mean(),
                    'inst_recall_50_mean': method_df['inst_recall_50'].mean(),
                    'inst_panoptic_quality_50_mean': method_df['inst_panoptic_quality_50'].mean(),
                })
            if 'inst_f1_80' in method_df.columns:
                stats.update({
                    'inst_f1_80_mean': method_df['inst_f1_80'].mean(),
                    'inst_f1_80_std': method_df['inst_f1_80'].std(),
                    'inst_f1_90_mean': method_df['inst_f1_90'].mean(),
                    'inst_f1_90_std': method_df['inst_f1_90'].std(),
                })
            summary_data.append(stats)

    summary_df = pd.DataFrame(summary_data)

    # Save summary results
    summary_output = f"{args.output}_summary.csv"
    summary_df.to_csv(summary_output, index=False)
    print(f"Summary results saved to: {summary_output}")

    # Print formatted summary table
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    print(f"\n{'Method':<30} {'N Files':<10} {'Dice':<15} {'IoU':<15}")
    print("-"*100)
    for _, row in summary_df.iterrows():
        print(f"{row['method']:<30} {row['n_files']:<10} "
              f"{row['dice_mean']:.4f}±{row['dice_std']:.4f}  "
              f"{row['iou_mean']:.4f}±{row['iou_std']:.4f}")

    print(f"\n{'Method':<30} {'Precision':<15} {'Recall':<15}")
    print("-"*100)
    for _, row in summary_df.iterrows():
        print(f"{row['method']:<30} "
              f"{row['precision_mean']:.4f}±{row['precision_std']:.4f}  "
              f"{row['recall_mean']:.4f}±{row['recall_std']:.4f}")

    # Print instance matching metrics if available
    if 'inst_f1_50_mean' in summary_df.columns:
        print(f"\n{'Method':<30} {'Inst F1@0.5':<15} {'Inst F1@0.7':<15} {'Inst F1@0.8':<15} {'Inst F1@0.9':<15} {'PQ@0.5':<15}")
        print("-"*120)
        for _, row in summary_df.iterrows():
            f1_80 = f"{row['inst_f1_80_mean']:.4f}±{row['inst_f1_80_std']:.4f}" if 'inst_f1_80_mean' in row else "N/A"
            f1_90 = f"{row['inst_f1_90_mean']:.4f}±{row['inst_f1_90_std']:.4f}" if 'inst_f1_90_mean' in row else "N/A"
            print(f"{row['method']:<30} "
                  f"{row['inst_f1_50_mean']:.4f}±{row['inst_f1_50_std']:.4f}  "
                  f"{row['inst_f1_70_mean']:.4f}±{row['inst_f1_70_std']:.4f}  "
                  f"{f1_80:<15} {f1_90:<15} "
                  f"{row['inst_panoptic_quality_50_mean']:.4f}")

        print(f"\n{'Method':<30} {'GT Cells':<12} {'Pred Cells':<12} {'Count Diff':<18}")
        print("-"*100)
        for _, row in summary_df.iterrows():
            print(f"{row['method']:<30} "
                  f"{row['n_true_mean']:<12.1f} "
                  f"{row['n_pred_mean']:<12.1f} "
                  f"{row['cell_count_diff_mean']:+.2f}±{row['cell_count_diff_std']:.2f}")

    print("="*100)

    # Visualize the best, median, and worst performing samples
    visualize_samples(df, root, args.ground_truth, args.methods, args.output, args.segment_step)


if __name__ == "__main__":
    main()
