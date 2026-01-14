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
    zarr_source_path = Path("/work/datasets/jump_target2_4plate/zstd.zarr") / source_id

    if zarr_source_path.exists():
        try:
            import zarr
            # Load the Zarr array
            store = zarr.open(str(zarr_source_path), mode='r')
            img = store[:]

            # img shape is [channels, height, width] for Zarr v3
            if img.ndim == 3 and img.shape[0] < img.shape[1]:
                # Select specified channels and create RGB (as float array)
                selected = np.zeros((*img.shape[1:], 3), dtype=np.float64)

                for i, ch in enumerate(channels):
                    if ch < img.shape[0]:
                        ch_data = img[ch].astype(np.float64)
                        # Use 1st and 99th percentiles for better contrast
                        ch_min = np.percentile(ch_data, 1)
                        ch_max = np.percentile(ch_data, 99)
                        if ch_max > ch_min:
                            # Clip values to percentile range then normalize
                            ch_clipped = np.clip(ch_data, ch_min, ch_max)
                            selected[..., i] = (ch_clipped - ch_min) / (ch_max - ch_min)

                return selected
        except Exception as e:
            warnings.warn(f"Failed to load Zarr image from {zarr_source_path}: {e}")

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


def main():
    parser = argparse.ArgumentParser(description="Compare segmentation masks from different compression methods")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing all methods")
    parser.add_argument("--ground-truth", type=str, required=True, help="Ground truth method name (e.g., zstd.zarr)")
    parser.add_argument("--methods", nargs='+', required=True, help="List of methods to compare against ground truth")
    parser.add_argument("--output", type=str, default="segmentation_comparison", help="Output file prefix")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--segment-step", type=str, default="segment_cell", help="Segmentation step name (e.g., segment_cell, segment_nuclei)")

    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise ValueError(f"Root directory does not exist: {root}")

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
    detailed_output = f"{args.output}_detailed.csv"
    df.to_csv(detailed_output, index=False)
    print(f"\nDetailed results saved to: {detailed_output}")

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
