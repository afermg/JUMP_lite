"""
Compare image quality metrics between lossy codecs and zstd reference.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import zarr
from lpips import LPIPS
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')

# Output directory (relative to this script)
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "output"

# Register imagecodecs numcodecs for JpegXL support
try:
    from imagecodecs.numcodecs import Brotli, Jpegxl
    import numcodecs
    numcodecs.register_codec(Brotli)
    numcodecs.register_codec(Jpegxl)
except (ImportError, AttributeError) as e:
    print(f"Warning: imagecodecs.numcodecs not available: {e}")


def compute_laplacian_variance(img: np.ndarray) -> float:
    """Variance of Laplacian response across all channels. Higher = sharper.

    Reference: squidpy _sharpness_metrics._laplacian_variance
    """
    from skimage.filters import laplace
    variances = []
    for c in range(img.shape[0]):  # img shape: (C, H, W)
        lap = laplace(img[c].astype(np.float32))
        variances.append(float(np.var(lap)))
    return float(np.mean(variances))


def compute_tenengrad(img: np.ndarray) -> float:
    """Mean Tenengrad energy (sum of squared Sobel gradients) across channels. Higher = sharper.

    Reference: squidpy _sharpness_metrics._tenengrad_mean
    """
    from skimage.filters import sobel_h, sobel_v
    energies = []
    for c in range(img.shape[0]):  # img shape: (C, H, W)
        channel = img[c].astype(np.float32)
        energy = sobel_h(channel) ** 2 + sobel_v(channel) ** 2
        energies.append(float(energy.mean()))
    return float(np.mean(energies))


def setup_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def open_zarr_store(zarr_path: Path):
    """Open a zarr store and return the root group."""
    store = zarr.storage.LocalStore(zarr_path)
    return zarr.group(store)


def normalize_image(img: np.ndarray) -> torch.Tensor:
    """Normalize image to [0, 1] range."""
    if img.dtype == np.uint16:
        img_float = img.astype(np.float32) / 65535.0
    elif img.dtype == np.uint8:
        img_float = img.astype(np.float32) / 255.0
    else:
        img_float = img.astype(np.float32)
    return torch.from_numpy(img_float).unsqueeze(0)


def compute_metrics(original, compressed, device=None, psnr_metric=None, ssim_metric=None, lpips_metric=None, sharpness_only=False):
    """Compute PSNR, SSIM, optionally LPIPS, and sharpness metrics for a pair of images."""
    result = {}

    if not sharpness_only:
        orig_tensor = normalize_image(original).to(device)
        comp_tensor = normalize_image(compressed).to(device)

        psnr_value = psnr_metric(comp_tensor, orig_tensor).item()
        ssim_value = ssim_metric(comp_tensor, orig_tensor).item()

        result = {'psnr': psnr_value, 'ssim': ssim_value}

        # LPIPS per channel (repeat to 3 channels) - optional
        if lpips_metric is not None:
            lpips_values = []
            for c in range(orig_tensor.shape[1]):
                orig_rgb = orig_tensor[:, c:c+1, :, :].repeat(1, 3, 1, 1)
                comp_rgb = comp_tensor[:, c:c+1, :, :].repeat(1, 3, 1, 1)
                lpips_values.append(lpips_metric(comp_rgb, orig_rgb).item())
            result['lpips'] = np.mean(lpips_values)
        else:
            result['lpips'] = np.nan

    # Sharpness metrics (computed on CPU numpy arrays)
    orig_lap = compute_laplacian_variance(original)
    comp_lap = compute_laplacian_variance(compressed)
    orig_ten = compute_tenengrad(original)
    comp_ten = compute_tenengrad(compressed)

    result.update({
        'laplacian_orig': orig_lap,
        'laplacian_comp': comp_lap,
        'laplacian_ratio': comp_lap / orig_lap if orig_lap > 0 else float('nan'),
        'laplacian_diff': orig_lap - comp_lap,
        'tenengrad_orig': orig_ten,
        'tenengrad_comp': comp_ten,
        'tenengrad_ratio': comp_ten / orig_ten if orig_ten > 0 else float('nan'),
        'tenengrad_diff': orig_ten - comp_ten,
    })

    return result


def main():
    import argparse
    import random
    parser = argparse.ArgumentParser(description="Compare image quality metrics between lossy codecs and zstd reference")
    parser.add_argument("--figures-only", action="store_true", help="Only generate figures from existing quality_metrics.csv")
    parser.add_argument("--data-dir", type=Path, default=Path("/work/datasets/jump_target2_4plate"), help="Directory containing zarr files")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of sites to sample (default: use all sites)")
    parser.add_argument("--skip-lpips", action="store_true", help="Skip LPIPS computation (10x faster, SSIM usually sufficient)")
    parser.add_argument("--lpips-net", type=str, default="alex", choices=["alex", "vgg", "squeeze"],
                        help="LPIPS network (alex=fast, vgg=accurate, squeeze=fastest)")
    parser.add_argument("--exclude-codecs", nargs="*", default=[], help="Codec names to exclude (without .zarr suffix)")
    parser.add_argument("--sharpness-only", action="store_true",
                        help="Only compute Laplacian variance and Tenengrad sharpness metrics (no GPU needed)")
    args = parser.parse_args()

    data_dir = args.data_dir
    reference_codec = "zstd"

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Figures-only mode: load existing results and regenerate plots
    if args.figures_only:
        csv_path = OUTPUT_DIR / "quality_metrics.csv"
        if not csv_path.exists():
            print(f"Error: {csv_path} not found. Run evaluation first.")
            return
        print(f"Loading existing results from {csv_path}")
        df = pd.read_csv(csv_path)
        generate_violin_plots(df, OUTPUT_DIR)
        return

    # Find all codecs
    codec_dirs = sorted([d for d in data_dir.glob("*.zarr") if d.is_dir()])
    codec_names = [d.stem for d in codec_dirs]

    # Exclude requested codecs
    if args.exclude_codecs:
        codec_names = [c for c in codec_names if c not in args.exclude_codecs]
        print(f"Excluding codecs: {args.exclude_codecs}")

    print(f"Found codecs: {codec_names}")
    print(f"Reference: {reference_codec}")

    # Setup device and metrics (skip GPU for sharpness-only mode)
    device = None
    psnr_metric = None
    ssim_metric = None
    lpips_metric = None

    if not args.sharpness_only:
        device = setup_device()
        psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        # LPIPS is optional (slow)
        if args.skip_lpips:
            print("Skipping LPIPS computation (much faster!)")
        else:
            lpips_metric = LPIPS(net=args.lpips_net).to(device)
            print(f"Using LPIPS with '{args.lpips_net}' network")
    else:
        print("\nSharpness-only mode: skipping GPU setup and PSNR/SSIM/LPIPS")

    # Open reference store (lazy - no data loaded yet)
    print(f"\nOpening reference ({reference_codec})...")
    ref_path = data_dir / f"{reference_codec}.zarr"
    ref_store = open_zarr_store(ref_path)
    site_names = list(ref_store.keys())
    print(f"Found {len(site_names)} total sites")

    # Sample sites if requested
    if args.n_samples and args.n_samples < len(site_names):
        site_names = random.sample(site_names, args.n_samples)
        print(f"Sampling {args.n_samples} sites for evaluation")
    else:
        print(f"Evaluating all {len(site_names)} sites")

    # Open all codec stores upfront
    print("\nOpening all codec stores...")
    codec_stores = {}
    codecs_to_compare = [c for c in codec_names if c != reference_codec]
    for codec_name in codecs_to_compare:
        codec_path = data_dir / f"{codec_name}.zarr"
        codec_stores[codec_name] = open_zarr_store(codec_path)
        print(f"  Opened {codec_name}")

    # Process all codecs per site (cache reference, avoid reloading)
    import time
    print(f"\nProcessing {len(site_names)} sites x {len(codecs_to_compare)} codecs...")
    print(f"Total comparisons: {len(site_names) * len(codecs_to_compare)}")
    results = []

    desc = "Computing sharpness" if args.sharpness_only else "Evaluating sites"
    start_time = time.time()
    for i, site_name in enumerate(tqdm(site_names, desc=desc)):
        # Load reference once per site
        try:
            original = ref_store[site_name][:]
        except KeyError:
            continue

        # Compare against all codecs
        for codec_name in codecs_to_compare:
            codec_store = codec_stores[codec_name]

            # Load compressed version
            try:
                compressed = codec_store[site_name][:]
            except KeyError:
                continue

            # Compute metrics
            metrics = compute_metrics(
                original, compressed, device,
                psnr_metric, ssim_metric, lpips_metric,
                sharpness_only=args.sharpness_only,
            )

            results.append({
                'site_name': site_name,
                'codec': codec_name,
                **metrics,
            })

        # Print timing estimate every 10 sites
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (len(site_names) - i - 1) / rate
            tqdm.write(f"  [{i+1}/{len(site_names)}] Rate: {rate:.1f} sites/sec, ETA: {remaining/60:.1f} min")

    df = pd.DataFrame(results)

    # Print summary
    print("\n" + "="*80)
    print(f"Quality Comparison vs {reference_codec} (reference)")
    print("="*80)

    for codec in codecs_to_compare:
        codec_df = df[df['codec'] == codec]
        print(f"\n{codec}:")
        if not args.sharpness_only:
            print(f"  PSNR:  {codec_df['psnr'].mean():.2f} +/- {codec_df['psnr'].std():.2f} dB")
            print(f"  SSIM:  {codec_df['ssim'].mean():.4f} +/- {codec_df['ssim'].std():.4f}")
            if not codec_df['lpips'].isna().all():
                print(f"  LPIPS: {codec_df['lpips'].mean():.4f} +/- {codec_df['lpips'].std():.4f}")
            else:
                print(f"  LPIPS: (skipped)")
        print(f"  Laplacian (orig):  {codec_df['laplacian_orig'].mean():.4f} +/- {codec_df['laplacian_orig'].std():.4f}")
        print(f"  Laplacian (comp):  {codec_df['laplacian_comp'].mean():.4f} +/- {codec_df['laplacian_comp'].std():.4f}")
        print(f"  Laplacian Ratio:   {codec_df['laplacian_ratio'].mean():.4f} +/- {codec_df['laplacian_ratio'].std():.4f}")
        print(f"  Laplacian Diff:    {codec_df['laplacian_diff'].mean():.4f} +/- {codec_df['laplacian_diff'].std():.4f}")
        print(f"  Tenengrad (orig):  {codec_df['tenengrad_orig'].mean():.4f} +/- {codec_df['tenengrad_orig'].std():.4f}")
        print(f"  Tenengrad (comp):  {codec_df['tenengrad_comp'].mean():.4f} +/- {codec_df['tenengrad_comp'].std():.4f}")
        print(f"  Tenengrad Ratio:   {codec_df['tenengrad_ratio'].mean():.4f} +/- {codec_df['tenengrad_ratio'].std():.4f}")
        print(f"  Tenengrad Diff:    {codec_df['tenengrad_diff'].mean():.4f} +/- {codec_df['tenengrad_diff'].std():.4f}")

    print("\n" + "="*80)
    if not args.sharpness_only:
        print("Interpretation: PSNR/SSIM higher=better, LPIPS lower=better")
    print("Interpretation: Laplacian/Tenengrad Ratio closer to 1.0 = better sharpness preservation")
    print("="*80)

    # Save results (merge into existing CSV if sharpness-only)
    output_path = OUTPUT_DIR / "quality_metrics.csv"
    if args.sharpness_only and output_path.exists():
        existing_df = pd.read_csv(output_path)
        sharpness_cols = [
            'laplacian_orig', 'laplacian_comp', 'laplacian_ratio', 'laplacian_diff',
            'tenengrad_orig', 'tenengrad_comp', 'tenengrad_ratio', 'tenengrad_diff',
        ]
        existing_df = existing_df.drop(
            columns=[c for c in sharpness_cols if c in existing_df.columns],
            errors='ignore',
        )
        merged = existing_df.merge(
            df[['site_name', 'codec'] + sharpness_cols],
            on=['site_name', 'codec'],
            how='left',
        )
        merged.to_csv(output_path, index=False)
        print(f"\nMerged sharpness metrics into: {output_path}")
    else:
        df.to_csv(output_path, index=False)
        print(f"\nDetailed results saved to: {output_path}")

    # Generate violin plots
    generate_violin_plots(df, OUTPUT_DIR)


def generate_violin_plots(df: pd.DataFrame, output_dir: Path):
    """Generate violin plots showing distribution of metrics for each codec."""
    df = df.copy()
    df['codec_display'] = df['codec'].str.replace('jpegxl_lossy_', 'jxl_')

    # Fixed codec order: highest quality to lowest
    CODEC_ORDER = ['jxl_hq', 'jxl_effort_3', 'jxl_d2_e8', 'jxl_mq', 'jxl_lq', 'jxl_d10', 'jxl_d15', 'jxl_d20_e2', 'jxl_d30']
    available_codecs = set(df['codec_display'].unique())
    label_order = [c for c in CODEC_ORDER if c in available_codecs]
    # Append any codecs not in the predefined order
    label_order += [c for c in sorted(available_codecs) if c not in label_order]

    has_psnr = 'psnr' in df.columns and not df['psnr'].isna().all()

    # Check which data exists
    has_lpips = 'lpips' in df.columns and not df['lpips'].isna().all()
    has_sharpness = 'laplacian_ratio' in df.columns

    # Build list of metrics to plot
    metrics = []
    if has_psnr:
        metrics.extend([('psnr', 'PSNR (dB)'), ('ssim', 'SSIM')])
    if has_lpips:
        metrics.append(('lpips', 'LPIPS'))
    if has_sharpness:
        metrics.extend([
            ('laplacian_orig', 'Laplacian (Orig)'),
            ('laplacian_comp', 'Laplacian (Comp)'),
            ('laplacian_ratio', 'Laplacian Ratio'),
            ('laplacian_diff', 'Laplacian Diff'),
            ('tenengrad_orig', 'Tenengrad (Orig)'),
            ('tenengrad_comp', 'Tenengrad (Comp)'),
            ('tenengrad_ratio', 'Tenengrad Ratio'),
            ('tenengrad_diff', 'Tenengrad Diff'),
        ])

    n_plots = len(metrics)
    n_codecs = len(label_order)
    ncols = min(n_plots, 4)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(ncols * 4.5, n_codecs * 1.2), nrows * 5))
    if n_plots == 1:
        axes = [axes]
    else:
        axes = list(np.array(axes).flatten())
    # Hide unused subplots
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    for ax, (metric, label) in zip(axes, metrics):
        sns.violinplot(
            data=df,
            x='codec_display',
            y=metric,
            hue='codec_display',
            order=label_order,
            hue_order=label_order,
            palette='viridis',
            inner='box',
            cut=0,
            legend=False,
            ax=ax
        )

        ax.set_xlabel('Codec', fontsize=14, fontweight='bold')
        ax.set_ylabel(label, fontsize=14, fontweight='bold')
        ax.set_title(f'{metric.upper()} Distribution', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=12)

        ax.set_xticks(range(len(label_order)))
        ax.set_xticklabels(label_order, fontsize=11, rotation=45, ha='right')

        if metric == 'ssim':
            ax.set_ylim(0.5, 1.05)
        elif metric.endswith('_ratio'):
            ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
        elif metric.endswith('_diff'):
            ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.5)

    title = 'Image Quality Metrics: JPEG-XL Lossy vs ZSTD Reference'
    if not has_lpips and has_psnr:
        title += ' (LPIPS skipped)'
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = output_dir / "quality_metrics_violin.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Violin plot saved to: {plot_path}")

    # Generate separate SSIM plot (only if SSIM data exists)
    if has_psnr:
        generate_ssim_plot(df, output_dir)


def generate_ssim_plot(df: pd.DataFrame, output_dir: Path):
    """Generate a separate violin plot for SSIM only."""
    df = df.copy()
    df['codec_display'] = df['codec'].str.replace('jpegxl_lossy_', 'jxl_')

    # Nice display names
    display_names = {
        'jxl_hq': 'High',
        'jxl_effort_3': 'Mid-High',
        'jxl_d2_e8': 'D2 E8',
        'jxl_mq': 'Medium',
        'jxl_lq': 'Low',
        'jxl_d10': 'D10',
        'jxl_d15': 'D15',
        'jxl_d20_e2': 'D20 E2',
        'jxl_d30': 'D25',
    }

    # Fixed codec order: highest quality to lowest
    CODEC_ORDER = ['jxl_hq', 'jxl_effort_3', 'jxl_d2_e8', 'jxl_mq', 'jxl_lq', 'jxl_d10', 'jxl_d15', 'jxl_d20_e2', 'jxl_d30']
    available_codecs = set(df['codec_display'].unique())
    label_order = [c for c in CODEC_ORDER if c in available_codecs]
    label_order += [c for c in sorted(available_codecs) if c not in label_order]

    for codec in label_order:
        if codec not in display_names:
            display_names[codec] = codec.replace('jxl_', '').upper()

    n_codecs = len(label_order)
    fig, ax = plt.subplots(figsize=(max(7, n_codecs * 1.2), 7))

    sns.violinplot(
        data=df,
        x='codec_display',
        y='ssim',
        hue='codec_display',
        order=label_order,
        hue_order=label_order,
        palette='viridis',
        inner='box',
        cut=0,
        legend=False,
        ax=ax
    )

    ax.set_xlabel('Compression Quality', fontsize=24, fontweight='bold')
    ax.set_ylabel('SSIM', fontsize=24, fontweight='bold')
    ax.set_title('SSIM - Image Similarity', fontsize=26, fontweight='bold')
    ax.tick_params(axis='both', labelsize=18)
    ax.set_xticks(range(len(label_order)))
    ax.set_xticklabels([display_names[c] for c in label_order], fontsize=18, rotation=45, ha='right')
    ax.set_ylim(0.9, 1.01)

    plt.tight_layout()
    plot_path = output_dir / "ssim_violin.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"SSIM plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
