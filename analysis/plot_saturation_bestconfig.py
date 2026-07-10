"""Plot saturation curves using the best config per (model, n, seed).

Produces one line per model with shaded variance across seeds (instead of
showing all configs separately). For each (model, n_actual, seed) we keep the
config with the highest metric value, then aggregate mean/std across seeds.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = PROJECT_ROOT / "analysis/output/saturation_proper"
RXRX3_CORE_COMPOUNDS = 1674
RXRX3_CORE_PERTURBATIONS = 1674 + 735


def best_config_per_seed(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    df = df.dropna(subset=[metric]).copy()
    idx = df.groupby(["model", "n_actual", "seed"])[metric].idxmax()
    return df.loc[idx].reset_index(drop=True)


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, title: str,
                output_path: Path):
    best = best_config_per_seed(df, metric)
    if best.empty:
        print(f"  no data for {metric} → skipping {output_path.name}")
        return

    cmap = plt.cm.tab10
    models = sorted(best["model"].unique())
    colors = {m: cmap(i) for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in models:
        sub = best[best["model"] == model]
        summary = sub.groupby("n_actual")[metric].agg(["mean", "std"]).reset_index()
        summary = summary.sort_values("n_actual")
        ax.plot(summary["n_actual"], summary["mean"], "o-",
                color=colors[model], label=model, markersize=4)
        ax.fill_between(summary["n_actual"],
                        summary["mean"] - summary["std"],
                        summary["mean"] + summary["std"],
                        color=colors[model], alpha=0.18)

    ax.axvline(x=RXRX3_CORE_COMPOUNDS, color="red", linestyle="--",
               alpha=0.7, linewidth=1.5)
    ax.axvline(x=RXRX3_CORE_PERTURBATIONS, color="darkred", linestyle=":",
               alpha=0.7, linewidth=1.5)
    ylim = ax.get_ylim()
    ax.text(RXRX3_CORE_COMPOUNDS * 0.85, ylim[1] * 0.95,
            f"RxRx3-core\ncompounds\n({RXRX3_CORE_COMPOUNDS})",
            color="red", fontsize=8, va="top", ha="right")
    ax.text(RXRX3_CORE_PERTURBATIONS * 1.05, ylim[1] * 0.95,
            f"RxRx3-core\nperturbations\n({RXRX3_CORE_PERTURBATIONS})",
            color="darkred", fontsize=8, va="top", ha="left")

    ax.set_xlabel("Number of Treatments")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close(fig)


def plot_group(csv_path: Path, output_dir: Path, group: str | None):
    if not csv_path.exists():
        print(f"missing: {csv_path}")
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"empty: {csv_path}")
        return

    suffix = f"_{group}" if group else ""
    label = f" ({group})" if group else ""

    plot_metric(
        df, "PA_mean_nap", "PA (NAP)",
        f"PA (NAP) vs. Treatment Count{label}\n(best config per seed; band = std across 5 seeds)",
        output_dir / f"saturation_proper_PA_mean_nap{suffix}_bestconfig.png",
    )

    if "PC_mean_nap" in df.columns and df["PC_mean_nap"].notna().any():
        plot_metric(
            df, "PC_mean_nap", "PC (NAP)",
            f"PC (NAP) vs. Treatment Count{label}\n(best config per seed; band = std across 5 seeds)",
            output_dir / f"saturation_proper_PC_mean_nap{suffix}_bestconfig.png",
        )
        zoomed = df[df["n_actual"] >= 300]
        if len(zoomed) > 0:
            plot_metric(
                zoomed, "PC_mean_nap", "PC (NAP)",
                f"PC (NAP) vs. Treatment Count{label}\n(n >= 300; best config per seed; band = std across 5 seeds)",
                output_dir / f"saturation_proper_PC_mean_nap{suffix}_bestconfig_zoomed.png",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write plots (default: same as --input-dir)")
    parser.add_argument("--groups", nargs="+",
                        default=["group_low", "group_orf", "group_high", "group_crispr"])
    parser.add_argument("--include-all", action="store_true",
                        help="Also plot the no-group all-treatments file")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir is not None else args.input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for group in args.groups:
        plot_group(args.input_dir / f"saturation_results_{group}.csv",
                   output_dir, group)

    if args.include_all:
        plot_group(args.input_dir / "saturation_results.csv",
                   output_dir, None)


if __name__ == "__main__":
    main()
