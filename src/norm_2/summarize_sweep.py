#!/usr/bin/env python3
"""Summarize sweep results by combining aggregated_results.csv files and plotting PA vs PC."""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path


def parse_folder_name(folder_name: str) -> tuple[str, str]:
    """
    Parse model and compression from folder name.

    Example: cp_measure_jump_target2_4plate_zstd_raw_features -> (cp_measure, zstd)
    """
    # Known compression types
    compressions = [
        "zstd", "jpegxl_lossy_effort_3", "jpegxl_lossy_hq",
        "jpegxl_lossy_mq", "jpegxl_lossy_lq"
    ]

    # Try to find compression in folder name
    compression = "unknown"
    for comp in compressions:
        if comp in folder_name:
            compression = comp
            break

    # Extract model (everything before _jump_)
    if "_jump_" in folder_name:
        model = folder_name.split("_jump_")[0]
    else:
        model = folder_name.split("_")[0]

    return model, compression


def main():
    parser = argparse.ArgumentParser(
        description="Summarize sweep results from aggregated_results.csv files"
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing subdirectories with aggregated_results.csv files"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output directory (default: same as input folder)"
    )
    args = parser.parse_args()

    if not args.folder.exists():
        print(f"Error: Folder {args.folder} does not exist")
        return 1

    # Find all aggregated_results.csv files
    csv_files = list(args.folder.rglob("aggregated_results.csv"))

    if not csv_files:
        print(f"No aggregated_results.csv files found in {args.folder}")
        return 1

    print(f"Found {len(csv_files)} aggregated_results.csv files:")

    # Read and combine
    dfs = []
    for csv_file in sorted(csv_files):
        # Get the parent folder name to extract model and compression
        parent_folder = csv_file.parent.name
        model, compression = parse_folder_name(parent_folder)

        print(f"  - {csv_file.relative_to(args.folder)} (model={model}, compression={compression})")

        df = pd.read_csv(csv_file)
        # Ignore clear outliers and artifacts (those with performance >= 100%)
        df_size_pre = len(df)
        df = df[
            (df["PA"] < 100) & (df["PC"] < 60)
        ]
        if len(df) < df_size_pre:
            print(f" WARNING   - Removed {df_size_pre - len(df)} outlier rows (PA = 100% or PC >= 60%)")
        df.insert(0, "model", model)
        df.insert(1, "compression", compression)
        
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    
    # Output directory
    output_dir = args.output if args.output else args.folder
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save combined CSV
    output_csv = output_dir / "sweep_summary.csv"
    combined.to_csv(output_csv, index=False)
    print(f"\nCombined {len(combined)} rows from {len(csv_files)} files")
    print(f"Saved to: {output_csv}")

    # Get best config per model/compression (highest Balance = PA * PC)
    if "PA" in combined.columns and "PC" in combined.columns:
        combined["Balance"] = combined["PA"] * combined["PC"]

        # Get best per model+compression
        best_configs = combined.loc[
            combined.groupby(["model", "compression"])["Balance"].idxmax()
        ].copy()

        best_csv = output_dir / "sweep_best_configs.csv"
        best_configs.to_csv(best_csv, index=False)
        print(f"Saved best configs to: {best_csv}")

        # Print summary table
        print("\n" + "=" * 100)
        print("BEST CONFIGURATIONS (by Balance = PA × PC)")
        print("=" * 100)
        summary_cols = ["model", "compression", "PA", "PC", "Balance"]
        if "Silhouette" in best_configs.columns:
            summary_cols.append("Silhouette")
        if "kBET" in best_configs.columns:
            summary_cols.append("kBET")
        print(best_configs[summary_cols].to_string(index=False))

        # Create PA vs PC plot
        plot_path = output_dir / "sweep_pa_vs_pc.png"
        create_pa_pc_plot(best_configs, plot_path)

        # Create detailed plot with all configs
        all_plot_path = output_dir / "sweep_pa_vs_pc_all.png"
        create_pa_pc_plot_all(combined, all_plot_path)

        # Create per-model subfigures plot
        by_model_plot_path = output_dir / "sweep_pa_vs_pc_by_model.png"
        create_pa_pc_plot_by_model(combined, by_model_plot_path)

        # Create KDE jointplot version per model
        kde_plot_path = output_dir / "sweep_pa_vs_pc_kde_by_model.png"
        create_pa_pc_kde_by_model(combined, kde_plot_path)

        # Create 5x5 KDE grid (models x compressions)
        kde_grid_path = output_dir / "sweep_pa_vs_pc_kde_grid.png"
        create_pa_pc_kde_grid(combined, kde_grid_path)

    return 0


def create_pa_pc_plot(df: pd.DataFrame, output_path: Path):
    """Create a PA vs PC scatter plot with color by model and size by compression."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get unique models and assign colors
    models = sorted(df["model"].unique())
    cmap = plt.colormaps.get_cmap("tab10")
    colors = {m: cmap(i) for i, m in enumerate(models)}

    # Get unique compressions and assign sizes
    compression_order = [
        "zstd", "jpegxl_lossy_hq", "jpegxl_lossy_effort_3",
        "jpegxl_lossy_mq", "jpegxl_lossy_lq"
    ]
    compressions = [c for c in compression_order if c in df["compression"].unique()]
    compressions += [c for c in df["compression"].unique() if c not in compression_order]

    size_min, size_max = 50, 600
    compression_sizes = {
        comp: size_max - (size_max - size_min) * i / max(1, len(compressions) - 1)
        for i, comp in enumerate(compressions)
    }

    # Plot each point
    for _, row in df.iterrows():
        ax.scatter(
            row["PC"],
            row["PA"],
            c=[colors[row["model"]]],
            s=compression_sizes.get(row["compression"], 100),
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5,
        )

    # Create legend for models (colors)
    color_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[m],
                   markersize=10, label=m)
        for m in models
    ]
    legend1 = ax.legend(
        handles=color_handles, title="Model", loc="upper left",
        bbox_to_anchor=(1.02, 1), fontsize=9
    )
    ax.add_artist(legend1)

    # Create legend for compressions (sizes)
    size_handles = [
        plt.Line2D([0], [0], marker="o", color="gray",
                   markersize=compression_sizes[comp] ** 0.5,
                   label=comp, linestyle="None")
        for comp in compressions
    ]
    ax.legend(
        handles=size_handles, title="Compression", loc="lower left",
        bbox_to_anchor=(1.02, 0), fontsize=9
    )

    ax.set_xlabel("Phenotypic Consistency (PC %)", fontsize=12)
    ax.set_ylabel("Phenotypic Activity (PA %)", fontsize=12)
    ax.set_title("PA vs PC: Best Configs by Model and Compression", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.4)
    plt.close()

    print(f"Saved PA vs PC plot to: {output_path}")


def create_pa_pc_plot_all(df: pd.DataFrame, output_path: Path):
    """Create a PA vs PC scatter plot showing ALL configurations, not just best."""
    fig, ax = plt.subplots(figsize=(12, 10))

    # Get unique models and assign colors
    models = sorted(df["model"].unique())
    cmap = plt.colormaps.get_cmap("tab10")
    colors = {m: cmap(i) for i, m in enumerate(models)}

    # Get unique compressions and assign markers
    compression_order = [
        "zstd", "jpegxl_lossy_hq", "jpegxl_lossy_effort_3",
        "jpegxl_lossy_mq", "jpegxl_lossy_lq"
    ]
    compressions = [c for c in compression_order if c in df["compression"].unique()]
    compressions += [c for c in df["compression"].unique() if c not in compression_order]

    markers = ["o", "s", "^", "D", "v", "p", "h", "*"]
    compression_markers = {comp: markers[i % len(markers)] for i, comp in enumerate(compressions)}

    # Plot each model+compression combination
    for model in models:
        for compression in compressions:
            subset = df[(df["model"] == model) & (df["compression"] == compression)]
            if len(subset) == 0:
                continue

            ax.scatter(
                subset["PC"],
                subset["PA"],
                c=[colors[model]],
                marker=compression_markers[compression],
                s=30,
                alpha=0.5,
                edgecolors="none",
            )

    # Highlight best configs
    if "Balance" in df.columns:
        best_configs = df.loc[
            df.groupby(["model", "compression"])["Balance"].idxmax()
        ]
        for _, row in best_configs.iterrows():
            ax.scatter(
                row["PC"],
                row["PA"],
                c=[colors[row["model"]]],
                marker=compression_markers[row["compression"]],
                s=200,
                alpha=1.0,
                edgecolors="black",
                linewidths=2,
            )

    # Create legend for models (colors)
    color_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors[m],
                   markersize=10, label=m)
        for m in models
    ]
    legend1 = ax.legend(
        handles=color_handles, title="Model", loc="upper left",
        bbox_to_anchor=(1.02, 1), fontsize=9
    )
    ax.add_artist(legend1)

    # Create legend for compressions (markers)
    marker_handles = [
        plt.Line2D([0], [0], marker=compression_markers[comp], color="gray",
                   markersize=8, label=comp, linestyle="None")
        for comp in compressions
    ]
    ax.legend(
        handles=marker_handles, title="Compression", loc="lower left",
        bbox_to_anchor=(1.02, 0), fontsize=9
    )

    ax.set_xlabel("Phenotypic Consistency (PC %)", fontsize=12)
    ax.set_ylabel("Phenotypic Activity (PA %)", fontsize=12)
    ax.set_title("PA vs PC: All Configurations (best highlighted)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.4)
    plt.close()

    print(f"Saved all-configs PA vs PC plot to: {output_path}")


def create_pa_pc_plot_by_model(df: pd.DataFrame, output_path: Path):
    """Create PA vs PC scatter plot with 5 subfigures (one per model), colors by compression."""
    models = sorted(df["model"].unique())
    n_models = len(models)

    # Create subfigures - arrange in a row or 2x3 grid depending on count
    if n_models <= 3:
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    else:
        ncols = 3
        nrows = (n_models + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
        axes = axes.flatten()

    # Handle single model case
    if n_models == 1:
        axes = [axes]

    # Get unique compressions and assign colors
    compression_order = [
        "zstd", "jpegxl_lossy_hq", "jpegxl_lossy_effort_3",
        "jpegxl_lossy_mq", "jpegxl_lossy_lq"
    ]
    compressions = [c for c in compression_order if c in df["compression"].unique()]
    compressions += [c for c in df["compression"].unique() if c not in compression_order]

    cmap = plt.colormaps.get_cmap("tab10")
    compression_colors = {comp: cmap(i) for i, comp in enumerate(compressions)}

    # Calculate global axis limits for consistency
    pc_min, pc_max = df["PC"].min(), df["PC"].max()
    pa_min, pa_max = df["PA"].min(), df["PA"].max()
    pc_margin = (pc_max - pc_min) * 0.05
    pa_margin = (pa_max - pa_min) * 0.05

    # Plot each model in its own subfigure
    for idx, model in enumerate(models):
        ax = axes[idx]
        model_df = df[df["model"] == model]

        for compression in compressions:
            subset = model_df[model_df["compression"] == compression]
            if len(subset) == 0:
                continue

            ax.scatter(
                subset["PC"],
                subset["PA"],
                c=[compression_colors[compression]],
                s=30,
                alpha=0.5,
                edgecolors="none",
                label=compression,
            )

        # Highlight best configs for this model
        if "Balance" in model_df.columns and len(model_df) > 0:
            best_configs = model_df.loc[
                model_df.groupby("compression")["Balance"].idxmax()
            ]
            for _, row in best_configs.iterrows():
                ax.scatter(
                    row["PC"],
                    row["PA"],
                    c=[compression_colors[row["compression"]]],
                    s=200,
                    alpha=1.0,
                    edgecolors="black",
                    linewidths=2,
                )

        ax.set_xlim(pc_min - pc_margin, pc_max + pc_margin)
        ax.set_ylim(pa_min - pa_margin, pa_max + pa_margin)
        ax.set_xlabel("Phenotypic Consistency (PC %)", fontsize=10)
        ax.set_ylabel("Phenotypic Activity (PA %)", fontsize=10)
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    # Hide unused axes if any
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    # Create shared legend
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=compression_colors[comp],
                   markersize=10, label=comp)
        for comp in compressions
    ]
    fig.legend(
        handles=legend_handles, title="Compression", loc="lower center",
        bbox_to_anchor=(0.5, -0.02), ncol=len(compressions), fontsize=9
    )

    fig.suptitle("PA vs PC by Model (best highlighted)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.4)
    plt.close()

    print(f"Saved by-model PA vs PC plot to: {output_path}")


def create_pa_pc_kde_by_model(df: pd.DataFrame, output_path: Path):
    """Create PA vs PC KDE jointplots, one per model, using seaborn."""
    sns.set_theme(style="ticks")

    models = sorted(df["model"].unique())
    n_models = len(models)

    # Get unique compressions in order
    compression_order = [
        "zstd", "jpegxl_lossy_hq", "jpegxl_lossy_effort_3",
        "jpegxl_lossy_mq", "jpegxl_lossy_lq"
    ]
    compressions = [c for c in compression_order if c in df["compression"].unique()]
    compressions += [c for c in df["compression"].unique() if c not in compression_order]

    # Use a distinct color palette for compressions
    distinct_colors = [
        "#e41a1c",  # red
        "#377eb8",  # blue
        "#4daf4a",  # green
        "#984ea3",  # purple
        "#ff7f00",  # orange
        "#a65628",  # brown
        "#f781bf",  # pink
        "#999999",  # gray
    ]
    compression_colors = {comp: distinct_colors[i % len(distinct_colors)] for i, comp in enumerate(compressions)}

    # Calculate global axis limits for consistency
    pc_min, pc_max = df["PC"].min(), df["PC"].max()
    pa_min, pa_max = df["PA"].min(), df["PA"].max()
    pc_margin = (pc_max - pc_min) * 0.1
    pa_margin = (pa_max - pa_min) * 0.1

    # Create a figure with subfigures for each model
    if n_models <= 3:
        ncols = n_models
        nrows = 1
    else:
        ncols = 3
        nrows = (n_models + ncols - 1) // ncols

    fig = plt.figure(figsize=(6 * ncols, 6 * nrows))
    subfigs = fig.subfigures(nrows, ncols, wspace=0.1, hspace=0.1)

    # Flatten subfigs for easier indexing
    if n_models == 1:
        subfigs = [subfigs]
    elif nrows == 1 or ncols == 1:
        subfigs = list(subfigs)
    else:
        subfigs = subfigs.flatten()

    for idx, model in enumerate(models):
        model_df = df[df["model"] == model].copy()

        # Create jointplot on the subfigure
        subfig = subfigs[idx]

        # Use JointGrid for more control
        g = sns.JointGrid(
            data=model_df,
            x="PC",
            y="PA",
            height=5,
        )

        # Plot KDE for each compression
        for compression in compressions:
            subset = model_df[model_df["compression"] == compression]
            if len(subset) < 2:
                continue

            color = compression_colors[compression]
            try:
                sns.kdeplot(
                    data=subset,
                    x="PC",
                    y="PA",
                    ax=g.ax_joint,
                    label=compression,
                    color=color,
                    fill=True,
                    alpha=0.3,
                    levels=5,
                )
                sns.kdeplot(
                    data=subset,
                    x="PC",
                    ax=g.ax_marg_x,
                    color=color,
                    fill=True,
                    alpha=0.3,
                )
                sns.kdeplot(
                    data=subset,
                    y="PA",
                    ax=g.ax_marg_y,
                    color=color,
                    fill=True,
                    alpha=0.3,
                )
            except Exception:
                # Fall back to scatter if KDE fails (not enough data)
                g.ax_joint.scatter(subset["PC"], subset["PA"], alpha=0.5, label=compression, s=20, color=color)

        g.ax_joint.set_xlim(pc_min - pc_margin, pc_max + pc_margin)
        g.ax_joint.set_ylim(pa_min - pa_margin, pa_max + pa_margin)
        g.ax_joint.set_xlabel("Phenotypic Consistency (PC %)", fontsize=10)
        g.ax_joint.set_ylabel("Phenotypic Activity (PA %)", fontsize=10)

        # Create manual legend handles
        legend_handles = [
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=compression_colors[comp],
                       markersize=8, label=comp)
            for comp in compressions if comp in model_df["compression"].values
        ]
        if legend_handles:
            g.ax_joint.legend(handles=legend_handles, title="Compression", fontsize=8, title_fontsize=9)

        # Copy the jointgrid figure content to our subfigure
        # Save individual model plot
        model_output = output_path.parent / f"{output_path.stem}_{model}.png"
        g.figure.suptitle(model, fontsize=12, fontweight="bold")
        g.savefig(model_output, dpi=150, bbox_inches="tight")
        plt.close(g.figure)
        print(f"Saved KDE plot for {model} to: {model_output}")

    plt.close(fig)

    # Also create a combined grid version using a simpler approach
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if n_models == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    axes_flat = [ax for row in axes for ax in row]

    for idx, model in enumerate(models):
        ax = axes_flat[idx]
        model_df = df[df["model"] == model].copy()

        for compression in compressions:
            subset = model_df[model_df["compression"] == compression]
            if len(subset) < 2:
                continue

            color = compression_colors[compression]
            try:
                sns.kdeplot(
                    data=subset,
                    x="PC",
                    y="PA",
                    ax=ax,
                    label=compression,
                    color=color,
                    fill=True,
                    alpha=0.3,
                    levels=5,
                )
            except Exception:
                ax.scatter(subset["PC"], subset["PA"], alpha=0.5, label=compression, s=20, color=color)

        ax.set_xlim(pc_min - pc_margin, pc_max + pc_margin)
        ax.set_ylim(pa_min - pa_margin, pa_max + pa_margin)
        ax.set_xlabel("PC %", fontsize=10)
        ax.set_ylabel("PA %", fontsize=10)
        ax.set_title(model, fontsize=12, fontweight="bold")

        # Create manual legend handles
        legend_handles = [
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=compression_colors[comp],
                       markersize=8, label=comp)
            for comp in compressions if comp in model_df["compression"].values
        ]
        if legend_handles:
            ax.legend(handles=legend_handles, title="Compression", fontsize=7, title_fontsize=8)

    # Hide unused axes
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("PA vs PC KDE by Model", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.4)
    plt.close()

    print(f"Saved combined KDE plot to: {output_path}")


def create_pa_pc_kde_grid(df: pd.DataFrame, output_path: Path):
    """Create a 5x5 KDE grid with models as rows and compressions as columns."""
    sns.set_theme(style="ticks")

    models = sorted(df["model"].unique())

    # Get unique compressions in order
    compression_order = [
        "zstd", "jpegxl_lossy_hq", "jpegxl_lossy_effort_3",
        "jpegxl_lossy_mq", "jpegxl_lossy_lq"
    ]
    compressions = [c for c in compression_order if c in df["compression"].unique()]
    compressions += [c for c in df["compression"].unique() if c not in compression_order]

    n_models = len(models)
    n_compressions = len(compressions)

    # Use distinct colors for compressions (for consistency with other plots)
    distinct_colors = [
        "#e41a1c",  # red
        "#377eb8",  # blue
        "#4daf4a",  # green
        "#984ea3",  # purple
        "#ff7f00",  # orange
        "#a65628",  # brown
        "#f781bf",  # pink
        "#999999",  # gray
    ]
    compression_colors = {comp: distinct_colors[i % len(distinct_colors)] for i, comp in enumerate(compressions)}

    # Calculate global axis limits for consistency
    pc_min, pc_max = df["PC"].min(), df["PC"].max()
    pa_min, pa_max = df["PA"].min(), df["PA"].max()
    pc_margin = (pc_max - pc_min) * 0.1
    pa_margin = (pa_max - pa_min) * 0.1

    # Create grid: rows = models, columns = compressions
    fig, axes = plt.subplots(
        n_models, n_compressions,
        figsize=(3 * n_compressions, 3 * n_models),
        sharex=True, sharey=True
    )

    # Handle edge cases
    if n_models == 1 and n_compressions == 1:
        axes = [[axes]]
    elif n_models == 1:
        axes = [axes]
    elif n_compressions == 1:
        axes = [[ax] for ax in axes]

    for row_idx, model in enumerate(models):
        for col_idx, compression in enumerate(compressions):
            ax = axes[row_idx][col_idx]
            subset = df[(df["model"] == model) & (df["compression"] == compression)]

            color = compression_colors[compression]

            if len(subset) >= 2:
                try:
                    sns.kdeplot(
                        data=subset,
                        x="PC",
                        y="PA",
                        ax=ax,
                        color=color,
                        fill=True,
                        alpha=0.5,
                        levels=5,
                    )
                except Exception:
                    ax.scatter(subset["PC"], subset["PA"], alpha=0.5, s=15, color=color)
            elif len(subset) == 1:
                ax.scatter(subset["PC"], subset["PA"], alpha=0.7, s=30, color=color)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="gray")

            ax.set_xlim(pc_min - pc_margin, pc_max + pc_margin)
            ax.set_ylim(pa_min - pa_margin, pa_max + pa_margin)

            # Only show axis labels on edges
            if row_idx == n_models - 1:
                ax.set_xlabel("PC %", fontsize=9)
            else:
                ax.set_xlabel("")
            if col_idx == 0:
                ax.set_ylabel("PA %", fontsize=9)
            else:
                ax.set_ylabel("")

            # Add row labels (model names) on the right
            if col_idx == n_compressions - 1:
                ax.annotate(
                    model, xy=(1.05, 0.5), xycoords="axes fraction",
                    fontsize=10, fontweight="bold", va="center", rotation=-90
                )

            # Add column labels (compression names) on top
            if row_idx == 0:
                ax.set_title(compression, fontsize=9, fontweight="bold", color=color)

    fig.suptitle("PA vs PC KDE: Models × Compressions", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.4)
    plt.close()

    print(f"Saved KDE grid plot to: {output_path}")


if __name__ == "__main__":
    exit(main())
