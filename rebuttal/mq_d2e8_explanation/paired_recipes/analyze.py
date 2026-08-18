#!/usr/bin/env python3
"""Paired MQ-versus-D2-E8 audit of the archived Target-2 recipe grid.

This analysis reads the frozen sweep summary only. It does not rerun extraction,
normalization, retrieval, or any canonical sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = Path(
    "/work/datasets/JUMP-lite-wacv/sweeps/"
    "MAIN_RESULTS__figure_4_variance_first_v11/sweep_results.csv"
)
DEFAULT_OUTPUT = HERE / "outputs" / "release_v1"
EXPECTED_INPUT_SIZE = 1_096_114
EXPECTED_INPUT_SHA256 = "08923c7bd27bca54c0a3f484429ced31d1b48ad097c974773591a89ac63eb53a"
EXPECTED_RECIPES = 48

FAMILY_MODELS: dict[str, dict[str, str]] = {
    "cp_measure": {
        "D2-E8": "jpegxl_lossy_d2_e8_raw",
        "MQ": "jpegxl_lossy_mq_raw",
    },
    "DINOv2": {
        "D2-E8": "dinov2_jpegxl_lossy_d2_e8_raw",
        "MQ": "dinov2_jpegxl_lossy_mq_raw",
    },
    "MorphEM": {
        "D2-E8": "morphem_jpegxl_lossy_d2_e8_raw",
        "MQ": "morphem_jpegxl_lossy_mq_raw",
    },
    "OpenPhenom": {
        "D2-E8": "openphenom_jpegxl_lossy_d2_e8_raw",
        "MQ": "openphenom_jpegxl_lossy_mq_raw",
    },
    "SubCell": {
        "D2-E8": "subcell__clip01_jpegxl_lossy_d2_e8_raw",
        "MQ": "subcell__clip01_jpegxl_lossy_mq_raw",
    },
}
REQUIRED_COLUMNS = {
    "model",
    "config",
    "PA_mean_nap",
    "PC_mean_nap",
}


class AnalysisError(RuntimeError):
    """Raised when an input or analysis invariant is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_production_input(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AnalysisError(f"missing input: {path}")
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != EXPECTED_INPUT_SIZE or digest != EXPECTED_INPUT_SHA256:
        raise AnalysisError(
            "canonical sweep drift: "
            f"expected ({EXPECTED_INPUT_SIZE}, {EXPECTED_INPUT_SHA256}), "
            f"observed ({size}, {digest})"
        )
    return {"path": str(path), "size_bytes": size, "sha256": digest}


def select_and_validate_grid(
    source: pd.DataFrame, expected_recipes: int = EXPECTED_RECIPES
) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(source.columns))
    if missing:
        raise AnalysisError(f"missing required columns: {missing}")
    if source.duplicated(["model", "config"]).any():
        duplicates = source.loc[
            source.duplicated(["model", "config"], keep=False), ["model", "config"]
        ]
        raise AnalysisError(f"duplicate model/config rows: {duplicates.head().to_dict('records')}")

    records: list[pd.DataFrame] = []
    for family, codecs in FAMILY_MODELS.items():
        config_sets: dict[str, set[str]] = {}
        for codec, model in codecs.items():
            selected = source.loc[source["model"] == model].copy()
            if len(selected) != expected_recipes:
                raise AnalysisError(
                    f"{family}/{codec}: expected {expected_recipes} rows, found {len(selected)}"
                )
            if selected["config"].duplicated().any():
                raise AnalysisError(f"{family}/{codec}: duplicate config")
            values = selected[["PA_mean_nap", "PC_mean_nap"]].to_numpy(float)
            if not np.isfinite(values).all():
                raise AnalysisError(f"{family}/{codec}: non-finite PA/PC NAP")
            selected["family"] = family
            selected["codec"] = codec
            selected["nap_product"] = (
                selected["PA_mean_nap"].astype(float)
                * selected["PC_mean_nap"].astype(float)
            )
            if not np.isfinite(selected["nap_product"]).all():
                raise AnalysisError(f"{family}/{codec}: non-finite NAP product")
            config_sets[codec] = set(selected["config"].astype(str))
            records.append(
                selected[
                    [
                        "family",
                        "codec",
                        "model",
                        "config",
                        "PA_mean_nap",
                        "PC_mean_nap",
                        "nap_product",
                    ]
                ]
            )
        if config_sets["D2-E8"] != config_sets["MQ"]:
            only_d2 = sorted(config_sets["D2-E8"] - config_sets["MQ"])
            only_mq = sorted(config_sets["MQ"] - config_sets["D2-E8"])
            raise AnalysisError(
                f"{family}: configuration mismatch; only D2-E8={only_d2[:5]}, "
                f"only MQ={only_mq[:5]}"
            )

    grid = pd.concat(records, ignore_index=True)
    expected_rows = len(FAMILY_MODELS) * 2 * expected_recipes
    if len(grid) != expected_rows:
        raise AnalysisError(f"expected {expected_rows} selected rows, found {len(grid)}")
    return grid.sort_values(["family", "codec", "config"], kind="stable").reset_index(drop=True)


def build_pairs(grid: pd.DataFrame) -> pd.DataFrame:
    value_columns = ["PA_mean_nap", "PC_mean_nap", "nap_product"]
    wide = grid.pivot(index=["family", "config"], columns="codec", values=value_columns)
    expected_columns = pd.MultiIndex.from_product([value_columns, ["D2-E8", "MQ"]])
    missing = expected_columns.difference(wide.columns)
    if len(missing):
        raise AnalysisError(f"missing paired columns: {list(missing)}")
    wide = wide.dropna(subset=list(expected_columns))
    expected_rows = len(FAMILY_MODELS) * EXPECTED_RECIPES
    # Synthetic unit tests may intentionally use another complete recipe count.
    observed_counts = wide.reset_index().groupby("family", observed=True)["config"].nunique()
    if observed_counts.nunique() != 1:
        raise AnalysisError(f"unequal paired recipe counts: {observed_counts.to_dict()}")

    pairs = wide.reset_index()[["family", "config"]].copy()
    for metric in value_columns:
        pairs[f"d2e8_{metric}"] = wide[(metric, "D2-E8")].to_numpy(float)
        pairs[f"mq_{metric}"] = wide[(metric, "MQ")].to_numpy(float)
        pairs[f"delta_{metric}"] = pairs[f"mq_{metric}"] - pairs[f"d2e8_{metric}"]
    if len(pairs) != expected_rows and set(pairs["family"]) == set(FAMILY_MODELS):
        raise AnalysisError(f"expected {expected_rows} production pairs, found {len(pairs)}")
    return pairs.sort_values(["family", "config"], kind="stable").reset_index(drop=True)


def summarize(grid: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    family_rows: list[dict[str, Any]] = []
    for family in FAMILY_MODELS:
        marginal = grid.loc[grid["family"] == family]
        paired = pairs.loc[pairs["family"] == family]
        row: dict[str, Any] = {
            "family": family,
            "n_recipes": len(paired),
            "d2e8_product_median": marginal.loc[
                marginal["codec"] == "D2-E8", "nap_product"
            ].median(),
            "mq_product_median": marginal.loc[
                marginal["codec"] == "MQ", "nap_product"
            ].median(),
        }
        row["marginal_median_difference"] = (
            row["mq_product_median"] - row["d2e8_product_median"]
        )
        for metric in ("nap_product", "PA_mean_nap", "PC_mean_nap"):
            delta = paired[f"delta_{metric}"]
            row[f"paired_{metric}_mean_delta"] = delta.mean()
            row[f"paired_{metric}_median_delta"] = delta.median()
            row[f"paired_{metric}_mq_greater_fraction"] = (delta > 0).mean()
            row[f"paired_{metric}_ties"] = int((delta == 0).sum())
        family_rows.append(row)
    family_summary = pd.DataFrame(family_rows)

    d2 = grid.loc[grid["codec"] == "D2-E8", "nap_product"]
    mq = grid.loc[grid["codec"] == "MQ", "nap_product"]
    product_delta = pairs["delta_nap_product"]
    pooled = pd.DataFrame(
        [
            {
                "n_families": len(FAMILY_MODELS),
                "n_recipes_per_family": int(pairs.groupby("family").size().iloc[0]),
                "n_paired_rows": len(pairs),
                "d2e8_marginal_product_median": d2.median(),
                "mq_marginal_product_median": mq.median(),
                "marginal_median_difference": mq.median() - d2.median(),
                "paired_product_mean_delta": product_delta.mean(),
                "paired_product_median_delta": product_delta.median(),
                "paired_product_mq_greater_fraction": (product_delta > 0).mean(),
                "paired_pa_mean_delta": pairs["delta_PA_mean_nap"].mean(),
                "paired_pa_median_delta": pairs["delta_PA_mean_nap"].median(),
                "paired_pa_mq_greater_fraction": (pairs["delta_PA_mean_nap"] > 0).mean(),
                "paired_pc_mean_delta": pairs["delta_PC_mean_nap"].mean(),
                "paired_pc_median_delta": pairs["delta_PC_mean_nap"].median(),
                "paired_pc_mq_greater_fraction": (pairs["delta_PC_mean_nap"] > 0).mean(),
                "pooled_median_inversion": bool(
                    (mq.median() - d2.median()) > 0 and product_delta.median() < 0
                ),
            }
        ]
    )
    return family_summary, pooled


def render_panel(pairs: pd.DataFrame, pooled: pd.DataFrame, png: Path, pdf: Path) -> None:
    order = list(FAMILY_MODELS)
    rng = np.random.Generator(np.random.PCG64DXSM(20_260_818))
    fig, ax = plt.subplots(figsize=(8.4, 5.7))
    colors = plt.get_cmap("Set2")(np.linspace(0, 1, len(order)))
    for index, (family, color) in enumerate(zip(order, colors, strict=True)):
        delta = pairs.loc[pairs["family"] == family, "delta_nap_product"].to_numpy()
        jitter = rng.uniform(-0.16, 0.16, size=len(delta))
        ax.scatter(
            index + jitter,
            delta,
            s=20,
            facecolor=color,
            edgecolor="black",
            linewidth=0.35,
            alpha=0.68,
            zorder=2,
        )
        ax.plot(
            [index - 0.22, index + 0.22],
            [np.median(delta), np.median(delta)],
            color="black",
            linewidth=2.3,
            zorder=4,
        )
        ax.scatter(index, delta.mean(), marker="D", s=40, color="white", edgecolor="black", zorder=5)
        ax.text(
            index,
            ax.get_ylim()[0] if ax.get_ylim()[0] < -0.018 else -0.019,
            f"{int((delta > 0).sum())}/48 >0",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.axhline(0, color="black", linewidth=1.1, linestyle="--", zorder=1)
    ax.set_xticks(range(len(order)), order, rotation=20, ha="right")
    ax.set_ylabel("MQ − D2-E8 NAP product")
    ax.set_xlabel("Same normalization recipe paired within each family")
    row = pooled.iloc[0]
    annotation = (
        f"Marginal pooled medians: MQ {row.mq_marginal_product_median:.5f} > "
        f"D2-E8 {row.d2e8_marginal_product_median:.5f}\n"
        f"Paired deltas (n={int(row.n_paired_rows)}): mean "
        f"{row.paired_product_mean_delta:+.5f}, median "
        f"{row.paired_product_median_delta:+.5f}; "
        f"MQ > D2-E8 in {100 * row.paired_product_mq_greater_fraction:.1f}%"
    )
    ax.text(
        0.015,
        0.985,
        annotation,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.6", "alpha": 0.95},
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", linewidth=0.7)
    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={"Creator": "paired_recipes/analyze.py", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def write_report(output: Path, family: pd.DataFrame, pooled: pd.DataFrame) -> None:
    p = pooled.iloc[0]
    lines = [
        "# Paired recipe audit of MQ versus D2-E8",
        "",
        "Figure 3c pools five representation families and 48 normalization recipes per codec. "
        "The pooled marginal median is therefore not a paired codec effect.",
        "",
        "## Result",
        "",
        f"The pooled marginal median is {p.mq_marginal_product_median:.8f} for MQ and "
        f"{p.d2e8_marginal_product_median:.8f} for D2-E8 (difference "
        f"{p.marginal_median_difference:+.8f}). After matching all {int(p.n_paired_rows)} "
        "family/recipe rows, the MQ-minus-D2-E8 mean and median differences are "
        f"{p.paired_product_mean_delta:+.8f} and {p.paired_product_median_delta:+.8f}; "
        f"MQ is higher in {100 * p.paired_product_mq_greater_fraction:.2f}% of pairs.",
        "",
        "This is a pooled-median inversion: the marginal medians place MQ above D2-E8, "
        "while the typical paired recipe difference is negative. It does not support a "
        "general MQ biological advantage.",
        "",
        "## Family summaries",
        "",
        "| Family | D2-E8 median | MQ median | Paired mean delta | Paired median delta | MQ higher |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in family.itertuples(index=False):
        lines.append(
            f"| {row.family} | {row.d2e8_product_median:.6f} | "
            f"{row.mq_product_median:.6f} | {row.paired_nap_product_mean_delta:+.6f} | "
            f"{row.paired_nap_product_median_delta:+.6f} | "
            f"{100 * row.paired_nap_product_mq_greater_fraction:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "The 48 recipes are a structured sensitivity grid, not independent biological "
            "replicates. No inferential p-values are computed over recipes. This analysis "
            "does not estimate sampling uncertainty over treatments or targets and does not "
            "attribute any difference to denoising or biological improvement.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))


def write_checksums(output: Path) -> None:
    records = []
    for path in sorted(p for p in output.rglob("*") if p.is_file() and p.name != "artifact_checksums.json"):
        records.append(
            {
                "path": path.relative_to(output).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    (output / "artifact_checksums.json").write_text(json.dumps({"artifacts": records}, indent=2) + "\n")


def verify_checksums(output: Path) -> None:
    inventory = output / "artifact_checksums.json"
    if not inventory.is_file():
        raise AnalysisError(f"missing checksum inventory: {inventory}")
    payload = json.loads(inventory.read_text())
    expected_paths = set()
    for record in payload.get("artifacts", []):
        rel = record["path"]
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise AnalysisError(f"unsafe artifact path: {rel}")
        path = output / rel
        expected_paths.add(rel)
        if not path.is_file():
            raise AnalysisError(f"missing artifact: {rel}")
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise AnalysisError(f"artifact drift: {rel}")
    observed = {
        p.relative_to(output).as_posix()
        for p in output.rglob("*")
        if p.is_file() and p.name != "artifact_checksums.json"
    }
    if observed != expected_paths:
        raise AnalysisError(
            f"artifact inventory mismatch; missing={sorted(expected_paths-observed)}, "
            f"extra={sorted(observed-expected_paths)}"
        )


def run(input_path: Path, output: Path) -> None:
    identity = validate_production_input(input_path)
    source = pd.read_csv(input_path)
    grid = select_and_validate_grid(source)
    pairs = build_pairs(grid)
    family, pooled = summarize(grid, pairs)

    output.mkdir(parents=True, exist_ok=True)
    results = output / "results"
    results.mkdir(exist_ok=True)
    grid.to_csv(results / "selected_grid.csv", index=False)
    pairs.to_csv(results / "paired_config_deltas.csv", index=False)
    family.to_csv(results / "family_summary.csv", index=False)
    pooled.to_csv(results / "pooled_summary.csv", index=False)
    render_panel(pairs, pooled, output / "paired_recipes.png", output / "paired_recipes.pdf")
    provenance = {
        "analysis": "paired_recipes",
        "metric": "PA_mean_nap * PC_mean_nap",
        "contrast": "MQ - D2-E8",
        "input": identity,
        "runner": {
            "path": Path(__file__).resolve().relative_to(HERE.parents[2]).as_posix(),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "families": list(FAMILY_MODELS),
        "expected_recipes_per_family_codec": EXPECTED_RECIPES,
        "canonical_inputs_read_only": True,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    write_report(output, family, pooled)
    write_checksums(output)
    verify_checksums(output)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_production_input(args.input)
        if args.verify_only:
            verify_checksums(args.output)
        else:
            run(args.input, args.output)
    except (AnalysisError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print("Verification passed." if args.verify_only else f"Wrote release to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
