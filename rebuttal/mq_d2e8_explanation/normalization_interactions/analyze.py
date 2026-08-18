#!/usr/bin/env python3
"""Describe normalization-recipe interactions in Target-2 MQ vs D2-E8.

This analysis reads the archived Target-2 sweep table only. It does not fit,
normalize, extract, or rescore profiles. The Figure-3c estimand is the unscaled
NAP product, PA_mean_nap * PC_mean_nap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

INPUT = Path(
    "/work/datasets/JUMP-lite-wacv/sweeps/"
    "MAIN_RESULTS__figure_4_variance_first_v11/sweep_results.csv"
)
INPUT_SIZE = 1_096_114
INPUT_SHA256 = "08923c7bd27bca54c0a3f484429ced31d1b48ad097c974773591a89ac63eb53a"
EXPECTED_INPUT_ROWS = 2_860
EXPECTED_CONFIGS = 48
RELEASE_FILES = {
    "REPORT.md",
    "artifact_checksums.json",
    "factor_summary.csv",
    "family_factor_summary.csv",
    "family_summary.csv",
    "normalization_interactions.pdf",
    "normalization_interactions.png",
    "paired_recipe_deltas.csv",
    "provenance.json",
    "recipe_sign_consistency.csv",
    "variation_decomposition.csv",
}
ARTIFACT_FILES = RELEASE_FILES - {"artifact_checksums.json"}

MODEL_PAIRS = {
    "cp_measure": (
        "jpegxl_lossy_d2_e8_raw",
        "jpegxl_lossy_mq_raw",
    ),
    "DINOv2": (
        "dinov2_jpegxl_lossy_d2_e8_raw",
        "dinov2_jpegxl_lossy_mq_raw",
    ),
    "MorphEM": (
        "morphem_jpegxl_lossy_d2_e8_raw",
        "morphem_jpegxl_lossy_mq_raw",
    ),
    "OpenPhenom": (
        "openphenom_jpegxl_lossy_d2_e8_raw",
        "openphenom_jpegxl_lossy_mq_raw",
    ),
    "SubCell": (
        "subcell__clip01_jpegxl_lossy_d2_e8_raw",
        "subcell__clip01_jpegxl_lossy_mq_raw",
    ),
}
FAMILY_ORDER = list(MODEL_PAIRS)

REQUIRED_COLUMNS = {
    "model",
    "config",
    "PA_mean_nap",
    "PC_mean_nap",
    "norm_method",
    "outlier_cutoff",
    "use_int",
    "prune_thresh",
    "use_pca",
    "pca_components",
    "batch_method",
    "tvn_epsilon",
    "tvn_efaar_n_components",
}
FACTOR_COLUMNS = [
    "norm_method",
    "outlier_cutoff",
    "use_int",
    "prune_thresh",
    "use_pca",
    "pca_components",
    "batch_method",
    "tvn_epsilon",
    "tvn_efaar_n_components",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def validate_input(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != INPUT_SIZE:
        raise ValueError(f"input size drift: {path.stat().st_size} != {INPUT_SIZE}")
    digest = sha256_file(path)
    if digest != INPUT_SHA256:
        raise ValueError(f"input hash drift: {digest} != {INPUT_SHA256}")


def _same_with_nan(a: pd.Series, b: pd.Series) -> bool:
    return bool(((a == b) | (a.isna() & b.isna())).all())


def _fit_scope(config: str) -> str:
    match = re.search(r"_(all|ctrl)(?:__|$)", config)
    if not match:
        raise ValueError(f"cannot parse fit scope from config: {config}")
    return match.group(1)


def _number_label(value: float | int, *, none: str = "none") -> str:
    if pd.isna(value):
        return none
    value_f = float(value)
    if value_f.is_integer():
        return str(int(value_f))
    return f"{value_f:g}"


def build_paired_table(
    frame: pd.DataFrame,
    model_pairs: dict[str, tuple[str, str]] = MODEL_PAIRS,
    expected_configs: int = EXPECTED_CONFIGS,
) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if frame.duplicated(["model", "config"]).any():
        dup = frame.loc[frame.duplicated(["model", "config"], keep=False), ["model", "config"]]
        raise ValueError(f"duplicate model/config rows: {dup.head().to_dict('records')}")

    records: list[pd.DataFrame] = []
    for family, (d2_model, mq_model) in model_pairs.items():
        d2 = frame.loc[frame["model"] == d2_model].copy()
        mq = frame.loc[frame["model"] == mq_model].copy()
        if len(d2) != expected_configs or len(mq) != expected_configs:
            raise ValueError(
                f"{family} coverage drift: D2-E8={len(d2)}, MQ={len(mq)}, "
                f"expected={expected_configs}"
            )
        if set(d2["config"]) != set(mq["config"]):
            only_d2 = sorted(set(d2["config"]) - set(mq["config"]))
            only_mq = sorted(set(mq["config"]) - set(d2["config"]))
            raise ValueError(f"{family} config mismatch: only D2={only_d2}, only MQ={only_mq}")

        merged = d2.merge(mq, on="config", how="inner", suffixes=("_d2e8", "_mq"), validate="one_to_one")
        for col in FACTOR_COLUMNS:
            if not _same_with_nan(merged[f"{col}_d2e8"], merged[f"{col}_mq"]):
                raise ValueError(f"{family} factor drift across codecs: {col}")
            merged[col] = merged[f"{col}_d2e8"]

        for codec in ("d2e8", "mq"):
            pa = pd.to_numeric(merged[f"PA_mean_nap_{codec}"], errors="raise")
            pc = pd.to_numeric(merged[f"PC_mean_nap_{codec}"], errors="raise")
            if not np.isfinite(pa).all() or not np.isfinite(pc).all():
                raise ValueError(f"{family} has non-finite {codec} PA/PC")
            merged[f"product_{codec}"] = pa * pc

        merged["family"] = family
        merged["delta_mq_minus_d2e8"] = merged["product_mq"] - merged["product_d2e8"]
        merged["delta_pa_mq_minus_d2e8"] = merged["PA_mean_nap_mq"] - merged["PA_mean_nap_d2e8"]
        merged["delta_pc_mq_minus_d2e8"] = merged["PC_mean_nap_mq"] - merged["PC_mean_nap_d2e8"]
        merged["fit_scope"] = merged["config"].map(_fit_scope)
        merged["normalization"] = merged["norm_method"].map(
            {"robustmad": "Robust MAD", "standardize": "Standardize"}
        )
        if merged["normalization"].isna().any():
            raise ValueError(f"{family} has unknown normalization method")
        merged["prune_value"] = merged["prune_thresh"].map(_number_label)
        # Align each family's two pruning settings by relative intensity. cp_measure
        # compares 0.90/0.95; learned representations compare none/0.90.
        unique_prune = sorted(
            merged["prune_thresh"].drop_duplicates().tolist(),
            key=lambda x: -math.inf if pd.isna(x) else float(x),
        )
        if len(unique_prune) != 2:
            raise ValueError(f"{family} expected two pruning levels, found {unique_prune}")
        prune_map = {
            ("none" if pd.isna(unique_prune[0]) else _number_label(unique_prune[0])): "lower",
            ("none" if pd.isna(unique_prune[1]) else _number_label(unique_prune[1])): "higher",
        }
        merged["prune_relative"] = merged["prune_value"].map(prune_map)
        merged["tvn_components"] = merged["tvn_efaar_n_components"].astype(int)
        merged["tvn_epsilon_label"] = merged["tvn_epsilon"].map(lambda x: f"{float(x):g}")
        merged["recipe_signature"] = (
            merged["normalization"].map({"Robust MAD": "R", "Standardize": "S"})
            + "-"
            + merged["fit_scope"].map({"all": "A", "ctrl": "C"})
            + "-"
            + merged["prune_relative"].map({"lower": "L", "higher": "H"})
            + "-e"
            + merged["tvn_epsilon_label"]
            + "-c"
            + merged["tvn_components"].astype(str)
        )
        records.append(merged)

    paired = pd.concat(records, ignore_index=True)
    expected_rows = len(model_pairs) * expected_configs
    if len(paired) != expected_rows:
        raise AssertionError(f"paired row count {len(paired)} != {expected_rows}")

    # Every family must instantiate the same 48 structural signatures. Exact
    # pruning values remain in prune_value; only relative intensity is aligned.
    signature_sets = {
        family: set(group["recipe_signature"])
        for family, group in paired.groupby("family", sort=False)
    }
    first = next(iter(signature_sets.values()))
    if len(first) != expected_configs or any(values != first for values in signature_sets.values()):
        raise ValueError("family recipe-structure signatures are not aligned")

    order_cols = [
        pd.Categorical(paired["normalization"], ["Robust MAD", "Standardize"], ordered=True),
        pd.Categorical(paired["fit_scope"], ["all", "ctrl"], ordered=True),
        pd.Categorical(paired["prune_relative"], ["lower", "higher"], ordered=True),
        paired["tvn_epsilon"],
        paired["tvn_components"],
    ]
    order_frame = pd.DataFrame(
        {
            "recipe_signature": paired["recipe_signature"],
            "normalization_order": order_cols[0].codes,
            "scope_order": order_cols[1].codes,
            "prune_order": order_cols[2].codes,
            "epsilon_order": order_cols[3],
            "components_order": order_cols[4],
        }
    ).drop_duplicates()
    order_frame = order_frame.sort_values(
        ["normalization_order", "scope_order", "prune_order", "epsilon_order", "components_order"]
    ).reset_index(drop=True)
    order_frame["recipe_order"] = np.arange(len(order_frame))
    paired = paired.merge(order_frame[["recipe_signature", "recipe_order"]], on="recipe_signature", validate="many_to_one")
    paired["family"] = pd.Categorical(paired["family"], list(model_pairs), ordered=True)
    return paired.sort_values(["family", "recipe_order"]).reset_index(drop=True)


def summarize_factors(paired: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    factor_specs = {
        "normalization": "normalization",
        "fit_scope": "fit_scope",
        "relative_pruning": "prune_relative",
        "exact_pruning": "prune_value",
        "TVN_epsilon": "tvn_epsilon_label",
        "TVN_components": "tvn_components",
    }
    rows: list[dict] = []
    family_rows: list[dict] = []
    for factor_name, column in factor_specs.items():
        for level, group in paired.groupby(column, dropna=False, observed=True):
            values = group["delta_mq_minus_d2e8"].to_numpy()
            rows.append(
                {
                    "factor": factor_name,
                    "level": str(level),
                    "n_family_recipe_cells": len(values),
                    "mean_delta": values.mean(),
                    "median_delta": np.median(values),
                    "positive_fraction": (values > 0).mean(),
                    "minimum_delta": values.min(),
                    "maximum_delta": values.max(),
                }
            )
        for (family, level), group in paired.groupby(["family", column], dropna=False, observed=True):
            values = group["delta_mq_minus_d2e8"].to_numpy()
            family_rows.append(
                {
                    "family": str(family),
                    "factor": factor_name,
                    "level": str(level),
                    "n_recipes": len(values),
                    "mean_delta": values.mean(),
                    "median_delta": np.median(values),
                    "positive_fraction": (values > 0).mean(),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(family_rows)


def two_way_decomposition(paired: pd.DataFrame) -> pd.DataFrame:
    matrix = paired.pivot(index="family", columns="recipe_signature", values="delta_mq_minus_d2e8")
    if matrix.isna().any().any():
        raise ValueError("incomplete family-by-recipe matrix")
    values = matrix.to_numpy(dtype=float)
    grand = values.mean()
    family_means = values.mean(axis=1)
    recipe_means = values.mean(axis=0)
    fitted = family_means[:, None] + recipe_means[None, :] - grand
    residual = values - fitted
    ss_total = float(np.square(values - grand).sum())
    ss_family = float(values.shape[1] * np.square(family_means - grand).sum())
    ss_recipe = float(values.shape[0] * np.square(recipe_means - grand).sum())
    ss_interaction = float(np.square(residual).sum())
    if not np.isclose(ss_total, ss_family + ss_recipe + ss_interaction, rtol=1e-10, atol=1e-14):
        raise AssertionError("two-way sums of squares do not close")
    rows = [
        ("family", ss_family, values.shape[0] - 1),
        ("recipe_structure", ss_recipe, values.shape[1] - 1),
        ("family_by_recipe_residual", ss_interaction, (values.shape[0] - 1) * (values.shape[1] - 1)),
    ]
    return pd.DataFrame(
        [
            {
                "component": name,
                "sum_squares": ss,
                "degrees_of_freedom": df,
                "fraction_of_total_variation": ss / ss_total if ss_total else np.nan,
            }
            for name, ss, df in rows
        ]
    )


def summarize_signs(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signature, group in paired.groupby("recipe_signature", observed=True):
        values = group["delta_mq_minus_d2e8"].to_numpy()
        rows.append(
            {
                "recipe_signature": signature,
                "recipe_order": int(group["recipe_order"].iloc[0]),
                "positive_families": int((values > 0).sum()),
                "negative_families": int((values < 0).sum()),
                "zero_families": int((values == 0).sum()),
                "mean_delta": values.mean(),
                "median_delta": np.median(values),
                "minimum_delta": values.min(),
                "maximum_delta": values.max(),
                "unanimous_sign": bool((values > 0).all() or (values < 0).all()),
            }
        )
    return pd.DataFrame(rows).sort_values("recipe_order").reset_index(drop=True)


def plot_panel(paired: pd.DataFrame, output_png: Path, output_pdf: Path) -> None:
    ordered = paired.sort_values("recipe_order").drop_duplicates("recipe_signature")
    signatures = ordered["recipe_signature"].tolist()
    matrix = (
        paired.pivot(index="family", columns="recipe_signature", values="delta_mq_minus_d2e8")
        .reindex(index=FAMILY_ORDER, columns=signatures)
    )
    family_means = matrix.mean(axis=1)
    vmax = float(np.nanmax(np.abs(matrix.to_numpy())))

    sns.set_theme(style="white", context="paper")
    # Deliberately wide and vector-backed so the 24 paired structural labels
    # remain readable when this panel is used at full manuscript text width.
    fig = plt.figure(figsize=(10.5, 4.6))
    grid = fig.add_gridspec(1, 2, width_ratios=[9.2, 1.45], wspace=0.20)
    ax = fig.add_subplot(grid[0, 0])
    cax = ax.inset_axes([0.02, -0.25, 0.52, 0.06])
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        vmin=-vmax,
        vmax=vmax,
        linewidths=0.05,
        linecolor="white",
        cbar=True,
        cbar_ax=cax,
        cbar_kws={"orientation": "horizontal", "label": "MQ − D2-E8 NAP product"},
        xticklabels=False,
        yticklabels=True,
        rasterized=False,
    )
    ax.set_xlabel("48 deterministic recipe structures", fontsize=9, fontweight="bold")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8, rotation=0)
    cax.tick_params(labelsize=6)
    cax.xaxis.label.set_size(7)

    # The 48 columns are 2 normalizations × 2 fit scopes × 2 relative
    # pruning levels × 3 epsilons × 2 component counts.
    for boundary, width in ((24, 1.2), (12, 0.8), (36, 0.8)):
        ax.axvline(boundary, color="black", lw=width)
    for boundary in (6, 18, 30, 42):
        ax.axvline(boundary, color="black", lw=0.45, alpha=0.65)
    for boundary in range(2, 48, 2):
        ax.axvline(boundary, color="black", lw=0.16, alpha=0.25)

    # Label 24 component-pairs rather than all 48 cells.
    pair_labels = []
    pair_positions = []
    for start in range(0, 48, 2):
        row = ordered.iloc[start]
        norm = "R" if row["normalization"] == "Robust MAD" else "S"
        scope = "A" if row["fit_scope"] == "all" else "C"
        prune = "L" if row["prune_relative"] == "lower" else "H"
        pair_labels.append(f"{norm}{scope}{prune}\n{float(row['tvn_epsilon']):g}")
        pair_positions.append(start + 1)
    ax.set_xticks(pair_positions)
    ax.set_xticklabels(pair_labels, fontsize=6.5, rotation=90, ha="center")
    ax.tick_params(axis="x", length=0, pad=1)
    ax.text(0.5, 1.03, "R/S: robust MAD/standardize; A/C: all/control fit; L/H: lower/higher pruning; each pair: c96/c128",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7.2)

    ax_bar = fig.add_subplot(grid[0, 1])
    colors = ["#b2182b" if value > 0 else "#2166ac" for value in family_means]
    y = np.arange(len(FAMILY_ORDER))
    ax_bar.barh(y, family_means.to_numpy(), color=colors, height=0.65)
    ax_bar.axvline(0, color="black", lw=0.8)
    ax_bar.set_yticks(y)
    ax_bar.set_yticklabels([])
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Mean\ndelta", fontsize=7, fontweight="bold")
    ax_bar.tick_params(axis="x", labelsize=6)
    ax_bar.spines[["top", "right", "left"]].set_visible(False)
    ax_bar.set_xlim(-0.0086, 0.0032)
    for yi, value in zip(y, family_means):
        ax_bar.text(
            0.0030,
            yi,
            f"{value:+.4f}",
            ha="right",
            va="center",
            fontsize=5.5,
            clip_on=True,
        )

    fig.subplots_adjust(left=0.09, right=0.99, top=0.87, bottom=0.30)
    metadata = {
        "Title": "Target-2 MQ minus D2-E8 normalization-recipe interactions",
        "Author": "JUMP-lite",
        "Subject": "Archived deterministic sensitivity analysis",
        "Keywords": "Target-2, MQ, D2-E8, normalization",
        "CreationDate": datetime(2026, 8, 18, tzinfo=timezone.utc),
        "ModDate": datetime(2026, 8, 18, tzinfo=timezone.utc),
    }
    fig.savefig(output_png, dpi=300, facecolor="white")
    fig.savefig(output_pdf, metadata=metadata, facecolor="white")
    plt.close(fig)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _artifact_records(paths: Iterable[Path], root: Path) -> list[dict[str, object]]:
    records = []
    for path in sorted(paths):
        records.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def verify_release(
    output_dir: Path,
    input_path: Path = INPUT,
    source_path: Path | None = None,
) -> None:
    """Fail closed on input, source, inventory, provenance, or artifact drift."""
    source_path = Path(__file__) if source_path is None else source_path
    validate_input(input_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not output_dir.is_dir() or output_dir.is_symlink():
        raise ValueError(f"release directory missing or unsafe: {output_dir}")
    entries = list(output_dir.iterdir())
    if any(not entry.is_file() or entry.is_symlink() for entry in entries):
        raise ValueError("release inventory contains a directory or symlink")
    actual = {entry.name for entry in entries}
    if actual != RELEASE_FILES:
        raise ValueError(
            f"release inventory drift: missing={sorted(RELEASE_FILES - actual)}, "
            f"extra={sorted(actual - RELEASE_FILES)}"
        )

    manifest = json.loads((output_dir / "artifact_checksums.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported checksum-manifest schema")
    expected_input = {
        "path": str(input_path),
        "bytes": INPUT_SIZE,
        "sha256": INPUT_SHA256,
        "rows": EXPECTED_INPUT_ROWS,
    }
    if manifest.get("input") != expected_input:
        raise ValueError("checksum-manifest input identity drift")
    expected_source = {"path": "analyze.py", "sha256": sha256_file(source_path)}
    if manifest.get("source") != expected_source:
        raise ValueError("checksum-manifest source identity drift")
    records = manifest.get("artifacts")
    if not isinstance(records, list) or {record.get("path") for record in records} != ARTIFACT_FILES:
        raise ValueError("checksum-manifest artifact inventory drift")
    if len(records) != len(ARTIFACT_FILES):
        raise ValueError("duplicate checksum-manifest artifact record")
    for record in records:
        rel = record["path"]
        if Path(rel).name != rel:
            raise ValueError(f"unsafe artifact path: {rel}")
        path = output_dir / rel
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"artifact size drift: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"artifact hash drift: {path}")

    provenance = json.loads((output_dir / "provenance.json").read_text())
    if provenance.get("input") != expected_input:
        raise ValueError("provenance input identity drift")
    if provenance.get("source_sha256") != expected_source["sha256"]:
        raise ValueError("provenance source identity drift")
    if provenance.get("paired_rows") != len(MODEL_PAIRS) * EXPECTED_CONFIGS:
        raise ValueError("provenance paired-row drift")
    paired = pd.read_csv(output_dir / "paired_recipe_deltas.csv")
    if len(paired) != len(MODEL_PAIRS) * EXPECTED_CONFIGS:
        raise ValueError("paired output row-count drift")
    if paired.duplicated(["family", "config"]).any():
        raise ValueError("paired output key duplication")
    if set(paired["family"]) != set(FAMILY_ORDER):
        raise ValueError("paired output family drift")
    if paired.groupby("family").size().to_dict() != {family: EXPECTED_CONFIGS for family in FAMILY_ORDER}:
        raise ValueError("paired output family coverage drift")
    decomposition = pd.read_csv(output_dir / "variation_decomposition.csv")
    if len(decomposition) != 3 or not np.isclose(
        decomposition["fraction_of_total_variation"].sum(), 1.0, rtol=0.0, atol=2e-12
    ):
        raise ValueError("variation decomposition drift")
    signs = pd.read_csv(output_dir / "recipe_sign_consistency.csv")
    if len(signs) != EXPECTED_CONFIGS or signs["recipe_order"].nunique() != EXPECTED_CONFIGS:
        raise ValueError("recipe-sign output drift")


def _publish_staged(staged: Path, output_dir: Path) -> None:
    """Replace a release directory by same-filesystem rename, restoring on error."""
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    if output_dir.exists():
        if not output_dir.is_dir() or output_dir.is_symlink():
            raise ValueError(f"refusing to replace unsafe output path: {output_dir}")
        backup = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.backup-", dir=output_dir.parent))
        backup.rmdir()
        output_dir.rename(backup)
    try:
        staged.rename(output_dir)
    except Exception:
        if backup is not None and backup.exists() and not output_dir.exists():
            backup.rename(output_dir)
        raise
    else:
        if backup is not None:
            shutil.rmtree(backup)


def _generate_release(input_path: Path, output_dir: Path, source_path: Path) -> None:
    if any(output_dir.iterdir()):
        raise ValueError(f"staging directory is not empty: {output_dir}")
    frame = pd.read_csv(input_path)
    if len(frame) != EXPECTED_INPUT_ROWS:
        raise ValueError(f"input row count drift: {len(frame)} != {EXPECTED_INPUT_ROWS}")
    paired = build_paired_table(frame)
    factor_summary, family_factor_summary = summarize_factors(paired)
    decomposition = two_way_decomposition(paired)
    signs = summarize_signs(paired)

    output_dir.mkdir(parents=True, exist_ok=True)
    paired_columns = [
        "family",
        "config",
        "recipe_signature",
        "recipe_order",
        "normalization",
        "fit_scope",
        "prune_value",
        "prune_relative",
        "tvn_epsilon",
        "tvn_components",
        "batch_method",
        "use_pca",
        "pca_components",
        "product_d2e8",
        "product_mq",
        "delta_mq_minus_d2e8",
        "delta_pa_mq_minus_d2e8",
        "delta_pc_mq_minus_d2e8",
    ]
    paired_out = paired[paired_columns].copy()
    paired_out["family"] = paired_out["family"].astype(str)
    paired_out.to_csv(output_dir / "paired_recipe_deltas.csv", index=False, float_format="%.12g")
    factor_summary.to_csv(output_dir / "factor_summary.csv", index=False, float_format="%.12g")
    family_factor_summary.to_csv(output_dir / "family_factor_summary.csv", index=False, float_format="%.12g")
    decomposition.to_csv(output_dir / "variation_decomposition.csv", index=False, float_format="%.12g")
    signs.to_csv(output_dir / "recipe_sign_consistency.csv", index=False, float_format="%.12g")
    plot_panel(paired, output_dir / "normalization_interactions.png", output_dir / "normalization_interactions.pdf")

    overall = paired["delta_mq_minus_d2e8"]
    family_summary = (
        paired.groupby("family", observed=True)
        .agg(
            n_recipes=("delta_mq_minus_d2e8", "size"),
            mean_delta=("delta_mq_minus_d2e8", "mean"),
            median_delta=("delta_mq_minus_d2e8", "median"),
            positive_fraction=("delta_mq_minus_d2e8", lambda x: float((x > 0).mean())),
        )
        .reset_index()
    )
    family_summary["family"] = family_summary["family"].astype(str)
    family_summary.to_csv(output_dir / "family_summary.csv", index=False, float_format="%.12g")

    interaction_fraction = float(
        decomposition.loc[
            decomposition["component"] == "family_by_recipe_residual", "fraction_of_total_variation"
        ].iloc[0]
    )
    unanimous = int(signs["unanimous_sign"].sum())
    report = "# Normalization-recipe interaction analysis\n\n"
    report += "## Direct result\n\n"
    report += (
        f"Across {len(paired)} exact family/recipe pairs, the mean MQ-minus-D2-E8 "
        f"NAP-product difference was {overall.mean():+.6f}, the median was "
        f"{overall.median():+.6f}, and MQ was higher in {(overall > 0).mean():.1%} of cells. "
        f"Only {unanimous} of {len(signs)} aligned recipe structures had the same sign in all five families.\n\n"
    )
    report += "## Family-specific summaries\n\n"
    report += "| Family | Mean delta | Median delta | MQ higher |\n|---|---:|---:|---:|\n"
    for row in family_summary.itertuples(index=False):
        report += f"| {row.family} | {row.mean_delta:+.6f} | {row.median_delta:+.6f} | {row.positive_fraction:.1%} |\n"
    report += "\n## Variation decomposition\n\n"
    report += (
        "A descriptive balanced two-way decomposition separates the 5-by-48 grid into family, "
        "aligned recipe-structure, and family-by-recipe residual components. It is not an "
        "inferential ANOVA because recipes are deterministic settings, not independent biological replicates.\n\n"
    )
    for row in decomposition.itertuples(index=False):
        report += f"- {row.component}: {row.fraction_of_total_variation:.1%} of grid sum-of-squares variation.\n"
    report += (
        f"\nThe family-by-recipe residual fraction was {interaction_fraction:.1%}. The heatmap therefore "
        "supports deterministic family/recipe sensitivity, not stochastic signal cleaning and not a universal MQ benefit.\n\n"
    )
    report += "## Design limitations\n\n"
    report += (
        "- Figure 3c pools deterministic configuration outputs; the 48 recipes are not biological replicates.\n"
        "- Exact pruning differs by family: cp_measure compares 0.90/0.95, while learned representations compare none/0.90. "
        "The aligned heatmap labels these as lower/higher intensity and preserves exact values in CSV.\n"
        "- All rows use TVN-EFAAR, no PCA, and no repeated stochastic run. Batch-method and PCA effects are therefore not identifiable here. "
        "TVN-EFAAR component count (96/128) and epsilon (0.05/0.1/0.2) do vary.\n"
        "- This analysis uses archived point estimates only and does not quantify compound/target sampling uncertainty.\n"
    )
    (output_dir / "REPORT.md").write_text(report)

    provenance = {
        "analysis": "normalization_interactions",
        "created_utc": "2026-08-18T00:00:00Z",
        "input": {
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "sha256": sha256_file(input_path),
            "rows": len(frame),
        },
        "estimand": "unscaled NAP product = PA_mean_nap * PC_mean_nap",
        "contrast": "MQ - D2-E8",
        "families": FAMILY_ORDER,
        "paired_rows": len(paired),
        "recipes_per_family": EXPECTED_CONFIGS,
        "source_sha256": sha256_file(source_path),
        "canonical_inputs_read_only": True,
        "normalization_or_extraction_rerun": False,
    }
    _write_json(output_dir / "provenance.json", provenance)

    artifacts = [output_dir / name for name in sorted(ARTIFACT_FILES)]
    _write_json(
        output_dir / "artifact_checksums.json",
        {
            "schema_version": 1,
            "input": {
                "path": str(input_path),
                "bytes": INPUT_SIZE,
                "sha256": INPUT_SHA256,
                "rows": EXPECTED_INPUT_ROWS,
            },
            "source": {"path": "analyze.py", "sha256": sha256_file(source_path)},
            "artifacts": _artifact_records(artifacts, output_dir),
        },
    )


def run(input_path: Path, output_dir: Path) -> None:
    """Generate a complete staged release and publish it only after verification."""
    validate_input(input_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    try:
        _generate_release(input_path, staged, Path(__file__))
        verify_release(staged, input_path, Path(__file__))
        _publish_staged(staged, output_dir)
        verify_release(output_dir, input_path, Path(__file__))
    finally:
        if staged.exists():
            shutil.rmtree(staged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs" / "release_v1",
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify_only:
        verify_release(args.output_dir, args.input, Path(__file__))
        print("input, source, inventory, provenance, and artifact checksums verified")
        return
    run(args.input, args.output_dir)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
