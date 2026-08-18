#!/usr/bin/env python3
"""Explain Target-2 MQ versus D2-E8 using frozen recipes and archived profiles.

This analysis (1) aligns archived PA compound and PC target units to quantify
influence/concentration and (2) re-scores each fixed recipe after omitting one
of the four Target-2 plate/laboratory pairs. Canonical inputs are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from norm_3.io import get_numeric_features, infer_columns  # noqa: E402
from norm_3.metrics import (  # noqa: E402
    calculate_phenotypic_activity,
    calculate_phenotypic_consistency,
)

SWEEP_ROOT = Path("/work/datasets/JUMP-lite-wacv/sweeps/MAIN_RESULTS__figure_4_variance_first_v11")
SWEEP_CSV = SWEEP_ROOT / "sweep_results.csv"
SELECTION_TOL = 1e-15
POINT_TOL = 1e-12
EXPECTED_CONFIGS = 48
EXPECTED_PA = 306
EXPECTED_PC = 201
EXPECTED_FULL_ROWS = 1536
MIN_PROFILE_ROWS = 1500
EXPECTED_PLATES = 4
CODECS = {"D2-E8": "jpegxl_lossy_d2_e8", "MQ": "jpegxl_lossy_mq"}
FAMILIES = {
    "cp_measure": {"display": "cp_measure", "sweep_model": "zstd_raw", "prefix": "cp_measure"},
    "dinov2": {"display": "DINOv2", "sweep_model": "dinov2_zstd_raw", "prefix": "dinov2"},
    "morphem": {"display": "MorphEM", "sweep_model": "morphem_zstd_raw", "prefix": "morphem"},
    "openphenom": {"display": "OpenPhenom", "sweep_model": "openphenom_zstd_raw", "prefix": "openphenom"},
    "subcell": {"display": "SubCell", "sweep_model": "subcell__clip01_zstd_raw", "prefix": "subcell__clip01"},
}

class AnalysisError(RuntimeError):
    pass

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise AnalysisError(f"missing or empty input: {path}")
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}

def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)

def write_csv(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        raise AnalysisError(f"refusing empty output: {path}")
    atomic_text(path, df.to_csv(index=False, lineterminator="\n"))

def write_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")

def select_recipes(sweep: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "config", "PA", "PC", "PA_mean_nap", "PC_mean_nap"}
    if missing := required.difference(sweep.columns):
        raise AnalysisError(f"sweep columns missing: {sorted(missing)}")
    rows = []
    for family, spec in FAMILIES.items():
        candidates = sweep[sweep.model == spec["sweep_model"]].copy()
        if len(candidates) != EXPECTED_CONFIGS or candidates.config.duplicated().any():
            raise AnalysisError(f"{family}: expected {EXPECTED_CONFIGS} unique Zstd recipes, got {len(candidates)}")
        candidates["selection_metric"] = candidates.PA * candidates.PC / 100.0
        best = candidates.selection_metric.max()
        tied = candidates[np.isclose(candidates.selection_metric, best, rtol=0, atol=SELECTION_TOL)]
        winner = tied.sort_values("config").iloc[0]
        rows.append({
            "family": family, "display_family": spec["display"], "config": str(winner.config),
            "selection_metric": "PA*PC/100 on Zstd only", "selection_value": float(winner.selection_metric),
            "zstd_pa_pct": float(winner.PA), "zstd_pc_pct": float(winner.PC), "n_tied": len(tied),
        })
    return pd.DataFrame(rows)

def variant_folder(family: str, codec_folder: str, config: str) -> Path:
    prefix = FAMILIES[family]["prefix"]
    return SWEEP_ROOT / f"{prefix}_jump_target2_4plate_{codec_folder}_raw_features" / config

def load_variant(family: str, codec: str, config: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pl.DataFrame, list[dict[str, Any]]]:
    folder = variant_folder(family, CODECS[codec], config)
    paths = {
        "metrics": folder / "results/metrics.json",
        "pa": folder / "results/phenotypic_activity_map.csv",
        "pc": folder / "results/phenotypic_consistency_per_target.csv",
        "output": folder / "output.parquet",
        "config": folder / "pipeline_config.yaml",
    }
    identities = [record(p) | {"family": family, "codec": codec, "role": role} for role, p in paths.items()]
    metrics = json.loads(paths["metrics"].read_text())
    pa = pd.read_csv(paths["pa"])
    pc = pd.read_csv(paths["pc"])
    profile = pl.read_parquet(paths["output"])
    if len(pa) != EXPECTED_PA or pa["Metadata_broad_sample"].duplicated().any():
        raise AnalysisError(f"{family}/{codec}: invalid PA units")
    if len(pc) != EXPECTED_PC or pc["Metadata_target"].duplicated().any():
        raise AnalysisError(f"{family}/{codec}: invalid PC units")
    if len(profile) < MIN_PROFILE_ROWS or len(profile) > EXPECTED_FULL_ROWS or profile["Metadata_id"].n_unique() != len(profile):
        raise AnalysisError(f"{family}/{codec}: invalid profile identity coverage")
    pa_point = float(pa.mean_normalized_average_precision.mean())
    pc_point = float(pc.mean_normalized_average_precision.mean())
    if abs(pa_point - float(metrics["PA_mean_nap"])) > POINT_TOL or abs(pc_point - float(metrics["PC_mean_nap"])) > POINT_TOL:
        raise AnalysisError(f"{family}/{codec}: archived per-unit/metrics mismatch")
    return metrics, pa, pc, profile, identities

def score_profile(profile: pl.DataFrame, require_expected_units: bool = True) -> dict[str, Any]:
    feature_cols, _ = infer_columns(profile)
    features = get_numeric_features(profile, feature_cols)
    if not features or not np.isfinite(profile.select(features).to_numpy()).all():
        raise AnalysisError("profile has absent/non-finite feature contract")
    pa = calculate_phenotypic_activity(
        profile, features, compound_col="Metadata_broad_sample", negcon_col="Metadata_negcon",
        batch_col="Metadata_Plate", seed=0,
    )
    pc = calculate_phenotypic_consistency(
        profile, features, compound_col="Metadata_broad_sample", target_col="Metadata_target_list",
        negcon_col="Metadata_negcon", seed=0,
    )
    n_pa, n_pc = len(pa["activity_map"]), len(pc["target_consistency"])
    if require_expected_units and (n_pa != EXPECTED_PA or n_pc != EXPECTED_PC):
        raise AnalysisError("full rescoring changed PA/PC unit counts")
    if n_pa <= 0 or n_pc <= 0:
        raise AnalysisError("rescoring produced empty PA/PC units")
    pa_point = float(pa["mean_normalized_average_precision"])
    pc_point = float(pc["mean_normalized_average_precision"])
    if not math.isfinite(pa_point) or not math.isfinite(pc_point):
        raise AnalysisError("non-finite rescored point")
    return {"pa": pa_point, "pc": pc_point, "product": pa_point * pc_point, "n_pa": n_pa, "n_pc": n_pc}

def aligned_units(family: str, tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pa_d2, pc_d2 = tables["D2-E8"]
    pa_mq, pc_mq = tables["MQ"]
    pa = pa_d2[["Metadata_broad_sample", "mean_normalized_average_precision"]].rename(columns={"mean_normalized_average_precision": "d2_value"}).merge(
        pa_mq[["Metadata_broad_sample", "mean_normalized_average_precision"]].rename(columns={"mean_normalized_average_precision": "mq_value"}),
        on="Metadata_broad_sample", how="inner", validate="one_to_one",
    )
    pc = pc_d2[["Metadata_target", "mean_normalized_average_precision"]].rename(columns={"mean_normalized_average_precision": "d2_value"}).merge(
        pc_mq[["Metadata_target", "mean_normalized_average_precision"]].rename(columns={"mean_normalized_average_precision": "mq_value"}),
        on="Metadata_target", how="inner", validate="one_to_one",
    )
    if len(pa) != EXPECTED_PA or len(pc) != EXPECTED_PC:
        raise AnalysisError(f"{family}: aligned unit coverage drift")
    return pa.sort_values("Metadata_broad_sample"), pc.sort_values("Metadata_target")

def unit_contributions(family: str, pa: pd.DataFrame, pc: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    pa_d2, pa_mq = pa.d2_value.mean(), pa.mq_value.mean()
    pc_d2, pc_mq = pc.d2_value.mean(), pc.mq_value.mean()
    full_delta = pa_mq * pc_mq - pa_d2 * pc_d2
    avg_pa, avg_pc = (pa_mq + pa_d2) / 2, (pc_mq + pc_d2) / 2
    parts = []
    for margin, frame, key, scale in (
        ("PA", pa, "Metadata_broad_sample", avg_pc), ("PC", pc, "Metadata_target", avg_pa)
    ):
        n = len(frame)
        for _, row in frame.iterrows():
            unit_delta = float(row.mq_value - row.d2_value)
            contribution = scale * unit_delta / n
            if margin == "PA":
                without = ((pa.mq_value.sum() - row.mq_value) / (n - 1)) * pc_mq - ((pa.d2_value.sum() - row.d2_value) / (n - 1)) * pc_d2
            else:
                without = pa_mq * ((pc.mq_value.sum() - row.mq_value) / (n - 1)) - pa_d2 * ((pc.d2_value.sum() - row.d2_value) / (n - 1))
            parts.append({
                "family": family, "display_family": FAMILIES[family]["display"], "margin": margin,
                "unit_id": str(row[key]), "d2_value": float(row.d2_value), "mq_value": float(row.mq_value),
                "unit_delta_mq_minus_d2": unit_delta, "symmetric_product_contribution": contribution,
                "absolute_contribution": abs(contribution), "leave_one_unit_influence": full_delta - without,
            })
    out = pd.DataFrame(parts)
    if abs(out.symmetric_product_contribution.sum() - full_delta) > 1e-14:
        raise AnalysisError(f"{family}: contributions do not sum to product delta")
    out["absolute_rank"] = out.absolute_contribution.rank(method="first", ascending=False).astype(int)
    total_abs = out.absolute_contribution.sum()
    summary = {
        "family": family, "display_family": FAMILIES[family]["display"], "product_delta_mq_minus_d2": full_delta,
        "pa_delta": pa_mq - pa_d2, "pc_delta": pc_mq - pc_d2,
        "pa_absolute_share": float(out.loc[out.margin == "PA", "absolute_contribution"].sum() / total_abs),
    }
    for k in (5, 10, 20):
        summary[f"top_{k}_absolute_share"] = float(out.nsmallest(k, "absolute_rank").absolute_contribution.sum() / total_abs)
    return out, summary

def make_panel(loo: pd.DataFrame, influence: pd.DataFrame, path_png: Path, path_pdf: Path) -> None:
    order = [FAMILIES[f]["display"] for f in FAMILIES]
    columns = ["Full"] + sorted([x for x in loo.omitted_label.unique() if x != "Full"])
    pivot = loo.pivot(index="display_family", columns="omitted_label", values="product_delta_mq_minus_d2").reindex(index=order, columns=columns)
    vals = pivot.to_numpy(float)
    limit = max(abs(vals.min()), abs(vals.max()), 1e-6)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), gridspec_kw={"width_ratios": [1.7, 1]})
    im = axes[0].imshow(vals, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axes[0].set_xticks(range(len(columns)), [c.replace(" / ", "\n") for c in columns], rotation=35, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(order)), order)
    axes[0].set_title("MQ − D2-E8 after omitting one plate/laboratory", fontweight="bold")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            axes[0].text(j, i, f"{vals[i,j]:+.4f}", ha="center", va="center", fontsize=8,
                         color="white" if abs(vals[i,j]) > 0.58 * limit else "black")
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.045, pad=0.03)
    cbar.set_label("NAP product difference")

    inf = influence.set_index("display_family").reindex(order)
    y = np.arange(len(order))
    axes[1].barh(y, 100 * inf.top_10_absolute_share, color="#4C78A8", label="Top 10 units")
    axes[1].scatter(100 * inf.pa_absolute_share, y, marker="D", color="#E45756", label="PA share of |contribution|", zorder=3)
    axes[1].set_yticks(y, order); axes[1].invert_yaxis(); axes[1].set_xlim(0, 100)
    axes[1].set_xlabel("Share of absolute unit contribution (%)")
    axes[1].set_title("Influence concentration", fontweight="bold")
    axes[1].grid(axis="x", alpha=.25); axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("Plate and unit influence on the Target-2 MQ/D2-E8 contrast", fontweight="bold", fontsize=14)
    fig.tight_layout()
    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight", metadata={"Creator": "JUMP-lite", "CreationDate": None, "ModDate": None})
    plt.close(fig)

def run(output: Path) -> None:
    if not SWEEP_CSV.is_file():
        raise AnalysisError(f"missing sweep: {SWEEP_CSV}")
    sweep_identity = record(SWEEP_CSV)
    sweep = pd.read_csv(SWEEP_CSV)
    selected = select_recipes(sweep)
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix="plate-unit-influence-", dir=output.parent))
    try:
        inputs = [sweep_identity]
        point_rows, loo_rows, coverage_rows, contribution_frames, influence_rows = [], [], [], [], []
        selected_by_family = selected.set_index("family")
        for family in FAMILIES:
            config = str(selected_by_family.loc[family, "config"])
            tables, profiles, archived = {}, {}, {}
            expected_plate_map = None
            for codec in CODECS:
                metrics, pa, pc, profile, identities = load_variant(family, codec, config)
                inputs.extend(identities); tables[codec] = (pa, pc); profiles[codec] = profile; archived[codec] = metrics
                mapping = profile.select(["Metadata_Plate", "Metadata_Source"]).unique().sort("Metadata_Plate")
                if len(mapping) != EXPECTED_PLATES or mapping["Metadata_Plate"].n_unique() != EXPECTED_PLATES or mapping["Metadata_Source"].n_unique() != EXPECTED_PLATES:
                    raise AnalysisError(f"{family}/{codec}: plate/source is not one-to-one")
                current = mapping.to_dicts()
                if expected_plate_map is None: expected_plate_map = current
                elif current != expected_plate_map: raise AnalysisError(f"{family}: codec plate/source mapping drift")
                full = score_profile(profile)
                if abs(full["pa"] - float(metrics["PA_mean_nap"])) > POINT_TOL or abs(full["pc"] - float(metrics["PC_mean_nap"])) > POINT_TOL:
                    raise AnalysisError(f"{family}/{codec}: full rescoring did not reproduce archive")
                point_rows.append({"family": family, "display_family": FAMILIES[family]["display"], "codec": codec, "config": config, **full,
                                   "archived_pa": float(metrics["PA_mean_nap"]), "archived_pc": float(metrics["PC_mean_nap"])})
            # Pair LOO scores on an explicit common Metadata_id population. The
            # cp_measure normalized outputs legitimately retain 1519/1520 rows,
            # so silently comparing their unequal populations would be invalid.
            id_sets = {codec: set(profile["Metadata_id"].to_list()) for codec, profile in profiles.items()}
            common_ids = set.intersection(*id_sets.values())
            if len(common_ids) < MIN_PROFILE_ROWS:
                raise AnalysisError(f"{family}: insufficient common codec coverage")
            common_profiles = {
                codec: profile.filter(pl.col("Metadata_id").is_in(sorted(common_ids))).sort("Metadata_id")
                for codec, profile in profiles.items()
            }
            for codec, profile in profiles.items():
                coverage_rows.append({"family": family, "codec": codec, "original_rows": len(profile), "common_rows": len(common_ids),
                                      "dropped_for_common": len(profile) - len(common_ids)})
            pa, pc = aligned_units(family, tables)
            contrib, influence = unit_contributions(family, pa, pc)
            contribution_frames.append(contrib); influence_rows.append(influence)
            common_full = {codec: score_profile(profile) for codec, profile in common_profiles.items()}
            loo_rows.append({"family": family, "display_family": FAMILIES[family]["display"], "omitted_plate": "", "omitted_source": "",
                             "omitted_label": "Full", "d2_product": common_full["D2-E8"]["product"], "mq_product": common_full["MQ"]["product"],
                             "product_delta_mq_minus_d2": common_full["MQ"]["product"] - common_full["D2-E8"]["product"],
                             "d2_pa": common_full["D2-E8"]["pa"], "d2_pc": common_full["D2-E8"]["pc"], "mq_pa": common_full["MQ"]["pa"], "mq_pc": common_full["MQ"]["pc"]})
            for plate_row in expected_plate_map or []:
                plate, source = str(plate_row["Metadata_Plate"]), str(plate_row["Metadata_Source"])
                scores = {codec: score_profile(profile.filter(pl.col("Metadata_Plate") != plate), require_expected_units=False) for codec, profile in common_profiles.items()}
                if scores["D2-E8"]["n_pa"] != scores["MQ"]["n_pa"] or scores["D2-E8"]["n_pc"] != scores["MQ"]["n_pc"]:
                    raise AnalysisError(f"{family}/{plate}: codec LOO unit counts differ")
                loo_rows.append({"family": family, "display_family": FAMILIES[family]["display"], "omitted_plate": plate, "omitted_source": source,
                                 "omitted_label": f"{source} / {plate}", "d2_product": scores["D2-E8"]["product"], "mq_product": scores["MQ"]["product"],
                                 "product_delta_mq_minus_d2": scores["MQ"]["product"] - scores["D2-E8"]["product"],
                                 "d2_pa": scores["D2-E8"]["pa"], "d2_pc": scores["D2-E8"]["pc"], "mq_pa": scores["MQ"]["pa"], "mq_pc": scores["MQ"]["pc"],
                                 "n_pa": scores["MQ"]["n_pa"], "n_pc": scores["MQ"]["n_pc"]})
        points = pd.DataFrame(point_rows); loo = pd.DataFrame(loo_rows); coverage = pd.DataFrame(coverage_rows); contributions = pd.concat(contribution_frames, ignore_index=True); influence = pd.DataFrame(influence_rows)
        top_parts = []
        for _, group in contributions.groupby("family", sort=True):
            top_parts.extend([
                group.nlargest(10, "symmetric_product_contribution").assign(direction="positive"),
                group.nsmallest(10, "symmetric_product_contribution").assign(direction="negative"),
            ])
        top = pd.concat(top_parts, ignore_index=True)
        write_csv(staged / "selected_recipes.csv", selected)
        write_csv(staged / "fixed_recipe_points.csv", points)
        write_csv(staged / "coverage_manifest.csv", coverage)
        write_csv(staged / "unit_contributions.csv", contributions)
        write_csv(staged / "influence_summary.csv", influence)
        write_csv(staged / "top_influential_units.csv", top)
        write_csv(staged / "leave_one_plate_out.csv", loo)
        make_panel(loo, influence, staged / "plate_unit_influence.png", staged / "plate_unit_influence.pdf")
        report_lines = [
            "# Plate and unit influence for MQ versus D2-E8", "",
            "One Zstd-selected recipe per Figure 3c family was frozen across MQ and D2-E8. Rescoring each original normalized output reproduced every archived PA/PC point within 1e-12 before leave-one-plate-out results were accepted. LOO contrasts use an explicitly recorded common Metadata_id population within each family; this matters for cp_measure, whose codec outputs retained slightly different rows.", "",
            "| Family | Full MQ-D2-E8 product | LOO min | LOO max | Top-10 absolute share | PA absolute share |", "|---|---:|---:|---:|---:|---:|",
        ]
        for family in FAMILIES:
            display = FAMILIES[family]["display"]; sub = loo[loo.family == family]; inf = influence[influence.family == family].iloc[0]
            report_lines.append(f"| {display} | {sub[sub.omitted_label == 'Full'].product_delta_mq_minus_d2.iloc[0]:+.6f} | {sub[sub.omitted_label != 'Full'].product_delta_mq_minus_d2.min():+.6f} | {sub[sub.omitted_label != 'Full'].product_delta_mq_minus_d2.max():+.6f} | {100*inf.top_10_absolute_share:.1f}% | {100*inf.pa_absolute_share:.1f}% |")
        report_lines += ["", "The cp_measure common-population full contrast (+0.001367) differs slightly from its archived unequal-population contrast (+0.001251); the four learned-family populations are already identical across codecs. Plate and laboratory are perfectly confounded in this four-plate pilot. Unit contributions are descriptive influence diagnostics, not independent causal effects. PA and PC are retrieval-derived and the product decomposition does not supply end-to-end uncertainty.", ""]
        atomic_text(staged / "REPORT.md", "\n".join(report_lines))
        provenance = {"analysis": "plate_unit_influence", "protocol_version": 1, "canonical_inputs_read_only": True,
                      "selection": "lexical tie-break after maximizing PA*PC/100 on Zstd only", "families": list(FAMILIES), "codecs": list(CODECS),
                      "expected_counts": {"configs_per_zstd_family": EXPECTED_CONFIGS, "maximum_profiles_per_variant": EXPECTED_FULL_ROWS, "minimum_profiles_per_variant": MIN_PROFILE_ROWS, "pa_units": EXPECTED_PA, "pc_units": EXPECTED_PC, "plates": EXPECTED_PLATES},
                      "inputs": sorted(inputs, key=lambda x: (x.get("family", ""), x.get("codec", ""), x.get("role", ""), x["path"])),
                      "qualification": "Plate and laboratory are confounded; influence is descriptive; no denoising or biological improvement is inferred."}
        write_json(staged / "provenance.json", provenance)
        artifacts = []
        for path in sorted(staged.iterdir()):
            if path.name == "artifact_checksums.json" or not path.is_file(): continue
            artifacts.append({"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
        write_json(staged / "artifact_checksums.json", {"artifacts": artifacts})
        if output.exists(): shutil.rmtree(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged, output)
    finally:
        if staged.exists(): shutil.rmtree(staged)

def verify(output: Path) -> None:
    payload = json.loads((output / "artifact_checksums.json").read_text())
    for row in payload["artifacts"]:
        path = output / row["path"]
        if path.stat().st_size != row["size_bytes"] or sha256_file(path) != row["sha256"]:
            raise AnalysisError(f"artifact drift: {path}")
    print(f"verified {len(payload['artifacts'])} artifacts")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "outputs/release_v1")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only: verify(args.output)
    else: run(args.output); verify(args.output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
