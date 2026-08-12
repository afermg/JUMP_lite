#!/usr/bin/env python3
"""Fail-closed current-release rescore on the frozen label-blind partition.

This runner never fits a cluster, PCA, or UMAP.  It derives the current compound
membership from the tracked release metadata, verifies that membership against
the historical manifest and fit, and writes a new, absent result root.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scipy
from scipy.stats import beta, rankdata
from sklearn import __version__ as sklearn_version
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

HERE = Path(__file__).resolve().parent
ANALYSIS_SOURCE = HERE / "analyze_cluster_representativeness.py"
EXPECTED_ANALYSIS_SOURCE = (
    31_366,
    "09ac9d4392c7573335d0ec5a4fd467070baeee3fb95d5f7e47306604ab29b645",
)
# Verify the imported scientific helper before Python executes it. This binding
# is intentionally independent of the fit completion marker and artifacts.
if (ANALYSIS_SOURCE.stat().st_size,
        hashlib.sha256(ANALYSIS_SOURCE.read_bytes()).hexdigest()) != EXPECTED_ANALYSIS_SOURCE:
    raise RuntimeError(f"Reviewed analysis helper identity drift: {ANALYSIS_SOURCE}")
_ANALYSIS_SPEC = importlib.util.spec_from_file_location("reviewed_cluster_analysis", ANALYSIS_SOURCE)
if _ANALYSIS_SPEC is None or _ANALYSIS_SPEC.loader is None:
    raise RuntimeError(f"Cannot load reviewed analysis helper: {ANALYSIS_SOURCE}")
analysis = importlib.util.module_from_spec(_ANALYSIS_SPEC)
_ANALYSIS_SPEC.loader.exec_module(analysis)

REPOSITORY_ROOT = HERE.parent.parent
HISTORICAL_ROOT = HERE / "outputs/profile_cluster_representativeness_v1"
DEFAULT_OUT = HERE / "outputs/profile_cluster_representativeness_release_v1"
METADATA = REPOSITORY_ROOT / "metadata/jump_lite_v1_perturbation_metadata.parquet"
DESIGN_ADDENDUM = HERE / "CLUSTER_SELECTION_RELEASE_V1_ADDENDUM.md"
HISTORICAL_MANIFEST = HERE / "outputs/profile_space/manifests/jump_lite_compounds.parquet"
FROZEN_UMAP = HISTORICAL_ROOT / "compound_cluster_umap.parquet"
REPOSITORY_SCOPE = "repository_root"
OUTPUT_SCOPE = "output_root"

EXPECTED_METADATA = (949_745, "bbedb37f12fdeb9a09e72abaa166159d286052dc1201166d811c5310db5cd7e1")
EXPECTED_HISTORICAL_MANIFEST = (33_252, "a0671dcaae029a2c32ac58fdaf09178806b495d33a5ea439ff859b3c0fbe74de")
EXPECTED_FROZEN_UMAP = (999_039, "3b8ac04e817dc65e89926048797bf330e8595966d9dd425de2491dab07f82bba")
EXPECTED_FIT_COMPLETION = (1_135, "fee7d3c85d2a4da2223c2962f2c75f247315419745d631785b84ea768036f332")
EXPECTED_FIT_ARTIFACTS = {
    "cluster_assignments_sensitivity_unlabeled.parquet": (11_203_503, "c3d021bf0c3c56488f5409228383f9cb7991e14307bfc2c4d072b9a2bfc4f45a"),
    "cluster_assignments_unlabeled.parquet": (1_161_683, "20a1282d0e81ad6d26cd443d327ba86282cd38981b7751d871bc170bec940613"),
    "clustering_diagnostics.csv": (4_708, "9df79ab9f92e6c3c1ca0b268e6544a9d59647a5c1e7bb345f86ce0b07e628709"),
    "computation_identity.json": (1_657, "153ddc59e136ae6bdb78ac61e76a311d306c73b4c52e4fae9b16c7bc1019941c"),
    "primary_model.npz": (31_712, "3546f6abe83ef85b2aa809cdbf4209dd0b2a979b3415c3e7eb956bf4339a8a54"),
}
METADATA_SCHEMA = {
    "Metadata_Well_Key": pl.String,
    "Metadata_Source": pl.String,
    "Metadata_Batch": pl.String,
    "Metadata_Plate": pl.String,
    "Metadata_Well": pl.String,
    "Metadata_JCP2022": pl.String,
    "Metadata_broad_sample": pl.String,
    "Metadata_Symbol": pl.String,
    "Metadata_pert_type": pl.String,
    "Metadata_Perturbation_Type": pl.String,
    "Metadata_Group": pl.String,
}
CURRENT_TREATMENTS = 3_775
CURRENT_COMPOUND_IDENTIFIERS = 3_776
HISTORICAL_ONLY = 57
NEGATIVE_CONTROL_ID = "JCP2022_033924"
PERMUTATIONS = 2_000
OUTPUT_NAMES = {
    "current_release_treatment_manifest.parquet",
    "current_release_manifest_provenance.json",
    "compound_cluster_assignments.parquet",
    "structural_counts.csv",
    "cluster_selection_table.csv",
    "cluster_selection_table.parquet",
    "retrieval_metrics.csv",
    "representation_summary.json",
    "permutation_summary.json",
    "cluster_selection_all_clusters.png",
    "retrieval_partition_sensitivity.csv",
    "selection_conclusion.json",
    "REPORT.md",
    "provenance.json",
    "computation_identity.json",
    "compound_cluster_umap.parquet",
    "cluster_selection_compound_map.png",
    "cluster_selection_compound_map.pdf",
    "cluster_selection_summary_table.csv",
    "cluster_selection_figure_provenance.json",
    "cluster_selection_figure_hashes.json",
    "output_hashes.json",
}


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path, *, base: Path, scope: str) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    resolved_base = base.resolve(strict=True)
    try:
        relative = resolved.relative_to(resolved_base)
    except ValueError as error:
        raise RuntimeError(f"Artifact escapes {scope}: {resolved}") from error
    if relative == Path(".") or ".." in relative.parts:
        raise RuntimeError(f"Invalid {scope} artifact: {relative}")
    return {"path": relative.as_posix(), "path_scope": scope,
            "bytes": resolved.stat().st_size, "sha256": sha256_file(resolved)}


def repository_record(path: Path) -> dict[str, object]:
    return record(path, base=REPOSITORY_ROOT, scope=REPOSITORY_SCOPE)


def output_record(path: Path, out: Path) -> dict[str, object]:
    return record(path, base=out, scope=OUTPUT_SCOPE)


def resolve_output_record(rec: dict[str, object], out: Path) -> Path:
    """Resolve an output record while rejecting traversal and symlink escapes."""
    if rec.get("path_scope") != OUTPUT_SCOPE:
        raise RuntimeError(f"Invalid output record scope: {rec.get('path_scope')!r}")
    raw = rec.get("path")
    if not isinstance(raw, str) or not raw:
        raise RuntimeError("Output record path is absent or invalid")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Output record path must be contained and relative: {raw}")
    base = out.resolve(strict=True)
    resolved = (base / relative).resolve(strict=False)
    try:
        resolved.relative_to(base)
    except ValueError as error:
        raise RuntimeError(f"Output record path escapes output root: {raw}") from error
    return resolved


def verify_exact(path: Path, expected: tuple[int, str]) -> dict[str, object]:
    rec = repository_record(path)
    if (rec["bytes"], rec["sha256"]) != expected:
        raise RuntimeError(f"Frozen input identity drift: {path}: {rec}")
    return rec


def write_json(path: Path, value: object) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing overwrite: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_reviewed_fit() -> dict[str, object]:
    """Independently pin helper, completion marker, and every frozen fit file."""
    helper = verify_exact(ANALYSIS_SOURCE, EXPECTED_ANALYSIS_SOURCE)
    fitdir = HISTORICAL_ROOT / "fit"
    completion = verify_exact(fitdir / "fit_complete.json", EXPECTED_FIT_COMPLETION)
    artifacts = {
        name: verify_exact(fitdir / name, expected)
        for name, expected in sorted(EXPECTED_FIT_ARTIFACTS.items())
    }
    # Also validate marker inventory, path scopes, and its internal hashes. The
    # constants above ensure changing marker and artifacts together still fails.
    analysis.verify_frozen_fit(HISTORICAL_ROOT)
    return {"analysis_helper": helper, "fit_completion": completion,
            "fit_artifacts": artifacts}


def derive_current_manifest() -> tuple[pl.DataFrame, dict[str, object]]:
    """Derive and fully validate the release treatment manifest in memory."""
    metadata_record = verify_exact(METADATA, EXPECTED_METADATA)
    historical_record = verify_exact(HISTORICAL_MANIFEST, EXPECTED_HISTORICAL_MANIFEST)
    metadata = pl.read_parquet(METADATA)
    if metadata.height != 163_776 or metadata.schema != METADATA_SCHEMA:
        raise RuntimeError(f"Current metadata schema/row drift: {metadata.height}, {metadata.schema}")
    if metadata["Metadata_Well_Key"].n_unique() != metadata.height:
        raise RuntimeError("Current metadata well keys are not unique")

    compounds = metadata.filter(pl.col("Metadata_Perturbation_Type") == "compound")
    compound_ids = compounds.select("Metadata_JCP2022").unique()
    treatments = compounds.filter(pl.col("Metadata_pert_type") == "trt")
    treatment_ids = treatments.select("Metadata_JCP2022").unique().sort("Metadata_JCP2022")
    controls = compounds.filter(pl.col("Metadata_pert_type") == "negcon").select("Metadata_JCP2022").unique()
    unexpected_types = set(compounds["Metadata_pert_type"].unique().to_list()) - {"trt", "negcon"}
    if (compound_ids.height != CURRENT_COMPOUND_IDENTIFIERS
            or treatment_ids.height != CURRENT_TREATMENTS
            or controls.height != 1
            or controls.item() != NEGATIVE_CONTROL_ID
            or unexpected_types):
        raise RuntimeError("Current compound treatment/control count or type drift")
    if treatments["Metadata_JCP2022"].null_count() or treatment_ids["Metadata_JCP2022"].n_unique() != CURRENT_TREATMENTS:
        raise RuntimeError("Current treatment identifiers are null or non-unique")

    historical = pl.read_parquet(HISTORICAL_MANIFEST).sort("Metadata_JCP2022")
    if historical.height != 3_832 or historical["Metadata_JCP2022"].n_unique() != 3_832:
        raise RuntimeError("Historical selected manifest count drift")
    current_set = set(treatment_ids["Metadata_JCP2022"].to_list())
    historical_set = set(historical["Metadata_JCP2022"].to_list())
    historical_only = sorted(historical_set - current_set)
    current_only = sorted(current_set - historical_set)
    if len(historical_only) != HISTORICAL_ONLY or current_only or current_set == historical_set:
        raise RuntimeError(
            f"Release/historical relation drift: historical-only={len(historical_only)}, "
            f"current-only={len(current_only)}"
        )

    manifest = (
        historical.filter(pl.col("Metadata_JCP2022").is_in(current_set))
        .with_columns(pl.lit("jump_lite_release_v1").alias("cohort"))
        .sort("Metadata_JCP2022")
    )
    if manifest.height != CURRENT_TREATMENTS or manifest["Metadata_JCP2022"].n_unique() != CURRENT_TREATMENTS:
        raise RuntimeError("Derived current manifest count/uniqueness drift")
    assignments = pl.read_parquet(HISTORICAL_ROOT / "fit/cluster_assignments_unlabeled.parquet")
    joined = manifest.join(
        assignments.select("Metadata_JCP2022", "fit_eligible"),
        on="Metadata_JCP2022", how="left", validate="1:1",
    )
    if joined["fit_eligible"].null_count() or not joined["fit_eligible"].all():
        raise RuntimeError("Current treatments lack frozen assignments or fit eligibility")
    facts = {
        "metadata": metadata_record,
        "historical_manifest": historical_record,
        "metadata_rows": metadata.height,
        "compound_identifiers": compound_ids.height,
        "treatments": treatment_ids.height,
        "negative_controls": controls.height,
        "negative_control_id": NEGATIVE_CONTROL_ID,
        "historical_treatments": historical.height,
        "historical_only_count": len(historical_only),
        "historical_only_ids": historical_only,
        "current_only_count": len(current_only),
        "current_only_ids": current_only,
        "all_assignments_present": True,
        "all_fit_eligible": True,
    }
    return manifest, facts


def partition_sensitivity(selected_set: set[str], fitdir: Path) -> pd.DataFrame:
    base = pl.read_parquet(fitdir / "cluster_assignments_unlabeled.parquet",
                           columns=["Metadata_JCP2022", "fit_eligible", "fold_id"])
    base = base.with_columns(pl.col("Metadata_JCP2022").is_in(selected_set).cast(pl.Int8).alias("selected"))
    sens = pl.read_parquet(fitdir / "cluster_assignments_sensitivity_unlabeled.parquet",
                           columns=["Metadata_JCP2022", "preprocessing", "k", "seed", "cluster_id"])
    rows: list[dict[str, object]] = []
    for key, group in sens.group_by(["preprocessing", "k", "seed"], maintain_order=True):
        prep, k, seed = key
        frame = group.join(base, on="Metadata_JCP2022", how="inner", validate="1:1").filter(pl.col("fit_eligible"))
        if frame.height != 95_426 or int(frame["selected"].sum()) != CURRENT_TREATMENTS:
            raise RuntimeError("Current partition sensitivity join/count drift")
        y = frame["selected"].to_numpy(); clusters = frame["cluster_id"].to_numpy(); folds = frame["fold_id"].to_numpy()
        score = np.empty(len(frame), float)
        for fold in range(5):
            test = folds == fold; train = ~test
            counts = np.bincount(clusters[train], minlength=int(k))
            positives = np.bincount(clusters[train], weights=y[train], minlength=int(k))
            score[test] = (positives[clusters[test]] + .5) / (counts[clusters[test]] + 1)
        prevalence = y.mean(); clipped = np.clip(score, 1e-9, 1 - 1e-9)
        total = np.bincount(clusters, minlength=int(k)); selected = np.bincount(clusters, weights=y, minlength=int(k))
        ps = selected / selected.sum(); pe = total / total.sum(); middle = .5 * (ps + pe)
        js = .5 * np.sum(ps[ps > 0] * np.log(ps[ps > 0] / middle[ps > 0])) + .5 * np.sum(pe * np.log(pe / middle))
        rows.append({"preprocessing": prep, "k": int(k), "seed": int(seed), "n": len(y),
                     "n_selected": int(y.sum()), "prevalence": prevalence,
                     "roc_auc": roc_auc_score(y, clipped),
                     "average_precision": average_precision_score(y, clipped),
                     "ap_lift": average_precision_score(y, clipped) / prevalence,
                     "log_loss": log_loss(y, clipped, labels=[0, 1]),
                     "brier": brier_score_loss(y, clipped),
                     "selected_occupied_clusters": int((selected > 0).sum()),
                     "eligible_mass_in_occupied_clusters": float(total[selected > 0].sum() / total.sum()),
                     "total_variation": float(.5 * np.abs(ps - pe).sum()),
                     "jensen_shannon_nats": float(js)})
    return pd.DataFrame(rows).sort_values(["preprocessing", "k", "seed"])


def render_figure(embedding: pl.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    data = embedding.to_pandas(); selected = data.selected.to_numpy(bool); eligible = data.fit_eligible.to_numpy(bool)
    clusters = data.cluster_id.to_numpy(int); x = data.umap_1.to_numpy(); y = data.umap_2.to_numpy()
    values = summary.set_index("metric")["value"]
    occupied_n = int(values["occupied_clusters"]); mass = values["eligible_mass_in_occupied_clusters"]
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.25), gridspec_kw={"width_ratios": [1.35, 1, .92]})
    ax = axes[0]
    ax.scatter(x[~selected], y[~selected], s=1, c="#8a8a8a", alpha=.075, linewidths=0, rasterized=True,
               label=f"Other compounds ({(~selected).sum():,})")
    ax.scatter(x[selected], y[selected], s=4.5, c="#0072B2", alpha=.52, linewidths=0, rasterized=True,
               label=f"JUMP-lite treatments ({selected.sum():,})")
    ax.set_title(f"A  Compound profile map ({selected.sum():,} treatments)", loc="left", fontweight="bold"); ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([]); ax.legend(frameon=False, loc="lower left", markerscale=2.8, fontsize=8)
    ax.text(.01, .99, "Frozen label-blind UMAP; 115,721 profiles (95,426 fit-eligible)", transform=ax.transAxes,
            va="top", ha="left", fontsize=7.5, color="#444444")
    ax = axes[1]
    eligible_counts = np.bincount(clusters[eligible], minlength=128).astype(float)
    selected_counts = np.bincount(clusters[selected], minlength=128).astype(float)
    pe = eligible_counts / eligible_counts.sum(); ps = selected_counts / selected_counts.sum(); occupied = selected_counts > 0
    upper = max(float(pe.max()), float(ps.max())) * 1.07
    ax.plot([0, upper], [0, upper], "--", color="#555555", lw=1, label="Proportional selection")
    ax.scatter(pe[occupied], ps[occupied], s=25, c="#0072B2", alpha=.72, linewidths=0, label=f"Occupied ({occupied_n})")
    ax.scatter(pe[~occupied], ps[~occupied], s=32, c="#D55E00", marker="x", linewidths=1.4,
               label=f"Unoccupied ({128 - occupied_n})")
    ax.set_xlim(-.001, upper); ax.set_ylim(-.001, upper); ax.set_aspect("equal", adjustable="box")
    ax.set_title("B  Coverage across all 128 clusters", loc="left", fontweight="bold")
    ax.set_xlabel("Share of eligible compounds"); ax.set_ylabel("Share of selected compounds"); ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.text(.98, .02, f"{occupied_n}/128 occupied\n{mass:.2%} of eligible mass", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8)
    ax = axes[2]
    labels = ["Constant", "Cluster only", "Acquisition\nstructure", "Structure\n+ cluster"]
    aps = [values["average_precision_constant"], values["average_precision_cluster_only"],
           values["average_precision_acquisition_structure"], values["average_precision_structure_plus_cluster"]]
    bars = ax.bar(np.arange(4), aps, color=["#B3B3B3", "#56B4E9", "#8A8A8A", "#0072B2"], width=.72)
    ax.set_ylim(0, max(.70, max(aps) * 1.18)); ax.set_xticks(np.arange(4), labels, fontsize=8)
    ax.set_ylabel("Out-of-fold average precision"); ax.set_title("C  Retrieval of selected treatments", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, aps, strict=True):
        ax.text(bar.get_x() + bar.get_width()/2, value + .016, f"{value:.3f}", ha="center", fontsize=8)
    # Keep the annotation below the value labels; in particular, do not cover
    # the 0.573 combined-model label above the rightmost bar.
    ax.text(.98, .04, f"Combined / structure = {values['combined_to_structure_ap_ratio']:.3f}×\n"
            f"Materiality gate = 1.25×\nConditional p = {values['conditional_permutation_p']:.6f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"})
    fig.suptitle("Current JUMP-lite treatments broadly span operational clusters; cluster signal is modestly additive",
                 x=.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(.02, .012, "Display projection only. Operational clusters have low stability (mean seed ARI = 0.162) and are not biological classes.",
             ha="left", va="bottom", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, .045, 1, .94), w_pad=2)
    fig.savefig(output_dir / "cluster_selection_compound_map.png", dpi=300,
                metadata={"Software": "rescore_cluster_representativeness_release_v1.py"})
    fig.savefig(output_dir / "cluster_selection_compound_map.pdf", dpi=300,
                metadata={"Creator": "rescore_cluster_representativeness_release_v1.py", "CreationDate": None, "ModDate": None})
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv); out = args.output_dir.resolve(strict=False)
    if out.exists():
        raise RuntimeError(f"Output root must be absent: {out}")
    started = time.perf_counter()
    reviewed_fit = verify_reviewed_fit()
    verify_exact(FROZEN_UMAP, EXPECTED_FROZEN_UMAP)
    manifest, facts = derive_current_manifest()
    out.mkdir(parents=True)
    manifest_path = out / "current_release_treatment_manifest.parquet"
    manifest.write_parquet(manifest_path, compression="zstd")
    manifest_provenance = {
        "version": "jump-lite-current-release-treatment-manifest-v1",
        "derivation": "distinct Metadata_JCP2022 where Metadata_Perturbation_Type == compound and Metadata_pert_type == trt; sorted by identifier; attributes joined from the frozen historical manifest",
        "exclusion": "The sole compound negative control is explicitly excluded.",
        "inputs": [facts["metadata"], facts["historical_manifest"]],
        "manifest": output_record(manifest_path, out),
        "validation": {key: value for key, value in facts.items() if key not in {"metadata", "historical_manifest"}},
    }
    write_json(out / "current_release_manifest_provenance.json", manifest_provenance)

    fitdir = HISTORICAL_ROOT / "fit"
    assignments = pl.read_parquet(fitdir / "cluster_assignments_unlabeled.parquet")
    selected_set = set(manifest["Metadata_JCP2022"].to_list())
    frame = assignments.with_columns(pl.col("Metadata_JCP2022").is_in(selected_set).alias("selected"))
    if int(frame["selected"].sum()) != CURRENT_TREATMENTS or not frame.filter(pl.col("selected"))["fit_eligible"].all():
        raise RuntimeError("Current assignment count/eligibility drift")
    pdf = frame.to_pandas(); y = pdf.selected.astype(int).to_numpy(); eligible = pdf.fit_eligible.to_numpy(bool)
    metrics: list[dict[str, object]] = []; score_sets: dict[str, dict[str, np.ndarray]] = {}
    for name, mask in (("eligible_primary", eligible), ("all_descriptive", np.ones(len(pdf), bool))):
        local = pdf.loc[mask].reset_index(drop=True); yy = y[mask]; scores = analysis.crossfit_scores(local, yy); score_sets[name] = scores
        for predictor, score in scores.items():
            metrics.append(analysis.metric_row(yy, score, predictor, name, int(yy.sum())))
    totals = np.bincount(pdf.cluster_id, minlength=analysis.PRIMARY_K)
    positives = np.bincount(pdf.cluster_id, weights=y, minlength=analysis.PRIMARY_K)
    fractions = positives / totals; global_prevalence = y.mean(); oof_all = score_sets["all_descriptive"]["cluster_only"]
    pdf["cluster_selected_fraction"] = fractions[pdf.cluster_id]
    pdf["cluster_selected_lift"] = pdf.cluster_selected_fraction / global_prevalence
    pdf["selection_probability_oof"] = oof_all
    pdf["selection_score_rank"] = rankdata(-oof_all, method="average")
    pdf.to_parquet(out / "compound_cluster_assignments.parquet", index=False)
    structural = (pdf.groupby("structural_stratum", sort=True).selected
                  .agg(n_compounds="size", n_selected="sum").reset_index())
    structural["selected_fraction"] = structural.n_selected / structural.n_compounds
    structural.to_csv(out / "structural_counts.csv", index=False)

    strata = pdf.structural_stratum.to_numpy(); clusters = pdf.cluster_id.to_numpy(); unique_strata = sorted(set(strata))
    expected = np.zeros(analysis.PRIMARY_K); variance = np.zeros(analysis.PRIMARY_K)
    for stratum in unique_strata:
        idx = strata == stratum; probability = y[idx].mean(); counts = np.bincount(clusters[idx], minlength=analysis.PRIMARY_K)
        expected += counts * probability; variance += counts * probability * (1 - probability)
    residual = (positives - expected) / np.sqrt(np.maximum(variance, 1e-12))
    observed = float(np.sum((positives[variance > 0] - expected[variance > 0]) ** 2 / variance[variance > 0]))
    rng = np.random.default_rng(20260811); exceed = np.zeros(analysis.PRIMARY_K, int); global_exceed = 0
    for _ in range(PERMUTATIONS):
        permuted = np.empty_like(y)
        for stratum in unique_strata:
            idx = np.flatnonzero(strata == stratum); permuted[idx] = rng.permutation(y[idx])
        counts = np.bincount(clusters, weights=permuted, minlength=analysis.PRIMARY_K)
        exceed += np.abs(counts - expected) >= np.abs(positives - expected) - 1e-12
        statistic = float(np.sum((counts[variance > 0] - expected[variance > 0]) ** 2 / variance[variance > 0]))
        global_exceed += statistic >= observed - 1e-12
    cluster_p = (exceed + 1) / (PERMUTATIONS + 1); cluster_q = analysis.bh_adjust(cluster_p)
    rows = []
    for cluster in range(analysis.PRIMARY_K):
        idx = clusters == cluster; n = int(idx.sum()); k = int(y[idx].sum()); smooth = (k + .5) / (n + 1)
        low, high = beta.ppf([.025, .975], k + .5, n - k + .5)
        rows.append({"cluster_id": cluster, "n_compounds": n, "n_selected": k, "n_nonselected": n-k,
                     "selected_fraction": k/n, "jeffreys_probability": smooth, "jeffreys_working_low": low,
                     "jeffreys_working_high": high, "overall_selected_fraction": global_prevalence,
                     "selection_lift": k/n/global_prevalence, "log2_selection_lift": math.log2(max(k/n/global_prevalence, 1e-12)),
                     "conditional_expected_selected": expected[cluster],
                     "conditional_lift": k/expected[cluster] if expected[cluster] > 0 else np.nan,
                     "conditional_standardized_residual": residual[cluster], "conditional_permutation_p": cluster_p[cluster],
                     "conditional_bh_q": cluster_q[cluster],
                     "cluster_distance_median": float(np.median(pdf.cluster_distance.to_numpy()[idx])),
                     "cluster_distance_p90": float(np.quantile(pdf.cluster_distance.to_numpy()[idx], .9)),
                     "n_wells_median": float(np.median(pdf.n_wells.to_numpy()[idx])),
                     "n_sources_median": float(np.median(pdf.n_sources.to_numpy()[idx])),
                     "n_plates_median": float(np.median(pdf.n_plates.to_numpy()[idx]))})
    table = pd.DataFrame(rows); table.to_csv(out / "cluster_selection_table.csv", index=False)
    table.to_parquet(out / "cluster_selection_table.parquet", index=False)
    metric_frame = pd.DataFrame(metrics); metric_frame.to_csv(out / "retrieval_metrics.csv", index=False)

    eligible_clusters = clusters[eligible]; eligible_y = y[eligible]
    eligible_counts = np.bincount(eligible_clusters, minlength=analysis.PRIMARY_K)
    eligible_selected = np.bincount(eligible_clusters, weights=eligible_y, minlength=analysis.PRIMARY_K)
    ps = eligible_selected / eligible_selected.sum(); pe = eligible_counts / eligible_counts.sum(); occupied = eligible_selected > 0
    middle = .5 * (ps + pe)
    js = .5 * np.sum(ps[ps > 0] * np.log(ps[ps > 0] / middle[ps > 0])) + .5 * np.sum(pe * np.log(pe / middle))
    representation = {"clusters": analysis.PRIMARY_K, "selected_occupied_clusters": int(occupied.sum()),
                      "occupied_fraction": float(occupied.mean()),
                      "eligible_compound_mass_in_occupied_clusters": float(eligible_counts[occupied].sum() / eligible_counts.sum()),
                      "total_variation_selected_vs_eligible": float(.5 * np.abs(ps - pe).sum()),
                      "jensen_shannon_divergence_nats": float(js),
                      "selection_lift_median": float(np.median(table.selection_lift)),
                      "selection_lift_iqr": [float(table.selection_lift.quantile(.25)), float(table.selection_lift.quantile(.75))]}
    write_json(out / "representation_summary.json", representation)
    permutation = {"permutations": PERMUTATIONS, "shuffle": "within structural_stratum",
                   "observed_statistic": observed, "one_sided_p": (global_exceed + 1) / (PERMUTATIONS + 1),
                   "interpretation": "finite-cohort design-null; not population inference"}
    write_json(out / "permutation_summary.json", permutation)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].bar(table.cluster_id, table.n_compounds, color="#777777"); axes[0].set_ylabel("Compounds")
    axes[1].bar(table.cluster_id, table.selected_fraction, color="#2b8cbe")
    axes[1].axhline(global_prevalence, color="black", ls="--", lw=1, label="overall")
    axes[1].set_ylabel("Selected fraction"); axes[1].set_xlabel("Canonical cluster ID"); axes[1].legend(); fig.tight_layout()
    fig.savefig(out / "cluster_selection_all_clusters.png", dpi=180); plt.close(fig)

    sensitivity = partition_sensitivity(selected_set, fitdir)
    sensitivity.to_csv(out / "retrieval_partition_sensitivity.csv", index=False)
    eligible_metrics = metric_frame[metric_frame.universe == "eligible_primary"].set_index("predictor")
    ap_ratio = float(eligible_metrics.loc["count_plus_cluster", "average_precision"] / eligible_metrics.loc["count_only", "average_precision"])
    detectable = bool(permutation["one_sided_p"] <= .05 and eligible_metrics.loc["count_plus_cluster", "log_loss"] < eligible_metrics.loc["count_only", "log_loss"])
    material = bool(detectable and ap_ratio >= 1.25)
    diagnostics = pd.read_csv(fitdir / "clustering_diagnostics.csv")
    mean_ari = float(diagnostics.query("preprocessing=='clip10' and k==128 and ari==ari")["ari"].mean())
    conclusion = {"detectably_better_than_structure": detectable, "materially_better_than_structure": material,
                  "combined_to_count_ap_ratio": ap_ratio, "conditional_permutation_p": permutation["one_sided_p"],
                  "primary_seed_mean_ari": mean_ari, "unique_cluster_biological_claims_allowed": bool(mean_ari >= .8)}
    write_json(out / "selection_conclusion.json", conclusion)

    summary_values = {
        "total_compounds": 115_721., "fit_eligible_compounds": 95_426., "selected_compounds": float(CURRENT_TREATMENTS),
        "all_compound_prevalence": float(y.mean()), "eligible_compound_prevalence": float(eligible_y.mean()),
        "occupied_clusters": float(occupied.sum()), "total_clusters": 128.,
        "eligible_mass_in_occupied_clusters": representation["eligible_compound_mass_in_occupied_clusters"],
        "total_variation_selected_vs_eligible": representation["total_variation_selected_vs_eligible"],
        "jensen_shannon_divergence_nats": representation["jensen_shannon_divergence_nats"],
        "average_precision_constant": float(eligible_metrics.loc["constant", "average_precision"]),
        "average_precision_acquisition_structure": float(eligible_metrics.loc["count_only", "average_precision"]),
        "average_precision_cluster_only": float(eligible_metrics.loc["cluster_only", "average_precision"]),
        "average_precision_structure_plus_cluster": float(eligible_metrics.loc["count_plus_cluster", "average_precision"]),
        "combined_to_structure_ap_ratio": ap_ratio, "conditional_permutation_p": permutation["one_sided_p"],
        "mean_seed_adjusted_rand_index": mean_ari,
    }
    qualifications = {
        "eligible_mass_in_occupied_clusters": "Broad occupancy, not proportional representation.",
        "total_variation_selected_vs_eligible": "Selection is non-uniform across operational clusters.",
        "jensen_shannon_divergence_nats": "Descriptive divergence, not a biological-class distance.",
        "average_precision_cluster_only": "Out-of-fold; cluster assignment is label-blind.",
        "average_precision_acquisition_structure": "Uses well/source/plate acquisition counts only.",
        "average_precision_structure_plus_cluster": "Evaluate against the preregistered 1.25x materiality gate.",
        "combined_to_structure_ap_ratio": "Preregistered materiality threshold: 1.25x.",
        "conditional_permutation_p": "2,000 permutations within acquisition strata; finite-cohort design null.",
        "mean_seed_adjusted_rand_index": "Low stability: do not interpret clusters as biological classes.",
    }
    summary = pd.DataFrame([{"metric": key, "value": value,
                             "unit": "count" if key in {"total_compounds", "fit_eligible_compounds", "selected_compounds", "occupied_clusters", "total_clusters"}
                                     else "nats" if key == "jensen_shannon_divergence_nats" else "fraction_or_score",
                             "qualification": qualifications.get(key, "")} for key, value in summary_values.items()])
    summary.to_csv(out / "cluster_selection_summary_table.csv", index=False, float_format="%.15g")

    frozen_embedding = pl.read_parquet(FROZEN_UMAP)
    expected_ids = assignments["Metadata_JCP2022"].to_list()
    if frozen_embedding["Metadata_JCP2022"].to_list() != expected_ids:
        raise RuntimeError("Frozen UMAP/assignment row-order drift")
    embedding = frozen_embedding.select("Metadata_JCP2022", "fit_eligible", "cluster_id", "umap_1", "umap_2").with_columns(
        pl.col("Metadata_JCP2022").is_in(selected_set).alias("selected")
    ).select("Metadata_JCP2022", "selected", "fit_eligible", "cluster_id", "umap_1", "umap_2")
    if int(embedding["selected"].sum()) != CURRENT_TREATMENTS:
        raise RuntimeError("Current UMAP selected-count drift")
    embedding.write_parquet(out / "compound_cluster_umap.parquet", compression="zstd")
    render_figure(embedding, summary, out)
    coordinate_digest = hashlib.sha256()
    for identifier in embedding["Metadata_JCP2022"].to_list():
        coordinate_digest.update(identifier.encode()); coordinate_digest.update(b"\0")
    coordinate_digest.update(np.ascontiguousarray(embedding.select("umap_1", "umap_2").to_numpy(), dtype="<f4").tobytes())
    figure_provenance = {
        "version": "cluster-selection-current-release-compound-figure-v1", "purpose": "visualization_only",
        "label_blind_contract": "The existing label-blind UMAP and cluster coordinates were reused byte-for-byte before current-release labels were joined; no fit was run.",
        "inputs": [repository_record(FROZEN_UMAP), output_record(manifest_path, out), output_record(out / "retrieval_metrics.csv", out),
                   output_record(out / "representation_summary.json", out), output_record(out / "permutation_summary.json", out)],
        "counts": {"compounds": embedding.height, "selected_treatments": CURRENT_TREATMENTS,
                   "fit_eligible": int(embedding["fit_eligible"].sum()), "clusters": int(embedding["cluster_id"].n_unique())},
        "coordinate_digest_sha256_ids_then_float32": coordinate_digest.hexdigest(),
        "qualifications": ["UMAP is display-only and not inferential.", "Coverage is broad, not proportional.",
                           "Low ARI prohibits biological-class interpretation."]}
    write_json(out / "cluster_selection_figure_provenance.json", figure_provenance)
    figure_files = [out / name for name in ("compound_cluster_umap.parquet", "cluster_selection_compound_map.png",
                                             "cluster_selection_compound_map.pdf", "cluster_selection_summary_table.csv",
                                             "cluster_selection_figure_provenance.json")]
    write_json(out / "cluster_selection_figure_hashes.json", {"path_base_definition": {OUTPUT_SCOPE: "current release output root"},
                                                               "files": [output_record(path, out) for path in figure_files]})

    report = f"""# Current-release cluster-selection representativeness report

## Scope and cohort identity

The tracked release metadata contains exactly **3,776 compound identifiers: 3,775 treatments plus one negative control** ({NEGATIVE_CONTROL_ID}). The negative control is excluded. The 3,775 treatments are a strict subset of the historical frozen 3,832-treatment manifest: 57 identifiers are historical-only and none are current-only. Every current treatment has an existing fit-eligible frozen assignment.

This rescore reuses the reviewed label-blind CellProfiler partition, assignments, diagnostics, model, and display UMAP without refitting. It changes only current-label-dependent calculations. The historical 3,832-sized matched-comparator sensitivities are deliberately omitted; they are not current-release evidence.

## Current results

The 3,775 treatments occupy {representation['selected_occupied_clusters']}/128 operational clusters, covering {representation['eligible_compound_mass_in_occupied_clusters']:.2%} of eligible compounds. Coverage is broad, not proportional: TV is {representation['total_variation_selected_vs_eligible']:.4f} and Jensen--Shannon divergence is {representation['jensen_shannon_divergence_nats']:.4f} nats.

Eligible-universe five-fold OOF AP is {eligible_metrics.loc['constant','average_precision']:.6f} constant, {eligible_metrics.loc['count_only','average_precision']:.6f} count/structure-only, {eligible_metrics.loc['cluster_only','average_precision']:.6f} cluster-only, and {eligible_metrics.loc['count_plus_cluster','average_precision']:.6f} count/structure-plus-cluster. The combined/count-only ratio is {ap_ratio:.6f}. The 2,000-shuffle within-stratum finite-cohort design-null p is {permutation['one_sided_p']:.6f}. The detectable gate is {str(detectable).lower()} and the 1.25x materiality gate is {str(material).lower()}.

## Qualifications

This is a finite-cohort design-null analysis, not population inference. Operational clusters have low stability (mean seed ARI {mean_ari:.3f}) and are not biological classes. Acquisition structure is strongly confounded with selection. Coverage is broad but not proportional. The scope is the fixed CellProfiler feature projection and makes no model-rank claim for broader JUMP or other representations. The frozen display UMAP is visualization only.
"""
    (out / "REPORT.md").write_text(report)
    runner_record = repository_record(Path(__file__))
    provenance = {"version": "profile-cluster-selection-current-release-v1", "runner": runner_record,
                  "design_addendum": repository_record(DESIGN_ADDENDUM), "cohort": manifest_provenance,
                  "reviewed_fit_identity": reviewed_fit, "frozen_umap": repository_record(FROZEN_UMAP),
                  "counts": {"all": len(pdf), "eligible": int(eligible.sum()), "selected_treatments": int(y.sum()),
                             "release_compound_identifiers": CURRENT_COMPOUND_IDENTIFIERS, "release_negative_controls_excluded": 1,
                             "historical_only": HISTORICAL_ONLY, "current_only": 0},
                  "matched_comparator_sensitivity": "omitted: historical comparators have 3,832 rows and are not reused as current evidence",
                  "conclusion": conclusion, "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
                  "polars": pl.__version__, "scipy": scipy.__version__, "sklearn": sklearn_version,
                  "runtime_seconds": time.perf_counter() - started}
    write_json(out / "provenance.json", provenance)
    identity = {"version": "profile-cluster-selection-current-release-v1", "runner": runner_record,
                "design_addendum": repository_record(DESIGN_ADDENDUM), "metadata": facts["metadata"], "manifest": output_record(manifest_path, out),
                "reviewed_fit_identity": reviewed_fit, "frozen_umap": repository_record(FROZEN_UMAP),
                "parameters": {"folds": 5, "permutations": PERMUTATIONS,
                               "permutation_strata": ["ineligible_lt4", "w4_7_single", "w4_7_multi", "w8plus"],
                               "refit_clusters": False, "refit_umap": False, "matched_comparators": "omitted"}}
    write_json(out / "computation_identity.json", identity)
    output_files = [path for path in sorted(out.iterdir()) if path.is_file() and path.name != "output_hashes.json"]
    write_json(out / "output_hashes.json", {"path_base_definition": {OUTPUT_SCOPE: "current release output root"},
                                             "files": [output_record(path, out) for path in output_files]})
    if {path.name for path in out.iterdir()} != OUTPUT_NAMES:
        raise RuntimeError(f"Current output inventory drift: {sorted(path.name for path in out.iterdir())}")
    print(json.dumps({"manifest": manifest_provenance["manifest"], "representation": representation,
                      "permutation": permutation, "conclusion": conclusion}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
