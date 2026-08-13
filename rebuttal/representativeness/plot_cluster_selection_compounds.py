#!/usr/bin/env python3
"""Render the frozen cluster-selection result at compound level.

The UMAP is a display-only projection. It is fitted without JUMP-lite labels on
all fit-eligible compounds in the frozen PCA32 space; selection labels are
joined only after the two-dimensional coordinates have been fixed. This script
never refits or changes the scientific clustering model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import sklearn
import umap

ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parent.parent
REPOSITORY_SCOPE = "repository_root"
FIGURE_OUTPUT_SCOPE = "figure_output_root"
PROFILE_ROOT = ROOT / "outputs/profile_space"
RESULT_ROOT = ROOT / "outputs/profile_cluster_representativeness_v1"
CONSENSUS = PROFILE_ROOT / "compound_consensus_plate_robust.parquet"
SELECTED = PROFILE_ROOT / "manifests/jump_lite_compounds.parquet"
MODEL = RESULT_ROOT / "fit/primary_model.npz"
UNLABELED = RESULT_ROOT / "fit/cluster_assignments_unlabeled.parquet"
ASSIGNMENTS = RESULT_ROOT / "compound_cluster_assignments.parquet"
CLUSTER_TABLE = RESULT_ROOT / "cluster_selection_table.parquet"
RETRIEVAL = RESULT_ROOT / "retrieval_metrics.csv"
REPRESENTATION = RESULT_ROOT / "representation_summary.json"
PERMUTATION = RESULT_ROOT / "permutation_summary.json"
DIAGNOSTICS = RESULT_ROOT / "fit/clustering_diagnostics.csv"

OUTPUT_NAMES = (
    "compound_cluster_umap.parquet",
    "cluster_selection_compound_map.png",
    "cluster_selection_compound_map.pdf",
    "cluster_selection_summary_table.csv",
    "cluster_selection_figure_provenance.json",
    "cluster_selection_figure_hashes.json",
)
EXPECTED = {
    CONSENSUS: (40_554_151, "dc2f84178a15f2e18177d4475b094af0da8fab10b1856bd3d1e4f6521d6c9d06"),
    SELECTED: (33_252, "a0671dcaae029a2c32ac58fdaf09178806b495d33a5ea439ff859b3c0fbe74de"),
    MODEL: (31_712, "3546f6abe83ef85b2aa809cdbf4209dd0b2a979b3415c3e7eb956bf4339a8a54"),
    UNLABELED: (1_161_683, "20a1282d0e81ad6d26cd443d327ba86282cd38981b7751d871bc170bec940613"),
    ASSIGNMENTS: (3_000_851, "01f1d45d9d05bd0d7498e39a48adaf0a2364fd6d205ca79dde37a7a88b984450"),
    CLUSTER_TABLE: (29_857, "06ad9a73de1e385491563b9be7e1e78fd6a7b11ff4df302d9d289e9f3c14a938"),
    RETRIEVAL: (1_786, "ae4b3945a82e60de7375ef217e66d0b68846ad0934d55c58cd46afd7c39e7949"),
    REPRESENTATION: (404, "fcacac30785d710693a2418226881e9779f06db2ea88bc69c4c20f7bc088e973"),
    PERMUTATION: (228, "b044f2d17ed64ac9843d62321d58026e9040e0dc268555236e3f6e963541aaaa"),
    DIAGNOSTICS: (4_708, "9df79ab9f92e6c3c1ca0b268e6544a9d59647a5c1e7bb345f86ce0b07e628709"),
}


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def record(
    path: Path,
    *,
    base: Path = REPOSITORY_ROOT,
    scope: str = REPOSITORY_SCOPE,
) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    resolved_base = base.resolve(strict=True)
    try:
        relative = resolved.relative_to(resolved_base)
    except ValueError as error:
        raise RuntimeError(f"Figure artifact escapes {scope}: {resolved}") from error
    if relative == Path(".") or ".." in relative.parts:
        raise RuntimeError(f"Invalid figure artifact path for {scope}: {relative}")
    return {
        "path": relative.as_posix(),
        "path_scope": scope,
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def verify(path: Path) -> dict[str, object]:
    rec = record(path)
    expected = EXPECTED[path]
    if (rec["bytes"], rec["sha256"]) != expected:
        raise RuntimeError(f"Frozen input identity drift: {path}: {rec}")
    return rec


def require_outputs_absent(output_dir: Path, names: Iterable[str] = OUTPUT_NAMES) -> None:
    existing = [str(output_dir / name) for name in names if (output_dir / name).exists()]
    if existing:
        raise RuntimeError("Refusing to overwrite figure outputs: " + ", ".join(existing))


def reconstruct_pca32(consensus: pl.DataFrame, model_path: Path = MODEL) -> np.ndarray:
    """Project profiles through the frozen preprocessing/PCA model."""
    with np.load(model_path) as model:
        features = model["feature_names"].tolist()
        raw = consensus.select(features).to_numpy().astype(np.float64, copy=False)
        impute = model["imputation_medians"].astype(np.float64)
        medians = model["scaling_medians"].astype(np.float64)
        iqrs = model["scaling_iqrs"].astype(np.float64)
        clip = float(model["clip_limit"][0])
        values = np.where(np.isfinite(raw), raw, impute)
        scaled = np.clip((values - medians) / iqrs, -clip, clip).astype(np.float32)
        coords = (scaled - model["pca_mean"]) @ model["pca_components"].T
    if coords.shape != (115_721, 32) or not np.isfinite(coords).all():
        raise RuntimeError(f"Frozen PCA reconstruction drift: {coords.shape}")
    return np.asarray(coords, dtype=np.float32)


def fit_label_blind_umap(pca32: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    """Fit display embedding before any selection labels are loaded or joined."""
    if pca32.shape != (115_721, 32) or eligible.shape != (115_721,):
        raise RuntimeError("Unexpected PCA/eligibility dimensions")
    if int(eligible.sum()) != 95_426:
        raise RuntimeError("Fit-eligible count drift")
    reducer = umap.UMAP(
        n_neighbors=30,
        min_dist=0.15,
        n_components=2,
        metric="euclidean",
        random_state=2026,
        transform_seed=2026,
        n_jobs=1,
        low_memory=True,
    )
    embedded = np.empty((len(pca32), 2), dtype=np.float32)
    embedded[eligible] = reducer.fit_transform(pca32[eligible]).astype(np.float32)
    embedded[~eligible] = reducer.transform(pca32[~eligible]).astype(np.float32)
    if not np.isfinite(embedded).all():
        raise RuntimeError("Non-finite UMAP coordinates")
    return embedded


def coordinate_digest(ids: list[str], coordinates: np.ndarray) -> str:
    if len(ids) != len(coordinates):
        raise ValueError("ID/coordinate length mismatch")
    digest = hashlib.sha256()
    for value in ids:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    digest.update(np.ascontiguousarray(coordinates, dtype="<f4").tobytes())
    return digest.hexdigest()


def join_selection_after_embedding(
    unlabeled: pl.DataFrame, coordinates: np.ndarray, selected_ids: set[str]
) -> pl.DataFrame:
    """Attach selection labels only after the label-blind embedding is fixed."""
    ids = unlabeled["Metadata_JCP2022"].to_list()
    if len(ids) != 115_721 or len(set(ids)) != len(ids):
        raise RuntimeError("Unlabeled compound identity drift")
    universe = set(ids)
    missing = selected_ids - universe
    if missing:
        raise RuntimeError(f"Selected IDs absent from frozen universe: {len(missing)}")
    selected = np.fromiter((value in selected_ids for value in ids), dtype=np.bool_, count=len(ids))
    if int(selected.sum()) != 3_832:
        raise RuntimeError("Selected count drift")
    return pl.DataFrame(
        {
            "Metadata_JCP2022": ids,
            "selected": selected,
            "fit_eligible": unlabeled["fit_eligible"],
            "cluster_id": unlabeled["cluster_id"],
            "umap_1": coordinates[:, 0],
            "umap_2": coordinates[:, 1],
        }
    )


def build_summary_table() -> pd.DataFrame:
    assignments = pl.read_parquet(ASSIGNMENTS)
    metrics = pd.read_csv(RETRIEVAL)
    representation = json.loads(REPRESENTATION.read_text())
    permutation = json.loads(PERMUTATION.read_text())
    diagnostics = pd.read_csv(DIAGNOSTICS)
    eligible = metrics[metrics["universe"] == "eligible_primary"].set_index("predictor")
    ari = diagnostics[
        (diagnostics["preprocessing"] == "clip10")
        & (diagnostics["k"] == 128)
        & diagnostics["ari"].notna()
    ]["ari"].mean()
    values = {
        "total_compounds": 115_721.0,
        "fit_eligible_compounds": 95_426.0,
        "selected_compounds": 3_832.0,
        "all_compound_prevalence": float(assignments["selected"].mean()),
        "eligible_compound_prevalence": float(eligible.loc["count_only", "prevalence"]),
        "occupied_clusters": float(representation["selected_occupied_clusters"]),
        "total_clusters": float(representation["clusters"]),
        "eligible_mass_in_occupied_clusters": float(
            representation["eligible_compound_mass_in_occupied_clusters"]
        ),
        "total_variation_selected_vs_eligible": float(
            representation["total_variation_selected_vs_eligible"]
        ),
        "jensen_shannon_divergence_nats": float(
            representation["jensen_shannon_divergence_nats"]
        ),
        "average_precision_constant": float(eligible.loc["constant", "average_precision"]),
        "average_precision_acquisition_structure": float(
            eligible.loc["count_only", "average_precision"]
        ),
        "average_precision_cluster_only": float(eligible.loc["cluster_only", "average_precision"]),
        "average_precision_structure_plus_cluster": float(
            eligible.loc["count_plus_cluster", "average_precision"]
        ),
        "combined_to_structure_ap_ratio": float(
            eligible.loc["count_plus_cluster", "average_precision"]
            / eligible.loc["count_only", "average_precision"]
        ),
        "conditional_permutation_p": float(permutation["one_sided_p"]),
        "mean_seed_adjusted_rand_index": float(ari),
    }
    expected = {
        "total_compounds": 115_721.0,
        "fit_eligible_compounds": 95_426.0,
        "selected_compounds": 3_832.0,
        "all_compound_prevalence": 0.03311412794566241,
        "eligible_compound_prevalence": 0.040156770691425814,
        "occupied_clusters": 120.0,
        "total_clusters": 128.0,
        "eligible_mass_in_occupied_clusters": 0.962368746463228,
        "total_variation_selected_vs_eligible": 0.47504893669424875,
        "jensen_shannon_divergence_nats": 0.17227756705149666,
        "average_precision_constant": 0.03946976570130267,
        "average_precision_acquisition_structure": 0.5158264268349165,
        "average_precision_cluster_only": 0.27766569255144513,
        "average_precision_structure_plus_cluster": 0.5764816772139572,
        "combined_to_structure_ap_ratio": 1.117588489506476,
        "conditional_permutation_p": 0.0004997501249375,
        "mean_seed_adjusted_rand_index": 0.16241422160818178,
    }
    for key, expected_value in expected.items():
        if not np.isclose(values[key], expected_value, rtol=0, atol=1e-12):
            raise RuntimeError(f"Summary value drift for {key}: {values[key]}")
    qualifications = {
        "eligible_mass_in_occupied_clusters": "Broad occupancy, not proportional representation.",
        "total_variation_selected_vs_eligible": "Selection is non-uniform across operational clusters.",
        "jensen_shannon_divergence_nats": "Descriptive divergence, not a biological-class distance.",
        "average_precision_cluster_only": "Out-of-fold; cluster assignment is label-blind.",
        "average_precision_acquisition_structure": "Uses well/source/plate acquisition counts only.",
        "average_precision_structure_plus_cluster": "Detectably additive, but below the 1.25x materiality gate.",
        "combined_to_structure_ap_ratio": "Preregistered materiality threshold: 1.25x.",
        "conditional_permutation_p": "2,000 permutations within acquisition strata; minimum attainable p.",
        "mean_seed_adjusted_rand_index": "Low stability: do not interpret clusters as biological classes.",
    }
    units = {
        key: "count" if key in {
            "total_compounds", "fit_eligible_compounds", "selected_compounds",
            "occupied_clusters", "total_clusters"
        } else "nats" if key == "jensen_shannon_divergence_nats" else "fraction_or_score"
        for key in values
    }
    return pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
                "unit": units[key],
                "qualification": qualifications.get(key, ""),
            }
            for key, value in values.items()
        ]
    )


def render_figure(embedding: pl.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    data = embedding.to_pandas()
    selected = data["selected"].to_numpy(dtype=bool)
    fit_eligible = data["fit_eligible"].to_numpy(dtype=bool)
    cluster_ids = data["cluster_id"].to_numpy(dtype=int)
    x = data["umap_1"].to_numpy()
    y = data["umap_2"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.25), gridspec_kw={"width_ratios": [1.35, 1, 0.92]})
    ax = axes[0]
    ax.scatter(x[~selected], y[~selected], s=1.0, c="#8a8a8a", alpha=0.075,
               linewidths=0, rasterized=True, label="Other compounds (111,889)")
    ax.scatter(x[selected], y[selected], s=4.5, c="#0072B2", alpha=0.52,
               linewidths=0, rasterized=True, label="JUMP-lite (3,832)")
    ax.set_title("A  Compound profile map", loc="left", fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(frameon=False, loc="lower left", markerscale=2.8, fontsize=8)
    ax.text(0.01, 0.99, "UMAP fit label-blind on 95,426 eligible profiles",
            transform=ax.transAxes, va="top", ha="left", fontsize=7.5, color="#444444")

    ax = axes[1]
    eligible_counts = np.bincount(cluster_ids[fit_eligible], minlength=128).astype(float)
    selected_counts = np.bincount(cluster_ids[selected], minlength=128).astype(float)
    population_share = eligible_counts / eligible_counts.sum()
    selected_share = selected_counts / selected_counts.sum()
    occupied = selected_counts > 0
    upper = max(float(population_share.max()), float(selected_share.max())) * 1.07
    ax.plot([0, upper], [0, upper], linestyle="--", color="#555555", linewidth=1,
            label="Proportional selection")
    ax.scatter(population_share[occupied], selected_share[occupied], s=25,
               c="#0072B2", alpha=0.72, linewidths=0, label="Occupied (120)")
    ax.scatter(population_share[~occupied], selected_share[~occupied], s=32,
               c="#D55E00", marker="x", linewidths=1.4, label="Unoccupied (8)")
    ax.set_xlim(-0.001, upper); ax.set_ylim(-0.001, upper)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("B  Coverage across all 128 clusters", loc="left", fontweight="bold")
    ax.set_xlabel("Share of eligible compounds")
    ax.set_ylabel("Share of selected compounds")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.text(0.98, 0.02, "120/128 occupied\n96.24% of eligible mass",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)

    ax = axes[2]
    summary_values = summary.set_index("metric")["value"]
    labels = ["Constant", "Cluster only", "Acquisition\nstructure", "Structure\n+ cluster"]
    values = [
        summary_values["average_precision_constant"],
        summary_values["average_precision_cluster_only"],
        summary_values["average_precision_acquisition_structure"],
        summary_values["average_precision_structure_plus_cluster"],
    ]
    colors = ["#B3B3B3", "#56B4E9", "#8A8A8A", "#0072B2"]
    bars = ax.bar(np.arange(4), values, color=colors, width=0.72)
    ax.set_ylim(0, 0.70)
    ax.set_xticks(np.arange(4), labels, fontsize=8)
    ax.set_ylabel("Out-of-fold average precision")
    ax.set_title("C  Retrieval of selected compounds", loc="left", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.016, f"{value:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.text(0.98, 0.96,
            "Combined / structure = 1.118×\nMateriality gate = 1.25×\nConditional p = 0.000500",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc"})

    fig.suptitle("JUMP-lite spans most operational profile clusters, but cluster signal is only modestly additive",
                 x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, 0.012,
             "Display projection only. Operational clusters are unstable (mean seed ARI = 0.162) and are not biological classes.",
             ha="left", va="bottom", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.045, 1, 0.94), w_pad=2.0)
    fig.savefig(output_dir / "cluster_selection_compound_map.png", dpi=300,
                metadata={"Software": "plot_cluster_selection_compounds.py"})
    fig.savefig(output_dir / "cluster_selection_compound_map.pdf", dpi=300,
                metadata={"Creator": "plot_cluster_selection_compounds.py", "CreationDate": None,
                          "ModDate": None})
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RESULT_ROOT)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve(strict=True)
    require_outputs_absent(output_dir)
    non_label_inputs = [path for path in EXPECTED if path != SELECTED]
    inputs = [verify(path) for path in non_label_inputs]

    # Selection labels and their manifest are intentionally untouched until the
    # label-blind embedding coordinates and digest have been fixed.
    consensus = pl.read_parquet(CONSENSUS)
    unlabeled = pl.read_parquet(UNLABELED)
    ids = consensus["Metadata_JCP2022"].to_list()
    if ids != unlabeled["Metadata_JCP2022"].to_list():
        raise RuntimeError("Consensus/unlabeled row-order drift")
    eligible = unlabeled["fit_eligible"].to_numpy()
    pca32 = reconstruct_pca32(consensus)
    coordinates = fit_label_blind_umap(pca32, eligible)
    digest = coordinate_digest(ids, coordinates)

    inputs.append(verify(SELECTED))
    selected_manifest = pl.read_parquet(SELECTED)
    selected_ids = set(selected_manifest["Metadata_JCP2022"].to_list())
    embedding = join_selection_after_embedding(unlabeled, coordinates, selected_ids)
    embedding_path = output_dir / "compound_cluster_umap.parquet"
    embedding.write_parquet(embedding_path, compression="zstd")

    summary = build_summary_table()
    summary_path = output_dir / "cluster_selection_summary_table.csv"
    summary.to_csv(summary_path, index=False, float_format="%.15g")
    render_figure(embedding, summary, output_dir)

    provenance_path = output_dir / "cluster_selection_figure_provenance.json"
    provenance = {
        "version": "cluster-selection-compound-figure-v1",
        "purpose": "visualization_only",
        "path_base_definitions": {
            REPOSITORY_SCOPE: "repository checkout root",
        },
        "label_blind_contract": (
            "UMAP was fit on frozen PCA32 coordinates for eligible compounds before "
            "the selected manifest was read or joined; scientific clusters were not refit."
        ),
        "runner": record(Path(__file__)),
        "inputs": inputs,
        "parameters": {
            "umap_fit_rows": 95_426,
            "umap_fit_rule": "all fit_eligible compounds (n_wells >= 4)",
            "umap_neighbors": 30,
            "umap_min_dist": 0.15,
            "umap_metric": "euclidean",
            "seed": 2026,
            "transform_ineligible_rows": 20_295,
            "n_jobs": 1,
        },
        "counts": {
            "compounds": embedding.height,
            "selected": int(embedding["selected"].sum()),
            "fit_eligible": int(embedding["fit_eligible"].sum()),
            "clusters": int(embedding["cluster_id"].n_unique()),
        },
        "coordinate_digest_sha256_ids_then_float32": digest,
        "environment": {
            "numpy": np.__version__,
            "polars": pl.__version__,
            "scikit_learn": sklearn.__version__,
            "umap_learn": umap.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "qualifications": [
            "UMAP is a display projection and not an inferential result.",
            "Low seed stability (mean ARI 0.162) prohibits biological class interpretation.",
            "Cluster occupancy is broad but selection is non-proportional.",
        ],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    generated = [
        embedding_path,
        output_dir / "cluster_selection_compound_map.png",
        output_dir / "cluster_selection_compound_map.pdf",
        summary_path,
        provenance_path,
    ]
    hashes_path = output_dir / "cluster_selection_figure_hashes.json"
    output_records = [
        record(path, base=output_dir, scope=FIGURE_OUTPUT_SCOPE) for path in generated
    ]
    hashes_path.write_text(json.dumps({
        "files": output_records,
        "path_base_definition": {
            FIGURE_OUTPUT_SCOPE: "directory containing cluster_selection_figure_hashes.json"
        },
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"coordinate_digest": digest, "outputs": output_records},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
