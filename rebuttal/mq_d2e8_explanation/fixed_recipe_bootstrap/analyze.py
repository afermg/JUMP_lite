#!/usr/bin/env python3
"""Paired fixed-recipe Target-2 MQ-versus-D2-E8 bootstrap.

Reads only archived sweep outputs. One recipe per Figure-3c family is selected
using Zstd PA*PC/100 and fixed across Zstd, D2-E8, and MQ. PA compound clusters
and PC target clusters are resampled independently with common codec weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SWEEP = Path("/work/datasets/JUMP-lite-wacv/sweeps/MAIN_RESULTS__figure_4_variance_first_v11")
SWEEP_CSV = SWEEP / "sweep_results.csv"
EXPECTED_SWEEP_SIZE = 1_096_114
EXPECTED_SWEEP_SHA256 = "08923c7bd27bca54c0a3f484429ced31d1b48ad097c974773591a89ac63eb53a"
SEED = 20_260_818
N_BOOTSTRAPS = 50_000
PA_CLUSTERS = 306
PC_CLUSTERS = 201
FAMILIES = ("cp_measure", "dinov2", "morphem", "openphenom", "subcell")
DISPLAY = {
    "cp_measure": "cp_measure",
    "dinov2": "DINOv2",
    "morphem": "MorphEM",
    "openphenom": "OpenPhenom",
    "subcell": "SubCell",
}
PREFIX = {
    "cp_measure": "cp_measure",
    "dinov2": "dinov2",
    "morphem": "morphem",
    "openphenom": "openphenom",
    "subcell": "subcell__clip01",
}
MODEL = {
    "cp_measure": "zstd_raw",
    "dinov2": "dinov2_zstd_raw",
    "morphem": "morphem_zstd_raw",
    "openphenom": "openphenom_zstd_raw",
    "subcell": "subcell__clip01_zstd_raw",
}
CODECS = (
    ("Zstd", "zstd", "Zstd"),
    ("D2-E8", "jpegxl_lossy_d2_e8", "D2-E8"),
    ("MQ", "jpegxl_lossy_mq", "MQ"),
)
EXPECTED_PA_QUERY_ROWS = {
    ("cp_measure", "Zstd"): 1266,
    ("cp_measure", "D2-E8"): 1263,
    ("cp_measure", "MQ"): 1264,
    **{
        (family, codec): 1280
        for family in ("dinov2", "morphem", "openphenom", "subcell")
        for codec in ("Zstd", "D2-E8", "MQ")
    },
}
EXPECTED_COMMON_PA_QUERY_ROWS = {
    "cp_measure": 1263,
    "dinov2": 1280,
    "morphem": 1280,
    "openphenom": 1280,
    "subcell": 1280,
}
CREATED_UTC = "2026-08-18T18:00:00+00:00"


class AnalysisError(RuntimeError):
    """Fail-closed validation error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise AnalysisError(f"missing or empty file: {path}")
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise AnalysisError(f"refusing to write empty CSV: {path}")
    atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def centered_pvalue(samples: np.ndarray, point: float) -> float:
    centered = samples - point
    return float((np.count_nonzero(np.abs(centered) >= abs(point)) + 1) / (len(samples) + 1))


def holm_adjust(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise AnalysisError("invalid p-values")
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate(ranked * (len(p) - np.arange(len(p))))
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def validate_sweep() -> pd.DataFrame:
    record = file_record(SWEEP_CSV)
    if record["size_bytes"] != EXPECTED_SWEEP_SIZE or record["sha256"] != EXPECTED_SWEEP_SHA256:
        raise AnalysisError("canonical Target-2 sweep_results.csv drift")
    sweep = pd.read_csv(SWEEP_CSV)
    required = {"model", "config", "PA", "PC", "PA_mean_nap", "PC_mean_nap"}
    if missing := required.difference(sweep.columns):
        raise AnalysisError(f"sweep columns missing: {sorted(missing)}")
    if len(sweep) != 2860:
        raise AnalysisError(f"unexpected sweep row count: {len(sweep)}")
    return sweep


def select_recipes(sweep: pd.DataFrame) -> dict[str, str]:
    winners: dict[str, str] = {}
    for family in FAMILIES:
        rows = sweep[sweep.model == MODEL[family]].copy()
        if len(rows) != 48 or rows.config.duplicated().any():
            raise AnalysisError(f"expected 48 unique Zstd recipes for {family}, found {len(rows)}")
        rows["selection_metric"] = rows.PA * rows.PC / 100.0
        best = rows.selection_metric.max()
        tied = rows[np.isclose(rows.selection_metric, best, rtol=0.0, atol=1e-15)]
        winners[family] = sorted(tied.config.astype(str))[0]
    return winners


def align_tables(tables: list[pd.DataFrame], key: str, expected: int) -> pd.DataFrame:
    if not tables:
        raise AnalysisError(f"no tables for {key}")
    base = tables[0]
    if len(base) != expected or base[key].duplicated().any():
        raise AnalysisError(f"invalid baseline {key} table")
    key_set = set(base[key].astype(str))
    for table in tables[1:]:
        if len(table) != expected or table[key].duplicated().any() or set(table[key].astype(str)) != key_set:
            raise AnalysisError(f"unaligned {key} tables")
        base = base.merge(table, on=key, validate="one_to_one")
    if len(base) != expected:
        raise AnalysisError(f"merged {key} count drift")
    return base.sort_values(key).reset_index(drop=True)


def restrict_to_common_pa_queries(
    queries: dict[str, pd.DataFrame], expected_common: int
) -> dict[str, pd.DataFrame]:
    """Return exact common-ID query tables after validating cluster mappings."""
    if len(queries) < 2:
        raise AnalysisError("at least two PA query variants are required")
    common_ids = set.intersection(*(set(table.Metadata_id.astype(str)) for table in queries.values()))
    if len(common_ids) != expected_common:
        raise AnalysisError(f"unexpected common PA query count: {len(common_ids)}")
    common_ids_sorted = sorted(common_ids)
    result: dict[str, pd.DataFrame] = {}
    expected_mapping: pd.DataFrame | None = None
    for codec, table in queries.items():
        current = table.copy()
        current["Metadata_id"] = current.Metadata_id.astype(str)
        current["Metadata_broad_sample"] = current.Metadata_broad_sample.astype(str)
        filtered = (
            current.loc[current.Metadata_id.isin(common_ids_sorted)]
            .sort_values("Metadata_id")
            .reset_index(drop=True)
        )
        if len(filtered) != expected_common or filtered.Metadata_id.duplicated().any():
            raise AnalysisError(f"common PA query coverage mismatch for {codec}")
        mapping = filtered[["Metadata_id", "Metadata_broad_sample"]]
        if expected_mapping is None:
            expected_mapping = mapping
        elif not mapping.equals(expected_mapping):
            raise AnalysisError(f"PA query-to-cluster mapping mismatch for {codec}")
        result[codec] = filtered
    return result


def validate_inputs(sweep: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Load fixed-recipe inputs and align PA on exact within-family query rows.

    Archived aggregate points are validated before restriction.  Each family's
    three codecs are then restricted to their exact common ``Metadata_id`` set,
    with a required one-to-one and codec-invariant mapping from query ID to broad
    sample.  This avoids silently comparing the unequal cp_measure query
    populations while preserving all 306 broad-sample bootstrap clusters.
    """
    winners = select_recipes(sweep)
    pa_tables: list[pd.DataFrame] = []
    pc_tables: list[pd.DataFrame] = []
    pa_cluster_key_sets: list[set[str]] = []
    records: list[dict[str, Any]] = []
    for family in FAMILIES:
        config = winners[family]
        prefix = PREFIX[family]
        variants: list[dict[str, Any]] = []
        for codec, folder_codec, label in CODECS:
            folder = SWEEP / f"{prefix}_jump_target2_4plate_{folder_codec}_raw_features" / config
            results = folder / "results"
            paths = {
                "metrics": results / "metrics.json",
                "pa_queries": results / "phenotypic_activity_per_compound.csv",
                "pa_map": results / "phenotypic_activity_map.csv",
                "pc": results / "phenotypic_consistency_per_target.csv",
                "config": folder / "pipeline_config.yaml",
                "output": folder / "output.parquet",
            }
            identities = {name: file_record(path) for name, path in paths.items()}
            metrics = json.loads(paths["metrics"].read_text())
            pa_query = pd.read_csv(paths["pa_queries"])
            pa_map = pd.read_csv(paths["pa_map"])
            pc = pd.read_csv(paths["pc"])
            pa_required = {"Metadata_broad_sample", "Metadata_id", "normalized_average_precision"}
            if missing := pa_required.difference(pa_query.columns):
                raise AnalysisError(f"PA query columns missing for {family}/{codec}: {sorted(missing)}")
            if pa_query[list(pa_required)].isna().any().any():
                raise AnalysisError(f"null PA query identity/value for {family}/{codec}")
            pa_query = pa_query.copy()
            pa_query["Metadata_id"] = pa_query.Metadata_id.astype(str)
            pa_query["Metadata_broad_sample"] = pa_query.Metadata_broad_sample.astype(str)
            if pa_query.Metadata_id.duplicated().any() or len(pa_query) != EXPECTED_PA_QUERY_ROWS[(family, codec)]:
                raise AnalysisError(f"invalid PA query keys/count for {family}/{codec}")
            original_agg = (
                pa_query.groupby("Metadata_broad_sample", as_index=False, sort=True)
                .normalized_average_precision.mean()
                .rename(columns={"normalized_average_precision": "mean_normalized_average_precision"})
            )
            if len(original_agg) != PA_CLUSTERS or original_agg.Metadata_broad_sample.duplicated().any():
                raise AnalysisError(f"unexpected PA cluster count for {family}/{codec}")
            expected_pa = pa_map[["Metadata_broad_sample", "mean_normalized_average_precision"]].copy()
            expected_pa["Metadata_broad_sample"] = expected_pa.Metadata_broad_sample.astype(str)
            expected_pa = expected_pa.sort_values("Metadata_broad_sample").reset_index(drop=True)
            if (
                len(expected_pa) != PA_CLUSTERS
                or not original_agg.Metadata_broad_sample.equals(expected_pa.Metadata_broad_sample)
                or not np.allclose(
                    original_agg.mean_normalized_average_precision,
                    expected_pa.mean_normalized_average_precision,
                    rtol=0.0,
                    atol=2e-15,
                )
            ):
                raise AnalysisError(f"PA query aggregation mismatch for {family}/{codec}")
            if len(pc) != PC_CLUSTERS or pc.Metadata_target.duplicated().any():
                raise AnalysisError(f"unexpected PC target keys for {family}/{codec}")
            archived_pa_point = float(original_agg.mean_normalized_average_precision.mean())
            pc_point = float(pc.mean_normalized_average_precision.mean())
            if not np.isfinite([archived_pa_point, pc_point]).all():
                raise AnalysisError(f"non-finite aggregate for {family}/{codec}")
            if (
                abs(archived_pa_point - float(metrics["PA_mean_nap"])) > 1e-12
                or abs(pc_point - float(metrics["PC_mean_nap"])) > 1e-12
            ):
                raise AnalysisError(f"metrics.json aggregate mismatch for {family}/{codec}")
            sweep_model = MODEL[family] if codec == "Zstd" else (
                f"{prefix}_{folder_codec}_raw" if family != "cp_measure" else f"{folder_codec}_raw"
            )
            row = sweep[(sweep.model == sweep_model) & (sweep.config == config)]
            if len(row) != 1:
                raise AnalysisError(f"sweep coverage mismatch for {family}/{codec}")
            for field in ("PA", "PC", "PA_mean_nap", "PC_mean_nap"):
                if abs(float(row.iloc[0][field]) - float(metrics[field])) > 1e-12:
                    raise AnalysisError(f"sweep/metrics {field} mismatch for {family}/{codec}")
            variants.append(
                {
                    "codec": codec,
                    "label": label,
                    "folder_codec": folder_codec,
                    "pa_query": pa_query,
                    "pc": pc,
                    "metrics": metrics,
                    "paths": paths,
                    "identities": identities,
                    "archived_pa_point": archived_pa_point,
                    "pc_point": pc_point,
                }
            )

        common_queries = restrict_to_common_pa_queries(
            {variant["codec"]: variant["pa_query"] for variant in variants},
            EXPECTED_COMMON_PA_QUERY_ROWS[family],
        )
        for variant in variants:
            common_query = common_queries[variant["codec"]]
            common_agg = (
                common_query.groupby("Metadata_broad_sample", as_index=False, sort=True)
                .normalized_average_precision.mean()
                .rename(columns={"normalized_average_precision": "mean_normalized_average_precision"})
            )
            if len(common_agg) != PA_CLUSTERS or common_agg.Metadata_broad_sample.duplicated().any():
                raise AnalysisError(f"common PA restriction lost broad-sample clusters for {family}/{variant['codec']}")
            pa_cluster_key_sets.append(set(common_agg.Metadata_broad_sample))
            common_pa_point = float(common_agg.mean_normalized_average_precision.mean())
            if not np.isfinite(common_pa_point):
                raise AnalysisError(f"non-finite common PA aggregate for {family}/{variant['codec']}")
            column = f"{family}__{variant['label']}"
            pa_tables.append(common_agg.rename(columns={"mean_normalized_average_precision": column}))
            pc_tables.append(
                variant["pc"][["Metadata_target", "mean_normalized_average_precision"]].rename(
                    columns={"mean_normalized_average_precision": column}
                )
            )
            paths = variant["paths"]
            identities = variant["identities"]
            original_rows = len(variant["pa_query"])
            records.append(
                {
                    "family": family,
                    "display_family": DISPLAY[family],
                    "codec": variant["label"],
                    "config": config,
                    "selection_metric": "PA*PC/100 on Zstd only; lexical exact-tie resolution",
                    "pa_clusters": PA_CLUSTERS,
                    "pc_clusters": PC_CLUSTERS,
                    "pa_query_rows_original": original_rows,
                    "pa_query_rows_common": len(common_query),
                    "pa_query_rows_dropped": original_rows - len(common_query),
                    "pa_point_archived": variant["archived_pa_point"],
                    "pa_point_common": common_pa_point,
                    "pc_point": variant["pc_point"],
                    "product_point_common": common_pa_point * variant["pc_point"],
                    **{f"{name}_path": str(paths[name]) for name in paths},
                    **{f"{name}_sha256": identities[name]["sha256"] for name in paths},
                    **{f"{name}_size_bytes": identities[name]["size_bytes"] for name in paths},
                }
            )
    baseline_pa_clusters = pa_cluster_key_sets[0]
    if any(keys != baseline_pa_clusters for keys in pa_cluster_key_sets[1:]):
        raise AnalysisError("PA Metadata_broad_sample key-set mismatch across family/codecs")
    pa = align_tables(pa_tables, "Metadata_broad_sample", PA_CLUSTERS)
    pc = align_tables(pc_tables, "Metadata_target", PC_CLUSTERS)
    return records, pa, pc

def bootstrap(pa: pd.DataFrame, pc: pd.DataFrame, replicates: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [c for c in pa.columns if c != "Metadata_broad_sample"]
    if columns != [c for c in pc.columns if c != "Metadata_target"]:
        raise AnalysisError("PA/PC variant columns differ")
    pa_values = pa[columns].to_numpy(float)
    pc_values = pc[columns].to_numpy(float)
    if not np.isfinite(pa_values).all() or not np.isfinite(pc_values).all():
        raise AnalysisError("non-finite per-unit values")
    if replicates < 100:
        raise AnalysisError("at least 100 bootstrap replicates required")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    pa_boot = np.empty((replicates, len(columns)))
    pc_boot = np.empty_like(pa_boot)
    for start in range(0, replicates, 250):
        size = min(250, replicates - start)
        pa_w = rng.multinomial(len(pa_values), np.full(len(pa_values), 1 / len(pa_values)), size=size)
        pc_w = rng.multinomial(len(pc_values), np.full(len(pc_values), 1 / len(pc_values)), size=size)
        pa_boot[start : start + size] = pa_w @ pa_values / len(pa_values)
        pc_boot[start : start + size] = pc_w @ pc_values / len(pc_values)
    prod_boot = pa_boot * pc_boot
    pa_points = pa_values.mean(axis=0)
    pc_points = pc_values.mean(axis=0)
    prod_points = pa_points * pc_points
    scores: list[dict[str, Any]] = []
    for i, column in enumerate(columns):
        family, codec = column.split("__")
        row: dict[str, Any] = {"family": family, "display_family": DISPLAY[family], "codec": codec, "replicates": replicates, "seed": seed}
        for metric, point, samples in (
            ("pa", pa_points[i], pa_boot[:, i]),
            ("pc", pc_points[i], pc_boot[:, i]),
            ("product", prod_points[i], prod_boot[:, i]),
        ):
            low, high = interval(samples)
            row.update({f"{metric}_point": point, f"{metric}_ci_low": low, f"{metric}_ci_high": high})
        scores.append(row)
    contrasts: list[dict[str, Any]] = []
    for family in FAMILIES:
        d2 = columns.index(f"{family}__D2-E8")
        mq = columns.index(f"{family}__MQ")
        row = {"family": family, "display_family": DISPLAY[family], "replicates": replicates, "seed": seed}
        for metric, points, samples in (
            ("pa", pa_points, pa_boot),
            ("pc", pc_points, pc_boot),
            ("product", prod_points, prod_boot),
        ):
            diff = samples[:, mq] - samples[:, d2]
            point = float(points[mq] - points[d2])
            low, high = interval(diff)
            row.update(
                {
                    f"{metric}_d2e8": float(points[d2]),
                    f"{metric}_mq": float(points[mq]),
                    f"{metric}_delta_mq_minus_d2e8": point,
                    f"{metric}_delta_ci_low": low,
                    f"{metric}_delta_ci_high": high,
                    f"{metric}_centered_bootstrap_p": centered_pvalue(diff, point),
                }
            )
        contrasts.append(row)
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame["product_holm_p"] = holm_adjust(contrast_frame.product_centered_bootstrap_p)
    contrast_frame["product_supported_direction"] = np.where(
        contrast_frame.product_holm_p < 0.05,
        np.where(contrast_frame.product_delta_mq_minus_d2e8 > 0, "MQ>D2-E8", "D2-E8>MQ"),
        "unresolved",
    )
    return pd.DataFrame(scores), contrast_frame


def plot_panel(contrasts: pd.DataFrame, output_dir: Path) -> None:
    order = list(DISPLAY.values())
    frame = contrasts.set_index("display_family").loc[order]
    y = np.arange(len(frame))
    points = frame.product_delta_mq_minus_d2e8.to_numpy()
    low = frame.product_delta_ci_low.to_numpy()
    high = frame.product_delta_ci_high.to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    colors = ["#2166ac" if value < 0 else "#b2182b" for value in points]
    for index, (point, lo, hi, color) in enumerate(zip(points, low, high, colors, strict=True)):
        ax.errorbar(point, index, xerr=[[point - lo], [hi - point]], fmt="none", ecolor=color, elinewidth=2, capsize=4)
    ax.scatter(points, y, c=colors, s=52, edgecolor="black", linewidth=0.5, zorder=3)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_yticks(y, frame.index)
    ax.invert_yaxis()
    ax.set_xlabel("MQ − D2-E8 fixed-recipe NAP product")
    ax.grid(axis="x", color="#dddddd", linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    metadata = {"Creator": "JUMP-lite fixed_recipe_bootstrap/analyze.py", "CreationDate": datetime(2026, 8, 18, tzinfo=UTC)}
    fig.savefig(output_dir / "fixed_recipe_bootstrap.pdf", bbox_inches="tight", metadata=metadata)
    fig.savefig(output_dir / "fixed_recipe_bootstrap.png", dpi=300, bbox_inches="tight", metadata={"Software": "JUMP-lite fixed_recipe_bootstrap/analyze.py"})
    plt.close(fig)


def report(records: list[dict[str, Any]], contrasts: pd.DataFrame, output_dir: Path) -> None:
    recipe_lines = []
    for family in FAMILIES:
        config = next(row["config"] for row in records if row["family"] == family)
        recipe_lines.append(f"- {DISPLAY[family]}: `{config}`")
    table = [
        "| Family | MQ-D2-E8 PA NAP | MQ-D2-E8 PC NAP | MQ-D2-E8 product (95% interval) | Holm result |",
        "|---|---:|---:|---:|---|",
    ]
    for row in contrasts.itertuples(index=False):
        table.append(
            f"| {row.display_family} | {row.pa_delta_mq_minus_d2e8:+.5f} | {row.pc_delta_mq_minus_d2e8:+.5f} | "
            f"{row.product_delta_mq_minus_d2e8:+.5f} [{row.product_delta_ci_low:+.5f}, {row.product_delta_ci_high:+.5f}] | "
            f"{row.product_supported_direction} (p_Holm={row.product_holm_p:.4g}) |"
        )
    cp_coverage = {
        row["codec"]: (row["pa_query_rows_original"], row["pa_query_rows_common"], row["pa_query_rows_dropped"])
        for row in records
        if row["family"] == "cp_measure"
    }
    coverage_line = ", ".join(
        f"{codec} {original}/{common}/{dropped}"
        for codec, (original, common, dropped) in cp_coverage.items()
    )
    content = "\n".join(
        [
            "# Fixed-recipe Target-2 MQ versus D2-E8 bootstrap",
            "",
            "## Design",
            "",
            "One recipe per Figure-3c family was selected using Zstd PA×PC/100 only, with lexical exact-tie resolution, then fixed across D2-E8 and MQ. Within each family, PA query rows were restricted to the exact Metadata_id intersection across Zstd, D2-E8, and MQ before aggregation to all 306 broad-sample clusters. The 306 PA clusters and 201 PC target clusters were then resampled independently with shared weights across every family/codec within each margin for 50,000 deterministic PCG64DXSM replicates.",
            "",
            f"cp_measure PA query original/common/dropped counts are: {coverage_line}. All learned-family variants retain 1,280/1,280/0 rows. Query-to-broad-sample mappings are required to be identical across codecs on the common rows.",
            "",
            "The product is a working product-of-margins model. Its interval and test omit unknown PA–PC covariance and are conditional on archived normalized outputs and the selected recipes; they are not end-to-end uncertainty.",
            "",
            "## Selected recipes",
            "",
            *recipe_lines,
            "",
            "## Results",
            "",
            *table,
            "",
            "Pointwise intervals are percentile intervals. Product p-values use a centered bootstrap and Holm adjustment across the five predeclared family contrasts. Non-support is not equivalence. Results do not establish denoising or biological improvement.",
            "",
            "![Fixed-recipe bootstrap](fixed_recipe_bootstrap.png)",
            "",
        ]
    )
    atomic_text(output_dir / "REPORT.md", content)


def write_checksums(output_dir: Path) -> None:
    files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name not in {"artifact_checksums.json"}
    )
    payload = {
        "root": ".",
        "artifacts": [
            {"path": path.relative_to(output_dir).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in files
        ],
    }
    atomic_json(output_dir / "artifact_checksums.json", payload)


def verify_release(output_dir: Path) -> None:
    checksums = output_dir / "artifact_checksums.json"
    provenance_path = output_dir / "provenance.json"
    if not checksums.is_file() or not provenance_path.is_file():
        raise AnalysisError("release checksums/provenance missing")
    payload = json.loads(checksums.read_text())
    listed = {record["path"] for record in payload["artifacts"]}
    actual = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "artifact_checksums.json"
    }
    if listed != actual:
        raise AnalysisError(f"release inventory drift: missing={sorted(listed - actual)}, extra={sorted(actual - listed)}")
    for record in payload["artifacts"]:
        path = output_dir / record["path"]
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise AnalysisError(f"release artifact drift: {path}")
    provenance = json.loads(provenance_path.read_text())
    sweep_record = file_record(SWEEP_CSV)
    if sweep_record["sha256"] != provenance["sweep_input"]["sha256"] or sweep_record["size_bytes"] != provenance["sweep_input"]["size_bytes"]:
        raise AnalysisError("sweep input drift from release provenance")
    if provenance.get("protocol_version") != 2:
        raise AnalysisError("unexpected release protocol version")
    selected = provenance["selected_inputs"]
    if len(selected) != len(FAMILIES) * len(CODECS):
        raise AnalysisError("selected input manifest count drift")
    for record in selected:
        expected_common = EXPECTED_COMMON_PA_QUERY_ROWS[record["family"]]
        if record["pa_query_rows_common"] != expected_common or record["pa_query_rows_original"] - record["pa_query_rows_dropped"] != expected_common:
            raise AnalysisError("PA query coverage provenance drift")
        for name in ("metrics", "pa_queries", "pa_map", "pc", "config", "output"):
            path = Path(record[f"{name}_path"])
            if sha256_file(path) != record[f"{name}_sha256"] or path.stat().st_size != record[f"{name}_size_bytes"]:
                raise AnalysisError(f"selected input drift: {path}")


def build_release(output_dir: Path, replicates: int, seed: int) -> None:
    sweep = validate_sweep()
    records, pa, pc = validate_inputs(sweep)
    scores, contrasts = bootstrap(pa, pc, replicates, seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_csv(output_dir / "manifests/selected_recipes.csv", pd.DataFrame(records))
    atomic_csv(output_dir / "results/score_intervals.csv", scores)
    atomic_csv(output_dir / "results/mq_vs_d2e8.csv", contrasts)
    plot_panel(contrasts, output_dir)
    report(records, contrasts, output_dir)
    provenance = {
        "protocol_version": 2,
        "created_utc": CREATED_UTC,
        "analysis": "fixed-recipe paired Target-2 MQ versus D2-E8 cluster bootstrap",
        "seed": seed,
        "replicates": replicates,
        "pa_clusters": PA_CLUSTERS,
        "pc_clusters": PC_CLUSTERS,
        "families": list(FAMILIES),
        "sweep_input": file_record(SWEEP_CSV),
        "selected_inputs": records,
        "selection": "Zstd PA*PC/100, lexical exact-tie resolution",
        "pa_query_alignment": "Exact within-family Metadata_id intersection before aggregation; codec-invariant Metadata_id-to-Metadata_broad_sample mapping required; all 306 broad-sample clusters retained.",
        "inference": "PA and PC margins independently resampled; common weights across family/codecs within margin; percentile intervals; centered bootstrap; Holm over five product contrasts",
        "qualification": "Conditional working product-of-margins inference omits unknown PA-PC covariance; no denoising or biological-improvement interpretation.",
    }
    atomic_json(output_dir / "provenance.json", provenance)
    write_checksums(output_dir)
    verify_release(output_dir)


def publish_release(output_dir: Path, builder: Callable[[Path], None]) -> None:
    """Build and verify in a fresh sibling directory before replacing release."""
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staged.", dir=output_dir.parent))
    # build_release requires its destination not to exist.
    staged.rmdir()
    backup = output_dir.parent / f".{output_dir.name}.backup"
    if backup.exists():
        raise AnalysisError(f"stale release backup exists: {backup}")
    try:
        builder(staged)
        verify_release(staged)
        if output_dir.exists():
            os.replace(output_dir, backup)
        try:
            os.replace(staged, output_dir)
        except Exception:
            if backup.exists() and not output_dir.exists():
                os.replace(backup, output_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if staged.exists():
            shutil.rmtree(staged)


def run(output_dir: Path, replicates: int, seed: int) -> None:
    publish_release(output_dir, lambda staged: build_release(staged, replicates, seed))
    verify_release(output_dir)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/release_v1")
    parser.add_argument("--replicates", type=int, default=N_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify_release(args.output_dir)
        print(f"Verified release: {args.output_dir}")
    else:
        run(args.output_dir, args.replicates, args.seed)
        print(f"Wrote and verified release: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
