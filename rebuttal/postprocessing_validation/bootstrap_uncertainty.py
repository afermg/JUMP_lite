#!/usr/bin/env python3
"""Paired cluster-bootstrap uncertainty for fixed-recipe held-out PA/PC scores.

The analysis is conditional on the saved per-unit PA treatment and PC target
retrieval results. It does not propagate treatment resampling through PC
retrieval or refit normalization/selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

METHOD_VERSION = 1
DEFAULT_REPLICATES = 50_000
DEFAULT_SEED = 20_260_812
DEFAULT_BATCH_SIZE = 100
CONFIDENCE_LEVEL = 0.95
ALPHA = 1.0 - CONFIDENCE_LEVEL
DIAGNOSTIC_PREFIXES = (10_000, 25_000)
CODEC_ORDER = ("Raw", "HQ", "MQ", "D20")
FAMILY_ORDER = (
    "cell_count",
    "cellprofiler",
    "dinov2",
    "dinov2_random",
    "morphem",
    "openphenom",
    "subcell",
)
LEARNED_FAMILIES = ("morphem", "dinov2", "subcell", "openphenom")
DISPLAY_NAMES = {
    "cell_count": "CellCount",
    "cellprofiler": "CellProfiler",
    "dinov2": "DINOv2",
    "dinov2_random": "ViT-rand",
    "morphem": "MorphEM",
    "openphenom": "OpenPhenom",
    "subcell": "SubCell",
}
CONDITIONAL_LABEL = (
    "Paired cluster-bootstrap uncertainty over the observed held-out PA treatment "
    "and PC target distributions, conditional on frozen retrieval calculations, "
    "target eligibility, selected recipes, and normalized profiles."
)
PA_KEYS = ("Metadata_JCP2022", "Metadata_Group")
PC_KEYS = ("Metadata_target", "Metadata_Group")
VALUE_COLUMN = "mean_normalized_average_precision"
FILE_RE = re.compile(
    r"^(?P<family>.+)__(?P<codec>Raw|HQ|MQ|D20)__(?P<metric>pa_treatments|pc_targets)\.csv$"
)


class BootstrapError(RuntimeError):
    """Fail-closed input or analysis error."""


@dataclass(frozen=True)
class AlignedMetric:
    metric: str
    keys: pd.DataFrame
    variants: tuple[tuple[str, str], ...]
    values: np.ndarray
    configs: tuple[str, ...]


@dataclass(frozen=True)
class ClusterMetric:
    metric: str
    cluster_ids: tuple[str, ...]
    strata: np.ndarray
    row_counts: np.ndarray
    value_sums: np.ndarray
    n_rows: int
    observed_memberships: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=here / "results",
        help="Directory containing heldout_test_scores.csv and per_unit/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: RESULTS_DIR/uncertainty).",
    )
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--diagnostic-prefixes",
        type=int,
        nargs="*",
        default=list(DIAGNOSTIC_PREFIXES),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_write_text(path, frame.to_csv(index=False, lineterminator="\n"))


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def family_sort_key(family: str) -> tuple[int, str]:
    try:
        return FAMILY_ORDER.index(family), family
    except ValueError:
        return len(FAMILY_ORDER), family


def variant_sort_key(variant: tuple[str, str]) -> tuple[int, int, str, str]:
    family, codec = variant
    return (
        family_sort_key(family)[0],
        CODEC_ORDER.index(codec) if codec in CODEC_ORDER else len(CODEC_ORDER),
        family,
        codec,
    )


def load_score_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise BootstrapError(f"missing held-out score table: {path}")
    frame = pd.read_csv(path)
    required = {
        "family",
        "codec",
        "config",
        "status",
        "test_pa_mean_nap",
        "test_pc_mean_nap",
        "test_balanced_nap_product",
    }
    if missing := required.difference(frame.columns):
        raise BootstrapError(f"score table lacks columns: {sorted(missing)}")
    if frame.duplicated(["family", "codec"]).any():
        raise BootstrapError("score table has duplicate family/codec rows")
    if not (frame["status"] == "ok").all():
        raise BootstrapError("score table contains non-ok rows")
    numeric = frame[
        ["test_pa_mean_nap", "test_pc_mean_nap", "test_balanced_nap_product"]
    ].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise BootstrapError("score table contains non-finite metrics")
    return frame


def discover_metric_files(per_unit_dir: Path) -> dict[str, dict[tuple[str, str], Path]]:
    found: dict[str, dict[tuple[str, str], Path]] = {
        "pa_treatments": {},
        "pc_targets": {},
    }
    if not per_unit_dir.is_dir():
        raise BootstrapError(f"missing per-unit directory: {per_unit_dir}")
    for path in sorted(per_unit_dir.glob("*.csv")):
        match = FILE_RE.match(path.name)
        if match is None:
            raise BootstrapError(f"unexpected per-unit CSV filename: {path.name}")
        metric = match.group("metric")
        variant = (match.group("family"), match.group("codec"))
        if variant in found[metric]:
            raise BootstrapError(f"duplicate {metric} file for {variant}")
        found[metric][variant] = path
    if not found["pa_treatments"] or not found["pc_targets"]:
        raise BootstrapError("no PA or PC per-unit files discovered")
    if set(found["pa_treatments"]) != set(found["pc_targets"]):
        raise BootstrapError("PA and PC family/codec sets differ")
    return found


def align_metric_tables(
    metric: str,
    paths: dict[tuple[str, str], Path],
    variants: tuple[tuple[str, str], ...],
) -> AlignedMetric:
    keys = PA_KEYS if metric == "pa_treatments" else PC_KEYS
    canonical_index: pd.MultiIndex | None = None
    canonical_keys: pd.DataFrame | None = None
    columns: list[np.ndarray] = []
    configs: list[str] = []
    for variant in variants:
        path = paths[variant]
        frame = pd.read_csv(path, dtype={keys[0]: "string", keys[1]: "string"})
        required = {"family", "codec", "config", VALUE_COLUMN, *keys}
        if missing := required.difference(frame.columns):
            raise BootstrapError(f"{path} lacks columns: {sorted(missing)}")
        if frame.empty:
            raise BootstrapError(f"empty per-unit file: {path}")
        observed_variants = set(zip(frame["family"], frame["codec"]))
        if observed_variants != {variant}:
            raise BootstrapError(
                f"filename/content family-codec mismatch in {path}: {observed_variants}"
            )
        config_values = frame["config"].drop_duplicates().tolist()
        if len(config_values) != 1:
            raise BootstrapError(f"expected one config in {path}: {config_values}")
        if frame[list(keys)].isnull().any().any():
            raise BootstrapError(f"null biological key in {path}")
        if frame.duplicated(list(keys)).any():
            raise BootstrapError(f"duplicate biological key in {path}")
        values = pd.to_numeric(frame[VALUE_COLUMN], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise BootstrapError(f"non-finite {VALUE_COLUMN} in {path}")
        indexed = frame.set_index(list(keys)).sort_index()
        if canonical_index is None:
            canonical_index = indexed.index
            canonical_keys = indexed.index.to_frame(index=False)
        elif not indexed.index.equals(canonical_index):
            missing_keys = canonical_index.difference(indexed.index)
            extra_keys = indexed.index.difference(canonical_index)
            raise BootstrapError(
                f"key-set mismatch in {path}: missing={len(missing_keys)}, extra={len(extra_keys)}"
            )
        columns.append(indexed[VALUE_COLUMN].to_numpy(float))
        configs.append(str(config_values[0]))
    assert canonical_keys is not None
    matrix = np.column_stack(columns)
    return AlignedMetric(
        metric=metric,
        keys=canonical_keys,
        variants=variants,
        values=matrix,
        configs=tuple(configs),
    )


def complete_membership(groups: Iterable[str]) -> str:
    return "|".join(sorted(set(groups)))


def build_cluster_metric(
    aligned: AlignedMetric,
    split: pd.DataFrame | None = None,
) -> ClusterMetric:
    id_column = PA_KEYS[0] if aligned.metric == "pa_treatments" else PC_KEYS[0]
    group_column = "Metadata_Group"
    cluster_ids = tuple(sorted(aligned.keys[id_column].astype(str).unique()))
    cluster_index = {cluster_id: i for i, cluster_id in enumerate(cluster_ids)}
    cluster_codes = aligned.keys[id_column].astype(str).map(cluster_index).to_numpy(int)
    row_counts = np.bincount(cluster_codes, minlength=len(cluster_ids)).astype(np.int64)
    value_sums = np.zeros((len(cluster_ids), aligned.values.shape[1]), dtype=np.float64)
    np.add.at(value_sums, cluster_codes, aligned.values)

    observed = (
        aligned.keys.groupby(id_column, sort=True)[group_column]
        .agg(complete_membership)
        .reindex(cluster_ids)
    )
    if observed.isnull().any():
        raise BootstrapError(f"failed to derive {aligned.metric} cluster memberships")
    observed_memberships = tuple(observed.astype(str))

    if aligned.metric == "pa_treatments":
        if split is None:
            raise BootstrapError("PA clustering requires treatment_split.csv")
        required = {"treatment_id", "stratum", "split"}
        if missing := required.difference(split.columns):
            raise BootstrapError(f"treatment split lacks columns: {sorted(missing)}")
        if split["treatment_id"].duplicated().any():
            raise BootstrapError("treatment split has duplicate treatment IDs")
        split_indexed = split.set_index("treatment_id")
        missing_ids = set(cluster_ids).difference(split_indexed.index.astype(str))
        if missing_ids:
            raise BootstrapError(f"PA clusters absent from treatment split: {len(missing_ids)}")
        selected = split_indexed.reindex(cluster_ids)
        if not (selected["split"] == "test").all():
            raise BootstrapError("saved PA clusters include non-test treatments")
        strata = selected["stratum"].astype(str).to_numpy()
        for cluster_id, observed_membership, stratum in zip(
            cluster_ids, observed_memberships, strata, strict=True
        ):
            if not set(observed_membership.split("|")).issubset(set(stratum.split("|"))):
                raise BootstrapError(
                    f"PA observed membership exceeds split stratum for {cluster_id}: "
                    f"observed={observed_membership}, stratum={stratum}"
                )
    else:
        strata = np.asarray(observed_memberships, dtype=object)

    return ClusterMetric(
        metric=aligned.metric,
        cluster_ids=cluster_ids,
        strata=np.asarray(strata, dtype=object),
        row_counts=row_counts,
        value_sums=value_sums,
        n_rows=len(aligned.keys),
        observed_memberships=observed_memberships,
    )


def bootstrap_component(
    metric: ClusterMetric,
    replicates: int,
    rng: np.random.Generator,
    batch_size: int,
) -> np.ndarray:
    if replicates <= 0:
        raise BootstrapError("replicates must be positive")
    if batch_size <= 0:
        raise BootstrapError("batch size must be positive")
    output = np.empty((replicates, metric.value_sums.shape[1]), dtype=np.float64)
    unique_strata = sorted(set(metric.strata.tolist()))
    positions = {
        stratum: np.flatnonzero(metric.strata == stratum) for stratum in unique_strata
    }
    for start in range(0, replicates, batch_size):
        batch = min(batch_size, replicates - start)
        numerator = np.zeros((batch, metric.value_sums.shape[1]), dtype=np.float64)
        denominator = np.zeros(batch, dtype=np.float64)
        for stratum in unique_strata:
            pos = positions[stratum]
            if not len(pos):
                raise BootstrapError(f"empty bootstrap stratum: {stratum}")
            weights = rng.multinomial(
                len(pos), np.full(len(pos), 1.0 / len(pos)), size=batch
            )
            numerator += weights @ metric.value_sums[pos]
            denominator += weights @ metric.row_counts[pos]
        if (denominator <= 0).any():
            raise BootstrapError("bootstrap produced an empty component sample")
        output[start : start + batch] = numerator / denominator[:, None]
    if not np.isfinite(output).all():
        raise BootstrapError(f"non-finite {metric.metric} bootstrap means")
    return output


def interval(samples: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    low, high = np.quantile(samples, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(low), float(high)


def centered_bootstrap_pvalue(samples: np.ndarray, point: float) -> float:
    centered = samples - point
    exceedances = int(np.count_nonzero(np.abs(centered) >= abs(point)))
    return (exceedances + 1.0) / (len(samples) + 1.0)


def holm_adjust(pvalues: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(pvalues), dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise BootstrapError("invalid p-values for Holm adjustment")
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate((len(p) - np.arange(len(p))) * ranked)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def validate_points(
    scores: pd.DataFrame,
    variants: tuple[tuple[str, str], ...],
    pa_points: np.ndarray,
    pc_points: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    product_points = pa_points * pc_points
    errors = {"pa": 0.0, "pc": 0.0, "product": 0.0}
    for index, variant in enumerate(variants):
        row = scores.loc[
            (scores["family"] == variant[0]) & (scores["codec"] == variant[1])
        ]
        if len(row) != 1:
            raise BootstrapError(f"score table missing or duplicates variant {variant}")
        row = row.iloc[0]
        errors["pa"] = max(
            errors["pa"], abs(float(row["test_pa_mean_nap"]) - pa_points[index])
        )
        errors["pc"] = max(
            errors["pc"], abs(float(row["test_pc_mean_nap"]) - pc_points[index])
        )
        errors["product"] = max(
            errors["product"],
            abs(float(row["test_balanced_nap_product"]) - product_points[index]),
        )
    if max(errors.values()) > 1e-12:
        raise BootstrapError(f"per-unit means do not reproduce held-out scores: {errors}")
    return product_points, errors


def summary_table(
    scores: pd.DataFrame,
    variants: tuple[tuple[str, str], ...],
    pa_points: np.ndarray,
    pc_points: np.ndarray,
    product_points: np.ndarray,
    pa_boot: np.ndarray,
    pc_boot: np.ndarray,
    product_boot: np.ndarray,
    pa_clusters: ClusterMetric,
    pc_clusters: ClusterMetric,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, (family, codec) in enumerate(variants):
        score = scores.loc[
            (scores["family"] == family) & (scores["codec"] == codec)
        ].iloc[0]
        pa_low, pa_high = interval(pa_boot[:, index])
        pc_low, pc_high = interval(pc_boot[:, index])
        product_low, product_high = interval(product_boot[:, index])
        rows.append(
            {
                "family": family,
                "display_name": DISPLAY_NAMES.get(family, family),
                "codec": codec,
                "config": score["config"],
                "pa_units": pa_clusters.n_rows,
                "pa_clusters": len(pa_clusters.cluster_ids),
                "pc_units": pc_clusters.n_rows,
                "pc_clusters": len(pc_clusters.cluster_ids),
                "pa_point": pa_points[index],
                "pa_ci_low": pa_low,
                "pa_ci_high": pa_high,
                "pc_point": pc_points[index],
                "pc_ci_low": pc_low,
                "pc_ci_high": pc_high,
                "product_point": product_points[index],
                "product_ci_low": product_low,
                "product_ci_high": product_high,
                "confidence_level": CONFIDENCE_LEVEL,
                "interval_method": "percentile",
                "replicates": replicates,
                "seed": seed,
                "conditional_inference": CONDITIONAL_LABEL,
            }
        )
    return pd.DataFrame(rows)


def codec_comparisons(
    variants: tuple[tuple[str, str], ...],
    pa_points: np.ndarray,
    pc_points: np.ndarray,
    product_points: np.ndarray,
    pa_boot: np.ndarray,
    pc_boot: np.ndarray,
    product_boot: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, tuple[float, np.ndarray]]]:
    index = {variant: i for i, variant in enumerate(variants)}
    families = sorted(
        {
            family
            for family, _ in variants
            if all((family, codec) in index for codec in CODEC_ORDER)
        },
        key=family_sort_key,
    )
    rows: list[dict[str, Any]] = []
    samples_by_name: dict[str, tuple[float, np.ndarray]] = {}
    for family in families:
        raw_i = index[(family, "Raw")]
        for codec in CODEC_ORDER[1:]:
            codec_i = index[(family, codec)]
            row: dict[str, Any] = {
                "family": family,
                "display_name": DISPLAY_NAMES.get(family, family),
                "codec": codec,
                "raw_codec": "Raw",
                "primary_learned_model": family in LEARNED_FAMILIES,
            }
            for name, points, boot in (
                ("pa", pa_points, pa_boot),
                ("pc", pc_points, pc_boot),
                ("product", product_points, product_boot),
            ):
                diff = boot[:, codec_i] - boot[:, raw_i]
                relative = 100.0 * (boot[:, codec_i] / boot[:, raw_i] - 1.0)
                diff_low, diff_high = interval(diff)
                denominator_crossing = float(np.mean(boot[:, raw_i] <= 0))
                row.update(
                    {
                        f"{name}_raw_point": points[raw_i],
                        f"{name}_codec_point": points[codec_i],
                        f"{name}_delta_point": points[codec_i] - points[raw_i],
                        f"{name}_delta_ci_low": diff_low,
                        f"{name}_delta_ci_high": diff_high,
                        f"{name}_raw_nonpositive_fraction": denominator_crossing,
                    }
                )
                if denominator_crossing == 0.0 and np.isfinite(relative).all():
                    relative_low, relative_high = interval(relative)
                    row.update(
                        {
                            f"{name}_relative_pct_point": 100.0
                            * (points[codec_i] / points[raw_i] - 1.0),
                            f"{name}_relative_pct_ci_low": relative_low,
                            f"{name}_relative_pct_ci_high": relative_high,
                            f"{name}_relative_pct_suppressed": False,
                        }
                    )
                else:
                    row.update(
                        {
                            f"{name}_relative_pct_point": math.nan,
                            f"{name}_relative_pct_ci_low": math.nan,
                            f"{name}_relative_pct_ci_high": math.nan,
                            f"{name}_relative_pct_suppressed": True,
                        }
                    )
                if name == "product":
                    row["product_centered_bootstrap_p"] = centered_bootstrap_pvalue(
                        diff, float(row["product_delta_point"])
                    )
                    samples_by_name[f"codec:{family}:{codec}-Raw"] = (
                        float(row["product_delta_point"]),
                        diff,
                    )
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame["product_holm_p"] = holm_adjust(frame["product_centered_bootstrap_p"])
    frame["product_supported_direction"] = np.where(
        frame["product_holm_p"] < ALPHA,
        np.where(frame["product_delta_point"] > 0, "increase", "decrease"),
        "unresolved",
    )
    frame["multiplicity_family"] = "all codec-vs-Raw product comparisons"
    frame["multiplicity_family_size"] = len(frame)
    return frame, samples_by_name


def pairwise_comparisons(
    variants: tuple[tuple[str, str], ...],
    pa_points: np.ndarray,
    pc_points: np.ndarray,
    product_points: np.ndarray,
    pa_boot: np.ndarray,
    pc_boot: np.ndarray,
    product_boot: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, tuple[float, np.ndarray]]]:
    index = {variant: i for i, variant in enumerate(variants)}
    rows: list[dict[str, Any]] = []
    samples_by_name: dict[str, tuple[float, np.ndarray]] = {}
    for codec in CODEC_ORDER:
        families = sorted(
            [family for family, candidate_codec in variants if candidate_codec == codec],
            key=family_sort_key,
        )
        for family_a, family_b in combinations(families, 2):
            a = index[(family_a, codec)]
            b = index[(family_b, codec)]
            row: dict[str, Any] = {
                "codec": codec,
                "family_a": family_a,
                "display_name_a": DISPLAY_NAMES.get(family_a, family_a),
                "family_b": family_b,
                "display_name_b": DISPLAY_NAMES.get(family_b, family_b),
                "primary_learned_pair": family_a in LEARNED_FAMILIES
                and family_b in LEARNED_FAMILIES,
            }
            for name, points, boot in (
                ("pa", pa_points, pa_boot),
                ("pc", pc_points, pc_boot),
                ("product", product_points, product_boot),
            ):
                diff = boot[:, a] - boot[:, b]
                low, high = interval(diff)
                row.update(
                    {
                        f"{name}_a_point": points[a],
                        f"{name}_b_point": points[b],
                        f"{name}_difference_a_minus_b": points[a] - points[b],
                        f"{name}_difference_ci_low": low,
                        f"{name}_difference_ci_high": high,
                    }
                )
                if name == "product":
                    point = float(row["product_difference_a_minus_b"])
                    row["product_probability_a_gt_b"] = float(np.mean(diff > 0))
                    row["product_centered_bootstrap_p"] = centered_bootstrap_pvalue(
                        diff, point
                    )
                    samples_by_name[
                        f"pair:{codec}:{family_a}-{family_b}"
                    ] = (point, diff)
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame["product_holm_p"] = holm_adjust(frame["product_centered_bootstrap_p"])
    directions: list[str] = []
    for row in frame.itertuples(index=False):
        if row.product_holm_p >= ALPHA:
            directions.append("unresolved")
        elif row.product_difference_a_minus_b > 0:
            directions.append(f"{row.family_a}>{row.family_b}")
        else:
            directions.append(f"{row.family_b}>{row.family_a}")
    frame["product_supported_direction"] = directions
    frame["multiplicity_family"] = "all same-codec product pairwise comparisons"
    frame["multiplicity_family_size"] = len(frame)
    return frame, samples_by_name


def rank_table(
    variants: tuple[tuple[str, str], ...],
    product_points: np.ndarray,
    product_boot: np.ndarray,
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    index = {variant: i for i, variant in enumerate(variants)}
    rows: list[dict[str, Any]] = []
    for codec in CODEC_ORDER:
        families = sorted(
            [family for family, candidate_codec in variants if candidate_codec == codec],
            key=family_sort_key,
        )
        columns = [index[(family, codec)] for family in families]
        points = product_points[columns]
        boot = product_boot[:, columns]
        point_order = np.argsort(-points, kind="stable")
        point_ranks = np.empty(len(families), dtype=int)
        point_ranks[point_order] = np.arange(1, len(families) + 1)
        boot_order = np.argsort(-boot, axis=1, kind="stable")
        boot_ranks = np.empty_like(boot_order)
        np.put_along_axis(
            boot_ranks,
            boot_order,
            np.broadcast_to(np.arange(1, len(families) + 1), boot_order.shape),
            axis=1,
        )
        relevant = pairwise.loc[pairwise["codec"] == codec]
        for local_index, family in enumerate(families):
            superiors: set[str] = set()
            inferiors: set[str] = set()
            for comparison in relevant.itertuples(index=False):
                if comparison.product_supported_direction == "unresolved":
                    continue
                winner, loser = comparison.product_supported_direction.split(">")
                if loser == family:
                    superiors.add(winner)
                if winner == family:
                    inferiors.add(loser)
            ranks = boot_ranks[:, local_index]
            rows.append(
                {
                    "codec": codec,
                    "family": family,
                    "display_name": DISPLAY_NAMES.get(family, family),
                    "product_point": points[local_index],
                    "point_rank": int(point_ranks[local_index]),
                    "bootstrap_rank_mean": float(ranks.mean()),
                    "bootstrap_rank_median": float(np.median(ranks)),
                    "bootstrap_rank_ci_low": float(np.quantile(ranks, ALPHA / 2.0)),
                    "bootstrap_rank_ci_high": float(
                        np.quantile(ranks, 1.0 - ALPHA / 2.0)
                    ),
                    "bootstrap_probability_best": float(np.mean(ranks == 1)),
                    "simultaneous_rank_lower": 1 + len(superiors),
                    "simultaneous_rank_upper": len(families) - len(inferiors),
                    "significantly_superior_models": "|".join(sorted(superiors)),
                    "significantly_inferior_models": "|".join(sorted(inferiors)),
                    "non_equivalence_warning": (
                        "Shared/overlapping rank bounds do not establish equivalence."
                    ),
                }
            )
    return pd.DataFrame(rows)


def audit_table(
    pa: ClusterMetric,
    pc: ClusterMetric,
    split: pd.DataFrame,
    point_errors: dict[str, float],
    n_variants: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split_test = split.loc[split["split"] == "test"]
    for metric in (pa, pc):
        for stratum in sorted(set(metric.strata.tolist())):
            mask = metric.strata == stratum
            assigned = (
                int((split_test["stratum"] == stratum).sum())
                if metric.metric == "pa_treatments"
                else math.nan
            )
            n_clusters = int(mask.sum())
            rows.append(
                {
                    "metric": metric.metric,
                    "stratum": stratum,
                    "n_units": int(metric.row_counts[mask].sum()),
                    "n_clusters": n_clusters,
                    "assigned_test_clusters": assigned,
                    "missing_assigned_clusters": (
                        assigned - n_clusters if metric.metric == "pa_treatments" else math.nan
                    ),
                    "n_family_codec_tables": n_variants,
                    "key_sets_identical": True,
                    "point_estimate_max_abs_error": max(point_errors.values()),
                }
            )
    return pd.DataFrame(rows)


def diagnostic_table(
    product_points: np.ndarray,
    product_boot: np.ndarray,
    variants: tuple[tuple[str, str], ...],
    comparison_samples: dict[str, tuple[float, np.ndarray]],
    prefixes: tuple[int, ...],
) -> pd.DataFrame:
    samples: dict[str, tuple[float, np.ndarray, float]] = {}
    for index, (family, codec) in enumerate(variants):
        samples[f"score:{family}:{codec}"] = (
            product_points[index],
            product_boot[:, index],
            math.nan,
        )
    for name, (point, values) in comparison_samples.items():
        samples[name] = (point, values, math.nan)
    rows: list[dict[str, Any]] = []
    for name, (point, values, denominator_crossing) in samples.items():
        final_low, final_high = interval(values)
        final_width = max(final_high - final_low, np.finfo(float).eps)
        row: dict[str, Any] = {
            "estimand": name,
            "point": point,
            "final_replicates": len(values),
            "final_ci_low": final_low,
            "final_ci_high": final_high,
            "denominator_nonpositive_fraction": denominator_crossing,
        }
        worst_drift = 0.0
        for prefix in prefixes:
            low, high = interval(values[:prefix])
            drift = max(abs(low - final_low), abs(high - final_high)) / final_width
            row[f"ci_low_{prefix}"] = low
            row[f"ci_high_{prefix}"] = high
            row[f"max_endpoint_drift_fraction_{prefix}"] = drift
            worst_drift = max(worst_drift, drift)
        row["worst_endpoint_drift_fraction"] = worst_drift
        row["monte_carlo_warning"] = worst_drift > 0.02
        rows.append(row)
    return pd.DataFrame(rows)


def render_report(
    summary: pd.DataFrame,
    codec: pd.DataFrame,
    pairwise: pd.DataFrame,
    ranks: pd.DataFrame,
    diagnostics: pd.DataFrame,
    replicates: int,
    seed: int,
    runtime_seconds: float,
) -> str:
    learned_summary = summary.loc[summary["family"].isin(LEARNED_FAMILIES)]
    learned_codec = codec.loc[codec["primary_learned_model"]]
    learned_pairs = pairwise.loc[pairwise["primary_learned_pair"]]
    lines = [
        "# Held-out paired cluster-bootstrap uncertainty",
        "",
        f"Generated from {replicates:,} deterministic resamples (seed `{seed}`) in "
        f"{runtime_seconds:.3f} seconds.",
        "",
        "## Scope and method",
        "",
        CONDITIONAL_LABEL,
        "",
        "PA treatment IDs were resampled with replacement within the frozen composite "
        "split strata, retaining all evaluable group rows for each sampled treatment. "
        "PC targets were resampled within their complete observed group-membership "
        "strata, retaining both high/low rows when present. The same cluster weights "
        "were applied to every model and codec, so contrasts are paired within each "
        "margin. PA and PC margins were independently resampled and multiplied within "
        "each replicate under a working product-of-margins model.",
        "",
        "All intervals below are pointwise conditional percentile-bootstrap intervals. "
        "The independent PA and PC streams omit their unknown covariance, so intervals "
        "and centered-bootstrap p-values may be either too narrow or too wide; they are "
        "not full end-to-end or unconditional sampling intervals.",
        "",
        "Displayed 95% intervals are not multiplicity-adjusted. Centered-bootstrap "
        "product tests were adjusted in two separate Holm families: all 15 codec-vs-Raw "
        "comparisons and all 51 same-codec model comparisons. No single correction "
        "across all 66 tests was applied.",
        "",
        "## Learned-model score intervals",
        "",
        "| Model | Codec | Product | Pointwise 95% interval |",
        "|---|---|---:|---:|",
    ]
    for codec_name in CODEC_ORDER:
        for family in LEARNED_FAMILIES:
            row = learned_summary.loc[
                (learned_summary["family"] == family)
                & (learned_summary["codec"] == codec_name)
            ].iloc[0]
            lines.append(
                f"| {DISPLAY_NAMES[family]} | {codec_name} | {row.product_point:.5f} | "
                f"[{row.product_ci_low:.5f}, {row.product_ci_high:.5f}] |"
            )
    lines.extend(
        [
            "",
            "## Codec changes from Raw",
            "",
            "| Model | Codec | Change | Pointwise 95% interval | Relative change (pointwise 95% interval) | Holm result |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for family in LEARNED_FAMILIES:
        for codec_name in CODEC_ORDER[1:]:
            row = learned_codec.loc[
                (learned_codec["family"] == family)
                & (learned_codec["codec"] == codec_name)
            ].iloc[0]
            relative = (
                "suppressed"
                if row.product_relative_pct_suppressed
                else f"{row.product_relative_pct_point:+.1f}% "
                f"[{row.product_relative_pct_ci_low:+.1f}%, "
                f"{row.product_relative_pct_ci_high:+.1f}%]"
            )
            lines.append(
                f"| {DISPLAY_NAMES[family]} | {codec_name} | "
                f"{row.product_delta_point:+.5f} | "
                f"[{row.product_delta_ci_low:+.5f}, {row.product_delta_ci_high:+.5f}] | "
                f"{relative} | {row.product_supported_direction} "
                f"($p_{{Holm}}={row.product_holm_p:.4g}$) |"
            )
    supported_middle = learned_pairs.loc[
        (learned_pairs["product_supported_direction"] != "unresolved")
        & ~learned_pairs["product_supported_direction"].str.startswith("morphem>")
        & ~learned_pairs["product_supported_direction"].str.endswith(">morphem")
    ]
    morph_rows = learned_pairs.loc[
        learned_pairs["family_a"].eq("morphem")
        | learned_pairs["family_b"].eq("morphem")
    ]
    morph_supported = int(
        (
            (morph_rows["product_supported_direction"] != "unresolved")
            & morph_rows["product_supported_direction"].str.startswith("morphem>")
        ).sum()
    )
    lines.extend(
        [
            "",
            "## Pairwise ranking result",
            "",
            f"Within the conditional two-margin bootstrap and the separate 51-comparison "
            f"same-codec Holm family, MorphEM is the highest point estimate at every "
            f"codec and is supported over each other learned model in "
            f"{morph_supported}/{len(morph_rows)} comparisons.",
            "",
        ]
    )
    if supported_middle.empty:
        lines.append(
            "No exact ordering among DINOv2, SubCell, and OpenPhenom is supported in "
            "the separate 51-comparison same-codec Holm family."
        )
    else:
        lines.append(
            "Supported middle-model directions in the separate 51-comparison "
            "same-codec Holm family:"
        )
        for row in supported_middle.itertuples(index=False):
            lines.append(
                f"- {row.codec}: `{row.product_supported_direction}` "
                f"($p_{{Holm}}={row.product_holm_p:.4g}$)."
            )
    raw_middle = ranks.loc[
        (ranks["codec"] == "Raw") & ranks["family"].isin(LEARNED_FAMILIES[1:])
    ]
    lines.extend(
        [
            "",
            "Among all seven Raw model families, simultaneous conditional rank bounds "
            "for the three middle learned models are: "
            + "; ".join(
                f"{DISPLAY_NAMES[row.family]} {row.simultaneous_rank_lower}--"
                f"{row.simultaneous_rank_upper}"
                for row in raw_middle.itertuples(index=False)
            )
            + ". Overlapping bounds do not establish equivalence.",
            "",
            "## Diagnostics",
            "",
            f"As an internal Monte Carlo convergence check, "
            f"{int(diagnostics.monte_carlo_warning.sum())}/{len(diagnostics)} tracked "
            "percentile intervals exceeded a 2% endpoint-drift threshold when the 10k "
            "and 25k nested prefixes were compared with the final run. This heuristic "
            "assesses interval-endpoint stability only; it does not validate the "
            f"resampling assumptions. Final intervals use all {replicates:,} replicates.",
            "",
            "## Limitations",
            "",
            "- The saved summaries cannot preserve dependence between PA treatments and "
            "PC targets; treatment-target incidence and query-level PC contributions are "
            "not present. The working-independence approximation omits PA--PC covariance "
            "and can make uncertainty either too narrow or too wide.",
            "- Target resampling does not remove dependence among targets sharing compounds.",
            "- Recipe selection, alternative treatment splits, normalization fitting, "
            "shared controls, wells/sites, target eligibility, and annotation uncertainty "
            "are not resampled.",
            "- The split is treatment-disjoint, not target-disjoint, and the archived "
            "transforms were fitted before splitting.",
            "- A non-supported difference is not evidence of equivalence.",
            "- CellProfiler is Raw-only and retains the known site-count asymmetry.",
            "",
            "## Outputs",
            "",
            "- `heldout_uncertainty.csv`",
            "- `codec_vs_raw_paired.csv`",
            "- `model_pairwise_by_codec.csv`",
            "- `model_rank_bounds.csv`",
            "- `resampling_unit_audit.csv`",
            "- `bootstrap_diagnostics.csv`",
            "- `provenance.json`",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if args.replicates < 100:
        raise BootstrapError("at least 100 bootstrap replicates are required")
    if args.seed < 0:
        raise BootstrapError("seed must be non-negative")
    prefixes = tuple(sorted(set(int(value) for value in args.diagnostic_prefixes)))
    if any(value <= 0 or value >= args.replicates for value in prefixes):
        raise BootstrapError(
            f"diagnostic prefixes must be positive and below replicates: {prefixes}"
        )

    results_dir = args.results_dir.resolve()
    output_dir = (args.output_dir or (results_dir / "uncertainty")).resolve()
    score_path = results_dir / "heldout_test_scores.csv"
    split_path = results_dir / "treatment_split.csv"
    scores = load_score_table(score_path)
    split = pd.read_csv(split_path, dtype={"treatment_id": "string", "stratum": "string"})
    paths = discover_metric_files(results_dir / "per_unit")
    variants = tuple(sorted(paths["pa_treatments"], key=variant_sort_key))
    score_variants = set(zip(scores["family"], scores["codec"]))
    if set(variants) != score_variants:
        raise BootstrapError(
            f"per-unit and score variant sets differ: per_unit={len(variants)}, "
            f"scores={len(score_variants)}"
        )

    pa_aligned = align_metric_tables("pa_treatments", paths["pa_treatments"], variants)
    pc_aligned = align_metric_tables("pc_targets", paths["pc_targets"], variants)
    pa_clusters = build_cluster_metric(pa_aligned, split)
    pc_clusters = build_cluster_metric(pc_aligned)
    pa_points = pa_aligned.values.mean(axis=0)
    pc_points = pc_aligned.values.mean(axis=0)
    product_points, point_errors = validate_points(
        scores, variants, pa_points, pc_points
    )

    seed_sequence = np.random.SeedSequence(args.seed)
    pa_seed, pc_seed = seed_sequence.spawn(2)
    pa_rng = np.random.Generator(np.random.PCG64DXSM(pa_seed))
    pc_rng = np.random.Generator(np.random.PCG64DXSM(pc_seed))
    pa_boot = bootstrap_component(
        pa_clusters, args.replicates, pa_rng, args.batch_size
    )
    pc_boot = bootstrap_component(
        pc_clusters, args.replicates, pc_rng, args.batch_size
    )
    product_boot = pa_boot * pc_boot

    summary = summary_table(
        scores,
        variants,
        pa_points,
        pc_points,
        product_points,
        pa_boot,
        pc_boot,
        product_boot,
        pa_clusters,
        pc_clusters,
        args.replicates,
        args.seed,
    )
    codec, codec_samples = codec_comparisons(
        variants,
        pa_points,
        pc_points,
        product_points,
        pa_boot,
        pc_boot,
        product_boot,
    )
    pairwise, pairwise_samples = pairwise_comparisons(
        variants,
        pa_points,
        pc_points,
        product_points,
        pa_boot,
        pc_boot,
        product_boot,
    )
    ranks = rank_table(variants, product_points, product_boot, pairwise)
    audit = audit_table(
        pa_clusters, pc_clusters, split, point_errors, len(variants)
    )
    diagnostics = diagnostic_table(
        product_points,
        product_boot,
        variants,
        {**codec_samples, **pairwise_samples},
        prefixes,
    )
    runtime_seconds = time.monotonic() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    output_frames = {
        "heldout_uncertainty.csv": summary,
        "codec_vs_raw_paired.csv": codec,
        "model_pairwise_by_codec.csv": pairwise,
        "model_rank_bounds.csv": ranks,
        "resampling_unit_audit.csv": audit,
        "bootstrap_diagnostics.csv": diagnostics,
    }
    for filename, frame in output_frames.items():
        atomic_write_csv(output_dir / filename, frame)
    report = render_report(
        summary,
        codec,
        pairwise,
        ranks,
        diagnostics,
        args.replicates,
        args.seed,
        runtime_seconds,
    )
    atomic_write_text(output_dir / "REPORT.md", report)

    script_path = Path(__file__).resolve()
    input_paths = [
        score_path,
        split_path,
        *[paths["pa_treatments"][variant] for variant in variants],
        *[paths["pc_targets"][variant] for variant in variants],
    ]
    output_paths = [output_dir / name for name in output_frames] + [
        output_dir / "REPORT.md"
    ]
    pa_strata = {
        stratum: {
            "clusters": int(np.sum(pa_clusters.strata == stratum)),
            "units": int(
                pa_clusters.row_counts[pa_clusters.strata == stratum].sum()
            ),
        }
        for stratum in sorted(set(pa_clusters.strata.tolist()))
    }
    pc_strata = {
        stratum: {
            "clusters": int(np.sum(pc_clusters.strata == stratum)),
            "units": int(
                pc_clusters.row_counts[pc_clusters.strata == stratum].sum()
            ),
        }
        for stratum in sorted(set(pc_clusters.strata.tolist()))
    }
    partial_membership_clusters = sum(
        observed != stratum
        for observed, stratum in zip(
            pa_clusters.observed_memberships, pa_clusters.strata, strict=True
        )
    )
    provenance = {
        "method_version": METHOD_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": runtime_seconds,
        "repository_head": subprocess_git_head(results_dir.parents[2]),
        "script": file_record(script_path),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "results_dir": str(results_dir),
        "output_dir": str(output_dir),
        "replicates": args.replicates,
        "seed": args.seed,
        "bit_generator": "PCG64DXSM",
        "pa_seed_spawn_key": list(pa_seed.spawn_key),
        "pc_seed_spawn_key": list(pc_seed.spawn_key),
        "batch_size": args.batch_size,
        "confidence_level": CONFIDENCE_LEVEL,
        "interval_method": "percentile",
        "diagnostic_prefixes": list(prefixes),
        "conditional_inference": CONDITIONAL_LABEL,
        "pairing": (
            "Identical PA cluster weights and identical PC cluster weights across all "
            "family/codec conditions within each replicate."
        ),
        "component_dependence": (
            "PA and PC margins use independent bootstrap streams under a working "
            "product-of-margins model because saved target summaries do not contain "
            "the treatment-target/query decomposition needed for a joint resample. "
            "Unknown PA-PC covariance may make intervals and tests too narrow or too wide."
        ),
        "dimensions": {
            "variants": len(variants),
            "pa_units": pa_clusters.n_rows,
            "pa_clusters": len(pa_clusters.cluster_ids),
            "pc_units": pc_clusters.n_rows,
            "pc_clusters": len(pc_clusters.cluster_ids),
            "pa_strata": pa_strata,
            "pc_strata": pc_strata,
            "pa_partial_membership_clusters": partial_membership_clusters,
        },
        "variants": [
            {"family": family, "codec": codec} for family, codec in variants
        ],
        "point_reproduction_max_abs_error": point_errors,
        "multiplicity": {
            "codec_vs_raw": {
                "method": "Holm FWER",
                "alpha": ALPHA,
                "comparisons": len(codec),
            },
            "same_codec_models": {
                "method": "Holm FWER",
                "alpha": ALPHA,
                "comparisons": len(pairwise),
            },
        },
        "limitations": [
            "Conditional two-margin uncertainty under a working-independence approximation, not end-to-end sampling uncertainty.",
            "No treatment-target/query decomposition for joint PA-PC resampling.",
            "No recipe-selection, split, transform-fit, target-eligibility, well/site, or control resampling.",
            "Treatment-disjoint rather than target-disjoint evaluation.",
            "A non-supported difference does not establish equivalence.",
        ],
        "inputs": [file_record(path) for path in input_paths],
        "outputs": [file_record(path) for path in output_paths],
    }
    atomic_write_json(output_dir / "provenance.json", provenance)
    print(output_dir)
    print(f"runtime_seconds={runtime_seconds:.3f}")
    print(f"max_point_error={max(point_errors.values()):.3g}")
    print(
        f"diagnostic_warnings={int(diagnostics.monte_carlo_warning.sum())}/"
        f"{len(diagnostics)}"
    )
    return {
        "summary": summary,
        "codec": codec,
        "pairwise": pairwise,
        "ranks": ranks,
        "audit": audit,
        "diagnostics": diagnostics,
        "provenance": provenance,
    }


def subprocess_git_head(repository: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    try:
        run(parse_args())
    except BootstrapError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
