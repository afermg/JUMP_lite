#!/usr/bin/env python3
"""Validation/test analysis for JUMP-lite post-processing configuration selection.

This runner never mutates the canonical sweep.  It selects one post-processing
configuration per model family from Raw validation treatments, then evaluates
that fixed configuration on held-out treatments and a frozen cross-model well
intersection for every available manuscript codec.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd
import polars as pl
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
NORM3_SRC = REPO_ROOT / "src"
if str(NORM3_SRC) not in sys.path:
    sys.path.insert(0, str(NORM3_SRC))

from norm_3.io import get_numeric_features, infer_columns  # noqa: E402
from norm_3.metrics import (  # noqa: E402
    calculate_phenotypic_activity,
    calculate_phenotypic_consistency,
)

DEFAULT_SWEEP = Path(
    "/work/datasets/JUMP-lite-wacv/sweeps/variance_first_v11_lite"
)
COMPOUND_COL = "Metadata_JCP2022"
TARGET_COL = "Metadata_RefChemDB_target"
NEGCON_COL = "Metadata_negcon"
GROUP_COL = "Metadata_Group"
PLATE_COL = "Metadata_Plate"
ID_COL = "Metadata_id"
PC_GROUPS = ("group_high", "group_low")
REQUIRED_ANNOTATION_COLS = (ID_COL, COMPOUND_COL, GROUP_COL, NEGCON_COL)
PROTOCOL_VERSION = 1


class AnalysisError(RuntimeError):
    """Fail-closed analysis error."""


class CoverageError(AnalysisError):
    """The fixed selected configuration is unavailable for a required codec."""


@dataclass(frozen=True)
class FamilySpec:
    key: str
    display: str
    codec_folders: Mapping[str, str]

    @property
    def raw_folder(self) -> str:
        return self.codec_folders["Raw"]


FAMILIES: dict[str, FamilySpec] = {
    "dinov2": FamilySpec(
        "dinov2",
        "DINOv2",
        {
            "Raw": "dinov2_jump_lite_updated_jpegxl_lossy_raw_raw_features",
            "HQ": "dinov2_jump_lite_updated_jpegxl_lossy_hq_raw_features",
            "MQ": "dinov2_jump_lite_updated_jpegxl_lossy_mq_raw_features",
            "D20": "dinov2_jump_lite_updated_jpegxl_lossy_d20_raw_features",
        },
    ),
    "morphem": FamilySpec(
        "morphem",
        "MorphEM",
        {
            "Raw": "morphem_jump_lite_updated_jpegxl_lossy_raw_raw_features",
            "HQ": "morphem_jump_lite_updated_jpegxl_lossy_hq_raw_features",
            "MQ": "morphem_jump_lite_updated_jpegxl_lossy_mq_raw_features",
            "D20": "morphem_jump_lite_updated_jpegxl_lossy_d20_raw_features",
        },
    ),
    "openphenom": FamilySpec(
        "openphenom",
        "OpenPhenom",
        {
            "Raw": "openphenom_jump_lite_updated_jpegxl_lossy_raw_raw_features",
            "HQ": "openphenom_jump_lite_updated_jpegxl_lossy_hq_raw_features",
            "MQ": "openphenom_jump_lite_updated_jpegxl_lossy_mq_raw_features",
            "D20": "openphenom_jump_lite_updated_jpegxl_lossy_d20_raw_features",
        },
    ),
    "subcell": FamilySpec(
        "subcell",
        "SubCell",
        {
            "Raw": "subcell__clip01_jump_lite_updated_jpegxl_lossy_raw_raw_features",
            "HQ": "subcell__clip01_jump_lite_updated_jpegxl_lossy_hq_raw_features",
            "MQ": "subcell__clip01_jump_lite_updated_jpegxl_lossy_mq_raw_features",
            "D20": "subcell__clip01_jump_lite_updated_jpegxl_lossy_d20_raw_features",
        },
    ),
    "cellprofiler": FamilySpec(
        "cellprofiler",
        "CellProfiler",
        {"Raw": "cellprofiler_raw_jump_lite_raw_features"},
    ),
    "dinov2_random": FamilySpec(
        "dinov2_random",
        "ViT-rand",
        {
            "Raw": "dinov2_random_jump_lite_updated_jpegxl_lossy_raw_raw_features",
            "HQ": "dinov2_random_jump_lite_updated_jpegxl_lossy_hq_raw_features",
            "MQ": "dinov2_random_jump_lite_updated_jpegxl_lossy_mq_raw_features",
            "D20": "dinov2_random_jump_lite_updated_jpegxl_lossy_d20_raw_features",
        },
    ),
    "cell_count": FamilySpec(
        "cell_count",
        "CellCount",
        {"Raw": "cell_count_jump_lite_raw_features"},
    ),
}


@dataclass(frozen=True)
class ProfileRef:
    family: str
    codec: str
    config: str
    folder: Path
    output_path: Path
    config_path: Path
    pa_map_path: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise AnalysisError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                extrasaction="ignore",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def atomic_write_pandas_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        raise AnalysisError(f"refusing to write empty result table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        frame.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_checkpoint(path: Path, protocol_hash: str, payload: Mapping[str, Any]) -> None:
    atomic_write_json(
        path,
        {
            "protocol_hash": protocol_hash,
            "completed_at": utc_now(),
            **dict(payload),
        },
    )


def load_checkpoint(path: Path, protocol_hash: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if data.get("protocol_hash") != protocol_hash:
        raise AnalysisError(f"checkpoint protocol mismatch: {path}")
    return data


def stable_assignment_key(seed: int, stratum: str, treatment_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{stratum}\0{treatment_id}".encode()).digest()


def make_treatment_split(
    memberships: Mapping[str, Iterable[str]],
    validation_fraction: float = 0.20,
    seed: int = 20260811,
) -> list[dict[str, Any]]:
    """Assign every treatment ID once, stratified by composite group membership."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    strata: dict[str, list[str]] = {}
    for treatment_id, groups in memberships.items():
        treatment_id = str(treatment_id)
        group_values = sorted({str(group) for group in groups})
        if not treatment_id or treatment_id.lower() == "none":
            raise AnalysisError("empty treatment identifier in split input")
        if not group_values:
            raise AnalysisError(f"treatment {treatment_id} has no group membership")
        stratum = "|".join(group_values)
        strata.setdefault(stratum, []).append(treatment_id)

    rows: list[dict[str, Any]] = []
    for stratum, ids in sorted(strata.items()):
        ordered = sorted(set(ids), key=lambda value: (stable_assignment_key(seed, stratum, value), value))
        n_validation = int(math.floor(len(ordered) * validation_fraction + 0.5))
        validation_ids = set(ordered[:n_validation])
        for treatment_id in sorted(ordered):
            rows.append(
                {
                    "treatment_id": treatment_id,
                    "stratum": stratum,
                    "split": "validation" if treatment_id in validation_ids else "test",
                    "assignment_hash": stable_assignment_key(seed, stratum, treatment_id).hex(),
                }
            )
    if len(rows) != len({row["treatment_id"] for row in rows}):
        raise AnalysisError("a treatment identifier was assigned more than once")
    if not any(row["split"] == "validation" for row in rows):
        raise AnalysisError("split has no validation treatments")
    if not any(row["split"] == "test" for row in rows):
        raise AnalysisError("split has no test treatments")
    return rows


def candidate_checkpoint_name(config: str) -> str:
    safe = hashlib.sha256(config.encode()).hexdigest()[:16]
    return f"{safe}.json"


def profile_ref(sweep_dir: Path, spec: FamilySpec, codec: str, config: str) -> ProfileRef:
    folder = sweep_dir / spec.codec_folders[codec] / config
    return ProfileRef(
        family=spec.key,
        codec=codec,
        config=config,
        folder=folder,
        output_path=folder / "output.parquet",
        config_path=folder / "pipeline_config.yaml",
        pa_map_path=folder / "results" / "phenotypic_activity_map.csv",
    )


def validate_nonempty_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise AnalysisError(f"missing {label}: {path}")
    if path.stat().st_size <= 0:
        raise AnalysisError(f"zero-byte {label}: {path}")


def validate_resolved_config(path: Path) -> dict[str, Any]:
    validate_nonempty_file(path, "resolved pipeline config")
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("steps"), list):
        raise AnalysisError(f"invalid resolved pipeline config: {path}")
    metric_steps = [step for step in config["steps"] if step.get("name") == "evaluate_metrics"]
    if len(metric_steps) != 1 or not metric_steps[0].get("enabled", False):
        raise AnalysisError(f"expected one enabled evaluate_metrics step: {path}")
    params = metric_steps[0].get("params", {})
    expected = {
        "compound_col": COMPOUND_COL,
        "target_col": TARGET_COL,
        "negcon_col": NEGCON_COL,
    }
    for key, value in expected.items():
        if params.get(key) != value:
            raise AnalysisError(f"unexpected {key}={params.get(key)!r} in {path}; expected {value!r}")
    if tuple(params.get("pc_groups", ())) != PC_GROUPS:
        raise AnalysisError(f"unexpected pc_groups in {path}: {params.get('pc_groups')!r}")
    return config


def assert_fixed_config_coverage(
    sweep_dir: Path, spec: FamilySpec, config: str
) -> dict[str, ProfileRef]:
    refs: dict[str, ProfileRef] = {}
    failures: list[str] = []
    for codec in spec.codec_folders:
        ref = profile_ref(sweep_dir, spec, codec, config)
        try:
            validate_nonempty_file(ref.output_path, "normalized profile")
            validate_resolved_config(ref.config_path)
        except AnalysisError as exc:
            failures.append(f"{codec}: {exc}")
        refs[codec] = ref
    if failures:
        raise CoverageError(
            f"selected config {config!r} for {spec.display} lacks exact fixed-config coverage; "
            + "; ".join(failures)
        )
    return refs


def discover_raw_configs(sweep_dir: Path, spec: FamilySpec) -> list[str]:
    folder = sweep_dir / spec.raw_folder
    if not folder.is_dir():
        raise AnalysisError(f"missing Raw family folder: {folder}")
    configs = sorted(
        child.name
        for child in folder.iterdir()
        if child.is_dir()
        and (child / "pipeline_config.yaml").is_file()
        and (child / "results" / "phenotypic_activity_map.csv").is_file()
    )
    if not configs:
        raise AnalysisError(f"no candidate configurations found under {folder}")
    return configs


def canonical_annotation_digest(frame: pl.DataFrame) -> str:
    missing = [column for column in REQUIRED_ANNOTATION_COLS if column not in frame.columns]
    if missing:
        raise AnalysisError(f"profile is missing annotation columns: {missing}")
    annotations = frame.select(REQUIRED_ANNOTATION_COLS).with_columns(
        pl.col(ID_COL).cast(pl.Utf8),
        pl.col(COMPOUND_COL).cast(pl.Utf8),
        pl.col(GROUP_COL).cast(pl.Utf8),
        pl.col(NEGCON_COL).cast(pl.Boolean),
    )
    if annotations.select(pl.col(ID_COL).is_null().any()).item():
        raise AnalysisError("profile contains null Metadata_id")
    if annotations[ID_COL].n_unique() != len(annotations):
        raise AnalysisError("Metadata_id is not unique")
    digest = hashlib.sha256()
    for row in annotations.sort(ID_COL).iter_rows():
        digest.update("\0".join(str(value) for value in row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def load_profile(path: Path) -> tuple[pl.DataFrame, list[str], str]:
    validate_nonempty_file(path, "normalized profile")
    frame = pl.read_parquet(path)
    missing = [column for column in (*REQUIRED_ANNOTATION_COLS, TARGET_COL, PLATE_COL) if column not in frame.columns]
    if missing:
        raise AnalysisError(f"{path} is missing required columns: {missing}")
    features, _metadata = infer_columns(frame, ["Metadata_"])
    features = get_numeric_features(frame, features)
    if not features:
        raise AnalysisError(f"no numeric features in {path}")
    if frame.select([pl.col(column).is_null().any().alias(column) for column in features]).row(0).count(True):
        raise AnalysisError(f"null numeric features in {path}")
    finite_failure = frame.select(
        pl.any_horizontal([~pl.col(column).is_finite() for column in features]).any()
    ).item()
    if finite_failure:
        raise AnalysisError(f"non-finite numeric features in {path}")
    return frame, features, canonical_annotation_digest(frame)


def run_pc(frame: pl.DataFrame, features: list[str]) -> dict[str, Any]:
    distance = "euclidean" if len(features) <= 2 else "cosine"
    result = calculate_phenotypic_consistency(
        frame,
        features,
        compound_col=COMPOUND_COL,
        target_col=TARGET_COL,
        negcon_col=NEGCON_COL,
        group_col=GROUP_COL,
        pc_groups=list(PC_GROUPS),
        distance=distance,
    )
    if result.get("target_consistency") is None or result.get("n_targets_total", 0) <= 0:
        raise AnalysisError("PC computation produced no target-level result")
    value = result.get("mean_normalized_average_precision")
    if value is None or not math.isfinite(float(value)):
        raise AnalysisError(f"PC computation produced invalid mean NAP: {value!r}")
    return result


def run_pa(frame: pl.DataFrame, features: list[str]) -> dict[str, Any]:
    distance = "euclidean" if len(features) <= 2 else "cosine"
    result = calculate_phenotypic_activity(
        frame,
        features,
        compound_col=COMPOUND_COL,
        negcon_col=NEGCON_COL,
        batch_col=PLATE_COL,
        group_col=GROUP_COL,
        distance=distance,
    )
    if result.get("activity_map") is None or result.get("n_compounds", 0) <= 0:
        raise AnalysisError("PA computation produced no treatment-level result")
    value = result.get("mean_normalized_average_precision")
    if value is None or not math.isfinite(float(value)):
        raise AnalysisError(f"PA computation produced invalid mean NAP: {value!r}")
    return result


def validation_pa_from_archive(path: Path, validation_ids: set[str]) -> dict[str, Any]:
    validate_nonempty_file(path, "archived PA map")
    frame = pl.read_csv(path).with_columns(pl.col(COMPOUND_COL).cast(pl.Utf8))
    required = {COMPOUND_COL, GROUP_COL, "mean_normalized_average_precision"}
    if missing := required.difference(frame.columns):
        raise AnalysisError(f"archived PA map {path} lacks columns: {sorted(missing)}")
    selected = frame.filter(pl.col(COMPOUND_COL).is_in(sorted(validation_ids)))
    if selected.is_empty():
        raise AnalysisError(f"validation split has no PA units in {path}")
    value = float(selected["mean_normalized_average_precision"].mean())
    if not math.isfinite(value):
        raise AnalysisError(f"validation PA mean NAP is not finite in {path}")
    per_group = {
        row[GROUP_COL]: row["mean_normalized_average_precision"]
        for row in selected.group_by(GROUP_COL)
        .agg(pl.col("mean_normalized_average_precision").mean())
        .iter_rows(named=True)
    }
    return {
        "mean_nap": value,
        "median_nap": float(selected["mean_normalized_average_precision"].median()),
        "n_units": len(selected),
        "n_treatments": selected[COMPOUND_COL].n_unique(),
        "per_group_mean_nap": per_group,
    }


def evaluate_selection_candidate(
    ref: ProfileRef,
    validation_ids: set[str],
    expected_annotation_digest: str | None,
    log_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    validate_resolved_config(ref.config_path)
    pa = validation_pa_from_archive(ref.pa_map_path, validation_ids)
    frame, features, annotation_digest = load_profile(ref.output_path)
    if expected_annotation_digest is not None and annotation_digest != expected_annotation_digest:
        raise AnalysisError(
            f"Raw candidate annotation coverage differs within {ref.family}: {ref.output_path}"
        )
    validation_pc = frame.filter(
        pl.col(COMPOUND_COL).cast(pl.Utf8).is_in(sorted(validation_ids))
        & ~pl.col(NEGCON_COL)
        & pl.col(GROUP_COL).is_in(list(PC_GROUPS))
    )
    if validation_pc.is_empty():
        raise AnalysisError(f"no validation PC rows in {ref.output_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        pc = run_pc(validation_pc, features)
    result = {
        "status": "ok",
        "family": ref.family,
        "codec": ref.codec,
        "config": ref.config,
        "output_path": str(ref.output_path),
        "config_path": str(ref.config_path),
        "config_sha256": sha256_file(ref.config_path),
        "output_size_bytes": ref.output_path.stat().st_size,
        "annotation_digest": annotation_digest,
        "n_profile_rows": len(frame),
        "n_features": len(features),
        "validation_pa_mean_nap": pa["mean_nap"],
        "validation_pa_median_nap": pa["median_nap"],
        "validation_pa_units": pa["n_units"],
        "validation_pa_treatments": pa["n_treatments"],
        "validation_pc_mean_nap": float(pc["mean_normalized_average_precision"]),
        "validation_pc_median_nap": float(pc["median_normalized_average_precision"]),
        "validation_pc_targets": int(pc["n_targets_total"]),
        "validation_pc_rows": len(validation_pc),
        "validation_pa_group_mean_nap": pa["per_group_mean_nap"],
        "elapsed_seconds": time.monotonic() - started,
    }
    del validation_pc, frame
    gc.collect()
    return result


def configure_worker_cache(output_dir: Path) -> None:
    """Give each process a private copairs cache to prevent concurrent .npy writes."""
    worker_home = output_dir / "worker_cache" / f"pid-{os.getpid()}"
    worker_home.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(worker_home)


def evaluate_selection_worker(
    ref: ProfileRef,
    validation_ids: set[str],
    log_path: Path,
) -> dict[str, Any]:
    """Process-pool boundary that converts candidate failures into records."""
    configure_worker_cache(log_path.parents[3])
    try:
        return evaluate_selection_candidate(ref, validation_ids, None, log_path)
    except Exception as exc:
        return {
            "status": "failed",
            "family": ref.family,
            "codec": "Raw",
            "config": ref.config,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "output_path": str(ref.output_path),
        }


def minmax_score_candidates(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if not records:
        raise AnalysisError("cannot select from zero successful candidates")
    pa_values = [float(row["validation_pa_mean_nap"]) for row in records]
    pc_values = [float(row["validation_pc_mean_nap"]) for row in records]
    if not all(math.isfinite(value) for value in (*pa_values, *pc_values)):
        raise AnalysisError("candidate metrics contain NaN or infinity")
    pa_min, pa_max = min(pa_values), max(pa_values)
    pc_min, pc_max = min(pc_values), max(pc_values)

    def scale(value: float, low: float, high: float) -> float:
        return 1.0 if high == low else (value - low) / (high - low)

    scored: list[dict[str, Any]] = []
    for row in records:
        item = dict(row)
        item["pa_scaled"] = scale(float(row["validation_pa_mean_nap"]), pa_min, pa_max)
        item["pc_scaled"] = scale(float(row["validation_pc_mean_nap"]), pc_min, pc_max)
        item["selection_score"] = item["pa_scaled"] * item["pc_scaled"]
        scored.append(item)
    scored.sort(key=lambda row: (-row["selection_score"], str(row["config"])))
    for rank, row in enumerate(scored, start=1):
        row["rank"] = rank
    ranges = {
        "pa_min": pa_min,
        "pa_max": pa_max,
        "pc_min": pc_min,
        "pc_max": pc_max,
    }
    return scored, ranges


def split_manifest_from_reference(
    sweep_dir: Path, validation_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], ProfileRef]:
    spec = FAMILIES["morphem"]
    configs = discover_raw_configs(sweep_dir, spec)
    reference: ProfileRef | None = None
    for config in configs:
        candidate = profile_ref(sweep_dir, spec, "Raw", config)
        if candidate.output_path.is_file() and candidate.output_path.stat().st_size > 0:
            reference = candidate
            break
    if reference is None:
        raise AnalysisError("no nonempty canonical MorphEM Raw profile for split construction")
    columns = [COMPOUND_COL, GROUP_COL, NEGCON_COL]
    frame = pl.read_parquet(reference.output_path, columns=columns).with_columns(
        pl.col(COMPOUND_COL).cast(pl.Utf8), pl.col(GROUP_COL).cast(pl.Utf8)
    )
    treatment_rows = frame.filter(~pl.col(NEGCON_COL))
    if treatment_rows.select(pl.col(COMPOUND_COL).is_null().any()).item():
        raise AnalysisError("canonical reference contains null non-control treatment IDs")
    memberships: dict[str, set[str]] = {}
    for treatment_id, group in treatment_rows.select(COMPOUND_COL, GROUP_COL).unique().iter_rows():
        memberships.setdefault(treatment_id, set()).add(group)
    return make_treatment_split(memberships, validation_fraction, seed), reference


def build_common_well_manifest(
    refs: Mapping[tuple[str, str], ProfileRef], split_by_id: Mapping[str, str]
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    metadata: dict[tuple[str, str], pl.DataFrame] = {}
    common_ids: set[str] | None = None
    coverage_rows: list[dict[str, Any]] = []
    for key, ref in refs.items():
        frame = pl.read_parquet(ref.output_path, columns=list(REQUIRED_ANNOTATION_COLS)).with_columns(
            pl.col(ID_COL).cast(pl.Utf8),
            pl.col(COMPOUND_COL).cast(pl.Utf8),
            pl.col(GROUP_COL).cast(pl.Utf8),
            pl.col(NEGCON_COL).cast(pl.Boolean),
        )
        canonical_annotation_digest(frame)
        ids = set(frame[ID_COL].to_list())
        common_ids = ids if common_ids is None else common_ids.intersection(ids)
        metadata[key] = frame
        coverage_rows.append(
            {
                "family": key[0],
                "codec": key[1],
                "config": ref.config,
                "native_wells": len(frame),
                "native_treatment_ids": frame.filter(~pl.col(NEGCON_COL))[COMPOUND_COL].n_unique(),
                "native_controls": frame.filter(pl.col(NEGCON_COL)).height,
                "output_path": str(ref.output_path),
            }
        )
    if not common_ids:
        raise AnalysisError("selected outputs have an empty Metadata_id intersection")
    ordered_common = sorted(common_ids)
    baseline_key = next(iter(metadata))
    baseline = metadata[baseline_key].filter(pl.col(ID_COL).is_in(ordered_common)).sort(ID_COL)
    baseline_digest = canonical_annotation_digest(baseline)
    for key, frame in metadata.items():
        subset = frame.filter(pl.col(ID_COL).is_in(ordered_common)).sort(ID_COL)
        if len(subset) != len(ordered_common):
            raise AnalysisError(f"failed to recover common wells for {key}")
        if canonical_annotation_digest(subset) != baseline_digest:
            raise AnalysisError(f"intersected annotations disagree for {key}")
    common = baseline.with_columns(
        pl.when(pl.col(NEGCON_COL))
        .then(pl.lit("shared_control"))
        .otherwise(pl.col(COMPOUND_COL).replace_strict(split_by_id, default=None))
        .alias("split")
    )
    if common.filter(~pl.col(NEGCON_COL) & pl.col("split").is_null()).height:
        raise AnalysisError("common-well treatments are absent from the frozen split manifest")
    for row in coverage_rows:
        row["common_wells"] = len(common)
        row["dropped_for_intersection"] = row["native_wells"] - len(common)
    return common, coverage_rows


def verify_plate_controls(frame: pl.DataFrame) -> None:
    controls = frame.filter(pl.col(NEGCON_COL))
    treatments = frame.filter(~pl.col(NEGCON_COL))
    if controls.is_empty():
        raise AnalysisError("held-out frame has no negative-control wells")
    required = treatments.select(PLATE_COL, GROUP_COL).unique()
    available = controls.select(PLATE_COL, GROUP_COL).unique()
    missing = required.join(available, on=[PLATE_COL, GROUP_COL], how="anti")
    if len(missing):
        examples = missing.head(10).to_dicts()
        raise AnalysisError(f"held-out treatment plate/group pairs lack controls: {examples}")


def flatten_group_summary(summary: pd.DataFrame | None, prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if summary is None or summary.empty:
        return result
    for _, row in summary.iterrows():
        group = row[GROUP_COL]
        for column, value in row.items():
            if column != GROUP_COL:
                result[f"{prefix}_{group}_{column}"] = value.item() if hasattr(value, "item") else value
    return result


def evaluate_final_profile(
    ref: ProfileRef,
    common_ids: set[str],
    test_ids: set[str],
    output_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    configure_worker_cache(output_dir)
    started = time.monotonic()
    validate_resolved_config(ref.config_path)
    frame, features, annotation_digest = load_profile(ref.output_path)
    frame = frame.filter(
        pl.col(ID_COL).cast(pl.Utf8).is_in(sorted(common_ids))
        & (pl.col(NEGCON_COL) | pl.col(COMPOUND_COL).cast(pl.Utf8).is_in(sorted(test_ids)))
    )
    if frame.is_empty():
        raise AnalysisError(f"held-out common-well frame is empty for {ref.output_path}")
    if frame[ID_COL].n_unique() != len(frame):
        raise AnalysisError(f"duplicate held-out Metadata_id in {ref.output_path}")
    verify_plate_controls(frame)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        pa = run_pa(frame, features)
        pc = run_pc(frame, features)
    pa_map = pa["activity_map"].copy()
    pa_map.insert(0, "codec", ref.codec)
    pa_map.insert(0, "family", ref.family)
    pa_map.insert(2, "config", ref.config)
    pc_map = pc["target_consistency"].copy()
    pc_map.insert(0, "codec", ref.codec)
    pc_map.insert(0, "family", ref.family)
    pc_map.insert(2, "config", ref.config)
    maps_dir = output_dir / "per_unit"
    atomic_write_pandas_csv(maps_dir / f"{ref.family}__{ref.codec}__pa_treatments.csv", pa_map)
    atomic_write_pandas_csv(maps_dir / f"{ref.family}__{ref.codec}__pc_targets.csv", pc_map)
    result = {
        "status": "ok",
        "family": ref.family,
        "codec": ref.codec,
        "config": ref.config,
        "output_path": str(ref.output_path),
        "config_path": str(ref.config_path),
        "config_sha256": sha256_file(ref.config_path),
        "output_size_bytes": ref.output_path.stat().st_size,
        "annotation_digest_native": annotation_digest,
        "n_native_profile_rows": pl.scan_parquet(ref.output_path).select(pl.len()).collect().item(),
        "n_evaluation_wells": len(frame),
        "n_evaluation_controls": frame.filter(pl.col(NEGCON_COL)).height,
        "n_evaluation_treatment_ids": frame.filter(~pl.col(NEGCON_COL))[COMPOUND_COL].n_unique(),
        "n_features": len(features),
        "distance": "euclidean" if len(features) <= 2 else "cosine",
        "test_pa_mean_nap": float(pa["mean_normalized_average_precision"]),
        "test_pa_median_nap": float(pa["median_normalized_average_precision"]),
        "test_pa_units": int(pa["n_compounds"]),
        "test_pc_mean_nap": float(pc["mean_normalized_average_precision"]),
        "test_pc_median_nap": float(pc["median_normalized_average_precision"]),
        "test_pc_targets": int(pc["n_targets_total"]),
        "test_balanced_nap_product": float(pa["mean_normalized_average_precision"])
        * float(pc["mean_normalized_average_precision"]),
        "elapsed_seconds": time.monotonic() - started,
        **flatten_group_summary(pa.get("group_summary"), "PA"),
        **flatten_group_summary(pc.get("group_summary"), "PC"),
    }
    del frame
    gc.collect()
    return result


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def protocol_payload(args: argparse.Namespace, families: Sequence[str]) -> dict[str, Any]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "sweep_dir": str(args.sweep_dir.resolve()),
        "families": list(families),
        "validation_fraction": args.validation_fraction,
        "seed": args.seed,
        "max_selection_configs": args.max_selection_configs,
        "workers": args.workers,
        "compound_col": COMPOUND_COL,
        "target_col": TARGET_COL,
        "negative_control_col": NEGCON_COL,
        "group_col": GROUP_COL,
        "pc_groups": list(PC_GROUPS),
        "selection_codec": "Raw",
        "selection_score": "within-family min-max(PA mean NAP) * min-max(PC mean NAP)",
        "repo_head": git_head(),
        "runner_sha256": sha256_file(Path(__file__)),
    }


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    path: Path,
) -> None:
    missing = required.difference(frame.columns)
    if missing:
        raise AnalysisError(
            f"downstream uncertainty file has missing columns: "
            f"path={path}, missing={sorted(missing)}"
        )
    if frame.empty:
        raise AnalysisError(f"downstream uncertainty file is empty: {path}")


def _require_finite(
    frame: pd.DataFrame,
    columns: Sequence[str],
    path: Path,
) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not values.map(math.isfinite).all():
            raise AnalysisError(
                f"downstream uncertainty contains non-finite {column}: {path}"
            )


def _record_relative_path(value: str) -> str:
    marker = "/rebuttal/postprocessing_validation/"
    normalized = value.replace("\\", "/")
    if marker in normalized:
        return normalized.split(marker, 1)[1]
    return normalized.lstrip("./")


def validate_downstream_uncertainty(
    output_dir: Path,
    final_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Validate and summarize the complete downstream bootstrap/figure bundle."""
    uncertainty_dir = output_dir / "uncertainty"
    named_paths = {
        "summary": uncertainty_dir / "heldout_uncertainty.csv",
        "codec": uncertainty_dir / "codec_vs_raw_paired.csv",
        "pairwise": uncertainty_dir / "model_pairwise_by_codec.csv",
        "ranks": uncertainty_dir / "model_rank_bounds.csv",
        "diagnostics": uncertainty_dir / "bootstrap_diagnostics.csv",
        "audit": uncertainty_dir / "resampling_unit_audit.csv",
        "report": uncertainty_dir / "REPORT.md",
        "provenance": uncertainty_dir / "provenance.json",
        "figure_pdf": output_dir / "heldout_codec_performance.pdf",
        "figure_png": output_dir / "heldout_codec_performance.png",
        "checksums": output_dir / "artifact_checksums.json",
    }
    uncertainty_paths = [
        path for name, path in named_paths.items() if name not in {"figure_pdf", "figure_png", "checksums"}
    ]
    if not any(path.is_file() for path in uncertainty_paths):
        return None
    missing = [
        str(path.relative_to(output_dir))
        for path in named_paths.values()
        if not path.is_file()
    ]
    if missing:
        raise AnalysisError(
            f"incomplete downstream uncertainty/figure bundle: missing={missing}"
        )

    try:
        summary = pd.read_csv(named_paths["summary"])
        codec = pd.read_csv(named_paths["codec"])
        pairwise = pd.read_csv(named_paths["pairwise"])
        ranks = pd.read_csv(named_paths["ranks"])
        diagnostics = pd.read_csv(named_paths["diagnostics"])
        audit = pd.read_csv(named_paths["audit"])
        provenance = json.loads(named_paths["provenance"].read_text())
        checksum_payload = json.loads(named_paths["checksums"].read_text())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"invalid downstream uncertainty bundle: {exc}") from exc

    _require_columns(
        summary,
        {
            "family", "codec", "pa_clusters", "pc_clusters", "pa_point",
            "pc_point", "product_point", "product_ci_low", "product_ci_high",
            "replicates", "seed",
        },
        named_paths["summary"],
    )
    _require_columns(
        codec,
        {
            "family", "codec", "product_holm_p", "product_supported_direction",
            "multiplicity_family_size",
        },
        named_paths["codec"],
    )
    _require_columns(
        pairwise,
        {
            "codec", "family_a", "family_b", "product_holm_p",
            "product_supported_direction", "multiplicity_family_size",
        },
        named_paths["pairwise"],
    )
    _require_columns(
        ranks,
        {
            "codec", "family", "product_point", "simultaneous_rank_lower",
            "simultaneous_rank_upper",
        },
        named_paths["ranks"],
    )
    _require_columns(
        diagnostics,
        {"point", "final_replicates", "final_ci_low", "final_ci_high"},
        named_paths["diagnostics"],
    )
    _require_columns(
        audit,
        {"n_family_codec_tables", "key_sets_identical", "point_estimate_max_abs_error"},
        named_paths["audit"],
    )
    for frame, columns, key in (
        (summary, ("pa_point", "pc_point", "product_point", "product_ci_low", "product_ci_high", "replicates", "seed"), "summary"),
        (codec, ("product_holm_p", "multiplicity_family_size"), "codec"),
        (pairwise, ("product_holm_p", "multiplicity_family_size"), "pairwise"),
        (ranks, ("product_point", "simultaneous_rank_lower", "simultaneous_rank_upper"), "ranks"),
        (diagnostics, ("point", "final_replicates", "final_ci_low", "final_ci_high"), "diagnostics"),
        (audit, ("n_family_codec_tables", "point_estimate_max_abs_error"), "audit"),
    ):
        _require_finite(frame, columns, named_paths[key])

    expected_points = {
        (str(row["family"]), str(row["codec"])): float(
            row["test_balanced_nap_product"]
        )
        for row in final_rows
    }
    if len(expected_points) != len(final_rows) or not all(
        math.isfinite(value) for value in expected_points.values()
    ):
        raise AnalysisError("held-out results contain duplicate or non-finite variants")
    observed_keys = list(zip(summary["family"].astype(str), summary["codec"].astype(str)))
    if len(observed_keys) != len(set(observed_keys)) or set(observed_keys) != set(expected_points):
        raise AnalysisError(
            "downstream uncertainty variants do not match held-out results"
        )
    uncertainty_points = {
        key: float(value)
        for key, value in zip(observed_keys, summary["product_point"], strict=True)
    }
    max_error = max(
        abs(uncertainty_points[key] - expected_points[key]) for key in expected_points
    )
    if not math.isfinite(max_error) or max_error > 1e-12:
        raise AnalysisError(
            "downstream uncertainty points are stale relative to held-out "
            f"results: max_error={max_error}"
        )
    if not (
        (summary["product_ci_low"] <= summary["product_point"])
        & (summary["product_point"] <= summary["product_ci_high"])
    ).all():
        raise AnalysisError("downstream uncertainty product interval excludes its point")

    variants = set(expected_points)
    expected_codec_pairs = {
        (family, codec_name)
        for family, codec_name in variants
        if codec_name != "Raw" and (family, "Raw") in variants
    }
    observed_codec_pairs = set(zip(codec["family"].astype(str), codec["codec"].astype(str)))
    if len(codec) != len(observed_codec_pairs) or observed_codec_pairs != expected_codec_pairs:
        raise AnalysisError("codec-vs-Raw uncertainty inventory is incomplete or duplicated")
    expected_pairwise = {
        (codec_name, family_a, family_b)
        for codec_name in {codec_name for _, codec_name in variants}
        for family_a in sorted(family for family, codec in variants if codec == codec_name)
        for family_b in sorted(family for family, codec in variants if codec == codec_name)
        if family_a < family_b
    }
    observed_pairwise = set(
        zip(
            pairwise["codec"].astype(str),
            pairwise["family_a"].astype(str),
            pairwise["family_b"].astype(str),
        )
    )
    if len(pairwise) != len(observed_pairwise) or observed_pairwise != expected_pairwise:
        raise AnalysisError("same-codec pairwise uncertainty inventory is incomplete or duplicated")
    rank_keys = set(zip(ranks["family"].astype(str), ranks["codec"].astype(str)))
    if len(ranks) != len(rank_keys) or rank_keys != variants:
        raise AnalysisError("rank-bound uncertainty inventory is incomplete or duplicated")

    replicates = int(summary["replicates"].iloc[0])
    seed = int(summary["seed"].iloc[0])
    if replicates < 100 or summary["replicates"].nunique() != 1 or summary["seed"].nunique() != 1:
        raise AnalysisError("uncertainty rows have inconsistent protocol parameters")
    if not (diagnostics["final_replicates"] == replicates).all():
        raise AnalysisError("bootstrap diagnostics use a different replicate count")
    if not audit["key_sets_identical"].astype(bool).all():
        raise AnalysisError("uncertainty resampling key sets are not identical")
    if int(audit["n_family_codec_tables"].max()) != len(variants):
        raise AnalysisError("uncertainty audit has the wrong variant count")
    if float(audit["point_estimate_max_abs_error"].max()) > 1e-12:
        raise AnalysisError("uncertainty audit point estimates are stale")

    dimensions = provenance.get("dimensions", {})
    multiplicity = provenance.get("multiplicity", {})
    if (
        int(provenance.get("method_version", -1)) != 1
        or int(provenance.get("replicates", -1)) != replicates
        or int(provenance.get("seed", -1)) != seed
        or int(dimensions.get("variants", -1)) != len(variants)
        or int(multiplicity.get("codec_vs_raw", {}).get("comparisons", -1)) != len(codec)
        or int(multiplicity.get("same_codec_models", {}).get("comparisons", -1)) != len(pairwise)
    ):
        raise AnalysisError("uncertainty provenance protocol or dimensions are stale")
    provenance_variants = {
        (str(row["family"]), str(row["codec"]))
        for row in provenance.get("variants", [])
    }
    if provenance_variants != variants:
        raise AnalysisError("uncertainty provenance variants are stale")

    provenance_outputs = {
        Path(str(row["path"])).name: row for row in provenance.get("outputs", [])
    }
    for path in uncertainty_paths[:-1]:
        row = provenance_outputs.get(path.name)
        if row is None or int(row.get("size_bytes", -1)) != path.stat().st_size or row.get("sha256") != sha256_file(path):
            raise AnalysisError(f"uncertainty provenance does not bind output: {path}")
    script_path = Path(__file__).with_name("bootstrap_uncertainty.py")
    script_record = provenance.get("script", {})
    if (
        int(script_record.get("size_bytes", -1)) != script_path.stat().st_size
        or script_record.get("sha256") != sha256_file(script_path)
    ):
        raise AnalysisError("uncertainty provenance does not bind the bootstrap source")

    checksum_records: dict[str, Mapping[str, Any]] = {}
    for section in ("result_artifacts", "uncertainty_outputs"):
        for row in checksum_payload.get(section, []):
            checksum_records[_record_relative_path(str(row["path"]))] = row
    bound_paths = [*uncertainty_paths, named_paths["figure_pdf"], named_paths["figure_png"]]
    for path in bound_paths:
        relative = path.relative_to(output_dir.parent).as_posix()
        row = checksum_records.get(relative)
        if row is None or int(row.get("size_bytes", -1)) != path.stat().st_size or row.get("sha256") != sha256_file(path):
            raise AnalysisError(f"artifact checksums do not bind downstream output: {path}")
    for field, path in (
        ("base_analysis_source", Path(__file__)),
        ("base_analysis_test_source", Path(__file__).with_name("test_run_analysis.py")),
    ):
        row = checksum_payload.get(field, {})
        if (
            int(row.get("size_bytes", -1)) != path.stat().st_size
            or row.get("sha256") != sha256_file(path)
        ):
            raise AnalysisError(f"artifact checksums do not bind {field}: {path}")

    report_text = named_paths["report"].read_text()
    if not report_text.startswith("# Held-out paired cluster-bootstrap uncertainty\n"):
        raise AnalysisError("downstream uncertainty report is malformed")
    learned = {"dinov2", "morphem", "openphenom", "subcell"}
    morph_rows = pairwise.loc[
        pairwise["codec"].isin({"Raw", "HQ", "MQ", "D20"})
        & pairwise["family_a"].isin(learned)
        & pairwise["family_b"].isin(learned)
        & ((pairwise["family_a"] == "morphem") | (pairwise["family_b"] == "morphem"))
    ]
    morph_supported = sum(
        str(row.product_supported_direction).startswith("morphem>")
        for row in morph_rows.itertuples(index=False)
    )
    middle = {"dinov2", "openphenom", "subcell"}
    raw_middle = pairwise.loc[
        (pairwise["codec"] == "Raw")
        & pairwise["family_a"].isin(middle)
        & pairwise["family_b"].isin(middle)
    ]
    raw_middle_unresolved = len(raw_middle) == 3 and (
        raw_middle["product_supported_direction"] == "unresolved"
    ).all()
    return {
        "replicates": replicates,
        "seed": seed,
        "pa_clusters": int(dimensions.get("pa_clusters", summary["pa_clusters"].iloc[0])),
        "pc_clusters": int(dimensions.get("pc_clusters", summary["pc_clusters"].iloc[0])),
        "codec_comparisons": len(codec),
        "pairwise_comparisons": len(pairwise),
        "morph_supported": morph_supported,
        "morph_total": len(morph_rows),
        "raw_middle_unresolved": raw_middle_unresolved,
    }


def render_report(
    output_dir: Path,
    protocol: Mapping[str, Any],
    split_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    final_rows: Sequence[Mapping[str, Any]],
    coverage_rows: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
) -> None:
    validation_count = sum(row["split"] == "validation" for row in split_rows)
    test_count = sum(row["split"] == "test" for row in split_rows)
    common_wells = int(coverage_rows[0]["common_wells"])
    evaluation_wells = int(final_rows[0]["n_evaluation_wells"])
    evaluation_controls = int(final_rows[0]["n_evaluation_controls"])
    evaluation_test_wells = evaluation_wells - evaluation_controls
    evaluation_test_ids = int(final_rows[0]["n_evaluation_treatment_ids"])
    evaluation_pa_units = int(final_rows[0]["test_pa_units"])
    evaluation_pc_targets = int(final_rows[0]["test_pc_targets"])
    tied_winners = sum(
        row.get("score_margin") not in (None, "")
        and float(row["score_margin"]) == 0.0
        for row in selected_rows
    )
    is_smoke = protocol.get("max_selection_configs") is not None or set(
        protocol["families"]
    ) != set(FAMILIES)
    uncertainty_summary = validate_downstream_uncertainty(output_dir, final_rows)
    uncertainty_ready = uncertainty_summary is not None
    figure_ready = all(
        (output_dir / f"heldout_codec_performance.{suffix}").is_file()
        for suffix in ("pdf", "png")
    )
    lines = [
        "# Post-processing validation/test analysis",
        "",
        f"Generated: {utc_now()}",
        "",
        f"**Run class:** {'SMOKE / INCOMPLETE' if is_smoke else 'PRODUCTION'}.",
        "",
        "## Protocol",
        "",
        f"- Frozen archive: `{protocol['sweep_dir']}` (read only).",
        f"- Split: SHA-256 seed `{protocol['seed']}`, {protocol['validation_fraction']:.0%} validation and {1-protocol['validation_fraction']:.0%} test, stratified by each treatment's composite group membership.",
        f"- Treatments: {validation_count:,} validation and {test_count:,} test `Metadata_JCP2022` IDs; controls are shared references and are not split treatments.",
        "- Selection: Raw validation treatments only; archived per-treatment PA NAP plus recomputed validation PC NAP. Each metric is min-max scaled within family, then multiplied.",
        "- Evaluation: the exact selected configuration is pinned across all intended codecs and recomputed on held-out test treatments over one frozen common-well population.",
        f"- Frozen common manifest: {common_wells:,} `Metadata_id` values.",
        f"- Held-out evaluation: {evaluation_wells:,} test-plus-control wells ({evaluation_test_wells:,} test wells and {evaluation_controls:,} shared controls), covering {evaluation_test_ids:,}/{test_count:,} assigned test IDs, {evaluation_pa_units:,} treatment/group PA units, and {evaluation_pc_targets:,} target/group PC units.",
        "",
        "## Selected configurations",
        "",
        "| Family | Config | Validation PA NAP | Validation PC NAP | Score | Runner-up |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in selected_rows:
        lines.append(
            f"| {row['display']} | `{row['config']}` | {row['validation_pa_mean_nap']:.4f} | "
            f"{row['validation_pc_mean_nap']:.4f} | {row['selection_score']:.4f} | "
            f"`{row.get('runner_up_config', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Held-out fixed-configuration results",
            "",
            "| Family | Codec | PA mean NAP | PC mean NAP | Product | Wells | PA units | PC targets |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(final_rows, key=lambda item: (item["family"], item["codec"])):
        lines.append(
            f"| {FAMILIES[row['family']].display} | {row['codec']} | {row['test_pa_mean_nap']:.4f} | "
            f"{row['test_pc_mean_nap']:.4f} | {row['test_balanced_nap_product']:.4f} | "
            f"{row['n_evaluation_wells']:,} | {row['test_pa_units']:,} | {row['test_pc_targets']:,} |"
        )
    lines.extend(
        [
            "",
            "## Completeness and failures",
            "",
            f"- Candidate failures: {len(failures)} (see `selection_failures.csv` when nonzero).",
            f"- Exact tied validation winners: {tied_winners}; ties were broken lexically and are not evidence of a uniquely optimal recipe.",
            "- Exact codec coverage was required for every selected configuration; no per-codec fallback was permitted.",
            "- Native and common-well counts are in `selected_codec_coverage.csv`.",
            "",
            "## Interpretation and limitations",
            "",
            "This analysis isolates **post-processing configuration selection**: no held-out treatment score was used to choose a configuration. It is not a strict inductive preprocessing holdout. The archived normalized matrices were fitted before the treatment split, and all-profile feature filtering or normalization can therefore expose test-distribution information even though labels and test scores were not used for configuration selection.",
            "",
            "The split is treatment-disjoint, not target-disjoint, and the point-estimate model ordering must remain descriptive until paired perturbation/target uncertainty is reported. Validation selection uses a within-family min-max-scaled product, whereas the held-out table reports the unscaled PA mean NAP times PC mean NAP; those products are not directly comparable.",
            "",
            "The archived PA validation scores are reusable because each treatment/group NAP was constructed from that treatment's replicates and same-plate/group negative controls; full-cohort PA significance rates were not reused. PC was recomputed because its retrieval population depends on which treatments are present.",
            "",
            "The active staging cleanup does not modify these frozen WACV normalized profiles. A fresh image/embedding extraction after deletion of negative-control images is not interchangeable: those controls are required for control-fitted normalization and PA reference construction.",
            "",
            "The existing full-data, cross-codec best-average fixed-configuration analysis remains a separate sensitivity check. Its configuration selection uses all treatments/codecs and must not be described as this validation/test result.",
            "",
            "CellProfiler and CellCount have Raw profiles only. CellProfiler also retains the manuscript's known site-count asymmetry relative to the four-site deep-learning inputs.",
            "",
            "## Output inventory",
            "",
            "- `treatment_split.csv` / `split_summary.csv`: frozen split and stratum counts.",
            "- `validation_config_scores.csv`: all successful Raw validation candidates and ranks.",
            "- `selected_configs.csv`: pinned family configurations and runner-up margins.",
            "- `heldout_test_scores.csv`: primary held-out results.",
            "- `per_unit/*`: held-out PA treatment and PC target tables for uncertainty analyses.",
            "- `provenance.json` and `selected_codec_coverage.csv`: archive/config paths, hashes, and coverage.",
        ]
    )
    pending_uncertainty = (
        "The split is treatment-disjoint, not target-disjoint, and the "
        "point-estimate model ordering must remain descriptive until paired "
        "perturbation/target uncertainty is reported. Validation selection uses "
        "a within-family min-max-scaled product, whereas the held-out table "
        "reports the unscaled PA mean NAP times PC mean NAP; those products are "
        "not directly comparable."
    )
    if uncertainty_ready:
        assert uncertainty_summary is not None
        morph_supported = int(uncertainty_summary["morph_supported"])
        morph_total = int(uncertainty_summary["morph_total"])
        if morph_supported == morph_total:
            morph_claim = (
                f"It supports MorphEM's lead in all {morph_total} learned-model "
                "pairwise comparisons"
            )
        else:
            morph_claim = (
                f"It supports MorphEM's lead in {morph_supported}/{morph_total} "
                "learned-model pairwise comparisons"
            )
        raw_ordering = (
            ", while the Raw middle-model ordering remains unresolved"
            if uncertainty_summary["raw_middle_unresolved"]
            else ""
        )
        uncertainty_index = lines.index(pending_uncertainty)
        lines[uncertainty_index : uncertainty_index + 1] = [
            (
                "The split is treatment-disjoint, not target-disjoint. A separate "
                f"{int(uncertainty_summary['replicates']):,}-replicate paired, "
                "stratified cluster bootstrap now quantifies conditional uncertainty "
                f"over the {int(uncertainty_summary['pa_clusters']):,} evaluable PA "
                f"treatment clusters and {int(uncertainty_summary['pc_clusters']):,} "
                f"PC target clusters. {morph_claim} after Holm correction across all "
                f"{int(uncertainty_summary['pairwise_comparisons'])} same-codec model "
                f"comparisons{raw_ordering}. Validation selection uses a within-family "
                "min-max-scaled product, whereas the held-out table reports the "
                "unscaled PA mean NAP times PC mean NAP; those products are not "
                "directly comparable."
            ),
            "",
            (
                "The bootstrap is conditional on the frozen per-unit retrieval "
                "results, target eligibility, selected recipes, and normalized "
                "profiles. PA and PC margins are resampled independently under a "
                "working product-of-margins model because the saved target summaries "
                "omit the treatment--target/query decomposition needed to propagate "
                "treatment resampling through PC retrieval. The resulting pointwise "
                "intervals and centered-bootstrap tests omit unknown PA--PC covariance "
                "and may therefore be too narrow or too wide. They do not quantify "
                "recipe-selection, split, transform-fitting, control, well/site, or "
                "annotation uncertainty; a non-supported contrast is not evidence of "
                "equivalence. Full methods, contrasts, rank bounds, diagnostics, and "
                "provenance are under `uncertainty/`."
            ),
        ]
        per_unit_index = lines.index(
            "- `per_unit/*`: held-out PA treatment and PC target tables for "
            "uncertainty analyses."
        )
        lines[per_unit_index : per_unit_index + 1] = [
            "- `heldout_codec_performance.pdf` and `.png`: reproducible paper "
            "figure with conditional intervals.",
            "- `per_unit/*`: held-out PA treatment and PC target tables used for "
            "uncertainty.",
            "- `uncertainty/*`: interval, codec contrast, pairwise model, rank-bound, "
            "diagnostic, report, and provenance outputs.",
        ]
    if figure_ready:
        output_inventory_index = lines.index("## Output inventory")
        if uncertainty_ready:
            assert uncertainty_summary is not None
            figure_description = (
                "The left panel reports the absolute unscaled PA--PC product; the "
                "right panel reports each learned model's point-estimate percentage "
                "change from its own Raw score. Whiskers are pointwise, conditional "
                "95% percentile intervals from "
                f"{int(uncertainty_summary['replicates']):,} paired, stratified "
                "cluster-bootstrap resamples of PA treatment IDs and PC target IDs "
                "under the working product-of-margins model. The same cluster weights "
                "are used across models/codecs within each margin."
            )
        else:
            figure_description = (
                "The left panel reports the absolute unscaled PA--PC product; the "
                "right panel reports each learned model's point-estimate percentage "
                "change from its own Raw score."
            )
        lines[output_inventory_index:output_inventory_index] = [
            "## Figure",
            "",
            "![Held-out fixed-recipe performance across codecs]"
            "(heldout_codec_performance.png)",
            "",
            figure_description,
            "",
        ]
    atomic_write_text(output_dir / "REPORT.md", "\n".join(lines) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=sorted(FAMILIES),
        default=list(FAMILIES),
        help="families to analyze (default: all primary models and controls)",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="parallel candidate/final workers (default: 4)",
    )
    parser.add_argument(
        "--max-selection-configs",
        type=int,
        default=None,
        help="lexically limit candidates per family for a smoke test; use a separate output directory",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    args.sweep_dir = args.sweep_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.sweep_dir.is_dir():
        raise AnalysisError(f"canonical sweep directory not found: {args.sweep_dir}")
    families = list(dict.fromkeys(args.families))
    if args.max_selection_configs is not None and args.max_selection_configs <= 0:
        raise AnalysisError("--max-selection-configs must be positive")
    if args.workers <= 0:
        raise AnalysisError("--workers must be positive")
    invocation_started_at = utc_now()
    protocol = protocol_payload(args, families)
    protocol_hash = sha256_bytes(json.dumps(protocol, sort_keys=True).encode())
    protocol = {**protocol, "protocol_hash": protocol_hash, "started_at": invocation_started_at}
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        comparable_previous = {key: previous.get(key) for key in protocol if key != "started_at"}
        comparable_current = {key: value for key, value in protocol.items() if key != "started_at"}
        if comparable_previous != comparable_current:
            raise AnalysisError(
                f"output directory contains a different protocol; choose another directory: {args.output_dir}"
            )
        protocol["started_at"] = previous["started_at"]
    else:
        atomic_write_json(protocol_path, protocol)

    split_rows, split_reference = split_manifest_from_reference(
        args.sweep_dir, args.validation_fraction, args.seed
    )
    atomic_write_csv_rows(args.output_dir / "treatment_split.csv", split_rows)
    split_summary = []
    for stratum in sorted({row["stratum"] for row in split_rows}):
        subset = [row for row in split_rows if row["stratum"] == stratum]
        split_summary.append(
            {
                "stratum": stratum,
                "validation": sum(row["split"] == "validation" for row in subset),
                "test": sum(row["split"] == "test" for row in subset),
                "total": len(subset),
            }
        )
    atomic_write_csv_rows(args.output_dir / "split_summary.csv", split_summary)
    validation_ids = {row["treatment_id"] for row in split_rows if row["split"] == "validation"}
    test_ids = {row["treatment_id"] for row in split_rows if row["split"] == "test"}

    successful: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    selected_refs: dict[tuple[str, str], ProfileRef] = {}
    selection_ranges: dict[str, Any] = {}

    for family in families:
        spec = FAMILIES[family]
        configs = discover_raw_configs(args.sweep_dir, spec)
        if args.max_selection_configs is not None:
            configs = configs[: args.max_selection_configs]
        family_success: list[dict[str, Any]] = []
        pending: list[tuple[ProfileRef, Path, Path]] = []
        for config in configs:
            ref = profile_ref(args.sweep_dir, spec, "Raw", config)
            checkpoint = (
                args.output_dir
                / "checkpoints"
                / "selection"
                / family
                / candidate_checkpoint_name(config)
            )
            cached = load_checkpoint(checkpoint, protocol_hash)
            if cached is not None:
                payload = cached["result"]
                if payload.get("status") == "ok":
                    family_success.append(payload)
                    successful.append(payload)
                else:
                    failures.append(payload)
                continue
            log_path = (
                args.output_dir
                / "logs"
                / "selection"
                / family
                / f"{candidate_checkpoint_name(config)[:-5]}.log"
            )
            pending.append((ref, checkpoint, log_path))

        if pending:
            with ProcessPoolExecutor(
                max_workers=min(args.workers, len(pending)),
                mp_context=mp.get_context("spawn"),
            ) as executor:
                future_to_task = {
                    executor.submit(evaluate_selection_worker, ref, validation_ids, log_path):
                    (ref, checkpoint)
                    for ref, checkpoint, log_path in pending
                }
                for future in as_completed(future_to_task):
                    ref, checkpoint = future_to_task[future]
                    result = future.result()
                    if result.get("status") == "ok":
                        family_success.append(result)
                        successful.append(result)
                    else:
                        failures.append(result)
                    write_checkpoint(checkpoint, protocol_hash, {"result": result})

        family_failures = [row for row in failures if row.get("family") == family]
        if family_failures and args.max_selection_configs is None:
            raise AnalysisError(
                f"production selection for {family} has {len(family_failures)} failed candidates"
            )
        annotation_digests = {row["annotation_digest"] for row in family_success}
        if len(annotation_digests) != 1:
            raise AnalysisError(
                f"Raw candidate annotation coverage differs within {family}: "
                f"{len(annotation_digests)} distinct digests"
            )
        family_success.sort(key=lambda row: str(row["config"]))
        scored, ranges = minmax_score_candidates(family_success)
        selection_ranges[family] = ranges
        successful_by_key = {(row["family"], row["config"]): row for row in successful}
        for row in scored:
            successful_by_key[(family, row["config"])].update(
                {
                    "pa_scaled": row["pa_scaled"],
                    "pc_scaled": row["pc_scaled"],
                    "selection_score": row["selection_score"],
                    "rank": row["rank"],
                }
            )
        winner = scored[0]
        runner_up = scored[1] if len(scored) > 1 else None
        try:
            coverage = assert_fixed_config_coverage(args.sweep_dir, spec, winner["config"])
        except CoverageError as exc:
            atomic_write_json(
                args.output_dir / "coverage_failure.json",
                {"family": family, "config": winner["config"], "error": str(exc)},
            )
            raise
        for codec, ref in coverage.items():
            selected_refs[(family, codec)] = ref
        selected_rows.append(
            {
                "family": family,
                "display": spec.display,
                "config": winner["config"],
                "validation_pa_mean_nap": winner["validation_pa_mean_nap"],
                "validation_pc_mean_nap": winner["validation_pc_mean_nap"],
                "pa_scaled": winner["pa_scaled"],
                "pc_scaled": winner["pc_scaled"],
                "selection_score": winner["selection_score"],
                "runner_up_config": runner_up["config"] if runner_up else "",
                "runner_up_score": runner_up["selection_score"] if runner_up else "",
                "score_margin": winner["selection_score"] - runner_up["selection_score"] if runner_up else "",
                "candidate_count": len(scored),
                **ranges,
            }
        )

    if successful:
        successful.sort(key=lambda row: (str(row["family"]), str(row["config"])))
        flat_success = []
        for row in successful:
            flat = {key: value for key, value in row.items() if not isinstance(value, dict)}
            flat_success.append(flat)
        atomic_write_csv_rows(args.output_dir / "validation_config_scores.csv", flat_success)
    if failures:
        atomic_write_csv_rows(args.output_dir / "selection_failures.csv", failures)
    atomic_write_csv_rows(args.output_dir / "selected_configs.csv", selected_rows)

    split_by_id = {row["treatment_id"]: row["split"] for row in split_rows}
    common_manifest, coverage_rows = build_common_well_manifest(selected_refs, split_by_id)
    atomic_write_csv_rows(args.output_dir / "selected_codec_coverage.csv", coverage_rows)
    common_manifest.write_csv(args.output_dir / ".common_wells.tmp.csv")
    os.replace(args.output_dir / ".common_wells.tmp.csv", args.output_dir / "common_wells.csv")
    common_ids = set(common_manifest[ID_COL].to_list())

    final_rows: list[dict[str, Any]] = []
    pending_final: list[tuple[tuple[str, str], ProfileRef, Path, Path]] = []
    for (family, codec), ref in selected_refs.items():
        checkpoint = (
            args.output_dir
            / "checkpoints"
            / "final"
            / family
            / f"{codec}.json"
        )
        cached = load_checkpoint(checkpoint, protocol_hash)
        if cached is not None:
            final_rows.append(cached["result"])
            continue
        log_path = args.output_dir / "logs" / "final" / family / f"{codec}.log"
        pending_final.append(((family, codec), ref, checkpoint, log_path))

    if pending_final:
        with ProcessPoolExecutor(
            max_workers=min(args.workers, len(pending_final)),
            mp_context=mp.get_context("spawn"),
        ) as executor:
            future_to_task = {
                executor.submit(
                    evaluate_final_profile,
                    ref,
                    common_ids,
                    test_ids,
                    args.output_dir,
                    log_path,
                ): (key, checkpoint)
                for key, ref, checkpoint, log_path in pending_final
            }
            for future in as_completed(future_to_task):
                key, checkpoint = future_to_task[future]
                result = future.result()
                write_checkpoint(checkpoint, protocol_hash, {"result": result})
                final_rows.append(result)
    final_rows.sort(key=lambda row: (str(row["family"]), str(row["codec"])))
    atomic_write_csv_rows(args.output_dir / "heldout_test_scores.csv", final_rows)

    provenance = {
        **protocol,
        "completed_at": utc_now(),
        "initial_started_at": protocol["started_at"],
        "invocation_started_at": invocation_started_at,
        "invocation_elapsed_seconds": time.monotonic() - started,
        "split_reference": {
            "output_path": str(split_reference.output_path),
            "config_path": str(split_reference.config_path),
            "config_sha256": sha256_file(split_reference.config_path),
        },
        "split_counts": {
            "validation_treatments": len(validation_ids),
            "test_treatments": len(test_ids),
        },
        "common_wells": len(common_ids),
        "selection_ranges": selection_ranges,
        "selected_outputs": [
            {
                "family": family,
                "codec": codec,
                "config": ref.config,
                "output_path": str(ref.output_path),
                "output_size_bytes": ref.output_path.stat().st_size,
                "config_path": str(ref.config_path),
                "config_sha256": sha256_file(ref.config_path),
            }
            for (family, codec), ref in selected_refs.items()
        ],
    }
    atomic_write_json(args.output_dir / "provenance.json", provenance)
    render_report(
        args.output_dir,
        protocol,
        split_rows,
        selected_rows,
        final_rows,
        coverage_rows,
        failures,
    )
    (args.output_dir / "FAILURE.json").unlink(missing_ok=True)
    print(f"Completed analysis in {time.monotonic() - started:.1f}s")
    print(f"Report: {args.output_dir / 'REPORT.md'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            args.output_dir / "FAILURE.json",
            {"failed_at": utc_now(), "error_type": type(exc).__name__, "error": str(exc)},
        )
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
