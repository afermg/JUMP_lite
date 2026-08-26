#!/usr/bin/env python3
"""Strict split-before-fit post-processing analysis.

The treatment split is constructed before any feature or transform state is fit.
Candidate recipes are fit on Raw validation treatments plus shared controls and
scored on that same selection partition.  The winning recipe is then fit
separately for every codec, still using only validation treatments plus shared
controls, and applied to held-out test treatments.

Canonical normalized sweep profiles and archived score maps are inventory-only
and are never read by this runner.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import norm
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
NORM_SRC = REPO_ROOT / "src"
if str(NORM_SRC) not in sys.path:
    sys.path.insert(0, str(NORM_SRC))

from norm_3.io import get_numeric_features, infer_columns  # noqa: E402
from norm_3.linalg import fractional_matrix_power  # noqa: E402
from norm_3.metrics import (  # noqa: E402
    calculate_phenotypic_activity,
    calculate_phenotypic_consistency,
)

from rebuttal.postprocessing_validation.run_analysis import (  # noqa: E402
    AnalysisError,
    FAMILIES,
    GROUP_COL,
    ID_COL,
    NEGCON_COL,
    PC_GROUPS,
    PLATE_COL,
    TARGET_COL,
    COMPOUND_COL,
    atomic_write_json,
    atomic_write_pandas_csv,
    atomic_write_text,
    make_treatment_split,
    minmax_score_candidates,
    sha256_file,
)

RAW_ROOT = Path("/work/datasets/JUMP-lite-wacv/raw_features")
SWEEP_ROOT = Path("/work/datasets/JUMP-lite-wacv/sweeps/variance_first_v11_lite")
PERTURBATION_METADATA = (
    REPO_ROOT / "metadata/jump_lite_v1_perturbation_metadata.parquet"
)
REFCHEM_METADATA = REPO_ROOT / "metadata/jump_lite_v1_refchem_annotations.parquet"
ARCHIVED_LABEL_PROFILE = (
    SWEEP_ROOT
    / "dinov2_jump_lite_updated_jpegxl_lossy_raw_raw_features"
    / "robustmad_all__tvn_efaar_e0.05"
    / "output.parquet"
)
PROTOCOL_VERSION = 5
SUPPORTED_STEPS = {
    "clean_nans",
    "filter_features",
    "merge_metadata",
    "normalize_standardize",
    "normalize_robustmad",
    "inverse_normal_transform",
    "prune_correlated",
    "normalize_tvn_efaar",
    "evaluate_metrics",
}


@dataclass(frozen=True)
class EffectiveRecipe:
    family: str
    canonical_name: str
    aliases: tuple[str, ...]
    signature: str
    config: Mapping[str, Any]
    config_paths: tuple[Path, ...]


@dataclass
class StrictFitState:
    family: str
    codec: str
    recipe: str
    effective_signature: str
    input_features: list[str]
    retained_features: list[str]
    input_schema_sha256: str
    canonical_schema_sha256: str
    fit_ids_sha256: str
    split_sha256: str
    step_states: list[dict[str, Any]] = field(default_factory=list)
    fit_audit: dict[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        digest = hashlib.sha256()

        def update(value: Any) -> None:
            if isinstance(value, np.ndarray):
                digest.update(b"array\0")
                digest.update(str(value.dtype).encode())
                digest.update(str(value.shape).encode())
                digest.update(np.ascontiguousarray(value).tobytes())
            elif isinstance(value, Mapping):
                digest.update(b"mapping\0")
                for key in sorted(value, key=str):
                    digest.update(str(key).encode())
                    digest.update(b"\0")
                    update(value[key])
            elif isinstance(value, (list, tuple)):
                digest.update(b"sequence\0")
                for item in value:
                    update(item)
            elif isinstance(value, np.generic):
                update(value.item())
            else:
                digest.update(json.dumps(value, sort_keys=True).encode())
                digest.update(b"\0")

        update(
            {
                "family": self.family,
                "codec": self.codec,
                "recipe": self.recipe,
                "effective_signature": self.effective_signature,
                "input_features": self.input_features,
                "input_schema_sha256": self.input_schema_sha256,
                "canonical_schema_sha256": self.canonical_schema_sha256,
                "fit_ids_sha256": self.fit_ids_sha256,
                "split_sha256": self.split_sha256,
                "retained_features": self.retained_features,
                "step_states": self.step_states,
            }
        )
        return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_strings(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(v) for v in values):
        digest.update(value.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def canonical_feature_name(family: str, name: str) -> str:
    """Map codec-specific names onto one deterministic family feature axis."""
    if family == "openphenom":
        return re.sub(r"^openphenom_", "", name)
    return name


def canonicalize_feature_schema(
    family: str, feature_names: Sequence[str]
) -> tuple[list[str], dict[str, str]]:
    mapping = {name: canonical_feature_name(family, name) for name in feature_names}
    canonical = list(mapping.values())
    if len(canonical) != len(set(canonical)):
        raise AnalysisError(f"canonical feature collision for {family}")
    order = sorted(
        canonical,
        key=lambda x: (
            int(x.rsplit("_", 1)[1]) if x.rsplit("_", 1)[-1].isdigit() else math.inf,
            x,
        ),
    )
    inverse = {value: key for key, value in mapping.items()}
    return order, {name: inverse[name] for name in order}


def ordered_schema_sha256(feature_names: Sequence[str]) -> str:
    """Hash an ordered feature axis without treating it as an unordered set."""
    return canonical_json_sha256(list(feature_names))


def require_canonical_schema(
    family: str,
    codec: str,
    expected: Sequence[str],
    observed: Sequence[str],
) -> None:
    if list(observed) != list(expected):
        missing = [name for name in expected if name not in observed]
        extra = [name for name in observed if name not in expected]
        reordered = not missing and not extra
        raise AnalysisError(
            f"canonical Raw schema mismatch for {family}/{codec}: "
            f"missing={missing[:5]}, extra={extra[:5]}, reordered={reordered}"
        )


def inspect_raw_profile(family: str, codec: str) -> dict[str, Any]:
    """Read only parquet schema metadata and bind the physical/canonical axes."""
    path = raw_feature_path(family, codec)
    empty = pl.read_parquet(path, n_rows=0)
    source_features, _ = infer_columns(empty, ["Metadata_"])
    source_features = get_numeric_features(empty, source_features)
    canonical, _ = canonicalize_feature_schema(family, source_features)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "source_features": source_features,
        "source_schema_sha256": ordered_schema_sha256(source_features),
        "canonical_features": canonical,
        "canonical_schema_sha256": ordered_schema_sha256(canonical),
    }


def _enabled_steps(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    steps = [
        {"name": str(step["name"]), "params": step.get("params", {})}
        for step in config.get("steps", [])
        if step.get("enabled", True)
    ]
    unknown = {step["name"] for step in steps}.difference(SUPPORTED_STEPS)
    if unknown:
        raise AnalysisError(f"unsupported strict-fit steps: {sorted(unknown)}")
    return steps


def effective_recipe_signature(config: Mapping[str, Any]) -> str:
    # Resolved step behavior is authoritative.  Unused top-level sweep aliases,
    # including learned-family use_prune_correlated, are intentionally excluded.
    return canonical_json_sha256(_enabled_steps(config))


def recipe_prefix_behavior_sha256(recipe: EffectiveRecipe) -> str | None:
    """Hash enabled operations strictly before TVN EFAAR, or decline caching."""
    steps = _enabled_steps(recipe.config)
    positions = [
        index
        for index, step in enumerate(steps)
        if step["name"] == "normalize_tvn_efaar"
    ]
    if len(positions) != 1:
        return None
    return canonical_json_sha256(steps[: positions[0]])


def contiguous_prefix_groups(
    recipes: Sequence[EffectiveRecipe],
) -> list[list[EffectiveRecipe]]:
    """Group only adjacent equal prefixes, preserving canonical candidate order."""
    groups: list[list[EffectiveRecipe]] = []
    keys: list[str] = []
    for recipe in recipes:
        behavior = recipe_prefix_behavior_sha256(recipe)
        key = behavior if behavior is not None else f"uncacheable:{recipe.signature}"
        if not groups or keys[-1] != key:
            groups.append([recipe])
            keys.append(key)
        else:
            groups[-1].append(recipe)
    return groups


def prefix_cache_signature(
    *,
    prefix_behavior_sha256: str,
    family: str,
    codec: str,
    input_sha256: str,
    source_schema_sha256: str,
    canonical_schema_sha256: str,
    split_sha256: str,
    fit_ids_sha256: str,
    code_sha256: str,
    protocol_sha256: str,
) -> str:
    """Bind an in-memory prefix to every identity that permits safe reuse."""
    return canonical_json_sha256(
        {
            "prefix_behavior_sha256": prefix_behavior_sha256,
            "family": family,
            "codec": codec,
            "input_sha256": input_sha256,
            "source_schema_sha256": source_schema_sha256,
            "canonical_schema_sha256": canonical_schema_sha256,
            "split_sha256": split_sha256,
            "fit_ids_sha256": fit_ids_sha256,
            "code_sha256": code_sha256,
            "protocol_sha256": protocol_sha256,
        }
    )


def _prefix_and_suffix_recipes(
    recipe: EffectiveRecipe,
) -> tuple[EffectiveRecipe, EffectiveRecipe]:
    steps = _enabled_steps(recipe.config)
    positions = [
        index
        for index, step in enumerate(steps)
        if step["name"] == "normalize_tvn_efaar"
    ]
    if len(positions) != 1:
        raise AnalysisError("prefix caching requires exactly one TVN EFAAR step")
    split_at = positions[0]

    def subrecipe(label: str, selected: Sequence[Mapping[str, Any]]) -> EffectiveRecipe:
        config = {"steps": [dict(step) for step in selected]}
        return EffectiveRecipe(
            family=recipe.family,
            canonical_name=f"{recipe.canonical_name}__{label}",
            aliases=recipe.aliases,
            signature=effective_recipe_signature(config),
            config=config,
            config_paths=recipe.config_paths,
        )

    return subrecipe("prefix", steps[:split_at]), subrecipe("suffix", steps[split_at:])


def discover_effective_recipes(
    family: str,
    sweep_root: Path = SWEEP_ROOT,
    limit: int | None = None,
) -> list[EffectiveRecipe]:
    spec = FAMILIES[family]
    folder = sweep_root / spec.raw_folder
    if not folder.is_dir():
        raise AnalysisError(f"missing candidate folder: {folder}")
    grouped: dict[str, list[tuple[str, Path, dict[str, Any]]]] = {}
    for config_path in sorted(folder.glob("*/pipeline_config.yaml")):
        config = yaml.safe_load(config_path.read_text())
        if not isinstance(config, dict):
            raise AnalysisError(f"invalid config: {config_path}")
        signature = effective_recipe_signature(config)
        grouped.setdefault(signature, []).append(
            (config_path.parent.name, config_path, config)
        )
    recipes: list[EffectiveRecipe] = []
    for signature, rows in sorted(
        grouped.items(), key=lambda item: min(r[0] for r in item[1])
    ):
        rows.sort(key=lambda row: row[0])
        aliases = tuple(row[0] for row in rows)
        if any(effective_recipe_signature(row[2]) != signature for row in rows):
            raise AnalysisError("effective recipe grouping drift")
        recipes.append(
            EffectiveRecipe(
                family=family,
                canonical_name=aliases[0],
                aliases=aliases,
                signature=signature,
                config=rows[0][2],
                config_paths=tuple(row[1] for row in rows),
            )
        )
    if not recipes:
        raise AnalysisError(f"no effective recipes for {family}")
    return recipes[:limit] if limit is not None else recipes


def protocol_recipe_inventory(
    inventories: Mapping[str, Sequence[EffectiveRecipe]], sweep_root: Path
) -> dict[str, list[dict[str, Any]]]:
    """Bind every effective recipe, alias, codec YAML, and resolved behavior."""
    result: dict[str, list[dict[str, Any]]] = {}
    for family, recipes in inventories.items():
        family_rows: list[dict[str, Any]] = []
        for recipe in recipes:
            codec_rows: dict[str, list[dict[str, Any]]] = {}
            for codec, folder_name in FAMILIES[family].codec_folders.items():
                alias_rows: list[dict[str, Any]] = []
                for alias in recipe.aliases:
                    path = sweep_root / folder_name / alias / "pipeline_config.yaml"
                    if not path.is_file():
                        raise AnalysisError(
                            f"recipe inventory lacks {family}/{codec}/{alias}: {path}"
                        )
                    config = yaml.safe_load(path.read_text())
                    signature = effective_recipe_signature(config)
                    if signature != recipe.signature:
                        raise AnalysisError(
                            f"recipe behavior differs for {family}/{codec}/{alias}"
                        )
                    alias_rows.append(
                        {
                            "alias": alias,
                            "path": str(path.resolve()),
                            "yaml_sha256": sha256_file(path),
                            "effective_signature": signature,
                        }
                    )
                codec_rows[codec] = alias_rows
            family_rows.append(
                {
                    "canonical_name": recipe.canonical_name,
                    "aliases": list(recipe.aliases),
                    "effective_signature": recipe.signature,
                    "resolved_codecs": codec_rows,
                }
            )
        result[family] = family_rows
    return result


def raw_feature_path(family: str, codec: str) -> Path:
    stem = FAMILIES[family].codec_folders[codec]
    subdir = (
        "jump_lite" if family in {"cellprofiler", "cell_count"} else "jump_lite_cl_3"
    )
    path = RAW_ROOT / subdir / f"{stem}.parquet"
    if not path.is_file() or path.stat().st_size == 0:
        raise AnalysisError(f"missing raw feature input: {path}")
    return path


def build_annotation_table() -> pl.DataFrame:
    perturb = pl.read_parquet(PERTURBATION_METADATA)
    required = {
        "Metadata_Source",
        "Metadata_Batch",
        "Metadata_Plate",
        "Metadata_Well",
        COMPOUND_COL,
        "Metadata_pert_type",
        "Metadata_Perturbation_Type",
        "Metadata_Symbol",
        GROUP_COL,
    }
    if missing := required.difference(perturb.columns):
        raise AnalysisError(f"perturbation metadata missing {sorted(missing)}")
    refchem = pl.read_parquet(REFCHEM_METADATA)
    targets = (
        refchem.filter(
            pl.col("WithinModalityTier").is_in(["Tier0", "Tier1", "Tier2", "Tier3"])
        )
        .filter(pl.col("target").is_not_null())
        .group_by(COMPOUND_COL)
        .agg(pl.col("target").unique().sort().str.join("|").alias("_compound_target"))
    )
    perturb = (
        perturb.join(targets, on=COMPOUND_COL, how="left")
        .with_columns(
            (pl.col("Metadata_pert_type") == "negcon").alias(NEGCON_COL),
            pl.when(pl.col("Metadata_pert_type") == "negcon")
            .then(pl.lit("unknown"))
            .when(pl.col("Metadata_Perturbation_Type").is_in(["orf", "crispr"]))
            .then(pl.col("Metadata_Symbol").cast(pl.Utf8))
            .when(pl.col("_compound_target").is_not_null())
            .then(pl.col("_compound_target"))
            .otherwise(pl.lit("unknown"))
            .alias(TARGET_COL),
        )
        .drop("_compound_target")
    )
    keys = ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"]
    if perturb.select(keys).n_unique() != len(perturb):
        raise AnalysisError("annotation table has duplicate physical wells")
    return perturb


def validate_projected_target_labels(
    annotations: pl.DataFrame, archived_profile: Path = ARCHIVED_LABEL_PROFILE
) -> dict[str, Any]:
    """Check labels only; no archived feature or score column is projected."""
    if not archived_profile.is_file():
        raise AnalysisError(
            f"missing canonical archived label profile: {archived_profile}"
        )
    projected = pl.read_parquet(
        archived_profile, columns=[ID_COL, TARGET_COL]
    ).with_columns(pl.col(ID_COL).cast(pl.Utf8), pl.col(TARGET_COL).cast(pl.Utf8))
    annotation_ids = annotations.select(
        pl.col("Metadata_Well_Key").cast(pl.Utf8).alias(ID_COL),
        pl.col(TARGET_COL).cast(pl.Utf8).alias("_projected_target"),
    )
    if annotation_ids[ID_COL].n_unique() != len(annotation_ids):
        raise AnalysisError("projected annotation labels have duplicate well IDs")
    compared = projected.join(annotation_ids, on=ID_COL, how="left")
    missing = compared.filter(pl.col("_projected_target").is_null()).height
    mismatches = compared.filter(
        pl.col("_projected_target") != pl.col(TARGET_COL)
    ).height
    if len(compared) != len(projected) or missing or mismatches:
        raise AnalysisError(
            "projected target-label parity failed: "
            f"rows={len(compared)}/{len(projected)}, missing={missing}, "
            f"mismatches={mismatches}"
        )
    labels = compared.select(ID_COL, TARGET_COL).sort(ID_COL)
    return {
        "archived_profile": str(archived_profile),
        "columns_read": [ID_COL, TARGET_COL],
        "rows": len(labels),
        "mismatches": 0,
        "projected_labels_sha256": canonical_json_sha256(labels.to_dicts()),
    }


def load_raw_profile(
    family: str,
    codec: str,
    annotations: pl.DataFrame,
    expected_canonical_schema: Sequence[str] | None = None,
    expected_source_schema_sha256: str | None = None,
) -> tuple[pl.DataFrame, list[str], Path]:
    path = raw_feature_path(family, codec)
    frame = pl.read_parquet(path)
    raw_features, _ = infer_columns(frame, ["Metadata_"])
    raw_features = get_numeric_features(frame, raw_features)
    source_schema_sha256 = ordered_schema_sha256(raw_features)
    canonical, source_by_canonical = canonicalize_feature_schema(family, raw_features)
    if expected_source_schema_sha256 is not None:
        if source_schema_sha256 != expected_source_schema_sha256:
            raise AnalysisError(f"source feature schema drift for {family}/{codec}")
    if expected_canonical_schema is not None:
        require_canonical_schema(family, codec, expected_canonical_schema, canonical)
    frame = frame.rename(
        {
            source: canonical_name
            for canonical_name, source in source_by_canonical.items()
        }
    )
    keys = ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"]
    frame = frame.with_columns([pl.col(key).cast(pl.Utf8) for key in keys])
    annotations = annotations.with_columns([pl.col(key).cast(pl.Utf8) for key in keys])
    overlapping = [
        column
        for column in annotations.columns
        if column in frame.columns and column not in keys
    ]
    if overlapping:
        frame = frame.drop(overlapping)
    frame = frame.join(annotations, on=keys, how="inner")
    if ID_COL not in frame.columns:
        frame = frame.with_columns(
            pl.concat_str([pl.col(key) for key in keys], separator="__").alias(ID_COL)
        )
    frame = frame.with_columns(
        pl.col(ID_COL).cast(pl.Utf8),
        pl.col(COMPOUND_COL).cast(pl.Utf8),
        pl.col(GROUP_COL).cast(pl.Utf8),
        pl.col(NEGCON_COL).cast(pl.Boolean),
    )
    if frame[ID_COL].n_unique() != len(frame):
        raise AnalysisError(f"nonunique Metadata_id in {path}")
    if frame.is_empty():
        raise AnalysisError(f"metadata join empty for {path}")
    return frame, canonical, path


def make_split_from_annotations(
    annotations: pl.DataFrame, fraction: float = 0.2, seed: int = 20260811
) -> list[dict[str, Any]]:
    memberships: dict[str, set[str]] = {}
    for treatment, group in (
        annotations.filter(~pl.col(NEGCON_COL))
        .select(COMPOUND_COL, GROUP_COL)
        .unique()
        .iter_rows()
    ):
        memberships.setdefault(str(treatment), set()).add(str(group))
    return make_treatment_split(memberships, fraction, seed)


def role_masks(
    frame: pl.DataFrame, split_by_id: Mapping[str, str]
) -> dict[str, np.ndarray]:
    negcon = frame[NEGCON_COL].to_numpy().astype(bool)
    compounds = frame[COMPOUND_COL].to_list()
    validation = np.array(
        [
            not negcon[i] and split_by_id.get(str(value)) == "validation"
            for i, value in enumerate(compounds)
        ]
    )
    test = np.array(
        [
            not negcon[i] and split_by_id.get(str(value)) == "test"
            for i, value in enumerate(compounds)
        ]
    )
    if np.any(~negcon & ~validation & ~test):
        raise AnalysisError("non-control rows are absent from split")
    return {"control": negcon, "validation": validation, "test": test}


def exact_fit_ids_hash(
    frame: pl.DataFrame,
    feature_names: Sequence[str],
    recipe: EffectiveRecipe,
    split_by_id: Mapping[str, str],
) -> str:
    """Hash the rows actually available to fit after fit-only missingness rules."""
    cp = _gpu()
    roles = role_masks(frame, split_by_id)
    permitted = roles["control"] | roles["validation"]
    clean_steps = [
        step for step in _enabled_steps(recipe.config) if step["name"] == "clean_nans"
    ]
    if len(clean_steps) != 1:
        raise AnalysisError("strict recipe must have exactly one clean_nans step")
    X = cp.asarray(frame.select(feature_names).to_numpy(), dtype=cp.float64)
    keep = _feature_validity_fit(
        X,
        permitted,
        float(clean_steps[0]["params"].get("na_cutoff", 0.3)),
        cp,
    )
    finite_rows = cp.asnumpy(cp.isfinite(X[:, cp.asarray(keep)]).all(axis=1)).astype(
        bool
    )
    exact_mask = permitted & finite_rows
    if not exact_mask.any():
        raise AnalysisError("missingness processing removed all permitted fit rows")
    return sha256_strings(frame.filter(pl.Series(exact_mask))[ID_COL].to_list())


def _gpu() -> Any:
    try:
        import cupy as cp
    except ImportError as exc:
        raise AnalysisError("CuPy is required for fitted numerical transforms") from exc
    count = cp.cuda.runtime.getDeviceCount()
    configured = os.environ.get("JUMP_LITE_STRICT_GPU_INDEX")
    if configured is not None:
        try:
            index = int(configured)
        except ValueError as exc:
            raise AnalysisError("invalid strict Hydra GPU index") from exc
        if index < 0 or index >= count:
            raise AnalysisError(
                f"strict Hydra GPU index {index} outside {count} visible devices"
            )
        cp.cuda.Device(index).use()
        return cp
    if count != 1:
        # The serial runner must expose exactly one GPU. Hydra workers instead
        # bind one explicit visible-device index from the immutable task manifest.
        raise AnalysisError("strict runner requires exactly one visible GPU")
    return cp


def _feature_validity_fit(
    X: Any, fit_mask: np.ndarray, cutoff: float, cp: Any
) -> np.ndarray:
    fit = X[cp.asarray(fit_mask)]
    invalid = ~cp.isfinite(fit)
    return cp.asnumpy(invalid.mean(axis=0) <= cutoff).astype(bool)


def _validate_cpu_workers(cpu_workers: int) -> None:
    if cpu_workers < 1:
        raise AnalysisError("cpu_workers must be at least 1")


def _feature_chunks(n_features: int, cpu_workers: int) -> list[tuple[int, int]]:
    """Return bounded contiguous chunks in canonical feature order."""
    _validate_cpu_workers(cpu_workers)
    if n_features == 0:
        return []
    chunk_size = math.ceil(n_features / min(cpu_workers, n_features))
    return [
        (start, min(start + chunk_size, n_features))
        for start in range(0, n_features, chunk_size)
    ]


def _variance_keep(
    X: Any,
    fit_mask: np.ndarray,
    freq_cut: float,
    unique_cut: float,
    cp: Any,
    cpu_workers: int = 1,
) -> np.ndarray:
    fit = cp.asnumpy(X[cp.asarray(fit_mask)])
    keep = np.zeros(fit.shape[1], dtype=bool)
    n = len(fit)

    def evaluate_chunk(bounds: tuple[int, int]) -> tuple[int, np.ndarray]:
        start, stop = bounds
        # Feature-major storage makes each stable per-feature sort contiguous and
        # lets NumPy release the GIL for substantial work rather than scalar tasks.
        sorted_chunk = np.array(fit[:, start:stop], order="F", copy=True)
        sorted_chunk.sort(axis=0, kind="mergesort")
        chunk_keep = np.zeros(stop - start, dtype=bool)
        for offset in range(stop - start):
            values = sorted_chunk[:, offset]
            finite = np.isfinite(values)
            if not finite.all():
                values = values[finite]
            if len(values) == 0:
                continue
            boundaries = np.flatnonzero(values[1:] != values[:-1]) + 1
            counts = np.diff(np.concatenate(([0], boundaries, [len(values)])))
            if len(counts) / n < unique_cut:
                continue
            if len(counts) >= 2:
                counts.sort()
                if counts[-2] / counts[-1] < freq_cut:
                    continue
            chunk_keep[offset] = True
        return start, chunk_keep

    chunks = _feature_chunks(fit.shape[1], cpu_workers)
    if cpu_workers == 1:
        for i in range(fit.shape[1]):
            values = fit[:, i]
            values = values[np.isfinite(values)]
            if len(values) == 0:
                continue
            _, counts = np.unique(values, return_counts=True)
            if len(counts) / n < unique_cut:
                continue
            if len(counts) >= 2:
                counts.sort()
                if counts[-2] / counts[-1] < freq_cut:
                    continue
            keep[i] = True
        return keep
    else:
        with ThreadPoolExecutor(max_workers=cpu_workers) as executor:
            results = executor.map(evaluate_chunk, chunks)
    for start, chunk_keep in results:
        keep[start : start + len(chunk_keep)] = chunk_keep
    return keep


def _fit_plate_scalers(
    X: Any,
    plates: np.ndarray,
    fit_mask: np.ndarray,
    method: str,
    epsilon: float,
    cp: Any,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for plate in sorted(set(plates.astype(str))):
        mask = fit_mask & (plates.astype(str) == plate)
        if not mask.any():
            raise AnalysisError(f"no permitted fit rows for plate {plate}")
        fit = X[cp.asarray(mask)]
        if method == "standardize":
            center = cp.mean(fit, axis=0)
            scale = cp.std(fit, axis=0) + 1e-18
        elif method == "robustmad":
            center = cp.median(fit, axis=0)
            scale = cp.median(cp.abs(fit - center), axis=0) + epsilon
        else:
            raise AnalysisError(f"unsupported scaler {method}")
        result[plate] = (cp.asnumpy(center), cp.asnumpy(scale))
    return result


def _apply_plate_scalers(
    X: Any,
    plates: np.ndarray,
    state: Mapping[str, tuple[np.ndarray, np.ndarray]],
    cp: Any,
) -> Any:
    output = cp.empty_like(X)
    plate_strings = plates.astype(str)
    unseen = sorted(set(plate_strings).difference(state))
    if unseen:
        raise AnalysisError(f"unseen transform plate strata: {unseen[:5]}")
    for plate, (center, scale) in state.items():
        mask = plate_strings == plate
        output[cp.asarray(mask)] = (
            X[cp.asarray(mask)] - cp.asarray(center)
        ) / cp.asarray(scale)
    return output


def fit_empirical_int(X_fit: np.ndarray, cpu_workers: int = 1) -> list[np.ndarray]:
    def fit_chunk(bounds: tuple[int, int]) -> tuple[int, list[np.ndarray]]:
        start, stop = bounds
        sorted_chunk = np.array(X_fit[:, start:stop], order="F", copy=True)
        sorted_chunk.sort(axis=0, kind="mergesort")
        return start, [sorted_chunk[:, i] for i in range(stop - start)]

    if cpu_workers == 1:
        return [np.sort(X_fit[:, i], kind="mergesort") for i in range(X_fit.shape[1])]
    chunks = _feature_chunks(X_fit.shape[1], cpu_workers)
    with ThreadPoolExecutor(max_workers=cpu_workers) as executor:
        results = executor.map(fit_chunk, chunks)
    sorted_fit: list[np.ndarray] = []
    for start, arrays in results:
        if start != len(sorted_fit):
            raise AnalysisError("INT fit chunks returned out of canonical order")
        sorted_fit.extend(arrays)
    return sorted_fit


def transform_empirical_int(
    X: np.ndarray,
    sorted_fit: Sequence[np.ndarray],
    cpu_workers: int = 1,
) -> np.ndarray:
    if X.shape[1] != len(sorted_fit):
        raise AnalysisError("INT fit distribution count does not match feature count")

    def transform_chunk(bounds: tuple[int, int]) -> tuple[int, np.ndarray]:
        start, stop = bounds
        chunk = np.empty((X.shape[0], stop - start), dtype=np.float64)
        for offset, i in enumerate(range(start, stop)):
            reference = sorted_fit[i]
            if len(reference) == 0 or not np.isfinite(reference).all():
                raise AnalysisError("invalid INT fit distribution")
            values = X[:, i]
            left = np.searchsorted(reference, values, side="left")
            right = np.searchsorted(reference, values, side="right")
            rank = np.clip((left + right + 1.0) / 2.0, 1.0, float(len(reference)))
            quantile = (rank - 0.375) / (len(reference) + 0.25)
            chunk[:, offset] = norm.ppf(quantile)
        return start, chunk

    output = np.empty_like(X, dtype=np.float64)
    if cpu_workers == 1:
        for i, reference in enumerate(sorted_fit):
            if len(reference) == 0 or not np.isfinite(reference).all():
                raise AnalysisError("invalid INT fit distribution")
            values = X[:, i]
            left = np.searchsorted(reference, values, side="left")
            right = np.searchsorted(reference, values, side="right")
            rank = np.clip((left + right + 1.0) / 2.0, 1.0, float(len(reference)))
            quantile = (rank - 0.375) / (len(reference) + 0.25)
            output[:, i] = norm.ppf(quantile)
        if not np.isfinite(output).all():
            raise AnalysisError("nonfinite out-of-sample INT output")
        return output
    chunks = _feature_chunks(X.shape[1], cpu_workers)
    with ThreadPoolExecutor(max_workers=cpu_workers) as executor:
        results = executor.map(transform_chunk, chunks)
    for start, chunk in results:
        output[:, start : start + chunk.shape[1]] = chunk
    if not np.isfinite(output).all():
        raise AnalysisError("nonfinite out-of-sample INT output")
    return output


def _correlation_keep(
    X: Any, fit_mask: np.ndarray, threshold: float, cp: Any
) -> np.ndarray:
    fit = X[cp.asarray(fit_mask)]
    corr = cp.nan_to_num(cp.corrcoef(fit, rowvar=False))
    adjacency = cp.asnumpy(cp.abs(corr) > threshold)
    np.fill_diagonal(adjacency, False)
    remaining = set(range(adjacency.shape[0]))
    independent: list[int] = []
    while remaining:
        degrees = adjacency.sum(axis=1)
        node = min(remaining, key=lambda index: (degrees[index], index))
        independent.append(node)
        neighbors = set(np.flatnonzero(adjacency[node]))
        remaining -= neighbors | {node}
        adjacency[node, :] = False
        adjacency[:, node] = False
        for neighbor in neighbors:
            adjacency[neighbor, :] = False
            adjacency[:, neighbor] = False
    keep = np.zeros(fit.shape[1], dtype=bool)
    keep[independent] = True
    return keep


def _fit_tvn_efaar(
    X: Any,
    controls: np.ndarray,
    batches: np.ndarray,
    params: Mapping[str, Any],
    cp: Any,
) -> tuple[Any, dict[str, Any]]:
    from cuml.decomposition import PCA
    from cuml.preprocessing import StandardScaler

    controls_gpu = cp.asarray(controls)
    if int(controls_gpu.sum()) < 2:
        raise AnalysisError("TVN EFAAR requires shared controls")
    global_scaler = StandardScaler().fit(X[controls_gpu])
    scaled = global_scaler.transform(X)
    batch_strings = batches.astype(str)
    control_counts = {
        batch: int(np.sum(controls & (batch_strings == batch)))
        for batch in sorted(set(batch_strings))
    }
    if any(count < 2 for count in control_counts.values()):
        bad = {batch: count for batch, count in control_counts.items() if count < 2}
        raise AnalysisError(f"batch lacks TVN controls: {bad}")
    requested = int(params.get("n_components", 128))
    min_controls = min(control_counts.values())
    components = min(requested, X.shape[1], int(controls.sum()) - 1)
    threshold = float(params.get("dim_ratio_threshold", 2.5))
    if components / min_controls > threshold:
        components = min_controls - 1
    if components < 1:
        raise AnalysisError("TVN dimension clamp produced no dimensions")
    pca = PCA(n_components=components, whiten=False).fit(scaled[controls_gpu])
    projected = pca.transform(scaled)
    batch_scalers: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    standardized = cp.empty_like(projected)
    for batch in sorted(control_counts):
        batch_mask = batch_strings == batch
        control_mask = controls & batch_mask
        scaler = StandardScaler().fit(projected[cp.asarray(control_mask)])
        standardized[cp.asarray(batch_mask)] = scaler.transform(
            projected[cp.asarray(batch_mask)]
        )
        batch_scalers[batch] = (
            cp.asnumpy(scaler.mean_),
            cp.asnumpy(scaler.scale_),
        )
    epsilon = float(params.get("epsilon", 0.5))
    controls_std = standardized[controls_gpu]
    target_cov = cp.cov(controls_std.T) + epsilon * cp.eye(components)
    target_sqrt = fractional_matrix_power(target_cov, 0.5).real
    source_inv: dict[str, np.ndarray] = {}
    transformed = cp.empty_like(standardized)
    for batch in sorted(control_counts):
        batch_mask = batch_strings == batch
        control_mask = controls & batch_mask
        source_cov = cp.cov(
            standardized[cp.asarray(control_mask)].T
        ) + epsilon * cp.eye(components)
        inv = fractional_matrix_power(source_cov, -0.5).real
        transformed[cp.asarray(batch_mask)] = (
            standardized[cp.asarray(batch_mask)] @ inv @ target_sqrt
        )
        source_inv[batch] = cp.asnumpy(inv)
    state = {
        "name": "normalize_tvn_efaar",
        "params": dict(params),
        "global_mean": cp.asnumpy(global_scaler.mean_),
        "global_scale": cp.asnumpy(global_scaler.scale_),
        "pca_mean": cp.asnumpy(pca.mean_),
        "pca_components": cp.asnumpy(pca.components_),
        "batch_scalers": batch_scalers,
        "target_sqrt": cp.asnumpy(target_sqrt),
        "source_inv": source_inv,
        "output_dimensions": components,
        "control_counts": control_counts,
    }
    return transformed, state


def _apply_tvn_efaar(
    X: Any, batches: np.ndarray, state: Mapping[str, Any], cp: Any
) -> Any:
    scaled = (X - cp.asarray(state["global_mean"])) / cp.asarray(state["global_scale"])
    projected = (scaled - cp.asarray(state["pca_mean"])) @ cp.asarray(
        state["pca_components"]
    ).T
    batch_strings = batches.astype(str)
    known = set(state["batch_scalers"])
    unseen = sorted(set(batch_strings).difference(known))
    if unseen:
        raise AnalysisError(f"unseen TVN batch strata: {unseen[:5]}")
    output = cp.empty_like(projected)
    target = cp.asarray(state["target_sqrt"])
    for batch in sorted(known):
        mask = batch_strings == batch
        center, scale = state["batch_scalers"][batch]
        standardized = (projected[cp.asarray(mask)] - cp.asarray(center)) / cp.asarray(
            scale
        )
        output[cp.asarray(mask)] = (
            standardized @ cp.asarray(state["source_inv"][batch]) @ target
        )
    return output


def fit_transform_recipe(
    frame: pl.DataFrame,
    feature_names: Sequence[str],
    recipe: EffectiveRecipe,
    split_by_id: Mapping[str, str],
    codec: str,
    cpu_workers: int = 1,
) -> tuple[pl.DataFrame, StrictFitState]:
    _validate_cpu_workers(cpu_workers)
    cp = _gpu()
    roles = role_masks(frame, split_by_id)
    permitted = roles["control"] | roles["validation"]
    if (
        not roles["control"].any()
        or not roles["validation"].any()
        or not roles["test"].any()
    ):
        raise AnalysisError("strict split roles are incomplete")
    fit_ids = frame.filter(pl.Series(permitted))[ID_COL].to_list()
    split_hash = canonical_json_sha256(
        sorted((str(k), str(v)) for k, v in split_by_id.items())
    )
    schema_hash = ordered_schema_sha256(feature_names)
    state = StrictFitState(
        family=recipe.family,
        codec=codec,
        recipe=recipe.canonical_name,
        effective_signature=recipe.signature,
        input_features=list(feature_names),
        retained_features=list(feature_names),
        input_schema_sha256=schema_hash,
        canonical_schema_sha256=schema_hash,
        fit_ids_sha256=sha256_strings(fit_ids),
        split_sha256=split_hash,
    )
    X = cp.asarray(frame.select(feature_names).to_numpy(), dtype=cp.float64)
    plates = frame[PLATE_COL].to_numpy()
    batches = frame["Metadata_Batch"].to_numpy()
    current = list(feature_names)
    steps = _enabled_steps(recipe.config)
    for step in steps:
        name, params = step["name"], dict(step["params"])
        if name in {"merge_metadata", "evaluate_metrics"}:
            continue
        if name == "clean_nans":
            keep = _feature_validity_fit(
                X, permitted, float(params.get("na_cutoff", 0.3)), cp
            )
            X, current = (
                X[:, cp.asarray(keep)],
                [f for f, flag in zip(current, keep, strict=True) if flag],
            )
            finite_rows = cp.asnumpy(cp.isfinite(X).all(axis=1)).astype(bool)
            if not finite_rows.all():
                X = X[cp.asarray(finite_rows)]
                frame = frame.filter(pl.Series(finite_rows))
                roles = {role: mask[finite_rows] for role, mask in roles.items()}
                permitted = roles["control"] | roles["validation"]
                plates = plates[finite_rows]
                batches = batches[finite_rows]
            if (
                not permitted.any()
                or not roles["control"].any()
                or not roles["test"].any()
            ):
                raise AnalysisError("nonfinite-row removal emptied a strict role")
            state.step_states.append({"name": name, "params": params, "keep": keep})
        elif name == "filter_features":
            operations = params.get("filters", [])
            if len(operations) != 1:
                raise AnalysisError(
                    "strict filter_features expects one resolved operation per step"
                )
            keep = np.ones(len(current), dtype=bool)
            for operation in operations:
                if operation["name"] == "variance_threshold":
                    keep = _variance_keep(
                        X,
                        permitted,
                        float(operation.get("freq_cut", 0.05)),
                        float(operation.get("unique_cut", 0.01)),
                        cp,
                        cpu_workers,
                    )
                elif operation["name"] == "drop_outliers":
                    fit = X[cp.asarray(permitted)]
                    z = (fit - cp.mean(fit, axis=0)) / (cp.std(fit, axis=0) + 1e-8)
                    keep = cp.asnumpy(
                        cp.max(cp.abs(z), axis=0)
                        <= float(operation.get("outlier_cutoff", 500))
                    )
                else:
                    raise AnalysisError(
                        f"unsupported strict feature filter: {operation['name']}"
                    )
                X, current = (
                    X[:, cp.asarray(keep)],
                    [f for f, flag in zip(current, keep, strict=True) if flag],
                )
            state.step_states.append({"name": name, "params": params, "keep": keep})
        elif name in {"normalize_standardize", "normalize_robustmad"}:
            fit_mask = (
                roles["control"] if params.get("fit_on_controls", False) else permitted
            )
            method = "standardize" if name.endswith("standardize") else "robustmad"
            epsilon = float(params.get("epsilon", 1e-18))
            scalers = _fit_plate_scalers(X, plates, fit_mask, method, epsilon, cp)
            X = _apply_plate_scalers(X, plates, scalers, cp)
            state.step_states.append(
                {"name": name, "params": params, "plate_scalers": scalers}
            )
        elif name == "inverse_normal_transform":
            sorted_fit = fit_empirical_int(
                cp.asnumpy(X[cp.asarray(permitted)]), cpu_workers
            )
            X = cp.asarray(
                transform_empirical_int(cp.asnumpy(X), sorted_fit, cpu_workers)
            )
            state.step_states.append(
                {
                    "name": name,
                    "params": params,
                    "sorted_fit": {str(i): arr for i, arr in enumerate(sorted_fit)},
                }
            )
        elif name == "prune_correlated":
            keep = _correlation_keep(
                X, permitted, float(params.get("threshold", 0.9)), cp
            )
            X, current = (
                X[:, cp.asarray(keep)],
                [f for f, flag in zip(current, keep, strict=True) if flag],
            )
            state.step_states.append({"name": name, "params": params, "keep": keep})
        elif name == "normalize_tvn_efaar":
            X, tvn_state = _fit_tvn_efaar(X, roles["control"], batches, params, cp)
            current = [f"PC_{i}" for i in range(X.shape[1])]
            state.step_states.append(tvn_state)
        else:
            raise AnalysisError(f"unsupported strict-fit step: {name}")
        if X.shape[1] == 0 or not bool(cp.isfinite(X).all()):
            raise AnalysisError(f"invalid output after strict step {name}")
    state.retained_features = current
    state.fit_ids_sha256 = sha256_strings(
        frame.filter(pl.Series(permitted))[ID_COL].to_list()
    )
    metadata = [column for column in frame.columns if column not in feature_names]
    result = frame.select(metadata)
    X_cpu = cp.asnumpy(X)
    result = result.with_columns(
        [
            pl.Series(name=feature, values=X_cpu[:, i])
            for i, feature in enumerate(current)
        ]
    )
    state.fit_audit = {
        "fit_rows": int(permitted.sum()),
        "validation_rows": int(roles["validation"].sum()),
        "control_rows": int(roles["control"].sum()),
        "test_rows_excluded_from_fit": int(roles["test"].sum()),
        "fit_treatments": int(
            frame.filter(pl.Series(roles["validation"]))[COMPOUND_COL].n_unique()
        ),
        "plates": int(frame.filter(pl.Series(permitted))[PLATE_COL].n_unique()),
        "batches": int(frame.filter(pl.Series(permitted))["Metadata_Batch"].n_unique()),
        "output_features": len(current),
    }
    return result, state


def fit_recipe_prefix(
    frame: pl.DataFrame,
    feature_names: Sequence[str],
    recipe: EffectiveRecipe,
    split_by_id: Mapping[str, str],
    codec: str,
    cpu_workers: int = 1,
) -> tuple[pl.DataFrame, StrictFitState]:
    prefix_recipe, _ = _prefix_and_suffix_recipes(recipe)
    return fit_transform_recipe(
        frame,
        feature_names,
        prefix_recipe,
        split_by_id,
        codec,
        cpu_workers,
    )


def complete_recipe_from_prefix(
    prefix_frame: pl.DataFrame,
    prefix_state: StrictFitState,
    recipe: EffectiveRecipe,
    split_by_id: Mapping[str, str],
    cpu_workers: int = 1,
) -> tuple[pl.DataFrame, StrictFitState]:
    """Fit the recipe suffix and reconstruct the exact canonical full state."""
    _, suffix_recipe = _prefix_and_suffix_recipes(recipe)
    transformed, suffix_state = fit_transform_recipe(
        prefix_frame,
        prefix_state.retained_features,
        suffix_recipe,
        split_by_id,
        prefix_state.codec,
        cpu_workers,
    )
    if suffix_state.fit_ids_sha256 != prefix_state.fit_ids_sha256:
        raise AnalysisError("prefix and suffix fit populations differ")
    state = StrictFitState(
        family=recipe.family,
        codec=prefix_state.codec,
        recipe=recipe.canonical_name,
        effective_signature=recipe.signature,
        input_features=list(prefix_state.input_features),
        retained_features=list(suffix_state.retained_features),
        input_schema_sha256=prefix_state.input_schema_sha256,
        canonical_schema_sha256=prefix_state.canonical_schema_sha256,
        fit_ids_sha256=prefix_state.fit_ids_sha256,
        split_sha256=prefix_state.split_sha256,
        step_states=[*prefix_state.step_states, *suffix_state.step_states],
        fit_audit=dict(suffix_state.fit_audit),
    )
    return transformed, state


def release_prefix_cache() -> None:
    """Collect dropped host prefixes and release unused CuPy pool blocks."""
    gc.collect()
    try:
        cp = _gpu()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except AnalysisError:
        pass


def apply_fitted_recipe(
    frame: pl.DataFrame,
    feature_names: Sequence[str],
    state: StrictFitState,
    cpu_workers: int = 1,
) -> pl.DataFrame:
    _validate_cpu_workers(cpu_workers)
    cp = _gpu()
    X = cp.asarray(frame.select(feature_names).to_numpy(), dtype=cp.float64)
    current = list(feature_names)
    plates = frame[PLATE_COL].to_numpy()
    batches = frame["Metadata_Batch"].to_numpy()
    for step in state.step_states:
        name = step["name"]
        if name in {"clean_nans", "filter_features", "prune_correlated"}:
            keep = np.asarray(step["keep"], dtype=bool)
            X, current = (
                X[:, cp.asarray(keep)],
                [f for f, flag in zip(current, keep, strict=True) if flag],
            )
            if name == "clean_nans":
                finite_rows = cp.asnumpy(cp.isfinite(X).all(axis=1)).astype(bool)
                X = X[cp.asarray(finite_rows)]
                frame = frame.filter(pl.Series(finite_rows))
                plates = plates[finite_rows]
                batches = batches[finite_rows]
        elif name in {"normalize_standardize", "normalize_robustmad"}:
            X = _apply_plate_scalers(X, plates, step["plate_scalers"], cp)
        elif name == "inverse_normal_transform":
            refs = [step["sorted_fit"][str(i)] for i in range(len(current))]
            X = cp.asarray(transform_empirical_int(cp.asnumpy(X), refs, cpu_workers))
        elif name == "normalize_tvn_efaar":
            X = _apply_tvn_efaar(X, batches, step, cp)
            current = [f"PC_{i}" for i in range(X.shape[1])]
        else:
            raise AnalysisError(f"unknown fit state step {name}")
    if not bool(cp.isfinite(X).all()):
        raise AnalysisError("nonfinite strict transform output")
    metadata = [column for column in frame.columns if column not in feature_names]
    result = frame.select(metadata)
    values = cp.asnumpy(X)
    return result.with_columns(
        [
            pl.Series(name=feature, values=values[:, i])
            for i, feature in enumerate(current)
        ]
    )


def score_partition(
    transformed: pl.DataFrame,
    features: Sequence[str],
    treatment_ids: set[str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    frame = transformed.filter(
        pl.col(NEGCON_COL) | pl.col(COMPOUND_COL).is_in(sorted(treatment_ids))
    )
    if frame.filter(pl.col(NEGCON_COL)).is_empty():
        raise AnalysisError("score partition has no controls")
    required = frame.filter(~pl.col(NEGCON_COL)).select(PLATE_COL, GROUP_COL).unique()
    available = frame.filter(pl.col(NEGCON_COL)).select(PLATE_COL, GROUP_COL).unique()
    missing = required.join(available, on=[PLATE_COL, GROUP_COL], how="anti")
    if len(missing):
        raise AnalysisError(
            f"score partition lacks plate/group controls: {missing.head(5).to_dicts()}"
        )
    distance = "euclidean" if len(features) <= 2 else "cosine"
    pa = calculate_phenotypic_activity(
        frame,
        list(features),
        compound_col=COMPOUND_COL,
        negcon_col=NEGCON_COL,
        batch_col=PLATE_COL,
        group_col=GROUP_COL,
        distance=distance,
    )
    pc = calculate_phenotypic_consistency(
        frame,
        list(features),
        compound_col=COMPOUND_COL,
        target_col=TARGET_COL,
        negcon_col=NEGCON_COL,
        group_col=GROUP_COL,
        pc_groups=list(PC_GROUPS),
        distance=distance,
    )
    pa_value = float(pa["mean_normalized_average_precision"])
    pc_value = float(pc["mean_normalized_average_precision"])
    if not all(map(math.isfinite, [pa_value, pc_value])):
        raise AnalysisError("nonfinite strict PA/PC")
    return (
        {
            "pa_mean_nap": pa_value,
            "pc_mean_nap": pc_value,
            "balanced_nap_product": pa_value * pc_value,
            "wells": len(frame),
            "controls": frame.filter(pl.col(NEGCON_COL)).height,
            "treatments": frame.filter(~pl.col(NEGCON_COL))[COMPOUND_COL].n_unique(),
            "pa_units": int(pa["n_compounds"]),
            "pc_targets": int(pc["n_targets_total"]),
            "features": len(features),
        },
        pa["activity_map"],
        pc["target_consistency"],
    )


def ensure_create_only(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=False)


def atomic_create_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create an immutable JSON package without replacement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp_path, path)
        except FileExistsError as exc:
            raise AnalysisError(
                f"refusing to replace immutable package: {path}"
            ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)


def candidate_cache_key(
    input_hash: str,
    split_hash: str,
    fit_ids_hash: str,
    code_hash: str,
    family: str,
    codec: str,
    config: str,
    recipe_signature: str,
    source_schema_hash: str,
    canonical_schema_hash: str,
    prefix_signature: str = "",
) -> str:
    identity = checkpoint_identity(
        code_sha256=code_hash,
        family=family,
        codec=codec,
        config=config,
        effective_signature=recipe_signature,
        input_sha256=input_hash,
        fit_ids_sha256=fit_ids_hash,
        split_sha256=split_hash,
        source_schema_sha256=source_schema_hash,
        canonical_schema_sha256=canonical_schema_hash,
    )
    identity["prefix_signature"] = prefix_signature
    return canonical_json_sha256(identity)


def recipe_for_codec(
    recipe: EffectiveRecipe,
    codec: str,
    sweep_root: Path,
) -> EffectiveRecipe:
    folder = sweep_root / FAMILIES[recipe.family].codec_folders[codec]
    rows: list[tuple[str, Path, dict[str, Any]]] = []
    for alias in recipe.aliases:
        path = folder / alias / "pipeline_config.yaml"
        if not path.is_file():
            raise AnalysisError(
                f"selected effective recipe lacks exact {codec} alias {alias}: {path}"
            )
        config = yaml.safe_load(path.read_text())
        if effective_recipe_signature(config) != recipe.signature:
            raise AnalysisError(
                f"selected recipe behavior differs for {recipe.family}/{codec}/{alias}"
            )
        rows.append((alias, path, config))
    return EffectiveRecipe(
        family=recipe.family,
        canonical_name=recipe.canonical_name,
        aliases=recipe.aliases,
        signature=recipe.signature,
        config=rows[0][2],
        config_paths=tuple(row[1] for row in rows),
    )


def checkpoint_identity(
    *,
    code_sha256: str,
    family: str,
    codec: str,
    config: str,
    effective_signature: str,
    input_sha256: str,
    fit_ids_sha256: str,
    split_sha256: str,
    source_schema_sha256: str,
    canonical_schema_sha256: str,
) -> dict[str, str]:
    return {
        "code_sha256": code_sha256,
        "family": family,
        "codec": codec,
        "config": config,
        "effective_signature": effective_signature,
        "input_sha256": input_sha256,
        "fit_ids_sha256": fit_ids_sha256,
        "split_sha256": split_sha256,
        "source_schema_sha256": source_schema_sha256,
        "canonical_schema_sha256": canonical_schema_sha256,
    }


def _checkpoint(
    path: Path,
    protocol_hash: str,
    payload: Mapping[str, Any],
    identity: Mapping[str, str] | None = None,
) -> None:
    atomic_write_json(
        path,
        {
            "protocol_hash": protocol_hash,
            "identity": dict(identity or {}),
            "identity_sha256": canonical_json_sha256(dict(identity or {})),
            "completed_at": utc_now(),
            **dict(payload),
        },
    )


def _load_checkpoint(
    path: Path,
    protocol_hash: str,
    expected_identity: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if payload.get("protocol_hash") != protocol_hash:
        raise AnalysisError(f"checkpoint protocol mismatch: {path}")
    observed_identity = payload.get("identity")
    if observed_identity != dict(expected_identity or {}):
        raise AnalysisError(f"checkpoint identity mismatch: {path}")
    if payload.get("identity_sha256") != canonical_json_sha256(observed_identity):
        raise AnalysisError(f"checkpoint identity digest mismatch: {path}")
    return payload


def _validate_cached_result_identity(
    result: Mapping[str, Any],
    expected_identity: Mapping[str, str],
    path: Path,
) -> None:
    for key, expected in expected_identity.items():
        if result.get(key) != expected:
            raise AnalysisError(f"cached result identity mismatch for {key}: {path}")


def _output_file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_visible_gpu_indices(
    gpu_indices: Sequence[int], visible_device_count: int
) -> None:
    missing = [index for index in gpu_indices if index >= visible_device_count]
    if missing:
        raise AnalysisError(
            "configured Hydra GPU indices are not visible: "
            f"missing={missing}, visible_device_count={visible_device_count}"
        )


def coordinator_lock_path(output: Path) -> Path:
    return output.parent / f".{output.name}.strict-coordinator.lock"


@contextmanager
def hold_coordinator_lock(output: Path) -> Iterable[Path]:
    """Lifetime-hold one nonblocking coordinator lock beside the output root."""
    path = coordinator_lock_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AnalysisError(f"strict coordinator lock is held: {path}") from exc
        yield path
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def parse_gpu_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "GPU indices must be comma-separated integers"
        ) from exc
    if not indices or any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError(
            "at least one nonnegative GPU index is required"
        )
    if len(indices) != len(set(indices)):
        raise argparse.ArgumentTypeError("GPU indices must be unique")
    return indices


def build_hydra_task_manifest(
    *,
    families: Sequence[str],
    inventories: Mapping[str, Sequence[EffectiveRecipe]],
    raw_inputs: Mapping[str, Mapping[str, Any]],
    canonical_raw_schemas: Mapping[str, Sequence[str]],
    annotations: pl.DataFrame,
    split_by_id: Mapping[str, str],
    gpu_indices: Sequence[int],
    coordinator_sha256: str,
    worker_sha256: str,
    sweep_root: Path,
) -> dict[str, Any]:
    """Build deterministic prefix-group tasks covering each candidate once."""
    tasks: list[dict[str, Any]] = []
    split_sha256 = canonical_json_sha256(
        sorted((str(key), str(value)) for key, value in split_by_id.items())
    )
    for family in families:
        raw_identity = raw_inputs[f"{family}/Raw"]
        frame, features, _ = load_raw_profile(
            family,
            "Raw",
            annotations,
            canonical_raw_schemas[family],
            str(raw_identity["source_schema_sha256"]),
        )
        fit_ids_by_clean: dict[str, str] = {}
        for group_index, group in enumerate(
            contiguous_prefix_groups(inventories[family])
        ):
            first = group[0]
            clean_signature = canonical_json_sha256(
                [
                    step
                    for step in _enabled_steps(first.config)
                    if step["name"] == "clean_nans"
                ]
            )
            if clean_signature not in fit_ids_by_clean:
                fit_ids_by_clean[clean_signature] = exact_fit_ids_hash(
                    frame, features, first, split_by_id
                )
            prefix_behavior = recipe_prefix_behavior_sha256(first) or ""
            task_index = len(tasks)
            task: dict[str, Any] = {
                "task_index": task_index,
                "family": family,
                "group_index": group_index,
                "gpu_index": int(gpu_indices[task_index % len(gpu_indices)]),
                "input_sha256": str(raw_identity["sha256"]),
                "source_schema_sha256": str(raw_identity["source_schema_sha256"]),
                "canonical_schema_sha256": str(raw_identity["canonical_schema_sha256"]),
                "fit_ids_sha256": fit_ids_by_clean[clean_signature],
                "split_sha256": split_sha256,
                "prefix_behavior_sha256": prefix_behavior,
                "coordinator_sha256": coordinator_sha256,
                "worker_sha256": worker_sha256,
                "sweep_root": str(sweep_root.resolve()),
                "recipes": [
                    {
                        "canonical_name": recipe.canonical_name,
                        "effective_signature": recipe.signature,
                        "aliases": list(recipe.aliases),
                        "config_paths": [
                            str(path.resolve()) for path in recipe.config_paths
                        ],
                        "config_sha256": [
                            sha256_file(path) for path in recipe.config_paths
                        ],
                    }
                    for recipe in group
                ],
                "result_relpath": f"hydra_worker_results/task-{task_index:04d}.json",
            }
            task["prefix_signature"] = canonical_json_sha256(
                {
                    key: task[key]
                    for key in (
                        "family",
                        "group_index",
                        "input_sha256",
                        "source_schema_sha256",
                        "canonical_schema_sha256",
                        "fit_ids_sha256",
                        "split_sha256",
                        "prefix_behavior_sha256",
                        "coordinator_sha256",
                        "worker_sha256",
                        "sweep_root",
                    )
                }
            )
            task["task_sha256"] = canonical_json_sha256(task)
            tasks.append(task)
        del frame
    candidate_keys = [
        (task["family"], recipe["effective_signature"])
        for task in tasks
        for recipe in task["recipes"]
    ]
    expected = [
        (family, recipe.signature)
        for family in families
        for recipe in inventories[family]
    ]
    if candidate_keys != expected or len(candidate_keys) != len(set(candidate_keys)):
        raise AnalysisError(
            "Hydra task manifest does not exactly close candidate inventory"
        )
    return {
        "schema_version": 1,
        "task_count": len(tasks),
        "candidate_count": len(candidate_keys),
        "gpu_indices": list(gpu_indices),
        "tasks": tasks,
    }


def validate_worker_package_payload(
    payload: Mapping[str, Any],
    *,
    protocol_hash: str,
    task: Mapping[str, Any],
    worker_sha256: str,
    coordinator_sha256: str,
) -> list[dict[str, Any]]:
    if payload.get("protocol_hash") != protocol_hash:
        raise AnalysisError("stale Hydra worker package protocol")
    if payload.get("worker_sha256") != worker_sha256:
        raise AnalysisError("stale Hydra worker package code")
    if payload.get("coordinator_sha256") != coordinator_sha256:
        raise AnalysisError("stale Hydra worker package coordinator")
    if payload.get("task") != task or payload.get("task_sha256") != task["task_sha256"]:
        raise AnalysisError("stale Hydra worker package task identity")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(task["recipes"]):
        raise AnalysisError("Hydra worker package row closure mismatch")
    expected = [recipe["effective_signature"] for recipe in task["recipes"]]
    observed = [row.get("effective_signature") for row in rows]
    if observed != expected or len(observed) != len(set(observed)):
        raise AnalysisError("Hydra worker package recipe order mismatch")
    for row in rows:
        for key in (
            "input_sha256",
            "source_schema_sha256",
            "canonical_schema_sha256",
            "split_sha256",
            "fit_ids_sha256",
            "prefix_signature",
            "task_sha256",
        ):
            if row.get(key) != task[key]:
                raise AnalysisError(f"Hydra worker package {key} mismatch")
        if row.get("worker_sha256") != worker_sha256:
            raise AnalysisError("Hydra worker package row code mismatch")
        if row.get("code_sha256") != coordinator_sha256:
            raise AnalysisError("Hydra worker package row coordinator mismatch")
    return [dict(row) for row in rows]


def _validate_existing_worker_package(
    output: Path,
    task: Mapping[str, Any],
    protocol_hash: str,
    protocol: Mapping[str, Any],
) -> bool:
    path = output / str(task["result_relpath"])
    if not path.exists():
        return False
    payload = json.loads(path.read_text())
    validate_worker_package_payload(
        payload,
        protocol_hash=protocol_hash,
        task=task,
        worker_sha256=str(protocol["hydra_selection"]["worker_sha256"]),
        coordinator_sha256=str(protocol["runner_sha256"]),
    )
    return True


def launch_hydra_selection(
    output: Path,
    protocol_hash: str,
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    hydra_jobs: int,
) -> None:
    """Launch bounded, resumable waves with distinct operational directories."""
    worker = Path(__file__).with_name("strict_split_fit_hydra_worker.py")
    sweep_root = output / "hydra_jobs"
    tasks = list(manifest["tasks"])
    for wave_index, start in enumerate(range(0, len(tasks), hydra_jobs)):
        wave_tasks = tasks[start : start + hydra_jobs]
        existing = [
            _validate_existing_worker_package(output, task, protocol_hash, protocol)
            for task in wave_tasks
        ]
        if all(existing):
            continue
        indices = [int(task["task_index"]) for task in wave_tasks]
        task_override = "task_index=" + ",".join(str(index) for index in indices)
        wave_root = sweep_root / f"wave-{wave_index:03d}"
        attempt = 0
        while (wave_root / f"attempt-{attempt:04d}").exists():
            attempt += 1
        attempt_root = wave_root / f"attempt-{attempt:04d}"
        attempt_root.mkdir(parents=True, exist_ok=False)
        command = [
            sys.executable,
            str(worker),
            "--multirun",
            f"coordinator_root={output}",
            task_override,
            "hydra/launcher=joblib",
            f"hydra.launcher.n_jobs={len(indices)}",
            f"hydra.sweep.dir={attempt_root}",
            "hydra.sweep.subdir=job-${hydra.job.num}",
            "hydra.job.chdir=false",
        ]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise AnalysisError(
                "Hydra strict selection wave "
                f"{wave_index} failed with exit status {completed.returncode}"
            )


def collect_hydra_selection(
    *,
    output: Path,
    protocol_hash: str,
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    inventories: Mapping[str, Sequence[EffectiveRecipe]],
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Validate packages and write canonical checkpoints in recipe order."""
    worker_sha256 = str(protocol["hydra_selection"]["worker_sha256"])
    coordinator_sha256 = str(protocol["runner_sha256"])
    rows_by_signature: dict[tuple[str, str], dict[str, Any]] = {}
    task_by_signature: dict[tuple[str, str], Mapping[str, Any]] = {}
    expected_paths: set[Path] = set()
    for task in manifest["tasks"]:
        result_path = output / task["result_relpath"]
        expected_paths.add(result_path.resolve())
        if not result_path.is_file():
            raise AnalysisError(f"missing Hydra worker package: {result_path}")
        payload = json.loads(result_path.read_text())
        rows = validate_worker_package_payload(
            payload,
            protocol_hash=protocol_hash,
            task=task,
            worker_sha256=worker_sha256,
            coordinator_sha256=coordinator_sha256,
        )
        for row in rows:
            key = (str(row["family"]), str(row["effective_signature"]))
            if key in rows_by_signature:
                raise AnalysisError("duplicate Hydra candidate result")
            rows_by_signature[key] = row
            task_by_signature[key] = task
    result_root = output / "hydra_worker_results"
    observed_paths = (
        {path.resolve() for path in result_root.glob("*.json")}
        if result_root.is_dir()
        else set()
    )
    if observed_paths != expected_paths:
        raise AnalysisError("Hydra worker result package closure mismatch")

    family_rows: dict[str, list[dict[str, Any]]] = {}
    for family, recipes in inventories.items():
        canonical_rows: list[dict[str, Any]] = []
        checkpoint_root = output / "checkpoints" / "selection" / family
        checkpoint_root.mkdir(parents=True, exist_ok=True)
        for recipe in recipes:
            key = (family, recipe.signature)
            row = rows_by_signature.get(key)
            task = task_by_signature.get(key)
            if row is None or task is None or row["config"] != recipe.canonical_name:
                raise AnalysisError("Hydra result missing from canonical recipe order")
            identity = checkpoint_identity(
                code_sha256=coordinator_sha256,
                family=family,
                codec="Raw",
                config=recipe.canonical_name,
                effective_signature=recipe.signature,
                input_sha256=str(task["input_sha256"]),
                fit_ids_sha256=str(task["fit_ids_sha256"]),
                split_sha256=str(task["split_sha256"]),
                source_schema_sha256=str(task["source_schema_sha256"]),
                canonical_schema_sha256=str(task["canonical_schema_sha256"]),
            )
            identity.update(
                {
                    "prefix_signature": str(task["prefix_signature"]),
                    "worker_sha256": worker_sha256,
                    "task_sha256": str(task["task_sha256"]),
                }
            )
            cache_key = canonical_json_sha256(identity)
            path = checkpoint_root / f"{cache_key}.json"
            cached = _load_checkpoint(path, protocol_hash, identity)
            if cached is None:
                _checkpoint(
                    path,
                    protocol_hash,
                    {"cache_key": cache_key, "result": row},
                    identity,
                )
            else:
                cached_row = dict(cached["result"])
                _validate_cached_result_identity(cached_row, identity, path)
                if cached_row != row:
                    raise AnalysisError(f"Hydra canonical checkpoint drift: {path}")
            canonical_rows.append(row)
        family_rows[family] = canonical_rows
    if set(rows_by_signature) != {
        (family, recipe.signature)
        for family, recipes in inventories.items()
        for recipe in recipes
    }:
        raise AnalysisError("Hydra collected result inventory mismatch")
    return [row for family in inventories for row in family_rows[family]], family_rows


def run_production(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    with hold_coordinator_lock(output) as lock_path:
        return _run_production_locked(args, lock_path)


def _run_production_locked(args: argparse.Namespace, lock_path: Path) -> int:
    output = args.output_dir.resolve()
    if args.hydra_jobs > len(args.hydra_gpus):
        raise AnalysisError(
            "Hydra jobs may not exceed bound GPUs; one concurrent task per GPU"
        )
    os.environ["JUMP_LITE_STRICT_GPU_INDEX"] = str(args.hydra_gpus[0])
    cp = _gpu()
    validate_visible_gpu_indices(args.hydra_gpus, cp.cuda.runtime.getDeviceCount())
    annotations = build_annotation_table()
    split_rows = make_split_from_annotations(
        annotations, args.validation_fraction, args.seed
    )
    split_by_id = {row["treatment_id"]: row["split"] for row in split_rows}
    validation_ids = {
        key for key, value in split_by_id.items() if value == "validation"
    }
    test_ids = {key for key, value in split_by_id.items() if value == "test"}
    families = list(args.families)
    inventories = {
        family: discover_effective_recipes(family, args.sweep_root, args.max_recipes)
        for family in families
    }
    target_label_parity = validate_projected_target_labels(annotations)
    raw_inputs: dict[str, dict[str, Any]] = {}
    canonical_raw_schemas: dict[str, list[str]] = {}
    for family in families:
        for codec in FAMILIES[family].codec_folders:
            identity = inspect_raw_profile(family, codec)
            raw_inputs[f"{family}/{codec}"] = identity
            if codec == "Raw":
                canonical_raw_schemas[family] = list(identity["canonical_features"])
            else:
                require_canonical_schema(
                    family,
                    codec,
                    canonical_raw_schemas[family],
                    identity["canonical_features"],
                )
    code_hash = sha256_file(Path(__file__))
    # The coordinator performs manifest fit-ID checks and final codec fits on
    # the first bound GPU. Hydra workers override this per immutable task.
    os.environ["JUMP_LITE_STRICT_GPU_INDEX"] = str(args.hydra_gpus[0])
    worker_path = Path(__file__).with_name("strict_split_fit_hydra_worker.py")
    worker_hash = sha256_file(worker_path)
    task_manifest = build_hydra_task_manifest(
        families=families,
        inventories=inventories,
        raw_inputs=raw_inputs,
        canonical_raw_schemas=canonical_raw_schemas,
        annotations=annotations,
        split_by_id=split_by_id,
        gpu_indices=args.hydra_gpus,
        coordinator_sha256=code_hash,
        worker_sha256=worker_hash,
        sweep_root=args.sweep_root,
    )
    task_manifest_hash = canonical_json_sha256(task_manifest)
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "strict_split_before_fit": True,
        "selection_codec": "Raw",
        "codec_fit_policy": (
            "fit the selected recipe separately per codec on validation treatments "
            "plus shared controls; exclude all test treatments from every fit"
        ),
        "seed": args.seed,
        "validation_fraction": args.validation_fraction,
        "cpu_workers": args.cpu_workers,
        "hydra_selection": {
            "launcher": "hydra_joblib_launcher.joblib_launcher.JoblibLauncher",
            "hydra_jobs": args.hydra_jobs,
            "gpu_indices": list(args.hydra_gpus),
            "dispatch_policy": (
                "bounded Hydra waves with at most one concurrent task per GPU"
            ),
            "coordinator_gpu_index": args.hydra_gpus[0],
            "worker_sha256": worker_hash,
            "task_manifest_sha256": task_manifest_hash,
            "task_count": task_manifest["task_count"],
            "candidate_count": task_manifest["candidate_count"],
            "worker_result_policy": (
                "unique immutable packages only; coordinator is sole canonical writer"
            ),
            "hydra_output_relpath": "hydra_jobs",
            "hydra_output_classification": (
                "operational logs excluded from scientific checksum closure"
            ),
            "worker_result_relpath": "hydra_worker_results",
        },
        "selection_prefix_cache": {
            "policy": (
                "one Hydra task per contiguous canonical pre-TVN prefix group; "
                "fit each prefix once and evaluate all suffix recipes"
            ),
            "single_canonical_writer": True,
            "group_counts": {
                family: len(contiguous_prefix_groups(recipes))
                for family, recipes in inventories.items()
            },
        },
        "families": families,
        "candidate_effective_counts": {
            family: len(recipes) for family, recipes in inventories.items()
        },
        "candidate_alias_counts": {
            family: sum(len(recipe.aliases) for recipe in recipes)
            for family, recipes in inventories.items()
        },
        "effective_recipe_inventory": protocol_recipe_inventory(
            inventories, args.sweep_root
        ),
        "raw_input_schemas": raw_inputs,
        "canonical_raw_schemas": canonical_raw_schemas,
        "canonical_raw_schema_sha256": {
            family: ordered_schema_sha256(schema)
            for family, schema in canonical_raw_schemas.items()
        },
        "target_label_parity": target_label_parity,
        "archived_label_use": (
            "metadata-only parity projection; archived feature and score columns "
            "are never read by strict fitting or scoring"
        ),
        "split_sha256": canonical_json_sha256(
            sorted((str(key), str(value)) for key, value in split_by_id.items())
        ),
        "split_artifact_sha256": canonical_json_sha256(split_rows),
        "runner_sha256": code_hash,
        "coordinator_lock": {
            "path": str(lock_path),
            "policy": "nonblocking exclusive OS flock held for coordinator lifetime",
            "outside_scientific_closure": True,
        },
        "sweep_root": str(args.sweep_root.resolve()),
        "raw_root": str(RAW_ROOT),
        "metadata": {
            str(PERTURBATION_METADATA): sha256_file(PERTURBATION_METADATA),
            str(REFCHEM_METADATA): sha256_file(REFCHEM_METADATA),
        },
    }
    protocol_hash = canonical_json_sha256(protocol)
    if args.resume:
        if not output.is_dir():
            raise AnalysisError(f"resume root does not exist: {output}")
        observed = json.loads((output / "protocol.json").read_text())
        if canonical_json_sha256(observed) != protocol_hash:
            raise AnalysisError("resume protocol mismatch")
        observed_manifest = json.loads(
            (output / "hydra_task_manifest.json").read_text()
        )
        if observed_manifest != task_manifest:
            raise AnalysisError("resume Hydra task manifest mismatch")
        observed_split = json.loads((output / "hydra_split.json").read_text())
        if observed_split != {"rows": split_rows}:
            raise AnalysisError("resume Hydra split artifact mismatch")
    else:
        ensure_create_only(output)
        atomic_write_json(output / "protocol.json", protocol)
        atomic_write_json(output / "hydra_task_manifest.json", task_manifest)
        atomic_write_json(output / "hydra_split.json", {"rows": split_rows})
        atomic_write_pandas_csv(
            output / "treatment_split.csv", pd.DataFrame(split_rows)
        )
    checkpoint_root = output / "checkpoints"
    checkpoint_root.mkdir(exist_ok=True)
    all_candidate_rows: list[dict[str, Any]] = []
    winners: dict[str, EffectiveRecipe] = {}
    selected_rows: list[dict[str, Any]] = []
    input_records: dict[str, dict[str, Any]] = {}
    started = time.monotonic()

    for family in families:
        input_records[f"{family}/Raw"] = dict(raw_inputs[f"{family}/Raw"])
    launch_hydra_selection(
        output, protocol_hash, protocol, task_manifest, args.hydra_jobs
    )
    collected_rows, collected_by_family = collect_hydra_selection(
        output=output,
        protocol_hash=protocol_hash,
        protocol=protocol,
        manifest=task_manifest,
        inventories=inventories,
    )
    all_candidate_rows.clear()
    for family in families:
        family_rows = collected_by_family[family]
        scored, ranges = minmax_score_candidates(family_rows)
        all_candidate_rows.extend(scored)
        winning_name = str(scored[0]["config"])
        winner = next(
            recipe
            for recipe in inventories[family]
            if recipe.canonical_name == winning_name
        )
        winners[family] = winner
        selected_rows.append(
            {
                **scored[0],
                "runner_up": scored[1]["config"] if len(scored) > 1 else "",
                **{f"selection_{key}": value for key, value in ranges.items()},
            }
        )
    if len(collected_rows) != len(all_candidate_rows):
        raise AnalysisError("Hydra collected/scored candidate count mismatch")

    atomic_write_pandas_csv(
        output / "validation_config_scores.csv",
        pd.DataFrame(all_candidate_rows),
    )
    atomic_write_pandas_csv(
        output / "selected_configs.csv", pd.DataFrame(selected_rows)
    )

    profile_paths: dict[tuple[str, str], Path] = {}
    fit_audits: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    profiles_root = output / "profiles"
    profiles_root.mkdir(exist_ok=True)
    final_checkpoint = checkpoint_root / "final_fit"
    final_checkpoint.mkdir(exist_ok=True)
    for family, winner in winners.items():
        for codec in FAMILIES[family].codec_folders:
            path = profiles_root / f"{family}__{codec}.parquet"
            codec_recipe = recipe_for_codec(winner, codec, args.sweep_root)
            raw_identity = raw_inputs[f"{family}/{codec}"]
            frame, features, input_path = load_raw_profile(
                family,
                codec,
                annotations,
                canonical_raw_schemas[family],
                raw_identity["source_schema_sha256"],
            )
            input_hash = str(raw_identity["sha256"])
            input_records[f"{family}/{codec}"] = dict(raw_identity)
            fit_ids_hash = exact_fit_ids_hash(
                frame, features, codec_recipe, split_by_id
            )
            identity = checkpoint_identity(
                code_sha256=code_hash,
                family=family,
                codec=codec,
                config=winner.canonical_name,
                effective_signature=winner.signature,
                input_sha256=input_hash,
                fit_ids_sha256=fit_ids_hash,
                split_sha256=protocol["split_sha256"],
                source_schema_sha256=raw_identity["source_schema_sha256"],
                canonical_schema_sha256=raw_identity["canonical_schema_sha256"],
            )
            checkpoint_path = final_checkpoint / (
                f"{family}__{codec}__{canonical_json_sha256(identity)}.json"
            )
            cached = _load_checkpoint(checkpoint_path, protocol_hash, identity)
            if cached is None:
                transformed, state = fit_transform_recipe(
                    frame,
                    features,
                    codec_recipe,
                    split_by_id,
                    codec,
                    args.cpu_workers,
                )
                if state.fit_ids_sha256 != fit_ids_hash:
                    raise AnalysisError(
                        "final cache fit-ID hash differs from fitted rows"
                    )
                transformed.write_parquet(path, compression="zstd")
                row = {
                    "family": family,
                    "codec": codec,
                    "config": winner.canonical_name,
                    "effective_signature": winner.signature,
                    "state_sha256": state.digest(),
                    "fit_ids_sha256": state.fit_ids_sha256,
                    "input_path": str(input_path),
                    "input_sha256": input_hash,
                    "source_schema_sha256": raw_identity["source_schema_sha256"],
                    "canonical_schema_sha256": raw_identity["canonical_schema_sha256"],
                    "split_sha256": state.split_sha256,
                    "code_sha256": code_hash,
                    "native_wells": len(transformed),
                    "native_controls": transformed.filter(pl.col(NEGCON_COL)).height,
                    "native_treatments": transformed.filter(~pl.col(NEGCON_COL))[
                        COMPOUND_COL
                    ].n_unique(),
                    "features": len(state.retained_features),
                    "profile_sha256": sha256_file(path),
                    "profile_size_bytes": path.stat().st_size,
                    **state.fit_audit,
                }
                _checkpoint(checkpoint_path, protocol_hash, {"result": row}, identity)
            else:
                row = dict(cached["result"])
                _validate_cached_result_identity(row, identity, checkpoint_path)
                if not path.is_file() or sha256_file(path) != row["profile_sha256"]:
                    raise AnalysisError(f"final profile checkpoint mismatch: {path}")
                if sha256_file(input_path) != row["input_sha256"]:
                    raise AnalysisError(f"resumed raw input drift: {input_path}")
            profile_paths[(family, codec)] = path
            fit_audits.append(row)

    common: set[str] | None = None
    annotation_digests: set[str] = set()
    for key, path in profile_paths.items():
        meta = pl.read_parquet(
            path, columns=[ID_COL, COMPOUND_COL, GROUP_COL, NEGCON_COL]
        ).with_columns(pl.col(ID_COL).cast(pl.Utf8))
        if meta[ID_COL].n_unique() != len(meta):
            raise AnalysisError(f"duplicate output wells: {path}")
        ids = set(meta[ID_COL].to_list())
        common = ids if common is None else common.intersection(ids)
        annotation_digests.add(canonical_json_sha256(meta.sort(ID_COL).to_dicts()))
        coverage_rows.append(
            {
                "family": key[0],
                "codec": key[1],
                "native_wells": len(meta),
                "native_controls": meta.filter(pl.col(NEGCON_COL)).height,
            }
        )
    if not common:
        raise AnalysisError("strict selected profiles have no common wells")
    ordered_common = sorted(common)
    # Native annotation digests may differ by codec-specific coverage.  The
    # intersected annotations below must be exactly identical.
    intersection_digest: str | None = None
    for key, path in profile_paths.items():
        meta = (
            pl.read_parquet(path, columns=[ID_COL, COMPOUND_COL, GROUP_COL, NEGCON_COL])
            .filter(pl.col(ID_COL).is_in(ordered_common))
            .sort(ID_COL)
        )
        digest = canonical_json_sha256(meta.to_dicts())
        if intersection_digest is None:
            intersection_digest = digest
        elif digest != intersection_digest:
            raise AnalysisError(f"strict common-well annotations disagree: {key}")
    atomic_write_pandas_csv(
        output / "common_wells.csv", pd.DataFrame({ID_COL: ordered_common})
    )
    for row in coverage_rows:
        row["common_wells"] = len(ordered_common)
        row["dropped_for_intersection"] = row["native_wells"] - len(ordered_common)
    atomic_write_pandas_csv(
        output / "selected_codec_coverage.csv", pd.DataFrame(coverage_rows)
    )
    atomic_write_pandas_csv(output / "fit_audit.csv", pd.DataFrame(fit_audits))

    final_rows: list[dict[str, Any]] = []
    per_unit = output / "per_unit"
    per_unit.mkdir(exist_ok=True)
    common_set = set(ordered_common)
    for (family, codec), path in profile_paths.items():
        frame = pl.read_parquet(path).filter(pl.col(ID_COL).is_in(ordered_common))
        features, _ = infer_columns(frame, ["Metadata_"])
        features = get_numeric_features(frame, features)
        metrics, pa_map, pc_map = score_partition(frame, features, test_ids)
        config = winners[family].canonical_name
        pa_map = pa_map.copy()
        pa_map.insert(0, "codec", codec)
        pa_map.insert(0, "family", family)
        pa_map.insert(2, "config", config)
        pc_map = pc_map.copy()
        pc_map.insert(0, "codec", codec)
        pc_map.insert(0, "family", family)
        pc_map.insert(2, "config", config)
        atomic_write_pandas_csv(
            per_unit / f"{family}__{codec}__pa_treatments.csv", pa_map
        )
        atomic_write_pandas_csv(per_unit / f"{family}__{codec}__pc_targets.csv", pc_map)
        fit_row = next(
            row
            for row in fit_audits
            if row["family"] == family and row["codec"] == codec
        )
        final_rows.append(
            {
                "status": "ok",
                "family": family,
                "codec": codec,
                "config": config,
                "state_sha256": fit_row["state_sha256"],
                "n_evaluation_wells": metrics["wells"],
                "n_evaluation_controls": metrics["controls"],
                "n_evaluation_treatment_ids": metrics["treatments"],
                "n_features": metrics["features"],
                "test_pa_mean_nap": metrics["pa_mean_nap"],
                "test_pc_mean_nap": metrics["pc_mean_nap"],
                "test_balanced_nap_product": metrics["balanced_nap_product"],
                "test_pa_units": metrics["pa_units"],
                "test_pc_targets": metrics["pc_targets"],
            }
        )
    atomic_write_pandas_csv(
        output / "heldout_test_scores.csv", pd.DataFrame(final_rows)
    )
    provenance = {
        "protocol": protocol,
        "protocol_sha256": protocol_hash,
        "completed_at": utc_now(),
        "runtime_seconds": time.monotonic() - started,
        "input_files": list(input_records.values()),
        "dimensions": {
            "validation_treatments_assigned": len(validation_ids),
            "test_treatments_assigned": len(test_ids),
            "effective_candidates": sum(len(rows) for rows in inventories.values()),
            "candidate_aliases": sum(
                len(recipe.aliases)
                for recipes in inventories.values()
                for recipe in recipes
            ),
            "selected_profiles": len(profile_paths),
            "common_wells": len(common_set),
        },
        "winners": [
            {
                "family": family,
                "config": recipe.canonical_name,
                "aliases": list(recipe.aliases),
                "effective_signature": recipe.signature,
            }
            for family, recipe in winners.items()
        ],
        "intersection_annotation_sha256": intersection_digest,
    }
    atomic_write_json(output / "provenance.json", provenance)
    closure_paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file()
        and path.name not in {"SHA256SUMS", "output_hashes.json"}
        and path.relative_to(output).parts[0] != "hydra_jobs"
    )
    atomic_write_json(
        output / "output_hashes.json",
        {"files": [_output_file_record(path, output) for path in closure_paths]},
    )
    checksum_paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file()
        and path.name != "SHA256SUMS"
        and path.relative_to(output).parts[0] != "hydra_jobs"
    )
    atomic_write_text(
        output / "SHA256SUMS",
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output).as_posix()}\n"
            for path in checksum_paths
        ),
    )
    return 0


def run_smoke(args: argparse.Namespace) -> int:
    output = args.output_dir.resolve()
    ensure_create_only(output)
    annotations = build_annotation_table()
    split_rows = make_split_from_annotations(
        annotations, args.validation_fraction, args.seed
    )
    split_by_id = {row["treatment_id"]: row["split"] for row in split_rows}
    validation_ids = {
        key for key, value in split_by_id.items() if value == "validation"
    }
    family = args.families[0]
    recipes = discover_effective_recipes(family, args.sweep_root, args.max_recipes)
    frame, features, input_path = load_raw_profile(family, "Raw", annotations)
    candidate_rows: list[dict[str, Any]] = []
    start = time.monotonic()
    for recipe in recipes:
        transformed, state = fit_transform_recipe(
            frame,
            features,
            recipe,
            split_by_id,
            "Raw",
            args.cpu_workers,
        )
        metrics, _, _ = score_partition(
            transformed, state.retained_features, validation_ids
        )
        candidate_rows.append(
            {
                "family": family,
                "codec": "Raw",
                "config": recipe.canonical_name,
                "aliases": "|".join(recipe.aliases),
                "effective_signature": recipe.signature,
                "state_sha256": state.digest(),
                "validation_pa_mean_nap": metrics["pa_mean_nap"],
                "validation_pc_mean_nap": metrics["pc_mean_nap"],
                **state.fit_audit,
            }
        )
    scored, _ = minmax_score_candidates(candidate_rows)
    atomic_write_pandas_csv(
        output / "validation_config_scores.csv", pd.DataFrame(scored)
    )
    atomic_write_pandas_csv(output / "treatment_split.csv", pd.DataFrame(split_rows))
    atomic_write_json(
        output / "provenance.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "created_at": utc_now(),
            "strict_split_before_fit": True,
            "family": family,
            "candidate_effective_count": len(recipes),
            "candidate_alias_count": sum(len(recipe.aliases) for recipe in recipes),
            "cpu_workers": args.cpu_workers,
            "input": {"path": str(input_path), "sha256": sha256_file(input_path)},
            "split_sha256": canonical_json_sha256(
                sorted((str(key), str(value)) for key, value in split_by_id.items())
            ),
            "split_artifact_sha256": canonical_json_sha256(split_rows),
            "code_sha256": sha256_file(Path(__file__)),
            "elapsed_seconds": time.monotonic() - start,
            "winner": scored[0]["config"],
        },
    )
    return 0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, default=SWEEP_ROOT)
    parser.add_argument(
        "--families", nargs="+", choices=sorted(FAMILIES), default=sorted(FAMILIES)
    )
    parser.add_argument("--max-recipes", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--cpu-workers",
        type=_positive_int,
        default=1,
        help="bounded CPU threads for canonical per-feature computations",
    )
    parser.add_argument(
        "--hydra-jobs",
        type=_positive_int,
        default=4,
        help="bounded Hydra Joblib selection processes",
    )
    parser.add_argument(
        "--hydra-gpus",
        type=parse_gpu_indices,
        default=(0, 1, 2, 3),
        help="comma-separated visible GPU indices assigned deterministically",
    )
    parser.add_argument("--mode", choices=["smoke", "production"], default="smoke")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "smoke":
        if len(args.families) != 1:
            raise AnalysisError("smoke mode requires exactly one family")
        return run_smoke(args)
    return run_production(args)


if __name__ == "__main__":
    raise SystemExit(main())
