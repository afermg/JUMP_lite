#!/usr/bin/env python3
"""Compression-order robustness analyses from frozen normalized profiles.

This runner never writes to the canonical archives.  It implements two analyses:
(A) stratified Target-2-sized subsamples of fixed-recipe JUMP-lite profiles and
(B) a paired Target-2 Zstd-selected D10-versus-D15 cluster bootstrap.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import polars as pl

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from norm_3.io import get_numeric_features, infer_columns  # noqa: E402
from norm_3.metrics import (  # noqa: E402
    calculate_phenotypic_activity,
    calculate_phenotypic_consistency,
)

PROTOCOL_VERSION = 2
SEED = 20_260_814
N_SAMPLES = 2_000
N_TREATMENTS = 306
WELLS_PER_UNIT = 4
TARGET_BOOTSTRAPS = 50_000
CODECS = ("Raw", "HQ", "MQ", "D20")
EXPECTED_PAIRS = (("Raw", "HQ"), ("HQ", "MQ"), ("MQ", "D20"))
FAMILIES = ("dinov2", "morphem", "openphenom", "subcell")
TARGET_FOLDERS = {
    "dinov2": "dinov2",
    "morphem": "morphem",
    "openphenom": "openphenom",
    "subcell": "subcell__clip01",
}
DISPLAY = {
    "dinov2": "DINOv2",
    "morphem": "MorphEM",
    "openphenom": "OpenPhenom",
    "subcell": "SubCell",
}
POST = REPO / "rebuttal/postprocessing_validation/results"
TARGET_SWEEP = Path(
    "/work/datasets/JUMP-lite-wacv/sweeps/MAIN_RESULTS__figure_4_variance_first_v11"
)
CHECKSUMS = POST / "artifact_checksums.json"
COMMON = POST / "common_wells.csv"
SPLIT = POST / "treatment_split.csv"
SELECTED = POST / "selected_configs.csv"
HELDOUT = POST / "heldout_test_scores.csv"
POST_PROVENANCE = POST / "provenance.json"
PER_UNIT = POST / "per_unit"
ID = "Metadata_id"
COMPOUND = "Metadata_JCP2022"
GROUP = "Metadata_Group"
TARGET = "Metadata_RefChemDB_target"
NEGCON = "Metadata_negcon"
PLATE = "Metadata_Plate"
PC_GROUPS = ("group_high", "group_low")


class AnalysisError(RuntimeError):
    """Fail-closed analysis error."""


@dataclass(frozen=True)
class Variant:
    family: str
    codec: str
    config: str
    output: Path
    config_path: Path
    sha256: str
    size: int


@dataclass(frozen=True)
class RunSpec:
    """Explicit protocol dimensions passed across spawn-process boundaries."""

    n_samples: int = N_SAMPLES
    target_bootstraps: int = TARGET_BOOTSTRAPS
    seed: int = SEED

    @property
    def production(self) -> bool:
        return self.n_samples == N_SAMPLES and self.target_bootstraps == TARGET_BOOTSTRAPS


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_text(path: Path, content: str) -> None:
    atomic_bytes(path, content.encode())


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    if frame.empty:
        raise AnalysisError(f"refusing to write empty CSV: {path}")
    atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))


def atomic_parquet(path: Path, frame: pl.DataFrame) -> None:
    if frame.is_empty():
        raise AnalysisError(f"refusing to write empty parquet: {path}")
    atomic_parquet_allow_empty(path, frame)


def atomic_parquet_allow_empty(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        frame.write_parquet(name, compression="zstd", statistics=True)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def file_record(path: Path, hash_content: bool = True) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise AnalysisError(f"missing or empty input: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path) if hash_content else None,
    }


def release_file_record(path: Path, root: Path) -> dict[str, Any]:
    """Return a relocatable, root-relative final-artifact identity."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise AnalysisError(f"release artifact escapes output root: {path}") from exc
    record = file_record(path)
    record["path"] = relative.as_posix()
    return record


def checksum_map() -> dict[str, dict[str, Any]]:
    payload = json.loads(CHECKSUMS.read_text())
    records = payload["result_artifacts"] + payload["selected_normalized_profiles"]
    return {record["path"]: record for record in records}


def verify_record(path: Path, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    key = str(path.resolve())
    if key not in records:
        raise AnalysisError(f"input absent from frozen checksum inventory: {path}")
    record = records[key]
    if path.stat().st_size != record["size_bytes"]:
        raise AnalysisError(f"input size drift: {path}")
    actual = sha256_file(path)
    if actual != record["sha256"]:
        raise AnalysisError(f"input hash drift: {path}")
    return record


def load_variants() -> tuple[list[Variant], dict[str, Any]]:
    selected = pd.read_csv(SELECTED).set_index("family")
    scores = pd.read_csv(HELDOUT)
    provenance = json.loads(POST_PROVENANCE.read_text())
    checksums = checksum_map()
    selected_outputs = {
        (row["family"], row["codec"]): row for row in provenance["selected_outputs"]
    }
    variants: list[Variant] = []
    for family in FAMILIES:
        if family not in selected.index:
            raise AnalysisError(f"missing selected family: {family}")
        config = str(selected.loc[family, "config"])
        for codec in CODECS:
            key = (family, codec)
            if key not in selected_outputs:
                raise AnalysisError(f"missing selected output: {key}")
            row = selected_outputs[key]
            if row["config"] != config:
                raise AnalysisError(f"selected config drift for {key}")
            output = Path(row["output_path"])
            config_path = Path(row["config_path"])
            verified = verify_record(output, checksums)
            if sha256_file(config_path) != row["config_sha256"]:
                raise AnalysisError(f"config hash drift: {config_path}")
            score = scores[(scores.family == family) & (scores.codec == codec)]
            if len(score) != 1 or score.iloc[0].config != config:
                raise AnalysisError(f"held-out score/config mismatch for {key}")
            variants.append(
                Variant(family, codec, config, output, config_path, verified["sha256"], verified["size_bytes"])
            )
    for path in (COMMON, SPLIT, SELECTED, HELDOUT, POST_PROVENANCE):
        verify_record(path, checksums)
    return variants, provenance


def annotation_digest(frame: pl.DataFrame) -> str:
    columns = [ID, COMPOUND, GROUP, NEGCON, PLATE, TARGET]
    missing = set(columns).difference(frame.columns)
    if missing:
        raise AnalysisError(f"missing annotation columns: {sorted(missing)}")
    selected = frame.select(columns).sort(ID)
    if selected[ID].n_unique() != len(selected):
        raise AnalysisError("Metadata_id is not unique")
    digest = hashlib.sha256()
    for row in selected.iter_rows():
        digest.update("\0".join("" if value is None else str(value) for value in row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def largest_remainder_quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    if total <= 0 or not counts or any(value <= 0 for value in counts.values()):
        raise AnalysisError("invalid quota inputs")
    available = sum(counts.values())
    if total > available:
        raise AnalysisError("quota exceeds eligible population")
    exact = {key: total * value / available for key, value in counts.items()}
    quotas = {key: math.floor(value) for key, value in exact.items()}
    remainder = total - sum(quotas.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remainder]:
        quotas[key] += 1
    if sum(quotas.values()) != total or any(quotas[key] > counts[key] for key in counts):
        raise AnalysisError("invalid largest-remainder result")
    return quotas


def build_manifests(
    output_dir: Path, variants: list[Variant], n_samples: int = N_SAMPLES
) -> dict[str, Any]:
    common = pl.read_csv(COMMON, schema_overrides={NEGCON: pl.Boolean})
    split = pl.read_csv(SPLIT).filter(pl.col("split") == "test")
    test_ids = set(split["treatment_id"].to_list())
    base = pl.read_parquet(variants[0].output)
    common_ids = set(common[ID].to_list())
    base = base.filter(
        pl.col(ID).is_in(sorted(common_ids))
        & pl.col(GROUP).is_in(PC_GROUPS)
        & (pl.col(NEGCON) | pl.col(COMPOUND).is_in(sorted(test_ids)))
    ).select([ID, COMPOUND, GROUP, NEGCON, PLATE, TARGET])
    if base[ID].n_unique() != len(base):
        raise AnalysisError("baseline common profile has duplicate IDs")
    expected_common = common.filter(
        pl.col(GROUP).is_in(PC_GROUPS)
        & (pl.col(NEGCON) | pl.col(COMPOUND).is_in(sorted(test_ids)))
    )
    if set(base[ID].to_list()) != set(expected_common[ID].to_list()):
        raise AnalysisError("baseline common held-out identity mismatch")
    digest = annotation_digest(base)

    # Every variant must have the same annotations on this frozen population.
    for variant in variants[1:]:
        candidate = pl.read_parquet(
            variant.output,
            columns=[ID, COMPOUND, GROUP, NEGCON, PLATE, TARGET],
        ).filter(pl.col(ID).is_in(sorted(base[ID].to_list())))
        if len(candidate) != len(base) or annotation_digest(candidate) != digest:
            raise AnalysisError(f"cross-variant annotation mismatch: {variant.family}/{variant.codec}")

    treatment = base.filter(~pl.col(NEGCON))
    counts = treatment.group_by([COMPOUND, GROUP]).agg(pl.len().alias("n_wells"))
    memberships = (
        counts.group_by(COMPOUND)
        .agg(
            pl.col(GROUP).sort().str.join("|").alias("stratum"),
            pl.col("n_wells").min().alias("min_unit_wells"),
            pl.col(GROUP).n_unique().alias("n_groups"),
        )
        .filter(pl.col("min_unit_wells") >= WELLS_PER_UNIT)
        .sort(COMPOUND)
    )
    allowed = {"group_high", "group_low", "group_high|group_low"}
    if not set(memberships["stratum"].to_list()).issubset(allowed):
        raise AnalysisError("unexpected membership stratum")
    eligible_ids = set(memberships[COMPOUND].to_list())
    eligible = treatment.filter(pl.col(COMPOUND).is_in(sorted(eligible_ids)))
    count_map = {
        row["stratum"]: int(row["len"])
        for row in memberships.group_by("stratum").len().iter_rows(named=True)
    }
    quotas = largest_remainder_quotas(count_map, N_TREATMENTS)

    by_stratum = {
        stratum: memberships.filter(pl.col("stratum") == stratum)[COMPOUND].to_numpy()
        for stratum in sorted(count_map)
    }
    well_map = {
        (compound, group): np.asarray(sorted(ids), dtype=object)
        for compound, group, ids in eligible.group_by([COMPOUND, GROUP]).agg(pl.col(ID)).iter_rows()
    }
    plate_map = dict(eligible.select(ID, PLATE).iter_rows())
    rng = np.random.Generator(np.random.PCG64DXSM(SEED))
    treatment_rows: list[dict[str, Any]] = []
    well_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for sample_id in range(n_samples):
        chosen: list[tuple[str, str]] = []
        for stratum in sorted(quotas):
            selected_ids = rng.choice(by_stratum[stratum], size=quotas[stratum], replace=False)
            chosen.extend((str(value), stratum) for value in selected_ids)
        if len({value for value, _ in chosen}) != N_TREATMENTS:
            raise AnalysisError("sample contains duplicate treatment clusters")
        represented: set[tuple[str, str]] = set()
        for compound, stratum in sorted(chosen):
            treatment_rows.append({"sample_id": sample_id, COMPOUND: compound, "stratum": stratum})
            for group in stratum.split("|"):
                wells = well_map[(compound, group)]
                selected_wells = rng.choice(wells, size=WELLS_PER_UNIT, replace=False)
                if len(set(selected_wells)) != WELLS_PER_UNIT:
                    raise AnalysisError("within-unit well replacement detected")
                for well_id in sorted(map(str, selected_wells)):
                    well_rows.append(
                        {"sample_id": sample_id, COMPOUND: compound, GROUP: group, ID: well_id}
                    )
                    represented.add((str(plate_map[well_id]), group))
        for plate, group in sorted(represented):
            pair_rows.append({"sample_id": sample_id, PLATE: plate, GROUP: group})

    treatments = pl.DataFrame(treatment_rows).sort(["sample_id", COMPOUND])
    wells = pl.DataFrame(well_rows).sort(["sample_id", COMPOUND, GROUP, ID])
    pairs = pl.DataFrame(pair_rows).sort(["sample_id", PLATE, GROUP])
    # Manifest-level invariants before any outcome is computed.
    if treatments.group_by("sample_id").len()["len"].unique().to_list() != [N_TREATMENTS]:
        raise AnalysisError("manifest treatment count mismatch")
    unit_counts = wells.group_by(["sample_id", COMPOUND, GROUP]).agg(
        pl.len().alias("n"), pl.col(ID).n_unique().alias("unique")
    )
    if unit_counts.filter((pl.col("n") != WELLS_PER_UNIT) | (pl.col("unique") != WELLS_PER_UNIT)).height:
        raise AnalysisError("manifest violates four-distinct-well rule")
    atomic_parquet(output_dir / "manifests/sample_treatments.parquet", treatments)
    atomic_parquet(output_dir / "manifests/sample_treatment_wells.parquet", wells)
    atomic_parquet(output_dir / "manifests/sample_plate_groups.parquet", pairs)
    atomic_parquet(output_dir / "manifests/eligible_treatments.parquet", memberships)
    return {
        "annotation_digest": digest,
        "eligible_stratum_counts": count_map,
        "quotas": quotas,
        "n_common_pc_population": len(base),
        "n_manifest_treatment_rows": len(treatments),
        "n_manifest_well_rows": len(wells),
        "n_manifest_plate_group_rows": len(pairs),
    }


def pc_query_eligibility(frame: pl.DataFrame) -> tuple[list[dict[str, Any]], int]:
    """Metadata-only undefined-query records and eligible consensus denominator."""
    metadata = frame.select([COMPOUND, TARGET, NEGCON, GROUP]).to_pandas()
    metadata[TARGET] = metadata[TARGET].fillna("unknown")
    consensus = metadata.groupby(
        [COMPOUND, TARGET, NEGCON, GROUP], as_index=False, observed=True
    ).size()
    consensus["targets"] = consensus[TARGET].str.split("|")
    consensus = consensus[consensus[NEGCON] == False].copy()
    counts_per_compound = consensus["targets"].apply(
        lambda value: len(value) if isinstance(value, list) else 0
    )
    consensus = consensus[counts_per_compound <= 50].copy()
    exploded = consensus.explode("targets")
    exploded = exploded[exploded.targets != "unknown"]
    valid = set(exploded.groupby("targets")[COMPOUND].nunique().loc[lambda value: value >= 3].index)
    consensus["eligible_targets"] = consensus.targets.apply(
        lambda values: tuple(sorted(set(values).intersection(valid)))
    )
    consensus = consensus[consensus.eligible_targets.str.len() > 0].copy()
    excluded: list[dict[str, Any]] = []
    for group, grouped in consensus.groupby(GROUP, sort=True, observed=True):
        rows = list(grouped[[COMPOUND, "eligible_targets"]].itertuples(index=False, name=None))
        for compound, targets in rows:
            target_set = set(targets)
            other_rows = [
                (other_compound, set(other_targets))
                for other_compound, other_targets in rows if other_compound != compound
            ]
            positive_count = sum(not target_set.isdisjoint(other) for _, other in other_rows)
            negative_count = sum(target_set.isdisjoint(other) for _, other in other_rows)
            reasons = []
            if positive_count == 0:
                reasons.append("no_positive")
            if negative_count == 0:
                reasons.append("no_negative")
            if reasons:
                excluded.append(
                    {
                        COMPOUND: str(compound), GROUP: str(group),
                        "eligible_targets": "|".join(targets), "reason": "|".join(reasons),
                        "positive_candidate_count": positive_count,
                        "negative_candidate_count": negative_count,
                    }
                )
    return sorted(excluded, key=lambda row: (row[GROUP], row[COMPOUND])), len(consensus)


def pc_undefined_query_exclusions(frame: pl.DataFrame) -> list[dict[str, Any]]:
    return pc_query_eligibility(frame)[0]


def pc_removal_records(exclusions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only records whose rows are physically removed from restricted PC."""
    if not exclusions:
        return []
    excluded = pd.DataFrame(exclusions)
    drop = excluded[excluded.reason.str.contains("no_negative")].copy()
    for _, grouped in excluded[excluded.reason == "no_positive"].groupby(GROUP, observed=True):
        total = int(grouped.negative_candidate_count.iloc[0]) + 1
        if len(grouped) == total:
            drop = pd.concat([drop, grouped], ignore_index=True)
    if drop.empty:
        return []
    return drop.sort_values([GROUP, COMPOUND]).to_dict("records")


def apply_pc_exclusions(
    frame: pl.DataFrame, exclusions: list[dict[str, Any]]
) -> pl.DataFrame:
    """Remove only rows that cannot serve as a query or valid reference.

    Individual no-positive compounds remain as negative references and copairs
    naturally never emits them as positive queries. If an entire group lacks
    positives, all of its eligible rows are removed so the empty query group is
    skipped. A no-negative row cannot be a negative reference for any other row
    (disjointness is symmetric), so removing it is also relation-preserving.
    """
    drop_records = pc_removal_records(exclusions)
    if not drop_records:
        return frame
    drop = pd.DataFrame(drop_records)
    pairs = pl.from_pandas(drop[[COMPOUND, GROUP]].drop_duplicates()).with_columns(
        pl.col(COMPOUND).cast(frame.schema[COMPOUND]),
        pl.col(GROUP).cast(frame.schema[GROUP]),
        pl.lit(True).alias("_exclude_pc_query"),
    )
    return (
        frame.join(pairs, on=[COMPOUND, GROUP], how="left")
        .filter(~pl.col("_exclude_pc_query").fill_null(False))
        .drop("_exclude_pc_query")
    )


def load_features(path: Path) -> tuple[pl.DataFrame, list[str]]:
    frame = pl.read_parquet(path)
    features, _ = infer_columns(frame, ["Metadata_"])
    features = get_numeric_features(frame, features)
    if not features:
        raise AnalysisError(f"no features: {path}")
    return frame, features


def score_frame(
    frame: pl.DataFrame,
    features: list[str],
    pc_exclusions: list[dict[str, Any]] | None = None,
) -> tuple[float, float, int, int]:
    distance = "euclidean" if len(features) <= 2 else "cosine"
    pa = calculate_phenotypic_activity(
        frame,
        features,
        compound_col=COMPOUND,
        negcon_col=NEGCON,
        batch_col=PLATE,
        group_col=GROUP,
        distance=distance,
    )
    pc_frame = apply_pc_exclusions(frame, pc_exclusions or [])
    pc = calculate_phenotypic_consistency(
        pc_frame,
        features,
        compound_col=COMPOUND,
        target_col=TARGET,
        negcon_col=NEGCON,
        group_col=GROUP,
        pc_groups=list(PC_GROUPS),
        distance=distance,
    )
    pa_value = float(pa["mean_normalized_average_precision"])
    pc_value_raw = pc.get("mean_normalized_average_precision")
    if pc_value_raw is None or pc.get("target_consistency") is None or int(pc.get("n_targets_total", 0)) <= 0:
        raise AnalysisError("PC computation produced no target-level result")
    pc_value = float(pc_value_raw)
    if not math.isfinite(pa_value) or not math.isfinite(pc_value):
        raise AnalysisError("non-finite subsample metric")
    return pa_value, pc_value, int(pa["n_compounds"]), int(pc["n_targets_total"])


def load_checkpoint(
    checkpoint: Path, expected_identity: dict[str, Any]
) -> list[dict[str, Any]]:
    """Load a resumable checkpoint only when its exact protocol identity matches."""
    identity_path = checkpoint.with_suffix(".protocol.json")
    if identity_path.is_file():
        if json.loads(identity_path.read_text()) != expected_identity:
            raise AnalysisError(f"checkpoint protocol drift: {identity_path}")
    elif checkpoint.is_file():
        raise AnalysisError(f"checkpoint lacks protocol identity: {checkpoint}")
    else:
        atomic_json(identity_path, expected_identity)
    if not checkpoint.is_file():
        return []
    prior = pd.read_csv(checkpoint)
    if prior.empty:
        return []
    required = {"sample_id", "family", "codec", "config", "pa", "pc", "product"}
    if missing := required.difference(prior.columns):
        raise AnalysisError(f"checkpoint columns missing {sorted(missing)}: {checkpoint}")
    expected = list(range(int(prior.sample_id.max()) + 1))
    if prior.sample_id.astype(int).tolist() != expected or len(prior) > expected_identity["n_samples"]:
        raise AnalysisError(f"noncontiguous/oversized checkpoint: {checkpoint}")
    if (set(prior.family) != {expected_identity["family"]}
            or set(prior.codec) != {expected_identity["codec"]}
            or set(prior.config) != {expected_identity["config"]}):
        raise AnalysisError(f"checkpoint variant drift: {checkpoint}")
    if not np.isfinite(prior[["pa", "pc", "product"]].to_numpy(float)).all():
        raise AnalysisError(f"checkpoint contains non-finite metrics: {checkpoint}")
    return prior.to_dict("records")


def full_worker(
    variant_dict: dict[str, Any],
    output_dir_str: str,
    expected_digest: str,
    n_samples: int,
    protocol_id: str,
) -> dict[str, Any]:
    os.environ.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "HOME": str(Path(output_dir_str) / "cache" / f"{variant_dict['family']}-{variant_dict['codec']}-{os.getpid()}"),
        }
    )
    Path(os.environ["HOME"]).mkdir(parents=True, exist_ok=True)
    # NumPy/Scipy are imported before the spawn worker enters this function;
    # apply runtime thread limits as well as environment caps.
    from threadpoolctl import threadpool_limits
    output_dir = Path(output_dir_str)
    checkpoint = output_dir / "checkpoints" / f"{variant_dict['family']}__{variant_dict['codec']}.csv"
    expected_identity = {
        "protocol_id": protocol_id,
        "family": variant_dict["family"],
        "codec": variant_dict["codec"],
        "config": variant_dict["config"],
        "output": variant_dict["output"],
        "output_sha256": variant_dict["sha256"],
        "output_size_bytes": variant_dict["size"],
        "config_sha256": variant_dict["config_sha256"],
        "runner_sha256": variant_dict["runner_sha256"],
        "n_samples": n_samples,
    }
    completed = load_checkpoint(checkpoint, expected_identity)
    start_sample = len(completed)
    frame, features = load_features(Path(variant_dict["output"]))
    common = pl.read_csv(COMMON, schema_overrides={NEGCON: pl.Boolean})
    split = pl.read_csv(SPLIT).filter(pl.col("split") == "test")
    test_ids = set(split["treatment_id"].to_list())
    ids = set(common[ID].to_list())
    frame = frame.filter(
        pl.col(ID).is_in(sorted(ids))
        & pl.col(GROUP).is_in(PC_GROUPS)
        & (pl.col(NEGCON) | pl.col(COMPOUND).is_in(sorted(test_ids)))
    )
    if annotation_digest(frame) != expected_digest:
        raise AnalysisError("worker annotation digest mismatch")
    id_to_index = {value: index for index, value in enumerate(frame[ID].to_list())}
    controls: dict[tuple[str, str], list[int]] = {}
    for row in frame.with_row_index("index").filter(pl.col(NEGCON)).select("index", PLATE, GROUP).iter_rows():
        controls.setdefault((row[1], row[2]), []).append(row[0])
    wells = pl.read_parquet(output_dir / "manifests/sample_treatment_wells.parquet")
    pairs = pl.read_parquet(output_dir / "manifests/sample_plate_groups.parquet")
    exclusions_path = output_dir / "manifests/pc_undefined_queries.parquet"
    exclusions_frame = pl.read_parquet(exclusions_path)
    started = time.monotonic()
    null = open(os.devnull, "w")
    try:
        for sample_id in range(start_sample, n_samples):
            treatment_ids = wells.filter(pl.col("sample_id") == sample_id)[ID].to_list()
            if any(value not in id_to_index for value in treatment_ids):
                raise AnalysisError("sample well absent from variant")
            indexes = [id_to_index[value] for value in treatment_ids]
            for plate, group in pairs.filter(pl.col("sample_id") == sample_id).select(PLATE, GROUP).iter_rows():
                if (plate, group) not in controls:
                    raise AnalysisError(f"represented plate/group lacks common controls: {plate}/{group}")
                indexes.extend(controls[(plate, group)])
            indexes = sorted(set(indexes))
            sample = frame[indexes]
            if sample.filter(~pl.col(NEGCON)).height != len(treatment_ids):
                raise AnalysisError("sample treatment reconstruction mismatch")
            sample_exclusions = (
                exclusions_frame.filter(pl.col("sample_id") == sample_id)
                .sort([GROUP, COMPOUND])
                .select(COMPOUND, GROUP, "eligible_targets", "reason", "positive_candidate_count", "negative_candidate_count")
                .to_dicts()
            )
            actual_exclusions = pc_undefined_query_exclusions(sample)
            if actual_exclusions != sample_exclusions:
                raise AnalysisError(
                    f"variant disagrees with frozen metadata-only PC exclusions at sample {sample_id}: "
                    f"actual={actual_exclusions[:2]} frozen={sample_exclusions[:2]}"
                )
            try:
                with threadpool_limits(limits=1), contextlib.redirect_stdout(null), contextlib.redirect_stderr(null):
                    pa, pc, pa_units, pc_targets = score_frame(sample, features, sample_exclusions)
            except Exception as exc:
                raise AnalysisError(
                    f"{variant_dict['family']}/{variant_dict['codec']} sample {sample_id}: {exc}"
                ) from exc
            completed.append(
                {
                    "sample_id": sample_id,
                    "family": variant_dict["family"],
                    "codec": variant_dict["codec"],
                    "config": variant_dict["config"],
                    "pa": pa,
                    "pc": pc,
                    "product": pa * pc,
                    "pa_units": pa_units,
                    "pc_targets": pc_targets,
                    "treatment_wells": len(treatment_ids),
                    "control_wells": int(sample.filter(pl.col(NEGCON)).height),
                    "total_wells": len(sample),
                }
            )
            if (sample_id + 1) % 25 == 0 or sample_id + 1 == n_samples:
                atomic_csv(checkpoint, pd.DataFrame(completed))
    finally:
        null.close()
        shutil.rmtree(os.environ["HOME"], ignore_errors=True)
    return {
        "family": variant_dict["family"],
        "codec": variant_dict["codec"],
        "rows": len(completed),
        "runtime_seconds": time.monotonic() - started,
        "checkpoint": str(checkpoint),
    }


def build_pc_exclusions(output_dir: Path, reference: Variant) -> dict[str, Any]:
    """Freeze metadata-only undefined-query exclusions for every sample."""
    frame = pl.read_parquet(
        reference.output, columns=[ID, COMPOUND, TARGET, NEGCON, GROUP, PLATE]
    )
    common = pl.read_csv(COMMON, schema_overrides={NEGCON: pl.Boolean})
    split = pl.read_csv(SPLIT).filter(pl.col("split") == "test")
    frame = frame.filter(
        pl.col(ID).is_in(common[ID].to_list())
        & pl.col(GROUP).is_in(PC_GROUPS)
        & (pl.col(NEGCON) | pl.col(COMPOUND).is_in(split["treatment_id"].to_list()))
    )
    id_to_index = {value: index for index, value in enumerate(frame[ID].to_list())}
    controls: dict[tuple[str, str], list[int]] = {}
    for index, plate, group in frame.with_row_index("index").filter(pl.col(NEGCON)).select(
        "index", PLATE, GROUP
    ).iter_rows():
        controls.setdefault((plate, group), []).append(index)
    wells = pl.read_parquet(output_dir / "manifests/sample_treatment_wells.parquet")
    pairs = pl.read_parquet(output_dir / "manifests/sample_plate_groups.parquet")
    rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    total_queries = 0
    for sample_id in sorted(wells["sample_id"].unique().to_list()):
        indexes = [id_to_index[value] for value in wells.filter(pl.col("sample_id") == sample_id)[ID]]
        for plate, group in pairs.filter(pl.col("sample_id") == sample_id).select(PLATE, GROUP).iter_rows():
            indexes.extend(controls[(plate, group)])
        sample_exclusions, sample_total = pc_query_eligibility(frame[sorted(set(indexes))])
        total_queries += sample_total
        for record in sample_exclusions:
            rows.append({"sample_id": sample_id, **record})
        for record in pc_removal_records(sample_exclusions):
            removed_rows.append({"sample_id": sample_id, **record})
    schema = {
        "sample_id": pl.Int64, COMPOUND: pl.String, GROUP: pl.String,
        "eligible_targets": pl.String, "reason": pl.String,
        "positive_candidate_count": pl.Int64, "negative_candidate_count": pl.Int64,
    }
    exclusions = pl.DataFrame(rows, schema=schema).sort(["sample_id", COMPOUND, GROUP])
    atomic_parquet_allow_empty(output_dir / "manifests/pc_undefined_queries.parquet", exclusions)
    removed = pl.DataFrame(removed_rows, schema=schema).sort(["sample_id", COMPOUND, GROUP])
    atomic_parquet_allow_empty(output_dir / "manifests/pc_removed_rows.parquet", removed)
    removed_profile_rows = (
        wells.join(removed.select(["sample_id", COMPOUND, GROUP]), on=["sample_id", COMPOUND, GROUP], how="inner").height
        if len(removed) else 0
    )
    return {
        "pc_consensus_query_count": total_queries,
        "pc_undefined_query_count": len(exclusions),
        "pc_undefined_query_fraction": len(exclusions) / total_queries,
        "pc_no_positive_query_count": exclusions.filter(pl.col("reason").str.contains("no_positive")).height,
        "pc_no_negative_query_count": exclusions.filter(pl.col("reason").str.contains("no_negative")).height,
        "pc_recorded_sample_count": exclusions["sample_id"].n_unique() if len(exclusions) else 0,
        "pc_recorded_sample_fraction": exclusions["sample_id"].n_unique() / wells["sample_id"].n_unique() if len(exclusions) else 0.0,
        "pc_removed_consensus_row_count": len(removed),
        "pc_removed_profile_row_count": removed_profile_rows,
        "pc_modified_sample_count": removed["sample_id"].n_unique() if len(removed) else 0,
        "pc_modified_sample_fraction": removed["sample_id"].n_unique() / wells["sample_id"].n_unique() if len(removed) else 0.0,
        "pc_record_only_query_count": len(exclusions) - len(removed),
    }


def summarize_pc_removals(output_dir: Path, manifest: dict[str, Any], n_samples: int) -> dict[str, Any]:
    """Reconstruct exact recorded-vs-removed PC incidence from frozen records."""
    exclusions = pl.read_parquet(output_dir / "manifests/pc_undefined_queries.parquet")
    removed_rows: list[dict[str, Any]] = []
    for sample_id, grouped in exclusions.to_pandas().groupby("sample_id", sort=True):
        for record in pc_removal_records(grouped.drop(columns="sample_id").to_dict("records")):
            removed_rows.append({"sample_id": int(sample_id), **record})
    schema = {
        "sample_id": pl.Int64, COMPOUND: pl.String, GROUP: pl.String,
        "eligible_targets": pl.String, "reason": pl.String,
        "positive_candidate_count": pl.Int64, "negative_candidate_count": pl.Int64,
    }
    removed = pl.DataFrame(removed_rows, schema=schema).sort(["sample_id", COMPOUND, GROUP])
    atomic_parquet_allow_empty(output_dir / "manifests/pc_removed_rows.parquet", removed)
    manifest = dict(manifest)
    wells = pl.read_parquet(output_dir / "manifests/sample_treatment_wells.parquet")
    removed_profile_rows = (
        wells.join(removed.select(["sample_id", COMPOUND, GROUP]), on=["sample_id", COMPOUND, GROUP], how="inner").height
        if len(removed) else 0
    )
    recorded_samples = exclusions["sample_id"].n_unique() if len(exclusions) else 0
    modified_samples = removed["sample_id"].n_unique() if len(removed) else 0
    manifest.update({
        "pc_recorded_sample_count": recorded_samples,
        "pc_recorded_sample_fraction": recorded_samples / n_samples,
        "pc_removed_consensus_row_count": len(removed),
        "pc_removed_profile_row_count": removed_profile_rows,
        "pc_modified_sample_count": modified_samples,
        "pc_modified_sample_fraction": modified_samples / n_samples,
        "pc_record_only_query_count": len(exclusions) - len(removed),
    })
    manifest.pop("pc_affected_sample_count", None)
    manifest.pop("pc_affected_sample_fraction", None)
    return manifest


def run_full(
    output_dir: Path,
    variants: list[Variant],
    manifest_info: dict[str, Any],
    workers: int,
    n_samples: int = N_SAMPLES,
    protocol_id: str = "test-protocol",
    write_combined: bool = True,
) -> list[dict[str, Any]]:
    payloads = [
        {
            "family": value.family,
            "codec": value.codec,
            "config": value.config,
            "output": str(value.output),
            "sha256": value.sha256,
            "size": value.size,
            "config_sha256": sha256_file(value.config_path),
            "runner_sha256": sha256_file(Path(__file__)),
        }
        for value in variants
    ]
    if workers <= 0 or n_samples <= 0:
        raise AnalysisError("workers and sample count must be positive")
    results: list[dict[str, Any]] = []
    # Set thread caps before spawn imports NumPy, then isolate mutable HOME/cache
    # inside each process. No worker mutates environment shared with another.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    with ProcessPoolExecutor(
        max_workers=min(workers, len(payloads)), mp_context=mp.get_context("spawn")
    ) as pool:
        futures = {
            pool.submit(
                full_worker, payload, str(output_dir), manifest_info["annotation_digest"],
                n_samples, protocol_id,
            ): payload
            for payload in payloads
        }
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: (FAMILIES.index(row["family"]), CODECS.index(row["codec"])))
    if write_combined:
        frames = [pd.read_csv(row["checkpoint"]) for row in results]
        combined = pd.concat(frames, ignore_index=True).sort_values(["sample_id", "family", "codec"])
        if len(combined) != n_samples * len(variants):
            raise AnalysisError("incomplete full subsample result")
        atomic_parquet(output_dir / "results/full_subsample_metrics.parquet", pl.from_pandas(combined))
    return results


def centered_pvalue(samples: np.ndarray, point: float) -> float:
    centered = samples - point
    return (np.count_nonzero(np.abs(centered) >= abs(point)) + 1.0) / (len(samples) + 1.0)


def holm_adjust(values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(values), dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise AnalysisError("invalid p-values")
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adjusted_ranked = np.maximum.accumulate((len(p) - np.arange(len(p))) * ranked)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def select_target_recipes(sweep: pd.DataFrame) -> dict[str, str]:
    required = {"model", "config", "PA", "PC", "PA_mean_nap", "PC_mean_nap"}
    if missing := required.difference(sweep.columns):
        raise AnalysisError(f"sweep summary lacks columns: {sorted(missing)}")
    winners: dict[str, str] = {}
    for family, prefix in TARGET_FOLDERS.items():
        model = f"{prefix}_zstd_raw"
        rows = sweep[sweep.model == model].copy()
        if rows.empty or rows.config.duplicated().any():
            raise AnalysisError(f"missing/duplicate Zstd candidates: {family}")
        rows["selection_metric"] = rows.PA * rows.PC / 100.0
        best = rows.selection_metric.max()
        tied = rows[np.isclose(rows.selection_metric, best, rtol=0.0, atol=1e-15)]
        winners[family] = sorted(tied.config.astype(str))[0]
    return winners


def target_input_hashes(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, str]]:
    fields = ("metrics_sha256", "pa_sha256", "pc_sha256", "config_sha256", "output_sha256")
    return {
        (str(row["family"]), str(row["codec"])): {field: str(row[field]) for field in fields}
        for row in records
    }


def verify_frozen_target_records(
    current: list[dict[str, Any]], frozen: list[dict[str, Any]]
) -> None:
    """Fail before summary regeneration if any production Target-2 input drifted."""
    if target_input_hashes(current) != target_input_hashes(frozen):
        raise AnalysisError("summarize-only Target-2 frozen input drift")


def validate_target_inputs(
    output_dir: Path, write_manifest: bool = True
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    sweep_path = TARGET_SWEEP / "sweep_results.csv"
    sweep = pd.read_csv(sweep_path)
    winners = select_target_recipes(sweep)
    pa_tables: list[pd.DataFrame] = []
    pc_tables: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for family, prefix in TARGET_FOLDERS.items():
        config = winners[family]
        for codec, folder_codec, display_codec in (
            ("Zstd", "zstd", "Zstd"),
            ("D10", "jpegxl_lossy_d10", "D10"),
            ("D15", "jpegxl_lossy_d15", "D15"),
        ):
            folder = TARGET_SWEEP / f"{prefix}_jump_target2_4plate_{folder_codec}_raw_features" / config
            if not folder.is_dir():
                raise AnalysisError(f"fixed Target-2 recipe missing: {folder}")
            metrics_path = folder / "results/metrics.json"
            pa_path = folder / "results/phenotypic_activity_map.csv"
            pc_path = folder / "results/phenotypic_consistency_per_target.csv"
            config_path = folder / "pipeline_config.yaml"
            output_path = folder / "output.parquet"
            for path in (metrics_path, pa_path, pc_path, config_path, output_path):
                file_record(path)
            metrics = json.loads(metrics_path.read_text())
            pa = pd.read_csv(pa_path)
            pc = pd.read_csv(pc_path)
            pa_key = "Metadata_broad_sample"
            pc_key = "Metadata_target"
            if pa_key not in pa or pc_key not in pc or pa[pa_key].duplicated().any() or pc[pc_key].duplicated().any():
                raise AnalysisError(f"invalid Target-2 per-unit keys: {folder}")
            if len(pa) != 306 or len(pc) != 201:
                raise AnalysisError(f"unexpected Target-2 per-unit counts: {folder}")
            pa_point = float(pa.mean_normalized_average_precision.mean())
            pc_point = float(pc.mean_normalized_average_precision.mean())
            if abs(pa_point - float(metrics["PA_mean_nap"])) > 1e-12 or abs(pc_point - float(metrics["PC_mean_nap"])) > 1e-12:
                raise AnalysisError(f"Target-2 metrics.json mismatch: {folder}")
            summary_row = sweep[(sweep.model == f"{prefix}_{folder_codec}_raw") & (sweep.config == config)]
            if len(summary_row) != 1:
                raise AnalysisError(f"Target-2 sweep summary coverage mismatch: {folder}")
            if abs(float(summary_row.iloc[0].PA) - float(metrics["PA"])) > 1e-12 or abs(float(summary_row.iloc[0].PC) - float(metrics["PC"])) > 1e-12:
                raise AnalysisError(f"Target-2 sweep summary point mismatch: {folder}")
            pa = pa[[pa_key, "mean_normalized_average_precision"]].rename(
                columns={"mean_normalized_average_precision": f"{family}__{display_codec}"}
            )
            pc = pc[[pc_key, "mean_normalized_average_precision"]].rename(
                columns={"mean_normalized_average_precision": f"{family}__{display_codec}"}
            )
            pa_tables.append(pa)
            pc_tables.append(pc)
            records.append(
                {
                    "family": family,
                    "codec": display_codec,
                    "config": config,
                    "selection_metric": "PA*PC/100 on Zstd only",
                    "pa_point": pa_point,
                    "pc_point": pc_point,
                    "product_point": pa_point * pc_point,
                    "metrics_path": str(metrics_path),
                    "metrics_sha256": sha256_file(metrics_path),
                    "pa_path": str(pa_path),
                    "pa_sha256": sha256_file(pa_path),
                    "pc_path": str(pc_path),
                    "pc_sha256": sha256_file(pc_path),
                    "config_path": str(config_path),
                    "config_sha256": sha256_file(config_path),
                    "output_path": str(output_path),
                    "output_sha256": sha256_file(output_path),
                }
            )
    pa_aligned = align_target_tables(pa_tables, "Metadata_broad_sample")
    pc_aligned = align_target_tables(pc_tables, "Metadata_target")
    if write_manifest:
        atomic_csv(output_dir / "manifests/target2_selected_recipes.csv", pd.DataFrame(records))
    return records, pa_aligned, pc_aligned


def align_target_tables(tables: list[pd.DataFrame], key: str) -> pd.DataFrame:
    if not tables or any(key not in table for table in tables):
        raise AnalysisError(f"Target-2 alignment lacks key {key}")
    merged = tables[0]
    if merged[key].duplicated().any():
        raise AnalysisError(f"Target-2 duplicate {key}")
    expected = set(merged[key].astype(str))
    for table in tables[1:]:
        if table[key].duplicated().any() or set(table[key].astype(str)) != expected:
            raise AnalysisError(f"Target-2 {key} key-set mismatch")
        merged = merged.merge(table, on=key, how="inner", validate="one_to_one")
    return merged.sort_values(key).reset_index(drop=True)


def target_bootstrap(
    output_dir: Path,
    records: list[dict[str, Any]],
    pa: pd.DataFrame,
    pc: pd.DataFrame,
    replicates: int = TARGET_BOOTSTRAPS,
    seed: int = SEED,
) -> dict[str, Any]:
    columns = [column for column in pa if column != "Metadata_broad_sample"]
    if columns != [column for column in pc if column != "Metadata_target"]:
        raise AnalysisError("Target-2 PA/PC variant columns differ")
    pa_values = pa[columns].to_numpy(float)
    pc_values = pc[columns].to_numpy(float)
    if not np.isfinite(pa_values).all() or not np.isfinite(pc_values).all():
        raise AnalysisError("non-finite Target-2 per-unit values")
    if replicates <= 0:
        raise AnalysisError("bootstrap replicate count must be positive")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    pa_boot = np.empty((replicates, len(columns)), dtype=float)
    pc_boot = np.empty_like(pa_boot)
    batch_size = 250
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        pa_weights = rng.multinomial(len(pa_values), np.full(len(pa_values), 1 / len(pa_values)), size=size)
        pc_weights = rng.multinomial(len(pc_values), np.full(len(pc_values), 1 / len(pc_values)), size=size)
        pa_boot[start:start + size] = pa_weights @ pa_values / len(pa_values)
        pc_boot[start:start + size] = pc_weights @ pc_values / len(pc_values)
    product_boot = pa_boot * pc_boot
    pa_points = pa_values.mean(axis=0)
    pc_points = pc_values.mean(axis=0)
    product_points = pa_points * pc_points
    summaries: list[dict[str, Any]] = []
    for index, column in enumerate(columns):
        family, codec = column.split("__")
        pa_low, pa_high = interval(pa_boot[:, index])
        pc_low, pc_high = interval(pc_boot[:, index])
        prod_low, prod_high = interval(product_boot[:, index])
        summaries.append(
            {
                "family": family,
                "codec": codec,
                "config": next(row["config"] for row in records if row["family"] == family),
                "pa_point": pa_points[index], "pa_ci_low": pa_low, "pa_ci_high": pa_high,
                "pc_point": pc_points[index], "pc_ci_low": pc_low, "pc_ci_high": pc_high,
                "product_point": product_points[index], "product_ci_low": prod_low, "product_ci_high": prod_high,
                "replicates": replicates, "seed": seed,
            }
        )
    contrasts: list[dict[str, Any]] = []
    for family in FAMILIES:
        d10 = columns.index(f"{family}__D10")
        d15 = columns.index(f"{family}__D15")
        product_diff = product_boot[:, d15] - product_boot[:, d10]
        product_low, product_high = interval(product_diff)
        contrasts.append(
            {
                "family": family,
                "d10_product": product_points[d10], "d15_product": product_points[d15],
                "product_delta_d15_minus_d10": product_points[d15] - product_points[d10],
                "product_delta_ci_low": product_low, "product_delta_ci_high": product_high,
                "product_centered_bootstrap_p": centered_pvalue(
                    product_diff, product_points[d15] - product_points[d10]
                ),
            }
        )
        row = contrasts[-1]
        for metric, points, boot in (("pa", pa_points, pa_boot), ("pc", pc_points, pc_boot)):
            diff = boot[:, d15] - boot[:, d10]
            low, high = interval(diff)
            row.update(
                {
                    f"{metric}_d10": points[d10], f"{metric}_d15": points[d15],
                    f"{metric}_delta_d15_minus_d10": points[d15] - points[d10],
                    f"{metric}_delta_ci_low": low, f"{metric}_delta_ci_high": high,
                }
            )
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame["product_holm_p"] = holm_adjust(contrast_frame.product_centered_bootstrap_p)
    contrast_frame["product_supported_direction"] = np.where(
        contrast_frame.product_holm_p < 0.05,
        np.where(contrast_frame.product_delta_d15_minus_d10 > 0, "D15>D10", "D10>D15"),
        "unresolved",
    )
    summary_frame = pd.DataFrame(summaries)
    atomic_csv(output_dir / "results/target2_score_intervals.csv", summary_frame)
    atomic_csv(output_dir / "results/target2_d15_vs_d10.csv", contrast_frame)
    return {"summary": summary_frame, "contrasts": contrast_frame}


def rank_correlation(values: dict[str, float], baseline: dict[str, float]) -> float:
    a = pd.Series([values[codec] for codec in CODECS]).rank(method="average").to_numpy()
    b = pd.Series([baseline[codec] for codec in CODECS]).rank(method="average").to_numpy()
    if np.std(a) == 0 or np.std(b) == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


MATCHED_FULL_IDENTITY_FIELDS = (
    "family", "codec", "pa_path", "pa_sha256", "pc_path", "pc_sha256",
)


def matched_full_input_identities(records: Iterable[dict[str, Any]]) -> list[dict[str, str]]:
    """Canonicalize the complete matched-full source-file identity."""
    identities = [
        {field: str(row[field]) for field in MATCHED_FULL_IDENTITY_FIELDS}
        for row in records
    ]
    identities.sort(key=lambda row: (row["family"], row["codec"]))
    expected = {(family, codec) for family in FAMILIES for codec in CODECS}
    observed = {(row["family"], row["codec"]) for row in identities}
    if len(identities) != len(expected) or observed != expected:
        raise AnalysisError("incomplete/duplicate matched-full frozen identity")
    return identities


def frozen_matched_full_inputs(provenance: dict[str, Any], output_dir: Path) -> list[dict[str, str]]:
    """Load canonical provenance identity, with a one-time legacy CSV migration path."""
    records = provenance.get("matched_full_inputs")
    if records is None:
        legacy = pd.read_csv(output_dir / "results/matched_full_baseline.csv")
        records = legacy.to_dict("records")
    return matched_full_input_identities(records)


def verify_frozen_matched_full_records(
    current: Iterable[dict[str, Any]], frozen: Iterable[dict[str, Any]]
) -> None:
    """Fail before summary regeneration if any matched-full source file drifted."""
    if matched_full_input_identities(current) != matched_full_input_identities(frozen):
        raise AnalysisError("summarize-only matched-full frozen input drift")


def matched_full_baseline() -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Build key-aligned group_high/group_low baseline from frozen per-unit tables."""
    baseline: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    key_sets: dict[str, set[tuple[str, str]]] = {}
    for family in FAMILIES:
        baseline[family] = {}
        for codec in CODECS:
            pa_path = PER_UNIT / f"{family}__{codec}__pa_treatments.csv"
            pc_path = PER_UNIT / f"{family}__{codec}__pc_targets.csv"
            pa = pd.read_csv(pa_path)
            pc = pd.read_csv(pc_path)
            pa = pa[pa[GROUP].isin(PC_GROUPS)].copy()
            pc = pc[pc[GROUP].isin(PC_GROUPS)].copy()
            pa_keys = set(map(tuple, pa[[COMPOUND, GROUP]].astype(str).to_numpy()))
            pc_keys = set(map(tuple, pc[["Metadata_target", GROUP]].astype(str).to_numpy()))
            for label, keys in (("pa", pa_keys), ("pc", pc_keys)):
                if label in key_sets and keys != key_sets[label]:
                    raise AnalysisError(f"matched full {label.upper()} key-set mismatch: {family}/{codec}")
                key_sets.setdefault(label, keys)
            if len(pa) != len(pa_keys) or len(pc) != len(pc_keys):
                raise AnalysisError(f"duplicate matched full per-unit key: {family}/{codec}")
            pa_mean = float(pa.mean_normalized_average_precision.mean())
            pc_mean = float(pc.mean_normalized_average_precision.mean())
            product = pa_mean * pc_mean
            baseline[family][codec] = product
            rows.append({
                "family": family, "codec": codec, "pa": pa_mean, "pc": pc_mean,
                "product": product, "pa_units": len(pa), "pc_targets": len(pc),
                "pa_path": str(pa_path), "pa_sha256": sha256_file(pa_path),
                "pc_path": str(pc_path), "pc_sha256": sha256_file(pc_path),
                "groups": "group_high|group_low",
            })
    return baseline, pd.DataFrame(rows)


def summarize_full(
    output_dir: Path,
    n_samples: int = N_SAMPLES,
    matched_full: tuple[dict[str, dict[str, float]], pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    metrics = pl.read_parquet(output_dir / "results/full_subsample_metrics.parquet").to_pandas()
    baseline, baseline_frame = matched_full or matched_full_baseline()
    sample_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for sample_id in range(n_samples):
            frame = metrics[(metrics.family == family) & (metrics.sample_id == sample_id)].set_index("codec")
            if set(frame.index) != set(CODECS):
                raise AnalysisError("incomplete sample family/codec result")
            products = frame["product"].to_dict()
            row = {
                "sample_id": sample_id,
                "family": family,
                "rank_correlation_vs_matched_full": rank_correlation(products, baseline[family]),
            }
            reversals = []
            for high, low in EXPECTED_PAIRS:
                reversal = products[low] > products[high]
                row[f"reversal_{low}_gt_{high}"] = reversal
                reversals.append(reversal)
            row["any_adjacent_reversal"] = any(reversals)
            sample_rows.append(row)
            for codec_a, codec_b in combinations(CODECS, 2):
                sample_delta = products[codec_a] - products[codec_b]
                matched_full_delta = baseline[family][codec_a] - baseline[family][codec_b]
                pair_rows.append(
                    {
                        "sample_id": sample_id, "family": family,
                        "codec_a": codec_a, "codec_b": codec_b,
                        "sample_delta_a_minus_b": sample_delta,
                        "matched_full_delta_a_minus_b": matched_full_delta,
                        "discordant_vs_matched_full": np.sign(sample_delta) != np.sign(matched_full_delta),
                    }
                )
    samples = pd.DataFrame(sample_rows)
    pairs = pd.DataFrame(pair_rows)
    distribution_rows: list[dict[str, Any]] = []
    for (family, codec), frame in metrics.groupby(["family", "codec"], sort=True):
        for metric in ("pa", "pc", "product"):
            values = frame[metric].to_numpy(float)
            distribution_rows.append(
                {
                    "family": family, "codec": codec, "metric": metric,
                    "mean": values.mean(), "sd": values.std(ddof=1 if len(values) > 1 else 0),
                    "q025": np.quantile(values, 0.025), "median": np.median(values),
                    "q975": np.quantile(values, 0.975), "min": values.min(), "max": values.max(),
                }
            )
    reversals = samples.groupby("family", sort=True).agg(
        any_adjacent_reversal_rate=("any_adjacent_reversal", "mean"),
        d20_gt_mq_rate=("reversal_D20_gt_MQ", "mean"),
        mq_gt_hq_rate=("reversal_MQ_gt_HQ", "mean"),
        hq_gt_raw_rate=("reversal_HQ_gt_Raw", "mean"),
        rank_correlation_mean=("rank_correlation_vs_matched_full", "mean"),
        rank_correlation_median=("rank_correlation_vs_matched_full", "median"),
    ).reset_index()
    discordance = pairs.groupby(["family", "codec_a", "codec_b"], sort=True).agg(
        discordance_rate=("discordant_vs_matched_full", "mean"),
        delta_mean=("sample_delta_a_minus_b", "mean"),
        delta_q025=("sample_delta_a_minus_b", lambda value: np.quantile(value, 0.025)),
        delta_q975=("sample_delta_a_minus_b", lambda value: np.quantile(value, 0.975)),
        matched_full_delta=("matched_full_delta_a_minus_b", "first"),
    ).reset_index()
    atomic_csv(output_dir / "results/matched_full_baseline.csv", baseline_frame)
    atomic_csv(output_dir / "results/full_sample_ordering.csv", samples)
    atomic_csv(output_dir / "results/full_pairwise_discordance.csv", discordance)
    atomic_csv(output_dir / "results/full_metric_distributions.csv", pd.DataFrame(distribution_rows))
    atomic_csv(output_dir / "results/full_reversal_summary.csv", reversals)
    return {"samples": samples, "pairs": discordance, "distributions": pd.DataFrame(distribution_rows), "reversals": reversals, "baseline": baseline_frame}


def plot_results(output_dir: Path, full: dict[str, pd.DataFrame], target: dict[str, Any]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    rev = full["reversals"].set_index("family").loc[list(FAMILIES)]
    x = np.arange(len(FAMILIES))
    width = 0.22
    for index, (column, label) in enumerate(
        (("hq_gt_raw_rate", "HQ > Raw"), ("mq_gt_hq_rate", "MQ > HQ"), ("d20_gt_mq_rate", "D20 > MQ"))
    ):
        axes[0].bar(x + (index - 1) * width, 100 * rev[column], width, label=label)
    axes[0].set_xticks(x, [DISPLAY[f] for f in FAMILIES], rotation=20, ha="right")
    axes[0].set_ylabel("Subsamples with reversal (%)")
    axes[0].set_title("A. JUMP-lite 306-ID subsamples")
    axes[0].legend(frameon=False, fontsize=8)

    contrast = target["contrasts"].set_index("family").loc[list(FAMILIES)]
    y = np.arange(len(FAMILIES))
    points = contrast.product_delta_d15_minus_d10.to_numpy()
    low = contrast.product_delta_ci_low.to_numpy()
    high = contrast.product_delta_ci_high.to_numpy()
    axes[1].errorbar(points, y, xerr=np.vstack([points - low, high - points]), fmt="o", color="#2166ac", capsize=3)
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_yticks(y, [DISPLAY[f] for f in FAMILIES])
    axes[1].set_xlabel("D15 − D10 PA–PC product")
    axes[1].set_title("B. Target-2 paired bootstrap")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"compression_order_robustness.{suffix}", dpi=220, bbox_inches="tight", metadata={"CreationDate": None, "ModDate": None} if suffix == "pdf" else None)
    plt.close(fig)


def render_report(
    output_dir: Path,
    manifest: dict[str, Any],
    full: dict[str, pd.DataFrame],
    target: dict[str, Any],
    runtimes: dict[str, Any],
    spec: RunSpec,
) -> None:
    rev = full["reversals"].set_index("family")
    target_rows = target["contrasts"].set_index("family")
    lines = [
        "# Compression-order robustness", "", "",
        "## Direct answer", "",
        "Small fixed-recipe cohorts frequently change adjacent codec orderings, so the non-monotonic Target-2 ordering cannot by itself be interpreted as compression improving biological signal. The direct D10/D15 result is model-dependent under a Zstd-selected fixed recipe.", "",
        "## JUMP-lite 306-treatment sensitivity", "",
        f"We generated {spec.n_samples:,} deterministic stratified samples (seed `{spec.seed}`), each with exactly {N_TREATMENTS} held-out treatment-ID clusters. Each observed treatment/group unit contributed exactly four distinct wells, and all frozen common controls from represented plate/group pairs were included. The same manifests were applied to all 16 fixed model/codec profiles.", "",
        "This is a stratified 306-ID/four-well-per-treatment-group sensitivity, not a literal four-plate replica and not an analysis of D10/D15. It conditions on archived transformations and the Raw-validation-selected recipes.", "",
        f"After the established PC target-count/promiscuity filtering, {manifest['pc_undefined_query_count']} of {manifest['pc_consensus_query_count']} compound/group query rows ({100*manifest['pc_undefined_query_fraction']:.2f}%) in {manifest['pc_recorded_sample_count']} samples ({100*manifest['pc_recorded_sample_fraction']:.2f}%) could not themselves define retrieval AP ({manifest['pc_no_positive_query_count']} lacked an eligible positive; {manifest['pc_no_negative_query_count']} lacked a disjoint eligible negative). Of these, {manifest['pc_record_only_query_count']} were record-only no-positive queries that copairs naturally does not emit. Actual row removal occurred for {manifest['pc_removed_consensus_row_count']} compound/group rows ({manifest['pc_removed_profile_row_count']} selected profile rows) in {manifest['pc_modified_sample_count']} of {spec.n_samples} samples ({100*manifest['pc_modified_sample_fraction']:.2f}%): two no-negative rows and one wholly no-positive group of 11 rows. For those affected samples only, PC is explicitly a modified restricted estimand after removing undefined rows; it is not the unchanged established PC estimand. Metadata-only records and removals were frozen in `manifests/pc_undefined_queries.parquet` and `manifests/pc_removed_rows.parquet` and applied identically to all 16 variants. PA and sample manifests were unchanged, and no other PC failure is suppressed.", "",
        "The primary inference is the adjacent within-sample reversal rate. As a secondary comparison, rank correlation and six-pair discordance use a directly matched group_high/group_low full-population baseline computed from key-aligned archived per-unit PA (3,654 treatment/group rows) and PC (669 target/group rows); this is not the all-group heldout_test_scores.csv product.", "",
        "| Model | HQ > Raw | MQ > HQ | D20 > MQ | Any adjacent reversal | Mean rank rho vs matched full |", "|---|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        row = rev.loc[family]
        lines.append(f"| {DISPLAY[family]} | {100*row.hq_gt_raw_rate:.1f}% | {100*row.mq_gt_hq_rate:.1f}% | {100*row.d20_gt_mq_rate:.1f}% | {100*row.any_adjacent_reversal_rate:.1f}% | {row.rank_correlation_mean:.3f} |")
    lines.extend(["", "The matched baseline is in `results/matched_full_baseline.csv`; all six pairwise codec discordance rates and PA/PC/product distributions are in `results/full_pairwise_discordance.csv` and `results/full_metric_distributions.csv`.", "", "## Target-2 fixed-recipe D10 versus D15", "", f"One recipe per learned model was selected using Zstd only by the manuscript metric PA×PC/100, with lexical tie resolution. That exact recipe was required for Zstd, D10, and D15. PA's 306 compound clusters and PC's 201 target clusters were resampled with common weights across all models/codecs within each margin for {spec.target_bootstraps:,} replicates (seed `{spec.seed}`). PA and PC were independently sampled and multiplied under a working product-of-margins model.", "", "| Model | Zstd product | D10 product | D15 product | D15 − D10 (95% interval) | Holm result |", "|---|---:|---:|---:|---:|---|",])
    summary = target["summary"]
    for family in FAMILIES:
        row = target_rows.loc[family]
        zstd = summary[(summary.family == family) & (summary.codec == "Zstd")].iloc[0]
        lines.append(f"| {DISPLAY[family]} | {zstd.product_point:.5f} | {row.d10_product:.5f} | {row.d15_product:.5f} | {row.product_delta_d15_minus_d10:+.5f} [{row.product_delta_ci_low:+.5f}, {row.product_delta_ci_high:+.5f}] | {row.product_supported_direction} ($p_{{Holm}}={row.product_holm_p:.4g}$) |")
    lines.extend(["", "Percentile intervals are pointwise. The four D15-vs-D10 product tests use Holm correction. The product intervals/tests omit unknown PA–PC covariance and may be too narrow or too wide; they are conditional on frozen per-unit retrieval outputs, target eligibility, and Zstd-selected recipes. Independent per-codec winners are not used.", "", "## Provenance and runtime", "", f"- Eligible stratum counts: `{manifest['eligible_stratum_counts']}`; quotas: `{manifest['quotas']}`.", f"- Full subsampling wall time was {runtimes['full_seconds']:.1f} seconds with spawn-isolated variant workers; Target-2 bootstrap took {runtimes['target_seconds']:.1f} seconds.", "- Canonical archives were read only; no normalization, extraction, or sweep was rerun.", "- See `provenance.json`, `artifact_checksums.json`, and `manifests/` for frozen identities and hashes.", "", "![Compression-order robustness](compression_order_robustness.png)", ""])
    atomic_text(output_dir / "REPORT.md", "\n".join(lines))


def write_provenance(
    output_dir: Path,
    variants: list[Variant],
    manifest: dict[str, Any],
    target_records: list[dict[str, Any]],
    matched_full_records: Iterable[dict[str, Any]],
    runtimes: dict[str, Any],
    spec: RunSpec,
    protocol_id: str,
) -> None:
    provenance = {
        "protocol_version": PROTOCOL_VERSION,
        "completed_at": runtimes.get("completed_at", utc_now()),
        "repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runner_sha256": sha256_file(Path(__file__)),
        "protocol_id": protocol_id,
        "production_protocol": spec.production,
        "seed": spec.seed, "n_samples": spec.n_samples, "n_treatments": N_TREATMENTS,
        "wells_per_treatment_group": WELLS_PER_UNIT, "target_bootstraps": spec.target_bootstraps,
        "manifest": manifest, "runtimes": runtimes,
        "software": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "polars": pl.__version__},
        "full_variants": [value.__dict__ | {"output": str(value.output), "config_path": str(value.config_path)} for value in variants],
        "target2_inputs": target_records,
        "matched_full_inputs": matched_full_input_identities(matched_full_records),
        "qualification": "Conditional archived-profile sensitivity; full samples are not literal four-plate replicas; affected subsample PCs are modified restricted estimands; PA and PC Target-2 bootstrap margins use a working independence approximation.",
    }
    atomic_json(output_dir / "provenance.json", provenance)


def write_checksums(output_dir: Path) -> None:
    final_paths = sorted(
        path for path in output_dir.rglob("*")
        if path.is_file() and "checkpoints" not in path.relative_to(output_dir).parts
        and "cache" not in path.relative_to(output_dir).parts
        and path.name != "artifact_checksums.json"
    )
    checkpoint_paths = sorted(
        path for path in (output_dir / "checkpoints").rglob("*") if path.is_file()
    ) if (output_dir / "checkpoints").is_dir() else []
    payload = {
        "path_semantics": "paths are relative to the directory containing this inventory",
        "final_release_artifacts": [release_file_record(path, output_dir) for path in final_paths],
        "retained_checkpoint_state": [release_file_record(path, output_dir) for path in checkpoint_paths],
    }
    atomic_json(output_dir / "artifact_checksums.json", payload)


def verify_checksums(output_dir: Path) -> None:
    inventory = output_dir / "artifact_checksums.json"
    payload = json.loads(inventory.read_text())
    if payload.get("path_semantics") != "paths are relative to the directory containing this inventory":
        raise AnalysisError("unsupported or non-relocatable artifact checksum inventory")
    for section in ("final_release_artifacts", "retained_checkpoint_state"):
        for record in payload.get(section, []):
            relative = Path(record["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise AnalysisError(f"unsafe artifact checksum path: {relative}")
            path = output_dir / relative
            if (not path.is_file() or path.stat().st_size != record["size_bytes"]
                    or sha256_file(path) != record["sha256"]):
                raise AnalysisError(f"artifact checksum mismatch: {relative}")


def protocol_payload(variants: list[Variant], spec: RunSpec) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "runner_sha256": sha256_file(Path(__file__)),
        "seed": spec.seed,
        "n_samples": spec.n_samples,
        "n_treatments": N_TREATMENTS,
        "wells_per_unit": WELLS_PER_UNIT,
        "target_bootstraps": spec.target_bootstraps,
        "pc_groups": list(PC_GROUPS),
        "codecs": list(CODECS),
        "expected_pairs": list(EXPECTED_PAIRS),
        "variants": [
            {
                "family": v.family, "codec": v.codec, "config": v.config,
                "output_sha256": v.sha256, "output_size_bytes": v.size,
                "config_sha256": sha256_file(v.config_path),
            }
            for v in variants
        ],
        "frozen_support_inputs": {
            str(path.resolve()): {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in (CHECKSUMS, COMMON, SPLIT, SELECTED, HELDOUT, POST_PROVENANCE)
        },
    }


def protocol_identity(variants: list[Variant], spec: RunSpec) -> str:
    return hashlib.sha256(json.dumps(protocol_payload(variants, spec), sort_keys=True).encode()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/release_v1")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true",
        help="run isolated 1-sample/250-bootstrap smoke protocol (requires non-release output dir)",
    )
    parser.add_argument(
        "--variant", choices=[f"{family}:{codec}" for family in FAMILIES for codec in CODECS],
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    if args.verify_only:
        verify_checksums(output_dir)
        print("checksum validation passed")
        return
    spec = RunSpec(n_samples=1, target_bootstraps=250) if args.smoke else RunSpec()
    release_dir = (HERE / "outputs/release_v1").resolve()
    if args.smoke and output_dir == release_dir:
        raise AnalysisError("smoke mode requires an output directory distinct from release_v1")
    variants, _ = load_variants()
    protocol_variants = variants
    protocol_id = protocol_identity(protocol_variants, spec)
    run_variants = variants
    if args.variant:
        if args.smoke or args.summarize_only:
            raise AnalysisError("--variant is only valid for production checkpoint completion")
        family, codec = args.variant.split(":")
        run_variants = [v for v in variants if v.family == family and v.codec == codec]
    if args.summarize_only:
        verify_checksums(output_dir)
        old = json.loads((output_dir / "provenance.json").read_text())
        # A runner hardening change intentionally changes protocol_id; summary
        # regeneration is safe only after all frozen full, Target-2, and
        # matched-full per-unit inputs are revalidated. The expensive metric
        # parquet is the sole scored subsample input.
        old_variants = {
            (row["family"], row["codec"]): (row["sha256"], row["size"])
            for row in old.get("full_variants", [])
        }
        current_variants = {
            (value.family, value.codec): (value.sha256, value.size) for value in variants
        }
        if old_variants != current_variants:
            raise AnalysisError("summarize-only full input drift")
        records, pa, pc = validate_target_inputs(output_dir, write_manifest=False)
        verify_frozen_target_records(records, old.get("target2_inputs", []))
        matched_full = matched_full_baseline()
        verify_frozen_matched_full_records(
            matched_full[1].to_dict("records"), frozen_matched_full_inputs(old, output_dir)
        )
        # All current frozen sources have now been validated; only now may
        # summarize-only replace any release artifact.
        atomic_csv(output_dir / "manifests/target2_selected_recipes.csv", pd.DataFrame(records))
        full = summarize_full(output_dir, spec.n_samples, matched_full=matched_full)
        target = target_bootstrap(
            output_dir, records, pa, pc, spec.target_bootstraps, spec.seed
        )
        runtimes = old["runtimes"]
        manifest = summarize_pc_removals(output_dir, old["manifest"], spec.n_samples)
        plot_results(output_dir, full, target)
        render_report(output_dir, manifest, full, target, runtimes, spec)
        write_provenance(
            output_dir, variants, manifest, records, full["baseline"].to_dict("records"),
            runtimes, spec, protocol_id,
        )
        write_checksums(output_dir)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    started_all = time.monotonic()
    manifest = build_manifests(output_dir, variants, spec.n_samples)
    exclusion_info = build_pc_exclusions(output_dir, variants[0])
    manifest.update(exclusion_info)
    started = time.monotonic()
    workers = run_full(
        output_dir, run_variants, manifest, args.workers, spec.n_samples, protocol_id,
        write_combined=not bool(args.variant),
    )
    full_seconds = time.monotonic() - started
    if args.variant:
        print(json.dumps({"output_dir": str(output_dir), "workers": workers}, indent=2))
        return
    full = summarize_full(output_dir, spec.n_samples)
    started = time.monotonic()
    records, pa, pc = validate_target_inputs(output_dir)
    target = target_bootstrap(
        output_dir, records, pa, pc, spec.target_bootstraps, spec.seed
    )
    target_seconds = time.monotonic() - started
    runtimes = {
        "full_seconds": full_seconds, "target_seconds": target_seconds,
        "total_seconds": time.monotonic() - started_all, "workers": workers,
        "completed_at": utc_now(),
    }
    plot_results(output_dir, full, target)
    render_report(output_dir, manifest, full, target, runtimes, spec)
    write_provenance(
        output_dir, variants, manifest, records, full["baseline"].to_dict("records"),
        runtimes, spec, protocol_id,
    )
    shutil.rmtree(output_dir / "cache", ignore_errors=True)
    write_checksums(output_dir)
    verify_checksums(output_dir)
    print(json.dumps({"output_dir": str(output_dir), "runtimes": runtimes}, indent=2))


if __name__ == "__main__":
    run(parse_args())
