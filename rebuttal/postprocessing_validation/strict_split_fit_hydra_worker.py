#!/usr/bin/env python3
"""Hydra Joblib worker for strict split-before-fit selection prefix groups.

Workers write one immutable result package per manifest task.  They never write
canonical selection checkpoints or final production outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from rebuttal.postprocessing_validation.run_analysis import AnalysisError, sha256_file
from rebuttal.postprocessing_validation.strict_split_fit import (
    EffectiveRecipe,
    _gpu,
    atomic_create_json,
    build_annotation_table,
    canonical_json_sha256,
    complete_recipe_from_prefix,
    discover_effective_recipes,
    exact_fit_ids_hash,
    fit_recipe_prefix,
    fit_transform_recipe,
    load_raw_profile,
    release_prefix_cache,
    score_partition,
)


@dataclass
class HydraWorkerConfig:
    coordinator_root: str = MISSING
    task_index: int = MISSING


ConfigStore.instance().store(name="strict_selection_worker", node=HydraWorkerConfig)


def _load_bound_task(
    root: Path, task_index: int
) -> tuple[dict[str, Any], dict[str, Any], str]:
    protocol = json.loads((root / "protocol.json").read_text())
    protocol_hash = canonical_json_sha256(protocol)
    manifest_payload = json.loads((root / "hydra_task_manifest.json").read_text())
    if (
        canonical_json_sha256(manifest_payload)
        != protocol["hydra_selection"]["task_manifest_sha256"]
    ):
        raise AnalysisError("Hydra task manifest protocol mismatch")
    tasks = manifest_payload.get("tasks", [])
    if task_index < 0 or task_index >= len(tasks):
        raise AnalysisError(f"Hydra task index out of range: {task_index}")
    task = tasks[task_index]
    if task.get("task_index") != task_index:
        raise AnalysisError("Hydra task index identity mismatch")
    if task.get("task_sha256") != canonical_json_sha256(
        {key: value for key, value in task.items() if key != "task_sha256"}
    ):
        raise AnalysisError("Hydra task identity digest mismatch")
    return protocol, task, protocol_hash


def bind_live_recipes(task: dict[str, Any], sweep_root: Path) -> list[EffectiveRecipe]:
    recipes = discover_effective_recipes(str(task["family"]), sweep_root)
    by_signature = {recipe.signature: recipe for recipe in recipes}
    selected: list[EffectiveRecipe] = []
    for expected in task["recipes"]:
        recipe = by_signature.get(expected["effective_signature"])
        if recipe is None or recipe.canonical_name != expected["canonical_name"]:
            raise AnalysisError("Hydra worker recipe inventory mismatch")
        observed_paths = [str(path.resolve()) for path in recipe.config_paths]
        observed_hashes = [sha256_file(path) for path in recipe.config_paths]
        if list(recipe.aliases) != expected["aliases"]:
            raise AnalysisError("Hydra worker recipe alias drift")
        if observed_paths != expected["config_paths"]:
            raise AnalysisError("Hydra worker recipe config-path drift")
        if observed_hashes != expected["config_sha256"]:
            raise AnalysisError("Hydra worker recipe YAML-byte drift")
        selected.append(recipe)
    return selected


def run_task(root: Path, task_index: int) -> Path:
    protocol, task, protocol_hash = _load_bound_task(root, task_index)
    worker_path = Path(__file__).resolve()
    worker_hash = sha256_file(worker_path)
    expected_worker_hash = protocol["hydra_selection"]["worker_sha256"]
    if worker_hash != expected_worker_hash:
        raise AnalysisError("Hydra worker code hash mismatch")
    coordinator_path = worker_path.with_name("strict_split_fit.py")
    if sha256_file(coordinator_path) != protocol["runner_sha256"]:
        raise AnalysisError("Hydra coordinator code hash mismatch")

    result_path = root / task["result_relpath"]
    if result_path.exists():
        payload = json.loads(result_path.read_text())
        validate_worker_package(payload, protocol_hash, task, worker_hash)
        return result_path

    os.environ["JUMP_LITE_STRICT_GPU_INDEX"] = str(task["gpu_index"])
    selected = bind_live_recipes(task, Path(protocol["sweep_root"]))
    _gpu()
    annotations = build_annotation_table()
    split_by_id = {
        str(row["treatment_id"]): str(row["split"])
        for row in json.loads((root / "hydra_split.json").read_text())["rows"]
    }
    split_hash = canonical_json_sha256(
        sorted((str(key), str(value)) for key, value in split_by_id.items())
    )
    if split_hash != protocol["split_sha256"] or split_hash != task["split_sha256"]:
        raise AnalysisError("Hydra worker split identity mismatch")
    validation_ids = {
        key for key, value in split_by_id.items() if value == "validation"
    }

    family = str(task["family"])
    raw_identity = protocol["raw_input_schemas"][f"{family}/Raw"]
    if sha256_file(Path(raw_identity["path"])) != task["input_sha256"]:
        raise AnalysisError("Hydra worker Raw input hash mismatch")
    canonical_schema = protocol["canonical_raw_schemas"][family]
    frame, features, _ = load_raw_profile(
        family,
        "Raw",
        annotations,
        canonical_schema,
        task["source_schema_sha256"],
    )
    observed_fit_hash = exact_fit_ids_hash(frame, features, selected[0], split_by_id)
    if observed_fit_hash != task["fit_ids_sha256"]:
        raise AnalysisError("Hydra worker fit-ID hash mismatch")

    prefix_behavior = task["prefix_behavior_sha256"] or None
    prefix_frame = None
    prefix_state = None
    if prefix_behavior is not None:
        prefix_frame, prefix_state = fit_recipe_prefix(
            frame, features, selected[0], split_by_id, "Raw", 1
        )
    rows: list[dict[str, Any]] = []
    for recipe_index, recipe in enumerate(selected):
        started = time.monotonic()
        if prefix_behavior is None:
            transformed, state = fit_transform_recipe(
                frame, features, recipe, split_by_id, "Raw", 1
            )
        else:
            if prefix_frame is None or prefix_state is None:
                raise AnalysisError("Hydra prefix result is incomplete")
            transformed, state = complete_recipe_from_prefix(
                prefix_frame, prefix_state, recipe, split_by_id, 1
            )
        metrics, _, _ = score_partition(
            transformed, state.retained_features, validation_ids
        )
        if state.fit_ids_sha256 != task["fit_ids_sha256"]:
            raise AnalysisError("Hydra candidate fit-ID hash differs from manifest")
        rows.append(
            {
                "status": "ok",
                "family": family,
                "codec": "Raw",
                "config": recipe.canonical_name,
                "aliases": "|".join(recipe.aliases),
                "alias_count": len(recipe.aliases),
                "effective_signature": recipe.signature,
                "prefix_behavior_sha256": prefix_behavior or "",
                "prefix_signature": task["prefix_signature"],
                "prefix_cache_hit": prefix_behavior is not None and recipe_index > 0,
                "state_sha256": state.digest(),
                "input_sha256": task["input_sha256"],
                "source_schema_sha256": task["source_schema_sha256"],
                "canonical_schema_sha256": task["canonical_schema_sha256"],
                "split_sha256": state.split_sha256,
                "code_sha256": protocol["runner_sha256"],
                "worker_sha256": worker_hash,
                "task_sha256": task["task_sha256"],
                "fit_ids_sha256": state.fit_ids_sha256,
                "validation_pa_mean_nap": metrics["pa_mean_nap"],
                "validation_pc_mean_nap": metrics["pc_mean_nap"],
                "validation_wells": metrics["wells"],
                "validation_controls": metrics["controls"],
                "validation_treatments": metrics["treatments"],
                "elapsed_seconds": time.monotonic() - started,
                **state.fit_audit,
            }
        )
        del transformed, state
    release_prefix_cache()
    payload = {
        "protocol_hash": protocol_hash,
        "worker_sha256": worker_hash,
        "coordinator_sha256": protocol["runner_sha256"],
        "task": task,
        "task_sha256": task["task_sha256"],
        "rows": rows,
    }
    validate_worker_package(payload, protocol_hash, task, worker_hash)
    atomic_create_json(result_path, payload)
    return result_path


def validate_worker_package(
    payload: dict[str, Any],
    protocol_hash: str,
    task: dict[str, Any],
    worker_hash: str,
) -> None:
    if payload.get("protocol_hash") != protocol_hash:
        raise AnalysisError("stale Hydra worker package protocol")
    if payload.get("worker_sha256") != worker_hash:
        raise AnalysisError("stale Hydra worker package code")
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
            expected_value = task[key]
            if row.get(key) != expected_value:
                raise AnalysisError(f"Hydra worker package {key} mismatch")
        if row.get("worker_sha256") != worker_hash:
            raise AnalysisError("Hydra worker package row code mismatch")


@hydra.main(version_base=None, config_name="strict_selection_worker")
def main(cfg: HydraWorkerConfig) -> str:
    return str(run_task(Path(cfg.coordinator_root).resolve(), int(cfg.task_index)))


if __name__ == "__main__":
    main()
