from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from rebuttal.postprocessing_validation.run_analysis import AnalysisError
from rebuttal.postprocessing_validation.strict_split_fit import (
    EffectiveRecipe,
    atomic_create_json,
    build_hydra_task_manifest,
    canonical_json_sha256,
    collect_hydra_selection,
    discover_effective_recipes,
    launch_hydra_selection,
    parse_args,
    validate_worker_package_payload,
)


def _raw_inputs(families: list[str]) -> dict[str, dict[str, object]]:
    return {
        f"{family}/Raw": {
            "sha256": f"input-{family}",
            "source_schema_sha256": f"source-{family}",
            "canonical_schema_sha256": f"canonical-{family}",
        }
        for family in families
    }


def _manifest() -> tuple[dict[str, object], dict[str, list[EffectiveRecipe]]]:
    families = [
        "cell_count",
        "cellprofiler",
        "dinov2",
        "dinov2_random",
        "morphem",
        "openphenom",
        "subcell",
    ]
    inventories = {family: discover_effective_recipes(family) for family in families}
    with (
        patch(
            "rebuttal.postprocessing_validation.strict_split_fit.load_raw_profile",
            return_value=(pl.DataFrame({"Metadata_id": ["x"]}), ["f0"], Path("raw")),
        ),
        patch(
            "rebuttal.postprocessing_validation.strict_split_fit.exact_fit_ids_hash",
            return_value="fit-ids",
        ),
    ):
        manifest = build_hydra_task_manifest(
            families=families,
            inventories=inventories,
            raw_inputs=_raw_inputs(families),
            canonical_raw_schemas={family: ["f0"] for family in families},
            annotations=pl.DataFrame(),
            split_by_id={"v": "validation", "t": "test"},
            gpu_indices=(0, 1, 2, 3),
            coordinator_sha256="coordinator",
            worker_sha256="worker",
            sweep_root=Path("/sweep"),
        )
    return manifest, inventories


def _package(
    task: dict[str, object], protocol_hash: str = "protocol"
) -> dict[str, object]:
    rows = []
    for recipe in task["recipes"]:
        rows.append(
            {
                "family": task["family"],
                "config": recipe["canonical_name"],
                "effective_signature": recipe["effective_signature"],
                "input_sha256": task["input_sha256"],
                "source_schema_sha256": task["source_schema_sha256"],
                "canonical_schema_sha256": task["canonical_schema_sha256"],
                "split_sha256": task["split_sha256"],
                "fit_ids_sha256": task["fit_ids_sha256"],
                "prefix_signature": task["prefix_signature"],
                "task_sha256": task["task_sha256"],
                "worker_sha256": "worker",
                "code_sha256": "coordinator",
            }
        )
    return {
        "protocol_hash": protocol_hash,
        "worker_sha256": "worker",
        "coordinator_sha256": "coordinator",
        "task": task,
        "task_sha256": task["task_sha256"],
        "rows": rows,
    }


class StrictHydraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest, cls.inventories = _manifest()

    def test_manifest_exactly_closes_inventory_and_prefix_groups(self) -> None:
        self.assertEqual(self.manifest["candidate_count"], 1160)
        self.assertEqual(self.manifest["task_count"], 38)
        candidate_keys = [
            (task["family"], recipe["effective_signature"])
            for task in self.manifest["tasks"]
            for recipe in task["recipes"]
        ]
        self.assertEqual(len(candidate_keys), len(set(candidate_keys)))
        self.assertEqual(
            [task["gpu_index"] for task in self.manifest["tasks"][:8]],
            [0, 1, 2, 3, 0, 1, 2, 3],
        )
        sizes: dict[str, list[int]] = {}
        for task in self.manifest["tasks"]:
            sizes.setdefault(task["family"], []).append(len(task["recipes"]))
        self.assertEqual(sizes["cell_count"], [1] * 5)
        self.assertEqual(sizes["cellprofiler"], [35] * 8)
        for family in ("dinov2", "dinov2_random", "morphem", "openphenom", "subcell"):
            self.assertEqual(sizes[family], [35] * 5)

    def test_manifest_is_deterministic_and_gpu_assignment_is_bound(self) -> None:
        again, _ = _manifest()
        self.assertEqual(self.manifest, again)
        changed = deepcopy(self.manifest)
        changed["tasks"][0]["gpu_index"] = 3
        self.assertNotEqual(
            canonical_json_sha256(self.manifest), canonical_json_sha256(changed)
        )

    def test_worker_package_identity_and_stale_rejection(self) -> None:
        task = self.manifest["tasks"][5]
        package = _package(task)
        rows = validate_worker_package_payload(
            package,
            protocol_hash="protocol",
            task=task,
            worker_sha256="worker",
            coordinator_sha256="coordinator",
        )
        self.assertEqual(len(rows), 35)
        for key in (
            "protocol_hash",
            "worker_sha256",
            "coordinator_sha256",
            "task_sha256",
        ):
            stale = deepcopy(package)
            stale[key] = "stale"
            with self.assertRaises(AnalysisError):
                validate_worker_package_payload(
                    stale,
                    protocol_hash="protocol",
                    task=task,
                    worker_sha256="worker",
                    coordinator_sha256="coordinator",
                )

    def test_atomic_worker_package_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.json"
            atomic_create_json(path, {"value": 1})
            with self.assertRaisesRegex(AnalysisError, "refusing to replace"):
                atomic_create_json(path, {"value": 2})
            self.assertEqual(json.loads(path.read_text()), {"value": 1})

    def test_collection_uses_canonical_order_and_one_checkpoint_writer(self) -> None:
        family = "cell_count"
        recipes = self.inventories[family][:2]
        tasks = []
        for index, recipe in enumerate(reversed(recipes)):
            task = next(
                deepcopy(item)
                for item in self.manifest["tasks"]
                if item["recipes"][0]["effective_signature"] == recipe.signature
            )
            task["task_index"] = index
            task["result_relpath"] = f"hydra_worker_results/task-{index:04d}.json"
            task["task_sha256"] = canonical_json_sha256(
                {key: value for key, value in task.items() if key != "task_sha256"}
            )
            tasks.append(task)
        manifest = {"tasks": tasks}
        protocol = {
            "runner_sha256": "coordinator",
            "hydra_selection": {"worker_sha256": "worker"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for task in tasks:
                atomic_create_json(root / task["result_relpath"], _package(task))
            with patch(
                "rebuttal.postprocessing_validation.strict_split_fit._checkpoint"
            ) as checkpoint:
                _, by_family = collect_hydra_selection(
                    output=root,
                    protocol_hash="protocol",
                    protocol=protocol,
                    manifest=manifest,
                    inventories={family: recipes},
                )
            self.assertEqual(checkpoint.call_count, 2)
            self.assertEqual(
                [row["effective_signature"] for row in by_family[family]],
                [recipe.signature for recipe in recipes],
            )

    def test_hydra_launcher_and_cli_guards(self) -> None:
        with patch("subprocess.run") as run:
            run.return_value.returncode = 0
            launch_hydra_selection(Path("/output"), 3, 4)
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertIn("hydra/launcher=joblib", command)
        self.assertIn("hydra.launcher.n_jobs=3", command)
        self.assertIn("task_index=0,1,2", command)
        args = parse_args(
            ["--output-dir", "/tmp/x", "--hydra-gpus", "3,1", "--hydra-jobs", "2"]
        )
        self.assertEqual(args.hydra_gpus, (3, 1))
        self.assertEqual(args.hydra_jobs, 2)
        with self.assertRaises(SystemExit):
            parse_args(["--output-dir", "/tmp/x", "--hydra-gpus", "1,1"])
        with self.assertRaises(SystemExit):
            parse_args(["--output-dir", "/tmp/x", "--hydra-jobs", "0"])


if __name__ == "__main__":
    unittest.main()
