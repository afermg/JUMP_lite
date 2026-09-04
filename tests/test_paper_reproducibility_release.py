from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from cpg_upload import build_paper_reproducibility_release as release


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_csv(path: Path, rows: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("name,value\n" + "".join(f"r{i},{i}\n" for i in range(rows)))


def compact_digest(root: Path, relatives: list[str]) -> tuple[int, int, str]:
    entries = [
        (relative, (root / relative).stat().st_size, digest(root / relative))
        for relative in relatives
    ]
    return (
        len(entries),
        sum(item[1] for item in entries),
        release._inventory_digest(entries),
    )


def make_sweep(
    root: Path, *, configs: int, missing: set[int] = frozenset()
) -> list[str]:
    names = []
    for index in range(configs):
        config = f"feature-{index}/config-{index}"
        names.append(config)
        base = root / config
        make_csv(base / "results/phenotypic_activity_per_compound.csv", 1)
        (base / "results/metrics.json").write_text('{"metric":1}\n')
        if index == 0:
            (base / "results/batch_effects.json").write_text("{}\n")
        (base / "pipeline_config.yaml").write_text(f"index: {index}\n")
        if index not in missing:
            (base / "output.parquet").write_bytes(b"PAR1fixture")
        (base / ".pipeline_cache/ignored.bin").parent.mkdir(parents=True)
        (base / ".pipeline_cache/ignored.bin").write_bytes(b"ignored")
    make_csv(root / "sweep_results.csv", configs)
    (root / "sweep_results_BAK.csv").write_text("ignored\n")
    (root / "plots_final").mkdir()
    (root / "plots_final/ignored.png").write_bytes(b"plot")
    return names


def sweep_values(root: Path, names: list[str]) -> dict[str, object]:
    included: list[tuple[str, int, None]] = []
    leaves: dict[str, int] = {}
    outputs = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        parts = PurePosixPath(relative).parts
        allowed = (
            relative == "sweep_results.csv"
            or (
                len(parts) == 3
                and parts[2] in {"output.parquet", "pipeline_config.yaml"}
            )
            or (len(parts) == 4 and parts[2] == "results")
        )
        if not allowed:
            continue
        included.append((relative, path.stat().st_size, None))
        if relative.endswith("/output.parquet"):
            outputs += 1
        if len(parts) == 4:
            leaves[parts[3]] = leaves.get(parts[3], 0) + 1
    return {
        "configs": len(names),
        "outputs": outputs,
        "config_digest": release._configuration_digest(names),
        "inventory_digest": release._inventory_digest(included),
        "file_count": len(included),
        "bytes": sum(item[1] for item in included),
        "leaves": dict(sorted(leaves.items())),
    }


def make_registry(
    tmp_path: Path, *, collision: bool = False, traversal: bool = False
) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    compact = tmp_path / "compact.csv"
    make_csv(compact, 2)
    compact_tree = tmp_path / "compact-tree"
    make_csv(compact_tree / "top.csv", 1)
    (compact_tree / "nested/data.json").parent.mkdir(parents=True)
    (compact_tree / "nested/data.json").write_text("{}\n")
    tree_count, tree_bytes, tree_digest = compact_digest(
        compact_tree, ["nested/data.json", "top.csv"]
    )

    sweep1 = tmp_path / "sweep1"
    names1 = make_sweep(sweep1, configs=2, missing={1})
    values1 = sweep_values(sweep1, names1)
    sweep2 = tmp_path / "sweep2"
    names2 = make_sweep(sweep2, configs=1)
    values2 = sweep_values(sweep2, names2)
    artifacts = [
        {
            "path": f"paper/generated-{index}.png",
            "mode": "generated",
            "producer": "fixture",
            "sha256": "f" * 64,
        }
        for index in range(28)
    ]
    (repo / "paper_artifacts.lock.json").write_text(
        json.dumps(
            {
                "manuscript_commit": "a" * 40,
                "tree": "b" * 40,
                "figures": artifacts,
                "tables": [],
            }
        )
    )
    compact_dest = "../escape.csv" if traversal else "compact/input.csv"
    duplicate = (
        f'''\n[[source]]
id = "duplicate"
role = "collision"
cohort = "fixture"
closure = "single-pinned"
root = "{compact}"
destination = "{compact_dest}"
expected_sha256 = "{digest(compact)}"
expected_bytes = {compact.stat().st_size}
'''
        if collision
        else ""
    )

    def sweep_block(
        identifier: str,
        root: Path,
        names: list[str],
        values: dict[str, object],
        missing: list[str],
    ) -> str:
        leaves = ", ".join(
            f'"{key}" = {value}' for key, value in values["leaves"].items()
        )
        return f'''[[sweep]]
id = "{identifier}"
role = "fixture sweep"
cohort = "fixture"
root = "{root}"
destination = "full_sweeps/{identifier}"
expected_configs = {values["configs"]}
expected_outputs = {values["outputs"]}
expected_summary_rows = {len(names)}
expected_summary_sha256 = "{digest(root / "sweep_results.csv")}"
expected_configuration_sha256 = "{values["config_digest"]}"
expected_inventory_sha256 = "{values["inventory_digest"]}"
expected_file_count = {values["file_count"]}
expected_bytes = {values["bytes"]}
allowed_results = {{ {leaves} }}
allowed_missing_output = {json.dumps(missing)}
'''

    registry = tmp_path / "registry.toml"
    registry.write_text(
        f'''schema_version = 2
package_version = "1.0"
analysis_code_release = "v1.1"
manuscript_commit = "{"a" * 40}"
manuscript_tree = "{"b" * 40}"
cohort_label = "fixture"
cohort_wells = 1
cohort_sites = 2
cohort_digest = "{"d" * 64}"

[[source]]
id = "compact"
role = "fixture compact data"
cohort = "fixture"
closure = "single-pinned"
root = "{compact}"
destination = "{compact_dest}"
expected_sha256 = "{digest(compact)}"
expected_bytes = {compact.stat().st_size}
expected_rows = 2
{duplicate}
[[source]]
id = "compact-tree"
role = "fixture closed compact tree"
cohort = "fixture"
closure = "curated-subset"
root = "{compact_tree}"
destination = "compact/tree"
include = ["*.csv", "nested/*.json"]
expected_file_count = {tree_count}
expected_bytes = {tree_bytes}
expected_inventory_sha256 = "{tree_digest}"

{sweep_block("sweep-one", sweep1, names1, values1, [names1[1]])}
{sweep_block("sweep-two", sweep2, names2, values2, [])}
[[figure_map]]
producer = "fixture"
artifacts = [{",".join(json.dumps(item["path"]) for item in artifacts)}]
source_ids = ["compact"]
'''
    )
    return registry, repo, compact


def test_plan_has_closed_sweep_and_compact_inventories(tmp_path: Path) -> None:
    registry, repo, _ = make_registry(tmp_path)
    result = release.plan(registry, repo)
    assert result["status"] == "ready"
    assert result["generated_artifact_mapping_count"] == 28
    assert result["compact"]["file_count"] == 3
    sweep = result["full_sweeps"]["sweeps"][0]
    assert sweep["configuration_count"] == 2
    assert sweep["output_parquet_count"] == 1
    assert sweep["missing_output_parquet"] == ["feature-1/config-1"]
    assert sweep["results_leaf_counts"] == {
        "batch_effects.json": 1,
        "metrics.json": 2,
        "phenotypic_activity_per_compound.csv": 2,
    }


@pytest.mark.parametrize(
    "mutation",
    ["renamed", "orphan", "missing-aux", "same-count", "result-log", "result-bak"],
)
def test_sweep_closure_mutations_fail(tmp_path: Path, mutation: str) -> None:
    registry, repo, _ = make_registry(tmp_path)
    root = tmp_path / "sweep1"
    if mutation == "renamed":
        (root / "feature-0").rename(root / "renamed-feature")
    elif mutation == "orphan":
        make_csv(root / "orphan/config/results/phenotypic_activity_per_compound.csv", 1)
    elif mutation == "missing-aux":
        (
            root / "feature-0/config-0/results/phenotypic_activity_per_compound.csv"
        ).unlink()
    elif mutation == "same-count":
        (
            root / "feature-1/config-1/results/phenotypic_activity_per_compound.csv"
        ).unlink()
        (root / "feature-1/config-1/results/batch_effects.json").write_text("{}\n")
    elif mutation == "result-log":
        (root / "feature-0/config-0/results/run.log").write_text("no\n")
    else:
        (root / "feature-0/config-0/results/metrics.json.bak").write_text("no\n")
    with pytest.raises(
        release.ReleaseError, match="metrics|results leaf|closure drift"
    ):
        release.plan(registry, repo)


def test_segment_aware_compact_patterns_and_inventory_drift(tmp_path: Path) -> None:
    assert release._matches(PurePosixPath("nested/a.json"), ["nested/*.json"])
    assert not release._matches(PurePosixPath("nested/deep/a.json"), ["nested/*.json"])
    assert not release._matches(PurePosixPath("nested/a.csv"), ["*.csv"])

    registry, repo, _ = make_registry(tmp_path / "content")
    path = tmp_path / "content/compact-tree/top.csv"
    path.write_text(path.read_text().replace("r0", "s0"))
    with pytest.raises(release.ReleaseError, match="compact inventory drift"):
        release.plan(registry, repo)

    registry, repo, _ = make_registry(tmp_path / "new")
    (tmp_path / "new/compact-tree/nested/new.json").write_text("{}\n")
    with pytest.raises(release.ReleaseError, match="compact inventory drift"):
        release.plan(registry, repo)


def test_compact_required_source_hash_traversal_collision_and_symlink_fail(
    tmp_path: Path,
) -> None:
    registry, repo, compact = make_registry(tmp_path / "missing")
    compact.unlink()
    with pytest.raises(release.ReleaseError, match="required path does not exist"):
        release.plan(registry, repo)

    registry, repo, _ = make_registry(tmp_path / "traversal", traversal=True)
    with pytest.raises(release.ReleaseError, match="unsafe destination"):
        release.plan(registry, repo)

    registry, repo, _ = make_registry(tmp_path / "collision", collision=True)
    with pytest.raises(release.ReleaseError, match="destination collision"):
        release.plan(registry, repo)

    registry, repo, compact = make_registry(tmp_path / "symlink")
    target = compact.with_suffix(".target")
    compact.rename(target)
    compact.symlink_to(target)
    with pytest.raises(release.ReleaseError, match="symlink"):
        release.plan(registry, repo)


def _build_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = "package"
) -> tuple[Path, Path, Path]:
    registry, repo, _ = make_registry(tmp_path / "inputs")
    monkeypatch.setattr(release, "_git_value", lambda repo_root, *args: "c" * 40)
    monkeypatch.setattr(
        release, "_packaging_code_commit", lambda repo_root, registry_path: "c" * 40
    )
    root = tmp_path / "release"
    root.mkdir()
    output = root / name
    release.build_release(
        output,
        registry_path=registry,
        repo_root=repo,
        allowed_root=root,
        copy_mode="copy",
        epoch=1234567890,
    )
    return output, registry, repo


def test_empty_csv_column_name_is_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, repo, compact = make_registry(tmp_path / "inputs")
    old_digest = digest(compact)
    compact.write_text(compact.read_text().replace("name,value\n", ",value1234\n"))
    registry.write_text(registry.read_text().replace(old_digest, digest(compact), 1))
    monkeypatch.setattr(
        release, "_packaging_code_commit", lambda repo_root, registry_path: "c" * 40
    )
    root = tmp_path / "release"
    root.mkdir()
    output = root / "package"

    release.build_release(
        output,
        registry_path=registry,
        repo_root=repo,
        allowed_root=root,
        copy_mode="copy",
        epoch=1234567890,
    )

    manifest = json.loads((output / "manifest.json").read_text())
    record = next(
        item for item in manifest["files"] if item["destination"] == "compact/input.csv"
    )
    assert record["columns"] == ["", "value1234"]
    assert release.verify_release(output)["status"] == "verified"


def test_table_record_validation_accepts_empty_names_and_rejects_malformed() -> None:
    record = {
        "bytes": 1,
        "cohort": "fixture",
        "destination": "compact/input.parquet",
        "role": "fixture",
        "sha256": "0" * 64,
        "source_id": "compact",
        "source_relative_path": "input.parquet",
        "tier": "compact",
        "format": "parquet",
        "row_count": 1,
        "columns": ["", "value"],
        "schema": {"": "Int64", "value": "String"},
    }
    release._validate_record(record)

    malformed = {**record, "columns": [1, "value"]}
    with pytest.raises(release.ReleaseError, match="invalid table columns"):
        release._validate_record(malformed)

    malformed = {**record, "schema": {"value": "String"}}
    with pytest.raises(release.ReleaseError, match="invalid Parquet schema"):
        release._validate_record(malformed)


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("non-string-column", "invalid table columns"),
        ("row-count", "source row count differs"),
        ("record-shape", "empty record role"),
    ],
)
def test_compact_record_preflight_fails_before_output_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    message: str,
) -> None:
    registry, repo, compact = make_registry(tmp_path / "inputs")
    root = tmp_path / "release"
    root.mkdir()
    monkeypatch.setattr(
        release, "_packaging_code_commit", lambda repo_root, registry_path: "c" * 40
    )

    if failure in {"non-string-column", "row-count"}:
        original_table_metadata = release._table_metadata

        def invalid_table_metadata(path: Path) -> dict[str, object]:
            if path == compact:
                if failure == "non-string-column":
                    return {"format": "csv", "row_count": 2, "columns": [1, "value"]}
                return {"format": "csv", "row_count": 3, "columns": ["name", "value"]}
            return original_table_metadata(path)

        monkeypatch.setattr(release, "_table_metadata", invalid_table_metadata)
    else:
        original_inventory = release.inventory

        def invalid_inventory(registry_path: Path, repo_root: Path) -> tuple:
            values = original_inventory(registry_path, repo_root)
            specs = [
                replace(spec, role="")
                if spec.tier == "compact" and spec.source_id == "compact"
                else spec
                for spec in values[1]
            ]
            return values[0], specs, *values[2:]

        monkeypatch.setattr(release, "inventory", invalid_inventory)

    def unexpected_materialization(*args: object, **kwargs: object) -> None:
        pytest.fail("payload materialization began before compact preflight completed")

    monkeypatch.setattr(release, "_copy_from_fd", unexpected_materialization)
    with pytest.raises(release.ReleaseError, match=message):
        release.build_release(
            root / "package",
            registry_path=registry,
            repo_root=repo,
            allowed_root=root,
            copy_mode="copy",
            epoch=1,
        )
    assert list(root.iterdir()) == []


def test_custom_basename_determinism_verify_attestation_and_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output1, registry, repo = _build_fixture(tmp_path, monkeypatch, "custom-name")
    output2 = tmp_path / "release/another-name"
    result = release.build_release(
        output2,
        registry_path=registry,
        repo_root=repo,
        allowed_root=tmp_path / "release",
        copy_mode="copy",
        epoch=1234567890,
    )
    manifest = json.loads((output2 / "manifest.json").read_text())
    payload_records = [item for item in manifest["files"] if item["tier"] != "metadata"]
    assert result["materialization"] == {
        "requested_mode": "copy",
        "file_counts": {"copy": len(payload_records)},
        "bytes": {"copy": sum(item["bytes"] for item in payload_records)},
    }
    assert (output1 / "manifest.json").read_bytes() == (
        output2 / "manifest.json"
    ).read_bytes()
    manifest_hash = digest(output1 / "manifest.json")
    assert (
        release.verify_release(output1, expected_manifest_sha256=manifest_hash)[
            "status"
        ]
        == "verified"
    )
    with pytest.raises(release.ReleaseError, match="external attestation"):
        release.verify_release(output1, expected_manifest_sha256="0" * 64)

    target = output1 / "compact/input.csv"
    target.chmod(0o644)
    target.write_text("tampered\n")
    with pytest.raises(release.ReleaseError, match="payload identity mismatch"):
        release.verify_release(output1)


def test_verifier_rejects_record_metadata_and_figure_map_rewrites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _, _ = _build_fixture(tmp_path / "metadata", monkeypatch)
    manifest_path = output / "manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    record = next(
        item for item in manifest["files"] if item["destination"] == "compact/input.csv"
    )
    record["row_count"] += 1
    manifest_path.write_bytes(release.canonical_json(manifest))
    with pytest.raises(release.ReleaseError, match="table metadata mismatch"):
        release.verify_release(output)

    output, _, _ = _build_fixture(tmp_path / "figure", monkeypatch)
    figure_path = output / release.FIGURE_MAP_NAME
    figure_path.chmod(0o644)
    figure = json.loads(figure_path.read_text())
    figure["mappings"][0]["source_ids"] = ["compact-tree"]
    figure_bytes = release.canonical_json(figure)
    figure_path.write_bytes(figure_bytes)
    manifest_path = output / "manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    record = next(
        item
        for item in manifest["files"]
        if item["destination"] == release.FIGURE_MAP_NAME
    )
    record["bytes"] = len(figure_bytes)
    record["sha256"] = hashlib.sha256(figure_bytes).hexdigest()
    manifest_path.write_bytes(release.canonical_json(manifest))
    with pytest.raises(release.ReleaseError, match="figure map differs"):
        release.verify_release(output)


def test_verifier_rejects_invalid_version_and_sweep_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _, _ = _build_fixture(tmp_path / "version", monkeypatch)
    manifest_path = output / "manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    manifest["package_version"] = "custom-name"
    manifest_path.write_bytes(release.canonical_json(manifest))
    with pytest.raises(release.ReleaseError, match="package_version"):
        release.verify_release(output)

    output, _, _ = _build_fixture(tmp_path / "sweep", monkeypatch)
    manifest_path = output / "manifest.json"
    manifest_path.chmod(0o644)
    manifest = json.loads(manifest_path.read_text())
    manifest["sweeps"][0]["missing_output_parquet"] = []
    manifest_path.write_bytes(release.canonical_json(manifest))
    with pytest.raises(release.ReleaseError, match="sweep summary mismatch"):
        release.verify_release(output)


def test_source_path_substitution_detected_after_descriptor_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, repo, compact = make_registry(tmp_path / "inputs")
    root = tmp_path / "release"
    root.mkdir()
    original = release._copy_from_fd
    substituted = False

    def substitute(source_fd: int, destination: Path, mode: str) -> None:
        nonlocal substituted
        if not substituted and os.path.samefile(f"/proc/self/fd/{source_fd}", compact):
            old = compact.with_suffix(".old")
            compact.rename(old)
            compact.write_bytes(old.read_bytes())
            substituted = True
        original(source_fd, destination, mode)

    monkeypatch.setattr(release, "_copy_from_fd", substitute)
    monkeypatch.setattr(
        release, "_packaging_code_commit", lambda repo_root, registry_path: "c" * 40
    )
    with pytest.raises(
        release.ReleaseError, match="opened source mutated|pathname identity changed"
    ):
        release.build_release(
            root / "package",
            registry_path=registry,
            repo_root=repo,
            allowed_root=root,
            copy_mode="copy",
            epoch=1,
        )


def test_cleanup_does_not_remove_replacement_temp_or_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, repo, _ = make_registry(tmp_path / "inputs")
    root = tmp_path / "release"
    root.mkdir()
    original = release._copy_from_fd
    replaced = False

    def replace_owned_paths(source_fd: int, destination: Path, mode: str) -> None:
        nonlocal replaced
        original(source_fd, destination, mode)
        if replaced:
            return
        temp = next(
            path for path in root.iterdir() if path.name.startswith(".package.tmp.")
        )
        stolen = root / ".stolen-temp"
        temp.rename(stolen)
        temp.mkdir()
        lock = root / ".package.build.lock"
        lock.unlink()
        lock.write_text("replacement\n")
        replaced = True

    monkeypatch.setattr(release, "_copy_from_fd", replace_owned_paths)
    monkeypatch.setattr(
        release, "_packaging_code_commit", lambda repo_root, registry_path: "c" * 40
    )
    with pytest.raises((release.ReleaseError, OSError)):
        release.build_release(
            root / "package",
            registry_path=registry,
            repo_root=repo,
            allowed_root=root,
            copy_mode="copy",
            epoch=1,
        )
    assert (root / ".package.build.lock").read_text() == "replacement\n"
    replacement_temp = next(
        path for path in root.iterdir() if path.name.startswith(".package.tmp.")
    )
    assert replacement_temp.is_dir()
    shutil.rmtree(root / ".stolen-temp")
    shutil.rmtree(replacement_temp)
    (root / ".package.build.lock").unlink()


def test_no_clobber_output_safety_and_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, repo, _ = make_registry(tmp_path / "inputs")
    monkeypatch.setattr(
        release, "_packaging_code_commit", lambda repo_root, registry_path: "c" * 40
    )
    root = tmp_path / "release"
    root.mkdir()
    existing = root / "existing"
    existing.mkdir()
    with pytest.raises(release.ReleaseError, match="already exists"):
        release.build_release(
            existing,
            registry_path=registry,
            repo_root=repo,
            allowed_root=root,
            copy_mode="copy",
        )
    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "link"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(release.ReleaseError, match="symlink|already exists"):
        release.build_release(
            link / "package",
            registry_path=registry,
            repo_root=repo,
            allowed_root=root,
            copy_mode="copy",
        )
    with pytest.raises(release.ReleaseError, match="traversal"):
        release.build_release(
            root / ".." / "outside-package",
            registry_path=registry,
            repo_root=repo,
            allowed_root=root,
            copy_mode="copy",
        )

    overlap_root = tmp_path / "inputs/sweep1/release-root"
    overlap_root.mkdir()
    with pytest.raises(release.ReleaseError, match="source/output overlap"):
        release.build_release(
            overlap_root / "package",
            registry_path=registry,
            repo_root=repo,
            allowed_root=overlap_root,
            copy_mode="copy",
        )


def test_auto_materialization_selects_reflink_and_cross_device_compact_copy() -> None:
    def spec(identifier: str, tier: str, device: int) -> release.FileSpec:
        return release.FileSpec(
            source_id=identifier,
            role="fixture",
            cohort="fixture",
            source=Path(f"/{identifier}"),
            source_relative=PurePosixPath(f"{identifier}.csv"),
            destination=PurePosixPath(f"{tier}/{identifier}.csv"),
            tier=tier,
            source_stat=(device, 1, 10, 1, 1),
            sweep_id=identifier if tier == "full_sweep" else None,
        )

    specs = [
        spec("compact-cross", "compact", 10),
        spec("compact-same", "compact", 20),
        spec("sweep-same", "full_sweep", 20),
    ]
    assert release._materialization_modes(specs, 20, "auto") == [
        "copy",
        "reflink",
        "reflink",
    ]
    assert release._materialization_modes(specs, 20, "copy") == ["copy"] * 3
    with pytest.raises(release.ReleaseError, match="strict reflink mode"):
        release._materialization_modes(specs, 20, "reflink")
    with pytest.raises(release.ReleaseError, match="cross-filesystem full sweep"):
        release._materialization_modes(
            [spec("sweep-cross", "full_sweep", 10)], 20, "auto"
        )


def test_materialization_preflights_each_selected_mode_and_device(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = []
    for index, content in enumerate((b"longer", b"x", b"copy-two")):
        source = tmp_path / f"source-{index}"
        source.write_bytes(content)
        sources.append(
            release.FileSpec(
                source_id=f"source-{index}",
                role="fixture",
                cohort="fixture",
                source=source,
                source_relative=PurePosixPath(source.name),
                destination=PurePosixPath(f"compact/{source.name}"),
                tier="compact",
                source_stat=release.stat_identity(source),
                planned_sha256=digest(source),
            )
        )
    seen: list[str] = []

    def descriptor_copy(source_fd: int, destination: Path, mode: str) -> None:
        seen.append(mode)
        destination_fd = os.open(
            destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444
        )
        try:
            os.lseek(source_fd, 0, os.SEEK_SET)
            while chunk := os.read(source_fd, 1024):
                os.write(destination_fd, chunk)
        finally:
            os.close(destination_fd)

    monkeypatch.setattr(release, "_copy_from_fd", descriptor_copy)
    release._preflight_materialization(sources, ["copy", "reflink", "copy"], tmp_path)
    assert seen == ["copy", "reflink"]
    assert not list(tmp_path.glob(".materialization-probe-*"))


def test_auto_cross_device_full_sweep_fails_before_any_build_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, repo, _ = make_registry(tmp_path / "inputs")
    root = tmp_path / "release"
    root.mkdir()
    output = root / "package"
    original_inventory = release.inventory

    def inventory_with_cross_device_sweeps(
        registry_path: Path, repo_root: Path
    ) -> tuple:
        values = original_inventory(registry_path, repo_root)
        specs = [
            replace(
                spec,
                source_stat=(
                    spec.source_stat[0] + 1,
                    *spec.source_stat[1:],
                ),
            )
            if spec.tier == "full_sweep"
            else spec
            for spec in values[1]
        ]
        return values[0], specs, *values[2:]

    monkeypatch.setattr(release, "inventory", inventory_with_cross_device_sweeps)
    monkeypatch.setattr(
        release, "_packaging_code_commit", lambda repo_root, registry_path: "c" * 40
    )
    before = list(root.iterdir())
    with pytest.raises(release.ReleaseError, match="cross-filesystem full sweep"):
        release.build_release(
            output,
            registry_path=registry,
            repo_root=repo,
            allowed_root=root,
            copy_mode="auto",
            epoch=1,
        )
    assert list(root.iterdir()) == before
    assert not output.exists()


def test_reflink_materialization_is_independent_when_supported(tmp_path: Path) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.write_bytes(b"independent")
    identity = release.stat_identity(source)
    fd = release._open_source(source, identity)
    try:
        try:
            release._copy_from_fd(fd, destination, "reflink")
        except release.ReleaseError as exc:
            if "Operation not supported" in str(exc) or "Invalid argument" in str(exc):
                pytest.skip(str(exc))
            raise
    finally:
        os.close(fd)
    assert source.stat().st_ino != destination.stat().st_ino
    destination.chmod(0o644)
    destination.write_bytes(b"changed")
    assert source.read_bytes() == b"independent"


def test_figure_map_requires_exact_generated_lock_coverage(tmp_path: Path) -> None:
    registry, repo, _ = make_registry(tmp_path)
    registry.write_text(
        registry.read_text().replace(
            '"paper/generated-27.png"', '"paper/not-in-lock.png"'
        )
    )
    with pytest.raises(release.ReleaseError, match="coverage mismatch"):
        release.plan(registry, repo)


def test_production_registry_cellpose_and_figure_dependencies() -> None:
    registry = release.load_registry()
    sources = {item["id"]: item for item in registry["source"]}
    assert all(not path.endswith(".tex") for path in sources["cellpose-count"]["files"])
    assert "cellpose-final-presentation" not in sources
    assert all(
        "fig5-s18-caret-marker" not in item.get("root", "") for item in sources.values()
    )
    assert sources["cellpose-final-figures"]["files"] == [
        "normalization_denominators.csv",
        "render_checksums.json",
        "s12_revised_full_data_points.csv",
        "s12_revised_rank_summary.csv",
        "s12_revised_task_scores.csv",
    ]
    assert sources["cellpose-final-table"]["files"] == [
        "detailed_scores.csv",
        "displayed_winners.csv",
        "grouped_scores.csv",
        "render_summary.json",
    ]
    assert all(
        not path.endswith(".tex")
        for source_id in ("cellpose-final-figures", "cellpose-final-table")
        for path in sources[source_id]["files"]
    )
    mappings = {
        artifact: set(group["source_ids"])
        for group in registry["figure_map"]
        for artifact in group["artifacts"]
    }
    for artifact in (
        "main/figures/main/fig5_heatmap/model_task_rank_best_avg_codec_mq.png",
        "main/figures/main/fig5_heatmap/model_task_rank_best_avg_codec_raw.png",
        "main/tables/combined_codec_delta_pct_table_summary.tex",
        "main/tables/performance_delta_table_combined_full.tex",
    ):
        assert "strict-motive-summary" in mappings[artifact]
    assert (
        "mq-heatmap-final-presentation"
        in mappings[
            "main/figures/main/fig5_heatmap/model_task_rank_best_avg_codec_mq.png"
        ]
    )
    assert (
        "mq-heatmap-final-presentation"
        not in mappings[
            "main/figures/main/fig5_heatmap/model_task_rank_best_avg_codec_raw.png"
        ]
    )
    for artifact in (
        "main/figures/rebuttal/model_task_rank_80pct_subsampling_mq.pdf",
        "main/figures/rebuttal/model_task_rank_80pct_subsampling_raw.pdf",
    ):
        assert "cellpose-final-figures" in mappings[artifact]
    assert (
        "cellpose-final-table"
        in mappings["main/sections/figures/figure5_s16_subsampling_scores.tex"]
    )
    assert mappings[
        "main/figures/rebuttal/rank_stability_correlation_scatter_all_configs.png"
    ] == {"jump-lite-sweep-summary"}
    assert (
        "target2-sweep-summary"
        not in mappings[
            "main/figures/main/fig3_segmentation_correlation/codec_balanced_absolute_boxen_y0.png"
        ]
    )


@pytest.mark.parametrize("epoch", [-1, release.MAX_SOURCE_DATE_EPOCH + 1])
def test_invalid_epoch_writes_nothing(tmp_path: Path, epoch: int) -> None:
    registry, repo, _ = make_registry(tmp_path / "inputs")
    allowed_root = tmp_path / "release"
    allowed_root.mkdir()
    output = allowed_root / "new-parent/package"
    before = list(allowed_root.iterdir())
    with pytest.raises(release.ReleaseError, match="source_date_epoch"):
        release.build_release(
            output,
            registry_path=registry,
            repo_root=repo,
            allowed_root=allowed_root,
            copy_mode="copy",
            epoch=epoch,
        )
    assert list(allowed_root.iterdir()) == before
    assert not output.parent.exists()


def test_default_epoch_comes_from_git_and_builder_has_no_network_upload_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    monkeypatch.setattr(release, "_git_value", lambda repo_root, *args: "123")
    assert release._source_date_epoch(None, Path(".")) == 123
    tree = ast.parse(Path(release.__file__).read_text())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (
            node.names
            if isinstance(node, ast.Import)
            else [ast.alias(name=node.module or "")]
        )
    }
    assert imported.isdisjoint({"boto3", "botocore", "requests", "socket", "urllib"})
    parser = release.make_parser()
    assert parser.parse_args(["build"]).copy_mode == "auto"
    assert "upload" not in {action.dest for action in parser._actions}
