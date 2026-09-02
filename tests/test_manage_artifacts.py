from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO / "scripts/manage_artifacts.py"
SPEC = importlib.util.spec_from_file_location("manage_artifacts", MODULE_PATH)
assert SPEC and SPEC.loader
manage = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(manage)
CHECK_SPEC = importlib.util.spec_from_file_location(
    "check_reproduce_inputs", REPO / "scripts/check_reproduce_inputs.py"
)
assert CHECK_SPEC and CHECK_SPEC.loader
check_inputs = importlib.util.module_from_spec(CHECK_SPEC)
CHECK_SPEC.loader.exec_module(check_inputs)


def test_registry_and_paper_producers_are_closed() -> None:
    registry = manage.load_registry()
    manage.validate_paper_producers(registry)
    bundles = manage.bundles_by_id(registry)
    assert "paper-post-sweep" in bundles
    assert "strict-heldout" in bundles
    assert "target-overlap" in bundles
    assert all(
        not bundle.get("regeneratable") or bundle["producer"] != "unmanaged"
        for bundle in bundles.values()
    )
    for bundle in bundles.values():
        if bundle["verify_mode"] != "producer":
            continue
        tracked = subprocess.check_output(
            ["git", "ls-files", "--", bundle["reference_root"]],
            cwd=REPO,
            text=True,
        ).splitlines()
        assert tracked, f"managed reference root is not tracked: {bundle['id']}"


def test_paper_lock_has_unique_complete_records() -> None:
    lock = json.loads((REPO / "paper_artifacts.lock.json").read_text())
    records = [*lock["figures"], *lock["tables"]]
    paths = [record["path"] for record in records]
    assert len(lock["figures"]) == 39
    assert len(lock["tables"]) == 3
    assert len(paths) == len(set(paths))
    for record in records:
        assert not Path(record["path"]).is_absolute()
        assert ".." not in Path(record["path"]).parts
        assert record["bytes"] > 0
        assert len(record["sha256"]) == 64


def test_destination_must_be_absent_and_inside_generated_root(tmp_path: Path) -> None:
    with pytest.raises(manage.ArtifactError, match="must remain under"):
        manage.check_destination(tmp_path, "strict-heldout")
    with pytest.raises(manage.ArtifactError, match="may not traverse"):
        manage.check_destination(
            Path("data/generated/artifacts/../escape"), "strict-heldout"
        )

    allowed = REPO / "data/generated/artifacts/test-destination-contract"
    try:
        destination = manage.check_destination(allowed, "strict-heldout")
        assert destination == allowed.resolve() / "strict-heldout"
        destination.mkdir()
        with pytest.raises(manage.ArtifactError, match="must not exist"):
            manage.check_destination(allowed, "strict-heldout")

        destination.rmdir()
        real_parent = allowed / "real"
        real_parent.mkdir(parents=True)
        linked_parent = allowed / "linked"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        with pytest.raises(manage.ArtifactError, match="symlinked output parent"):
            manage.check_destination(linked_parent, "strict-heldout")
    finally:
        shutil.rmtree(allowed, ignore_errors=True)


def test_render_and_compare_bundles_match_locked_outputs(tmp_path: Path) -> None:
    registry = manage.bundles_by_id(manage.load_registry())
    for bundle_id in ("strict-heldout", "target-overlap"):
        bundle = registry[bundle_id]
        output = tmp_path / bundle_id
        manage.run(manage.command_for(bundle, output, verify_only=False))
        manage.compare_expected(output, bundle)


def test_active_tex_graph_excludes_dormant_assets_and_closes_tables(
    tmp_path: Path,
) -> None:
    (tmp_path / "main/tables").mkdir(parents=True)
    (tmp_path / "main/figures").mkdir()
    (tmp_path / "main.tex").write_text("\\input{main/active}\n")
    (tmp_path / "main/active.tex").write_text(
        "\\includegraphics{main/figures/active.png}\n\\input{main/tables/generated}\n"
    )
    (tmp_path / "main/dormant.tex").write_text(
        "\\includegraphics{main/figures/dormant.png}\n"
    )
    (tmp_path / "main/tables/generated.tex").write_text(
        "\\begin{table}\n\\end{table}\n"
    )
    active = manage.active_tex_files(tmp_path)
    assert manage.active_graphics(tmp_path, active) == {"main/figures/active.png"}
    assert manage.active_generated_tables(tmp_path, active) == {
        "main/tables/generated.tex"
    }


def test_reproduce_preflight_rejects_60_file_motive_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    relative = Path("data/intermediate/motive_eval/large_strict")
    root = tmp_path / relative
    for index in range(60):
        path = root / f"config-{index}/metrics.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    monkeypatch.setattr(check_inputs, "EXPECTED_METRIC_COUNTS", {relative: 1_055})
    monkeypatch.setattr(check_inputs, "EXPECTED_FILES", {})
    monkeypatch.setattr(check_inputs, "REQUIRED_NONEMPTY", ())
    monkeypatch.setattr(check_inputs, "REQUIRED_GLOBS", {})
    errors = check_inputs.validate(tmp_path)
    assert errors == [f"checkpoint count mismatch for {relative}: 60 != 1055"]


def test_cli_list_and_verify_all_need_no_external_dataset() -> None:
    result = subprocess.run(
        [sys.executable, str(MODULE_PATH), "list"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "paper-post-sweep" in result.stdout
    assert "target-overlap" in result.stdout

    verified = subprocess.run(
        [sys.executable, str(MODULE_PATH), "verify", "all", "--skip-external"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert "artifact verification:" in verified.stdout
    assert (
        "SKIP compression-order-robustness: external checks disabled" in verified.stdout
    )

    listed = subprocess.run(
        ["just", "--list"], cwd=REPO, text=True, capture_output=True, check=True
    )
    assert "artifacts-verify" in listed.stdout
    assert "paper-artifacts-verify" in listed.stdout
