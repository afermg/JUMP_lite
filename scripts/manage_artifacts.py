#!/usr/bin/env python3
"""List, regenerate, and verify JUMP-lite figure/data artifact bundles safely."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO / "artifacts.toml"
AUTHORED_PRODUCERS = {"authored-diagram", "frozen-example-images"}
MANAGED_PRODUCERS = {
    "target_overlap",
    "strict_heldout",
    "compression_order",
    "mq_effort",
    "mq_fixed_recipe",
    "mq_normalization",
    "mq_paired",
    "mq_plate",
    "mq_synthesis",
    "pretraining_legacy",
}


class ArtifactError(RuntimeError):
    """Fail-closed artifact-management error."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    with path.open("rb") as stream:
        data = tomllib.load(stream)
    if data.get("schema_version") != 1:
        raise ArtifactError("unsupported artifact registry schema")
    output_root = Path(data.get("default_output_root", ""))
    if not output_root.parts or output_root.is_absolute() or ".." in output_root.parts:
        raise ArtifactError(
            "default_output_root must be a safe repository-relative path"
        )
    bundles = data.get("bundle")
    if not isinstance(bundles, list) or not bundles:
        raise ArtifactError("artifact registry has no bundles")
    ids = [bundle.get("id") for bundle in bundles]
    if any(not isinstance(item, str) or not item for item in ids):
        raise ArtifactError("every bundle requires a non-empty string id")
    if len(ids) != len(set(ids)):
        raise ArtifactError("duplicate artifact bundle id")
    allowed = MANAGED_PRODUCERS | {"unmanaged"}
    unknown = sorted({bundle.get("producer") for bundle in bundles} - allowed)
    if unknown:
        raise ArtifactError(f"unknown producer allowlist key(s): {unknown}")
    for bundle in bundles:
        root = bundle.get("reference_root", "")
        if root and (Path(root).is_absolute() or ".." in Path(root).parts):
            raise ArtifactError(f"unsafe reference root for {bundle['id']}: {root}")
        if bundle.get("regeneratable") and bundle["producer"] == "unmanaged":
            raise ArtifactError(
                f"unmanaged bundle marked regeneratable: {bundle['id']}"
            )
    return data


def bundles_by_id(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {bundle["id"]: bundle for bundle in registry["bundle"]}


def expected_outputs(bundle: dict[str, Any]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in bundle.get("expected_outputs", []):
        if not isinstance(value, str) or ":" not in value:
            raise ArtifactError(
                f"invalid expected output for {bundle['id']}: {value!r}"
            )
        name, digest = value.rsplit(":", 1)
        path = Path(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ArtifactError(f"unsafe expected output for {bundle['id']}: {value!r}")
        if name in parsed:
            raise ArtifactError(f"duplicate expected output for {bundle['id']}: {name}")
        parsed[name] = digest
    return parsed


def command_for(
    bundle: dict[str, Any], output: Path, *, verify_only: bool
) -> list[str]:
    py = sys.executable
    producer = bundle["producer"]
    commands: dict[str, tuple[str, str]] = {
        "compression_order": (
            "rebuttal/compression_order_robustness/analyze.py",
            "--output-dir",
        ),
        "mq_effort": (
            "scripts/run_effort_sensitivity_isolated.py",
            "--output-dir",
        ),
        "mq_fixed_recipe": (
            "rebuttal/mq_d2e8_explanation/fixed_recipe_bootstrap/analyze.py",
            "--output-dir",
        ),
        "mq_normalization": (
            "rebuttal/mq_d2e8_explanation/normalization_interactions/analyze.py",
            "--output-dir",
        ),
        "mq_paired": (
            "rebuttal/mq_d2e8_explanation/paired_recipes/analyze.py",
            "--output",
        ),
        "mq_plate": (
            "rebuttal/mq_d2e8_explanation/plate_unit_influence/analyze.py",
            "--output",
        ),
        "mq_synthesis": (
            "rebuttal/mq_d2e8_explanation/synthesis/analyze.py",
            "--output",
        ),
        "pretraining_legacy": (
            "rebuttal/pretraining_overlap_figure/plot.py",
            "--output-dir",
        ),
    }
    if producer in commands:
        script, option = commands[producer]
        command = [py, str(REPO / script), option, str(output)]
        if verify_only:
            command.append("--verify-only")
        return command
    if producer == "target_overlap":
        if verify_only:
            raise ArtifactError("target-overlap uses render-and-compare verification")
        return [
            py,
            str(REPO / "paper_artifacts/target_overlap/render.py"),
            "--output-dir",
            str(output),
        ]
    if producer == "strict_heldout":
        if verify_only:
            raise ArtifactError("strict-heldout uses render-and-compare verification")
        return [
            py,
            str(REPO / "paper_artifacts/strict_heldout/render.py"),
            "--input-dir",
            str(REPO / "paper_artifacts/strict_heldout/inputs"),
            "--output-dir",
            str(output),
        ]
    raise ArtifactError(f"bundle has no managed command: {bundle['id']}")


def run(command: list[str]) -> None:
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(REPO / "src"),
        }
    )
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO, env=env, check=True)


def compare_expected(root: Path, bundle: dict[str, Any]) -> None:
    expected = expected_outputs(bundle)
    if not expected:
        raise ArtifactError(f"bundle has no locked expected outputs: {bundle['id']}")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != set(expected):
        raise ArtifactError(
            f"output inventory mismatch for {bundle['id']}: "
            f"missing={sorted(set(expected) - actual)}, extra={sorted(actual - set(expected))}"
        )
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ArtifactError(f"missing or symlinked output: {path}")
        observed = sha256(path)
        if observed != digest:
            raise ArtifactError(
                f"output hash mismatch for {path}: {observed} != {digest}"
            )


def verify_checksum_map(bundle: dict[str, Any]) -> None:
    manifest = REPO / bundle["checksum_manifest"]
    root = REPO / bundle["reference_root"]
    values = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or not values:
        raise ArtifactError(f"invalid checksum map: {manifest}")
    tracked = subprocess.check_output(
        ["git", "ls-files", "--", str(root.relative_to(REPO))],
        cwd=REPO,
        text=True,
    ).splitlines()
    checked = 0
    for repo_relative in tracked:
        path = REPO / repo_relative
        if path == manifest or not path.is_file():
            continue
        relative = str(path.relative_to(root))
        expected = values.get(relative)
        if expected is None:
            raise ArtifactError(f"tracked artifact omitted from checksum map: {path}")
        if path.is_symlink() or sha256(path) != expected:
            raise ArtifactError(f"tracked artifact checksum mismatch: {path}")
        checked += 1
    print(f"verified {checked} tracked artifacts from {manifest.relative_to(REPO)}")


def unavailable_paths(bundle: dict[str, Any]) -> list[Path]:
    missing: list[Path] = []
    for value in bundle.get("availability_paths", []):
        path = Path(value)
        path = path if path.is_absolute() else REPO / path
        if not path.exists():
            missing.append(path)
    return missing


def verify_bundle(
    bundle: dict[str, Any],
    root: Path | None = None,
    *,
    skip_external: bool = False,
) -> str:
    probes = bundle.get("availability_paths", [])
    missing = unavailable_paths(bundle)
    if probes and (skip_external or missing):
        reason = "external checks disabled" if skip_external else f"missing {missing}"
        print(f"SKIP {bundle['id']}: {reason}")
        return "skipped"
    mode = bundle["verify_mode"]
    if mode == "producer":
        selected_root = root or (REPO / bundle["reference_root"])
        run(command_for(bundle, selected_root, verify_only=True))
        return "verified"
    if mode == "render-and-compare":
        if root is not None:
            compare_expected(root, bundle)
        else:
            with tempfile.TemporaryDirectory(prefix=f"verify-{bundle['id']}-") as tmp:
                rendered = Path(tmp) / bundle["id"]
                run(command_for(bundle, rendered, verify_only=False))
                compare_expected(rendered, bundle)
        return "verified"
    if mode == "checksum-map":
        if root is not None:
            raise ArtifactError(
                "alternate roots are not supported for checksum-map bundles"
            )
        verify_checksum_map(bundle)
        return "verified"
    if mode in {"external-checkpoints", "external-snapshot"}:
        print(f"SKIP {bundle['id']}: {bundle.get('notes', mode)}")
        return "skipped"
    raise ArtifactError(f"unsupported verify mode for {bundle['id']}: {mode}")


def check_destination(
    output_root: Path, bundle_id: str, *, allowed_root: Path | None = None
) -> Path:
    allowed_root = (
        REPO / "data/generated/artifacts" if allowed_root is None else allowed_root
    ).resolve()
    if ".." in output_root.parts:
        raise ArtifactError(f"output root may not traverse parents: {output_root}")
    lexical_root = output_root if output_root.is_absolute() else REPO / output_root
    current = Path(lexical_root.anchor)
    for part in lexical_root.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ArtifactError(f"symlinked output parent is not allowed: {current}")
    requested_root = lexical_root.resolve()
    if not is_relative_to(requested_root, allowed_root):
        raise ArtifactError(
            f"output root must remain under {allowed_root}; got {requested_root}"
        )
    destination = requested_root / bundle_id
    if destination.exists() or destination.is_symlink():
        raise ArtifactError(f"destination must not exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def regenerate_bundle(
    bundle: dict[str, Any], output_root: Path, *, allowed_root: Path | None = None
) -> Path:
    if not bundle.get("regeneratable"):
        raise ArtifactError(f"bundle is not safely managed yet: {bundle['id']}")
    missing = unavailable_paths(bundle)
    if missing:
        raise ArtifactError(f"required external inputs are unavailable: {missing}")
    destination = check_destination(
        output_root, bundle["id"], allowed_root=allowed_root
    )
    try:
        run(command_for(bundle, destination, verify_only=False))
        verify_bundle(bundle, destination)
    except BaseException:
        if destination.exists():
            failed = destination.with_name(destination.name + ".FAILED")
            if failed.exists():
                shutil.rmtree(failed)
            destination.rename(failed)
            print(f"preserved failed output at {failed}", file=sys.stderr)
        raise
    print(f"generated and verified: {destination}")
    return destination


def strip_tex_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def active_tex_files(manuscript_root: Path) -> set[Path]:
    root = manuscript_root.resolve(strict=True)
    entrypoint = root / "main.tex"
    if not entrypoint.is_file():
        raise ArtifactError(f"missing manuscript entrypoint: {entrypoint}")
    include_pattern = re.compile(r"\\(?:input|include)\{([^}]+)\}")
    active: set[Path] = set()
    pending = [entrypoint]
    while pending:
        path = pending.pop()
        if path in active:
            continue
        if not path.is_file() or path.is_symlink():
            raise ArtifactError(f"missing or symlinked active TeX input: {path}")
        active.add(path)
        text = strip_tex_comments(path.read_text(encoding="utf-8"))
        for value in include_pattern.findall(text):
            relative = Path(value)
            if relative.is_absolute() or ".." in relative.parts:
                raise ArtifactError(f"unsafe TeX input from {path}: {value}")
            candidate = root / relative
            if candidate.suffix == "":
                candidate = candidate.with_suffix(".tex")
            candidate = candidate.resolve()
            if not is_relative_to(candidate, root):
                raise ArtifactError(f"TeX input escapes manuscript root: {value}")
            pending.append(candidate)
    return active


def active_graphics(
    manuscript_root: Path, tex_files: set[Path] | None = None
) -> set[str]:
    paths = active_tex_files(manuscript_root) if tex_files is None else tex_files
    found: set[str] = set()
    pattern = re.compile(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}")
    for path in paths:
        found.update(
            pattern.findall(strip_tex_comments(path.read_text(encoding="utf-8")))
        )
    return found


def active_generated_tables(manuscript_root: Path, tex_files: set[Path]) -> set[str]:
    root = manuscript_root.resolve(strict=True)
    found: set[str] = set()
    table_start = re.compile(r"^\s*\\begin\{table\*?\}")
    for path in tex_files:
        relative = path.relative_to(root)
        text = strip_tex_comments(path.read_text(encoding="utf-8"))
        if relative.parts[:2] == ("main", "tables") or table_start.match(text):
            found.add(str(relative))
    return found


def verify_paper(manuscript_root: Path, *, allow_other_commit: bool = False) -> None:
    registry = load_registry()
    lock_path = REPO / registry["paper_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    root = manuscript_root.resolve(strict=True)
    if not allow_other_commit:
        observed = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        if observed != lock["manuscript_commit"]:
            raise ArtifactError(
                f"manuscript commit mismatch: {observed} != {lock['manuscript_commit']}"
            )
    records = [*lock["figures"], *lock["tables"]]
    for record in records:
        path = root / record["path"]
        if not path.is_file() or path.is_symlink():
            raise ArtifactError(f"missing or symlinked paper artifact: {path}")
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ArtifactError(f"paper artifact drift: {path}")
    tex_files = active_tex_files(root)
    declared = {record["path"] for record in lock["figures"]}
    active = active_graphics(root, tex_files)
    if declared != active:
        raise ArtifactError(
            "paper graphic inventory drift: "
            f"undeclared={sorted(active - declared)}, inactive={sorted(declared - active)}"
        )
    declared_tables = {record["path"] for record in lock["tables"]}
    active_tables = active_generated_tables(root, tex_files)
    if declared_tables != active_tables:
        raise ArtifactError(
            "paper generated-table inventory drift: "
            f"undeclared={sorted(active_tables - declared_tables)}, "
            f"inactive={sorted(declared_tables - active_tables)}"
        )
    print(
        f"verified {len(lock['figures'])} figures and {len(lock['tables'])} tables "
        f"at manuscript commit {lock['manuscript_commit']}"
    )


def validate_paper_producers(registry: dict[str, Any]) -> None:
    lock = json.loads((REPO / registry["paper_lock"]).read_text(encoding="utf-8"))
    known = set(bundles_by_id(registry)) | AUTHORED_PRODUCERS
    unknown = sorted(
        {record["producer"] for record in [*lock["figures"], *lock["tables"]]} - known
    )
    if unknown:
        raise ArtifactError(f"paper lock references unknown producer(s): {unknown}")


def list_bundles(registry: dict[str, Any]) -> None:
    validate_paper_producers(registry)
    print("ID\tSCOPE\tREGENERATE\tRUNTIME\tDESCRIPTION")
    for bundle in registry["bundle"]:
        value = "yes" if bundle.get("regeneratable") else "no"
        print(
            f"{bundle['id']}\t{bundle['scope']}\t{value}\t"
            f"{bundle['runtime']}\t{bundle['description']}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list artifact bundles and reproduction status")
    verify_parser = sub.add_parser(
        "verify", help="verify one bundle or all local bundles"
    )
    verify_parser.add_argument("bundle", nargs="?", default="all")
    verify_parser.add_argument(
        "--skip-external",
        action="store_true",
        help="skip bundles whose verification requires canonical external inputs",
    )
    regenerate = sub.add_parser("regenerate", help="regenerate one bundle safely")
    regenerate.add_argument("bundle")
    regenerate.add_argument("--output-root", type=Path)
    paper = sub.add_parser("verify-paper", help="verify exact active manuscript assets")
    paper.add_argument("--manuscript-root", type=Path, required=True)
    paper.add_argument("--allow-other-commit", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = load_registry()
        validate_paper_producers(registry)
        by_id = bundles_by_id(registry)
        if args.command == "list":
            list_bundles(registry)
            return 0
        if args.command == "verify-paper":
            verify_paper(
                args.manuscript_root, allow_other_commit=args.allow_other_commit
            )
            return 0
        if args.bundle != "all" and args.bundle not in by_id:
            raise ArtifactError(f"unknown bundle: {args.bundle}")
        if args.command == "regenerate":
            if args.bundle == "all":
                raise ArtifactError("regenerate requires one explicit bundle id")
            configured_root = REPO / registry["default_output_root"]
            output_root = args.output_root or configured_root
            regenerate_bundle(
                by_id[args.bundle], output_root, allowed_root=configured_root
            )
            return 0
        selected = (
            list(by_id.values()) if args.bundle == "all" else [by_id[args.bundle]]
        )
        counts = {"verified": 0, "skipped": 0}
        for bundle in selected:
            result = verify_bundle(bundle, skip_external=args.skip_external)
            counts[result] += 1
        print(
            f"artifact verification: {counts['verified']} verified, {counts['skipped']} skipped"
        )
        return 0
    except (
        ArtifactError,
        OSError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
