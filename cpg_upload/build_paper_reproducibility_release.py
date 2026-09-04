#!/usr/bin/env python3
"""Plan, build, and verify the local JUMP-Lite paper-data release.

This module has no network or upload operation. Planning hashes the compact
inputs (about 813 MB) but validates full sweeps using path/size inventories only.
Building is create-only and publishes with Linux renameat2(RENAME_NOREPLACE).
"""

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import tomllib
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = Path(__file__).with_name("paper_reproducibility_sources.toml")
DEFAULT_OUTPUT = Path("/work/datasets/jump_lite/cpg_release/paper_artifacts/v1.0")
ALLOWED_OUTPUT_ROOT = Path("/work/datasets/jump_lite/cpg_release")
MANIFEST_NAME = "manifest.json"
FIGURE_MAP_NAME = "figure_to_inputs.json"
PACKAGED_LOCK = "metadata/paper_artifacts.lock.json"
PACKAGED_REGISTRY = "metadata/source_registry.json"
FICLONE = 0x40049409
CHUNK = 8 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+(?:\.[0-9]+)?$")
MAX_SOURCE_DATE_EPOCH = 253402300799


class ReleaseError(RuntimeError):
    """A fail-closed release validation error."""


Identity = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class FileSpec:
    source_id: str
    role: str
    cohort: str
    source: Path
    source_relative: PurePosixPath
    destination: PurePosixPath
    tier: str
    source_stat: Identity
    planned_sha256: str | None = None
    expected_rows: int | None = None
    sweep_id: str | None = None
    configuration: str | None = None


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _identity(info: os.stat_result) -> Identity:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def stat_identity(path: Path) -> Identity:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(info.st_mode):
        raise ReleaseError(f"source is not a regular file: {path}")
    return _identity(info)


def _directory_identity(path: Path) -> tuple[int, int]:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode):
        raise ReleaseError(f"expected directory: {path}")
    return info.st_dev, info.st_ino


def _fd_identity(fd: int) -> Identity:
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        raise ReleaseError("opened source descriptor is not a regular file")
    return _identity(info)


def _open_source(path: Path, expected: Identity) -> int:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as exc:
        raise ReleaseError(f"cannot safely open source {path}: {exc}") from exc
    if _fd_identity(fd) != expected:
        os.close(fd)
        raise ReleaseError(f"source descriptor identity changed: {path}")
    return fd


def _hash_fd(fd: int) -> str:
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while chunk := os.read(fd, CHUNK):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    identity = stat_identity(path)
    fd = _open_source(path, identity)
    try:
        digest = _hash_fd(fd)
        if _fd_identity(fd) != identity or stat_identity(path) != identity:
            raise ReleaseError(f"source mutated while hashing: {path}")
        return digest
    finally:
        os.close(fd)


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def validate_relative(value: str, *, label: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in ("", ".", "..") for part in path.parts)
        or "\\" in value
    ):
        raise ReleaseError(f"unsafe {label}: {value!r}")
    return path


def ensure_no_symlink_components(path: Path, *, require_exists: bool = True) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            if require_exists:
                raise ReleaseError(f"required path does not exist: {current}") from None
            return
        if stat.S_ISLNK(info.st_mode):
            raise ReleaseError(f"symlink path component refused: {current}")


def walk_regular_files(root: Path) -> list[tuple[PurePosixPath, Path]]:
    ensure_no_symlink_components(root)
    if not root.is_dir():
        raise ReleaseError(f"source tree is not a directory: {root}")
    found: list[tuple[PurePosixPath, Path]] = []
    stack: list[tuple[PurePosixPath, Path]] = [(PurePosixPath(), root)]
    while stack:
        relative_dir, directory = stack.pop()
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda value: value.name, reverse=True):
                relative = relative_dir / entry.name
                if entry.is_symlink():
                    raise ReleaseError(f"source symlink refused: {entry.path}")
                if entry.is_dir(follow_symlinks=False):
                    stack.append((relative, Path(entry.path)))
                elif entry.is_file(follow_symlinks=False):
                    found.append((relative, Path(entry.path)))
                else:
                    raise ReleaseError(
                        f"non-regular source entry refused: {entry.path}"
                    )
    return sorted(found, key=lambda item: item[0].as_posix())


def resolve_source(value: str, repo_root: Path) -> Path:
    if value == "repo://":
        return repo_root
    if value.startswith("repo://"):
        relative = validate_relative(
            value.removeprefix("repo://"), label="repository source"
        )
        return repo_root.joinpath(*relative.parts)
    path = Path(value)
    if not path.is_absolute():
        raise ReleaseError(f"source root must be absolute or repo://: {value}")
    return path


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    ensure_no_symlink_components(path)
    with path.open("rb") as handle:
        registry = tomllib.load(handle)
    if registry.get("schema_version") != 2:
        raise ReleaseError("registry schema_version must be 2")
    if not isinstance(registry.get("package_version"), str) or not VERSION_RE.fullmatch(
        registry["package_version"]
    ):
        raise ReleaseError("package_version must be a semantic numeric version")
    ids = [item["id"] for key in ("source", "sweep") for item in registry.get(key, [])]
    if len(ids) != len(set(ids)):
        raise ReleaseError("registry source IDs must be unique")
    return registry


def _matches(rel: PurePosixPath, patterns: Iterable[str]) -> bool:
    """Match path patterns segment by segment; '*' never crosses '/'."""
    for pattern in patterns:
        pattern_path = validate_relative(pattern, label="include pattern")
        if len(rel.parts) == len(pattern_path.parts) and all(
            fnmatchcase(part, wanted)
            for part, wanted in zip(rel.parts, pattern_path.parts, strict=True)
        ):
            return True
    return False


def _inventory_digest(entries: Iterable[tuple[str, int, str | None]]) -> str:
    digest = hashlib.sha256()
    for path, size, content_hash in sorted(entries):
        line = f"{path}\0{size}"
        if content_hash is not None:
            line += f"\0{content_hash}"
        digest.update((line + "\n").encode())
    return digest.hexdigest()


def _configuration_digest(configurations: Iterable[str]) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in sorted(configurations)).encode()
    ).hexdigest()


def _make_spec(
    item: dict[str, Any],
    source: Path,
    source_relative: PurePosixPath,
    destination: PurePosixPath,
    tier: str,
    *,
    planned_sha256: str | None = None,
    sweep_id: str | None = None,
    configuration: str | None = None,
) -> FileSpec:
    ensure_no_symlink_components(source)
    return FileSpec(
        source_id=item["id"],
        role=item["role"],
        cohort=item["cohort"],
        source=source,
        source_relative=source_relative,
        destination=destination,
        tier=tier,
        source_stat=stat_identity(source),
        planned_sha256=planned_sha256,
        expected_rows=item.get("expected_rows"),
        sweep_id=sweep_id,
        configuration=configuration,
    )


def _validate_checksum_closure(
    item: dict[str, Any], root: Path, specs: list[FileSpec]
) -> None:
    selected = {spec.source_relative.as_posix(): spec for spec in specs}
    if manifest_name := item.get("checksum_manifest"):
        manifest = root / manifest_name
        entries: dict[str, str] = {}
        for line in manifest.read_text().splitlines():
            digest, separator, name = line.partition("  ")
            if not separator or not SHA256_RE.fullmatch(digest):
                raise ReleaseError(f"invalid checksum line in {manifest}: {line!r}")
            entries[name] = digest
        unlisted = set(item.get("checksum_unlisted", []))
        expected = set(selected) - {manifest_name} - unlisted
        if not unlisted <= set(selected) or set(entries) != expected:
            raise ReleaseError(f"checksum closure mismatch for {item['id']}")
        for name, digest in entries.items():
            if selected[name].planned_sha256 != digest:
                raise ReleaseError(f"checksum mismatch for {item['id']}/{name}")
        if "completion.json" in unlisted:
            completion = json.loads(selected["completion.json"].source.read_text())
            if (
                completion.get("sha256s_sha256")
                != selected[manifest_name].planned_sha256
            ):
                raise ReleaseError(
                    f"completion receipt checksum mismatch for {item['id']}"
                )
    if manifest_name := item.get("json_checksum_manifest"):
        data = json.loads((root / manifest_name).read_text())
        entries = {
            entry["path"]: entry["sha256"] for entry in data.get("artifacts", [])
        }
        if set(entries) != set(selected) - {manifest_name}:
            raise ReleaseError(f"JSON checksum closure mismatch for {item['id']}")
        for name, digest in entries.items():
            if selected[name].planned_sha256 != digest:
                raise ReleaseError(f"checksum mismatch for {item['id']}/{name}")
    if manifest_name := item.get("subset_checksum_manifest"):
        if manifest_name not in selected:
            raise ReleaseError(
                f"subset checksum manifest is not selected: {item['id']}"
            )
        entries: dict[str, str] = {}
        for line in (root / manifest_name).read_text().splitlines():
            digest, separator, name = line.partition("  ")
            if not separator or not SHA256_RE.fullmatch(digest):
                raise ReleaseError(
                    f"invalid checksum line in {manifest_name}: {line!r}"
                )
            entries[name] = digest
        for name, spec in selected.items():
            if name != manifest_name and entries.get(name) != spec.planned_sha256:
                raise ReleaseError(f"subset checksum mismatch for {item['id']}/{name}")
    if receipt_name := item.get("receipt_file"):
        if receipt_name not in selected:
            raise ReleaseError(f"receipt is not selected: {item['id']}")
        receipt = json.loads((root / receipt_name).read_text())
        for path, expected in item.get("expected_json_values", {}).items():
            if _json_value(receipt, path) != expected:
                raise ReleaseError(f"receipt value mismatch for {item['id']}/{path}")
        artifacts = receipt.get("artifacts", {})
        for name, expected in item.get("expected_artifact_checksums", {}).items():
            if (
                artifacts.get(name) != expected
                or selected[name].planned_sha256 != expected
            ):
                raise ReleaseError(f"receipt artifact mismatch for {item['id']}/{name}")
        for path, expected in item.get("expected_json_lists", {}).items():
            if _json_value(receipt, path) != expected:
                raise ReleaseError(f"receipt list mismatch for {item['id']}/{path}")


def _json_value(data: dict[str, Any], path: str) -> Any:
    if path in data:
        return data[path]
    value: Any = data
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ReleaseError(f"receipt JSON path is absent: {path}")
        value = value[part]
    return value


def _validate_strict_motive_path(item: dict[str, Any], path: Path) -> None:
    if "expected_distinct_keys" not in item:
        return
    key_columns = item["key_columns"]
    task_column = item["task_column"]
    counts: Counter[tuple[str, ...]] = Counter()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = tuple(row[column] for column in key_columns)
            counts[key] += 1
            if not row[task_column]:
                raise ReleaseError("strict-MOTIVE has an empty task value")
    if len(counts) != item["expected_distinct_keys"]:
        raise ReleaseError(f"strict-MOTIVE distinct-key drift: {len(counts)}")
    expected_tasks = item["expected_tasks_per_key"]
    if any(count != expected_tasks for count in counts.values()):
        raise ReleaseError(
            "strict-MOTIVE does not have exactly eight task rows per key"
        )


def scan_compact(
    registry: dict[str, Any], repo_root: Path
) -> tuple[list[FileSpec], list[dict[str, Any]], float]:
    started = time.monotonic()
    all_specs: list[FileSpec] = []
    summaries: list[dict[str, Any]] = []
    for item in registry.get("source", []):
        root = resolve_source(item["root"], repo_root)
        destination = validate_relative(item["destination"], label="destination")
        includes = item.get("include")
        files = item.get("files")
        selected: list[tuple[PurePosixPath, Path]] = []
        if includes is not None and files is not None:
            raise ReleaseError(
                f"compact source {item['id']} cannot define both include and files"
            )
        if files is not None:
            if not files:
                raise ReleaseError(
                    f"compact source {item['id']} has an empty files list"
                )
            selected = [
                (
                    validate_relative(value, label=f"{item['id']} file"),
                    root.joinpath(
                        *validate_relative(value, label=f"{item['id']} file").parts
                    ),
                )
                for value in files
            ]
        elif includes is not None:
            selected = [
                entry
                for entry in walk_regular_files(root)
                if _matches(entry[0], includes)
            ]
            if not selected:
                raise ReleaseError(f"compact source {item['id']} matched no files")
        else:
            selected = [(PurePosixPath(root.name), root)]
            root = root.parent

        specs: list[FileSpec] = []
        for relative, source in selected:
            initial = _make_spec(
                item,
                source,
                relative,
                destination
                if len(selected) == 1 and "include" not in item and "files" not in item
                else destination / relative,
                "compact",
            )
            digest = sha256_file(source)
            specs.append(FileSpec(**{**initial.__dict__, "planned_sha256": digest}))
        count = len(specs)
        size = sum(spec.source_stat[2] for spec in specs)
        inventory_sha256 = _inventory_digest(
            (spec.source_relative.as_posix(), spec.source_stat[2], spec.planned_sha256)
            for spec in specs
        )
        if item["closure"] == "single-pinned":
            if count != 1 or specs[0].planned_sha256 != item["expected_sha256"]:
                raise ReleaseError(f"SHA-256 drift for {item['id']}")
            if specs[0].source_stat[2] != item["expected_bytes"]:
                raise ReleaseError(f"byte-size drift for {item['id']}")
        else:
            expected = (
                item["expected_file_count"],
                item["expected_bytes"],
                item["expected_inventory_sha256"],
            )
            observed = (count, size, inventory_sha256)
            if observed != expected:
                raise ReleaseError(
                    f"compact inventory drift for {item['id']}: {observed} != {expected}"
                )
        if (
            item.get("expected_rows") is not None
            and csv_rows(specs[0].source) != item["expected_rows"]
        ):
            raise ReleaseError(f"row-count drift for {item['id']}")
        _validate_strict_motive_path(item, specs[0].source)
        _validate_checksum_closure(item, root, specs)
        all_specs.extend(specs)
        summaries.append(
            {
                "id": item["id"],
                "role": item["role"],
                "cohort": item["cohort"],
                "closure": item["closure"],
                "file_count": count,
                "bytes": size,
                "inventory_sha256": inventory_sha256,
                "destinations": sorted(spec.destination.as_posix() for spec in specs),
            }
        )
    return all_specs, summaries, time.monotonic() - started


def _sweep_excluded(relative: PurePosixPath) -> bool:
    if relative == PurePosixPath("sweep_results_BAK.csv"):
        return True
    lowered = [part.lower() for part in relative.parts]
    return any(
        part == ".pipeline_cache"
        or part == "__pycache__"
        or part.startswith("plots")
        or part in {"cache", "tmp", "temp", "logs"}
        or part.endswith("_bak")
        or part.endswith(".bak")
        or part.endswith("~")
        for part in lowered
    )


def scan_sweep(
    item: dict[str, Any], repo_root: Path = REPO_ROOT
) -> tuple[list[FileSpec], dict[str, Any]]:
    root = resolve_source(item["root"], repo_root)
    destination = validate_relative(item["destination"], label="sweep destination")
    allowed_results = item["allowed_results"]
    included: list[tuple[PurePosixPath, Path]] = []
    excluded: list[str] = []
    for relative, source in walk_regular_files(root):
        parts = relative.parts
        if len(parts) == 4 and parts[2] == "results":
            if parts[3] not in allowed_results:
                raise ReleaseError(
                    f"unrecognized results leaf in {item['id']}: {relative}"
                )
            included.append((relative, source))
        elif relative == PurePosixPath("sweep_results.csv") or (
            len(parts) == 3 and parts[2] in {"output.parquet", "pipeline_config.yaml"}
        ):
            included.append((relative, source))
        elif _sweep_excluded(relative):
            excluded.append(relative.as_posix())
        else:
            raise ReleaseError(f"unrecognized file in sweep {item['id']}: {relative}")

    by_config: dict[str, set[str]] = defaultdict(set)
    result_counts: Counter[str] = Counter()
    for relative, _ in included:
        if len(relative.parts) < 3:
            continue
        config = "/".join(relative.parts[:2])
        leaf = "/".join(relative.parts[2:])
        by_config[config].add(leaf)
        if len(relative.parts) == 4:
            result_counts[relative.parts[3]] += 1
    all_configs = set(by_config)
    metric_configs = {
        key for key, values in by_config.items() if "results/metrics.json" in values
    }
    pipeline_configs = {
        key for key, values in by_config.items() if "pipeline_config.yaml" in values
    }
    output_configs = {
        key for key, values in by_config.items() if "output.parquet" in values
    }
    if all_configs != metric_configs:
        raise ReleaseError(
            f"{item['id']} has configuration files/results without metrics: {sorted(all_configs - metric_configs)[:5]}"
        )
    if pipeline_configs != metric_configs:
        raise ReleaseError(f"{item['id']} pipeline_config set differs from metrics set")
    missing = sorted(metric_configs - output_configs)
    observed_inventory = _inventory_digest(
        (relative.as_posix(), stat_identity(source)[2], None)
        for relative, source in included
    )
    observed = {
        "configuration_count": len(metric_configs),
        "output_parquet_count": len(output_configs),
        "missing_output_parquet": missing,
        "configuration_sha256": _configuration_digest(metric_configs),
        "inventory_sha256": observed_inventory,
        "included_file_count": len(included),
        "included_bytes": sum(stat_identity(source)[2] for _, source in included),
        "results_leaf_counts": dict(sorted(result_counts.items())),
    }
    expected = {
        "configuration_count": item["expected_configs"],
        "output_parquet_count": item["expected_outputs"],
        "missing_output_parquet": sorted(item.get("allowed_missing_output", [])),
        "configuration_sha256": item["expected_configuration_sha256"],
        "inventory_sha256": item["expected_inventory_sha256"],
        "included_file_count": item["expected_file_count"],
        "included_bytes": item["expected_bytes"],
        "results_leaf_counts": dict(sorted(item["allowed_results"].items())),
    }
    if observed != expected:
        raise ReleaseError(
            f"sweep closure drift for {item['id']}: {observed} != {expected}"
        )
    summary = root / "sweep_results.csv"
    if (
        sha256_file(summary) != item["expected_summary_sha256"]
        or csv_rows(summary) != item["expected_summary_rows"]
    ):
        raise ReleaseError(f"summary identity drift for {item['id']}")

    specs = [
        _make_spec(
            item,
            source,
            relative,
            destination / relative,
            "full_sweep",
            sweep_id=item["id"],
            configuration="/".join(relative.parts[:2])
            if len(relative.parts) >= 3
            else None,
        )
        for relative, source in included
    ]
    return specs, {
        "id": item["id"],
        "role": item["role"],
        "cohort": item["cohort"],
        **observed,
        "excluded_file_count": len(excluded),
        "excluded_policy": [
            ".pipeline_cache",
            "plots*",
            "sweep_results_BAK.csv",
            "cache/temp/log/backup material",
        ],
    }


def validate_collisions(specs: list[FileSpec]) -> None:
    destinations: dict[PurePosixPath, FileSpec] = {}
    for spec in specs:
        if spec.destination in destinations:
            raise ReleaseError(f"destination collision: {spec.destination}")
        destinations[spec.destination] = spec
    for destination in destinations:
        if any(parent in destinations for parent in destination.parents):
            raise ReleaseError(f"destination file/directory collision: {destination}")


def generated_artifacts(lock: dict[str, Any]) -> set[str]:
    return {
        item["path"]
        for section in ("figures", "tables")
        for item in lock.get(section, [])
        if item.get("mode") == "generated"
    }


def _figure_map_from_destinations(
    registry: dict[str, Any], destinations: dict[str, list[str]], lock: dict[str, Any]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in registry.get("figure_map", []):
        source_ids = sorted(group["source_ids"])
        unknown = set(source_ids) - set(destinations)
        if unknown:
            raise ReleaseError(
                f"figure map names unknown source IDs: {sorted(unknown)}"
            )
        for artifact in group["artifacts"]:
            if artifact in seen:
                raise ReleaseError(f"duplicate figure-map artifact: {artifact}")
            seen.add(artifact)
            records.append(
                {
                    "artifact": artifact,
                    "producer": group["producer"],
                    "source_ids": source_ids,
                    "package_inputs": sorted(
                        path
                        for source_id in source_ids
                        for path in destinations[source_id]
                    ),
                }
            )
    expected = generated_artifacts(lock)
    if seen != expected:
        raise ReleaseError(
            f"figure-map coverage mismatch; missing={sorted(expected - seen)}, extra={sorted(seen - expected)}"
        )
    return sorted(records, key=lambda item: item["artifact"])


def _validate_lock_hash_assertions(
    registry: dict[str, Any], lock: dict[str, Any]
) -> None:
    locked = {
        item["path"]: item["sha256"]
        for section in ("figures", "tables")
        for item in lock.get(section, [])
    }
    for source in registry.get("source", []):
        for path, expected in source.get("lock_artifact_hashes", {}).items():
            if locked.get(path) != expected:
                raise ReleaseError(
                    f"source receipt hash does not match manuscript lock: {source['id']}/{path}"
                )


def build_figure_map(
    registry: dict[str, Any], specs: list[FileSpec], lock: dict[str, Any]
) -> list[dict[str, Any]]:
    destinations: dict[str, list[str]] = defaultdict(list)
    for spec in specs:
        if spec.tier == "compact":
            destinations[spec.source_id].append(spec.destination.as_posix())
    return _figure_map_from_destinations(registry, destinations, lock)


def inventory(
    registry_path: Path = REGISTRY_PATH, repo_root: Path = REPO_ROOT
) -> tuple[
    dict[str, Any],
    list[FileSpec],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    float,
]:
    registry = load_registry(registry_path)
    compact, compact_summaries, compact_seconds = scan_compact(registry, repo_root)
    sweeps: list[FileSpec] = []
    sweep_summaries: list[dict[str, Any]] = []
    for item in registry.get("sweep", []):
        specs, summary = scan_sweep(item, repo_root)
        sweeps.extend(specs)
        sweep_summaries.append(summary)
    specs = compact + sweeps
    validate_collisions(specs)
    lock_path = repo_root / "paper_artifacts.lock.json"
    lock = json.loads(lock_path.read_text())
    if (
        lock.get("manuscript_commit") != registry["manuscript_commit"]
        or lock.get("tree") != registry["manuscript_tree"]
    ):
        raise ReleaseError(
            "registry manuscript identity differs from paper_artifacts.lock.json"
        )
    _validate_lock_hash_assertions(registry, lock)
    figure_map = build_figure_map(registry, specs, lock)
    return (
        registry,
        specs,
        sweep_summaries,
        figure_map,
        compact_summaries,
        compact_seconds,
    )


def plan(
    registry_path: Path = REGISTRY_PATH, repo_root: Path = REPO_ROOT
) -> dict[str, Any]:
    registry, specs, sweeps, figure_map, compact_summaries, compact_seconds = inventory(
        registry_path, repo_root
    )
    _preflight_compact_records(specs)
    compact = [item for item in specs if item.tier == "compact"]
    full = [item for item in specs if item.tier == "full_sweep"]
    return {
        "status": "ready",
        "package_version": registry["package_version"],
        "compact": {
            "file_count": len(compact),
            "bytes": sum(item.source_stat[2] for item in compact),
            "source_count": len(compact_summaries),
            "inventory_sha256": _inventory_digest(
                (
                    f"{item.source_id}/{item.source_relative.as_posix()}",
                    item.source_stat[2],
                    item.planned_sha256,
                )
                for item in compact
            ),
            "inventory_hash_seconds": round(compact_seconds, 6),
            "sources": compact_summaries,
        },
        "full_sweeps": {
            "file_count": len(full),
            "bytes": sum(item.source_stat[2] for item in full),
            "sweeps": sweeps,
        },
        "generated_artifact_mapping_count": len(figure_map),
    }


def _validate_output(
    output: Path, allowed_root: Path, *, create_direct_parent: bool = False
) -> tuple[Path, Path]:
    if not output.is_absolute() or not allowed_root.is_absolute():
        raise ReleaseError("output and allowed output root must be absolute")
    if ".." in output.parts or ".." in allowed_root.parts:
        raise ReleaseError("output traversal is not permitted")
    ensure_no_symlink_components(allowed_root)
    root = allowed_root.resolve(strict=True)
    candidate = Path(os.path.abspath(output))
    if candidate == root or root not in candidate.parents:
        raise ReleaseError(f"output must be below {root}: {candidate}")
    if candidate.exists() or candidate.is_symlink():
        raise ReleaseError(f"output already exists: {candidate}")
    if not candidate.parent.exists() and create_direct_parent:
        if candidate.parent.parent != root:
            raise ReleaseError(
                f"only a direct child package parent may be created below {root}"
            )
        ensure_no_symlink_components(candidate.parent.parent)
        try:
            candidate.parent.mkdir(mode=0o755)
        except FileExistsError:
            pass
    ensure_no_symlink_components(candidate.parent)
    if not candidate.parent.is_dir():
        raise ReleaseError(f"output parent must already exist: {candidate.parent}")
    return candidate, root


def _ensure_no_overlap(
    specs: list[FileSpec], output: Path, registry: dict[str, Any], repo_root: Path
) -> None:
    output_abs = output.resolve(strict=False)
    sources = {spec.source.resolve(strict=True) for spec in specs}
    sources.update(
        resolve_source(item["root"], repo_root).resolve(strict=True)
        for key in ("source", "sweep")
        for item in registry.get(key, [])
    )
    if any(
        source == output_abs
        or source in output_abs.parents
        or output_abs in source.parents
        for source in sources
    ):
        raise ReleaseError("source/output overlap refused")


def _atomic_publish_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise ReleaseError("Linux renameat2(RENAME_NOREPLACE) is required")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise ReleaseError(f"output appeared during build: {destination}")
        raise ReleaseError(f"atomic no-replace rename failed: {os.strerror(error)}")


def _copy_from_fd(source_fd: int, destination: Path, mode: str) -> None:
    destination_fd = os.open(
        destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444
    )
    try:
        if mode == "reflink":
            try:
                fcntl.ioctl(destination_fd, FICLONE, source_fd)
            except OSError as exc:
                raise ReleaseError(f"reflink failed for {destination}: {exc}") from exc
        elif mode == "copy":
            os.lseek(source_fd, 0, os.SEEK_SET)
            while chunk := os.read(source_fd, CHUNK):
                view = memoryview(chunk)
                while view:
                    view = view[os.write(destination_fd, view) :]
        else:
            raise ReleaseError(f"unsupported copy mode: {mode}")
        os.fsync(destination_fd)
    finally:
        os.close(destination_fd)


def _materialize(spec: FileSpec, destination: Path, mode: str) -> None:
    source_fd = _open_source(spec.source, spec.source_stat)
    try:
        _copy_from_fd(source_fd, destination, mode)
        if _fd_identity(source_fd) != spec.source_stat:
            raise ReleaseError(f"opened source mutated during copy: {spec.source}")
    finally:
        os.close(source_fd)
    if stat_identity(spec.source) != spec.source_stat:
        raise ReleaseError(
            f"source pathname identity changed during copy: {spec.source}"
        )


def _materialization_modes(
    specs: list[FileSpec], output_device: int, requested: str
) -> list[str]:
    if requested not in {"auto", "reflink", "copy"}:
        raise ReleaseError(f"unsupported copy mode: {requested}")
    modes: list[str] = []
    for spec in specs:
        same_device = spec.source_stat[0] == output_device
        if requested == "copy":
            mode = "copy"
        elif requested == "reflink":
            if not same_device:
                raise ReleaseError(
                    f"strict reflink mode cannot cross filesystems: {spec.source_id}"
                )
            mode = "reflink"
        elif same_device:
            mode = "reflink"
        elif spec.tier == "compact":
            mode = "copy"
        else:
            raise ReleaseError(
                f"auto mode refuses to copy a cross-filesystem full sweep: "
                f"{spec.sweep_id or spec.source_id}"
            )
        modes.append(mode)
    return modes


def _preflight_materialization(
    specs: list[FileSpec], modes: list[str], temp: Path
) -> None:
    if len(specs) != len(modes):
        raise ReleaseError("materialization preflight mode/spec mismatch")
    representatives: dict[tuple[str, int], FileSpec] = {}
    for spec, mode in zip(specs, modes, strict=True):
        key = (mode, spec.source_stat[0])
        current = representatives.get(key)
        if current is None or spec.source_stat[2] < current.source_stat[2]:
            representatives[key] = spec
    for index, ((mode, _), spec) in enumerate(sorted(representatives.items())):
        probe = temp / f".materialization-probe-{index}"
        _materialize(spec, probe, mode)
        probe_identity = stat_identity(probe)
        if probe_identity[:2] == spec.source_stat[:2]:
            raise ReleaseError("materialization preflight created a hard link")
        if probe_identity[2] != spec.source_stat[2]:
            raise ReleaseError("materialization preflight size mismatch")
        expected_digest = spec.planned_sha256 or sha256_file(spec.source)
        if sha256_file(probe) != expected_digest:
            raise ReleaseError("materialization preflight content mismatch")
        probe.unlink()


def _table_metadata(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                columns = next(reader)
            except StopIteration:
                columns = []
            return {
                "format": "csv",
                "row_count": sum(1 for _ in reader),
                "columns": columns,
            }
    if path.suffix.lower() == ".parquet":
        # The packaging process reads only Parquet footer metadata. Bound Polars'
        # global worker pool before its first import so this local verifier also
        # works inside conservative TasksMax cgroups on high-core-count hosts.
        os.environ["POLARS_MAX_THREADS"] = "1"
        try:
            import polars as pl
        except ImportError as exc:
            raise ReleaseError(
                "polars is required for compact Parquet metadata"
            ) from exc
        schema = pl.read_parquet_schema(path)
        rows = pl.scan_parquet(path).select(pl.len()).collect().item()
        return {
            "format": "parquet",
            "row_count": rows,
            "columns": list(schema),
            "schema": {key: str(value) for key, value in schema.items()},
        }
    return {}


def _git_value(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args], check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def _packaging_code_commit(repo_root: Path, registry_path: Path) -> str:
    commit = _git_value(repo_root, "rev-parse", "HEAD")
    for path in (Path(__file__).resolve(), registry_path.resolve()):
        try:
            relative = path.relative_to(repo_root.resolve()).as_posix()
        except ValueError as exc:
            raise ReleaseError(
                f"packaging input must be inside the Git checkout: {path}"
            ) from exc
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit}:{relative}"],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0 or result.stdout != path.read_bytes():
            raise ReleaseError(
                f"packaging code must be committed at HEAD before build: {relative}"
            )
    return commit


def _source_date_epoch(epoch: int | None, repo_root: Path) -> int:
    if epoch is not None:
        return epoch
    if value := os.environ.get("SOURCE_DATE_EPOCH"):
        return int(value)
    return int(_git_value(repo_root, "show", "-s", "--format=%ct", "HEAD"))


def _timestamp(epoch: int) -> str:
    return (
        datetime.fromtimestamp(epoch, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _validated_source_date_epoch(epoch: int | None, repo_root: Path) -> tuple[int, str]:
    value = _source_date_epoch(epoch, repo_root)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReleaseError("source_date_epoch must be an integer")
    if value < 0 or value > MAX_SOURCE_DATE_EPOCH:
        raise ReleaseError(
            f"source_date_epoch must be between 0 and {MAX_SOURCE_DATE_EPOCH}"
        )
    try:
        formatted = _timestamp(value)
    except (OverflowError, OSError, ValueError) as exc:
        raise ReleaseError(f"source_date_epoch is not representable: {value}") from exc
    return value, formatted


def _portable_registry(
    registry: dict[str, Any], preparation_registry_sha256: str
) -> dict[str, Any]:
    portable = copy.deepcopy(registry)
    portable["preparation_registry_sha256"] = preparation_registry_sha256
    for key in ("source", "sweep"):
        for item in portable.get(key, []):
            item.pop("root", None)
            item["logical_locator"] = item["id"]
    return portable


def _package_readme(registry: dict[str, Any], sweeps: list[dict[str, Any]]) -> str:
    facts = "\n".join(
        f"- `{item['id']}` ({item['cohort']}): {item['configuration_count']:,} configurations, {item['output_parquet_count']:,} outputs, {item['included_file_count']:,} files."
        for item in sweeps
    )
    return f"""# JUMP-Lite paper reproducibility data v{registry["package_version"]}\n\nThis local package is bound to manuscript `{registry["manuscript_commit"]}` and\nanalysis-code release `{registry["analysis_code_release"]}`. The manifest separately records\n`packaging_code_git_commit`, the exact post-release builder implementation used.\n`compact/` contains plot-ready inputs; `full_sweeps/` contains curated scientific\nsweeps. No uploader is provided.\n\nThe global release identity is the corrected JUMP-Lite cohort: {registry["cohort_wells"]:,}\nwells and {registry["cohort_sites"]:,} sites (`{registry["cohort_digest"]}`). Separate\nTarget-2, strict-MOTIVE, held-out, and broader-JUMP/nonlite inputs retain explicit\ncohort labels and are not represented as corrected-cohort aggregates.\n\n## Curated sweeps\n\n{facts}\n\nEvery configuration has metrics and `pipeline_config.yaml`; `output.parquet` is\nincluded when retained. Only registry-allowlisted `results/*` leaves are accepted.\nCaches, plots, backups, logs, and transient state are excluded.\n\nCellpose core scores intentionally omit jobs, logs, the whole-run checksum, and\nold presentation files. Final measured-Cellpose figure and table CSV/JSON receipts\nare stored separately; staging/final TeX is not packaged. The main-text MQ heatmap\nsubset retains its validation receipt, which binds the excluded PNG to the lock.\n"""


def _data_dictionary() -> str:
    return """# Data dictionary\n\n`manifest.json` is canonical JSON and records bytes and SHA-256 for every payload.\nCompact CSV/Parquet records also contain reproducibly verified row/schema metadata.\nFull-sweep closure is bound by sorted configuration and path+size digests, explicit\nresults leaf counts, and the declared output exception. `metadata/` contains the\nexact manuscript lock and a portable source registry with local roots removed.\n`figure_to_inputs.json` maps every generated locked artifact to exact package inputs.\n`analysis_code_release` names the scientific release; `packaging_code_git_commit`\nnames the exact committed builder/registry implementation used for packaging.\n"""


def _safe_remove_tree(path: Path, owned_identity: tuple[int, int] | None) -> None:
    if owned_identity is None:
        return
    try:
        info = path.stat(follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode) and (info.st_dev, info.st_ino) == owned_identity:
            shutil.rmtree(path)
    except FileNotFoundError:
        pass


def _safe_unlink(path: Path, owned_identity: tuple[int, int] | None) -> None:
    if owned_identity is None:
        return
    try:
        info = path.stat(follow_symlinks=False)
        if (info.st_dev, info.st_ino) == owned_identity and stat.S_ISREG(info.st_mode):
            path.unlink()
    except FileNotFoundError:
        pass


def _fsync_tree(root: Path) -> None:
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in directories + [root]:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _record_for_spec(
    spec: FileSpec, *, size: int, digest: str, metadata: dict[str, Any]
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "bytes": size,
        "cohort": spec.cohort,
        "destination": spec.destination.as_posix(),
        "role": spec.role,
        "sha256": digest,
        "source_id": spec.source_id,
        "source_relative_path": spec.source_relative.as_posix(),
        "tier": spec.tier,
        **metadata,
    }
    if spec.sweep_id:
        record["sweep_id"] = spec.sweep_id
    if spec.configuration:
        record["configuration"] = spec.configuration
    return record


def _preflight_compact_records(
    specs: list[FileSpec],
) -> dict[PurePosixPath, dict[str, Any]]:
    metadata_by_destination: dict[PurePosixPath, dict[str, Any]] = {}
    for spec in specs:
        if spec.tier != "compact":
            continue
        metadata = _table_metadata(spec.source)
        if stat_identity(spec.source) != spec.source_stat:
            raise ReleaseError(
                f"source mutated during compact metadata preflight: {spec.source}"
            )
        if (
            spec.expected_rows is not None
            and metadata.get("row_count") != spec.expected_rows
        ):
            raise ReleaseError(f"source row count differs for {spec.source_id}")
        record = _record_for_spec(
            spec,
            size=spec.source_stat[2],
            digest=spec.planned_sha256 or "0" * 64,
            metadata=metadata,
        )
        _validate_record(record)
        metadata_by_destination[spec.destination] = metadata
    return metadata_by_destination


def build_release(
    output: Path = DEFAULT_OUTPUT,
    *,
    registry_path: Path = REGISTRY_PATH,
    repo_root: Path = REPO_ROOT,
    allowed_root: Path = ALLOWED_OUTPUT_ROOT,
    copy_mode: str = "auto",
    epoch: int | None = None,
) -> dict[str, Any]:
    # Validate and format caller-controlled time before any output-parent, lock,
    # temporary-directory, or payload write can occur.
    source_epoch, created_at = _validated_source_date_epoch(epoch, repo_root)
    packaging_code_git_commit = _packaging_code_commit(repo_root, registry_path)
    output, _ = _validate_output(output, allowed_root, create_direct_parent=True)
    registry, specs, sweeps, figure_map, compact_summaries, _ = inventory(
        registry_path, repo_root
    )
    compact_metadata = _preflight_compact_records(specs)
    _ensure_no_overlap(specs, output, registry, repo_root)
    output_parent_identity = _directory_identity(output.parent)
    materialization_modes = _materialization_modes(
        specs, output_parent_identity[0], copy_mode
    )
    materialization_counts = Counter(materialization_modes)
    materialization_bytes: Counter[str] = Counter()
    for spec, mode in zip(specs, materialization_modes, strict=True):
        materialization_bytes[mode] += spec.source_stat[2]
    lock_path = repo_root / "paper_artifacts.lock.json"
    preparation_registry_sha256 = sha256_file(registry_path)
    portable_registry_bytes = canonical_json(
        _portable_registry(registry, preparation_registry_sha256)
    )
    lock_bytes = lock_path.read_bytes()

    lock_file = output.parent / f".{output.name}.build.lock"
    temp = output.parent / f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    lock_fd: int | None = None
    lock_identity: tuple[int, int] | None = None
    temp_identity: tuple[int, int] | None = None
    try:
        try:
            lock_fd = os.open(
                lock_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444
            )
        except FileExistsError as exc:
            raise ReleaseError(f"exclusive build lock exists: {lock_file}") from exc
        lock_info = os.fstat(lock_fd)
        lock_identity = (lock_info.st_dev, lock_info.st_ino)
        os.write(lock_fd, f"pid={os.getpid()}\noutput={output.name}\n".encode())
        os.fsync(lock_fd)
        temp.mkdir(mode=0o755)
        temp_identity = _directory_identity(temp)
        if (
            temp_identity[0] != output_parent_identity[0]
            or _directory_identity(output.parent) != output_parent_identity
        ):
            raise ReleaseError("output parent identity changed before preflight")
        _preflight_materialization(specs, materialization_modes, temp)

        records: list[dict[str, Any]] = []
        for index, (spec, mode) in enumerate(
            zip(specs, materialization_modes, strict=True), start=1
        ):
            destination = temp.joinpath(*spec.destination.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _materialize(spec, destination, mode)
            destination_identity = stat_identity(destination)
            if destination_identity[:2] == spec.source_stat[:2]:
                raise ReleaseError(
                    f"materialized hard link refused: {spec.destination}"
                )
            size = destination_identity[2]
            digest = sha256_file(destination)
            if size != spec.source_stat[2] or (
                spec.planned_sha256 and digest != spec.planned_sha256
            ):
                raise ReleaseError(
                    f"materialized identity differs for {spec.destination}"
                )
            metadata = _table_metadata(destination) if spec.tier == "compact" else {}
            if (
                spec.tier == "compact"
                and metadata != compact_metadata[spec.destination]
            ):
                raise ReleaseError(
                    f"copied table metadata differs for {spec.source_id}"
                )
            record = _record_for_spec(spec, size=size, digest=digest, metadata=metadata)
            _validate_record(record)
            records.append(record)
            if index % 1000 == 0:
                print(f"materialized {index:,}/{len(specs):,} files", file=sys.stderr)

        generated = {
            "README.md": _package_readme(registry, sweeps).encode(),
            "DATA_DICTIONARY.md": _data_dictionary().encode(),
            FIGURE_MAP_NAME: canonical_json(
                {"schema_version": 1, "mappings": figure_map}
            ),
            PACKAGED_LOCK: lock_bytes,
            PACKAGED_REGISTRY: portable_registry_bytes,
        }
        for name, content in generated.items():
            destination = temp / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            records.append(
                {
                    "bytes": len(content),
                    "cohort": "package-metadata",
                    "destination": name,
                    "role": "portable package metadata",
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "source_id": "package-builder",
                    "source_relative_path": name,
                    "tier": "metadata",
                }
            )

        manifest = {
            "schema_version": 2,
            "package_version": registry["package_version"],
            "source_date_epoch": source_epoch,
            "created_at": created_at,
            "analysis_code_release": registry["analysis_code_release"],
            "packaging_code_git_commit": packaging_code_git_commit,
            "manuscript_commit": registry["manuscript_commit"],
            "manuscript_tree": registry["manuscript_tree"],
            "cohort": {
                "label": registry["cohort_label"],
                "wells": registry["cohort_wells"],
                "sites": registry["cohort_sites"],
                "site_key_sha256": registry["cohort_digest"],
            },
            "bindings": {
                "builder_sha256": sha256_file(Path(__file__)),
                "paper_artifacts_lock_sha256": hashlib.sha256(lock_bytes).hexdigest(),
                "portable_registry_sha256": hashlib.sha256(
                    portable_registry_bytes
                ).hexdigest(),
                "preparation_registry_sha256": preparation_registry_sha256,
            },
            "compact_sources": compact_summaries,
            "files": sorted(records, key=lambda item: item["destination"]),
            "sweeps": sweeps,
            "exclusions": [
                ".pipeline_cache",
                "plots*",
                "sweep_results_BAK.csv",
                "logs",
                "cache/temp/backup material",
                "Cellpose jobs/logs/old presentation/whole-run checksum",
            ],
        }
        manifest_bytes = canonical_json(manifest)
        with (temp / MANIFEST_NAME).open("xb") as handle:
            handle.write(manifest_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        verify_release(temp)
        for path in temp.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        _fsync_tree(temp)
        if output.exists() or output.is_symlink():
            raise ReleaseError(f"output appeared during build: {output}")
        _atomic_publish_no_replace(temp, output)
        temp_identity = None
        parent_fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return {
            "status": "built",
            "output": str(output),
            "file_count": len(records),
            "bytes": sum(item["bytes"] for item in records),
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "materialization": {
                "requested_mode": copy_mode,
                "file_counts": dict(sorted(materialization_counts.items())),
                "bytes": dict(sorted(materialization_bytes.items())),
            },
        }
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        _safe_remove_tree(temp, temp_identity)
        _safe_unlink(lock_file, lock_identity)


def _require(value: Any, expected_type: type, label: str) -> Any:
    if not isinstance(value, expected_type) or (
        expected_type is int and isinstance(value, bool)
    ):
        raise ReleaseError(f"invalid {label}")
    return value


def _validate_manifest_header(manifest: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "package_version",
        "source_date_epoch",
        "created_at",
        "analysis_code_release",
        "packaging_code_git_commit",
        "manuscript_commit",
        "manuscript_tree",
        "cohort",
        "bindings",
        "compact_sources",
        "files",
        "sweeps",
        "exclusions",
    }
    if set(manifest) != required or manifest.get("schema_version") != 2:
        raise ReleaseError("invalid manifest schema")
    if not isinstance(manifest.get("package_version"), str) or not VERSION_RE.fullmatch(
        manifest["package_version"]
    ):
        raise ReleaseError("invalid package_version")
    epoch = _require(manifest.get("source_date_epoch"), int, "source_date_epoch")
    if (
        epoch < 0
        or epoch > MAX_SOURCE_DATE_EPOCH
        or manifest.get("created_at") != _timestamp(epoch)
    ):
        raise ReleaseError("invalid deterministic timestamp")
    for key in (
        "analysis_code_release",
        "packaging_code_git_commit",
        "manuscript_commit",
        "manuscript_tree",
    ):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ReleaseError(f"invalid manifest {key}")
    for key in ("packaging_code_git_commit", "manuscript_commit", "manuscript_tree"):
        if not re.fullmatch(r"[0-9a-f]{40}", manifest[key]):
            raise ReleaseError(f"invalid manifest {key}")
    cohort = _require(manifest.get("cohort"), dict, "cohort")
    if set(cohort) != {"label", "wells", "sites", "site_key_sha256"}:
        raise ReleaseError("invalid cohort schema")
    if not isinstance(cohort["label"], str) or not cohort["label"]:
        raise ReleaseError("invalid cohort label")
    if any(
        _require(cohort[key], int, f"cohort {key}") <= 0 for key in ("wells", "sites")
    ):
        raise ReleaseError("invalid cohort counts")
    if not isinstance(cohort["site_key_sha256"], str) or not SHA256_RE.fullmatch(
        cohort["site_key_sha256"]
    ):
        raise ReleaseError("invalid cohort digest")
    bindings = _require(manifest.get("bindings"), dict, "bindings")
    if set(bindings) != {
        "builder_sha256",
        "paper_artifacts_lock_sha256",
        "portable_registry_sha256",
        "preparation_registry_sha256",
    } or any(
        not isinstance(value, str) or not SHA256_RE.fullmatch(value)
        for value in bindings.values()
    ):
        raise ReleaseError("invalid bindings")
    for key in ("compact_sources", "files", "sweeps", "exclusions"):
        _require(manifest.get(key), list, key)


def _validate_record(record: dict[str, Any]) -> None:
    required = {
        "bytes",
        "cohort",
        "destination",
        "role",
        "sha256",
        "source_id",
        "source_relative_path",
        "tier",
    }
    optional = {"format", "row_count", "columns", "schema", "sweep_id", "configuration"}
    if set(record) - required - optional or not required <= set(record):
        raise ReleaseError(f"invalid record keys for {record.get('destination')}")
    for key in (
        "cohort",
        "destination",
        "role",
        "sha256",
        "source_id",
        "source_relative_path",
        "tier",
    ):
        if not _require(record[key], str, f"record {key}"):
            raise ReleaseError(f"empty record {key}")
    if _require(record["bytes"], int, "record bytes") < 0 or not SHA256_RE.fullmatch(
        record["sha256"]
    ):
        raise ReleaseError("invalid record size or SHA-256")
    validate_relative(record["destination"], label="manifest destination")
    validate_relative(record["source_relative_path"], label="source relative path")
    if record["tier"] not in {"compact", "full_sweep", "metadata"}:
        raise ReleaseError("invalid record tier")
    if record["tier"] == "full_sweep":
        if "sweep_id" not in record or (
            record["source_relative_path"] != "sweep_results.csv"
            and "configuration" not in record
        ):
            raise ReleaseError("full-sweep record lacks identity")
    if "sweep_id" in record and (
        not isinstance(record["sweep_id"], str) or not record["sweep_id"]
    ):
        raise ReleaseError("invalid sweep_id")
    if "configuration" in record:
        validate_relative(
            _require(record["configuration"], str, "configuration"),
            label="configuration",
        )
    if "format" in record:
        if record["tier"] != "compact" or record["format"] not in {"csv", "parquet"}:
            raise ReleaseError("invalid table metadata scope")
        if _require(record.get("row_count"), int, "row_count") < 0:
            raise ReleaseError("invalid row_count")
        columns = _require(record.get("columns"), list, "columns")
        # Empty names are legitimate in accepted historical CSV and Parquet
        # inputs (typically a serialized index); preserve them byte-for-byte.
        if any(not isinstance(column, str) for column in columns):
            raise ReleaseError("invalid table columns")
        if record["format"] == "parquet":
            schema = _require(record.get("schema"), dict, "Parquet schema")
            if (
                len(schema) != len(columns)
                or set(schema) != set(columns)
                or any(
                    not isinstance(key, str) or not isinstance(value, str)
                    for key, value in schema.items()
                )
            ):
                raise ReleaseError("invalid Parquet schema")
        elif "schema" in record:
            raise ReleaseError("CSV record must not contain Parquet schema")


def _verify_sweeps(
    root: Path,
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    registry: dict[str, Any],
) -> None:
    manifest_sweeps = {item["id"]: item for item in manifest["sweeps"]}
    registry_sweeps = {item["id"]: item for item in registry["sweep"]}
    record_sweep_ids = {
        item["sweep_id"] for item in records if item["tier"] == "full_sweep"
    }
    if (
        len(manifest_sweeps) != len(manifest["sweeps"])
        or len(registry_sweeps) != len(registry["sweep"])
        or set(manifest_sweeps) != set(registry_sweeps)
        or record_sweep_ids != set(registry_sweeps)
    ):
        raise ReleaseError("sweep ID set differs from portable registry")
    for sweep_id, source in registry_sweeps.items():
        summary = manifest_sweeps[sweep_id]
        expected_summary_keys = {
            "id",
            "role",
            "cohort",
            "configuration_count",
            "output_parquet_count",
            "missing_output_parquet",
            "configuration_sha256",
            "inventory_sha256",
            "included_file_count",
            "included_bytes",
            "results_leaf_counts",
            "excluded_file_count",
            "excluded_policy",
        }
        if (
            set(summary) != expected_summary_keys
            or summary.get("role") != source["role"]
            or summary.get("cohort") != source["cohort"]
            or not isinstance(summary.get("excluded_file_count"), int)
            or summary["excluded_file_count"] < 0
            or not isinstance(summary.get("excluded_policy"), list)
        ):
            raise ReleaseError(f"invalid sweep summary schema: {sweep_id}")
        sweep_records = [item for item in records if item.get("sweep_id") == sweep_id]
        prefix = validate_relative(
            source["destination"], label="sweep destination"
        ).as_posix()
        if any(
            item["source_id"] != sweep_id
            or item["destination"] != f"{prefix}/{item['source_relative_path']}"
            for item in sweep_records
        ):
            raise ReleaseError(f"sweep record namespace mismatch: {sweep_id}")
        relative = {item["source_relative_path"]: item for item in sweep_records}
        if len(relative) != len(sweep_records):
            raise ReleaseError(f"duplicate sweep relative path: {sweep_id}")
        by_config: dict[str, set[str]] = defaultdict(set)
        leaves: Counter[str] = Counter()
        for path, record in relative.items():
            parts = PurePosixPath(path).parts
            if path == "sweep_results.csv":
                if "configuration" in record:
                    raise ReleaseError("sweep summary has a configuration")
                continue
            if not (
                len(parts) == 3
                and parts[2] in {"output.parquet", "pipeline_config.yaml"}
                or len(parts) == 4
                and parts[2] == "results"
                and parts[3] in source["allowed_results"]
            ):
                raise ReleaseError(f"invalid sweep relative path: {path}")
            configuration = "/".join(parts[:2])
            if record.get("configuration") != configuration:
                raise ReleaseError(f"configuration field mismatch: {path}")
            leaf = "/".join(parts[2:])
            by_config[configuration].add(leaf)
            if len(parts) == 4 and parts[2] == "results":
                leaves[parts[3]] += 1
        summary_record = relative.get("sweep_results.csv")
        if (
            summary_record is None
            or summary_record["sha256"] != source["expected_summary_sha256"]
            or csv_rows(root / summary_record["destination"])
            != source["expected_summary_rows"]
        ):
            raise ReleaseError(f"sweep summary identity mismatch: {sweep_id}")
        configs = set(by_config)
        metrics = {
            key for key, values in by_config.items() if "results/metrics.json" in values
        }
        pipelines = {
            key for key, values in by_config.items() if "pipeline_config.yaml" in values
        }
        outputs = {
            key for key, values in by_config.items() if "output.parquet" in values
        }
        missing = sorted(metrics - outputs)
        observed = {
            "configuration_count": len(metrics),
            "output_parquet_count": len(outputs),
            "missing_output_parquet": missing,
            "configuration_sha256": _configuration_digest(metrics),
            "inventory_sha256": _inventory_digest(
                (path, item["bytes"], None) for path, item in relative.items()
            ),
            "included_file_count": len(sweep_records),
            "included_bytes": sum(item["bytes"] for item in sweep_records),
            "results_leaf_counts": dict(sorted(leaves.items())),
        }
        expected = {
            "configuration_count": source["expected_configs"],
            "output_parquet_count": source["expected_outputs"],
            "missing_output_parquet": sorted(source["allowed_missing_output"]),
            "configuration_sha256": source["expected_configuration_sha256"],
            "inventory_sha256": source["expected_inventory_sha256"],
            "included_file_count": source["expected_file_count"],
            "included_bytes": source["expected_bytes"],
            "results_leaf_counts": dict(sorted(source["allowed_results"].items())),
        }
        if configs != metrics or pipelines != metrics or observed != expected:
            raise ReleaseError(f"verified sweep closure mismatch: {sweep_id}")
        for key, value in observed.items():
            if summary.get(key) != value:
                raise ReleaseError(f"manifest sweep summary mismatch: {sweep_id}/{key}")


def _validate_packaged_source_metadata(
    root: Path, source: dict[str, Any], items: list[dict[str, Any]]
) -> None:
    selected = {item["source_relative_path"]: item for item in items}
    if len(selected) != len(items):
        raise ReleaseError(f"duplicate compact relative path: {source['id']}")

    def payload(name: str) -> Path:
        if name not in selected:
            raise ReleaseError(
                f"selected metadata path is absent: {source['id']}/{name}"
            )
        return root / selected[name]["destination"]

    if manifest_name := source.get("checksum_manifest"):
        entries: dict[str, str] = {}
        for line in payload(manifest_name).read_text().splitlines():
            digest, separator, name = line.partition("  ")
            if not separator or not SHA256_RE.fullmatch(digest):
                raise ReleaseError(f"invalid packaged checksum line: {source['id']}")
            entries[name] = digest
        expected = (
            set(selected) - {manifest_name} - set(source.get("checksum_unlisted", []))
        )
        if set(entries) != expected or any(
            selected[name]["sha256"] != digest for name, digest in entries.items()
        ):
            raise ReleaseError(f"packaged checksum closure mismatch: {source['id']}")
        if "completion.json" in source.get("checksum_unlisted", []):
            completion = json.loads(payload("completion.json").read_text())
            if completion.get("sha256s_sha256") != selected[manifest_name]["sha256"]:
                raise ReleaseError(
                    f"packaged completion receipt mismatch: {source['id']}"
                )
    if manifest_name := source.get("json_checksum_manifest"):
        ledger = json.loads(payload(manifest_name).read_text())
        entries = {
            entry["path"]: entry["sha256"] for entry in ledger.get("artifacts", [])
        }
        if set(entries) != set(selected) - {manifest_name} or any(
            selected[name]["sha256"] != digest for name, digest in entries.items()
        ):
            raise ReleaseError(
                f"packaged JSON checksum closure mismatch: {source['id']}"
            )
    if manifest_name := source.get("subset_checksum_manifest"):
        entries = {}
        for line in payload(manifest_name).read_text().splitlines():
            digest, separator, name = line.partition("  ")
            if not separator or not SHA256_RE.fullmatch(digest):
                raise ReleaseError(
                    f"invalid packaged subset checksum line: {source['id']}"
                )
            entries[name] = digest
        if any(
            name != manifest_name and entries.get(name) != item["sha256"]
            for name, item in selected.items()
        ):
            raise ReleaseError(f"packaged subset checksum mismatch: {source['id']}")
    if receipt_name := source.get("receipt_file"):
        receipt = json.loads(payload(receipt_name).read_text())
        for path, expected in source.get("expected_json_values", {}).items():
            if _json_value(receipt, path) != expected:
                raise ReleaseError(f"packaged receipt mismatch: {source['id']}/{path}")
        artifacts = receipt.get("artifacts", {})
        for name, expected in source.get("expected_artifact_checksums", {}).items():
            if artifacts.get(name) != expected or selected[name]["sha256"] != expected:
                raise ReleaseError(
                    f"packaged receipt artifact mismatch: {source['id']}/{name}"
                )
        for path, expected in source.get("expected_json_lists", {}).items():
            if _json_value(receipt, path) != expected:
                raise ReleaseError(
                    f"packaged receipt list mismatch: {source['id']}/{path}"
                )


def verify_release(
    root: Path, *, expected_manifest_sha256: str | None = None
) -> dict[str, Any]:
    ensure_no_symlink_components(root)
    if not root.is_dir():
        raise ReleaseError(f"package is not a directory: {root}")
    manifest_path = root / MANIFEST_NAME
    manifest_digest = sha256_file(manifest_path)
    if expected_manifest_sha256 is not None:
        if (
            not SHA256_RE.fullmatch(expected_manifest_sha256)
            or manifest_digest != expected_manifest_sha256
        ):
            raise ReleaseError("manifest SHA-256 does not match external attestation")
    manifest = json.loads(manifest_path.read_text())
    if canonical_json(manifest) != manifest_path.read_bytes():
        raise ReleaseError("manifest is not canonical sorted JSON")
    _validate_manifest_header(manifest)
    records = manifest["files"]

    expected_paths = {MANIFEST_NAME}
    total = 0
    for record in records:
        _validate_record(_require(record, dict, "file record"))
        value = record["destination"]
        if value in expected_paths:
            raise ReleaseError(f"duplicate manifest destination: {value}")
        expected_paths.add(value)
        path = root.joinpath(*PurePosixPath(value).parts)
        ensure_no_symlink_components(path)
        identity = stat_identity(path)
        if identity[2] != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ReleaseError(f"payload identity mismatch: {value}")
        if "format" in record and _table_metadata(path) != {
            key: record[key]
            for key in ("format", "row_count", "columns", "schema")
            if key in record
        }:
            raise ReleaseError(f"table metadata mismatch: {value}")
        total += identity[2]
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ReleaseError(f"invalid package entry: {path}")
    if actual != expected_paths:
        raise ReleaseError(
            f"package closure mismatch; missing={sorted(expected_paths - actual)}, extra={sorted(actual - expected_paths)}"
        )

    record_by_path = {item["destination"]: item for item in records}
    for path in (PACKAGED_LOCK, PACKAGED_REGISTRY, FIGURE_MAP_NAME):
        if path not in record_by_path:
            raise ReleaseError(f"missing portable metadata: {path}")
    lock_bytes = (root / PACKAGED_LOCK).read_bytes()
    registry_bytes = (root / PACKAGED_REGISTRY).read_bytes()
    if (
        manifest["bindings"]["paper_artifacts_lock_sha256"]
        != hashlib.sha256(lock_bytes).hexdigest()
        or manifest["bindings"]["portable_registry_sha256"]
        != hashlib.sha256(registry_bytes).hexdigest()
    ):
        raise ReleaseError("portable metadata binding mismatch")
    lock = json.loads(lock_bytes)
    registry = json.loads(registry_bytes)
    if (
        canonical_json(registry) != registry_bytes
        or registry.get("schema_version") != 2
        or any(
            "root" in item or item.get("logical_locator") != item.get("id")
            for key in ("source", "sweep")
            for item in registry.get(key, [])
        )
    ):
        raise ReleaseError("invalid portable source registry")
    if (
        registry["package_version"] != manifest["package_version"]
        or registry["analysis_code_release"] != manifest["analysis_code_release"]
        or registry.get("preparation_registry_sha256")
        != manifest["bindings"]["preparation_registry_sha256"]
        or lock.get("manuscript_commit") != manifest["manuscript_commit"]
        or lock.get("tree") != manifest["manuscript_tree"]
    ):
        raise ReleaseError("portable metadata identity mismatch")

    compact_records = [item for item in records if item["tier"] == "compact"]
    destinations: dict[str, list[str]] = defaultdict(list)
    compact_relative: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in compact_records:
        destinations[record["source_id"]].append(record["destination"])
        compact_relative[record["source_id"]].append(record)
    if any(
        not isinstance(item, dict) or "id" not in item
        for item in manifest["compact_sources"]
    ):
        raise ReleaseError("invalid compact source summaries")
    source_summaries = {item["id"]: item for item in manifest["compact_sources"]}
    registry_sources = {item["id"]: item for item in registry["source"]}
    if (
        len(registry_sources) != len(registry["source"])
        or len(source_summaries) != len(manifest["compact_sources"])
        or set(destinations) != set(source_summaries)
        or set(destinations) != set(registry_sources)
    ):
        raise ReleaseError("compact source ID closure mismatch")
    for source_id, items in compact_relative.items():
        summary = source_summaries[source_id]
        source = registry_sources[source_id]
        if set(summary) != {
            "id",
            "role",
            "cohort",
            "closure",
            "file_count",
            "bytes",
            "inventory_sha256",
            "destinations",
        } or any(
            summary.get(key) != source[key]
            for key in ("id", "role", "cohort", "closure")
        ):
            raise ReleaseError(f"invalid compact source summary: {source_id}")
        base_destination = source["destination"]
        if any(
            item["role"] != source["role"]
            or item["cohort"] != source["cohort"]
            or item["destination"]
            != (
                base_destination
                if "files" not in source and "include" not in source
                else f"{base_destination}/{item['source_relative_path']}"
            )
            for item in items
        ):
            raise ReleaseError(f"compact record namespace mismatch: {source_id}")
        observed = {
            "file_count": len(items),
            "bytes": sum(item["bytes"] for item in items),
            "inventory_sha256": _inventory_digest(
                (item["source_relative_path"], item["bytes"], item["sha256"])
                for item in items
            ),
            "destinations": sorted(item["destination"] for item in items),
        }
        if any(summary.get(key) != value for key, value in observed.items()):
            raise ReleaseError(f"compact manifest inventory mismatch: {source_id}")
        if source["closure"] == "single-pinned":
            if (
                items[0]["sha256"] != source["expected_sha256"]
                or items[0]["bytes"] != source["expected_bytes"]
            ):
                raise ReleaseError(f"single compact identity mismatch: {source_id}")
        elif (
            observed["file_count"],
            observed["bytes"],
            observed["inventory_sha256"],
        ) != (
            source["expected_file_count"],
            source["expected_bytes"],
            source["expected_inventory_sha256"],
        ):
            raise ReleaseError(f"compact registry inventory mismatch: {source_id}")
        if (
            source.get("expected_rows") is not None
            and items[0].get("row_count") != source["expected_rows"]
        ):
            raise ReleaseError(f"compact row count mismatch: {source_id}")
        if source.get("expected_distinct_keys") is not None:
            _validate_strict_motive_path(source, root / items[0]["destination"])
        _validate_packaged_source_metadata(root, source, items)

    figure = json.loads((root / FIGURE_MAP_NAME).read_text())
    if (
        canonical_json(figure) != (root / FIGURE_MAP_NAME).read_bytes()
        or figure.get("schema_version") != 1
    ):
        raise ReleaseError("invalid figure map")
    _validate_lock_hash_assertions(registry, lock)
    expected_figure = _figure_map_from_destinations(registry, destinations, lock)
    if figure.get("mappings") != expected_figure:
        raise ReleaseError(
            "figure map differs from lock/registry/record reconstruction"
        )
    _verify_sweeps(root, records, manifest, registry)
    return {
        "status": "verified",
        "package_version": manifest["package_version"],
        "file_count": len(records),
        "bytes": total,
        "manifest_sha256": manifest_digest,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("plan", help="read-only closed inventory validation")
    build = commands.add_parser("build", help="create a new atomic local package")
    build.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build.add_argument(
        "--copy-mode", choices=("auto", "reflink", "copy"), default="auto"
    )
    build.add_argument("--source-date-epoch", type=int)
    verify = commands.add_parser(
        "verify", help="read-only complete package verification"
    )
    verify.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    verify.add_argument("--expected-manifest-sha256")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        if args.command == "plan":
            result = plan(args.registry, REPO_ROOT)
        elif args.command == "build":
            result = build_release(
                args.output,
                registry_path=args.registry,
                copy_mode=args.copy_mode,
                epoch=args.source_date_epoch,
            )
        else:
            result = verify_release(
                args.root, expected_manifest_sha256=args.expected_manifest_sha256
            )
    except (
        ReleaseError,
        OSError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
