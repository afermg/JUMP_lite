"""Fail-closed resource governor; it mutates only compression control JSON."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any

from .model import (
    COMPRESSION_CPUS,
    INITIAL_WORKERS,
    LIVE_STATE_PARENT,
    MAX_CUMULATIVE_ERRORS,
    MAX_WORKERS,
    assert_no_symlinks,
    atomic_json,
)


@dataclass(frozen=True)
class GovernorPaths:
    candidate_id: str
    config_sha256: str
    state_root: Path
    mq_state: Path
    lossless_state: Path
    mq_mask_receipts: Path
    lossless_mask_receipts: Path
    mq_profiles: Path
    lossless_profiles: Path
    output_filesystem: Path = Path("/work/datasets")
    test_mode: bool = False

    @property
    def compression_state(self) -> Path:
        return self.state_root / "compression.json"

    @property
    def control(self) -> Path:
        return self.state_root / "control.json"

    @property
    def snapshots(self) -> Path:
        return self.state_root / "governor_snapshots"

    def validate(self) -> None:
        if not self.candidate_id or any(
            char not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
            for char in self.candidate_id
        ):
            raise ValueError("invalid governor candidate id")
        if len(self.config_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.config_sha256
        ):
            raise ValueError("invalid governor config digest")
        if self.test_mode:
            boundary = self.state_root.parent
            if self.state_root.name != self.candidate_id:
                raise ValueError("test governor state root must end in candidate id")
        else:
            boundary = LIVE_STATE_PARENT
            expected = LIVE_STATE_PARENT / self.candidate_id
            if self.state_root.absolute() != expected.absolute():
                raise ValueError("live governor state root literal path drift")
        if boundary.is_symlink() or (
            boundary.exists() and boundary.resolve() != boundary.absolute()
        ):
            raise ValueError("governor state parent redirect rejected")
        assert_no_symlinks(self.state_root, boundary)
        if self.state_root.exists() and not self.state_root.is_dir():
            raise ValueError("governor state root is not a directory")
        for path in (
            self.compression_state,
            self.control,
            self.snapshots,
            self.snapshots / "latest.json",
        ):
            assert_no_symlinks(path, self.state_root)
            if path.is_symlink():
                raise ValueError(f"governor writable symlink rejected: {path}")


def _affinity(pid: int) -> list[int]:
    if pid <= 0 or not Path(f"/proc/{pid}").exists():
        raise RuntimeError(f"PID unavailable: {pid}")
    return sorted(os.sched_getaffinity(pid))


def _service(name: str) -> dict[str, Any]:
    text = subprocess.check_output(
        [
            "systemctl",
            "--user",
            "show",
            name,
            "-p",
            "ActiveState",
            "-p",
            "MainPID",
            "-p",
            "ControlGroup",
        ],
        text=True,
    )
    fields = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    pid = int(fields.get("MainPID", "0"))
    cgroup = Path("/sys/fs/cgroup") / fields.get("ControlGroup", "").lstrip("/")
    pids = []
    if pid:
        if not (cgroup / "cgroup.procs").is_file():
            raise RuntimeError(f"cgroup telemetry missing: {name}")
        pid_set = {pid}
        for process_file in cgroup.rglob("cgroup.procs"):
            pid_set.update(int(x) for x in process_file.read_text().split())
        pids = sorted(pid_set)
    return {
        "name": name,
        "active": fields.get("ActiveState"),
        "pid": pid,
        "process_affinities": {str(x): _affinity(x) for x in pids},
    }


def _memory_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable missing")


def _io_pressure() -> float:
    for line in Path("/proc/pressure/io").read_text().splitlines():
        if line.startswith("some "):
            return float(dict(x.split("=") for x in line.split()[1:])["avg10"])
    raise RuntimeError("I/O pressure missing")


def _bounded_artifact_count(root: Path, suffix: str, expected: int) -> int:
    """Count current artifacts read-only, without following links or unbounded trees."""
    if expected < 0 or expected > 2_000_000:
        raise RuntimeError("authoritative expected-site bound invalid")
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"authoritative progress root unavailable: {root}")
    pending = [root]
    directories = 0
    count = 0
    while pending:
        current = pending.pop()
        directories += 1
        if directories > 100_000:
            raise RuntimeError("authoritative progress directory bound exceeded")
        with os.scandir(current) as entries:
            for entry in entries:
                if entry.is_symlink():
                    raise RuntimeError(
                        f"symlink in authoritative progress root: {entry.path}"
                    )
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False) and entry.name.endswith(
                    suffix
                ):
                    count += 1
                    if count > expected:
                        raise RuntimeError(
                            "authoritative progress exceeds expected sites"
                        )
    return count


def collect_metrics(
    paths: GovernorPaths, now_unix: float | None = None
) -> dict[str, Any]:
    paths.validate()
    timestamp = time.time() if now_unix is None else now_unix
    states = {}
    worker_affinities = {}
    for name, path in (("MQ", paths.mq_state), ("lossless", paths.lossless_state)):
        if not path.is_file():
            raise RuntimeError(f"feature telemetry missing: {path}")
        state = json.loads(path.read_text())
        states[name] = state
        pids = state.get("worker_pids")
        if not isinstance(pids, list) or not pids:
            raise RuntimeError(f"profile worker PID telemetry missing: {name}")
        worker_affinities[name] = {str(pid): _affinity(int(pid)) for pid in pids}
    authoritative = {}
    progress_paths = {
        "MQ": (paths.mq_mask_receipts, paths.mq_profiles),
        "lossless": (paths.lossless_mask_receipts, paths.lossless_profiles),
    }
    for codec, (mask_root, profile_root) in progress_paths.items():
        expected = int(states[codec]["expected_sites"])
        authoritative[codec] = {
            "receipt_backed_masks": _bounded_artifact_count(
                mask_root, ".json", expected
            ),
            "canonical_profiles": _bounded_artifact_count(
                profile_root, ".parquet", expected
            ),
            "mask_receipt_root": str(mask_root.absolute()),
            "profile_root": str(profile_root.absolute()),
        }
    if not paths.compression_state.is_file() or paths.compression_state.is_symlink():
        raise RuntimeError("compression telemetry missing")
    compression = json.loads(paths.compression_state.read_text())
    return {
        "observed_at_unix": timestamp,
        "candidate_id": paths.candidate_id,
        "config_sha256": paths.config_sha256,
        "feature_states": states,
        "profile_worker_affinities": worker_affinities,
        "authoritative_progress": authoritative,
        "feature_services": [
            _service("jump-lite-cp-segment-MQ-direct-v2.service"),
            _service("jump-lite-cp-segment-lossless-direct-v2.service"),
        ],
        "compression": compression,
        "load1": os.getloadavg()[0],
        "memory_available_bytes": _memory_available(),
        "storage_available_bytes": shutil.disk_usage(paths.output_filesystem).free,
        "io_pressure_avg10": _io_pressure(),
        "compression_cpus": list(COMPRESSION_CPUS),
    }


def _overlap(affinities: dict[str, list[int]]) -> bool:
    return any(set(value) & set(COMPRESSION_CPUS) for value in affinities.values())


def _validated_previous(
    previous: dict[str, Any] | None,
    metrics: dict[str, Any],
    errors: int,
) -> tuple[dict[str, Any], str | None]:
    if previous is None:
        return {}, "prior control missing"
    try:
        acknowledgement = int(previous["acknowledged_error_count"])
        valid = (
            previous.get("format_version") == "full-jump-compression-control-v2"
            and previous.get("candidate_id") == metrics.get("candidate_id")
            and previous.get("config_sha256") == metrics.get("config_sha256")
            and previous.get("compression_cpus") == list(COMPRESSION_CPUS)
            and previous.get("max_workers") == MAX_WORKERS
            and previous.get("feature_processes_mutated") is False
            and isinstance(previous.get("paused"), bool)
            and type(previous.get("desired_workers")) is int
            and 1 <= previous["desired_workers"] <= MAX_WORKERS
            and type(previous.get("consecutive_healthy_windows")) is int
            and 0 <= previous["consecutive_healthy_windows"] < 2
            and type(previous.get("acknowledged_error_count")) is int
            and 0 <= acknowledgement <= min(errors, MAX_CUMULATIVE_ERRORS)
            and (
                (
                    acknowledgement == 0
                    and previous.get("acknowledgement_source") == "zero"
                )
                or (
                    acknowledgement > 0
                    and previous.get("acknowledgement_source") == "explicit-cli"
                )
            )
            and 0
            <= float(metrics.get("observed_at_unix", 0))
            - float(previous.get("observed_at_unix", 0))
            <= 4 * 3600
        )
    except Exception:
        valid = False
    if not valid:
        return {}, "prior control identity/shape/age/acknowledgement invalid"
    return previous, None


def decide(
    metrics: dict[str, Any],
    previous: dict[str, Any] | None = None,
    previous_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failures = []
    now_ts = float(metrics.get("observed_at_unix", 0))
    compression = metrics.get("compression", {})
    try:
        observed_errors = int(compression["cumulative_errors"])
        if not 0 <= observed_errors <= MAX_CUMULATIVE_ERRORS:
            raise ValueError("error count outside bound")
    except Exception:
        observed_errors = 0
        failures.append("compression cumulative errors invalid")
    effective_previous, prior_failure = _validated_previous(
        previous, metrics, observed_errors
    )
    if prior_failure:
        failures.append(prior_failure)
    states = metrics.get("feature_states", {})
    if set(states) != {"MQ", "lossless"}:
        failures.append("missing feature state")
    for codec in ("MQ", "lossless"):
        state = states.get(codec, {})
        try:
            if not 0 <= now_ts - float(state["heartbeat_unix"]) <= 300:
                failures.append(f"{codec} heartbeat stale")
            expected = int(state["expected_sites"])
            current = metrics.get("authoritative_progress", {}).get(codec, {})
            masks = int(current["receipt_backed_masks"])
            profiles = int(current["canonical_profiles"])
            if not 0 <= masks <= expected or not 0 <= profiles <= expected:
                failures.append(f"{codec} counters invalid")
            old = (
                (previous_metrics or {})
                .get("authoritative_progress", {})
                .get(codec, {})
            )
            if previous_metrics is not None:
                if masks < expected and masks <= int(
                    old.get("receipt_backed_masks", -1)
                ):
                    failures.append(f"{codec} masks stagnated")
                if profiles < expected and profiles <= int(
                    old.get("canonical_profiles", -1)
                ):
                    failures.append(f"{codec} profiles stagnated")
            workers = metrics.get("profile_worker_affinities", {}).get(codec)
            if not isinstance(workers, dict) or not workers or _overlap(workers):
                failures.append(f"{codec} profile worker affinity invalid/overlap")
        except Exception:
            failures.append(f"{codec} telemetry invalid")
    services = metrics.get("feature_services")
    if not isinstance(services, list) or len(services) != 2:
        failures.append("segmentation service telemetry missing")
    else:
        for service in services:
            affinities = service.get("process_affinities")
            if (
                not isinstance(affinities, dict)
                or (
                    service.get("active") == "active"
                    and (int(service.get("pid", 0)) <= 0 or not affinities)
                )
                or (affinities and _overlap(affinities))
            ):
                failures.append(
                    f"segmentation cgroup affinity invalid/overlap: {service.get('name')}"
                )
            codec = "MQ" if "MQ" in str(service.get("name")) else "lossless"
            state = states.get(codec, {})
            current_masks = (
                metrics.get("authoritative_progress", {})
                .get(codec, {})
                .get("receipt_backed_masks")
            )
            if service.get("active") != "active" and current_masks != state.get(
                "expected_sites"
            ):
                failures.append(f"segmentation inactive before completion: {codec}")
    try:
        if compression.get("format_version") != "full-jump-compression-state-v2":
            failures.append("compression format invalid")
        if metrics.get("candidate_id") != compression.get(
            "candidate_id"
        ) or metrics.get("config_sha256") != compression.get("config_sha256"):
            failures.append("compression identity drift")
        if not 0 <= now_ts - float(compression["heartbeat_unix"]) <= 300:
            failures.append("compression heartbeat stale")
        errors = int(compression["cumulative_errors"])
        acknowledged = int(effective_previous.get("acknowledged_error_count", 0))
        if errors != acknowledged:
            failures.append("compression error acknowledgement mismatch")
        if previous_metrics is not None and compression.get("state") not in {
            "complete",
            "paused",
        }:
            old = (previous_metrics or {}).get("compression", {})
            if int(compression.get("processed", -1)) <= int(old.get("processed", -1)):
                failures.append("compression progress stagnated")
    except Exception:
        failures.append("compression telemetry invalid")
    if float(metrics.get("load1", 9999)) > 256:
        failures.append("host load guard")
    if int(metrics.get("memory_available_bytes", 0)) < 128 * 1024**3:
        failures.append("memory guard")
    if int(metrics.get("storage_available_bytes", 0)) < 8 * 1024**4:
        failures.append("storage guard")
    if float(metrics.get("io_pressure_avg10", 9999)) > 20:
        failures.append("I/O pressure guard")
    old_workers = int(effective_previous.get("desired_workers", INITIAL_WORKERS))
    windows = int(effective_previous.get("consecutive_healthy_windows", 0))
    acknowledged = int(effective_previous.get("acknowledged_error_count", 0))
    if failures:
        paused = True
        workers = INITIAL_WORKERS
        windows = 0
    else:
        paused = False
        windows += 1
        workers = old_workers
        if windows >= 2 and workers < MAX_WORKERS:
            workers = min(MAX_WORKERS, workers + 4)
            windows = 0
    return {
        "format_version": "full-jump-compression-control-v2",
        "candidate_id": metrics.get("candidate_id"),
        "config_sha256": metrics.get("config_sha256"),
        "paused": paused,
        "desired_workers": workers,
        "max_workers": MAX_WORKERS,
        "consecutive_healthy_windows": windows,
        "compression_cpus": list(COMPRESSION_CPUS),
        "acknowledged_error_count": acknowledged,
        "acknowledgement_source": ("explicit-cli" if acknowledged > 0 else "zero"),
        "reasons": failures or ["healthy; feature extraction retains priority"],
        "observed_at_unix": now_ts,
        "feature_processes_mutated": False,
    }


def run_governor(paths: GovernorPaths, dry_run: bool = True) -> dict[str, Any]:
    paths.validate()
    latest = paths.snapshots / "latest.json"
    error = None
    previous = None
    try:
        previous = (
            json.loads(paths.control.read_text()) if paths.control.is_file() else None
        )
        snap = json.loads(latest.read_text()) if latest.is_file() else None
        previous_metrics = snap.get("metrics") if isinstance(snap, dict) else None
        metrics = collect_metrics(paths)
        decision = decide(metrics, previous, previous_metrics)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        metrics = {"observed_at_unix": time.time(), "telemetry_error": error}
        decision = {
            "format_version": "full-jump-compression-control-v2",
            "candidate_id": paths.candidate_id,
            "config_sha256": paths.config_sha256,
            "paused": True,
            "desired_workers": INITIAL_WORKERS,
            "max_workers": MAX_WORKERS,
            "consecutive_healthy_windows": 0,
            "compression_cpus": list(COMPRESSION_CPUS),
            "acknowledged_error_count": 0,
            "acknowledgement_source": "zero",
            "reasons": [f"telemetry failure: {error}"],
            "observed_at_unix": metrics["observed_at_unix"],
            "feature_processes_mutated": False,
        }
    result = {
        "metrics": metrics,
        "decision": decision,
        "dry_run": dry_run,
        "telemetry_error": error,
    }
    if not dry_run:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        paths.validate()
        paths.snapshots.mkdir(parents=True, exist_ok=True)
        paths.validate()
        stamped = paths.snapshots / f"{stamp}.json"
        assert_no_symlinks(stamped, paths.state_root)
        if stamped.is_symlink():
            raise ValueError("governor snapshot symlink rejected")
        atomic_json(
            stamped,
            {"metrics": metrics, "decision": decision},
        )
        atomic_json(latest, {"metrics": metrics, "decision": decision})
        atomic_json(paths.control, decision)
    return result
