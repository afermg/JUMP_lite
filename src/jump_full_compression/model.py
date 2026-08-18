"""Identity and configuration primitives for candidate full-JUMP compression."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

CHANNELS = ("AGP", "DNA", "ER", "Mito", "RNA")
SOURCE_15_CHANNELS = ("AGP", "DNA", "ER", "Mito")
CODECS = {
    "jpegxl_lossy_hq": {"distance": 1.0, "lossless": False, "numthreads": 1},
    "jpegxl_lossy_mq": {"distance": 3.0, "lossless": False, "numthreads": 1},
}
IDENTITY_COLUMNS = (
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Site",
)
LIVE_CANDIDATE_PARENT = Path(
    "/work/datasets/jump_lite/images/compressed/.jump_full_candidate"
)
LIVE_STATE_PARENT = Path("/work/datasets/jump_lite/full_jump_compression_state/v1.0")
COMPRESSION_CPUS = tuple(range(64, 81))
INITIAL_WORKERS = 4
MAX_WORKERS = 16
MAX_CANDIDATE_ROWS = 256
MAX_CUMULATIVE_ERRORS = 1_000_000
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TBB_NUM_THREADS",
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.partial.{os.getpid()}")
    data = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    with tmp.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def channels_for_source(source: str) -> tuple[str, ...]:
    return SOURCE_15_CHANNELS if source == "source_15" else CHANNELS


def site_key(row: Mapping[str, Any]) -> str:
    values = [str(row[column]) for column in IDENTITY_COLUMNS]
    if any(
        not value
        or value in {"None", "nan"}
        or "__" in value
        or "/" in value
        or "\\" in value
        for value in values
    ):
        raise ValueError(f"malformed identity: {values}")
    return "__".join(values)


def assert_no_symlinks(path: Path, boundary: Path) -> None:
    """Reject symlinks in every existing component between boundary and path."""
    absolute = path.absolute()
    limit = boundary.absolute()
    if absolute != limit and limit not in absolute.parents:
        raise ValueError(f"path escapes boundary: {absolute} not under {limit}")
    current = limit
    for part in absolute.relative_to(limit).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"symlink candidate component rejected: {current}")


@dataclass(frozen=True)
class CandidateConfig:
    candidate_id: str
    manifest: Path
    audit_report: Path
    output_root: Path
    state_root: Path
    inventory_digest: str
    manifest_sha256: str
    manifest_size: int
    batch_size: int = 4
    max_workers: int = MAX_WORKERS
    test_mode: bool = False

    def validate(self) -> None:
        if not self.candidate_id or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in self.candidate_id
        ):
            raise ValueError(
                "candidate-id must use lowercase letters, digits, '-' or '_'"
            )
        for value in (self.inventory_digest, self.manifest_sha256):
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("identity digests must be lowercase SHA-256")
        if (
            self.manifest_size < 1
            or self.batch_size < 1
            or self.max_workers != MAX_WORKERS
        ):
            raise ValueError("invalid manifest size/batch/max-workers contract")
        if not self.manifest.is_file() or self.manifest.is_symlink():
            raise ValueError("manifest must be a regular non-symlink file")
        if self.test_mode:
            if not self.candidate_id.startswith("test-"):
                raise ValueError("test-mode candidate ids must start with test-")
            output_boundary, state_boundary = (
                self.output_root.parent,
                self.state_root.parent,
            )
        else:
            expected_output = LIVE_CANDIDATE_PARENT / self.candidate_id
            expected_state = LIVE_STATE_PARENT / self.candidate_id
            if (
                self.output_root.absolute() != expected_output.absolute()
                or self.state_root.absolute() != expected_state.absolute()
            ):
                raise ValueError(
                    "live candidate output/state path must be literal expected paths"
                )
            if LIVE_CANDIDATE_PARENT.is_symlink() or LIVE_STATE_PARENT.is_symlink():
                raise ValueError("live candidate parents cannot be symlinks")
            if (
                LIVE_CANDIDATE_PARENT.exists()
                and LIVE_CANDIDATE_PARENT.resolve() != LIVE_CANDIDATE_PARENT.absolute()
            ):
                raise ValueError("resolved live output parent drift")
            if (
                LIVE_STATE_PARENT.exists()
                and LIVE_STATE_PARENT.resolve() != LIVE_STATE_PARENT.absolute()
            ):
                raise ValueError("resolved live state parent drift")
            output_boundary, state_boundary = LIVE_CANDIDATE_PARENT, LIVE_STATE_PARENT
        assert_no_symlinks(self.output_root, output_boundary)
        assert_no_symlinks(self.state_root, state_boundary)
        if self.output_root.name == "jump_full" or "cellpainting-gallery" in str(
            self.output_root
        ):
            raise ValueError("candidate cannot target final/CPG paths")

    @property
    def digest(self) -> str:
        payload = asdict(self)
        for key in ("manifest", "audit_report", "output_root", "state_root"):
            payload[key] = str(Path(payload[key]).absolute())
        return digest_json(payload)
