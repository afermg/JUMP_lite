#!/usr/bin/env python3
"""Build a complete ordered full-JUMP manifest from explicitly staged inputs.

This command is intentionally offline.  It reuses a content-pinned preliminary
wide Parquet, removes approved exclusions, and adds six separately staged gray
plate load-data CSVs.  It never downloads inputs and never freezes release
identity; the separate frozen inventory audit owns that decision.
"""

from __future__ import annotations

import argparse
import csv
import gc
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Iterable

# Override every relevant native pool before the first PyArrow/Numpy/native import.
THREAD_ENV = (
    "ARROW_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "RAYON_NUM_THREADS",
    "TBB_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
for _name in THREAD_ENV:
    os.environ[_name] = "1"

import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.csv as pacsv  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

pa.set_cpu_count(1)
pa.set_io_thread_count(1)

from jump_full_compression.inventory import _validate_frozen_policy  # noqa: E402

UPSTREAM_COMMIT = "016e865fa0691244e0860943e41c7d6a88ed2580"
PLATE_CSV_SHA256 = "541ada1f64816166509a4e2328316d2a6662ba67e257b7ae134cbec9d7079319"
PRELIMINARY_SHA256 = "cb31bb16bee36d126e45a9588d80b9e8cef5e16f468a9b7b2738f0b618152a44"
PRELIMINARY_BYTES = 145_201_434
UPSTREAM_PLATE_COUNT = 2_525
IDENTITY = (
    "Metadata_Source",
    "Metadata_Batch",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_Site",
)
CHANNELS = ("AGP", "DNA", "ER", "Mito", "RNA")
CORE_COLUMNS = IDENTITY + tuple(f"URL_Orig{x}" for x in CHANNELS)
DAMAGED_KEYS = {
    ("source_7", "20210727_Run3", "CP3-SC1-18", "I22", 2),
    ("source_7", "20210727_Run3", "CP3-SC1-18", "I22", 3),
}
GRAY_RECEIPT_FORMAT = "full-jump-gray-input-receipt-v1"
REPORT_FORMAT = "full-jump-production-manifest-build-v1"
MINIMUM_MEMORY_HEADROOM = 64 * 1024**3
MEMORY_MULTIPLIER = 6


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: Path, label: str) -> None:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RuntimeError(
            f"{label} must be an absolute regular non-symlink file: {path}"
        )


def _stable_stat(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def capture_regular_file(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    """Read and bind exactly one stable regular-file snapshot without following links."""
    if not path.is_absolute():
        raise RuntimeError(f"{label} path must be absolute: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"cannot safely open {label}: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} is not a regular file: {path}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
            digest.update(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = os.stat(path, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError(f"{label} path changed during capture: {path}") from error
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or _stable_stat(before) != _stable_stat(after)
        or _stable_stat(after) != _stable_stat(current)
    ):
        raise RuntimeError(f"{label} changed during capture: {path}")
    payload = b"".join(chunks)
    if len(payload) != before.st_size:
        raise RuntimeError(f"{label} byte-count changed during capture: {path}")
    return payload, {"bytes": len(payload), "sha256": digest.hexdigest()}


def binding(path: Path) -> dict[str, Any]:
    regular_file(path, "artifact")
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except Exception as error:
        raise RuntimeError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object")
    return value


def pinned_plate_universe(checkout: Path) -> set[tuple[str, str, str]]:
    if not checkout.is_absolute() or not checkout.is_dir() or checkout.is_symlink():
        raise RuntimeError(
            "datasets checkout must be an absolute non-symlink directory"
        )
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"datasets checkout commit drift: {commit}")
    plate_csv = checkout / "metadata/plate.csv.gz"
    plate_bytes, plate_binding = capture_regular_file(
        plate_csv, "pinned plate metadata"
    )
    if plate_binding["sha256"] != PLATE_CSV_SHA256:
        raise RuntimeError("pinned plate metadata SHA-256 drift")
    with gzip.open(io.BytesIO(plate_bytes), "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"Metadata_Source", "Metadata_Batch", "Metadata_Plate"}
    if len(rows) != UPSTREAM_PLATE_COUNT or not rows or not required <= set(rows[0]):
        raise RuntimeError("pinned plate metadata schema/count drift")
    universe = {
        (row["Metadata_Source"], row["Metadata_Batch"], row["Metadata_Plate"])
        for row in rows
    }
    if len(universe) != len(rows):
        raise RuntimeError("pinned plate metadata contains duplicate identities")
    return universe


def _require_core_schema(table: pa.Table, label: str, *, allow_extra: bool) -> pa.Table:
    names = set(table.column_names)
    missing = set(CORE_COLUMNS) - names
    unsupported_urls = sorted(
        name
        for name in names
        if name.startswith("URL_Orig") and name not in CORE_COLUMNS
    )
    extras = names - set(CORE_COLUMNS)
    if missing or unsupported_urls or (extras and not allow_extra):
        raise RuntimeError(
            f"{label} schema drift: missing={sorted(missing)} "
            f"unsupported_urls={unsupported_urls} extras={sorted(extras)}"
        )
    table = table.select(CORE_COLUMNS)
    if not pa.types.is_integer(table.schema.field("Metadata_Site").type):
        raise RuntimeError(f"{label} Metadata_Site must be integer")
    for name in CORE_COLUMNS:
        if name == "Metadata_Site":
            continue
        if not pa.types.is_string(table.schema.field(name).type):
            raise RuntimeError(f"{label} {name} must be string")
    return table.set_column(
        table.schema.get_field_index("Metadata_Site"),
        "Metadata_Site",
        pc.cast(table["Metadata_Site"], pa.int64()),
    )


def _row_batches(table: pa.Table) -> Iterable[tuple[Any, ...]]:
    for batch in table.select(IDENTITY).to_batches(max_chunksize=65_536):
        values = batch.to_pydict()
        for index in range(batch.num_rows):
            yield tuple(values[name][index] for name in IDENTITY)


def validate_rows(
    table: pa.Table, label: str, *, allow_source_15_missing_rna: bool = False
) -> tuple[set[tuple[str, str, str]], dict[str, int]]:
    for name in CORE_COLUMNS:
        invalid = pc.is_null(table[name])
        if name != "Metadata_Site":
            invalid = pc.or_(
                invalid,
                pc.fill_null(pc.equal(pc.utf8_trim_whitespace(table[name]), ""), True),
            )
        if allow_source_15_missing_rna and name == "URL_OrigRNA":
            invalid = pc.and_(
                invalid, pc.invert(pc.equal(table["Metadata_Source"], "source_15"))
            )
        if pc.any(invalid).as_py():
            raise RuntimeError(
                f"{label} contains missing/empty required values in {name}"
            )
    plates: set[tuple[str, str, str]] = set()
    counts: dict[str, int] = {}
    for key in _row_batches(table):
        source, batch, plate, _well, site = key
        if isinstance(site, bool):
            raise RuntimeError(f"{label} contains boolean site")
        plates.add((source, batch, plate))
        counts[source] = counts.get(source, 0) + 1
    return plates, dict(sorted(counts.items()))


def parquet_uncompressed_bytes(payload: bytes) -> int:
    parquet = pq.ParquetFile(pa.BufferReader(payload))
    metadata = parquet.metadata
    return sum(
        metadata.row_group(row_group).column(column).total_uncompressed_size
        for row_group in range(metadata.num_row_groups)
        for column in range(metadata.num_columns)
    )


def _host_available_memory() -> int:
    try:
        fields = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in Path("/proc/meminfo").read_text().splitlines()
            if ":" in line
        }
        amount, unit = fields["MemAvailable"].split()
        if unit != "kB":
            raise ValueError(unit)
        return int(amount) * 1024
    except Exception as error:
        raise RuntimeError("cannot determine Linux MemAvailable") from error


def _cgroup_available_memory() -> int | None:
    try:
        relative = next(
            line.split("::", 1)[1]
            for line in Path("/proc/self/cgroup").read_text().splitlines()
            if line.startswith("0::")
        )
        root = Path("/sys/fs/cgroup") / relative.lstrip("/")
        maximum = (root / "memory.max").read_text().strip()
        current = int((root / "memory.current").read_text().strip())
        if maximum == "max":
            return None
        return max(0, int(maximum) - current)
    except (FileNotFoundError, StopIteration):
        return None
    except Exception as error:
        raise RuntimeError("cannot determine cgroup-v2 available memory") from error


def memory_preflight(
    payload: bytes, test_available: int | None = None
) -> dict[str, Any]:
    uncompressed = parquet_uncompressed_bytes(payload)
    required = max(MINIMUM_MEMORY_HEADROOM, MEMORY_MULTIPLIER * uncompressed)
    if test_available is None:
        host_available = _host_available_memory()
        cgroup_available = _cgroup_available_memory()
        available = min(
            value for value in (host_available, cgroup_available) if value is not None
        )
        source = (
            "min(linux-memavailable,cgroup-v2)"
            if cgroup_available is not None
            else "linux-memavailable"
        )
    else:
        host_available = None
        cgroup_available = None
        available = test_available
        source = "injected-test-only"
    result = {
        "parquet_uncompressed_column_chunk_bytes": uncompressed,
        "multiplier": MEMORY_MULTIPLIER,
        "minimum_headroom_bytes": MINIMUM_MEMORY_HEADROOM,
        "required_headroom_bytes": required,
        "host_mem_available_bytes": host_available,
        "cgroup_available_bytes": cgroup_available,
        "effective_available_bytes": available,
        "effective_available_source": source,
    }
    if available < required:
        raise RuntimeError(
            f"insufficient memory headroom: available={available} required={required}"
        )
    return result


def load_gray_inputs(
    receipt_path: Path, expected_gray: set[tuple[str, str, str]]
) -> tuple[pa.Table, list[dict[str, Any]], dict[str, Any]]:
    receipt_bytes, receipt_binding = capture_regular_file(
        receipt_path, "gray input receipt"
    )
    receipt = json_bytes(receipt_bytes, "gray input receipt")
    if (
        set(receipt) != {"format_version", "entries"}
        or receipt.get("format_version") != GRAY_RECEIPT_FORMAT
    ):
        raise RuntimeError("gray input receipt format drift")
    entries = receipt.get("entries")
    if not isinstance(entries, list) or len(entries) != 6:
        raise RuntimeError("gray input receipt must contain exactly six entries")
    identities: set[tuple[str, str, str]] = set()
    tables: list[pa.Table] = []
    observations: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "source",
            "batch",
            "plate",
            "public_uri",
            "path",
            "bytes",
            "sha256",
        }:
            raise RuntimeError("gray input receipt entry schema drift")
        identity = (entry["source"], entry["batch"], entry["plate"])
        if identity in identities:
            raise RuntimeError("duplicate gray receipt identity")
        identities.add(identity)
        path = Path(entry["path"])
        csv_bytes, observed = capture_regular_file(path, "staged gray CSV")
        if observed != {"bytes": entry["bytes"], "sha256": entry["sha256"]}:
            raise RuntimeError(f"gray input receipt binding drift for {identity}")
        expected_uri = (
            "s3://cellpainting-gallery/cpg0016-jump/"
            f"{entry['source']}/workspace/load_data_csv/{entry['batch']}/"
            f"{entry['plate']}/load_data_with_illum.csv"
        )
        if entry["public_uri"] != expected_uri:
            raise RuntimeError(f"gray public URI drift for {identity}")
        table = pacsv.read_csv(
            pa.BufferReader(csv_bytes),
            read_options=pacsv.ReadOptions(use_threads=False),
            convert_options=pacsv.ConvertOptions(
                column_types={"Metadata_Site": pa.int64()}
            ),
        )
        table = _require_core_schema(table, f"gray CSV {identity}", allow_extra=True)
        plates, _ = validate_rows(table, f"gray CSV {identity}")
        if plates != {identity}:
            raise RuntimeError(
                f"gray CSV metadata identity drift for {identity}: {plates}"
            )
        tables.append(table)
        observations.append({**entry, "path": None})
    if identities != expected_gray:
        raise RuntimeError("gray receipt identities differ from canonical gray ledger")
    observations.sort(key=lambda item: (item["source"], item["batch"], item["plate"]))
    return pa.concat_tables(tables), observations, receipt_binding


def _filter_preliminary(
    table: pa.Table,
) -> tuple[pa.Table, dict[str, int]]:
    source_15 = pc.equal(table["Metadata_Source"], "source_15")
    damaged = None
    for source, batch, plate, well, site in sorted(DAMAGED_KEYS):
        match = pc.and_(
            pc.and_(
                pc.and_(
                    pc.equal(table["Metadata_Source"], source),
                    pc.equal(table["Metadata_Batch"], batch),
                ),
                pc.and_(
                    pc.equal(table["Metadata_Plate"], plate),
                    pc.equal(table["Metadata_Well"], well),
                ),
            ),
            pc.equal(table["Metadata_Site"], site),
        )
        damaged = match if damaged is None else pc.or_(damaged, match)
    keep = pc.invert(pc.or_(source_15, damaged))
    return table.filter(keep), {
        "source_15_rows": pc.sum(pc.cast(source_15, pa.int64())).as_py(),
        "known_damaged_site_rows": pc.sum(pc.cast(damaged, pa.int64())).as_py(),
    }


def validate_sorted_unique(table: pa.Table) -> None:
    previous: tuple[Any, ...] | None = None
    for key in _row_batches(table):
        if previous is not None and key <= previous:
            reason = "duplicate" if key == previous else "decreasing"
            raise RuntimeError(
                f"manifest identity order is not strict ({reason}): {key}"
            )
        previous = key


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    explicit_paths = (
        args.preliminary,
        args.datasets_checkout,
        args.exclusion_policy,
        args.damaged_objects,
        args.damaged_sites,
        args.qc_plates,
        args.gray_receipt,
        args.output,
        args.report,
    )
    if any(not path.is_absolute() for path in explicit_paths):
        raise RuntimeError("all input and output paths must be absolute")
    output = args.output.resolve()
    report_path = args.report.resolve()
    if output.exists() or report_path.exists():
        raise FileExistsError("output and report must not already exist")
    if output.parent != report_path.parent:
        raise RuntimeError("output and report must share an existing parent directory")
    if not output.parent.is_dir() or output.parent.is_symlink():
        raise RuntimeError("output parent must be an existing non-symlink directory")
    temporary = output.with_name(f".{output.name}.building-{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(f"temporary output already exists: {temporary}")

    preliminary = args.preliminary
    preliminary_bytes, prelim_binding = capture_regular_file(
        preliminary, "preliminary inventory"
    )
    configured_preliminary = {
        "bytes": args.preliminary_bytes,
        "sha256": args.preliminary_sha256,
    }
    if configured_preliminary != {
        "bytes": PRELIMINARY_BYTES,
        "sha256": PRELIMINARY_SHA256,
    }:
        raise RuntimeError("configured preliminary identity is not the canonical input")
    if prelim_binding != configured_preliminary:
        raise RuntimeError("preliminary inventory binding drift")
    memory = memory_preflight(
        preliminary_bytes, getattr(args, "_test_memory_available_bytes", None)
    )
    universe = pinned_plate_universe(args.datasets_checkout)
    for path, label in (
        (args.exclusion_policy, "exclusion policy"),
        (args.damaged_objects, "damaged-object ledger"),
        (args.damaged_sites, "damaged-site ledger"),
        (args.qc_plates, "QC plate ledger"),
    ):
        regular_file(path, label)
    policy = _validate_frozen_policy(
        args.exclusion_policy,
        args.damaged_objects,
        args.damaged_sites,
        args.qc_plates,
    )
    if policy["red_gray_action"] != "exclude_red_include_gray":
        raise RuntimeError("production policy must exclude red and include gray")
    red = policy.pop("red_plate_keys")
    gray = policy.pop("gray_plate_keys")
    expected_preliminary = universe - red - gray

    table = pq.read_table(pa.BufferReader(preliminary_bytes), use_threads=False)
    del preliminary_bytes
    table = _require_core_schema(table, "preliminary inventory", allow_extra=False)
    preliminary_plates, preliminary_counts = validate_rows(
        table, "preliminary inventory", allow_source_15_missing_rna=True
    )
    if preliminary_plates != expected_preliminary:
        missing = sorted(expected_preliminary - preliminary_plates)[:20]
        extra = sorted(preliminary_plates - expected_preliminary)[:20]
        raise RuntimeError(
            f"preliminary plate-universe drift: missing={missing} extra={extra}"
        )
    filtered, exclusions = _filter_preliminary(table)
    if exclusions["known_damaged_site_rows"] != 2:
        raise RuntimeError(
            "preliminary inventory does not contain exactly both damaged sites"
        )
    gray_table, gray_entries, gray_receipt_binding = load_gray_inputs(
        args.gray_receipt, gray
    )
    combined = pa.concat_tables([filtered, gray_table])
    final_plates, _ = validate_rows(combined, "combined manifest")
    expected_final_plates = {
        plate for plate in universe - red if plate[0] != "source_15"
    }
    if final_plates != expected_final_plates:
        raise RuntimeError("combined final plate universe drift")
    if pc.any(pc.equal(combined["Metadata_Source"], "source_15")).as_py():
        raise RuntimeError("combined manifest retains source_15")
    damaged_mask = None
    for source, batch, plate, well, site in sorted(DAMAGED_KEYS):
        match = pc.and_(
            pc.and_(
                pc.and_(
                    pc.equal(combined["Metadata_Source"], source),
                    pc.equal(combined["Metadata_Batch"], batch),
                ),
                pc.and_(
                    pc.equal(combined["Metadata_Plate"], plate),
                    pc.equal(combined["Metadata_Well"], well),
                ),
            ),
            pc.equal(combined["Metadata_Site"], site),
        )
        damaged_mask = match if damaged_mask is None else pc.or_(damaged_mask, match)
    if damaged_mask is not None and pc.any(damaged_mask).as_py():
        raise RuntimeError("combined manifest retains a known damaged site")
    order = pc.sort_indices(
        combined, sort_keys=[(name, "ascending") for name in IDENTITY]
    )
    ordered = pc.take(combined, order)
    validate_sorted_unique(ordered)
    _, final_counts = validate_rows(ordered, "ordered manifest")
    preliminary_rows = table.num_rows
    gray_rows = gray_table.num_rows
    final_rows = ordered.num_rows
    output_columns = list(ordered.column_names)
    output_schema = str(ordered.schema)

    try:
        pq.write_table(
            ordered,
            temporary,
            compression="zstd",
            use_dictionary=True,
            row_group_size=65_536,
            write_statistics=True,
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        # Drop every full-table/sort reference before independently rereading
        # the just-written output for its publication checks.
        del table, filtered, gray_table, combined, order, ordered
        gc.collect()
        observed = pq.read_table(temporary, use_threads=False)
        observed = _require_core_schema(observed, "written manifest", allow_extra=False)
        validate_sorted_unique(observed)
        if observed.num_rows != final_rows:
            raise RuntimeError("written manifest row-count drift")
        del observed
        gc.collect()
        output_binding = binding(temporary.resolve())
        script = Path(__file__).resolve()
        report = {
            "format_version": REPORT_FORMAT,
            "build_success": True,
            "release_identity_frozen": False,
            "policy_action": "exclude_red_include_gray",
            "inputs": {
                "preliminary": prelim_binding,
                "datasets": {
                    "commit": UPSTREAM_COMMIT,
                    "plate_csv_sha256": PLATE_CSV_SHA256,
                    "plate_count": len(universe),
                },
                "exclusion_policy": policy["policy"],
                "damaged_objects": policy["damaged_objects"],
                "damaged_sites": policy["damaged_sites"],
                "qc_plates": policy["qc_plates"],
                "gray_receipt": gray_receipt_binding,
                "gray_entries": gray_entries,
            },
            "counts": {
                "preliminary_rows": preliminary_rows,
                "preliminary_source_counts": preliminary_counts,
                "excluded": exclusions,
                "gray_rows_added": gray_rows,
                "final_rows": final_rows,
                "final_source_counts": final_counts,
                "red_plate_count": len(red),
                "gray_plate_count": len(gray),
                "final_plate_count": len(final_plates),
            },
            "output": {
                **output_binding,
                "rows": final_rows,
                "columns": output_columns,
                "schema": output_schema,
                "strict_identity_order": True,
                "unique_identities": True,
            },
            "memory_preflight": memory,
            "software": {
                "python": sys.version,
                "pyarrow": pa.__version__,
                "script_sha256": sha256_file(script),
                "thread_counts": {
                    "pyarrow_cpu": pa.cpu_count(),
                    "pyarrow_io": pa.io_thread_count(),
                },
                "thread_env": {name: os.environ[name] for name in THREAD_ENV},
            },
            "next_step": "run the separate frozen inventory audit; this report does not freeze identity",
        }
        report["build_digest"] = hashlib.sha256(
            json.dumps(report, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        report_temporary = report_path.with_name(
            f".{report_path.name}.building-{os.getpid()}"
        )
        with report_temporary.open("x") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        output_published = False
        try:
            # Hard-link publication is same-filesystem, atomic, and refuses an
            # output that appeared after preflight instead of overwriting it.
            os.link(temporary, output)
            output_published = True
            os.link(report_temporary, report_path)
        except Exception:
            if output_published and not report_path.exists():
                output.unlink(missing_ok=True)
            raise
        temporary.unlink()
        report_temporary.unlink()
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return report
    finally:
        temporary.unlink(missing_ok=True)
        report_path.with_name(f".{report_path.name}.building-{os.getpid()}").unlink(
            missing_ok=True
        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--preliminary", type=Path, required=True)
    result.add_argument("--preliminary-sha256", required=True)
    result.add_argument("--preliminary-bytes", type=int, required=True)
    result.add_argument("--datasets-checkout", type=Path, required=True)
    result.add_argument("--exclusion-policy", type=Path, required=True)
    result.add_argument("--damaged-objects", type=Path, required=True)
    result.add_argument("--damaged-sites", type=Path, required=True)
    result.add_argument("--qc-plates", type=Path, required=True)
    result.add_argument("--gray-receipt", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--report", type=Path, required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    print(json.dumps(build_manifest(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
