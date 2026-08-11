#!/usr/bin/env python3
"""Upload JUMP-Lite per-site Parquets to CPG's embedding hierarchy.

The local Parquets use flat, self-describing filenames. This uploader maps each
file to the CPG model/source/batch/plate/well-site hierarchy without creating
millions of local hard links. Uploads are resumable through deterministic,
per-variant checkpoints and temporary S3 Access Grant credentials are refreshed
before expiration.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from validate_release import (
    EXPECTED_PROFILE_VARIANTS,
    EXPECTED_SITE_COUNT,
    DEFAULT_PROFILE_ROOT,
)

ACCOUNT_ID = "309624411020"
REGION = "us-east-1"
BUCKET = "staging-cellpainting-gallery"
GRANT_TARGET = "s3://staging-cellpainting-gallery/cpg0016-jump/*"
DESTINATION_ROOT = "cpg0016-jump/source_all/workspace_dl/embeddings"
RELEASE_BATCH = "2026_jump_lite_v1.0"
CANONICAL_DIGEST = "399e703bc924a19f7c3827db3c711373306e3d943d2f12cf56d0a368f5d13961"
MODEL_NAMES = {
    "dinov2": "dinov2",
    "dinov2_random": "dinov2_random",
    "morphem": "morphem",
    "openphenom_confusing": "openphenom",
    "subcell": "subcell",
    "subcell__clip01": "subcell_clip01",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, default=DEFAULT_PROFILE_ROOT)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/work/datasets/jump_lite/cpg_upload_state/v1.0/profiles"),
    )
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--model", choices=sorted(EXPECTED_PROFILE_VARIANTS))
    parser.add_argument("--codec")
    parser.add_argument("--max-files", type=int, help="limit each variant for testing")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform uploads; otherwise only validate and print the key mapping",
    )
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def validate_report(path: Path | None) -> None:
    if path is None:
        command = [sys.executable, str(Path(__file__).with_name("validate_release.py"))]
        subprocess.run(command, check=True)
        return

    report = json.loads(path.read_text())
    errors = report.get("errors")
    if errors != []:
        raise RuntimeError(f"validation report is not successful: errors={errors!r}")
    if report.get("canonical_site_key_sha256") != CANONICAL_DIGEST:
        raise RuntimeError("validation report has the wrong canonical site-key digest")
    validated_at = report.get("validated_at")
    if not validated_at:
        raise RuntimeError("validation report does not contain validated_at")


def variants(args: argparse.Namespace) -> list[tuple[str, str]]:
    result = []
    for model, codecs in sorted(EXPECTED_PROFILE_VARIANTS.items()):
        if args.model and model != args.model:
            continue
        for codec in sorted(codecs):
            if args.codec and codec != args.codec:
                continue
            result.append((model, codec))
    if not result:
        raise RuntimeError("no profile variants matched the requested filters")
    return result


def feature_set_name(model: str, codec: str) -> str:
    public_model = MODEL_NAMES[model]
    public_codec = codec.removesuffix(".zarr")
    return f"{public_model}-{public_codec}"


def destination_prefix(model: str, codec: str) -> str:
    return f"{DESTINATION_ROOT}/{feature_set_name(model, codec)}/{RELEASE_BATCH}"


def destination_key(model: str, codec: str, filename: str) -> str:
    if not filename.endswith(".parquet"):
        raise ValueError(f"not a Parquet filename: {filename}")
    fields = filename.removesuffix(".parquet").split("__")
    if len(fields) != 5:
        raise ValueError(f"malformed site filename: {filename}")
    source, batch, plate, well, site = fields
    return (
        f"{destination_prefix(model, codec)}/"
        f"{source}/{batch}/{plate}/{well}-{site}/embedding.parquet"
    )


class RefreshingS3Client:
    """Vend and proactively refresh a thread-safe S3 client."""

    def __init__(self, workers: int) -> None:
        self.workers = workers
        self._lock = threading.Lock()
        self._client: Any = None
        self._expiration = 0.0

    @staticmethod
    def _base_credentials() -> tuple[str, str]:
        key_path = Path(os.environ.get("CPG_KEY_ID_FILE", Path.home() / ".cpg_key_id"))
        secret_path = Path(
            os.environ.get("CPG_SECRET_FILE", Path.home() / ".cpg_access_key")
        )
        return key_path.read_text().strip(), secret_path.read_text().strip()

    def _refresh(self) -> None:
        access_key, secret_key = self._base_credentials()
        control = boto3.client(
            "s3control",
            region_name=REGION,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(retries={"mode": "adaptive", "max_attempts": 10}),
        )
        response = control.get_data_access(
            AccountId=ACCOUNT_ID,
            Target=GRANT_TARGET,
            Permission="READWRITE",
            DurationSeconds=43_200,
        )
        credentials = response["Credentials"]
        expiration = credentials["Expiration"]
        self._client = boto3.client(
            "s3",
            region_name=REGION,
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
            config=Config(
                max_pool_connections=self.workers + 16,
                retries={"mode": "adaptive", "max_attempts": 12},
                connect_timeout=20,
                read_timeout=120,
            ),
        )
        self._expiration = expiration.timestamp()
        print(
            f"[{datetime.now(timezone.utc).isoformat()}] "
            f"refreshed temporary profile-upload credentials; expires={expiration.isoformat()}",
            flush=True,
        )

    def client(self, *, force_refresh: bool = False) -> Any:
        # Refresh with a 15-minute safety margin.
        if (
            not force_refresh
            and self._client is not None
            and time.time() < self._expiration - 900
        ):
            return self._client
        with self._lock:
            if (
                force_refresh
                or self._client is None
                or time.time() >= self._expiration - 900
            ):
                self._refresh()
            return self._client


def upload_file(
    manager: RefreshingS3Client,
    source: Path,
    key: str,
    *,
    max_attempts: int = 20,
) -> int:
    size = source.stat().st_size
    for attempt in range(1, max_attempts + 1):
        try:
            with source.open("rb") as stream:
                manager.client().put_object(
                    Bucket=BUCKET,
                    Key=key,
                    Body=stream,
                    ContentLength=size,
                    ContentType="application/vnd.apache.parquet",
                    ExpectedBucketOwner=ACCOUNT_ID,
                )
            return size
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "unknown")
            if code in {
                "AccessDenied",
                "ExpiredToken",
                "InvalidAccessKeyId",
                "RequestExpired",
                "TokenRefreshRequired",
            }:
                manager.client(force_refresh=True)
            if attempt == max_attempts:
                raise
            time.sleep(min(60.0, 0.5 * (2 ** min(attempt, 7))))
        except (BotoCoreError, OSError):
            if attempt == max_attempts:
                raise
            time.sleep(min(60.0, 0.5 * (2 ** min(attempt, 7))))
    raise AssertionError("unreachable")


def upload_variant(
    args: argparse.Namespace,
    manager: RefreshingS3Client,
    model: str,
    codec: str,
) -> None:
    source_dir = args.profile_root / model / codec / "profiles"
    if not source_dir.is_dir():
        raise RuntimeError(f"missing profile directory: {source_dir}")

    entries = sorted(
        (entry for entry in os.scandir(source_dir) if entry.is_file() and entry.name.endswith(".parquet")),
        key=lambda entry: entry.name,
    )
    if len(entries) != EXPECTED_SITE_COUNT:
        raise RuntimeError(
            f"{model}/{codec} contains {len(entries):,} Parquets; "
            f"expected {EXPECTED_SITE_COUNT:,}"
        )
    if args.max_files is not None:
        entries = entries[: args.max_files]

    checkpoint_path = args.checkpoint_root / model / f"{codec}.json"
    start = 0
    uploaded_bytes = 0
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text())
        expected_destination = destination_prefix(model, codec)
        recorded_destination = checkpoint.get("destination_prefix")
        if recorded_destination != expected_destination:
            raise RuntimeError(
                "checkpoint destination mismatch; use a new checkpoint for the "
                f"CPG-compliant path: {checkpoint_path} "
                f"(recorded={recorded_destination!r}, expected={expected_destination!r})"
            )
        start = int(checkpoint["next_index"])
        uploaded_bytes = int(checkpoint.get("uploaded_bytes", 0))
        if start > len(entries):
            raise RuntimeError(f"checkpoint exceeds file count: {checkpoint_path}")
        if start and checkpoint.get("last_filename") != entries[start - 1].name:
            raise RuntimeError(f"checkpoint ordering mismatch: {checkpoint_path}")

    print(
        f"[{datetime.now(timezone.utc).isoformat()}] variant={model}/{codec} "
        f"files={len(entries):,} resume_index={start:,}",
        flush=True,
    )
    if start == len(entries):
        print(f"variant already complete: {model}/{codec}", flush=True)
        return

    if not args.apply:
        for entry in entries[start : min(start + 5, len(entries))]:
            print(f"DRY-RUN {entry.path} -> s3://{BUCKET}/{destination_key(model, codec, entry.name)}")
        return

    started = time.monotonic()
    completed = start
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while completed < len(entries):
            end = min(completed + args.batch_size, len(entries))
            batch = entries[completed:end]
            results = executor.map(
                lambda entry: upload_file(
                    manager,
                    Path(entry.path),
                    destination_key(model, codec, entry.name),
                ),
                batch,
            )
            for entry, size in zip(batch, results, strict=True):
                completed += 1
                uploaded_bytes += size
                if completed % 1_000 == 0 or completed == len(entries):
                    elapsed = max(time.monotonic() - started, 0.001)
                    rate = (completed - start) / elapsed
                    payload = {
                        "model": model,
                        "codec": codec,
                        "destination_prefix": destination_prefix(model, codec),
                        "next_index": completed,
                        "file_count": len(entries),
                        "last_filename": entry.name,
                        "uploaded_bytes": uploaded_bytes,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "complete": completed == len(entries),
                    }
                    atomic_write_json(checkpoint_path, payload)
                    if completed % 10_000 == 0 or completed == len(entries):
                        print(
                            f"[{payload['updated_at']}] variant={model}/{codec} "
                            f"uploaded={completed:,}/{len(entries):,} rate={rate:,.1f} files/s",
                            flush=True,
                        )


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.batch_size < 1:
        raise ValueError("workers and batch-size must be positive")
    validate_report(args.validation_report)
    selected = variants(args)

    # Validate mappings before any write.
    for model, codec in selected:
        profile_dir = args.profile_root / model / codec / "profiles"
        sample = next(
            (entry.name for entry in os.scandir(profile_dir) if entry.name.endswith(".parquet")),
            None,
        )
        if sample is None:
            raise RuntimeError(f"no Parquets found in {profile_dir}")
        destination_key(model, codec, sample)

    manager = RefreshingS3Client(args.workers)
    if args.apply:
        manager.client()
    for model, codec in selected:
        upload_variant(args, manager, model, codec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
