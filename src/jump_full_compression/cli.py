"""CLI for bounded full-JUMP candidate compression."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from .governor import GovernorPaths, run_governor
from .inventory import audit_inventory
from .model import CandidateConfig, assert_runtime_task_ceiling
from .pipeline import (
    acknowledge_errors,
    bootstrap_candidate,
    install_signal_handlers,
    run_candidate,
    validate_adoption_seam,
)

DEFAULT_FEATURE = Path(
    "/work/scratch/amunoz/jump-lite-cp-measure-provisional-20260813/canonical-control"
)
DEFAULT_CANONICAL = Path("/work/datasets/jump_lite/rebuttal/cp_measure_codec_v1/full")


def _config(args) -> CandidateConfig:
    return CandidateConfig(
        args.candidate_id,
        args.manifest,
        args.audit_report,
        args.output_root,
        args.state_root,
        args.inventory_digest,
        args.manifest_sha256,
        args.manifest_size,
        args.batch_size,
        16,
        args.test_mode,
    )


def _add_config_arguments(command) -> None:
    for name in ("manifest", "audit-report", "output-root", "state-root"):
        command.add_argument(f"--{name}", type=Path, required=True)
    command.add_argument("--candidate-id", required=True)
    command.add_argument("--inventory-digest", required=True)
    command.add_argument("--manifest-sha256", required=True)
    command.add_argument("--manifest-size", type=int, required=True)
    command.add_argument("--batch-size", type=int, default=4)
    command.add_argument("--test-mode", action="store_true", help=argparse.SUPPRESS)


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--input", type=Path, required=True)
    audit.add_argument("--report", type=Path, required=True)
    audit.add_argument("--kind", choices=("raw", "candidate", "frozen"), default="raw")
    audit.add_argument(
        "--exclusion-policy",
        type=Path,
        help="required for frozen audits; production exclusion-policy JSON",
    )
    audit.add_argument(
        "--damaged-objects",
        type=Path,
        help="required for frozen audits; known damaged-object ledger JSON",
    )
    audit.add_argument(
        "--damaged-sites",
        type=Path,
        help="required for frozen audits; derived damaged-site ledger JSON",
    )
    audit.add_argument(
        "--qc-plates",
        type=Path,
        help="required for frozen audits; pinned red/gray plate ledger JSON",
    )
    run = commands.add_parser("run")
    _add_config_arguments(run)
    run.add_argument("--apply", action="store_true")
    bootstrap = commands.add_parser("bootstrap")
    _add_config_arguments(bootstrap)
    bootstrap.add_argument("--apply", action="store_true")
    acknowledge = commands.add_parser("acknowledge-errors")
    _add_config_arguments(acknowledge)
    acknowledge.add_argument("--expected-count", type=int, required=True)
    acknowledge.add_argument("--apply", action="store_true")
    gov = commands.add_parser("governor")
    gov.add_argument("--apply", action="store_true")
    gov.add_argument("--candidate-id", required=True)
    gov.add_argument("--config-sha256", required=True)
    gov.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE)
    gov.add_argument("--state-root", type=Path, required=True)
    gov.add_argument("--canonical-root", type=Path, default=DEFAULT_CANONICAL)
    gov.add_argument("--output-filesystem", type=Path, default=Path("/work/datasets"))
    gov.add_argument("--test-mode", action="store_true", help=argparse.SUPPRESS)
    adopt = commands.add_parser("validate-adoption")
    _add_config_arguments(adopt)
    for name in (
        "frozen-manifest",
        "frozen-audit",
        "exclusion-policy",
        "damaged-objects",
        "damaged-sites",
        "qc-plates",
    ):
        adopt.add_argument(f"--{name}", type=Path, required=True)
    adopt.add_argument("--frozen-inventory-digest", required=True)
    status = commands.add_parser("status")
    status.add_argument("--state-root", type=Path, required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "audit":
        assert_runtime_task_ceiling()
        print(
            json.dumps(
                audit_inventory(
                    args.input,
                    args.report,
                    kind=args.kind,
                    exclusion_policy=args.exclusion_policy,
                    damaged_objects=args.damaged_objects,
                    damaged_sites=args.damaged_sites,
                    qc_plates=args.qc_plates,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "run":
        install_signal_handlers()
        print(json.dumps(run_candidate(_config(args), args.apply), indent=2))
        return 0
    if args.command == "bootstrap":
        print(json.dumps(bootstrap_candidate(_config(args), args.apply), indent=2))
        return 0
    if args.command == "acknowledge-errors":
        print(
            json.dumps(
                acknowledge_errors(_config(args), args.expected_count, args.apply),
                indent=2,
            )
        )
        return 0
    if args.command == "governor":
        paths = GovernorPaths(
            args.candidate_id,
            args.config_sha256,
            args.state_root,
            args.feature_root / "MQ_state.json",
            args.feature_root / "lossless_state.json",
            args.canonical_root / "masks/MQ/receipts",
            args.canonical_root / "masks/lossless/receipts",
            args.canonical_root / "profiles/MQ",
            args.canonical_root / "profiles/lossless",
            args.output_filesystem,
            args.test_mode,
        )
        print(json.dumps(run_governor(paths, dry_run=not args.apply), indent=2))
        return 0
    if args.command == "validate-adoption":
        print(
            json.dumps(
                validate_adoption_seam(
                    _config(args),
                    args.frozen_manifest,
                    args.frozen_audit,
                    args.frozen_inventory_digest,
                    args.exclusion_policy,
                    args.damaged_objects,
                    args.damaged_sites,
                    args.qc_plates,
                ),
                indent=2,
            )
        )
        return 0
    if args.command == "status":
        result = {}
        for name in (
            "control.json",
            "compression.json",
            "checkpoint.json",
            "governor_snapshots/latest.json",
        ):
            path = args.state_root / name
            result[name] = json.loads(path.read_text()) if path.is_file() else None
        print(json.dumps(result, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
