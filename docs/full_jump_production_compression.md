# Full-JUMP production compression (quarantined)

The native production runner consumes only the accepted frozen inventory. It writes a
quarantined, non-published HQ/MQ build under
`/work/datasets/jump_lite/images/compressed/.jump_full_production/full-jump-hq-mq-v1`.
It does not create codec-root `.zgroup`/`.zattrs`, publish a CPG namespace, or upload.

## Identity and exclusions

Production configuration binds the manifest, frozen audit, build report, exclusion
policy, damaged-object/site ledgers, QC plate ledger, their SHA-256 values, the
7,544,417-site count, codec parameters, producer checkout, CPU/task policy, and fixed
256-site tranche size. Source 15, red plates, and the two known damaged Source-7 sites
are absent; six gray negative-control plates are retained.

## Durability

Sites are published codec-by-codec and then receive a production-specific receipt.
Only a complete ordered 256-site tranche receives a hash-chained tranche record and
then advances the compact checkpoint. A signal finishes the active worker sub-batch
but never commits a partial tranche. Restart validates the complete record chain and
fully revalidates the most recently committed tranche. It may reuse valid sites only
in the next uncommitted tranche; committed drift is fatal.

`production-verify-tranche` fully revalidates one selected committed tranche.
`production-finalize-validation` exhaustively validates every site, receipt, chunk,
and tranche, and is permitted only after the checkpoint is complete.

## One-tranche gate and continuous authorization

Bootstrap starts paused and requires a governor pass. The original unit remains an
unaltered one-tranche gate using exactly `--max-tranches 1`; all other finite values
fail. After tranche 0, independently review all 256 receipts/sites, the chain, feature
progress, restart behavior, and resource telemetry, then create a durable independent
acceptance receipt. A successor deployment must first run the one-time
`production-migrate-producer` dry/apply sequence below; authorization before that
migration is rejected. `production-authorize-continuous` parses the receipt (not just
its caller-supplied SHA-256), rechecks its artifacts and semantics, authenticates the
applied transition and old tranche under its predecessor producer, then atomically
creates a non-overwritable authorization marker and republishes control paused.

### Exact acceptance schemas

Every object below rejects extra or missing keys. All SHA values are lowercase
64-character SHA-256 strings; all timestamps are timezone-aware ISO-8601 strings; all
artifact paths are absolute regular non-symlink files whose bytes match `sha256`.
`full-jump-one-tranche-acceptance-v1` has exactly:

```json
{
  "format_version": "full-jump-one-tranche-acceptance-v1",
  "decision": "GO",
  "production_id": "...", "config_sha256": "...", "inventory_digest": "...",
  "frozen_manifest": {"sha256": "...", "bytes": 0, "site_count": 7544417},
  "checkpoint": {"sha256": "...", "next_index": 256, "completed_tranches": 1, "cumulative_errors": 0, "chain_head": "..."},
  "tranche0": {"record_sha256": "...", "tranche_digest": "...", "site_count": 256},
  "verification": {"artifact": {"path": "/...", "sha256": "..."}, "status": "valid", "tranche": 0, "sites": 256, "tranche_digest": "..."},
  "governor": {
    "before": {"path": "/...", "sha256": "..."},
    "post": {"path": "/...", "sha256": "..."},
    "feature_deltas": {
      "MQ": {"receipt_backed_masks": 1, "canonical_profiles": 1},
      "lossless": {"receipt_backed_masks": 1, "canonical_profiles": 1}
    },
    "io_pressure": {"before_some_avg10": 0, "after_some_avg10": 0, "max_some_avg10": 0}
  },
  "predecessor_producer": {"sha256": "...", "git_commit": "..."},
  "reviews": {
    "code": {"identifier": "...", "reviewed_at": "...+00:00"},
    "science": {"identifier": "...", "reviewed_at": "...+00:00"},
    "ops": {"identifier": "...", "reviewed_at": "...+00:00"}
  },
  "accepted_at": "...+00:00"
}
```

All four feature deltas must be positive integers and must equal the post-minus-before
values in the two bound governor artifacts. The three review identifiers must be
nonempty and distinct. Verification artifact fields must themselves report valid
tranche 0, 256 sites, and the accepted digest. Every I/O-pressure value must be zero.

`producer-migration-acceptance-v1` has exactly:

```json
{
  "format_version": "producer-migration-acceptance-v1", "decision": "GO",
  "production_id": "...", "config_sha256": "...", "inventory_digest": "...",
  "checkpoint_sha256": "...", "tranche0_record_sha256": "...", "tranche0_digest": "...",
  "one_tranche_acceptance": {"path": "/...", "sha256": "..."},
  "predecessor": {"producer_sha256": "...", "software": {}},
  "successor": {"software": {}},
  "review": {"identifier": "...", "reviewed_at": "...+00:00"},
  "approved_at": "...+00:00"
}
```

The `software` objects are exact complete values from predecessor `producer.json` and
successor `software_identity(require_clean=True)`, not partial examples. `decision`
other than literal `GO`, arbitrary text, malformed JSON, non-positive evidence,
identity mismatch, or extra fields fail closed even when the caller supplies the
matching file hash. Live migration additionally requires predecessor producer
`eea9ed8964f7d2f3ce9a164becdfa0530818b07855cde1b578f22e8c686d469a` at commit
`75b18904ea0fe18610feb840888794733fea2fd0`.

### One-time producer migration

Stop both bounded and continuous units. Point `DEPLOY_ROOT`/`PYTHONPATH` at the clean
successor but retain the accepted 75b1890 policy-ledger paths and hashes in the config
arguments; this preserves the accepted config digest. Run
`production-migrate-producer` with both acceptance paths/hashes, first dry and then
`--apply`. The nonblocking production lock proves no compressor owns the state.
Migration fully verifies checkpoint/tranche 0 with the predecessor, verifies clean
successor software, and creates immutable `producers/<sha>.json` histories and
`transitions/00000001.json`. It then changes only current `producer.json`, the compact
checkpoint producer binding, terminal telemetry, and paused control. It never rewrites
an existing site receipt, chunk, or tranche record. Repeating apply after interruption
at any migration durability boundary converges to the same transition. Any conflicting
history/transition/current producer fails closed. Run the governor only after migration
and continuous authorization.

Only the separate `jump-full-production-continuous.service` uses `--continuous`.
Continuous execution fails closed without the authenticated marker and proves the
checkpoint is not behind its authorized chain head. At every tranche boundary it
snapshots the governor's `desired_workers`; allocation changes therefore take effect
for the next tranche. Each tranche uses one persistent worker pool, Zarr's global
thread and async limits remain four, and the 24-task runtime check is sampled after
every sub-batch. Pause and stop are honored between sub-batches and tranches. Terminal
telemetry distinguishes `session-complete`, final `complete`, `stopped`, `paused`, and
`error` and includes current/peak tasks, RSS/max RSS, and CPU affinity.

The service is constrained to CPUs 64-80, `Nice=19`, CPU/IO weight 1, `TasksMax=256`,
one codec thread, no GPU, and the literal production-id output/state writable roots.
Those exact parent/root paths must be created by reviewed bootstrap before installing
the templates because systemd never receives a broader writable parent. The separate
three-hour governor reuses the candidate-compatible telemetry fields but mutates only
compression control. It never changes feature extraction.

## Deployment and commands

Bootstrap and every service invocation must use one clean detached deployment at the
reviewed commit. Remove transient files (including `.pi-subagents`), require a clean
tracked tree, and make the deployment read-only before bootstrap. Do not bootstrap
from a development worktree and then point systemd at a different checkout. The
interpreter must be the same resolved executable with the same file identity and
installed dependency versions used by the unit:
`/work/users/amunoz/projects/JUMP_lite/.venv/bin/python`.

Both bootstrap dry-run and bootstrap apply must be launched with the exact systemd
environment, not merely an interactive-shell approximation. In particular, set the
ten captured thread variables to literal `1`:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TBB_NUM_THREADS=1 ARROW_NUM_THREADS=1 POLARS_MAX_THREADS=1
export RAYON_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=
export GIT_EXECUTABLE=/nix/store/1k2lblqlj39azh6wn1sffa2869vrg3mr-git-2.54.0/bin/git
export DEPLOY_ROOT=/the/reviewed/read-only/detached-deployment
export PYTHONPATH="$DEPLOY_ROOT/src"
PYTHON=/work/users/amunoz/projects/JUMP_lite/.venv/bin/python
```

Use that same `DEPLOY_ROOT`, `PYTHON`, `PYTHONPATH`, `GIT_EXECUTABLE`, empty
`CUDA_VISIBLE_DEVICES`, and ten literal thread values for dry-run, apply, validation,
and the installed units. `producer.json` binds the clean Git tree, resolved Git
executable and hash, resolved interpreter and hash, dependency versions, thread
environment, and exact Zarr pool/concurrency limits; any mismatch makes later
production operations fail closed.

All commands require the complete explicit production identity argument set shown by
`$PYTHON -m jump_full_compression production-bootstrap --help`.

1. Run `production-bootstrap` without `--apply`, review, then apply once, using the
   exact environment and deployment above for both commands.
2. Run the production governor with apply and verify fresh unpaused control.
3. Start `jump-full-production-compress.service`; it can commit at most one tranche.
4. Run `production-verify-tranche --tranche 0`, complete independent review, and write
   the independent one-tranche acceptance receipt.
5. Create and independently review `producer-migration-acceptance-v1`; with both
   compressor units stopped, run `production-migrate-producer` dry and apply once from
   the clean successor while retaining the old ledger paths/config digest.
6. Run `production-verify-tranche --tranche 0` again under transition-aware validation.
7. Run `production-authorize-continuous` with the tranche acceptance receipt path and
   SHA-256, review the dry result, then apply exactly once.
8. Run the governor with apply from the newly paused control and confirm it unpauses.
9. Start `jump-full-production-continuous.service`, and only then enable the separate
   three-hour governor timer.

Continuous output remains quarantined. Do not publish or upload CPG artifacts.

Each live run and validation recomputes the clean checkout, interpreter, dependency,
and executable identity and requires exact equality with `producer.json`. Manifest
authentication and Parquet scans share one open inode snapshot; row-group metadata
seeks directly near the checkpoint rather than rescanning the prefix. Before a tranche
record is committed, all 256 receipts and both outputs are decoded and revalidated.

Error acknowledgement is exact and leaves control paused with governor evaluation
required. Unknown structural entries, identity drift, stale/malformed control,
unacknowledged errors, task excess, or committed output drift fail closed.
