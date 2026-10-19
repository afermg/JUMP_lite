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
Only a complete ordered 256-site tranche (successful or explicitly excluded sites)
receives a hash-chained tranche record and then advances the compact checkpoint. A signal finishes the active worker sub-batch
but never commits a partial tranche. Restart validates the complete record chain and
fully revalidates the most recently committed tranche. It may reuse valid sites only
in the next uncommitted tranche; committed drift is fatal.

`production-verify-tranche` fully revalidates one selected committed tranche.
`production-finalize-validation` exhaustively validates every site, receipt, chunk,
and tranche, and is permitted only after the checkpoint is complete.

## Missing sources: record-and-skip with a rate guard

The production default is `--missing-source-policy record-and-skip` with
`--max-missing-fraction 0.05`. Both settings are part of the configuration digest and
must be identical in bootstrap, run, status, validation, and acknowledgement commands.
`--missing-source-policy stop` retains strict stop-and-review behavior.

A site is excluded only when **all failed channels** have exhausted five read
attempts with `NoSuchKey` / HTTP 404, and an independent `HeadObject` check also
confirms each object is absent (HTTP 404 with `NoSuchKey`, `NotFound`, or `404`).
Mixed retry histories, timeouts, access denial, missing buckets/versions, and unknown
errors are not missing-object proof. If HEAD succeeds or fails for another reason,
the site is not excluded: it produces a blocking source incident instead. Each
request also retains the S3 client's bounded retry policy.

For confirmed missing objects, the runner:

1. Writes `OUTPUT_ROOT/receipts/<tranche>/<site>.json` with format
   `full-jump-production-excluded-source-v1` and status `excluded-missing-source`.
2. Records the zero-based manifest index, site identity, all expected channels,
   failed channels/URIs in manifest order, every GET retry, HEAD confirmation,
   object count, and production/config/inventory/producer identities.
3. Logs a warning and excludes the **whole site** from both codecs, then continues
   without requiring acknowledgement for routine gaps.

There are no partial-channel images or synthetic pixels. Exclusion receipts are
hash-bound into the ordinary tranche chain. On restart, they are validated and reused
without new source reads, just like successful site receipts. They represent a frozen
build decision, not a live view of S3: a later reappearing object is not silently
reintroduced. Any image output for an excluded site fails validation.

### Missing-site limit

Before starting the next worker sub-batch or committing a tranche, the runner checks
`excluded_sites / manifest_sites_in_this_tranche <= max_missing_fraction`. The default
allows at most 12 excluded sites in a full 256-site tranche; the 13th pauses the run.
The final short tranche uses its actual size, with no grace rounding (e.g. one missing
site in a 10-site final tranche exceeds 5%). The setting accepts finite values from
0 to 1; 0 allows no exclusions, and 1 is an explicit decision to allow even an empty
output tranche. Counts are sites, not channels, and include reused exclusion receipts.

A limit breach writes a blocking incident with reason `missing-source-rate-exceeded`
and the observed count, tranche denominator, and configured limit. It stops before
tranche commit, leaving the checkpoint unchanged. Valid site/exclusion receipts in
that uncommitted tranche remain available for review. Acknowledgement does **not**
bypass the limit: a retry with the same over-limit receipts pauses again. Investigate
before arranging a reviewed policy/inventory change; do not delete receipts to evade
the guard or change the configuration of an existing build in place.

### Counts, validation, and compatibility

Checkpoint and tranche formats are now `full-jump-production-checkpoint-v2` and
`full-jump-production-tranche-v2`. They count `excluded_missing` separately from
`created` and `skipped` (the latter still means successful outputs reused on restart).
Their sum equals processed manifest sites. `production-status` reports manifest,
processed, available, and excluded-site counts; verification/finalization also report
excluded-object counts. Codec inventories must contain exactly the available sites.
The legacy `sites` field in verification/finalization remains the manifest denominator,
**not** the number of usable image sites.

Downstream consumers must use successful receipts/available-site counts and the
explicit exclusion receipts when defining cohorts and denominators; a missing output
without a valid exclusion receipt is still an error. This compressor change does not
rewrite downstream analysis code or publish a dataset.

**This is not a drop-in upgrade of an existing v1 production state.** New policy
identity and v2 accounting require reviewed migration/new-build approval. Existing
v1 checkpoints, producer bindings, and pinned continuous approvals are not silently
rewritten or bypassed. Do not point an active signed deployment at this working tree.

### Blocking source incidents and recovery

Non-missing read failures, rate-limit breaches, or any read failure in `stop` mode
finish only the current worker sub-batch and write bounded evidence under
`STATE_ROOT/source-incidents/<sequence>.json`, followed by an atomic `index.json`.
These receipts include the unchanged checkpoint's SHA-256/tranche boundary and the
exact site/channel/source/retry evidence. They leave the checkpoint byte-for-byte
unchanged, including its legacy `cumulative_errors` field. A separate durable gate
blocks resume regardless of governor control; the governor also refuses unpause while
source incidents are unacknowledged. Generic non-read errors retain their existing
error-counter/stop behavior.

After investigating/resolving a blocking incident, stop the compressor and review the
complete incident list. With the same production identity/policy arguments:

```bash
python -m jump_full_compression production-status "${ARGS[@]}"
# Copy count and list_sha256 from the reviewed source_incidents snapshot.
python -m jump_full_compression production-acknowledge-source-incidents "${ARGS[@]}" \
  --expected-count N --expected-list-sha256 REVIEWED_LIST_SHA256
python -m jump_full_compression production-acknowledge-source-incidents "${ARGS[@]}" \
  --expected-count N --expected-list-sha256 REVIEWED_LIST_SHA256 --apply
```

`N` counts blocking incident receipts, not routine exclusion receipts or objects.
Acknowledgement checks the exact count and ordered receipt-name/hash digest under
the production lock, then leaves control paused for fresh governor evaluation. It
permits retry, not an override of classification, exclusions, or the missing-site
limit. Repeated blocking failures require a new acknowledgement; dry-run changes no
acknowledgement or control. Legacy `production-acknowledge-errors` cannot bypass this
gate.

All reports are credential-safe: only public allowlisted S3 URIs, sanitized error
codes/statuses, SDK retry counts, and retry delays are retained. Signed query strings,
fragments, userinfo, and non-public sources are rejected; raw exception messages,
headers, endpoints, request IDs, and whole botocore responses are never copied.
Receipts are limited to 256 KiB; the blocking ledger to 10,000 receipts and a 2 MiB
index. Unsafe files, hash drift, and interrupted append artifacts fail closed and
require review. Declared damaged-site exclusions, successful all-zero reads, and
candidate callers' original read-exception contract remain unchanged.

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

### Signed acceptance authority and exact schemas

Both acceptance JSON files require detached OpenSSH signatures from the pinned
`jump-full-review` Ed25519 key. The canonical allowed-signers file is
`metadata/full_jump_compression/continuous_approval_allowed_signers`; runtime pins its
exact bytes, SHA-256, public-key fingerprint, and the exact `ssh-keygen` executable.
The signing operator signs each final, immutable receipt (the private-key location is
intentionally not documented):

```bash
ssh-keygen -Y sign -f "$REVIEW_SIGNING_KEY" -n jump-full-production-v1 one-tranche-acceptance.json
ssh-keygen -Y sign -f "$REVIEW_SIGNING_KEY" -n jump-full-production-v1 producer-migration-acceptance.json
ssh-keygen -Y verify -f metadata/full_jump_compression/continuous_approval_allowed_signers \
  -I jump-full-review -n jump-full-production-v1 -s one-tranche-acceptance.json.sig \
  < one-tranche-acceptance.json
```

Pass both absolute signature paths to migration as
`--one-tranche-acceptance-signature` and `--migration-acceptance-signature`; pass the
former again to authorization. Signature and receipt bytes are captured and verified
from regular non-symlink files. The marker records their hashes and runtime revalidates
both signatures before every continuous invocation.

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
tranche 0, 256 sites, and the accepted digest. `metrics.io_pressure_avg10` must be
numeric and exactly `0.0` in each bound governor artifact, and those derived values
must exactly equal the receipt's three I/O-pressure assertions.

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
history/transition/current producer fails closed. After any producer transition, bounded
production is forbidden as well as paused: no further tranche can run until the signed
continuous marker exists. Run the governor only after migration and continuous
authorization.

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
record is committed, all receipts are revalidated, both outputs of every retained site
are decoded, and excluded sites are verified to have no outputs.

Error acknowledgement is exact and leaves control paused with governor evaluation
required. Unknown structural entries, identity drift, stale/malformed control,
unacknowledged errors, task excess, or committed output drift fail closed.
