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

## One-tranche launch gate

Bootstrap starts paused and requires a governor pass. The installed-template command
hardcodes `--max-tranches 1`. Every `production-run` invocation requires exactly
`--max-tranches 1`; larger, zero, or negative values fail until a future reviewed
authorization marker is implemented. There is intentionally no continuous mode or
authorization marker. Do not edit the unit to bypass this gate. After the first live
tranche, stop and independently review resource telemetry, all 256 receipts/sites,
the tranche chain, feature progress, and restart behavior. A future continuous mode
requires a separate reviewed implementation.

The service is constrained to CPUs 64-80, `Nice=19`, CPU/IO weight 1, `TasksMax=256`,
one codec thread, no GPU, and the literal production-id output/state writable roots.
Those exact parent/root paths must be created by reviewed bootstrap before installing
the templates because systemd never receives a broader writable parent. The separate
three-hour governor reuses the candidate-compatible telemetry fields but mutates only
compression control. It never changes feature extraction.

## Commands

All commands require the complete explicit production identity argument set shown by
`python -m jump_full_compression production-bootstrap --help`.

1. Run `production-bootstrap` without `--apply`, review, then apply once.
2. Run the production governor with apply and verify fresh unpaused control.
3. Start `jump-full-production-compress.service`; it can commit at most one tranche.
4. Run `production-verify-tranche --tranche 0` and independent review.
5. Do not continue automatically. Continuous execution is impossible by design.

Each live run and validation recomputes the clean checkout, interpreter, dependency,
and executable identity and requires exact equality with `producer.json`. Manifest
authentication and Parquet scans share one open inode snapshot; row-group metadata
seeks directly near the checkpoint rather than rescanning the prefix. Before a tranche
record is committed, all 256 receipts and both outputs are decoded and revalidated.

Error acknowledgement is exact and leaves control paused with governor evaluation
required. Unknown structural entries, identity drift, stale/malformed control,
unacknowledged errors, task excess, or committed output drift fail closed.
