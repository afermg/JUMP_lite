# Full-JUMP candidate compression (pre-production)

This package creates bounded, quarantined JPEG XL candidates. It cannot publish
a final `jump_full` dataset or a CPG prefix.

## Contract and quarantine

- Zarr v2, one site-major array/full-site `uint16` chunk.
- HQ is JPEG XL distance 1.0; MQ is distance 3.0; `numthreads=1`.
- Channel order is `AGP,DNA,ER,Mito,RNA`; bounded historical candidates may
  contain `source_15` as exactly `AGP,DNA,ER,Mito`. RNA is never synthesized.
  Production frozen manifests exclude all `source_15` rows.
- One source decode feeds both codecs.
- OMP/OpenBLAS/MKL/BLIS/VecLib/NumExpr/TBB/PyArrow/Polars/Rayon are capped at
  one before third-party imports and recorded.
- Python DuckDB and Polars are not part of the compressor/audit runtime.
  Preparation rejects more than 24 existing Linux tasks, and each worker batch
  is rejected unless current tasks plus workers remains at most 24.

Live candidate data and operational state are candidate-bound and separate:

```text
/work/datasets/jump_lite/images/compressed/.jump_full_candidate/<candidate-id>/
/work/datasets/jump_lite/full_jump_compression_state/v1.0/<candidate-id>/
```

Every existing writable path component is checked for symlinks. Candidate,
receipt, codec, site, checkpoint, and state roots reject unknown entries.
Candidate codec roots intentionally omit `.zgroup` and `.zattrs`. Site files
are fsynced before atomic rename, destination parents are fsynced, and only
then is the durable receipt written. A state-root lock excludes concurrent
controllers, bootstrap, and acknowledgement.

## Inventory identity

`audit --kind raw` streams an arbitrary inventory and is not runnable input. A
separately selected `--kind candidate` manifest is capped at 256 rows and keeps
all selection/provenance columns. Every column participates in row identity.
PyArrow reads Parquet batches with `use_threads=False`; no in-process sort is
performed. All raw, candidate, and frozen manifests must already be in strict
physical order by `Metadata_Source,Metadata_Batch,Metadata_Plate,Metadata_Well,Metadata_Site`.
Duplicate or decreasing identity tuples fail closed.

If a large raw inventory is not ordered, sort it as a separate pre-processing
operation outside this package. For example, a reviewed DuckDB **CLI** job may
use one thread and an explicitly provisioned temporary directory:

```bash
duckdb -c "SET threads=1; SET temp_directory='/bounded/scratch/duckdb'; \
COPY (SELECT * FROM read_parquet('raw.parquet') ORDER BY Metadata_Source, \
Metadata_Batch, Metadata_Plate, Metadata_Well, Metadata_Site) \
TO 'canonical.parquet' (FORMAT PARQUET);"
```

Validate storage bounds separately. Never import Python `duckdb` into the
candidate deployment or service runtime.

`inventory_digest` is portable content identity: schema, manifest SHA-256 and
byte count, ordered keys/rows, source counts, audit fields and anomaly results.
It does not include absolute path, mtime, device, or inode, so byte-identical
relocation preserves the digest. Local path/stat evidence and its own digest
are recorded separately. Loading a report recomputes its content digest,
checks local-evidence integrity, and validates the current file SHA-256/bytes.
Changing a report field without regenerating the accepted digest is rejected.

The preliminary `misc/jump_index.parquet` remains raw input only. The production
policies for `source_15` and known damaged objects are resolved as exclusions:
all `source_15` rows are excluded, and both affected `source_7` sites are
excluded whole. The exact four zero-filled objects and derived two-site set are
preserved under `metadata/full_jump_compression/`; unknown decode failures
remain fatal. Historical candidate behavior is unchanged, including the
completed four-channel `source_15` canary.

A frozen audit validates but never filters these exclusions. New reports bind
an explicit `audit_success` field; failed reports remain useful diagnostics but
set `release_identity_frozen=false`, and audit loading rejects them. Legacy v2
raw/candidate reports without that field remain loadable only when their exact
row count, empty anomaly fields, non-frozen marker, digest, and file identity
all pass; legacy frozen reports are never accepted. Every audit reads schema,
row count, rows, and hashes from one captured Parquet byte snapshot and rejects
a manifest path changed during the audit. Frozen audits require the explicit
policy, all three independently pinned classification ledgers, and the
content-bound successful builder completion report:

```bash
python -m jump_full_compression audit --kind frozen \
  --input /absolute/frozen.parquet --report /absolute/frozen-audit.json \
  --exclusion-policy metadata/full_jump_compression/production_exclusion_policy_v1.json \
  --damaged-objects metadata/full_jump_compression/known_damaged_objects_v1.json \
  --damaged-sites metadata/full_jump_compression/known_damaged_sites_v1.json \
  --qc-plates metadata/full_jump_compression/qc_plate_classification_v1.json \
  --build-report /absolute/full-jump-production-manifest-build.json
```

The QC ledger is generated offline by
`prep/generate_full_jump_qc_ledger.py` from `jump-cellpainting/datasets` commit
`016e865` and binds the exact 169 red and six gray source/batch/plate identities.
It records classification only, not the release action. The reviewed release
policy is `exclude_red_include_gray`: all 169 technically problematic red plates
are excluded, while the six technically valid negative-control-only gray plates
are included. Frozen auditing rejects every red row and permits gray rows.
Production remains blocked on a commit-pinned complete ordered inventory and the
final CPG namespace.

### Offline production-manifest build

Building the complete manifest is deliberately split into three reviewed steps:

1. **Stage the six gray load-data CSVs.** This is the only network-fetch step.
   Fetch each canonical public URI outside the builder, retain it as a regular
   non-symlink file, and write a `full-jump-gray-input-receipt-v1` JSON object
   with exactly six entries. Each entry contains `source`, `batch`, `plate`, the
   canonical `public_uri`, absolute local `path`, `bytes`, and `sha256`. Review
   those bytes and the receipt before proceeding.
2. **Build offline.** `prep/build_full_jump_manifest.py` makes no network calls.
   It binds the exact preliminary Parquet, pinned datasets checkout and all
   policy ledgers; proves the preliminary plate universe; excludes all
   `source_15` rows and both damaged sites; and adds only the six receipted gray
   CSVs. Preliminary Parquet and gray CSV parsing uses the exact stable bytes
   captured and hashed in memory. The builder publishes a unique, strictly
   identity-sorted Parquet first, then publishes its content-bound report as the
   completion marker. The report explicitly keeps
   `release_identity_frozen=false`.
3. **Freeze separately.** Run `jump_full_compression audit --kind frozen` with
   the checked-in policy, three ledgers, and builder completion report. The
   audit validates the report digest, manifest bytes/schema/rows/order claims,
   policy action, and ledger bindings against its own captured manifest bytes.
   Only a successful frozen-audit report may set
   `release_identity_frozen=true`; the builder never does so.

Example offline build (all paths must be absolute):

```bash
PYTHONPATH=src python prep/build_full_jump_manifest.py \
  --preliminary /work/datasets/jump_lite/misc/jump_index.parquet \
  --preliminary-sha256 cb31bb16bee36d126e45a9588d80b9e8cef5e16f468a9b7b2738f0b618152a44 \
  --preliminary-bytes 145201434 \
  --datasets-checkout /absolute/pinned/datasets-checkout \
  --exclusion-policy /absolute/metadata/full_jump_compression/production_exclusion_policy_v1.json \
  --damaged-objects /absolute/metadata/full_jump_compression/known_damaged_objects_v1.json \
  --damaged-sites /absolute/metadata/full_jump_compression/known_damaged_sites_v1.json \
  --qc-plates /absolute/metadata/full_jump_compression/qc_plate_classification_v1.json \
  --gray-receipt /absolute/staging/gray-input-receipt.json \
  --output /absolute/identity/full-jump-production-manifest.parquet \
  --report /absolute/identity/full-jump-production-manifest-build.json
```

The builder overrides all native thread-pool environment variables to one,
caps PyArrow CPU and I/O pools at one, and imports no Python DuckDB. Before
materializing the preliminary table it sums Parquet uncompressed column-chunk
bytes and requires effective available memory of at least 64 GiB or six times
that payload, whichever is larger. Effective availability is the minimum of
Linux `MemAvailable` and every finite cgroup-v2 ancestor's remaining allowance;
the build report records this preflight.
Gray CSVs may contain non-URL load-data columns, but the builder selects exactly
the five identity and five `URL_Orig*` columns and rejects missing channels or
unsupported URL channels.

Output and report are not an atomic pair. An exclusive directory build lock
serializes cooperating builders. The output is published first without
overwrite, and the report is the completion marker. Report-publication failures
remove the output only if it is still the inode hard-linked by that builder;
foreign paths are never unlinked. A process or host crash between publications
can still leave a fail-closed orphan output. A path with output but no valid
report must never be frozen, adopted, or overwritten; explicitly quarantine or
remove that orphan after review before rerunning. The builder otherwise refuses
existing outputs and removes unpublished temporary files after failures.

## Reviewed launch sequence

No step below should be run from this mutable worktree. First create and review
an immutable deployment. Let `ARGS` denote these exact arguments:

```bash
ARGS=(
  --candidate-id candidate-001 --manifest /absolute/candidate.parquet
  --audit-report /absolute/candidate-audit.json --inventory-digest SHA
  --manifest-sha256 SHA --manifest-size BYTES
  --output-root /work/datasets/jump_lite/images/compressed/.jump_full_candidate/candidate-001
  --state-root /work/datasets/jump_lite/full_jump_compression_state/v1.0/candidate-001
  --batch-size 4
)
```

The required order is:

1. **Audit, review, commit, and prove the producer is clean.**
   ```bash
   PYTHONPATH=/immutable/deployment/src .venv/bin/python -m jump_full_compression audit \
     --kind candidate --input /absolute/candidate.parquet \
     --report /absolute/candidate-audit.json
   git status --short --untracked-files=all # must be empty in deployment
   ```
2. **Bootstrap once, paused.** Dry-run first, then explicit apply:
   ```bash
   python -m jump_full_compression bootstrap "${ARGS[@]}"
   python -m jump_full_compression bootstrap "${ARGS[@]}" --apply
   ```
   Bootstrap validates config/audit and the clean producer, then atomically
   publishes fresh candidate/config-bound compression telemetry followed by a
   paused control requiring governor evaluation. It refuses initialized state.
3. **Run the governor explicitly with apply.**
   ```bash
   python -m jump_full_compression governor --apply \
     --candidate-id candidate-001 --config-sha256 CONFIG_SHA \
     --feature-root FEATURE_ROOT --canonical-root CANONICAL_ROOT \
     --state-root STATE_ROOT --output-filesystem /work/datasets
   ```
4. **Verify control before starting compression.** `status` must show a fresh,
   identity-bound `control.json`, expected CPUs `64..80`, no telemetry errors,
   and a reviewed paused/unpaused decision:
   ```bash
   python -m jump_full_compression status --state-root STATE_ROOT
   ```
5. **Start the compressor explicitly.** Do not enable it implicitly:
   ```bash
   systemctl --user start jump-full-candidate-compress.service
   ```
6. **Only after observing the canary, enable the three-hour timer:**
   ```bash
   systemctl --user enable --now jump-full-compression-governor.timer
   ```

The systemd `candidate.env` is parsed as deterministic runtime fields, not a
free-form argument string. It must contain one reviewed, whitespace-free value
for each of:

```text
DEPLOY_ROOT GIT_EXECUTABLE CANDIDATE_ID MANIFEST AUDIT_REPORT INVENTORY_DIGEST
MANIFEST_SHA256 MANIFEST_SIZE OUTPUT_ROOT STATE_ROOT BATCH_SIZE
CONFIG_SHA256 FEATURE_ROOT CANONICAL_ROOT
```

`DEPLOY_ROOT` must be an immutable reviewed checkout; all paths are absolute.
`GIT_EXECUTABLE` must be the absolute executable path recorded in every receipt;
the service never depends on an ambient `PATH`. Untracked non-ignored files
anywhere in that deployment, including importable
shadow modules under `src/`, block live bootstrap and apply.
The service templates remain uninstalled until this file and every field are
reviewed together.

## Errors, restart, and adoption

Persistent `cumulative_errors` must equal control
`acknowledged_error_count`; otherwise the controller performs no site work.
Acknowledgement is bounded and can only equal the current observed count:

```bash
python -m jump_full_compression acknowledge-errors "${ARGS[@]}" --expected-count N
python -m jump_full_compression acknowledge-errors "${ARGS[@]}" --expected-count N --apply
```

The apply form writes a fresh **paused** control and marks governor evaluation
required. Future/oversized/stale expected counts fail. Run the governor with
`--apply` again and verify control before restarting compression.

Each checkpoint stores SHA-256 for every receipt in its prefix. Restart checks
the hash, exact receipt fields/source observations, both codec objects, and
producer/config identity. The earliest bad prefix is removed and rebuilt.

`validate-adoption` is validation-only. It requires a complete original bounded
candidate: complete checkpoint, exact row/receipt set, exact site directories
in both codecs, and checkpoint-bound receipt hashes. It also requires the same
explicit `--exclusion-policy`, `--damaged-objects`, `--damaged-sites`,
`--qc-plates`, and `--build-report` artifacts as a frozen audit. It freshly
audits the frozen manifest
and checks each original row is unchanged and present. It never moves, promotes,
publishes, or uploads data.

## Governor boundary

The governor accepts live state only at the literal
`/work/datasets/jump_lite/full_jump_compression_state/v1.0/<candidate-id>` path.
It rejects redirected shared parents and symlinks in state, control, compression,
or snapshot paths before reading or writing. Temporary roots require the hidden
test-mode seam and are never used by the service.

The three-hour persistent timer is `OnCalendar=*-*-* 00/3:00:00`. The governor
checks feature heartbeats and independently scans the canonical mask-receipt
and profile-Parquet roots for exact current unfinished progress, then checks
every profile PID, all segmentation cgroup descendants, active-service MainPID,
compression heartbeat/progress/errors, load, memory, storage, and I/O pressure.
Missing or racy telemetry fails closed. Two healthy windows permit one
four-worker ramp and reset the window count.

The governor writes only compression control/snapshots. It never signals,
restarts, repins, or edits feature processes. Expanding profile extraction
after segmentation is a separately reviewed manual migration. CPG upload and
production promotion remain outside this package.
