# JUMP-Lite CPG upload machinery

This directory contains the preparation and fail-closed upload tools for the
JUMP-Lite contribution beneath `cpg0016-jump/source_all`.

No script stores AWS credentials in the repository. The long-lived CPG grant
keys remain in `~/.cpg_key_id` and `~/.cpg_access_key`; temporary credentials
are requested through S3 Access Grants.

## Safety invariant

All compressed image datasets and every corresponding per-site Parquet variant
must contain the exact same frozen set of 655,101 paper-cohort site keys. The MQ Zarr is the
canonical set for this release.

Uploads must use `upload_to_staging.sh` for one unchanged directory,
`run_background_upload.sh` for the main release, or the checkpoint-aware
`upload_zstd_to_staging.py` for a concurrently changing Zstd build. The normal
workflows run `validate_release.py` before writes. The Zstd workflow instead
uploads only builder-checkpoint-confirmed arrays, withholds the group root, and
performs full local and remote validation before making the store complete.

## Quick runbook

From the repository root, the normal order for a release is:

```bash
# 0. Keep user services alive after SSH logout (one-time, requires permission).
sudo loginctl enable-linger "$USER"
loginctl show-user "$USER" -p Linger

# 1. Audit identities; review any discrepancy before applying a repair.
.venv/bin/python cpg_upload/reconcile_site_sets.py

# 2. Freeze metadata and run the fail-closed preflight.
.venv/bin/python cpg_upload/build_cpg_metadata.py
.venv/bin/python cpg_upload/validate_release.py

# 3. Install services once, then start the main release upload.
mkdir -p ~/.config/systemd/user
for unit in jump-lite-cpg-upload jump-lite-zstd-rebuild jump-lite-zstd-upload; do
  ln -sfn "$PWD/cpg_upload/systemd/$unit.service" \
    "$HOME/.config/systemd/user/$unit.service"
done
systemctl --user daemon-reload
systemctl --user enable --now jump-lite-cpg-upload.service

# 4. If the release includes Zstd, overlap rebuilding and transfer.
systemctl --user enable --now jump-lite-zstd-rebuild.service
systemctl --user enable --now jump-lite-zstd-upload.service

# 5. Monitor read-only status.
cpg_upload/upload_status.sh
cpg_upload/zstd_rebuild_status.sh
cpg_upload/zstd_upload_status.sh
```

Do not notify the CPG maintainer merely because site uploads reach 100%. Wait
for final local validation, publication of the root Zarr metadata, recursive S3
verification, refreshed release metadata, and the final validation report. The
complete finalization checklist is below.

## Files

- `reconcile_site_sets.py`: audits MQ/HQ/D20 and reversibly quarantines surplus
  image arrays and per-site Parquets. It never deletes data.
- `build_cpg_metadata.py`: joins the exact MQ keys to the full JUMP-Lite index
  and writes frozen release metadata under
  `/work/datasets/jump_lite/cpg_release/metadata/`.
- `validate_release.py`: read-only, fail-closed release preflight.
- `activate_cpg_credentials.sh`: exchanges the CPG grant keys for temporary,
  prefix-scoped READWRITE credentials. This file must be sourced.
- `upload_to_staging.sh`: validates and then runs one constrained CPG staging
  sync. It is an AWS dry run unless `--apply` is supplied.
- `run_background_upload.sh`: supervises the complete v1.0 upload, renews
  temporary credentials, and safely resumes interrupted image syncs.
- `upload_profiles_to_staging.py`: concurrently uploads flat local Parquets
  into the CPG model/source/batch/plate/well-site hierarchy without creating
  millions of local hard links. Per-variant checkpoints make it resumable.
- `upload_status.sh`: reports supervisor, component, log, and profile-checkpoint
  progress for the active background run.
- `rebuild_zstd_from_originals.py`: streams the five original public TIFFs for
  each frozen MQ site into memory and writes a matching site-major Zarr v3 array
  with lossless Blosc/Zstd compression. It does not cache TIFFs.
- `run_zstd_rebuild.sh`: background wrapper with durable logs and completion
  markers for the resumable Zstd rebuild.
- `zstd_rebuild_status.sh`: reports the active rebuild checkpoint and log tail.
- `upload_zstd_to_staging.py`: streams only checkpoint-confirmed complete Zstd
  arrays to v1.0, withholds the group root until final local validation, and
  verifies final S3 object count and bytes.
- `run_zstd_upload.sh` and `zstd_upload_status.sh`: durable wrapper and status
  report for the concurrent Zstd upload.
- `systemd/*.service`: persistent user-service definitions that resume the CPG
  upload, Zstd rebuild, and Zstd upload after the user manager starts.
- `verify_staging.sh`: compares one local directory's file count with one
  recursive staging object count.
- `verify_complete_staging.py`: restart-safe, read-only final audit of all three
  JPEG XL and all 16 transformed embedding prefixes. It compares both object
  counts and bytes and checkpoints each completed prefix.
- `JUMP_LITE_README.md`: dataset-facing README copied to the release root by
  the metadata builder and uploaded with the release.

## 1. Reconcile site sets when inputs change

Audit only:

```bash
python cpg_upload/reconcile_site_sets.py
```

For v1.0, this audit found 274 surplus HQ/D20 image-array directories and 1,370
matching surplus Parquets. They were moved to a timestamped quarantine rather
than deleted. If a future audit reports a discrepancy, first save its output
and review whether the canonical manifest intentionally changed. To print paths
that require write ACLs:

```bash
python cpg_upload/reconcile_site_sets.py \
  --print-surplus-image-paths > /tmp/jump_lite_surplus_image_paths.txt
```

Only after review and, if necessary, an administrator grants access, apply the
reversible repair:

```bash
python cpg_upload/reconcile_site_sets.py --apply
```

Surplus entries are moved under `/work/datasets/jump_lite/quarantine/` with a
JSON manifest. Never delete or reuse a quarantine while preparing a release.

## 2. Build frozen release metadata

```bash
python cpg_upload/build_cpg_metadata.py
```

This deliberately uses the existing MQ keys rather than resampling sites. It
writes wide and tidy image indices, perturbation metadata, release-filtered
annotations, a plate manifest, and a metadata manifest. It also copies
`cpg_upload/JUMP_LITE_README.md` to
`/work/datasets/jump_lite/cpg_release/README.md` and records its byte size.
Therefore rerun this command after every dataset-facing README change and before
the final metadata sync.

## 3. Validate

```bash
python cpg_upload/validate_release.py
```

A successful validation reports `CPG release status: ready` and exits zero.
Anything else blocks upload. Before Zstd exists, the default validates the
three JPEG XL stores, embeddings, and metadata. For a final release that
requires the completed lossless store, use:

```bash
python cpg_upload/validate_release.py \
  --require-zstd \
  --json-output /work/datasets/jump_lite/cpg_release/final_validation_report.json
```

The finalized Zstd store is also validated automatically whenever its final
path exists.

## 4. CPG object layout

The release uses the following agreed `source_all` namespace:

```text
cpg0016-jump/source_all/
├── images/2026_jump_lite_v1.0/images_compressed/<codec>.zarr/
├── workspace/publication_data/2026_jump_lite/metadata/v1.0/
│   └── <release metadata files>
└── workspace_dl/embeddings/
    └── <model>-<codec>/2026_jump_lite_v1.0/
        └── <source>/<batch>/<plate>/<well>-<site>/embedding.parquet
```

The image codecs are lossless `zstd` plus `jpegxl_lossy_mq`,
`jpegxl_lossy_hq`, and `jpegxl_lossy_d20`. Metadata includes the dataset README,
wide and tidy image indices, perturbation metadata, RefChemDB annotations, plate
manifest, and release manifest.

Frozen v1.0 facts useful for auditing:

| Invariant | Value |
|---|---:|
| Sites per image/embedding variant | 655,101 |
| Modality-specific perturbations | 24,356 |
| Site-key SHA-256 | `4ea6ea3f5457c33a1412a80a89d8696d4f8e77474cf449e75db7ce6ba98685e2` |
| Embedding variants | 16 |
| Embedding Parquets | 10,481,616 |
| Final Zstd objects | 1,310,203 |
| Final Zstd bytes | 4,812,456,031,773 (4.812 TB; 4.377 TiB) |

Local embedding files are flat and named:

```text
<source>__<batch>__<plate>__<well>__<site>.parquet
```

`upload_profiles_to_staging.py` parses that identity and writes the standard CPG
well-site object key shown above. It maps internal names
`openphenom_confusing` to `openphenom` and `subcell__clip01` to
`subcell_clip01`; local data are not renamed.

The original TIFFs and source-specific `load_data_csv` files already exist in
the six contributing JUMP source folders. They are referenced by the deposited
indices and are not duplicated under `source_all`.

## 5. Activate temporary CPG credentials

The maintainer-provided long-lived grant files are expected at
`~/.cpg_key_id` and `~/.cpg_access_key`. Keep them outside the repository and
restrict their permissions:

```bash
chmod 600 ~/.cpg_key_id ~/.cpg_access_key
```

Install AWS CLI v2 first, then source temporary credentials only for manual
commands:

```bash
source cpg_upload/activate_cpg_credentials.sh
```

The credentials are scoped to:

```text
s3://staging-cellpainting-gallery/cpg0016-jump/*
```

and are requested with `READWRITE` permission in `us-east-1` for 12 hours.
AWS CLI v2 can be used without permanent installation through:

```bash
nix shell nixpkgs#awscli2
```

Never print, log, commit, or pass the long-lived keys on a command line. The
Python uploaders request and refresh their own temporary credentials; systemd
services do not need a manually sourced shell environment.

## 6. Run the complete upload in the background

Install and start the persistent user service from the repository root:

```bash
mkdir -p ~/.config/systemd/user
ln -sfn "$PWD/cpg_upload/systemd/jump-lite-cpg-upload.service" \
  ~/.config/systemd/user/jump-lite-cpg-upload.service
systemctl --user daemon-reload
systemctl --user enable --now jump-lite-cpg-upload.service
```

The enabled service resumes whenever the user manager starts. To keep user
services running after the final login session closes, an administrator must
run `loginctl enable-linger amunoz`. Check state with
`systemctl --user status jump-lite-cpg-upload.service`.

The supervisor validates once before writes, creates a clean metadata view,
starts metadata plus three image syncs, and starts the transformed embedding
upload. Image syncs stop after 11 hours, renew their 12-hour credentials, and
resume. The Python embedding uploader refreshes credentials proactively and
stores deterministic checkpoints under:

```text
/work/datasets/jump_lite/cpg_upload_state/v1.0/profiles/
```

Run logs are stored under a UTC timestamp in:

```text
/work/datasets/jump_lite/cpg_upload_logs/
```

The `latest` symlink points to the active or most recent run. Monitor without
changing S3:

```bash
cpg_upload/upload_status.sh
```

Re-running the supervisor is safe when its checkpoints were created for the
same destination: `aws s3 sync` skips matching image objects, and profile
checkpoints skip successfully uploaded Parquets. Profile checkpoints now record
the full CPG destination prefix and fail closed if an older-layout checkpoint
is reused; use a new checkpoint root rather than relabeling an old checkpoint.
For the supervisor, set `CPG_UPLOAD_STATE_ROOT` to that new directory. No upload
command uses `--delete` or follows symlinks.

## 7. Rebuild and stream the lossless Zstd store

The legacy local `zstd.zarr` is an interrupted one-plate experiment with an
incompatible well/channel layout. Install the persistent rebuild service to
create the v1.0 lossless store directly from original TIFFs:

```bash
mkdir -p ~/.config/systemd/user
ln -sfn "$PWD/cpg_upload/systemd/jump-lite-zstd-rebuild.service" \
  ~/.config/systemd/user/jump-lite-zstd-rebuild.service
systemctl --user daemon-reload
systemctl --user enable --now jump-lite-zstd-rebuild.service
```

Check service state with
`systemctl --user status jump-lite-zstd-rebuild.service`.

The frozen site index supplies exactly the 655,101 MQ keys and five original
TIFF URLs per site. For each site, the builder:

1. downloads AGP, DNA, ER, Mito, and RNA directly from the public CPG;
2. decodes them in memory without retaining raw TIFFs;
3. checks shape and dtype against the corresponding MQ array;
4. writes one `(5, y, x)` Zarr v3 array and one Blosc/Zstd level-9 chunk; and
5. checkpoints progress after each bounded batch.

Four `source_7` inputs are known zero-filled objects rather than TIFFs: ER for
`CP3-SC1-18/I22/site 2`, and DNA, Mito, and RNA for site 3. They share the
frozen size/ETag documented in
[jump-cellpainting/datasets#177](https://github.com/jump-cellpainting/datasets/issues/177).
The builder permits a zero-plane reconstruction only for those exact
URI/size/ETag combinations; every other undecodable TIFF remains a hard error.
Keep this exception list explicit and remove it when corrected upstream
metadata excludes those sites.

The original image-store parent is not writable by the uploader, so the
resumable building store, validated replacement, and state are kept at:

```text
/work/datasets/jump_lite/zstd_rebuild/v1.0/zstd.building.zarr/
/work/datasets/jump_lite/zstd_rebuild/v1.0/zstd.zarr/
/work/datasets/jump_lite/zstd_rebuild_state/v1.0/checkpoint.json
```

Monitor it with:

```bash
cpg_upload/zstd_rebuild_status.sh
```

On completion, the builder verifies the full site count and canonical SHA-256
digest and atomically renames the writable building store to `zstd.zarr`. The
legacy incomplete store in the protected image directory remains untouched. A
truncated test run never performs final renaming.

The streaming uploader can safely overlap the multi-terabyte transfer with the
rebuild. It follows only completed manifest batches and uploads each chunk before
its array metadata. It deliberately withholds the group-level `zarr.json` until
the builder has finalized all 655,101 release arrays and the uploader has repeated full
local validation:

```bash
ln -sfn "$PWD/cpg_upload/systemd/jump-lite-zstd-upload.service" \
  ~/.config/systemd/user/jump-lite-zstd-upload.service
systemctl --user daemon-reload
systemctl --user enable --now jump-lite-zstd-upload.service
cpg_upload/zstd_upload_status.sh
```

State is stored at
`/work/datasets/jump_lite/cpg_upload_state/v1.0/zstd/checkpoint.json`. The final
destination is
`cpg0016-jump/source_all/images/2026_jump_lite_v1.0/images_compressed/zstd.zarr/`.

A builder restart scans the manifest from its beginning and skips existing
complete arrays. Its displayed `processed_manifest_sites` may therefore fall
from a large number to zero; this is a rescan, not lost data. The streaming
uploader waits whenever that safe checkpoint falls behind its own upload
checkpoint and resumes automatically after the scan catches up. Do not edit
either checkpoint by hand.

## 8. Monitor, interrupt, and recover

Useful read-only commands:

```bash
systemctl --user --no-pager status jump-lite-cpg-upload.service
systemctl --user --no-pager status jump-lite-zstd-rebuild.service
systemctl --user --no-pager status jump-lite-zstd-upload.service

journalctl --user -fu jump-lite-cpg-upload.service
journalctl --user -fu jump-lite-zstd-rebuild.service
journalctl --user -fu jump-lite-zstd-upload.service

cpg_upload/upload_status.sh
cpg_upload/zstd_rebuild_status.sh
cpg_upload/zstd_upload_status.sh
```

Durable state and logs are deliberately outside the repository:

| Component | Checkpoint/state | Timestamped logs |
|---|---|---|
| Embeddings/main release | `/work/datasets/jump_lite/cpg_upload_state/v1.0/` | `/work/datasets/jump_lite/cpg_upload_logs/` |
| Zstd rebuild | `/work/datasets/jump_lite/zstd_rebuild_state/v1.0/` | `/work/datasets/jump_lite/zstd_rebuild_logs/` |
| Zstd upload | `/work/datasets/jump_lite/cpg_upload_state/v1.0/zstd/` | `/work/datasets/jump_lite/zstd_upload_logs/` |

A final SSH logout with `Linger=no` cleanly stops the entire user manager and
all three services. Enabled services resume on the next login, but long jobs do
not run while logged out. Check and fix this before starting:

```bash
loginctl show-user "$USER" -p Linger
sudo loginctl enable-linger "$USER"
```

If a process fails, inspect the newest log and `FAILED` marker before restarting:

```bash
readlink -f /work/datasets/jump_lite/zstd_rebuild_logs/latest
systemctl --user restart jump-lite-zstd-rebuild.service
```

Restarts are expected to rescan local files and safely skip completed work.
Never remove checkpoints merely to make a status display look current. Never
use `aws s3 sync --delete`; partial staging objects and local quarantine data
are evidence needed for recovery.

Expected terminal states are:

- `upload_status.sh`: `Overall state: COMPLETE` and all 16 profile checkpoints
  at `655,101/655,101`;
- `zstd_rebuild_status.sh`: `State: COMPLETE`, `complete: 1`, and final path
  `.../zstd.zarr`;
- `zstd_upload_status.sh`: `Run state: COMPLETE`, `complete: true`,
  `root_metadata_published: true`, and a `remote_verification` block.

A service becoming `inactive (dead)` with `Result=success` after reaching one
of these states is normal completion, not a crash.

## 9. Upload one unchanged directory

For a one-component dry run:

```bash
cpg_upload/upload_to_staging.sh LOCAL_PATH RELATIVE_DESTINATION
```

Add `--apply` only after reviewing its destination. Destination arguments are
relative to `cpg0016-jump/source_all/`.

## 10. Verify staging

After transfer, compare local and staging object counts:

```bash
cpg_upload/verify_staging.sh LOCAL_PATH RELATIVE_DESTINATION
```

For the complete bulk audit, install the restart-safe verification service:

```bash
ln -sfn "$PWD/cpg_upload/systemd/jump-lite-staging-verify.service" \
  ~/.config/systemd/user/jump-lite-staging-verify.service
systemctl --user daemon-reload
systemctl --user enable --now jump-lite-staging-verify.service
journalctl --user -fu jump-lite-staging-verify.service
```

The service compares exact local/checkpoint object counts and byte totals for
all three JPEG XL stores and all 16 transformed embedding prefixes. It writes
`/work/datasets/jump_lite/cpg_release/staging_bulk_verification.json` after each
completed prefix, so a restart skips already verified prefixes. Recursive S3
listing covers roughly 21 million objects and can take well over 30 minutes.
Do not substitute upload checkpoint counts for this remote verification.

## 11. Finalize and hand off a release

Use this checklist after all bulk transfers stop changing:

1. Confirm the three terminal states listed above. For Zstd, 100% of site
   objects is not enough: wait for root metadata publication and the built-in
   remote object/byte verification.
2. Update `cpg_upload/JUMP_LITE_README.md` with final sizes, anomalies, and
   citations, then rebuild the release metadata:

   ```bash
   .venv/bin/python cpg_upload/build_cpg_metadata.py
   ```

3. Run and retain the final fail-closed report:

   ```bash
   .venv/bin/python cpg_upload/validate_release.py \
     --require-zstd \
     --json-output /work/datasets/jump_lite/cpg_release/final_validation_report.json
   ```

4. Upload the refreshed README and metadata together. Build a temporary view so
   `README.md` shares the metadata version prefix without mutating the frozen
   metadata directory:

   ```bash
   metadata_view=$(mktemp -d)
   cp -a /work/datasets/jump_lite/cpg_release/metadata/. "$metadata_view/"
   cp /work/datasets/jump_lite/cpg_release/README.md "$metadata_view/README.md"

   # Enter a shell that has AWS CLI v2 if needed, then activate temporary keys.
   source cpg_upload/activate_cpg_credentials.sh
   cpg_upload/upload_to_staging.sh --apply \
     "$metadata_view" workspace/publication_data/2026_jump_lite/metadata/v1.0
   rm -rf "$metadata_view"
   ```

5. Verify metadata, each JPEG XL store, all transformed embedding prefixes, and
   the Zstd uploader's final report. Record counts, byte totals, validation
   digest, and verification timestamps in the handoff note.
6. Commit and push only the intended scripts/docs. Preserve unrelated working
   tree files. Tag the release only after the deposited README and manifest
   match staging.
7. Notify the CPG maintainer with the staging prefix, version, canonical digest,
   validation report, object counts, final sizes, and known upstream issues.
   Promotion to the public bucket is a maintainer action.
8. After every final verification is recorded, disable completed one-shot
   services so a future login does not launch another no-op scan:

   ```bash
   systemctl --user stop jump-lite-cpg-upload.service \
     jump-lite-zstd-rebuild.service jump-lite-zstd-upload.service \
     jump-lite-staging-verify.service
   systemctl --user disable jump-lite-cpg-upload.service \
     jump-lite-zstd-rebuild.service jump-lite-zstd-upload.service \
     jump-lite-staging-verify.service
   ```

Do not modify a promoted version in place. Corrections after promotion require
agreement with the CPG maintainer and normally a new version.

## 12. Prepare a future version

The current scripts intentionally freeze v1.0 paths, counts, and digest. Before
v1.1 or another iteration:

1. Agree on the public CPG layout and version with the maintainer.
2. Generate the proposed site manifest deterministically and review its count,
   source/plate/well totals, and sorted-key digest. Do not reuse v1.0's MQ keys
   accidentally.
3. Create new local build, state, checkpoint, and log namespaces. Never point a
   new version at `cpg_upload_state/v1.0` or `zstd_rebuild/v1.0`.
4. Audit every hard-coded version, count, and digest before any write:

   ```bash
   rg -n 'v1\.0|655_101|655,101|4ea6ea3f' cpg_upload
   ```

   In particular update `run_background_upload.sh`,
   `upload_profiles_to_staging.py`, `upload_status.sh`,
   `rebuild_zstd_from_originals.py`, `zstd_rebuild_status.sh`,
   `upload_zstd_to_staging.py`, `zstd_upload_status.sh`,
   `verify_complete_staging.py`, `validate_release.py`, and both READMEs. A
   future refactor should expose one
   shared release-version/config object instead of duplicating these constants.
5. Revisit the explicit zero-filled TIFF allowlist. Remove entries excluded by
   corrected upstream metadata; never broaden it to accept arbitrary corrupt
   inputs.
6. Run reconciliation and all upload commands in audit/dry-run mode first.
   Review sample local-to-S3 mappings and confirm that the destination contains
   the new version before using `--apply`.
7. Syntax-check and inspect the exact diff:

   ```bash
   .venv/bin/python -m py_compile cpg_upload/*.py
   for script in cpg_upload/*.sh; do bash -n "$script"; done
   git diff --check
   git status --short
   ```

8. Start with metadata or a bounded test prefix where possible, verify it, then
   launch the restart-safe services. Preserve the previous public and local
   release unchanged.

## Deterministic prevention

Site sampling in `prep/build_jl_index.sql` now partitions on the complete
source/batch/plate/well identity and orders by `Metadata_Site` before retaining
at most four sites. This prevents reruns from selecting a different four-site
subset. Compression jobs should additionally start from a clean store whenever
the input manifest changes; skip-existing mode must not be used to merge data
from different site manifests.
