# Public CPG release audit

## Current status, 2026-10-19

The public image repair is complete. All **119 previously missing MQ payloads** were regenerated from the original TIFFs, promoted, and downloaded again on October 19. Every payload matches its repair SHA-256 and size, totaling **13,947,011 bytes**. The affected keys and hashes are in [mq_repaired_sites_20261001.csv](mq_repaired_sites_20261001.csv).

The fresh bounded check also verified all seven metadata hashes against September 29, four image root markers, and one listed object from each of 16 embedding and four mask collections. All 24 entry points are accessible. Evidence: [cpg-public-20261019.json](../reproducibility/validation/cpg-public-20261019.json). This is not a fresh full-release listing or a scientific revalidation of all derived outputs.

The repair closes the missing-public-image issue, but **derived embedding/mask input provenance remains unresolved** for the affected sites. The audited manifest contains masks for 113 of them: every affected MQ cell and nuclei mask has zero reported objects, whereas the matching lossless masks contain 135 to 665 cells and 122 to 740 nuclei. Those MQ masks are not validated segmentations of the repaired images. The [manifest intersection evidence](../reproducibility/validation/mq-derived-coverage-20261019.json) records the exact query and source hash. Nonzero embedding values alone cannot establish which pixels were used. The local canonical MQ store still lacks all 119 chunks and was not modified by the public repair. The [data card](../DATA_CARD.md) records these limitations.

The deposited README and mask image-lineage paths remain stale. No further CPG uploads or replacements are planned. The repository [public path map](PUBLIC_PATHS.md) records corrected access locations while preserving deposited metadata and hashes. Correct public paths are also listed below. The remainder of this report preserves the original September 29 findings as history.

## Original audit, 2026-09-29

The v1.0 release is public in `s3://cellpainting-gallery/cpg0016-jump/source_all/`.
All remote checks used anonymous HTTPS requests against the **public** bucket,
not staging credentials or upload checkpoints. No CPG objects were changed.

Machine-readable results: [cpg-public-20260929.json](../reproducibility/validation/cpg-public-20260929.json).

At the original audit, the documentation push was held because 119 missing MQ chunks contradicted the release's coverage claim. Those payloads have since been repaired as described above; the historical observation below is retained for provenance.

## Historical data discrepancy: 119 missing MQ chunks

All four image stores have metadata for the 655,101 frozen sites, but MQ has
only **654,982 image chunks**. Exactly 119 `source_13` site directories contain
`.zarray` and `.zattrs` but no `0.0.0` payload. All 119 chunk URLs returned HTTP
404; their public array metadata specifies `fill_value: 0`, so ordinary Zarr
reads silently produce all-zero five-channel images instead of an error.

The complete public listing's key-set digest matches the expected site set
minus these exact 119 keys. The same 119 chunks are absent in the local canonical
MQ store, so this is not merely a public-promotion mismatch. Three affected
sites were downloaded from public MQ, Zstd, and HQ stores: MQ decoded to zeros,
while Zstd/HQ contained nonzero pixels throughout. For example:

```text
source_13__20220914_Run1__CP-CC9-R1-03__O02__1
```

Its Zstd pixels range from 53 to 23,162; its MQ read is entirely zero.
The JSON receipt's `mq_missing_chunks.affected_sites` lists all 119 keys, public
metadata hashes, shapes, and HTTP results. There are also published mask records
for 113 of these sites under both codecs. The cause and downstream embedding or
mask provenance impact are **not established**; do not silently rebuild or
replace promoted data without investigation and maintainer coordination.

All other audited image/embedding/mask key inventories match; segmentation
manifest byte totals and every Target-2 data object's key/size also match.
Observed image-store totals, including group metadata, are:

| Store | Objects | Bytes |
|---|---:|---:|
| Zstd | 1,310,203 | 4,812,456,031,773 |
| HQ | 1,965,305 | 237,706,877,551 |
| MQ (119 missing chunks) | 1,965,186 | 92,032,255,077 |
| D20 | 1,965,305 | 16,235,617,193 |

Thus the README's rounded byte sizes match what is public, but the MQ size does
not establish complete image coverage.

## Correct public paths

| Component | Path below `cpg0016-jump/source_all/` |
|---|---|
| Images | `images/jump_lite_v1.0/images_compressed/<codec>.zarr/` |
| Embeddings | `workspace_dl/embeddings/<model>-<codec>/jump_lite/v1.0/<source>/<batch>/<plate>/<well>-<site>/embedding.parquet` |
| Metadata | `workspace/publication_data/2026_jump_lite/metadata/v1.0/` |
| JUMP-Lite masks | `workspace/publication_data/2026_jump_lite/segmentation/2026_jump_lite_v1.0/` |
| Target-2 artifacts | `workspace/publication_data/2026_jump_lite/target_2/v1.0/` |

The previously documented image and embedding paths using
`2026_jump_lite_v1.0` are empty in the public bucket. That name **is correct**
for the segmentation package; it must not be replaced globally.

## Validation scope

- Downloaded all seven metadata objects and verified their recorded sizes.
  Recomputed the 655,101-site digest
  `4ea6ea3f5457c33a1412a80a89d8696d4f8e77474cf449e75db7ce6ba98685e2`,
  all source/batch/plate/well counts, the five-channel image-index expansion,
  24,356 modality-specific perturbations, and 29,142 annotations across 1,526
  perturbations. The site-index file matches the committed smoke-test hash.
- Recursively listed all four image stores and all 16 embedding collections,
  comparing exact object-key sets to the public site index, partitioned by
  source. This checks metadata **and** chunk coverage, not just root markers.
- Recursively listed all four segmentation stores against the public mask
  manifest's site sets and byte totals. Its 632,672 sites are all in the image
  release; 22,429 image sites have no masks, as implied by the README's counts.
- Compared every Target-2 data object's key and byte size to its public manifest:
  163,786 masks, 86,989 object-feature files, and 66 compact profiles.
- Downloaded and decoded 24 source-stratified image arrays (six sources × four
  codecs). Every metadata/chunk file matches the committed
  [smoke-test attestation](../reproducibility/validation/dataset-smoke-20260901.json)
  byte-for-byte by SHA-256. Read 96 embedding Parquets (six sources × 16
  variants) and confirmed the documented long-form columns.
- Verified segmentation and Target-2 manifest hashes against their published
  provenance. Downloaded and decoded 24 mask arrays (six sources × four stores),
  80 Target-2 NPZ masks, 40 Target-2 object-feature Parquets, and **all 66** compact
  profiles, checking content hashes, sizes, shapes, dtypes, or row/column counts
  as applicable.

This is a complete **key-inventory** audit with bounded content checks, not a
rehash or scientific revalidation of every multi-terabyte image/embedding
payload. Counts and sizes are observations at the audit date.

Repository checks also passed: `just --list`, `just artifacts-list`, and
`just artifacts-verify` (10 verified, seven explicitly skipped external or
machine-local bundles). The manuscript lock contains 39 figures and three
tables; this audit did not rerun full manuscript reproduction.

## Discrepancies and follow-up

1. **Repository access paths and release-status caveat:** corrected in
   local drafts of `README.md` and `JUMP_LITE_README.md`, with a new warning
   about the MQ data gap. The dataset specification also
   incorrectly described public embedding filenames as site-key stems; the
   actual filename is `embedding.parquet`, with identity in its parent path.
2. **Deposited metadata README is stale:** the public
   `metadata/v1.0/README.md` is the older 12,791-byte document. It still gives
   the nonexistent image/embedding paths and omits the now-public JUMP-Lite
   segmentation package. Its recorded size in `metadata_manifest.json` is
   internally consistent, but it is not the current repository specification.
3. **Segmentation image-lineage paths are stale:** the public mask manifest's
   `source_image_store` column names
   `images/2026_jump_lite_v1.0/images_compressed/{zstd,jpegxl_lossy_mq}.zarr`.
   The corresponding public stores are under `images/jump_lite_v1.0/`.
   The mask object paths themselves are correct. Any manifest correction must
   also update its provenance hash; do not silently rewrite it in place.
4. **Upload tooling is not a public-path reference:** `cpg_upload/README.md`,
   `run_background_upload.sh`, `upload_profiles_to_staging.py`,
   `upload_zstd_to_staging.py`, and `segmentation_zarr.py` retain historical
   staging destinations or lineage paths. Coordinate their eventual update
   with the CPG maintainer, including checkpoint identities. Do not rerun
   uploaders merely to make public documentation match.

The historical MQ payload discrepancy has been repaired. Repository documentation now supplies the corrected access paths without modifying CPG objects. The derivative-provenance investigation remains open.
