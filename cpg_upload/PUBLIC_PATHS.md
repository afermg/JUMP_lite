# Verified public JUMP-lite access paths

Checked anonymously on 2026-10-19. No uploads or replacements are made by this
repository documentation update. Deposited metadata and hashes remain unchanged.

## Start with the release index

The direct data entry point is the
[655,101-site index](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/workspace/publication_data/2026_jump_lite/metadata/v1.0/jump_lite_site_index.parquet).
It contains the exact source, batch, plate, well, site, and five original TIFF URLs.

[Croissant metadata](../croissant.json) provide direct URLs and SHA-256 hashes for
all five metadata tables and their manifest. Access requires no credentials.

## Image stores

The following links resolve to the image stores' root metadata, not a single
archive containing all images. Use the site index and a compatible Zarr reader to
retrieve site arrays beneath each root.

| Variant | Root metadata URL |
|---|---|
| Lossless Zstd | [zarr.json](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/images/jump_lite_v1.0/images_compressed/zstd.zarr/zarr.json) |
| HQ | [.zgroup](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/images/jump_lite_v1.0/images_compressed/jpegxl_lossy_hq.zarr/.zgroup) |
| MQ | [.zgroup](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/images/jump_lite_v1.0/images_compressed/jpegxl_lossy_mq.zarr/.zgroup) |
| D20 | [.zgroup](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/images/jump_lite_v1.0/images_compressed/jpegxl_lossy_d20.zarr/.zgroup) |

For example, the MQ payload for site
`source_13__20220914_Run1__CP-CC9-R1-03__O02__1` is available at
[its repaired chunk URL](https://cellpainting-gallery.s3.amazonaws.com/cpg0016-jump/source_all/images/jump_lite_v1.0/images_compressed/jpegxl_lossy_mq.zarr/source_13__20220914_Run1__CP-CC9-R1-03__O02__1/0.0.0).
Its `.zarray` and `.zattrs` are in the same site directory.

## Other components

These are S3 key prefixes below `s3://cellpainting-gallery/cpg0016-jump/source_all/`;
angle-bracket fields are placeholders, not literal URLs.

| Component | Published prefix or key pattern |
|---|---|
| Embeddings | `workspace_dl/embeddings/<model>-<codec>/jump_lite/v1.0/<source>/<batch>/<plate>/<well>-<site>/embedding.parquet` |
| Metadata | `workspace/publication_data/2026_jump_lite/metadata/v1.0/` |
| Cell/nuclei masks | `workspace/publication_data/2026_jump_lite/segmentation/2026_jump_lite_v1.0/{cell_masks,nuclei_masks}/{zstd,jpegxl_lossy_mq}.zarr/` |
| Target-2 artifacts | `workspace/publication_data/2026_jump_lite/target_2/v1.0/` |

Model/codec combinations are listed in [the dataset specification](JUMP_LITE_README.md).
The 16 embedding collections cover lossy image variants; no lossless embedding
collection is included in this release.

## Historical paths in deposited documents

The deposited README and segmentation manifest still contain older image and
embedding paths. Apply this access-path mapping without modifying the manifest:

| Historical value | Actual public location |
|---|---|
| `images/2026_jump_lite_v1.0/images_compressed/` | `images/jump_lite_v1.0/images_compressed/` |
| `workspace_dl/embeddings/<model>-<codec>/2026_jump_lite_v1.0/` | `workspace_dl/embeddings/<model>-<codec>/jump_lite/v1.0/` |

Do **not** replace `2026_jump_lite_v1.0` globally; it is the correct segmentation
package directory. The mapping corrects access locations, not scientific source
provenance for the repaired MQ sites. That separate limitation remains in the
[release audit](PUBLIC_RELEASE_AUDIT.md).

## Verification scope

The [October 19 receipt](../reproducibility/validation/cpg-public-20261019.json)
records all seven metadata file hashes, four image root markers, one object in
each embedding and mask collection, and every one of the 119 repaired MQ chunks.
The [September 29 audit](PUBLIC_RELEASE_AUDIT.md) records the complete original
key inventories and bounded content tests. Root availability and sample checks
alone are not a complete new release inventory or validation of every model output.
