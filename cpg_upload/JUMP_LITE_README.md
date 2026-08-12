# JUMP-Lite

JUMP-Lite is a compact, source-spanning subset of the JUMP Cell Painting
dataset. It provides lossless and lossy-compressed multichannel image arrays together with
per-site deep-learning feature Parquets, image provenance metadata, perturbation
metadata, and curated RefChemDB-derived annotations.

The original TIFF images are already public under `cpg0016-jump`. This release
is deposited as an aggregated multi-source subset beneath
`cpg0016-jump/source_all`; it adds compressed derivatives and links every
compressed site back to its five original TIFF URLs.

## Dataset scope

| Item | Count |
|---|---:|
| JUMP sources | 6 |
| Batches | 30 |
| Plates | 551 |
| Wells with images | 163,776 |
| Modality-specific perturbations | 24,356 |
| Sites | 655,101 |
| Channels per site | 5 |
| Original image references | 3,275,505 |

Included sources are `source_2`, `source_4`, `source_6`, `source_7`, `source_8`,
and `source_13`.

For benchmark accounting, a perturbation is the combination of modality and
biological entity (gene or compound). The release contains 3,776 compound
identities, 7,977 CRISPR-targeted genes, and 12,603 ORF-overexpressed genes, for
24,356 modality-specific perturbations. ORF overexpression and CRISPR knockout
of the same gene are distinct because these opposing interventions can produce
different cellular phenotypes.

## Cell Painting Gallery layout

The release is stored as loose, directly addressable CPG objects under:

```text
cpg0016-jump/source_all/
├── images/
│   └── 2026_jump_lite_v1.0/
│       └── images_compressed/
│           ├── zstd.zarr/
│           ├── jpegxl_lossy_hq.zarr/
│           ├── jpegxl_lossy_mq.zarr/
│           └── jpegxl_lossy_d20.zarr/
├── workspace/
│   └── publication_data/
│       └── 2026_jump_lite/
│           └── metadata/
│               └── v1.0/
│                   ├── README.md
│                   ├── jump_lite_site_index.parquet
│                   ├── jump_lite_image_index.parquet
│                   ├── jump_lite_perturbation_metadata.parquet
│                   ├── jump_lite_refchem_annotations.parquet
│                   ├── jump_lite_plate_manifest.parquet
│                   └── metadata_manifest.json
└── workspace_dl/
    └── embeddings/
        └── <model>-<image-codec>/
            └── 2026_jump_lite_v1.0/
                └── <source>/
                    └── <batch>/
                        └── <plate>/
                            └── <well>-<site>/
                                └── embedding.parquet
```

The embedding feature-set identifiers combine model and image input, for
example `dinov2-jpegxl_lossy_mq` and
`openphenom-jpegxl_lossy_d20`. Public model labels are `dinov2`,
`dinov2_random`, `morphem`, `openphenom`, `subcell`, and
`subcell_clip01`.

JUMP-Lite is an addition to the existing JUMP project, not a new acquisition.
The original TIFFs and their source-specific `load_data_csv` files remain in
`source_2`, `source_4`, `source_6`, `source_7`, `source_8`, and `source_13`.
They are not duplicated under `source_all`; the deposited site and image
indices retain the original public TIFF URLs.

The release contains the exact well cohort used in the paper after plate,
quality-control, annotation, and replicate filtering. Six negative-control-only
graylisted plates are excluded, leaving 551 plates. At most four sites are
retained per well; 163,773 wells have four sites and three wells have three
available sites. The release freezes these 655,101 site keys across images and
embeddings; no unavailable site is synthesized.

## Compressed images

Each image dataset contains one array per site. The JPEG XL stores use Zarr v2;
the lossless Zstd store uses Zarr v3. Site keys use:

```text
<source>__<batch>__<plate>__<well>__<site>
```

Each site array has dimensions `(channel, y, x)`, uses unsigned 16-bit values,
and stores all five channels in one chunk. Channel order is:

```text
AGP, DNA, ER, Mito, RNA
```

Four compressed image variants are included:

| Dataset | Format | Size | Description |
|---|---|---:|---|
| `zstd.zarr` | Zarr v3, Blosc/Zstd level 9 with bit shuffle | 4.8 TB (4.4 TiB) | Lossless site-major copy of the original TIFF pixels |
| `jpegxl_lossy_hq.zarr` | Zarr v2, JPEG XL distance 1.0 | 237.7 GB (221.4 GiB) | High quality |
| `jpegxl_lossy_mq.zarr` | Zarr v2, JPEG XL distance 3.0 | 92.0 GB (85.7 GiB) | Medium quality and canonical site manifest |
| `jpegxl_lossy_d20.zarr` | Zarr v2, JPEG XL distance 20.0 | 16.2 GB (15.1 GiB) | High-compression comparison variant |

The finalized `zstd.zarr` contains 1,310,203 loose objects and totals exactly
4,812,456,031,773 bytes (4.812 TB; 4.377 TiB).

The corrected cohort excludes the `source_7` sites affected by the malformed
public TIFF objects tracked in
[jump-cellpainting/datasets#177](https://github.com/jump-cellpainting/datasets/issues/177).

The JPEG XL arrays are lossy derivatives and should not be interpreted as
replacing the original JUMP TIFFs. Their decoding requires a Zarr-compatible
registration of the `imagecodecs_jpegxl` codec, such as
`imagecodecs.numcodecs.Jpegxl`. The Zstd arrays were rebuilt directly from the
five original public TIFFs for each frozen site without caching those TIFFs.

## Per-site Parquet outputs

Feature outputs intentionally use one Parquet file per image site. Their file
stem is identical to the corresponding Zarr site key. Each Parquet is in long
form with the columns:

```text
tile, label, branch, metric, value, object, tp
```

The release contains outputs from the following model families and image
compression variants:

| Model family | Image variants |
|---|---|
| DINOv2 ViT-S/14 | MQ, HQ, D20 |
| Randomly initialized DINOv2 ViT-S/14 | MQ, HQ, D20 |
| MorphEm | MQ, HQ, D20 |
| OpenPhenom | MQ, HQ, D20 |
| SubCell | MQ |
| SubCell clipped-input variant | MQ, HQ, D20 |

Broadly, DINOv2 uses AGP/DNA/ER 224-pixel tiles; MorphEm uses all five channels
with 224-pixel tiles; OpenPhenom uses all five channels with 256-pixel tiles,
outlier clipping, and 8-bit conversion; and SubCell uses Mito/ER/DNA/AGP with
448-pixel tiles. The processing code in the JUMP-Lite repository and the
frozen metadata manifest provide the authoritative record of run parameters
and release identity.

## Metadata files

The release metadata are generated from the exact canonical MQ keys rather than
by resampling:

- `jump_lite_site_index.parquet`: one row per compressed site, including source,
  batch, plate, well, site, and the five original JUMP TIFF URLs.
- `jump_lite_image_index.parquet`: tidy expansion with one row per site/channel
  and 3,275,505 total rows.
- `jump_lite_perturbation_metadata.parquet`: 163,776 benchmark wells with JUMP
  identifiers, perturbation type, symbols, and grouping information.
- `jump_lite_plate_manifest.parquet`: per-plate well and site counts.
- `metadata_manifest.json`: counts, channel order, and artifact sizes.

## Annotations

`jump_lite_refchem_annotations.parquet` contains the release-relevant subset of
curated RefChemDB/JUMP confidence matches:

- 29,142 annotation rows
- 1,526 distinct JUMP perturbation identifiers
- target genes, target type, mode and activity fields
- cross-modality and within-modality confidence tiers
- compound/perturbation direction-match indicators

Annotations whose query perturbation is absent from the frozen JUMP-Lite
release are excluded from this deposited table.

## Data integrity

Before upload, fail-closed validation and upload guards require:

1. The frozen manifest to contain exactly 655,101 internally consistent site
   keys with the v1.0 digest.
2. Zstd, MQ, HQ, D20, and every per-site Parquet collection to contain all
   frozen release keys; extra source objects are reported rather than counted
   as release members.
3. Unfiltered image syncs to use already-subsetted source trees, while the Zstd
   and profile uploaders select the exact frozen manifest and verify its digest.
4. The frozen site and image indices to match the canonical keys and contain all
   five original image URLs.
5. Metadata, plate, source, well, and annotation invariants to pass.

Release validation checks object counts and metadata invariants following Cell
Painting Gallery upload guidance.

## Provenance

### Image and site lineage

The release follows this lineage for every site:

```text
original JUMP TIFFs in cpg0016-jump/source_<n>/images/
    ↓ five URLs frozen in jump_lite_site_index.parquet
site-major uint16 array in <image-codec>.zarr/<site-key>
    ↓ model-specific channel selection, preprocessing, and tiling
per-site embedding Parquet in <model>-<image-codec>/2026_jump_lite_v1.0/
```

The JUMP-Lite plate list was filtered against the JUMP redlist and graylist,
and at most four sites per source/batch/plate/well were selected in
`Metadata_Site` order. The exact 655,101-site release manifest is frozen from
the completed MQ store. HQ and D20 contain the same site identities, but their
profile values come from their respective compressed pixels: an HQ profile is
computed from `jpegxl_lossy_hq.zarr`, a D20 profile from
`jpegxl_lossy_d20.zarr`, and an MQ profile from `jpegxl_lossy_mq.zarr`. Profiles
are never substituted across codecs.

Each compressed site array contains the five original channels in
`AGP, DNA, ER, Mito, RNA` order. The profile inputs are:

| Public model label | Model/checkpoint identifier | Input channels | Tile and preprocessing | Uploaded image variants |
|---|---|---|---|---|
| `dinov2` | `facebookresearch/dinov2`, `dinov2_vits14` | AGP, DNA, ER | 224 px | MQ, HQ, D20 |
| `dinov2_random` | `dinov2_vits14`, randomly initialized (`pretrained=False`) | AGP, DNA, ER | 224 px | MQ, HQ, D20 |
| `morphem` | `CaicedoLab/MorphEm` | AGP, DNA, ER, Mito, RNA | 224 px | MQ, HQ, D20 |
| `openphenom` | `recursionpharma/OpenPhenom` | AGP, DNA, ER, Mito, RNA | 256 px, outlier clipping, 8-bit conversion | MQ, HQ, D20 |
| `subcell` | SubCell `mae_contrast_supcon_model`, channels `rybg` | Mito, ER, DNA, AGP | 448 px | MQ |
| `subcell_clip01` | Same SubCell model | Mito, ER, DNA, AGP | 448 px, clipped input | MQ, HQ, D20 |

These are per-site embedding outputs, not well-level profiles. There are
655,101 Parquets in each of 16 model/codec variants, for 10,481,616 Parquets in
total. The lossless Zstd store is a pixel reference and does not have a
corresponding embedding variant in v1.0.

The featurization driver is `prep/aliby_featurize.py`; it dispatches each Zarr
site through Aliby and Nahual model servers. The release-building and validation
implementation is under `cpg_upload/` in:

- <https://github.com/afermg/JUMP_lite>

The index-generation inputs and related JUMP/JUMP-Lite tables are described at:

- Zenodo: <https://doi.org/10.5281/zenodo.18705140>
- Cell Painting Gallery JUMP project:
  <https://registry.opendata.aws/cellpainting-gallery/>

JUMP-Lite is derived from `cpg0016-jump`; users should cite the primary JUMP
Cell Painting dataset and the feature-model publications appropriate to their
use.
