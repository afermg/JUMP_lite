# JUMP-Lite

**JUMP-Lite** is a compact, analysis-ready subset of the [JUMP Cell Painting dataset](https://registry.opendata.aws/cellpainting-gallery/). It combines compressed five-channel microscopy images with per-site deep-learning embeddings, Cellpose cell and nuclei instance masks, exact links to the original TIFFs, perturbation metadata, curated RefChemDB annotations, and publication artifacts for the four-plate Target-2 compression analysis.

> **Release status:** JUMP-Lite v1.0 has been deposited in the Cell Painting Gallery (CPG) and is awaiting gallery promotion. The public paths below will be available by mid-September 2026 at the latest.

## Dataset at a glance

|                             |                              v1.0 |
|-----------------------------|----------------------------------:|
| JUMP sources                |                                 6 |
| Batches                     |                                30 |
| Plates                      |                               551 |
| Wells                       |                           163,776 |
| Benchmark perturbations     |                            24,356 |
| Image sites                 |                           655,101 |
| Mask-covered sites          |                           632,672 |
| Cell/nuclei mask arrays     |                         2,530,688 |
| Channels                    |        5: AGP, DNA, ER, Mito, RNA |
| Compressed image variants   |              4: Zstd, HQ, MQ, D20 |
| Lossless Zstd               |              4.8 TB (4.4 TiB) |
| JPEG XL HQ                  |          237.7 GB (221.4 GiB) |
| JPEG XL MQ                  |            92.0 GB (85.7 GiB) |
| JPEG XL D20                 |            16.2 GB (15.1 GiB) |
| Embedding variants          |                                16 |
| Per-site embedding Parquets |                        10,481,616 |
| Curated annotation rows     | 29,142 across 1,526 perturbations |

The release covers `source_2`, `source_4`, `source_6`, `source_7`, `source_8`, and `source_13`, with at most four sites per well. Benchmark perturbations are defined by modality and biological entity, so ORF overexpression and CRISPR knockout of the same gene count separately.

## Access

Public root:

```text
s3://cellpainting-gallery/cpg0016-jump/source_all/
```

| Component | Path below the public root |
|---|---|
| Compressed images | `images/2026_jump_lite_v1.0/images_compressed/<codec>.zarr/` |
| Metadata and annotations | `workspace/publication_data/2026_jump_lite/metadata/v1.0/` |
| Cell and nuclei instance masks | `workspace/publication_data/2026_jump_lite/segmentation/2026_jump_lite_v1.0/{cell_masks,nuclei_masks}/{zstd,jpegxl_lossy_mq}.zarr/` |
| Target-2 masks, object features, and compact profiles | `workspace/publication_data/2026_jump_lite/target_2/v1.0/` |
| Per-site embeddings | `workspace_dl/embeddings/<model>-<codec>/2026_jump_lite_v1.0/` |

Start by downloading the small metadata bundle; AWS credentials are not required:

```bash
aws s3 cp --no-sign-request --recursive \
  s3://cellpainting-gallery/cpg0016-jump/source_all/workspace/publication_data/2026_jump_lite/metadata/v1.0/ \
  jump_lite_metadata/
```

## What is included

- **Images:** a 4.8 TB (4.4 TiB) lossless Zarr v3 store (`zstd`) plus Zarr v2 JPEG XL stores at high quality (`jpegxl_lossy_hq`, 237.7 GB), medium quality (`jpegxl_lossy_mq`, 92.0 GB), and high compression (`jpegxl_lossy_d20`, 16.2 GB). Each site is one unsigned 16-bit `(channel, y, x)` array. Reading JPEG XL arrays requires a compatible NumCodecs implementation such as `imagecodecs`.
- **Segmentation masks:** 2,530,688 losslessly encoded Zarr v3 arrays provide Cellpose cell and nuclei instance labels for 632,672 sites under both lossless Zstd and MQ inputs. Every mask is `uint16` with shape `(1, height, width)`. Cell and nuclei labels are independent: label IDs are not paired across object types, sites, or codecs.
- **Embeddings:** per-site long-form Parquets from DINOv2, randomly initialized DINOv2, MorphEm, OpenPhenom, and two SubCell input variants. These are site-level outputs, not aggregated well profiles.
- **Metadata:** `jump_lite_site_index.parquet` is the primary index and links every compressed site to its five original JUMP TIFF URLs. The release also provides a channel-level image index, perturbation metadata, a plate manifest, and a machine-readable manifest.
- **Annotations:** `jump_lite_refchem_annotations.parquet` contains the release-relevant RefChemDB/JUMP confidence matches, including targets, genes, activity fields, confidence tiers, and direction-match indicators.
- **Target-2 artifacts:** 163,786 cell/nuclei instance-mask NPZ files (4.304 GB), 86,989 per-site object-level `cp_measure` Parquets (253.990 GB), and 66 canonical compact raw-feature Parquets (506.6 MB) from the four-plate compression subanalysis. Separate manifests record exact codec coverage, mask availability, shapes, empty files, schema and site-set digests, checksums, recorded mask scripts, and a release-time profile-producer code snapshot; partial variants are labeled rather than silently intersected.

Site identifiers have the form:

```text
<source>__<batch>__<plate>__<well>__<site>
```

The original TIFFs remain in their source-specific `cpg0016-jump/source_<n>/` locations and are not duplicated in JUMP-Lite.

## Provenance

The exact 655,101-site paper benchmark cohort is frozen across all four image variants. It contains 163,773 wells with four available sites and three wells with three available sites. The embedding collections share this same site set for the lossy image variants they cover.

## How JUMP-Lite was generated

```text
public JUMP TIFFs
  → plate filtering and deterministic site selection
  → lossless Zstd and JPEG XL site-major Zarr arrays
  → Cellpose cell/nuclei masks for lossless Zstd and MQ
  → model-specific tiling and per-site embeddings
  → frozen metadata and annotation tables
  → cross-variant validation and CPG deposit
```

The generation and analysis resources remain part of this repository:

- [Bootstrap and source-data preparation](prep/README.md), including the [deterministic site-selection query](prep/build_jl_index.sql)
- [Validated release image compressor](src/compress_tif_release.py) and [embedding driver](prep/aliby_featurize.py)
- [Release metadata, validation, and CPG layout](cpg_upload/README.md)
- [Complete dataset specification and model provenance](cpg_upload/JUMP_LITE_README.md)
- [Target-2 masks, object-feature, and compact-profile artifact specification](cpg_upload/TARGET2_ARTIFACTS_README.md)
- [Technical analysis pipeline](PIPELINE.md) and [paper reproduction guide](REPRODUCE.md)

## Reproducibility

Run `nix develop` followed by `just --list` to inspect the generation and
reproduction recipes. The bounded source-stratified dataset check validates the
complete 655,101-site release index and reproduces one five-channel site from
each of the six release sources under all four published image codecs:

```bash
just dataset-smoke jump_lite_metadata/jump_lite_site_index.parquet
```

It writes only below `data/generated/dataset-smoke/` and refuses existing or
symlinked output paths. Maintainers can use `just dataset-smoke-local` to compare
every generated Zarr metadata/chunk file with canonical release stores. The
2026-09-01 public-download and local-reference validations each reproduced
24/24 sampled arrays byte-for-byte and verified six lossless Zstd round trips;
their portable hashes and environment are retained
in [`reproducibility/validation/dataset-smoke-20260901.json`](reproducibility/validation/dataset-smoke-20260901.json).

For figures and tables, `just artifacts-list` inventories active-paper and
supporting bundles, `just artifacts-verify` performs non-mutating checksum and
provenance verification, and `just paper-artifacts-verify` checks the exact 39
figures and three generated tables in the final manuscript. Full commands,
inputs, safety rules, known external snapshots, and validation scope are in
[REPRODUCE.md](REPRODUCE.md).

## Citation

Please cite the JUMP-Lite manuscript:

> Muñoz AF, Fredin Haslum J, Shen R, Carpenter AE, Singh S. (2026). **JUMP-lite: Compact, reproducible benchmarking of cell representations.** [arXiv:2608.07632](https://arxiv.org/abs/2608.07632).

Release indices and related tables are archived at [Zenodo](https://doi.org/10.5281/zenodo.18705140).

## License

Repository code is available under the [MIT License](LICENSE). Data usage is governed by the terms of the Cell Painting Gallery and the original JUMP dataset.
