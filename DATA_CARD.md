# JUMP-lite data card and responsible-use documentation

Version: JUMP-lite data v1.0. Documentation checked on 2026-10-19.

This document describes provenance, intended use, limitations, access, and
maintenance. It provides human-readable responsible-use documentation.
[Croissant 1.0 metadata](croissant.json) describe the release's tabular entry points.

## Purpose and intended use

JUMP-lite supports research on representations of five-channel Cell Painting
images, image compression, phenotypic activity, and retrieval of annotated
biological relationships. It combines a selected JUMP cohort with lossless and
lossy image derivatives and derived representations.

The benchmark is annotation-enriched. It is not a random or representative sample
of all JUMP phenotypes, cell types, acquisition conditions, compounds, or patients.
Retrieval scores are research measurements, not evidence of clinical efficacy,
safety, a causal mechanism, or a treatment recommendation. Clinical decisions,
patient risk prediction, and biological claims outside the measured assay require
separate data, validation, and appropriate oversight.

## Origin, selection, and processing

The source images are public JUMP Cell Painting TIFFs in the
[Cell Painting Gallery](https://registry.opendata.aws/cellpainting-gallery/).
The deposited site index retains the exact five source URLs for each site.
JUMP-lite adds compressed derivatives; it does not replace the source images or
collect new participant data. The released metadata describe acquisitions,
experimental wells, perturbations, and annotations rather than clinical records.
We do not claim an independent audit of all upstream consent or ethics approvals;
users should consult the source dataset publications for the original acquisition
and governance details.

Available CRISPR and ORF perturbations were retained after documented quality and
cohort filters. Compound candidates were linked to MOTIVE's compound-compound or
compound-gene graphs, then filtered by plate quality, available profiles, and
replication within compound groups. RefChemDB was joined downstream for target
annotations and evaluation. Consequently, MOTIVE-based selection and MOTIVE
retrieval are not independent, and annotation availability affects what can be
evaluated. RefChemDB coverage is not a compound-inclusion requirement.

The exact release has six sources, 30 batches, and 551 plates. At most four sites
per well were retained. Source-specific TIFF links, the frozen site-key digest,
and validation records allow users to recover the selected acquisition identities.
The [preparation documentation](prep/README.md), [site-selection SQL](prep/build_jl_index.sql),
and [dataset specification](cpg_upload/JUMP_LITE_README.md) describe the process.

Image variants are lossless Zstd and JPEG XL HQ, MQ, and D20. Their JPEG XL distances
are 1, 3, and 20, respectively, with effort not explicitly set in the release
producer. Target-2 pilot settings are separate; in particular its D20 used effort
2. HQ and MQ preserve more of the evaluated signal than the aggressive D20 setting,
but no setting is guaranteed to preserve all fine image structure or downstream
tasks. Prefer the original TIFFs or lossless reference when loss is unacceptable.

## Counts and coverage

| Unit | Count | Definition |
|---|---:|---|
| Image sites | 655,101 | Selected five-channel fields of view |
| Wells | 163,776 | 163,773 wells have four sites; three have three sites |
| Treatments | 24,347 | 3,775 compounds, 7,974 CRISPR symbols, 12,598 ORF symbols |
| Control identities | 9 | One compound, three CRISPR, and five ORF control identities |
| Modality-specific identities | 24,356 | Treatments plus controls; the same gene in CRISPR and ORF counts separately |
| Distinct JCP reagent IDs | 26,884 | Distinct `Metadata_JCP2022` values across all wells |
| Sum of subgroup JCP counts | 27,710 | 7,977 CRISPR + 15,131 ORF + 1,187 Diversity + 3,415 Bioactive |
| Mask-covered sites | 632,672 | Frozen segmentation subset in 158,168 four-site wells |
| Mask arrays | 2,530,688 | Cell and nuclei labels for both lossless and MQ inputs |

Multiple ORF reagent identifiers map to the same gene: 15,131 ORF JCP IDs map to
12,603 ORF gene/control identities. There are 826 compound IDs in both compound
groups, so their subgroup sum of 4,602 represents 3,776 distinct compounds.
The 27,710 subgroup sum must not be interpreted as a globally distinct count.
[Verified count queries](reproducibility/validation/dataset-counts-20261019.json)
record the public input hash and SQL.

The mask subset leaves 22,429 image sites without released masks. Consult the mask
manifest for exact membership instead of silently intersecting cohorts. Mask
coverage and all-site object counts do not imply complete morphological feature
extraction. Cell and nuclei instance labels are independent and cannot be matched
by label number across object types, sites, or codecs.

The 16 published embedding collections each contain 655,101 site files. They cover
lossy variants, not lossless Zstd embeddings. Files retain tile-level measurements
and require an explicit aggregation and normalization protocol for well-level
analysis. The separate Target-2 package has its own 9,216-site universe and openly
records incomplete codec coverage. Full non-lite extraction and ongoing matched
cp_measure production are separate work, not additional completed v1.0 artifacts.

## Known limitations and data-quality notices

- **Annotation and acquisition bias.** Selection emphasizes annotated treatments.
  Source, plate, and batch effects may influence representations. Cluster occupancy
  does not establish that JUMP-lite reproduces the full phenotypic distribution.
- **Pretraining exposure and comparisons.** MorphEM and OpenPhenom have documented
  exposure to JUMP-derived images. Excluding matched wells or plates is a sensitivity
  check, not proof of leakage-free training. Published Raw CellProfiler references
  aggregate more sites per well than the learned models. Exact middle-model ranks
  are not consistently resolved by the reported uncertainty analyses.
- **Compression and segmentation.** Compression may remove information relevant to
  untested tasks. Automated mask agreement with lossless-derived masks is not manual
  ground-truth accuracy. Empty, failed, and absent artifacts must remain distinct.
- **Repaired images and unresolved derivative provenance.** The September audit found
  119 MQ arrays with metadata but missing payloads, which ordinary Zarr reads could
  silently replace with zeros. All 119 public payloads were repaired and hash-verified
  again on October 19. The audited mask manifest reports zero MQ cell and nuclei
  objects at all 113 affected sites with mask records, while the corresponding
  lossless masks contain 135 to 665 cells and 122 to 740 nuclei per site. These MQ
  masks must not be treated as validated segmentations of the repaired images.
  Historical input bytes and the effects on published embeddings remain unresolved.
  The [affected-site inventory](cpg_upload/mq_repaired_sites_20261001.csv) and
  [mask coverage evidence](reproducibility/validation/mq-derived-coverage-20261019.json)
  support explicit sensitivity checks. Do not silently exclude these sites, treat
  nonzero embeddings as proof of correct inputs, or substitute regenerated outputs
  without recording a new version and provenance.

The [public release audit](cpg_upload/PUBLIC_RELEASE_AUDIT.md) distinguishes complete
key-inventory checks from bounded content checks. Presence of files and matching
sizes alone do not establish scientific validity. Deposited documentation still
contains some stale paths; use this repository's corrected access table and
[public path map](cpg_upload/PUBLIC_PATHS.md). The deposited files remain unchanged.

## Access and reproducibility

No AWS account or credentials are needed to read the public release. Start with
the small metadata bundle described in the [README](README.md#access).
[Croissant metadata](croissant.json) contain exact public URLs, byte sizes, SHA-256
values, schemas, and references for five Parquet tables and their manifest. They
do not enumerate millions of image or feature files. After downloading the metadata
bundle, verify its hashes, fields, and count queries with the following command
(requires DuckDB):

```bash
python scripts/validate_release_documentation.py --metadata-dir jump_lite_metadata
```

The [Croissant validation record](reproducibility/validation/croissant-20261019.json)
reports zero validation errors and successful record extraction from all five
tables. It retains the recommended-field warning for the unknown publication date
rather than inventing a date.

The [reproduction guide](REPRODUCE.md) describes the bounded dataset smoke test,
figure registry, frozen artifact versions, and external inputs. Preserve the
manifest, model/checkpoint, channel order, compression setting, source hashes,
cohort exclusions, normalization fit population, and evaluation split when making
comparisons. Technical reproducibility does not remove selection bias or prove
biological generalization.

## Licensing and attribution

The [Cell Painting Gallery registry](https://registry.opendata.aws/cellpainting-gallery/)
lists its data under CC0 1.0 and asks users to cite each source dataset publication.
Repository code is under [MIT](LICENSE). This documentation does not relicense
upstream model weights, software, or external annotation databases; consult their
own licenses and terms. Cite the JUMP dataset, the JUMP-lite manuscript, and the
models and annotation resources used in an analysis.

## Maintenance and reporting

Report reproducibility or data-quality problems through
[repository issues](https://github.com/afermg/JUMP_lite/issues), including release
version, site key, codec/model, expected and observed behavior, and hashes when
available. Do not post credentials or sensitive information. Contact information
for the authors is in the manuscript; issue reporting does not imply a guaranteed
response time or indefinite hosting commitment.

Corrections should preserve the frozen cohort and historical evidence, or clearly
introduce a new version when either changes. This documentation update does not
upload or replace CPG objects. Repository documentation records corrected access
paths without modifying deposited metadata or checksums. Source files, model
outputs, and partial failures must not be silently overwritten to make counts
appear complete.
