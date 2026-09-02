# Reproducing JUMP-lite figures and data

The repository uses [just](https://just.systems/) as the public command index.
Run `just --list` to see every recipe.

There are four intentionally separate workflows:

1. **Dataset generation smoke test** (minutes): validate the complete release
   index and reproduce a source-stratified set of image arrays.
2. **Artifact verification** (seconds): verify committed analysis bundles and
   the exact active manuscript assets without recomputation.
3. **Post-sweep reproduction** (about 55 minutes): regenerate the original
   paper plots/tables from frozen sweep and evaluation checkpoints.
4. **Full production** (hours to days, GPUs, hundreds of GB): regenerate images,
   features, sweeps, and evaluations. This is never invoked by an artifact
   verification command.

## 1. Environment

The Nix shell supplies `just`, `uv`, `pixi`, system libraries, and the pinned
Python environment:

```bash
nix develop
just --list
```

Recipes intentionally use the runner required by each analysis (`uv run`,
`pixi run`, or the active Python interpreter). They do not recursively enter
`nix develop`; enter the shell once before running them.

## 2. Dataset generation smoke test

The bounded dataset test first validates the complete JUMP-Lite v1.0 index:
655,101 distinct site keys, the locked site-key digest, and the exact counts for
all six release sources. It then selects the lexical middle site within each
source, reads all five original TIFF channels, and regenerates each site's Zstd,
HQ, MQ, and D20 arrays. The result is 30 source TIFF inputs and 24 generated Zarr
site arrays.

With a downloaded metadata bundle, run:

```bash
just dataset-smoke \
  jump_lite_metadata/jump_lite_site_index.parquet \
  data/generated/dataset-smoke/my-run
```

This downloads only the 30 selected public TIFFs through their release-index
URLs. The output root must be new and must remain below
`data/generated/dataset-smoke/`; existing paths, path traversal, and symlinked
output parents are refused.

Maintainers with local source TIFFs and canonical stores can also require exact
file-tree equality:

```bash
just dataset-smoke-local \
  /work/datasets/jump_lite/cpg_release/metadata/jump_lite_site_index.parquet \
  /work/datasets/jump_lite/images/raw/jump_lite \
  /work/datasets/jump_lite/images/compressed/compressed_test/jump_lite_updated \
  /work/datasets/jump_lite/zstd_rebuild/v1.0/zstd.zarr \
  data/generated/dataset-smoke/my-local-run
```

Every run verifies the exact site-index file, the 30 selected source TIFF hashes,
and all 24 generated site trees against the tracked attestation, so the public
download path can fail closed without local canonical stores. The public recipe
was exercised end to end on 2026-09-01: it anonymously downloaded all 30 TIFFs
(71,186,977 bytes), reproduced 24/24 attested site trees, and passed six lossless
round trips. The local validation additionally reproduced all six Zstd arrays
losslessly and matched all 24 canonical site trees byte-for-byte, including Zarr
metadata and compressed chunks. The portable source/output hashes, public-download receipt, and environment are retained in
`reproducibility/validation/dataset-smoke-20260901.json`; the ignored local run
report is `data/generated/dataset-smoke/validation-20260901-v4/reproduction_report.json`.
This source-stratified test covers every release source, channel, and published
image codec, but it is a structural smoke test rather than a statistically
representative biological sample or a replacement for whole-release validation.

The test exercises the same `src/compress_tif_release.py` implementation exposed
by the full compression recipes. The historical `src/compress_tif.py` remains
unchanged because accepted effort-sensitivity provenance locks its source bytes.
The release CLI validates the six-part TIFF filename, groups
by the five-part site key, requires exactly the canonical AGP/DNA/ER/Mito/RNA
channel inventory, supports the release D20 setting, stages each site before an
atomic publish, validates existing arrays before skipping them, and exits
nonzero if any site fails. Only the explicitly destructive `--overwrite` option
can replace an existing codec store; use fresh output directories near canonical
data.

## 3. Artifact registry

List every active-paper and supporting bundle without accessing external data:

```bash
just artifacts-list
```

Verify every locally available committed bundle:

```bash
just artifacts-verify
# or one bundle
just artifacts-verify mq-d2e8-synthesis
```

Bundles that still depend on separately retained immutable production snapshots
are reported explicitly as `SKIP`; they are never silently treated as verified.
`artifacts.toml` records each bundle's producer, scope, runtime class, reference
root, and safe-regeneration status. Run the focused artifact and dataset-generation tests with:

```bash
uv run pytest -q tests/test_manage_artifacts.py tests/test_dataset_reproduction.py
```

Run the complete repository suite with the source roots exposed explicitly:

```bash
PYTHONPATH=.:src uv run pytest -q tests
```

The final 2026-09-01 validation passed 87 tests and 14 subtests.

Regenerate one managed bundle into an isolated, ignored destination:

```bash
just artifacts-regenerate target-overlap
just artifacts-regenerate strict-heldout
just artifacts-regenerate mq-d2e8-synthesis
```

The destination is `data/generated/artifacts/<bundle>/`. Regeneration refuses an
existing destination, symlinked output parents, paths outside that root, and
unmanaged/heavy bundles. It never overwrites committed results or manuscript
assets.

### Exact final-manuscript verification

`paper_artifacts.lock.json` records all 39 active graphics and three generated
TeX tables in final manuscript commit
`20a1fdaf32a425b6e20c1e18fa12bbf193405518`. Verify an exact checkout with:

```bash
just paper-artifacts-verify /path/to/jump-lite-wacv-manuscript
```

This checks the manuscript commit, complete active `\\includegraphics`
inventory, byte sizes, and SHA-256 hashes. It distinguishes computed artifacts,
authored SVG/PNG diagrams, and frozen example-image sources.

## 4. Post-sweep reproduction

The lightweight path reuses canonical checkpoints. It must not regenerate the
approximately 968--976 GB JUMP-lite normalization sweep.

### Required staged paths

Stage real directories or symlinks at these repository-relative paths:

```text
data/intermediate/sweep_v11_lite/                 JUMP-lite v11 sweep
data/features/variance_first_v11/                 Target-2 v11 sweep
data/intermediate/motive_eval/large_strict/       production strict MOTIVE evaluation
data/intermediate/segmentation_comparison/
  instance_mappings/
  detailed_results/
data/intermediate/image_quality/quality_metrics.csv
data/intermediate/saturation_proper/              persisted saturation CSVs
metadata/metadata_dataset_filtered_4reps.parquet
```

Canonical sweep archives are read-only inputs. The known local archive roots are:

```text
/work/datasets/JUMP-lite-wacv/sweeps/variance_first_v11_lite/
/work/datasets/JUMP-lite-wacv/sweeps/MAIN_RESULTS__figure_4_variance_first_v11/
```

The production strict MOTIVE tree has historically lived in a collaborator
checkout and still needs a stable public archive. Do not substitute the reduced
60-file public smoke tree for the 1,055-file production top-config tree.

Example staging:

```bash
mkdir -p data/intermediate data/features
ln -s /work/datasets/JUMP-lite-wacv/sweeps/variance_first_v11_lite \
  data/intermediate/sweep_v11_lite
ln -s /work/datasets/JUMP-lite-wacv/sweeps/MAIN_RESULTS__figure_4_variance_first_v11 \
  data/features/variance_first_v11
# Stage the remaining validated checkpoints at the paths above.
```

Preflight first:

```bash
just reproduce-check
```

The preflight requires 7,285 JUMP-lite sweep metrics, 2,860 Target-2 sweep
metrics, 1,055 strict-MOTIVE metrics, hash-pinned image-quality and segmentation
sentinels, non-empty instance mappings, the analysis metadata, and persisted
saturation results. Empty directories and the 60-file MOTIVE smoke tree fail.

Then regenerate into `data/results/`:

```bash
just clean-results
just reproduce
```

`gather_sweep_results.py` is always given an explicit output under
`data/results/summaries/`. It does **not** write `sweep_results.csv` through a
symlink into a canonical sweep tree. Downstream rank and table recipes consume
those isolated summaries explicitly.

`just reproduce` performs:

1. Target-2 and JUMP-lite sweep aggregation (per-codec and best-average views).
2. Strict MOTIVE summary, delta, cross-task, and table rendering.
3. Model-task heatmaps and combined RefChem/MOTIVE tables.
4. Rank-stability rendering from the isolated JUMP-lite summary.
5. Segmentation IoU, matched-feature, feature-correlation, and image-quality
   rendering from staged checkpoints.
6. Saturation replotting from the staged saturation summaries.

It writes to `data/results/` and to new derived subdirectories under
`data/intermediate/`; it does not modify the staged sweep or evaluation roots.

## 5. Full production

Only use this path when the raw inputs, storage, GPUs, model servers, and runtime
have been planned:

```bash
just produce-paper
```

This runs compression, feature extraction, normalization sweeps, MOTIVE
curation/evaluation, segmentation comparison, and then `just reproduce`.
Prerequisite helpers include:

```bash
just build-jl-index
just download-raw
just fetch-cp-profiles
just prep-annotations
just aliby-featurize     # requires the external Aliby/Nahual GPU deployment
```

The full path is operational production, not a smoke test. It can write hundreds
of GB and use multiple GPUs.

## 6. Rebuttal-derived active figures

Later WACV analyses are self-contained under `rebuttal/` and are registered in
`artifacts.toml`. Managed bundles expose both isolated regeneration and
non-mutating verification. Examples:

```bash
just artifacts-verify compression-order-robustness
just artifacts-verify mq-d2e8-synthesis
just artifacts-regenerate mq-d2e8-fixed-recipe
```

The final target-overlap and five-seed held-out renderers are under
`paper_artifacts/`; their compact frozen inputs are committed so the published
figures can be reproduced without rerunning sweeps or model fitting.

Some final presentation packages remain tied to separately retained immutable
snapshots (Cellpose-count subsampling, broader-JUMP cluster coverage, preliminary
non-JUMP-lite PA/PC, and five-seed CHAMMI exclusion). The registry reports these
as external snapshots and `paper_artifacts.lock.json` still verifies their exact
published bytes. They should be promoted into managed repository bundles before
claiming a fully public clean-checkout regeneration of every active figure.

## 7. Validation and caveats

- Never silently intersect incomplete Target-2 codec outputs.
- Keep canonical raw features, sweep trees, and accepted checkpoints read-only.
- Generated PDF bytes can depend on the pinned Matplotlib/font stack; every
  managed deterministic renderer has locked hashes.
- The previous post-sweep validation produced hundreds of figures and matched
  selected tracked references, but the complete final WACV manuscript asset set
  is now governed by `paper_artifacts.lock.json`, not by basename guesses.
- `PIPELINE.md` is a historical technical map. `just --list`, this document,
  `artifacts.toml`, and bundle READMEs are the current execution contract.
