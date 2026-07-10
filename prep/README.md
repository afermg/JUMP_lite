# Bootstrap from raw

These scripts produce the raw inputs that `just produce-paper` assumes exist:
the raw TIFF images and aliby segmentation/feature outputs.

**Most public reproducers can skip this directory.** You only need it if
you're starting from a clean disk. If you already have raw images and an
aliby output directory, jump straight to `just produce-paper` (see the
top-level README).

`prep/` covers the JUMP-portrait images and the aliby pipeline driver.
Several upstream annotation artifacts still have no automatic fetch path
in this repo — those are listed in the top-level `DATA_SOURCES.md`.

## Step 1 — Build the URI manifest

Generates `data/manifest/jl_index_tidy.parquet` from the
JUMP-cellpainting/datasets GitHub repo plus the per-plate `load_data` CSVs
on S3 (filters red-listed and gray-listed plates).

    just build-jl-index

Inputs: none beyond network access.
Output: `data/manifest/jl_index_tidy.parquet` (plus sibling indexes).

## Step 2 — Download raw TIFFs

Pulls images from the public `cellpainting-gallery` S3 bucket using the
manifest from Step 1. Anonymous S3 access — no AWS credentials required.

    just download-raw                    # 16 parallel workers
    just download-raw n_jobs=128         # full bandwidth on a fat link

Output: TIFFs under `$DATA_ROOT/jump_lite/imgs/raw/` named
`{source}__{batch}__{plate}__{well}__{site}__{channel}.tif`.

## Step 3 — Aliby featurization (EXTERNAL DEPENDENCIES)

Runs aliby to segment images and extract DL-model embeddings into
`$ALIBY_OUTPUT/<model>/<dataset>/<codec>/`. Consumed by `extract-cp-*`
and the cell-count features.

This step is *not* fully automatic:

- `aliby` Python package must be installed separately
  (`pip install aliby` or add to your pixi env).
- Nahual model-serving GPU servers must be running, one per model.
  See `archive/analysis/deploy_nahual_featurizers.sh` for the legacy
  launcher pattern; adapt it to your hardware.
- Edit the constants block at the top of `prep/aliby_featurize.py` —
  `dataset`, `codec_glob_prefix`, `n_devices`, `n_addresses` — to match
  what you downloaded.

Then:

    just aliby-featurize

## Step 4 — CellProfiler profiles (~13.5 GB)

Downloads `profiles.parquet` from `cpg0016-jump-assembled v1.0c` — the
assembled CellProfiler features consumed by `just extract-cp-lite` /
`extract-cp-target2`. Anonymous S3, size-verified (13,550,356,031 bytes),
idempotent.

    just fetch-cp-profiles

Output: `$DATA_ROOT/jump_core_annotated/raw_jump_CP_profiles/profiles.parquet`.

## Step 5 — RefChemDB confidence-tier annotations (optional)

`data/refchemdb/` ships three parquets:

| File | Size | Role |
|---|---|---|
| `refchemdb_inchikey.parquet` | 11 MB | Raw RefChemDB with InChI keys (citation: Judson et al. 2019 ALTEX, PMID 30570668) |
| `ref_chem_overlap.parquet` | 1 MB | RefChemDB filtered to compounds present in JUMP |
| `refchemdb_conf_jump_matched.parquet` | 616 KB | Tier-annotated, joined to JUMP CRISPR/ORF JCPs — consumed by `just metadata` |

Two producers regenerate the chain end-to-end:

    just build-refchemdb-overlap   # raw → overlap (joins to JUMP compound InChIs)
    just build-refchemdb-matched   # overlap → tier-annotated matched parquet
    # or in one shot:
    just build-refchemdb

Both producers reproduce the committed parquets exactly (verified: 181,732
overlap rows; 34,004 matched rows; identical Cross/WithinModalityTier counts
Tier0=7, Tier1=49, Tier2=2488, Tier3=31460).

## Step 6 — Annotation curation (optional)

The `metadata/*.parquet` files are committed, so most reproducers skip this.
You only need it if you want to regenerate `metadata_dataset_filtered_4reps.parquet`,
`metadata.parquet`, `motive_annotations.parquet`, and
`motive_annotations_strict.parquet` from upstream sources.

To run the full annotation chain from a clean clone:

    just prep-annotations

That umbrella runs:
1. `just fetch-annotations` — downloads the 4 Zenodo parquets + duckdb to `data/annotations/` (md5-verified)
2. `just build-refchemdb` — regenerates the RefChemDB overlap + matched parquet
3. `just metadata` — `scripts/build_metadata_dataset.py` (8-step pipeline)
4. `just motive-curate` — `curate_motive.py --mode full` (uses the committed `metadata/motive_splits.parquet` by default; override the positional arg to point at a different upstream file)
5. `just motive-curate-strict` — `curate_motive.py --mode strict --skip-splits`

You can also run any step individually.

**Inputs needed from outside this repo**: none — everything is either committed
or auto-fetched.

## What this directory deliberately does NOT cover

| Artifact | Why no recipe | See |
|---|---|---|
| JUMP annotation DB (`jump_metadata.duckdb`) | Upstream artifact; not ours to host | `DATA_SOURCES.md` |
| MOTIVE compound-compound / compound-gene parquets | Sourced from MOTIVE publication | `DATA_SOURCES.md` |
| RefChemDB matched parquet | Built by `archive/analysis/04_refchemdb_match.py` from RefChemDB raw data | `DATA_SOURCES.md` |
