# Data sources & provenance

This document does two jobs:

1. **§1** — lists what a public reproducer needs to obtain to run `just produce-paper`.
2. **§2** — documents how each file under `metadata/` was produced from upstream
   sources, with the addresses (citations / URLs) that still need to be filled in.

---

## §1. What `just produce-paper` needs from outside the repo

| Required input | Provided by | Status |
|---|---|---|
| Raw JUMP TIFFs | `just build-jl-index download-raw` | Wired (`prep/`) |
| Aliby segmentation + per-model embeddings | `just aliby-featurize` | Wired (`prep/`), but requires external `aliby` + Nahual GPU servers |
| CellProfiler `profiles.parquet` | `just fetch-cp-profiles` (anonymous S3, ~13.5 GB) | Wired (`prep/`) |

The only annotation file committed to the repo is `metadata/motive_eval_compounds.parquet`
(the MOTIVE compound allowlist). The rest of `metadata/*.parquet` (incl.
`motive_annotations*`, `metadata_dataset_filtered_4reps.parquet`) is gitignored
and must be regenerated via `just prep-annotations` — see §2 below. A reproducer
who already has those files locally can skip §2 entirely.

---

## §2. Provenance of the committed `metadata/` files

The 11 files in `metadata/` were produced by three scripts in this repo. Each
script's external inputs (TODO ADDRESS entries) are listed.

All four producer scripts are wrapped by the `prep-annotations` recipe:

    just prep-annotations /path/to/motive_eval_compounds.parquet

(symlink the annotation bundle to `data/annotations/` first; see `prep/README.md`).
You can also invoke each step individually (`just metadata`, `just motive-curate`,
`just motive-curate-strict`, `just motive-curate-ultra-strict`).

### 2a. Metadata bundle (`scripts/build_metadata_dataset.py`, 8 steps unified)

| Reads | Writes (all committed) |
|---|---|
| `jump_metadata.duckdb` (well / plate / compound / crispr / orf tables) | `metadata.parquet` |
| `annotations_compound_compound.parquet` (MOTIVE) | `metadata_filtered.parquet` |
| `annotations_compound_gene.parquet` (MOTIVE) | `metadata_negative_controls.parquet` |
| `refchemdb_conf_jump_matched.parquet` (RefChemDB priors) | `metadata_dataset.parquet` |
| `profiles.parquet` (CellProfiler) | `metadata_dataset_filtered_4reps.parquet` ← **master** |

### 2b. MOTIVE annotations (`scripts/curate_motive.py --mode {full,strict,ultra_strict}`)

| Reads | Writes (all committed) |
|---|---|
| `metadata.parquet` (from 2a) | `motive_annotations.parquet` (full) |
| `metadata_dataset_filtered_4reps.parquet` (from 2a) | `motive_annotations_strict.parquet` |
| `annotations_compound_compound.parquet` (MOTIVE) | `motive_annotations_ultra_strict.parquet` |
| `annotations_compound_gene.parquet` (MOTIVE) | `motive_eval_compounds.parquet` (canonicalised copy of the input allowlist) |
| `annotations_compound_gene_curated.parquet` (curated MOTIVE CG; strict / ultra modes) | |
| `annotations_gene_gene.parquet` (MOTIVE gene-gene) | |
| `inchikey_to_jcp2022_mapping_compound_compound.csv` | |
| `inchikey_to_jcp2022_mapping_compound_gene.csv` | |
| MOTIVE eval-compound allowlist (`--motive-splits-path`, published with the MOTIVE paper) | |

### 2c. Top-config list (`analysis/filter_top_configs.py` via `just motive-filter-top`)

| Reads | Writes (committed) |
|---|---|
| `sweep_results.csv` (output of `just sweep-v11{,-lite}`) | `motive_top_configs.txt` |

---

## §3. Addresses still to fill before public release

Each TODO below blocks "rebuild metadata from source" but does **not** block
running `produce-paper` against the committed parquets.

| # | Artifact | Used by | Address (TODO) |
|---|---|---|---|
| 1 | `jump_metadata.duckdb` | 2a | **TODO** — JUMP annotation DB. Internal Broad source? Rebuildable via `broad_babel`? |
| 2 | `annotations_compound_compound.parquet` | 2a, 2b | **TODO** — MOTIVE C–C relationships |
| 3 | `annotations_compound_gene.parquet` | 2a, 2b | **TODO** — MOTIVE C–G relationships |
| 4 | `annotations_compound_gene_curated.parquet` | 2b (strict / ultra) | **TODO** — curated MOTIVE C–G (drugbank / biokg subset) |
| 5 | `annotations_gene_gene.parquet` | 2b | **TODO** — MOTIVE G–G relationships |
| 6 | `inchikey_to_jcp2022_mapping_compound_{compound,gene}.csv` | 2b | **TODO** — produced by `archive/analysis/04_refchemdb_match.py` or an upstream standardisation step; pin the canonical version |
| 7 | RefChemDB raw data | needed to (re)produce `refchemdb_conf_jump_matched.parquet` | ✅ **Cited and shipped**: Judson et al. 2019 ALTEX, PMID 30570668. Raw + overlap + matched parquets in `data/refchemdb/` (12 MB total, derived from local 158 MB CSVs). Producer: `prep/build_refchemdb_matched.py` (`just build-refchemdb`); reproduces tier distributions exactly. |
| 8 | MOTIVE eval-compound allowlist | 2b (`metadata/motive_eval_compounds.parquet`) AND `evaluate_motive.py` | ✅ **Committed** at `metadata/motive_eval_compounds.parquet` (94 KB; 26,450 rows; not a train/test partition — all rows labelled `"test"`, used as a JCP2022 allowlist of MOTIVE benchmark perturbations). Source: canonicalised from the MOTIVE publication supplement. |
| 9 | CellProfiler `profiles.parquet` | §1 (eval) **AND** 2a | ✅ **Cited and fetchable**: `cpg0016-jump-assembled v1.0c` (13,550,356,031 bytes, last modified 2025-06-30). Source: `s3://cellpainting-gallery/cpg0016-jump-assembled/source_all/workspace/profiles_assembled/ALL/v1.0c/profiles.parquet`. Fetcher: `prep/fetch_cp_profiles.py` (`just fetch-cp-profiles`); anonymous S3, size-verified. |
| 10 | `aliby` package | `prep/aliby_featurize.py` | **TODO** — Aliby project URL + supported version |
| 11 | Nahual model-server | `prep/aliby_featurize.py` | **TODO** — Nahual project URL + deploy guide |

For each row, replace **TODO** with a citation (DOI, Zenodo record, GitHub URL,
or "request from authors at …") before publishing.
