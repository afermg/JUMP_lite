# Portable real-input JUMP-lite smoke run

**This is a real-data software test, not reproduction of published benchmark
scores.** The synthetic `uv_profile_demo.py` remains a separate, faster example.
The real runner downloads public images and annotations, executes a frozen MAE
locally, and evaluates a deliberately small, annotation-enriched cohort. It does
not use deposited embeddings as a substitute for fresh inference.

## One command, no project environment

Install [uv](https://docs.astral.sh/uv/), then run:

```sh
uv run examples/uv_jump_lite.py --output real-run
```

For fully locked dependency resolution, retain the adjacent `.py.lock` file and
use `uv run --locked examples/uv_jump_lite.py --output real-run`. The script alone
also works when copied into an unrelated directory. There are no
repository imports, private default paths, model servers, credentials, Nix,
Conda, Docker, segmentation jobs, or background services. Python 3.12.12 and
pinned dependencies are declared in the script. PyTorch CPU wheels are selected
on Linux/Windows; macOS uses PyPI. Only Linux x86_64 has been exercised.

First execution downloads roughly 0.7 GB of weights, 8 MB of metadata, 55 TIFFs,
and the Python/runtime dependencies. The validated download cache occupied about
778 MiB, excluding Python dependencies. Expect additional runtime disk space and
several GiB of RAM. Inference uses CPU, one thread, and one tile per batch. It
never uses a GPU occupied by another job.

Use fresh output directories. `--cache` selects an explicit reusable download
cache (default `jump-lite-download-cache`); every reused entry is rehashed.
Neither outputs nor corrupt/incomplete cache entries are silently overwritten.
Network failures leave no success marker. SHA256 and size mismatches fail closed.
TLS verification stays enabled; a pinned certifi CA bundle supports uv Python on
hosts where OpenSSL cannot discover the system CA path.

Metadata-only preparation avoids model/image downloads and inference:

```sh
uv run examples/uv_jump_lite.py --prepare-only --output prepared
```

This still resolves the script's declared dependencies. Success means **prepared
only**, recorded in `prepared.json`, not inference or benchmark completion.

## Frozen data/model contract

- All three metadata tables are fetched anonymously from Cell Painting Gallery,
  with the exact file sizes/SHA256 from the release Croissant metadata.
- Selection is frozen in `configuration.json` **before** reading feature values
  or scores: lexical first eligible `group_high` plate; five negative-control
  wells and six distinct treatment identities, one well each. Include the first
  lexical shared-target pair, then four lexical filler identities. Choose the
  first lexical site in each well. No fallback/replacement on input failure.
- RefChem relationships require `activity_class == active` and support >=5.
  Shared-target pairs are deduplicated into directed edges with target evidence.
  This is **not** official PC/NAP, MOTIVE CC, or confirmation of annotation truth.
- Original TIFF channels are explicitly AGP, DNA, ER, Mito, RNA. Each must be a
  single uint16 plane with matching dimensions. Take one top-left 256×256 tile.
  No resizing, lossy-codec substitution, padding, whole-well inference or masks.
- Model: `recursionpharma/OpenPhenom`, revision
  `0f92333685f6e9f031b804c70fe246f9b05ae90d`. The seven source files, config and
  712,434,294-byte checkpoint have predeclared SHA256 hashes. Only verified source
  is imported. The checkpoint is actually a torch archive despite its name;
  loading uses `weights_only=True` and strict state-dict matching.
- Frozen `predict()` applies its own pixels/255 and per-tile instance
  normalization, then returns the final mean of non-CLS tokens (384 features).
  Inference repeats the first input to check deterministic output. This exact
  prospective contract does **not** resolve historical deposited-profile lineage
  or claim equivalence to historical whole-site preprocessing.

TIFFs do not have SHA256 values in the deposited site index. The first download
records their hashes in `image_lock.json`; the cache enforces those receipts.
To enforce exactly those pixels on later runs, even with a new cache:

```sh
uv run examples/uv_jump_lite.py --image-lock real-run/image_lock.json --output real-repeat
```

A supplied lock must name exactly the 55 frozen URLs. SHA256 drift aborts; it does
not silently update the lock. The manifest is not an authentication mechanism
for untrusted third-party data.

## Normalization and evaluation

Fit medians and **raw** MADs on the five declared negative-control wells only,
then transform all wells as `(x - median) / (MAD + 1e-18)`. Reject zero/tiny MAD,
NaN/Inf, and zero cosine norms; do not drop difficult features or choose a new
cohort after seeing scores. This formula is tested against the unchanged
production `RobustMAD_CPU` class. That test is **primitive parity only**, not the
full GPU pipeline, its scaled-MAD variant, TVN, whitening or feature selection.

Take median treatment profiles per identity (one well each here), rank by cosine,
exclude self before computing cutoffs, and resolve exact ties lexically. Report
Recall@1/5/10% for queries with annotated positives and explicit coverage for the
others. With five nonself candidates, all three cutoffs are one: they are **not
three independent demonstrations of performance**. This tiny, selected cohort
is not suitable for model ranking, annotation correction or biological claims.
It is same-plate control calibration, not held-out-plate generalization.

Outputs include raw and normalized well-feature CSVs, fitted parameters,
per-query results, metadata/model/image receipts, configuration and runtime
versions. `provenance.json` is written last and hashes the outputs. Partial output
without that marker is not a successful run. Local measurement inputs, if used,
may expose local source paths in local provenance; review before sharing.

## Reuse cp_measure outputs without extraction

Optionally add an explicit manifest:

```sh
uv run examples/uv_jump_lite.py --cp-measure measurements.json --output real-with-measurements
```

The manifest must contain **exactly the selected 11 sites**. Obtain their keys
with `--prepare-only`. The current adapter accepts existing codec-pipeline
Parquet tables and their original completion receipts. It does not discover
arbitrary local datasets or infer object types/units. A minimal schema is:

```json
{
  "schema_version": 1,
  "measurement_contract": {
    "contract_version": "<from original receipt>",
    "channels": ["AGP", "DNA", "ER", "Mito", "RNA"],
    "codec": "lossless",
    "schema_sha256": "<from original receipt>",
    "parity_mode": "<from original receipt>",
    "legacy_intensity_semantics": true,
    "cp_measure": {"version": "<version>", "commit": "<commit>", "source_sha256": "<hash>", "source_files": 20, "url": "https://github.com/afermg/cp_measure.git"}
  },
  "features": ["0/max/sizeshapeArea", "0/max/sizeshapeEccentricity"],
  "units": {"0/max/sizeshapeArea": "pixels^2", "0/max/sizeshapeEccentricity": "dimensionless"},
  "object_type": {"column": "metadata_object", "value": "cell"},
  "tables": [{
    "site_key": "<exact selected site key>",
    "path": "profiles/site.parquet",
    "sha256": "<table SHA256>",
    "object_id": "metadata_label",
    "receipt": {"path": "profiles/site.parquet.receipt.json", "sha256": "<receipt SHA256>"}
  }]
}
```

The example shows one table; supply all 11. Relative paths resolve against the
manifest's directory. Copy `measurement_contract` values exactly from receipts:
`cp_measure` comes from `dependency_contract.cp_measure`. Require the same contract
for all tables; never mix versions/codecs silently. Each receipt must bind the
site, complete status, file hash/size and compartment count. Validate in-table
site keys, nonnull unique object IDs, numeric finite descriptors and declared
units. Aggregate all selected compartment objects by median. No segmentation or
cp_measure extraction is rerun; original tables remain untouched.

**Spatial-support caveat:** these are full-site cell medians, whereas the MAE
uses one tile. This interface tests data reuse, not a controlled head-to-head
morphology comparison. Original receipt source/mask provenance is retained, but
site-key agreement is not independent verification that every historical input
pixel matches a newly downloaded TIFF. Physical micron calibration is not assumed.

## Separate production-metric replay

The [dual-protocol CPU runner](uv_jump_benchmark.md) now accepts checksum-bound
raw well profiles and resolved recipes. It separates v11-style cohort fitting from
strict split-before-fit, runs production PA/PC/NAP, and compares target-specific
references. Its bounded validation is not full published-score reproduction.
Fresh-image inference here and historical-profile replay there remain separate
contracts; unresolved checkpoint lineage prevents treating them as interchangeable.

## Full-benchmark mode: explicitly blocked, not implemented execution

```sh
uv run examples/uv_jump_lite.py --mode full --output full-readiness
```

This writes `full_readiness.json` and exits **3**. It reports known unresolved
requirements, performs no benchmark or heavy download, and does not write a
success marker. It is a blocker report, **not a dynamic inspection of a supplied
production installation**. In particular:

1. Public, frozen production MOTIVE inputs/mappings and the complete reference
   evaluation archive have not been bound to this runner. See `DATA_SOURCES.md`
   and `REPRODUCE.md`; a reduced smoke archive is not a replacement.
2. CPU OpenPhenom recipe replay and production PA/PC/NAP are implemented in the
   separate runner, but full-grid/GPU numerical parity and full MOTIVE remain
   unverified. This readiness mode does not execute that runner automatically.
3. Historical checkpoint/preprocessing provenance and the exact published
   cohort/recipes/reference-score contract must be resolved before comparing
   fresh inference to published results.

These are remaining deliverables, not features claimed complete. No new
annotation/normalization choices can establish equality to an unspecified
historical benchmark.

## Validation

```sh
uv run tests/test_uv_jump_lite.py -v
uv run tests/test_uv_profile_demo.py -v
```

Offline fixtures are explicitly synthetic. They test cohort selection, ordering,
metadata and checksum drift, bounded downloads/cache validation, fit isolation,
exact CPU RobustMAD primitive agreement, retrieval ties/self-exclusion, receipt
contracts and explicit incomplete/blocked states. They are not biological evidence.

Real-input validation on Linux x86_64 completed four CPU runs. The second ran a
copied script from an unrelated directory with an empty environment, isolated
HOME and uv-downloaded Python, an enforced image lock, and receipt-bound existing
measurements (11 sites, 1,719 cell objects, two predeclared shape descriptors).
MAE raw/normalized CSVs, fit parameters and image-lock bytes matched the first
run exactly. The third used the dependency lock and repeated the second run's
inputs: all recorded output hashes and provenance bytes were identical. A fourth
run of the then-lint-clean script matched those output hashes; only the script
hash in provenance changed after formatting-only edits. The later dual-protocol
extension updated full-readiness messaging, not image-inference logic. No cached embeddings
were used. The portable [validation receipt](../reproducibility/validation/uv-public-smoke-20261029.json)
records the checks and limits. Both representations' small-cohort
recalls were zero; that is a diagnostic result, not a failure disguised as success
or evidence of incorrect biological annotations. macOS/Windows are untested.
