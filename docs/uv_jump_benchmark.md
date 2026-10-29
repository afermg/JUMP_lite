# Two separate CPU benchmark-replay protocols

`examples/uv_jump_benchmark.py` complements the [public-image smoke runner](uv_jump_lite.md).
It consumes **actual raw well profiles**, authoritative metadata and resolved
production recipes. It does not download an invented benchmark bundle, rerun
segmentation, infer missing annotations or replace unavailable metrics with recall.

**Implemented:** CPU OpenPhenom recipe replay, production PA/PC/NAP, strict Raw-only
recipe selection, serialized fitted state, and separately bound reference comparisons.
**Not established:** complete published-benchmark reproduction, CuPy/cuML numerical
parity, historical model lineage, or full MOTIVE CC/CG/GG execution. Replaying
archived raw features does not validate the separate fresh-image inference contract.

## Run

```sh
uv run --locked examples/uv_jump_benchmark.py \
  --manifest benchmark-inputs.json --target both --output replay
```

The script and its adjacent lock can be copied outside the repository. Without
the lock, use `uv run` instead of `uv run --locked`. Python 3.12.12 and direct
dependencies are pinned; no repository imports, CUDA, services or model downloads
are required. Only Linux x86_64 is validated. `--target paper-v11` and
`--target strict` execute just one protocol. Separate directories are always used
when executing both.

The manifest is required: there is **no public full-benchmark input manifest yet**.
Do not pass the smoke runner's 11-well diagnostic outputs as historical raw profiles.
The default bounded budget allows up to 20,000 metadata rows, four recipes,
four codecs, 4,096 input features and 512 MiB per profile/metadata file. It does
not truncate inputs to meet those budgets. `--allow-large` explicitly permits
up to 200,000 rows, 175 effective recipes and 8 GiB per file. Full-cohort declarations
also require all 384 OpenPhenom coordinates. These are admission limits, not a
promise of low peak memory: pairwise PA/PC can be expensive. No large run is
started implicitly by the small-image example or its full-readiness report.

Use a new output directory with no symlinked parents. Input SHA256s are checked
before execution and again before the completion record. All null caches and
runtime source files live under the output directory. BLAS/Polars/Numba and the
copairs thread pools are bounded to one worker. Numerical seeds and null sizes
are not reduced to obtain a faster diagnostic result.

## Manifest contract

Top-level fields:

- `schema_version`: `1`.
- `family`: `"openphenom"`; `profile_kind`: `"raw_well_features"`.
- `cohort_scope`: `"bounded-validation"` or `"full-declared-cohort"`. The latter
  declares the supplied cohort, **not completeness of the entire paper benchmark**.
- `seed`: strict split seed, default `20260811`. Run other seeds separately.
- `metadata`: an artifact record, `{"path": "metadata.parquet", "sha256": "<64 lowercase hex characters>"}`.
- `recipes`: ordered entries with a safe unique `id` and a `config` artifact
  record for a resolved norm_3 YAML file. Declare each effective recipe once;
  duplicate enabled-step signatures are rejected. Disabled pruning aliases do
  not become separate candidates.
- `codecs`: ordered entries with `name`, a `profiles` artifact record, and
  `features`: the complete ordered list of actual numeric column names.
  Raw must come first; other allowed names are HQ, MQ and D20. The axis length
  must agree across codecs. Explicit names accommodate the historical
  `nahualX_` versus `openphenom_nahualX_` prefixes without guessing coordinates.
- Optional `references`: separate `paper-v11` and `strict` artifact records
  for reference JSON documents. Never reuse one target's scores for the other.
- Optional explanatory source provenance is retained in `input_manifest.json`.
  Review local paths before sharing outputs.

Paths resolve relative to the manifest, not the working directory. Metadata must
contain exactly the intended cohort, with these nonnull columns:

| Column | Contract |
|---|---|
| `Metadata_id` | Unique physical well key; shared exactly across all codec inputs |
| `Metadata_JCP2022` | Treatment identity; cannot be both control and treatment |
| `Metadata_Group` | Authoritative group, not inferred from feature values |
| `Metadata_Plate` | Physical plate key, unambiguous across sources/batches |
| `Metadata_Batch` | Declared normalization batch key |
| `Metadata_negcon` | Typed Boolean control flag |
| `Metadata_RefChemDB_target` | Frozen pipe-separated targets; `unknown` when absent |

All other metadata columns are ignored. Raw tables must cover every declared well
exactly once. Feature matrices are read only after freezing metadata, recipes and
the strict treatment split. PyArrow reads and fully validates the selected raw
Parquet data; no normalized archive or saved PA map is read for strict scoring.
The manifest supplies prejoined metadata in place of the resolved recipe's
machine-specific metadata path. This equivalence is the input preparer's explicit
responsibility; an unverified join must not be represented as paper parity.

## Distinct normalization and scoring rules

### `paper-v11`

Replay each declared resolved recipe on the full declared cohort. Missingness and
feature filters see that population; plate scalers fit all rows unless the recipe
explicitly requires controls. TVN always fits controls. All surviving treatments
are scored. Every declared recipe/codec is reported; this is not an implementation
of the paper's complete cross-model/grid/figure-selection workflow.

### `strict`

Before reading features, group treatment identities by their complete set of
metadata group memberships. Within each stratum, sort SHA256 of
`seed + "\0" + stratum + "\0" + identity` and assign rounded 20% to selection,
80% to held-out test. All wells of an identity have one role. This matches the
historical `make_treatment_split` helper; default seed is `20260811`.

Feature selection, missingness thresholds and all-treatments plate-scaler fitting
use **selection treatments plus shared controls only**. Control-only operations
fit only controls. No fallback to held-out rows is allowed when calibration data
are missing. Choose one recipe using Raw selection PA/PC, independently min-max
scaling the two metrics and maximizing their product; a constant range scales to
one and ties are lexical. Freeze that recipe across codecs; each codec fits its
own numerical state from its permitted rows. Report test treatments with the
shared controls as retrieval/calibration references. Codec held-out well sets must
match exactly after cleaning, or execution fails instead of silently intersecting.

Shared controls mean **held-out treatments**, not held-out plates or batches.
This runner evaluates one seed and one model family, not the full five-seed,
all-family strict production workflow. With only two candidates, opposing PA/PC
rankings can give two zero products and a lexical winner; this is a software
validation of the selection rule, not evidence that the winning recipe is optimal.

### CPU numerical port

Supported enabled steps are missingness removal, variance/outlier feature filters,
plate standardization or raw-MAD normalization, TVN_EFAAR, empirical inverse-normal
transform, correlation pruning, metadata binding and PA/PC evaluation. Unsupported
steps/parameters and unresolved interpolation fail rather than being ignored.

TVN uses control-fitted global standardization, full-SVD PCA, per-batch control
standardization, batch-control covariance whitening and pooled-control covariance
recoloring. Covariances are regularized by the recipe epsilon. The dimension/control
ratio guard uses the recipe's clamp rule. Every batch needs at least two controls;
unseen or uncalibrated batches fail. State arrays are saved without pickle.

The CPU port uses float64, sklearn full-SVD PCA and NumPy symmetric eigendecomposition.
It is **not a bitwise CuPy/cuML claim**, and its stricter failure checks deliberately
refuse production fallback behavior that would invalidate strict fitting. The saved
states, audit lists and scores make numerical comparisons possible without hiding
backend differences.

PA/PC use the unchanged hash-verified `norm_3/metrics.py` and `io.py` from commit
`bc4f74838cad1441bd3723b37e4ad78c5cceaa9b`, obtained anonymously over verified TLS.
An offline `--metrics-source /path/to/norm_3` must match the same hashes.
Runtime is pinned to copairs 0.5.4. Null size is 10,000, seed 0; PC requires at
least three compounds per target and keeps the production maximum of 50 targets
per compound. PC uses group_high/group_low. Distance is Euclidean for at most two
features, cosine otherwise. Undefined/missing/nonfinite metrics fail; zero is
never substituted. PA and PC unit tables are retained alongside aggregate NAP.

## Reference comparisons and completion

Each target writes `scores.json` with a `protocol` identity, `scores` and
`reference_kind`. Local output uses `local_cpu_replay_not_published`. Identities
bind cohort, artifact hashes, ordered feature axes, target, split and metric source.
Reference identities must match exactly; population counts and score keys must
also agree. Numerical comparisons use fixed `rtol=1e-5`, `atol=1e-6`: these are
engineering tolerances, **not independently established CPU/GPU parity tolerances**.
A metric difference returns exit 4. Invalid inputs/reference identity return exit 2.

Exit 0 and a final `provenance.json` mean computations completed, not that the
paper was reproduced. A supplied prior CPU run can verify repeatability but is
not an independent historical reference. Even a caller-declared
`published_frozen_reference` label does not certify publication provenance or
whole-paper coverage; `published_parity` remains false. No reference is reported
as `not_supplied`. Missing public archives, MOTIVE inputs, full inventory coverage
and historical inference lineage remain explicit outstanding work.

## Actual validation

Two complete CPU runs used the previously frozen real-data validation cohort:
4,564 wells, 328 chemical identities, 2,686 controls, all 384 raw coordinates,
Raw/HQ/MQ/D20 and two predeclared resolved recipes. Strict fitting used 3,065
selection/control wells and excluded 1,499 held-out treatment wells. The test
partition had 262 treatment identities. All recipes produced 96 features.

The second run used a copied script/lock, an empty inherited environment, isolated
HOME and downloaded Python outside the repository. **All 87 target-specific
output files were byte-identical**, and both separately bound local references
matched. Independently mutating real held-out values in memory left both strict
recipes' fitted state and selection/control outputs identical; the v11 all-row
standardization state changed as expected. No source profile was modified.

These are bounded **real-data protocol tests**, not synthetic substitutes for
published metrics. The metadata/annotation-rich cohort and two recipes do not
replace the full 175-effective-recipe OpenPhenom search or other model families.
See the [validation receipt](../reproducibility/validation/uv-dual-protocol-20261029.json)
and [remaining work](uv_reproduction_progress.md).

```sh
uv run tests/test_uv_jump_benchmark.py -v
```

Offline unit fixtures are explicitly synthetic and test split equivalence,
held-out isolation, state replay, transformations, failures and reference binding.
