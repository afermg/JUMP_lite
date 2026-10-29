# One-file feature → normalization → evaluation demo

For public-image downloads and fresh pinned OpenPhenom inference, see the separate
[real-input uv runner](uv_jump_lite.md). Neither example is a full reproduction
of published benchmark scores.

**The default data and relationships are entirely synthetic. The scores are a
software smoke test, NOT scientific evidence or JUMP-lite benchmark results.**
No images, models, private files, GPU, repository imports, services, shell tools,
submodules, Conda, Nix, Docker, or heavyweight project environment are needed by
the example. Only [uv](https://docs.astral.sh/uv/) must already be installed.

## Run anywhere

Copy `examples/uv_profile_demo.py` into any unrelated directory, then run:

```text
uv run uv_profile_demo.py --output demo-output
```

Or, from this repository: `uv run examples/uv_profile_demo.py --output demo-output`.
PEP 723 metadata pins Python **3.12.12** and NumPy **2.2.6**. uv can download the
required Python automatically. First use needs network access to uv's Python
distribution host and PyPI/package files; subsequent runs can reuse uv's cache.
There is no data download. NumPy is the only dependency, with no runtime package
installation performed by the script itself. Copy the adjacent
`uv_profile_demo.py.lock` too if you want wheel hashes/resolution locked, then use
`uv run --locked uv_profile_demo.py --output demo-output`. The script alone also
works; the lock is not required. Use a **new** output directory each time.

Expected synthetic output: 22 four-feature profiles (10 training-control wells,
12 evaluation wells, six treatment identities), six directed relationships,
and Recall@1%, @5%, @10% each equal to **1.0**. The fixture deliberately makes
paired identities similar. With five eligible references these cutoffs all mean
one reference, so these are not three independent demonstrations of performance.

Seven small output files (normally under 20 KB total):

- `input_profiles.csv`, `input_annotations.csv`: exact input bytes, including the
  generated synthetic fixture for the default run.
- `normalized_profiles.csv`: all input rows, metadata and selected transformed
  features; metadata never enters the feature matrix.
- `metrics.json`: mean directed CC recalls and annotation/evaluation coverage.
- `per_query.csv`: each scored query's positive/candidate counts, top reference,
  effective cutoffs and recalls.
- `configuration.json`: selected features, fit sample identities, plate medians
  and raw MADs, algorithm choices and resource limits.
- `provenance.json`: completion marker written last, input/script/output SHA-256
  hashes, dependency/runtime versions, reference commit, limitations and source
  label. No timestamps or machine-local paths, so same-environment reruns are
  byte-identical. Cross-platform bitwise equality is not promised.

Validation errors exit nonzero before creating outputs. Existing output paths
are never overwritten. An I/O failure can leave partial files; do not treat a
missing or invalid `provenance.json` as a completed run. Failure is never encoded
as a score of zero.

## Use your precomputed features / swap models

```text
uv run uv_profile_demo.py --profiles my-wells.csv --annotations my-edges.csv --features Feature_1,Feature_2,Feature_3 --output my-model-results
```

Supply all three input options together. CSV only; no implicit joins, feature
inference, cohort intersections, imputation or annotation downloading. Each row
must be an already aggregated **well-level** profile, not a mixture of cells and
wells. This narrow demo supports **one group and directed CC relationships**.
The CC name does not independently verify that identifiers denote compounds.

Required profile columns:

| Column | Contract |
| --- | --- |
| `Metadata_Sample` | Globally unique, nonempty well/sample string |
| `Metadata_JCP2022` | Perturbation identifier; arbitrary strings allowed |
| `Metadata_Group` | Exactly one group per run |
| `Metadata_Plate` | Globally unique physical plate identifier (include batch/source) |
| `Metadata_negcon` | Literal CSV `true` or `false` |
| `Metadata_Split` | `train` or `eval`; training rows must be negative controls |
| `Feature_*` | Explicitly selected finite numeric features |

At least three **training negative controls on every supplied plate** are
required, with nonzero raw MAD in every selected feature. All other metadata,
including numeric metadata, must use a `Metadata_` prefix. All feature columns
must use `Feature_`; select the desired ones with `--features`. Unselected
`Feature_` columns are ignored and not emitted as normalized features. Duplicate
sample keys, conflicting control/treatment identities, missing/whitespace-padded
values, malformed CSV, selected NaN/Inf/non-numeric features, zero/tiny control
MAD and zero/nonfinite consensus cosine norms are errors, not silently removed
rows or features. No ID format is inferred from the historical column name.

Annotation header and example (the actual IDs must occur in your profiles):

```csv
task,query_group,query_id,reference_group,reference_id
CC,compounds,drug_a,compounds,drug_b
CC,compounds,drug_b,compounds,drug_a
```

The header order above is required. Relationships are directed; include both
directions for an undirected pair. Both groups must match the profile group.
Both endpoints must be evaluation treatment identities. Unknown identities,
control endpoints, duplicate edges, self-edges and empty annotations fail closed.
Unannotated treatment identities remain reference candidates, but queries with
no positives are excluded from the mean and explicitly listed in coverage.

To swap representations (CellProfiler, a pretrained encoder, another model),
export that model's precomputed well features into this interface, keep the
identities, plates, splits and annotation rules fixed, and run to a different
output directory. Models can have different feature counts. This script does not
run or compare models, enforce cross-run cohort equality, or validate their
training exposure. You must ensure matched cohorts and independently establish
that upstream models/features did not use held-out outcomes. Do not select
features, normalization settings or models using these evaluation labels.

## Scientific slice and deliberate differences

1. For each plate, fit feature medians and **raw** median absolute deviations
   only on rows explicitly marked `train` **and** negative control. Transform all
   rows as `(x - median) / (raw_MAD + 1e-18)`. Evaluation controls, treatments and
   annotations never contribute to the fit. Every plate needs its own declared
   calibration controls; this is not held-out-plate generalization.
2. Take feature-wise medians over evaluation, non-control wells for each
   perturbation. Controls are not retrieval references or queries.
3. Compute cosine similarities between consensus profiles. Exclude the query's
   own identity entirely. Rank by descending similarity, breaking exact ties
   by ascending reference identity. Recall is positives retrieved / all supplied
   positives for that query, averaged equally over queries with positives.
   Cutoff = `max(1, floor(number_of_nonself_candidates * percentage / 100))`.

This independently implemented tiny CPU slice follows the mathematics of
`src/norm_3/core.py:RobustMAD_CPU` and the median/cosine/recall primitives in
`src/motive/evaluate_cross_modality_retrieval.py` at JUMP-lite commit
`94f19cfe1865ad9a07858908dce0caa679627204`. The CPU raw MAD convention is **not**
the GPU helper's scaled MAD. Unlike the production recall primitive's
`argpartition`, ties here have a declared lexical order; self-exclusion also
removes self from the cutoff denominator. Zero-norm profiles and missing
positives are errors rather than returning a misleading zero. These choices
mean this is **not a drop-in reproduction of production benchmark scores**.

Not reproduced: full TVN/whitening/covariance batch correction, production feature
selection, global training/evaluation recipe, PA/PC/NAP and null significance,
GG/CG or target/tier/activity-filtered evaluations, model training, image feature
extraction, segmentation, or biological validation. Plate location and annotation
coverage can confound retrieval. Synthetic success demonstrates plumbing only.

Inputs are bounded at 8 MiB each, 10,000 rows each, 256 selected features and
1,000,000 pairwise similarities (at most 1,000 treatment identities). NumPy's
thread environment is set to one before import. Large analyses need a separately
reviewed workflow, not relaxed limits in this demo.

## Tests and portability evidence

```text
uv run tests/test_uv_profile_demo.py -v
```

For copied tests: `uv run test_uv_profile_demo.py --demo-script uv_profile_demo.py -v`.
Tests cover repeatability, fit isolation, formula, metadata separation, malformed
and nonfinite inputs, control/positive coverage, annotation alignment, self
matches, ties, bounds, non-overwrite behavior and nonzero CLI failures.

Validated on **Linux x86_64**, uv 0.11.21, uv-downloaded CPython 3.12.12 and NumPy
2.2.6: empty ambient environment, fresh owned uv cache/Python installation/home,
no PYTHONPATH/project environment, two isolated locked executions plus a
script-only execution without the lock, and custom-input execution. All seven
outputs, including provenance, matched byte-for-byte across default reruns.
The Linux host was NixOS, but the demo did not enter a Nix shell or use a Nix
Python/package environment; only the preinstalled uv binary was required.
**macOS, Windows and non-x86_64 platforms were not tested.** No background job or
service is created by the example.
