# Replay OpenPhenom normalization recipes

Run CPU normalization and production PA/PC/NAP on raw well features:

```sh
uv run --locked examples/uv_jump_benchmark.py \
  --manifest benchmark-inputs.json --target both --output replay
```

Keep the script's adjacent `.py.lock` file and choose a new output directory.
Use `--target paper-v11` or `--target strict` to run one protocol.
For introductory model and annotation workflows, see [the feature guide](uv_profile_demo.md).

## Inputs

Supply a JSON manifest with the following fields. Artifact records use
`{"path": "file.parquet", "sha256": "<file SHA256>"}`; paths are relative to the manifest.

| Field | Contents |
|---|---|
| `schema_version` | `1` |
| `family`, `profile_kind` | `"openphenom"`, `"raw_well_features"` |
| `cohort_scope` | `"bounded-validation"` or `"full-declared-cohort"` |
| `metadata` | Parquet artifact record for the selected wells |
| `recipes` | Entries with unique `id` and `config` artifact record for resolved norm_3 YAML |
| `codecs` | Entries with `name`, `profiles` artifact record and ordered `features` column names |
| `seed` | Optional strict split seed; default `20260811` |
| `references` | Optional separate `paper-v11` and `strict` score-JSON artifact records |

Raw comes first; other codec names are HQ, MQ and D20. Each profile table must
cover the selected wells exactly once, with the same feature-axis length.
Metadata requires these nonnull columns:

- `Metadata_id`: unique physical well ID.
- `Metadata_JCP2022`, `Metadata_Group`: perturbation identity and group.
- `Metadata_Plate`, `Metadata_Batch`: physical plate and normalization batch.
- `Metadata_negcon`: Boolean control flag.
- `Metadata_RefChemDB_target`: pipe-separated targets, or `unknown`.

The default budget is 20,000 wells, four recipes and four codecs, with 512 MiB per
input file. `--allow-large` permits 200,000 wells, 175 recipes and 8 GiB per file.
Full-cohort inputs require all 384 OpenPhenom coordinates. Pairwise scoring can
require substantial memory. A public full-benchmark manifest is still pending.

## Protocols

| | `paper-v11` | `strict` |
|---|---|---|
| Fitting population | Declared cohort; control-only steps use controls | Selection treatments plus shared controls |
| Recipe selection | Report every supplied recipe | Select on Raw selection PA/PC, then freeze across codecs |
| Evaluation | All surviving treatments | Held-out treatments with shared controls |

Strict splitting assigns 20% of treatment identities to selection and 80% to test,
stratified by group memberships, before reading features. Its selection score is
min-max(PA) × min-max(PC), with lexical ties. Each codec fits its own numerical
state; held-out well sets must agree across codecs.

The CPU implementation supports feature filtering, plate scaling, TVN_EFAAR,
inverse-normal transformation and correlation pruning. PA/PC use pinned production
metric code and copairs 0.5.4, with 10,000 null samples and seed 0.

## Results

Each target writes normalized profiles, fitted state, fit-row audits and scores.
References must match the target, input hashes, split and population counts.
Comparisons use `rtol=1e-5`, `atol=1e-6`. Exit codes: 0 completed, 2 invalid input,
4 score mismatch. `provenance.json` records completion and reference status.

Validation covered 4,564 real wells, four codecs and two recipes: 87 output files
matched across two runs, and held-out perturbation checks passed. Full-grid,
GPU-parity and published-reference validation remain [follow-up work](uv_reproduction_progress.md).

Tests: `uv run tests/test_uv_examples.py -v`.
