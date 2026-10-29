# Portable reproduction implementation checkpoint

Scope approved: small real-data default, explicit full-benchmark mode, one pinned
OpenPhenom model first; **both** frozen-v11 and strict split-before-fit targets,
with separate fitting rules, references and reports. Extend the uv-example branch; do not touch production
services, the dirty main checkout, source datasets, or published release objects.

Review: [PR #11](https://github.com/afermg/JUMP_lite/pull/11) contains both
reproduction targets and the public-image smoke runner. The unchecked items below
remain follow-up work; this PR does not claim full published-score reproduction.

## Acceptance checklist

- [x] Download hash-pinned public metadata and annotations anonymously.
- [x] Freeze a bounded real cohort before reading features or computing scores.
- [x] Download original TIFFs with recorded hashes and explicit channel order.
- [x] Load a pinned OpenPhenom revision strictly, with verified source/weight hashes.
- [x] Infer deterministic image features on CPU without a model server.
- [x] Accept receipt-bound existing cp_measure cell tables, preserving units,
      object identities, aggregation rules, and measurement contracts.
- [x] Produce diagnostic normalization, retrieval, coverage, configuration and provenance.
- [x] Report known full-benchmark blockers separately from smoke-test completion.
- [x] Test invalid inputs, checksum drift, cohort identity, repeats and resource bounds.
- [x] Validate fresh public-input execution and document actual portability evidence.
- [x] Implement separate CPU OpenPhenom recipe replay for v11 and strict targets.
- [x] Execute production PA/PC/NAP with frozen null settings on actual raw profiles.
- [x] Select strict recipes using Raw selection only; serialize fitted-state audits.
- [x] Bind separate target references and validate isolated, deterministic replay.
- [ ] Validate the complete recipe/model/seed inventory and historical GPU numerical parity.
- [ ] Implement full MOTIVE CC/CG/GG execution with authoritative public inputs.
- [ ] Bind the published cohort/recipes/reference scores and verify reproduction.
- [ ] Resolve and distribute the missing public production inputs with approval.
- [ ] Validate macOS/Windows portability.

## Completed validation: public-image checkpoint

`examples/uv_jump_lite.py` is a standalone PEP 723 CPU script; its optional lock
pins transitive dependencies. Four real runs completed on 11 wells / 55 TIFFs.
Runs two and three reused verified original measurements for 1,719 cell objects
and produced byte-identical recorded outputs and provenance. The second used a
copied script, an empty inherited environment, an isolated HOME and uv-downloaded
Python outside the repository. No cached embeddings, GPUs or production services
were used. The final lint-clean script was rerun with identical recorded output
hashes; only its own source hash changed. Offline coverage: 34 new tests, 16
original demo tests and two release-documentation tests. Ruff and diff checks
passed.

Evidence: `reproducibility/validation/uv-public-smoke-20261029.json`.
The complete private execution artifacts are retained on Oppy under
`/work/users/amunoz/artifacts/jump-lite-followups/uv-public-20261029/`;
raw images, profiles and weights are not included in this branch.

The initial metadata-only candidate required two treatment wells per identity
on one plate; no plate met that rule. Before any image inference or scores, the
rule was corrected to one well per treatment identity (five controls plus six
treatments). The failed preparation directory was retained. No score-driven
cohort changes or model tuning occurred. The first TLS attempt also failed due
to uv Python's missing system CA discovery; pinned certifi fixed this without
disabling certificate verification.

## Completed validation: separate v11 / strict CPU protocols

`examples/uv_jump_benchmark.py` and its lock run without repository imports.
The manifest binds raw well profiles, metadata, resolved recipes and separate
reference files. The strict split is frozen before reading features. It preserves
the historical SHA256 stratified 20/80 treatment split and Raw-only min-max
PA/PC recipe-selection rule; no held-out numerical state or archived PA maps are
used for selection. Shared negative controls remain calibration/retrieval references.

Two real runs completed on the prior frozen 4,564-well / 328-compound cohort,
2,686 controls, all 384 raw coordinates, four codecs and two predeclared recipes.
Strict fitting permitted 3,065 selection/control wells, excluding 1,499 test wells;
its held-out score tables cover 262 treatment identities. Eight v11 recipe/codec
results and four strict held-out codec results were produced, plus two Raw
selection candidate results. All 87 target-specific output files matched byte for
byte on an isolated-HOME, empty-environment run of a copied script and lock.
Both separately bound local references matched; neither is a published reference.

Real-data in-memory perturbation sentinels changed only held-out values: both
strict fits and selection/control transforms were identical, while v11 all-row
standardization changed. No source file was modified. Offline coverage adds 25
protocol tests to the earlier 52 tests. Full-grid, all-family, five-seed and
MOTIVE validation remain outstanding. See [protocol details](uv_jump_benchmark.md)
and `reproducibility/validation/uv-dual-protocol-20261029.json`. Private artifacts:
`/work/users/amunoz/artifacts/jump-lite-followups/uv-dual-20261029/`.

Two input-preparation attempts failed in the native Polars reader with invalid
UTF-8/panic errors before any metric computation. The selected original data
validated fully through PyArrow; the loader was changed, not the data or cohort.
Failed preparation directories were retained. No source/release objects or
production services were changed. Only source, documentation, dependency locks,
tests and compact validation receipts are included in the PR; execution datasets
and caches remain outside the repository.

## Known blockers to exact published-score reproduction

The release metadata are public and hash-pinned in `croissant.json`. This does
not make the historical analysis intermediates reproducible:

- `DATA_SOURCES.md` identifies missing public addresses for the JUMP annotation
  database, MOTIVE relationships and identifier mappings.
- `REPRODUCE.md` requires 7,285 JUMP-lite sweep metrics, 2,860 Target-2 sweep
  metrics and 1,055 production strict-MOTIVE metrics. The 60-file smoke bundle
  is not a substitute.
- `src/norm_3/pipeline.py` imports CuPy/cuML unconditionally. The new CPU port
  supports declared OpenPhenom recipes and production PA/PC but uses float64,
  sklearn full-SVD PCA and NumPy eigendecomposition. Repeated CPU execution is
  not independent GPU numerical validation. The two-recipe test is not the
  full 175-effective-recipe family inventory or the full multi-model grinder.
- Historical OpenPhenom profile checkpoint/preprocessing lineage is unresolved.
  Fresh inference can have exact provenance but cannot retroactively establish
  the lineage of deposited profiles.
- A small annotation-enriched cohort is a software test, not an unbiased
  biological benchmark or the published evaluation cohort.

Do not fabricate public URLs, reference scores, historical parity guarantees or
benchmark completion. The runner's fixed comparison tolerances are engineering
checks, not validated CPU/GPU acceptance thresholds. Keep the synthetic example available and clearly named as such.
