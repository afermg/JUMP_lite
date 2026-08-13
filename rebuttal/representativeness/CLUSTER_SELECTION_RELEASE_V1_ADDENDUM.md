# Current-release cohort rescore addendum

This addendum applies only to
`outputs/profile_cluster_representativeness_release_v1/`. The historical
3,832-treatment analysis, its preregistered design, and all of its input and
output bytes remain frozen and are not current-release evidence.

## Cohort identity

The only membership source is tracked
`metadata/jump_lite_v1_perturbation_metadata.parquet`, required to be exactly
949,745 bytes with SHA-256
`bbedb37f12fdeb9a09e72abaa166159d286052dc1201166d811c5310db5cd7e1` and the
11-column schema checked by the rescore runner. Among rows with
`Metadata_Perturbation_Type == "compound"`, there are exactly 3,776 unique
compound identifiers: 3,775 `trt` identifiers and one `negcon` identifier
(`JCP2022_033924`). The negative control is excluded explicitly.

The 3,775 treatments must be unique and a strict subset of the frozen
3,832-treatment historical manifest. Exactly 57 identifiers must be
historical-only and zero current-only. Every current treatment must have a
frozen assignment and be fit-eligible. The sorted treatment manifest and a
provenance record binding it to the metadata and historical-manifest hashes are
written into the new result root.

## Frozen fit and current-label scoring

The reviewed label-blind fit, assignments, diagnostics, PCA/model, partition
sensitivities, and UMAP coordinates are reused without refitting. The rescore
independently pins the imported analysis helper, the reviewed 1,135-byte
`fit_complete.json` marker (SHA-256
`fee7d3c85d2a4da2223c2962f2c75f247315419745d631785b84ea768036f332`),
and every fit artifact by size and SHA-256. Thus helper drift or coordinated
changes to an artifact and its marker fail closed. Current labels are joined
only after those identities are verified. Current-label-dependent outputs are
recomputed using the original design:

- structural counts;
- deterministic SHA-based five-fold OOF constant, count-only, cluster-only, and
  count-plus-cluster metrics on eligible-primary and all-descriptive universes;
- all 128 cluster rows, occupied coverage, eligible mass coverage, total
  variation, and Jensen--Shannon divergence;
- 2,000 deterministic permutations within frozen structural strata; and
- K/preprocessing partition sensitivity using the frozen assignments.

The historical matched comparator manifests have 3,832 rows. They are omitted,
not resized or reused, because doing otherwise would either mismatch the current
cohort or introduce a new selection procedure beyond this narrow rescore.

## Interpretation gates and qualifications

The original gates remain fixed. Detectably better requires within-stratum
permutation p <= 0.05 and lower combined OOF log loss than count-only.
Materially better additionally requires combined/count-only AP >= 1.25.

The permutation is a finite-cohort design null, not population inference.
Broad cluster occupancy is not proportional coverage. Low ARI prohibits
interpreting operational clusters as biological classes. Selection is strongly
confounded with acquisition structure. Results are restricted to the fixed
CellProfiler feature scope and make no model-rank claim about broader JUMP,
other representations, or other biological endpoints. The frozen label-blind
UMAP is display-only.
