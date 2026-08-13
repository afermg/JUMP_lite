# Preregistered compression-order robustness design

## Question

Does the non-monotonic codec ordering seen in the four-plate Target-2 analysis remain interpretable after accounting for small-cohort variability and fixed post-processing recipes?

## Analysis A: archived JUMP-lite profiles

Before outcomes are read, freeze 2,000 PCG64DXSM-seeded (`20260814`) samples from the common held-out JUMP-lite population. Restrict to `group_high`/`group_low`; retain treatment IDs for which every observed treatment/group unit has at least four wells. Form complete membership strata (`group_high`, `group_low`, or both), allocate exactly 306 treatment IDs by deterministic largest-remainder quotas, select four distinct wells per selected treatment/group unit without replacement, and add every frozen common negative control for represented plate/group pairs.

Apply each frozen manifest identically to the 16 Raw-validation-selected normalized profiles (four learned models × Raw/HQ/MQ/D20). Recompute PA, PC, and their product using the established metadata columns and `norm_3.metrics`. After the established PC target-count/promiscuity filtering, a compound/group consensus row defines a retrieval query only if it has at least one same-group eligible positive sharing an eligible target and at least one same-group eligible negative with a disjoint eligible-target set; otherwise its query AP is undefined. Freeze these metadata-only eligibility records once per sample and record positive- and negative-undefined reasons separately. Individual no-positive rows remain available as negative references because the established copairs implementation naturally emits no positive query for them; wholly no-positive groups and no-negative rows are removed from PC. Apply this handling identically across variants and report incidence; PA and sample manifests remain unchanged, and no other PC failure is suppressed. Report all six codec differences, adjacent reversals (HQ>Raw, MQ>HQ, D20>MQ), and any adjacent reversal as the primary inference. As a secondary comparison, compute discordance and rank correlation against a matched group_high/group_low full-population baseline directly from the key-aligned archived per-unit PA/PC tables; do not compare the restricted subsample products to the all-group held-out product.

This is a stratified 306-ID/four-well-per-treatment-group sensitivity, not a literal four-plate replica and not the D10/D15 setting. It conditions on archived transforms and selected recipes.

## Analysis B: direct Target-2 D10 versus D15

For each learned model, select one recipe using only Zstd and the manuscript selection metric PA×PC/100; break exact ties lexically. Require the identical recipe under Zstd, D10, and D15. Validate archived PA compound and PC target key sets and reproduce `metrics.json` points.

Use 50,000 PCG64DXSM replicates (`20260814`). Resample the 306 PA compound clusters and 201 PC target clusters independently with replacement while sharing weights across models/codecs within each margin. Multiply independently sampled PA and PC means under a qualified working product-of-margins model. Report percentile intervals and centered-bootstrap D15−D10 product tests, with Holm adjustment over four models.

## Guardrails

- Frozen normalized outputs and per-unit results are read only.
- No extraction, normalization, retrieval, or sweep is rerun.
- Missing, partial, drifted, or key-mismatched inputs abort; summarize-only revalidates current Target-2 hashes against production provenance before regeneration.
- No codec-specific recipe fallback is allowed.
- Independent per-codec winners are not the fixed-recipe analysis.
- Product inference omits unknown PA–PC covariance and may be too narrow or too wide.
