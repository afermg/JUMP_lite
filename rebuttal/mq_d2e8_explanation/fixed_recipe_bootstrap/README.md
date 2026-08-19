# Fixed-recipe Target-2 MQ versus D2-E8 bootstrap

This analysis asks whether the apparent pooled MQ-over-D2-E8 ordering in Figure 3c persists after removing codec-specific recipe selection.

## Protocol

- Read the frozen 2,860-row Target-2 sweep only.
- For each Figure-3c family (`cp_measure`, DINOv2, MorphEM, OpenPhenom, SubCell), select one recipe using Zstd `PA*PC/100`; resolve exact ties lexically.
- Require that exact recipe under Zstd, D2-E8, and MQ.
- Validate all archived aggregate points and pin the observed PA query counts (1,280 for learned families; 1,263--1,266 for cp_measure).
- Within each family, restrict PA to the exact common `Metadata_id` intersection across Zstd, D2-E8, and MQ, require an identical query-to-broad-sample mapping, and retain all 306 broad-sample clusters. The common cp_measure population is 1,263 rows (dropping 3/0/1 from Zstd/D2-E8/MQ); learned families retain all 1,280 rows.
- Exactly align the 306 common-population PA broad-sample clusters and 201 PC target clusters across all 15 family/codec variants.
- Draw 50,000 deterministic PCG64DXSM replicates. PA broad samples and PC targets are independently resampled; weights are shared across every family/codec within each margin.
- Report MQ-minus-D2-E8 PA NAP, PC NAP, and product contrasts. Product tests use a centered bootstrap and Holm adjustment across five families.

The PA–PC product uses a working-independence approximation because joint PA–PC cluster covariance is unavailable. Intervals are pointwise and conditional on archived outputs and selected recipes. Non-support is not equivalence, and a positive contrast is not evidence of denoising or biological improvement.

## Run

```bash
/work/users/amunoz/projects/JUMP_lite/.venv/bin/python \
  rebuttal/mq_d2e8_explanation/fixed_recipe_bootstrap/analyze.py
/work/users/amunoz/projects/JUMP_lite/.venv/bin/python \
  rebuttal/mq_d2e8_explanation/fixed_recipe_bootstrap/analyze.py --verify-only
/work/users/amunoz/projects/JUMP_lite/.venv/bin/python \
  rebuttal/mq_d2e8_explanation/fixed_recipe_bootstrap/test_analyze.py
```

Release outputs are under `outputs/release_v1/`.
