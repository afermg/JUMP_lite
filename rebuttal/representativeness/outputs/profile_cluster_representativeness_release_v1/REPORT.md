# Current-release cluster-selection representativeness report

## Scope and cohort identity

The tracked release metadata contains exactly **3,776 compound identifiers: 3,775 treatments plus one negative control** (JCP2022_033924). The negative control is excluded. The 3,775 treatments are a strict subset of the historical frozen 3,832-treatment manifest: 57 identifiers are historical-only and none are current-only. Every current treatment has an existing fit-eligible frozen assignment.

This rescore reuses the reviewed label-blind CellProfiler partition, assignments, diagnostics, model, and display UMAP without refitting. It changes only current-label-dependent calculations. The historical 3,832-sized matched-comparator sensitivities are deliberately omitted; they are not current-release evidence.

## Current results

The 3,775 treatments occupy 120/128 operational clusters, covering 96.24% of eligible compounds. Coverage is broad, not proportional: TV is 0.4750 and Jensen--Shannon divergence is 0.1723 nats.

Eligible-universe five-fold OOF AP is 0.038887 constant, 0.513537 count/structure-only, 0.273105 cluster-only, and 0.573313 count/structure-plus-cluster. The combined/count-only ratio is 1.116402. The 2,000-shuffle within-stratum finite-cohort design-null p is 0.000500. The detectable gate is true and the 1.25x materiality gate is false.

## Qualifications

This is a finite-cohort design-null analysis, not population inference. Operational clusters have low stability (mean seed ARI 0.162) and are not biological classes. Acquisition structure is strongly confounded with selection. Coverage is broad but not proportional. The scope is the fixed CellProfiler feature projection and makes no model-rank claim for broader JUMP or other representations. The frozen display UMAP is visualization only.
