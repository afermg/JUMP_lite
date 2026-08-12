# Broader-JUMP compound-profile coverage control

This result-only package documents **broad but non-uniform coverage of operational CellProfiler compound-profile space in JUMP** for the cleaned JUMP-lite compound cohort. It intentionally excludes code, the 40.6 MB consensus, assignments, models, and other large inputs.

## Cohort reconciliation

The cleaned release metadata contains **3,776 compound identities**. Exactly **3,775** have `Metadata_Perturbation_Type == "compound"` and `Metadata_pert_type == "trt"`; these treatment IDs define selection here. The remaining identity, `JCP2022_033924`, is the shared DMSO negative control and is not scored as a selected treatment. The 3,775 IDs are a strict subset of the reviewed 3,832-candidate manifest: 57 candidates were removed by the cleaned release cohort and none were added. See `release_selected_compounds.csv` and `RESULTS_PROVENANCE.json`.

## Frozen label-blind analysis

The scientific partition was not refit. We reused the reviewed clip-10/PCA32/K=128 partition and display-only UMAP from source commit `5b8ab8c6f8dc3616a6ddf6a9310c6bb98a0e551f`. Both representations were frozen without selection labels. Only afterward were the 3,775 cleaned release treatment labels joined and the selection summaries recomputed. The partition fit used 95,426 compounds with at least four wells and assigned all 115,721 compounds.

JUMP-lite treatment compounds occupy **120/128 operational clusters**, whose eligible members comprise **96.24%** of all fit-eligible compounds. Coverage is broad but non-proportional (TV **0.4750**; Jensen--Shannon divergence **0.1723 nats**). Eligible out-of-fold AP is **0.5135** for acquisition structure, **0.2731** for cluster only, and **0.5733** for structure plus cluster. The combined/structure ratio is **1.116**, below the predeclared **1.25x** materiality gate, although the 2,000-permutation within-stratum finite-cohort design null is detectable (`p=0.000500`).

These results support **broad but non-uniform coverage of operational CellProfiler compound-profile space in JUMP**. They do not prove random/proportional sampling, coverage of every phenotype, genetic-perturbation representativeness, representativeness under other feature models, or stable biological cluster classes. Mean seed ARI is only **0.162**. The UMAP is visualization only.

## Files

- `cluster_selection_compound_map.pdf` / `.png`: SI-ready three-panel figure.
- `cluster_selection_summary_table.csv`: compact reported metrics and qualifications.
- `cluster_selection_table.csv`: all 128 operational clusters.
- `retrieval_metrics.csv`: eligible and all-compound OOF retrieval metrics.
- `clustering_diagnostics.csv`: exact frozen partition diagnostics.
- `release_selected_compounds.csv`: 3,775 treatment IDs from cleaned release metadata.
- `RESULTS_PROVENANCE.json`: source identities, cohort reconciliation, gates, and validation.
- `SHA256SUMS`: package checksums.
