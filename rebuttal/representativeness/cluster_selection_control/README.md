# Broader-JUMP compound-profile coverage control

This compact Supplementary Information package is exported from the reviewed,
committed current-release results. The complete scientific fit and rescore
pipeline lives in `rebuttal/representativeness/`:
`analyze_cluster_representativeness.py` fits the label-blind historical
partition, and `rescore_cluster_representativeness_release_v1.py` derives and
scores the current 3,775-treatment release cohort without refitting.
`export_cluster_selection_control.py` only verifies and copies canonical
artifacts; it does not fit, rescore, or recompute scientific metrics.

The tracked release contains **3,776 compound identities**: **3,775 treatment
compounds** plus one excluded shared DMSO negative-control identity. The selected
treatments occupy **120/128 operational clusters**, whose eligible compounds
comprise **96.24%** of fit-eligible compound mass. Coverage is non-uniform
(TV **0.4750**). Eligible out-of-fold average precision is **0.5135** for
acquisition structure, **0.2731** for cluster alone, and **0.5733** for structure
plus cluster; the combined/structure ratio is **1.116**, below the predeclared
**1.25x** materiality gate. The 2,000-permutation finite-cohort conditional test
has `p=0.000500`, and mean seed ARI is **0.162**.

**Bounded interpretation:** Broad but non-uniform coverage of operational CellProfiler compound-profile space in JUMP. This does not establish proportional
or random sampling, stable biological classes, coverage of every phenotype,
representativeness under other feature models, or genetic-perturbation
representativeness. The UMAP is visualization only.

## Files

- `cluster_selection_compound_map.pdf` / `.png`: corrected SI figure.
- `cluster_selection_summary_table.csv`: compact reported metrics.
- `cluster_selection_table.csv`: all 128 operational clusters.
- `retrieval_metrics.csv`: eligible and all-compound OOF metrics.
- `clustering_diagnostics.csv`: frozen partition diagnostics.
- `release_selected_compounds.csv`: sorted current 3,775 treatment identifiers.
- `RESULTS_PROVENANCE.json`: source and package records.
- `SHA256SUMS`: checksums for every other package file.

Large consensus, assignment, UMAP-coordinate, and model files are intentionally
excluded from this compact export and remain in the full scientific package.
