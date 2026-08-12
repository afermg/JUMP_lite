# Cluster-selection representativeness report

## Direct answer

JUMP-lite membership is **detectably but not materially more retrievable after accounting for acquisition structure**, under the preregistered finite-cohort thresholds. The selected compounds broadly occupy the partitioned profile space, but they are not distributed like a uniform sample of eligible JUMP compounds. This does not establish or refute biological representativeness because the phenotypic partitions are unstable and selection is strongly confounded with replicate structure.

## Scope and label separation

The CPU-only `fit-clusters` stage partitioned the frozen 115,721-compound CellProfiler consensus without reading selection labels. Only after the unlabeled assignments/model were hash-frozen did `score-selection` join the frozen 3,832-ID JUMP-lite manifest. Selected prevalence is 3.311413% overall and 4.015677% among the 95,426 fit-eligible (`n_wells >= 4`) compounds.

The primary partition used finite filtering, eligible-compound median/IQR scaling, clip `[-10,10]`, PCA32, and label-blind minimum-inertia MiniBatchKMeans K=128 across five fixed seeds. It retained 94/96 features and explained 90.639% of variance.

## Strong acquisition-structure confounding

| Structural stratum | Compounds | Selected | Selected fraction |
|---|---:|---:|---:|
| fewer than four wells | 20,295 | 0 | 0.000% |
| 4–7 wells, multiple sources | 77,697 | 315 | 0.405% |
| 4–7 wells, one source | 2,855 | 1,234 | 43.222% |
| 8+ wells | 14,874 | 2,283 | 15.349% |

Consequently, a naive cluster-only result must not be interpreted as phenotype-driven selection.

## Representation across partitions

Selected compounds occupied 120/128 primary clusters. Those occupied clusters contained 96.24% of eligible compounds. The selected-versus-eligible cluster distributions had total-variation distance 0.4750 and Jensen–Shannon divergence 0.1723 nats. Thus coverage was broad, but allocation was non-uniform.

Primary raw cluster selected fractions ranged from 0 to 67.83%; eight clusters contained no selected compound. These cluster-specific values are tabulated for completeness, not as stable biological classes.

## Out-of-fold retrieval

Primary eligible-universe metrics:

| Predictor | ROC-AUC | Average precision | AP lift over 4.016% prevalence | Log loss |
|---|---:|---:|---:|---:|
| Count/structure only | 0.9316 | 0.5158 | 12.85× | 0.09460 |
| Cluster only | 0.8199 | 0.2777 | 6.91× | 0.13220 |
| Count/structure + cluster | 0.9562 | 0.5765 | 14.36× | 0.08274 |

Adding cluster identity increased AP by 11.76% relative to the count-only baseline and improved log loss. A 2,000-shuffle label permutation within structural strata gave design-null p=0.000500. This passes the preregistered “detectable” gate but not the 1.25× AP-ratio “material” gate.

Across ten separate balanced selected-versus-matched-comparator sensitivities, combined-model AP was 0.8121 median (0.8092–0.8182), versus count-only AP 0.7411 (0.7336–0.7559) and cluster-only AP 0.7071 (0.6988–0.7147). These ten values are descriptive sensitivities, not p-values or population percentiles.

## Partition robustness

Mean pairwise primary-seed ARI was only 0.162, below the preregistered 0.8 gate; unique cluster-specific biological claims are therefore prohibited. Cluster-only eligible AP varied with partition resolution/preprocessing:

- clip-10 K=64: median 0.2124 (0.1998–0.2251)
- clip-10 K=128: median 0.2615 (0.2571–0.2777)
- clip-10 K=256: median 0.3223 (0.3141–0.3238)
- rank-Gaussian K=128: median 0.3044 (0.3031–0.3195)

The continuous conclusion—some phenotypic-region enrichment—persists, while its magnitude is resolution-dependent.

## Interpretation and limitations

JUMP-lite spans most primary partitions, but membership is enriched in particular regions and is modestly more predictable after controlling for coarse replication structure. This is compatible with broad-but-nonuniform phenotypic coverage. It is not a population probability, causal claim, or proof that annotation-driven selection directly targeted morphology. Clusters discretize a continuum, source identity is absent from the cached consensus, the fixed 94-feature projection is not biology-optimized, and negligible effects can be detectable at this cohort size.

## Validation evidence

The committed package is self-validating: `test_cluster_representativeness.py` checks frozen inputs, portable fit-marker resolution, overwrite refusal, model portability, counts, manifests, and every record in `output_hashes.json`; `test_cluster_selection_plot.py` checks label-blind ordering, the exact coordinate digest, summary values, rendered files, and `cluster_selection_figure_hashes.json`. The exact fit and score runner snapshots are committed beside the results. Historical append-only run logs are not included in or cited as evidence for this package. These checks do not rerun or mutate clustering, scoring, permutations, models, or scientific output tables.
