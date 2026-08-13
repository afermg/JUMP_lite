# Preregistered cluster-selection analysis

Frozen before fitting/scoring the new analysis. Selection labels will not be read by the `fit-clusters` command.

## Question and estimand

Partition the finite broader-JUMP compound-profile universe without selection labels, then ask whether frozen JUMP-lite membership is distributed uniformly across those phenotypic partitions and whether partition identity incrementally retrieves membership beyond acquisition structure. This is a finite-cohort/design-null analysis, not a population likelihood, causal effect, or proof of biological representativeness.

## Frozen inputs

- Plate-negative-control-robust compound consensus: exactly 115,721 unique `Metadata_JCP2022` rows, SHA-256 `dc2f84178a15f2e18177d4475b094af0da8fab10b1856bd3d1e4f6521d6c9d06`.
- Fixed 96-feature projection from `selected_features.txt`.
- Membership is read only during scoring from the frozen 3,832-ID manifest, SHA-256 `a0671dcaae029a2c32ac58fdaf09178806b495d33a5ea439ff859b3c0fbe74de`.
- One fixed 20,000-ID evaluation manifest and ten matched non-selected manifests are scoring-only sensitivity inputs.

## Label-blind fit

Fit only compounds with `n_wells >= 4` (expected 95,426); assign the remaining 20,295 compounds afterward to their nearest centroid. Stable-feature rules are >=95% finite values, finite median/IQR, and positive IQR. Median-impute, scale by eligible-compound median/IQR, clip to [-10,10], and fit randomized PCA32 with seed 2026.

Primary partition: MiniBatchKMeans K=128, batch size 4,096, n_init=5 at seeds 13, 42, 2026, 31415, and 65537. Choose the lowest-inertia fit without labels and canonicalize IDs by lexicographic ordering of PCA centroids. Sensitivities are clip-10 K=64 and K=256 plus rank-Gaussian/PCA32/K=128, each at the same seeds. Diagnostics use a fixed SHA-ordered sample of at most 20,000 and report silhouette, inertia, cluster-size summaries, ARI, and AMI. Mean primary-seed ARI below 0.8 prohibits unique cluster-specific biological claims.

The fit command writes and hashes unlabeled assignments, long sensitivity assignments, diagnostics, and a portable NPZ containing feature order, imputation medians, scaling medians/IQRs, PCA mean/components, and primary centroids. It has no selected-manifest argument.

## Scoring

Join membership exactly after fit completion. All 3,832 selected IDs must be present and fit-eligible. Structural strata are `ineligible_lt4`, `w4_7_single`, `w4_7_multi`, and `w8plus`.

Use deterministic SHA-based five-fold cross-fitting on identical rows. Compare constant prevalence, structural counts (stratum plus standardized log1p n_wells/n_sources/n_plates), cluster-only, and structure-plus-cluster, without class weighting. Report ROC-AUC, average precision, AP lift, log loss, Brier score, and tie-aware precision/recall at the selected-count cutoff for eligible-primary and all-descriptive universes.

Use 2,000 deterministic shuffles within structural strata for a fast conditional cluster-association contingency statistic. Detectably better requires conditional p<=0.05 and lower combined OOF log loss than count-only. Materially better additionally requires combined/count-only AP >=1.25. Ten matched-manifest comparisons remain separate descriptive sensitivities; their median/range are not p-values or percentiles.

Report every cluster, including counts, raw and Jeffreys-smoothed membership fractions with working intervals, global/structure-conditional lift and residuals, BH-adjusted conditional enrichment values, and distance/count summaries. Also report occupied-cluster coverage, eligible mass in occupied clusters, total variation, Jensen-Shannon divergence, and lift distribution. Low retrieval supports mixing but does not prove representativeness; high retrieval can reflect annotation/acquisition confounding.
