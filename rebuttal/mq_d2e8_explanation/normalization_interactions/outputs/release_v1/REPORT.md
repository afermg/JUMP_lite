# Normalization-recipe interaction analysis

## Direct result

Across 240 exact family/recipe pairs, the mean MQ-minus-D2-E8 NAP-product difference was -0.001323, the median was -0.000771, and MQ was higher in 40.0% of cells. Only 1 of 48 aligned recipe structures had the same sign in all five families.

## Family-specific summaries

| Family | Mean delta | Median delta | MQ higher |
|---|---:|---:|---:|
| cp_measure | +0.000089 | +0.000964 | 54.2% |
| DINOv2 | +0.001232 | +0.000582 | 54.2% |
| MorphEM | -0.007494 | -0.006466 | 0.0% |
| OpenPhenom | -0.000931 | -0.000813 | 33.3% |
| SubCell | +0.000491 | +0.001504 | 58.3% |

## Variation decomposition

A descriptive balanced two-way decomposition separates the 5-by-48 grid into family, aligned recipe-structure, and family-by-recipe residual components. It is not an inferential ANOVA because recipes are deterministic settings, not independent biological replicates.

- family: 50.8% of grid sum-of-squares variation.
- recipe_structure: 10.4% of grid sum-of-squares variation.
- family_by_recipe_residual: 38.9% of grid sum-of-squares variation.

The family-by-recipe residual fraction was 38.9%. The heatmap therefore supports deterministic family/recipe sensitivity, not stochastic signal cleaning and not a universal MQ benefit.

## Design limitations

- Figure 3c pools deterministic configuration outputs; the 48 recipes are not biological replicates.
- Exact pruning differs by family: cp_measure compares 0.90/0.95, while learned representations compare none/0.90. The aligned heatmap labels these as lower/higher intensity and preserves exact values in CSV.
- All rows use TVN-EFAAR, no PCA, and no repeated stochastic run. Batch-method and PCA effects are therefore not identifiable here. TVN-EFAAR component count (96/128) and epsilon (0.05/0.1/0.2) do vary.
- This analysis uses archived point estimates only and does not quantify compound/target sampling uncertainty.
