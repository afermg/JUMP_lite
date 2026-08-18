# Fixed-recipe Target-2 MQ versus D2-E8 bootstrap

## Design

One recipe per Figure-3c family was selected using Zstd PA×PC/100 only, with lexical exact-tie resolution, then fixed across D2-E8 and MQ. Within each family, PA query rows were restricted to the exact Metadata_id intersection across Zstd, D2-E8, and MQ before aggregation to all 306 broad-sample clusters. The 306 PA clusters and 201 PC target clusters were then resampled independently with shared weights across every family/codec within each margin for 50,000 deterministic PCG64DXSM replicates.

cp_measure PA query original/common/dropped counts are: Zstd 1266/1263/3, D2-E8 1263/1263/0, MQ 1264/1263/1. All learned-family variants retain 1,280/1,280/0 rows. Query-to-broad-sample mappings are required to be identical across codecs on the common rows.

The product is a working product-of-margins model. Its interval and test omit unknown PA–PC covariance and are conditional on archived normalized outputs and the selected recipes; they are not end-to-end uncertainty.

## Selected recipes

- cp_measure: `std_ctrl__outlier100__INT__prune0.9__tvn_efaar_e0.05`
- DINOv2: `robustmad_ctrl__prune0.9__tvn_efaar_e0.05`
- MorphEM: `robustmad_all__prune0.9__tvn_efaar_e0.05`
- OpenPhenom: `std_all__prune0.9__tvn_efaar_e0.05`
- SubCell: `robustmad_ctrl__prune0.9__tvn_efaar_e0.05`

## Results

| Family | MQ-D2-E8 PA NAP | MQ-D2-E8 PC NAP | MQ-D2-E8 product (95% interval) | Holm result |
|---|---:|---:|---:|---|
| cp_measure | +0.03365 | -0.00360 | +0.00127 [-0.00340, +0.00616] | unresolved (p_Holm=0.7063) |
| DINOv2 | -0.00469 | +0.00453 | +0.00185 [-0.00208, +0.00581] | unresolved (p_Holm=0.7063) |
| MorphEM | -0.00828 | -0.01620 | -0.00895 [-0.01365, -0.00444] | D2-E8>MQ (p_Holm=0.001) |
| OpenPhenom | +0.00603 | -0.00861 | -0.00340 [-0.00727, +0.00029] | unresolved (p_Holm=0.3087) |
| SubCell | -0.01330 | -0.00348 | -0.00240 [-0.00528, +0.00036] | unresolved (p_Holm=0.3087) |

Pointwise intervals are percentile intervals. Product p-values use a centered bootstrap and Holm adjustment across the five predeclared family contrasts. Non-support is not equivalence. Results do not establish denoising or biological improvement.

![Fixed-recipe bootstrap](fixed_recipe_bootstrap.png)
