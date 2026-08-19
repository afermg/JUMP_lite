# Plate and unit influence for MQ versus D2-E8

One Zstd-selected recipe per Figure 3c family was frozen across MQ and D2-E8. Rescoring each original normalized output reproduced every archived PA/PC point within 1e-12 before leave-one-plate-out results were accepted. LOO contrasts use an explicitly recorded common Metadata_id population within each family; this matters for cp_measure, whose codec outputs retained slightly different rows.

| Family | Full MQ-D2-E8 product | LOO min | LOO max | Top-10 absolute share | PA absolute share |
|---|---:|---:|---:|---:|---:|
| cp_measure | +0.001367 | -0.006557 | +0.005164 | 15.1% | 43.7% |
| DINOv2 | +0.001845 | -0.000464 | +0.006767 | 20.1% | 34.1% |
| MorphEM | -0.008954 | -0.008191 | -0.000442 | 14.6% | 29.9% |
| OpenPhenom | -0.003402 | -0.004624 | -0.000607 | 16.0% | 27.4% |
| SubCell | -0.002395 | -0.008154 | +0.001125 | 18.2% | 32.8% |

The cp_measure common-population full contrast (+0.001367) differs slightly from its archived unequal-population contrast (+0.001251); the four learned-family populations are already identical across codecs. All unit contributions, the Full column, and every leave-one-plate-out score use the same family-specific common Metadata_id population. Plate and laboratory are perfectly confounded in this four-plate pilot. Unit contributions are descriptive influence diagnostics, not independent causal effects. PA and PC are retrieval-derived and the product decomposition does not supply end-to-end uncertainty.
