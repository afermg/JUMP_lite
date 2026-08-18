# Paired recipe audit of MQ versus D2-E8

Figure 3c pools five representation families and 48 normalization recipes per codec. The pooled marginal median is therefore not a paired codec effect.

## Result

The pooled marginal median is 0.02405992 for MQ and 0.02250419 for D2-E8 (difference +0.00155573). After matching all 240 family/recipe rows, the MQ-minus-D2-E8 mean and median differences are -0.00132265 and -0.00077061; MQ is higher in 40.00% of pairs.

This is a pooled-median inversion: the marginal medians place MQ above D2-E8, while the typical paired recipe difference is negative. It does not support a general MQ biological advantage.

## Family summaries

| Family | D2-E8 median | MQ median | Paired mean delta | Paired median delta | MQ higher |
|---|---:|---:|---:|---:|---:|
| cp_measure | 0.029109 | 0.028909 | +0.000089 | +0.000964 | 54.2% |
| DINOv2 | 0.020096 | 0.020033 | +0.001232 | +0.000582 | 54.2% |
| MorphEM | 0.041424 | 0.034419 | -0.007494 | -0.006466 | 0.0% |
| OpenPhenom | 0.013522 | 0.011669 | -0.000931 | -0.000813 | 33.3% |
| SubCell | 0.021498 | 0.022733 | +0.000491 | +0.001504 | 58.3% |

## Interpretation limits

The 48 recipes are a structured sensitivity grid, not independent biological replicates. No inferential p-values are computed over recipes. This analysis does not estimate sampling uncertainty over treatments or targets and does not attribute any difference to denoising or biological improvement.
