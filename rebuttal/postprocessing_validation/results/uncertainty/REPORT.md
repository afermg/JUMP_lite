# Held-out paired cluster-bootstrap uncertainty

Generated from 50,000 deterministic resamples (seed `20260812`) in 87.092 seconds.

## Scope and method

Paired cluster-bootstrap uncertainty over the observed held-out PA treatment and PC target distributions, conditional on frozen retrieval calculations, target eligibility, selected recipes, and normalized profiles.

PA treatment IDs were resampled with replacement within the frozen composite split strata, retaining all evaluable group rows for each sampled treatment. PC targets were resampled within their complete observed group-membership strata, retaining both high/low rows when present. The same cluster weights were applied to every model and codec, so contrasts are paired within each margin. PA and PC margins were independently resampled and multiplied within each replicate under a working product-of-margins model.

All intervals below are pointwise conditional percentile-bootstrap intervals. The independent PA and PC streams omit their unknown covariance, so intervals and centered-bootstrap p-values may be either too narrow or too wide; they are not full end-to-end or unconditional sampling intervals.

Displayed 95% intervals are not multiplicity-adjusted. Centered-bootstrap product tests were adjusted in two separate Holm families: all 15 codec-vs-Raw comparisons and all 51 same-codec model comparisons. No single correction across all 66 tests was applied.

## Learned-model score intervals

| Model | Codec | Product | Pointwise 95% interval |
|---|---|---:|---:|
| MorphEM | Raw | 0.02801 | [0.02254, 0.03382] |
| DINOv2 | Raw | 0.01161 | [0.00891, 0.01454] |
| SubCell | Raw | 0.01126 | [0.00869, 0.01402] |
| OpenPhenom | Raw | 0.01114 | [0.00841, 0.01406] |
| MorphEM | HQ | 0.02686 | [0.02159, 0.03244] |
| DINOv2 | HQ | 0.01017 | [0.00776, 0.01278] |
| SubCell | HQ | 0.01107 | [0.00852, 0.01382] |
| OpenPhenom | HQ | 0.01202 | [0.00923, 0.01497] |
| MorphEM | MQ | 0.02300 | [0.01852, 0.02780] |
| DINOv2 | MQ | 0.00771 | [0.00581, 0.00976] |
| SubCell | MQ | 0.01086 | [0.00836, 0.01354] |
| OpenPhenom | MQ | 0.00845 | [0.00631, 0.01074] |
| MorphEM | D20 | 0.01392 | [0.01068, 0.01741] |
| DINOv2 | D20 | 0.00552 | [0.00388, 0.00733] |
| SubCell | D20 | 0.00934 | [0.00714, 0.01173] |
| OpenPhenom | D20 | 0.00745 | [0.00549, 0.00955] |

## Codec changes from Raw

| Model | Codec | Change | Pointwise 95% interval | Relative change (pointwise 95% interval) | Holm result |
|---|---|---:|---:|---:|---|
| MorphEM | HQ | -0.00115 | [-0.00317, +0.00100] | -4.1% [-10.9%, +3.8%] | unresolved ($p_{Holm}=1$) |
| MorphEM | MQ | -0.00501 | [-0.00729, -0.00276] | -17.9% [-24.1%, -10.8%] | decrease ($p_{Holm}=0.00078$) |
| MorphEM | D20 | -0.01409 | [-0.01718, -0.01116] | -50.3% [-55.7%, -44.9%] | decrease ($p_{Holm}=0.0003$) |
| DINOv2 | HQ | -0.00144 | [-0.00311, +0.00001] | -12.4% [-24.8%, +0.1%] | unresolved ($p_{Holm}=0.4948$) |
| DINOv2 | MQ | -0.00389 | [-0.00569, -0.00236] | -33.5% [-43.7%, -23.5%] | decrease ($p_{Holm}=0.0012$) |
| DINOv2 | D20 | -0.00609 | [-0.00790, -0.00447] | -52.4% [-61.3%, -44.0%] | decrease ($p_{Holm}=0.0003$) |
| SubCell | HQ | -0.00019 | [-0.00098, +0.00060] | -1.7% [-8.6%, +5.6%] | unresolved ($p_{Holm}=1$) |
| SubCell | MQ | -0.00040 | [-0.00152, +0.00070] | -3.5% [-12.9%, +6.7%] | unresolved ($p_{Holm}=1$) |
| SubCell | D20 | -0.00193 | [-0.00345, -0.00045] | -17.1% [-28.4%, -4.4%] | unresolved ($p_{Holm}=0.1029$) |
| OpenPhenom | HQ | +0.00088 | [-0.00045, +0.00221] | +7.9% [-3.9%, +21.7%] | unresolved ($p_{Holm}=1$) |
| OpenPhenom | MQ | -0.00269 | [-0.00419, -0.00138] | -24.2% [-34.0%, -14.2%] | decrease ($p_{Holm}=0.00506$) |
| OpenPhenom | D20 | -0.00369 | [-0.00607, -0.00132] | -33.1% [-47.9%, -14.0%] | decrease ($p_{Holm}=0.0284$) |

## Pairwise ranking result

Within the conditional two-margin bootstrap and the separate 51-comparison same-codec Holm family, MorphEM is the highest point estimate at every codec and is supported over each other learned model in 12/12 comparisons.

Supported middle-model directions in the separate 51-comparison same-codec Holm family:
- MQ: `subcell>dinov2` ($p_{Holm}=0.00102$).
- MQ: `subcell>openphenom` ($p_{Holm}=0.00264$).
- D20: `subcell>dinov2` ($p_{Holm}=0.00102$).

Among all seven Raw model families, simultaneous conditional rank bounds for the three middle learned models are: DINOv2 3--5; OpenPhenom 3--5; SubCell 3--5. Overlapping bounds do not establish equivalence.

## Diagnostics

As an internal Monte Carlo convergence check, 0/88 tracked percentile intervals exceeded a 2% endpoint-drift threshold when the 10k and 25k nested prefixes were compared with the final run. This heuristic assesses interval-endpoint stability only; it does not validate the resampling assumptions. Final intervals use all 50,000 replicates.

## Limitations

- The saved summaries cannot preserve dependence between PA treatments and PC targets; treatment-target incidence and query-level PC contributions are not present. The working-independence approximation omits PA--PC covariance and can make uncertainty either too narrow or too wide.
- Target resampling does not remove dependence among targets sharing compounds.
- Recipe selection, alternative treatment splits, normalization fitting, shared controls, wells/sites, target eligibility, and annotation uncertainty are not resampled.
- The split is treatment-disjoint, not target-disjoint, and the archived transforms were fitted before splitting.
- A non-supported difference is not evidence of equivalence.
- CellProfiler is Raw-only and retains the known site-count asymmetry.

## Outputs

- `heldout_uncertainty.csv`
- `codec_vs_raw_paired.csv`
- `model_pairwise_by_codec.csv`
- `model_rank_bounds.csv`
- `resampling_unit_audit.csv`
- `bootstrap_diagnostics.csv`
- `provenance.json`
