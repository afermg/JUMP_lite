# Target-2 MQ/D2-E8 normalization interactions

This self-contained sensitivity reads the archived Target-2 `sweep_results.csv`
and pairs MQ and D2-E8 within each of the five Figure-3c families and every
identical deterministic normalization configuration. The outcome is the same
unscaled NAP product plotted in Figure 3c:

```text
PA_mean_nap * PC_mean_nap
```

It describes how the MQ-minus-D2-E8 contrast changes across normalization
method, control/all fit scope, pruning intensity, TVN-EFAAR epsilon, and
TVN-EFAAR component count. The 48 settings are a deterministic grid, not 48
biological replicates. Consequently the factor summaries and two-way variation
decomposition are descriptive and carry no p-values or causal interpretation.

All rows use TVN-EFAAR and no PCA, so batch-method and PCA effects cannot be
estimated from this grid. Exact pruning differs by family: cp_measure compares
0.90 with 0.95, whereas the learned representations compare no pruning with
0.90. The aligned heatmap therefore calls these lower/higher pruning and keeps
the exact values in the CSV.

## Run

```bash
python analyze.py
python test_analyze.py
python analyze.py --verify-only
```

The production input is pinned by byte count and SHA-256. The script aborts on
input drift, missing codecs/families/configurations, duplicate keys, factor
mismatch across codecs, nonfinite metrics, or output checksum drift. It never
writes to the canonical archive and performs no extraction, normalization, or
retrieval rescoring.
