# Paired MQ versus D2-E8 recipe audit

This analysis explains the marginal codec ordering in Target-2 Figure 3c without rerunning extraction, normalization, retrieval, or the canonical sweep.

Figure 3c pools five representation families and 48 normalization recipes per codec. `analyze.py` restricts the frozen `sweep_results.csv` to the exact five displayed families, requires complete D2-E8 and MQ recipe sets, and pairs the unscaled NAP product (`PA_mean_nap * PC_mean_nap`) by `(family, config)`.

## Run

```bash
.venv/bin/python rebuttal/mq_d2e8_explanation/paired_recipes/analyze.py
.venv/bin/python rebuttal/mq_d2e8_explanation/paired_recipes/test_analyze.py
.venv/bin/python rebuttal/mq_d2e8_explanation/paired_recipes/analyze.py --verify-only
```

The canonical input is read-only and pinned by exact byte size and SHA-256. Outputs are written beneath `outputs/release_v1/` and inventoried by relocatable checksums.

## Interpretation

The normalization recipes form a structured sensitivity grid, not 48 independent biological experiments. Recipe-pair summaries are descriptive and carry no inferential p-values. They do not establish denoising or biological improvement.
