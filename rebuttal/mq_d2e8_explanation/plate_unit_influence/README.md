# Plate and unit influence analysis

This analysis explains the Target-2 Figure 3c MQ/D2-E8 contrast using only archived normalized outputs and retrieval tables.

1. It selects one recipe per Figure 3c family using Zstd `PA*PC/100` only, with lexical tie-breaking.
2. It verifies the archived PA/PC unit tables and re-scores every selected MQ/D2-E8 `output.parquet`, requiring exact agreement with archived NAP points before sensitivity results are written.
3. It explicitly intersects codec profile identities within each family, re-scores the exact common population, and derives the 306 PA compound and 201 PC target units used for the symmetric contribution decomposition.
4. It uses that same common population for the full contrast and after omitting each of the four plate/laboratory pairs.

The common-population step is material only for `cp_measure`: its selected D2-E8 and MQ outputs contain 1,519 and 1,520 rows, with 1,519 common. The four learned-family pairs each contain the same 1,536 identities. Intersections are recorded in `coverage_manifest.csv`; they are never silent.

## Run

```bash
PYTHONPATH=src /work/users/amunoz/projects/JUMP_lite/.venv/bin/python \
  rebuttal/mq_d2e8_explanation/plate_unit_influence/analyze.py
PYTHONPATH=src /work/users/amunoz/projects/JUMP_lite/.venv/bin/python \
  rebuttal/mq_d2e8_explanation/plate_unit_influence/test_analyze.py
```

The deterministic release is under `outputs/release_v1/`. `--verify-only` validates the exact output inventory plus the frozen sweep, selected metrics/config/profile/per-unit inputs, and scoring source hashes. Repository-owned scoring sources are recorded with normalized repository-relative paths so verification remains valid after checkout relocation; canonical external data inputs retain canonical absolute paths. Unsafe or traversing relative paths fail closed.

## Interpretation limits

Plate and laboratory are perfectly confounded. Leave-one-out results are finite-cohort sensitivity estimates, and unit contributions are descriptive influence diagnostics rather than causal effects. The PA/PC product does not encode their unknown covariance. No result establishes denoising or improved biological signal.
