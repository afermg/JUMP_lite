# Compression-order robustness

Reproducible archived-profile analyses for small-cohort codec-order instability and fixed-recipe Target-2 D10 versus D15 uncertainty.

## Run

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python rebuttal/compression_order_robustness/analyze.py
```

The production runner uses up to 16 spawn-isolated persistent variant processes and protocol-identified checkpoint/resume files under its output directory. Checkpoint identities include the runner hash plus complete frozen support/config/profile identities, so code or input drift rejects reuse. It never writes to `/work/datasets`.

## Bounded smoke

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .venv/bin/python rebuttal/compression_order_robustness/analyze.py \
  --smoke --output-dir /tmp/compression_order_smoke_v3 --workers 16
```

Smoke mode follows the real 16-variant worker path with one subsample and 250 bootstrap replicates. It refuses the production `release_v1` output path, and checkpoint protocol identities prevent cross-mode reuse.

## Validate

```bash
.venv/bin/python rebuttal/compression_order_robustness/test_analyze.py
.venv/bin/python -m py_compile rebuttal/compression_order_robustness/analyze.py \
  rebuttal/compression_order_robustness/test_analyze.py
.venv/bin/python rebuttal/compression_order_robustness/analyze.py --verify-only
```

`--summarize-only` deterministically regenerates compact summary tables, the report, and the figure from final `results/full_subsample_metrics.parquet`, not checkpoints. Before writing any regenerated output, it fail-closed validates the full-profile identity, all current Target-2 hashes, and every matched-full per-unit PA/PC source hash against canonical frozen identities in production provenance.

`artifact_checksums.json` uses paths relative to the output root, so `--verify-only --output-dir <copied-tree>` validates a relocated copy. The inventory separates final release artifacts from `retained_checkpoint_state`; checkpoints are retained execution state, not scientific final results. Because runner hardening changes protocol identity, the retained production checkpoints are checksummed for audit but no longer resumable by the current runner; any future scoring run must start fresh checkpoints.

See `DESIGN.md` for the frozen protocol and `outputs/release_v1/REPORT.md` for results and qualifications.
