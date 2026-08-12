# Smoke validation receipt

Date: 2026-08-11

Command (cold run):

```bash
cd src/norm_3
pixi run python ../../rebuttal/postprocessing_validation/run_analysis.py \
  --families cell_count \
  --max-selection-configs 2 \
  --output-dir ../../rebuttal/postprocessing_validation/smoke
```

Result: exit 0 in 25.4 seconds reported by the runner (26.2 seconds shell wall
time). The run created the deterministic 5,375/21,502 validation/test split,
evaluated two CellCount candidates, selected `robustmad_all`, constructed the
166,080-well Raw intersection, and produced a held-out result over 137,005 wells
(18,091 shared controls and 21,502 test treatment IDs).

The identical command was then rerun to test resume behavior. It exited 0 in
2.2 seconds reported by the runner (3.0 seconds shell wall time) and reused the
selection and final checkpoints.

Additional validation:

```text
python -m py_compile ...                                      exit 0
src/norm_3/.pixi/envs/default/bin/python test_run_analysis.py exit 0 (4 tests)
git diff --check                                             exit 0
```

This is a bounded smoke result, not the production model comparison.
