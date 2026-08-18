# Effort sensitivity

This self-contained archived-output analysis compares Target-2 HQ and E3. Both
were declared with JPEG XL distance 1 in `src/compress_tif.py`; HQ omitted the
`effort` argument and E3 set it to 3. Because the historical numeric default and
encoder build were not frozen, the contrast is explicitly approximate.

Run from the repository root:

```bash
/work/users/amunoz/projects/JUMP_lite/.venv/bin/python rebuttal/mq_d2e8_explanation/effort_sensitivity/analyze.py
/work/users/amunoz/projects/JUMP_lite/.venv/bin/python rebuttal/mq_d2e8_explanation/effort_sensitivity/test_analyze.py
```

The runner reads archived sweep/per-unit, image-quality, feature-correlation,
and segmentation summaries. It never writes canonical inputs. One recipe per
family is selected using Zstd PA×PC/100 only and frozen across HQ/E3. PA and PC
margins are independently cluster-bootstrapped under a working-independence
approximation. See `outputs/REPORT.md`.
