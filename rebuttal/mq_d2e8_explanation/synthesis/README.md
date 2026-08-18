# MQ/D2-E8 synthesis

This directory combines five independently generated, committed Target-2 analyses into the supplementary figure used to explain the pooled Figure 3c MQ/D2-E8 ordering.

`analyze.py` reads only pinned sibling release CSVs. It does not read canonical datasets and does not rerun compression, extraction, normalization, retrieval, or bootstrapping. Every input is checked by exact size and SHA-256 before use. Generation is transactional and deterministic.

```bash
python rebuttal/mq_d2e8_explanation/synthesis/analyze.py
python rebuttal/mq_d2e8_explanation/synthesis/analyze.py --verify-only
python rebuttal/mq_d2e8_explanation/synthesis/test_analyze.py
```

Release artifacts are under `outputs/release_v1/`. `CAPTION.md` contains the complete manuscript caption. The figure supports a pooled-median and analysis-pipeline-sensitivity explanation; it does not support denoising or improved biological signal.
