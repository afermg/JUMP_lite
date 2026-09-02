# MQ/D2-E8 synthesis

This directory combines five independently generated, committed Target-2 analyses to explain the pooled Figure 3c MQ/D2-E8 ordering. The active supplementary figure retains only the fixed-recipe paired-bootstrap panel, showing that D2-E8 is not consistently higher than MQ. The exact matched-recipe, normalization-interaction, plate/laboratory, and encoder-effort analyses remain verified supporting sensitivities summarized in the report.

`analyze.py` reads only pinned sibling release CSVs. It does not read canonical datasets and does not rerun compression, extraction, normalization, retrieval, or bootstrapping. Every input is checked by exact size and SHA-256 before use. Generation is transactional and deterministic.

```bash
python rebuttal/mq_d2e8_explanation/synthesis/analyze.py
python rebuttal/mq_d2e8_explanation/synthesis/analyze.py --verify-only
python rebuttal/mq_d2e8_explanation/synthesis/test_analyze.py
```

Release artifacts are under `outputs/release_v1/`. `CAPTION.md` contains the complete manuscript caption. The figure explains why the pooled median should not be interpreted as a general paired codec advantage; it does not establish a causal mechanism, equivalence for unresolved contrasts, denoising, improved biological signal, or broad model-rank stability.
