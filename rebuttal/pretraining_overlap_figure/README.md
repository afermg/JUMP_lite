# MorphEM pretraining-overlap exclusion figure

This package renders the supplementary forest plot for the two exclusion
sensitivities supported by MorphEM's acquisition-level CHAMMI-75 inventory:

1. exclude directly matched held-out wells; and
2. exclude every held-out well on a matched plate.

It intentionally does **not** construct or display an OpenPhenom-based
exclusion. OpenPhenom's model card names JUMP collections but provides no
image- or plate-level membership manifest from which to define a defensible
sample exclusion. OpenPhenom remains a comparator in the two MorphEM-defined
sensitivities.

The two frozen input CSVs contain the six displayed contrasts and the two
subset summaries extracted from the archived 50,000-replicate Raw sensitivity.
Their parent-source hashes are recorded in `outputs/provenance.json`. No
feature extraction, normalization, recipe selection, retrieval, or bootstrap
was rerun to make this figure.

## Reproduce

```bash
python rebuttal/pretraining_overlap_figure/plot.py
python rebuttal/pretraining_overlap_figure/plot.py --verify-only
python rebuttal/pretraining_overlap_figure/test_plot.py

# Preferred safe interface (isolated output, no tracked-file overwrite):
just artifacts-regenerate pretraining-overlap-legacy
just artifacts-verify pretraining-overlap-legacy
```

The figure is descriptive and conditional on the archived normalized profiles,
selected Raw recipes, and working product-of-margins bootstrap. The exclusions
change task composition and do not estimate a causal benefit from leakage.
