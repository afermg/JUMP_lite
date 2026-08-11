# Cluster-based JUMP-lite representativeness analysis

This directory contains the frozen, CPU-only analysis used to ask whether the
3,832-compound JUMP-lite set spans operational phenotypic profile clusters and
whether cluster assignment retrieves selected compounds beyond acquisition
structure alone. The cluster fit is label-blind: selection labels are not read
until the partition has been frozen.

## Result in one paragraph

The selected set occupies 120 of 128 operational clusters, and those clusters
contain 96.24% of the 95,426 fit-eligible compounds. Selection is nevertheless
non-proportional across clusters (total variation 0.4750; Jensen--Shannon
0.1723 nats). On the eligible universe, out-of-fold average precision is 0.5158
for acquisition structure alone, 0.2777 for cluster alone, and 0.5765 for
structure plus cluster. The combined/structure ratio is 1.118: a conditional
permutation detects an increment (`p=0.000500`), but it is below the
preregistered 1.25x materiality gate. Mean seed ARI is only 0.162, so the
clusters must not be interpreted as stable biological classes.

The publication-ready `cluster_selection_compound_map.{png,pdf}` makes the
three parts of this result explicit: compound-level coverage in a display-only
UMAP, coverage of all 128 clusters, and the out-of-fold retrieval comparison.
The UMAP is fit without labels on the frozen PCA32 profiles before selection
labels are joined. It is visualization, not additional inferential evidence.

## Contents

- `CLUSTER_SELECTION_DESIGN.md`: frozen design and interpretation gates.
- `analyze_cluster_representativeness.py`: label-blind fit and frozen-label
  scoring.
- `score_cluster_partition_sensitivity.py`: fixed partition-sensitivity score.
- `plot_cluster_selection_compounds.py`: deterministic compound figure and
  compact summary table; it never refits the scientific clusters.
- `test_cluster_representativeness.py` and
  `test_cluster_selection_plot.py`: scientific and figure contracts.
- `outputs/profile_space/`: exact derived consensus, selected feature list,
  provenance, and frozen selection/evaluation manifests needed by the runners.
- `outputs/profile_cluster_representativeness_v1/`: reviewed model,
  assignments, tables, reports, snapshots, hashes, and figures.

The 40,554,151-byte consensus is included to avoid recomputing the 13.5 GB
full-JUMP consensus. Input identity checks fail closed on size or SHA-256 drift.

## Reproduce

From the repository root, using the locked project environment:

```bash
uv sync
PY=.venv/bin/python
ROOT=rebuttal/representativeness
OUT=/tmp/profile_cluster_representativeness_reproduction

# OUT must not exist. Fit first, then expose labels only during scoring.
$PY "$ROOT/analyze_cluster_representativeness.py" fit-clusters --output-dir "$OUT"
$PY "$ROOT/analyze_cluster_representativeness.py" score-selection --output-dir "$OUT"
$PY "$ROOT/score_cluster_partition_sensitivity.py" --output-dir "$OUT"

# Regenerate the display artifacts from the committed frozen model/results.
mkdir /tmp/cluster_selection_figure_reproduction
$PY "$ROOT/plot_cluster_selection_compounds.py" \
  --output-dir /tmp/cluster_selection_figure_reproduction

$PY "$ROOT/test_cluster_representativeness.py"
$PY "$ROOT/test_cluster_selection_plot.py"
```

The analysis runners refuse to overwrite an existing result root or partial
result. The figure runner likewise refuses to overwrite any of its outputs.
Runtime fields can differ across a fresh scientific reproduction. Artifact
records use explicit checkout-, output-, or fit-relative path scopes, while the
committed snapshots, frozen hashes, and numeric contracts record the reviewed
run.

## Required qualifications

- “Operational cluster” does not mean a biological class.
- Broad occupancy does not mean proportional or random representation.
- The conditional permutation is a finite-cohort design null, not population
  inference.
- Matched subsets are descriptive sensitivities, not p-values or population
  percentiles.
- The display UMAP does not repair the low stability of cluster boundaries.
