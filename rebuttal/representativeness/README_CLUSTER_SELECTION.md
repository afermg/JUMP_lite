# Cluster-based JUMP-lite representativeness analysis

This directory contains a frozen, CPU-only label-blind cluster fit and two
separate scoring packages. The historical package scores 3,832 treatments. The
primary current-release package scores the 3,775 treatments in tracked release
metadata. These packages are versioned separately and their cohort identities
must not be confused.

## Current-release result

Tracked release metadata has exactly 3,776 compound identifiers: 3,775
treatments plus one excluded negative control. The 3,775 treatments are a
strict subset of the historical manifest (57 historical-only; no current-only
identifiers), and all have fit-eligible frozen assignments.

The current treatments occupy 120 of 128 operational clusters, and those
clusters contain 96.24% of the 95,426 fit-eligible compounds. Coverage is broad
but non-proportional (total variation 0.4750; Jensen--Shannon 0.1723 nats).
Eligible-universe OOF average precision is 0.5135 for acquisition structure,
0.2731 for cluster alone, and 0.5733 for structure plus cluster. The
combined/structure ratio is 1.116: the conditional permutation detects an
increment (`p=0.000500`), but it remains below the preregistered 1.25x
materiality gate. Mean seed ARI is only 0.162, so clusters are not biological
classes. The historical 3,832-sized matched comparators are omitted rather than
reused as current evidence.

## Historical result

The frozen 3,832-treatment package reported 120/128 occupied clusters, 96.24%
eligible mass coverage, and a combined/structure AP ratio of 1.118. These
values remain reproducible historical evidence but are not primary
current-release values.

The publication-ready `cluster_selection_compound_map.{png,pdf}` makes the
three parts of this result explicit: compound-level coverage in a display-only
UMAP, coverage of all 128 clusters, and the out-of-fold retrieval comparison.
The UMAP is fit without labels on the frozen PCA32 profiles before selection
labels are joined. It is visualization, not additional inferential evidence.

## Contents

- `CLUSTER_SELECTION_DESIGN.md`: unchanged historical frozen design.
- `CLUSTER_SELECTION_RELEASE_V1_ADDENDUM.md`: current cohort contract and
  unchanged interpretation gates.
- `analyze_cluster_representativeness.py`: historical label-blind fit and
  frozen-label scoring.
- `rescore_cluster_representativeness_release_v1.py`: fail-closed current
  manifest derivation, scoring, and figure regeneration without refitting.
- `export_cluster_selection_control.py`: fail-closed, result-only exporter for
  the compact SI package; it verifies and copies canonical outputs without
  fitting, rescoring, or recomputing scientific metrics.
- `score_cluster_partition_sensitivity.py`: fixed partition-sensitivity score.
- `plot_cluster_selection_compounds.py`: deterministic compound figure and
  compact summary table; it never refits the scientific clusters.
- `test_cluster_representativeness.py`, `test_cluster_selection_plot.py`, and
  `test_cluster_representativeness_release_v1.py`: historical and current
  scientific/figure contracts.
- `outputs/profile_space/`: exact derived consensus, selected feature list,
  provenance, and frozen selection/evaluation manifests needed by the runners.
- `outputs/profile_cluster_representativeness_v1/`: immutable historical
  3,832-treatment model/results package.
- `outputs/profile_cluster_representativeness_release_v1/`: current-release
  3,775-treatment manifest, scores, report, hashes, and figures; it references
  the hash-frozen historical fit and UMAP and contains no matched-comparator
  output.

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

# Reproduce current-label scoring and figures without any fit/refit.
$PY "$ROOT/rescore_cluster_representativeness_release_v1.py" \
  --output-dir /tmp/profile_cluster_representativeness_release_v1_reproduction

# Export the compact SI package from committed canonical results only.
$PY "$ROOT/export_cluster_selection_control.py" \
  --output-dir /tmp/cluster_selection_control

# Historical validation remains separate.
$PY "$ROOT/test_cluster_representativeness.py"
$PY "$ROOT/test_cluster_selection_plot.py"
$PY "$ROOT/test_cluster_representativeness_release_v1.py"
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
