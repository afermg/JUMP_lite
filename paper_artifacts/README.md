# Final paper artifact sources

This directory contains small, deterministic renderers and frozen derived inputs
needed to regenerate final active manuscript figures that are not part of the
original post-sweep pipeline.

Use the top-level `just` interface rather than invoking these scripts directly:

```bash
just artifacts-list
just artifacts-regenerate target-overlap
just artifacts-regenerate strict-heldout
just paper-artifacts-verify /path/to/final/manuscript
```

Generation always targets `data/generated/artifacts/<bundle>/` and refuses an
existing destination. It never overwrites committed reference results or the
manuscript checkout.

- `target_overlap/` reads the committed release perturbation metadata and
  RefChemDB snapshot.
- `strict_heldout/` renders the accepted five-seed figure from three immutable
  N=5 summary tables. The expensive split/refit/scoring run is a separate
  upstream checkpoint; its summaries and hashes are preserved here so the
  published visualization can be regenerated without rerunning normalization.

`paper_artifacts.lock.json` at the repository root records every active figure
and generated table in final manuscript commit `20a1fdaf`. It also distinguishes
computed artifacts from authored diagrams and frozen example-image sources.
