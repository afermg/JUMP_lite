# JUMP_lite

Reproduction code for the WACV submission **"<paper title>"**, which evaluates
how lossy image compression affects downstream cell-painting analysis (model
ranking, MOTIVE retrieval, segmentation, feature stability).

## What's here

```
.
├── README.md                  this file
├── REPRODUCE.md               step-by-step reproduction guide
├── PIPELINE.md                technical pipeline reference
├── justfile                   all recipes (run `just --list`)
├── flake.nix                  Nix dev shell (provides pixi + system libs)
├── pyproject.toml / uv.lock   Python deps (managed by uv)
├── src/                       compression, feature extraction, motive eval
├── src/norm_3/                normalization sweep pipeline
├── analysis/                  per-figure analysis scripts
├── scripts/                   one-shot data-prep utilities
├── metadata/                  curated metadata used by the pipeline
```

## Quick start

```bash
# 1. Environment
nix develop                    # pixi + system libs auto-activated

# 2. Verify
just --list                    # show recipes
just check-env                 # confirm Python + CUDA work

# 3. Reproduce paper figures (assumes you have the per-(model, codec) sweep
#    outputs — see REPRODUCE.md §2 for what to drop into data/)
just clean-results
just reproduce                 # ~55 min, ~350 MB written to data/results/
```

End-to-end from raw images is `just produce-paper` — many hours, hundreds of
GB of disk. See REPRODUCE.md for the full pipeline.

## Smoke test

To validate the sweep machinery without committing to a full run, the smoke
recipes collapse the per-(model, codec) grid from 48 → 4:

```bash
just sweep-v11-lite-smoke 0 4  # GPU 0, 4 joblib workers — ~20–40 min
```

## Reproducing on a different machine

Set `DATA_ROOT` to point at wherever your raw data lives:

```bash
export DATA_ROOT=/my/storage/jump_data
just check-data                # verifies all upstream paths resolve
just produce-paper             # end-to-end
```

The default is `./data`, so a self-contained `data/` directory inside the repo
works without configuration.

## License

MIT. See [LICENSE](LICENSE).

## Citation

```
TODO bibtex
```
