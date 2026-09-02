# Deferred speed benchmark suite

Status: planned for after the current manuscript and release deadlines. No benchmark run is authorized by this document.

## Goal

Measure individual extraction throughput for DINOv2, ViT-rand, MorphEM, OpenPhenom, SubCell, and Cellpose plus `cp_measure` under one reproducible protocol. Replace the manuscript's mixed historical/live timing comparison with a common-hardware benchmark when results are complete.

## Benchmark cohort

- Freeze at least 30 local five-channel JUMP-lite sites.
- Stratify sites across plates and low, median, and high object-count ranges.
- Give every method the same decoded `uint16` site inputs in the same deterministic order.
- Record the manifest, dimensions, channel order, input hashes, and selection seed.
- Document model-specific channel selection, tiling, and scaling rather than treating computational inputs as identical.

## Runs

Run every method in two modes:

1. **Steady state:** model/server loaded and inputs warm in the OS cache. This is the primary throughput comparison.
2. **Cold end to end:** include process startup, weight loading, extraction, serialization, and output flush.

For each mode:

- use three unmeasured warm-up sites;
- run five measured repeats over the full frozen manifest;
- deterministically shuffle site order per repeat;
- synchronize CUDA before and after timed GPU stages;
- use a fresh output directory for every repeat;
- run one model at a time on an otherwise idle device;
- pin CPU thread/core settings and record them.

Primary tests use one GPU. GPU scaling at 1, 2, and 4 devices and CPU scaling for Cellpose plus `cp_measure` are separate experiments.

## Timing boundaries and units

Record these stages separately when applicable:

- input decode and preprocessing;
- tiling;
- segmentation;
- embedding inference or `cp_measure` measurement;
- serialization;
- optional site-to-well aggregation.

Primary reported units:

- wall seconds/site;
- sites/hour/device;
- wall seconds/source megapixel;
- GPU-seconds/site;
- peak GPU memory;
- CPU time and peak host RSS;
- input and output bytes/site.

Record tile count for learned models and object count for `cp_measure` as method-specific diagnostics. Do not compare neural forward time alone with the complete Cellpose plus `cp_measure` pipeline. Report `src/extract_features.py --n-jobs 1` aggregation separately from image extraction.

## Provenance and outputs

Build a thin benchmark CLI around the pipeline builders in `analysis/aliby_featurize.py`. Move model definitions into a shared registry and expose model, dataset, device, output, mode, and repeat arguments. Generalize the guarded runner in `rebuttal/representativeness/run_cp_measure_pilot.py` to a frozen multi-site manifest while preserving its CUDA, model-hash, commit, containment, and stale-output checks.

Suggested output layout:

```text
benchmarks/speed/results/<run-id>/
  config.json
  hardware.json
  input_manifest.parquet
  events.jsonl
  per_site.parquet
  per_repeat.parquet
  summary.csv
  summary.json
  validation.json
  logs/
```

Capture hostname, UTC time, command, Git commit/status, CPU and RAM, kernel, GPU model/index/UUID, driver/CUDA, package versions, model identifier and weight hash, `pretrained` flag, thread limits, worker count, input identity, and cache policy.

## Validation and acceptance

A result is publishable only if:

- all methods use identical site keys and input hashes;
- outputs are nonempty and contain finite features;
- expected output counts, schemas, and hashes are recorded;
- ViT-rand records `pretrained=false`;
- no timed run reuses or skips an existing output;
- CUDA synchronization and timing order are validated;
- cold and steady-state results remain separate;
- primary runs do not share GPUs or use inconsistent CPU limits;
- failures, timeouts, and harness cancellations remain in the denominator and are classified;
- a fixed subset rerun confirms determinism or quantifies expected tolerance.

Report per-site median, interquartile range, p95, repeat-level throughput, and bootstrap 95% intervals. Update manuscript speed claims only after this suite passes validation.
