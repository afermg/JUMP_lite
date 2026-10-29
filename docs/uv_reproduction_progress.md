# Example status

[PR #11](https://github.com/afermg/JUMP_lite/pull/11) provides:

- [Feature and annotation workflows](uv_profile_demo.md): export model features,
  normalize them and compare annotation versions.
- [Public-image inference](uv_jump_lite.md): a pinned OpenPhenom example and
  reuse of existing cp_measure results.
- [Recipe replay](uv_jump_benchmark.md): separate v11-style and strict CPU paths.

Validation covered four image runs and two recipe-replay runs. The latter used
4,564 wells, four codecs and two recipes; 87 output files matched between runs.
Execution artifacts remain outside the repository; six shared smoke tests cover
the main input, fitting and reference checks.

Remaining work: complete recipe/model/seed coverage, independent published-score
and GPU-parity comparisons, historical model lineage, full MOTIVE inputs/evaluation,
and macOS/Windows testing. Public input sources are tracked in [DATA_SOURCES.md](../DATA_SOURCES.md).
