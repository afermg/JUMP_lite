# Effort sensitivity

This self-contained archived-output analysis compares Target-2 HQ and E3. Both
were declared with JPEG XL distance 1 in `src/compress_tif.py`; HQ omitted the
`effort` argument and E3 set it to 3. Because the historical numeric default and
encoder build were not frozen, the contrast is explicitly approximate.

Run from the repository root:

```bash
uv run python rebuttal/mq_d2e8_explanation/effort_sensitivity/analyze.py
uv run python rebuttal/mq_d2e8_explanation/effort_sensitivity/test_analyze.py

# Preferred safe interface: writes only below data/generated/artifacts/.
just artifacts-regenerate mq-d2e8-effort
```

The runner reads archived sweep/per-unit, image-quality, feature-correlation,
and segmentation summaries. It never writes canonical inputs. All 82 inputs,
the five Zstd-selected recipes, and the durable repository declaration of the
codec parameters are frozen in `frozen_inputs.json`; drift aborts. Releases are
built in a clean staging directory and promoted only after checksum validation.
Validate the frozen inputs and committed output without writing via:

```bash
uv run python rebuttal/mq_d2e8_explanation/effort_sensitivity/analyze.py --verify-only
# or
just artifacts-verify mq-d2e8-effort
```

One recipe per family is selected using Zstd PA×PC/100 only and frozen across
HQ/E3. PA and PC margins are independently cluster-bootstrapped under a
working-independence approximation. This approximate HQ/E3 sensitivity is not
a distance-by-effort factorial and cannot explain MQ versus D2-E8. See
`outputs/REPORT.md`.
