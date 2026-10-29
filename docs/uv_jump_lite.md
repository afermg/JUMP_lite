# Extract features from public images

Run a small OpenPhenom example with `uv`:

```sh
uv run examples/uv_jump_lite.py --output real-run
```

The runner downloads public metadata and 55 TIFFs for 11 wells, extracts 384
features per well on CPU, normalizes against five controls and evaluates shared-target
retrieval. It uses one top-left 256×256 tile per site, in AGP/DNA/ER/Mito/RNA order.
The OpenPhenom revision is pinned to `0f92333685f6e9f031b804c70fe246f9b05ae90d`.

Allow roughly 0.8 GB for image/model downloads, plus dependencies, and several GiB
of RAM. Linux x86_64 is tested. For locked dependencies, retain the adjacent lock
file and use `uv run --locked`. Output and cache directories must be separate;
each run needs a new output directory.

## Useful options

| Option | Use |
|---|---|
| `--prepare-only` | Download metadata and select wells before inference |
| `--cache PATH` | Reuse verified downloads; default `jump-lite-download-cache` |
| `--image-lock RUN/image_lock.json` | Require the same TIFF hashes on a repeat run |
| `--cp-measure measurements.json` | Reuse existing cell measurements |

Outputs include raw and normalized feature CSVs, fitted parameters, retrieval
scores, input receipts and `provenance.json`. The latter is written on completion.
Use the [feature guide](uv_profile_demo.md) for custom models and annotations.

## Reuse cp_measure outputs

Supply a manifest covering the 11 selected sites. Obtain their keys with
`--prepare-only`. The manifest has this structure; repeat `tables` for every site:

```json
{
  "schema_version": 1,
  "measurement_contract": {"...": "copy contract fields from the original receipts"},
  "features": ["0/max/sizeshapeArea"],
  "units": {"0/max/sizeshapeArea": "pixels^2"},
  "object_type": {"column": "metadata_object", "value": "cell"},
  "tables": [{
    "site_key": "<selected site key>",
    "path": "profiles/site.parquet",
    "sha256": "<table hash>",
    "object_id": "metadata_label",
    "receipt": {"path": "profiles/site.parquet.receipt.json", "sha256": "<receipt hash>"}
  }]
}
```

Copy these contract fields from the receipts: `contract_version`, `channels`,
`codec`, `schema_sha256`, `parity_mode`, `legacy_intensity_semantics`, and
`cp_measure` from `dependency_contract.cp_measure`. Paths are relative to the
manifest. All sites must use the same contract.

The runner checks table hashes, site/object identities and units, then takes
median cell features per site. It reuses extraction results. These full-site
measurements have broader spatial coverage than the single image tile.

## Scope and checks

This small example uses control-fitted raw MAD and cosine recall. For TVN and
production PA/PC/NAP, use [recipe replay](uv_jump_benchmark.md).
`--mode full` writes the outstanding requirements and exits 3.

Four public-image runs completed, including an isolated-environment repeat.
Repeated image features matched; measurement reuse covered 1,719 cells.
Run the offline checks with `uv run tests/test_uv_examples.py -v`.
