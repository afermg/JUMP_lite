# /// script
# requires-python = "==3.12.12"
# dependencies = [
#   "numpy==1.26.4", "polars==1.35.2", "tifffile==2025.10.16",
#   "imagecodecs==2025.3.30", "torch==2.10.0", "torchvision==0.25.0",
#   "transformers==4.46.1", "timm==1.0.24", "certifi==2025.10.5",
# ]
# [tool.uv.sources]
# torch = [{ index = "pytorch-cpu", marker = "sys_platform != 'darwin'" }]
# torchvision = [{ index = "pytorch-cpu", marker = "sys_platform != 'darwin'" }]
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
# ///
"""Real public JUMP-lite -> frozen MAE -> diagnostic retrieval, not paper parity.

Copy this file anywhere; run with uv. CPU only, one thread, no model server.
The default downloads ~0.7 GB of model weights plus metadata and <=55 TIFFs.
--mode full is a readiness check, NOT a silently reduced benchmark execution.
See docs/uv_jump_lite.md for scope, limitations and cp_measure reuse.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import ssl
import sys
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ[_name] = "1"

import certifi  # noqa: E402 - set numeric thread limits before importing libraries
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

BASE = "https://cellpainting-gallery.s3.amazonaws.com/"
METADATA = BASE + "cpg0016-jump/source_all/workspace/publication_data/2026_jump_lite/metadata/v1.0/"
RELEASE = {
    "jump_lite_site_index.parquet": (6914554, "ec0e8a87ee66d640025617c360165d20d16a35bde655e7f78854a69651e87eec"),
    "jump_lite_perturbation_metadata.parquet": (949745, "bbedb37f12fdeb9a09e72abaa166159d286052dc1201166d811c5310db5cd7e1"),
    "jump_lite_refchem_annotations.parquet": (318521, "5a766a222907e1b3805f84cc2eccad0e5cd2f3eaac1add4983689853601db960"),
}
REVISION = "0f92333685f6e9f031b804c70fe246f9b05ae90d"
MODEL_URL = f"https://huggingface.co/recursionpharma/OpenPhenom/resolve/{REVISION}/"
MODEL_HASHES = {
    "model.safetensors": "f6e5f1c97101331b1574c9e4b99623260191c55eea5d98e40460849c0e4c4d47",
    "config.json": "cdbf5b22f3aed6a729fa690c804861c931b511e2de74c8526189a38b1c3a52cf",
    "huggingface_mae.py": "2fd6da12b32a548db38e271d67072f85c82358e1e5aceac19adf06d151af68f1",
    "loss.py": "6b3fa60b6f7b89d90def3b9151e67f6c393d6de2f7895da1eada545a070ff776",
    "mae_modules.py": "5bea326273b234824e0fdfb31fe7a226a1ea1d5ff5502448c7ad675d48d90dfc",
    "mae_utils.py": "3fc34b749011afc7053b038745bddce02d34061f7e098a216ed8133195bf2cef",
    "masking.py": "b6eea7c755171aea3722d4d9d755e1ddb0e2db358d4e22bd4f87a3d25438ba33",
    "normalizer.py": "78923002ab47743d3866e7008e809187cd5b104dc66e60695519a4d6fc54ae2f",
    "vit.py": "964f7a1e59e7b3a3edca0289ca35c387fd186305d3ef2aed7d245855c53b9bef",
}
KEYS = ["Metadata_Source", "Metadata_Batch", "Metadata_Plate", "Metadata_Well"]
CHANNELS = ["AGP", "DNA", "ER", "Mito", "RNA"]
MAX_WELLS = 11
MAX_IMAGE_BYTES = 32 * 1024**2
MAX_MEASUREMENT_BYTES = 32 * 1024**2


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dump(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def no_symlinks(path):
    path = Path(os.path.abspath(path))
    require(all(not parent.is_symlink() for parent in (path, *path.parents)),
            f"Symlinked destination refused: {path}")


def checked_url(url):
    """No credentials, arbitrary hosts, private S3 buckets or query parameters."""
    if url.startswith("s3://cellpainting-gallery/"):
        url = BASE + url[len("s3://cellpainting-gallery/"):]
    parsed = urlparse(url)
    require(parsed.scheme == "https" and parsed.hostname in
            {"cellpainting-gallery.s3.amazonaws.com", "huggingface.co"}
            and parsed.username is None and parsed.password is None
            and not parsed.query and not parsed.fragment and parsed.port is None,
            f"Not an approved public input URL: {url}")
    return url


def fetch(url, cache, expected=None, size=None, maximum=MAX_IMAGE_BYTES):
    """Bounded download; cache entries are immutable and rehashed on every reuse.

    Metadata/weights have predeclared SHA256. TIFFs have first-download receipts;
    use --image-lock to require prior hashes, including on a fresh cache.
    """
    url = checked_url(url)
    no_symlinks(cache)
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode()).hexdigest()
    target, receipt = cache / key, cache / (key + ".json")
    no_symlinks(target)
    no_symlinks(receipt)
    if target.exists() or receipt.exists():
        require(target.is_file() and receipt.is_file(), "Incomplete cache entry; use a new cache")
        record = json.loads(receipt.read_text())
        require(record["url"] == url and record["sha256"] == digest(target), "Cache identity/hash drift")
        require(record["bytes"] == target.stat().st_size, "Cache size drift")
    else:
        # mkstemp prevents collisions between writers; do not replace existing entries.
        fd, temporary = tempfile.mkstemp(prefix="download-", dir=cache)
        try:
            count = 0
            with os.fdopen(fd, "wb") as out, urlopen(
                    url, timeout=60, context=ssl.create_default_context(cafile=certifi.where())) as response:
                require(urlparse(response.url).scheme == "https", "Insecure redirect")
                length = response.headers.get("Content-Length")
                if length is not None:
                    require(int(length) <= maximum, "Download Content-Length exceeds bound")
                while chunk := response.read(1024 * 1024):
                    count += len(chunk)
                    require(count <= maximum, "Download exceeds byte bound")
                    out.write(chunk)
            record = {"url": url, "bytes": count, "sha256": digest(temporary)}
            require(size is None or count == size, "Downloaded size mismatch")
            require(expected is None or record["sha256"] == expected, "Downloaded SHA256 mismatch")
            # Exclusive target creation: another writer cannot be overwritten.
            with target.open("xb") as out, open(temporary, "rb") as inp:
                import shutil
                shutil.copyfileobj(inp, out)
            dump(receipt, record)
        finally:
            Path(temporary).unlink(missing_ok=True)
    require(record["bytes"] <= maximum, "Cached input exceeds bound")
    require(size is None or record["bytes"] == size, "Input size mismatch")
    require(expected is None or record["sha256"] == expected, "Input SHA256 mismatch")
    return target, record


def select_cohort(sites, wells, annotations):
    """One fixed metadata-only rule; never inspect image values or model scores.

    Lexical first eligible group_high plate; five controls, six identities (one
    well each). Include the first annotated target pair, then lexical fillers.
    One lexical first site per selected well. This is annotation-enriched QC.
    """
    require(wells["Metadata_Well_Key"].n_unique() == len(wells), "Duplicate release well")
    require(sites["Metadata_Site_Key"].n_unique() == len(sites), "Duplicate release site")
    pairs = (annotations.filter((pl.col("activity_class") == "active") &
                                (pl.col("support") >= 5) & pl.col("target").is_not_null())
             .select("Metadata_JCP2022", "target").unique().sort("target", "Metadata_JCP2022"))
    targets = {}
    for row in pairs.to_dicts():
        targets.setdefault(row["target"], set()).add(row["Metadata_JCP2022"])
    available = sites.select(KEYS).unique()
    candidates = wells.filter(pl.col("Metadata_Group") == "group_high").join(
        available, on=KEYS, how="semi").with_columns(
        pl.concat_str(KEYS[:3], separator="__").alias("physical_plate"))
    for plate in sorted(candidates["physical_plate"].unique()):
        on_plate = candidates.filter(pl.col("physical_plate") == plate)
        controls = on_plate.filter(pl.col("Metadata_pert_type") == "negcon").sort("Metadata_Well_Key").head(5)
        treatments = on_plate.filter(pl.col("Metadata_pert_type") == "trt")
        counts = treatments.group_by("Metadata_JCP2022").len()
        eligible = set(counts.filter(pl.col("len") >= 1)["Metadata_JCP2022"])
        if len(controls) < 5 or len(eligible) < 6:
            continue
        pair = next((sorted(ids & eligible)[:2] for _, ids in sorted(targets.items())
                     if len(ids & eligible) >= 2), None)
        if pair is None:
            continue
        ids = sorted(set(pair) | set(sorted(eligible - set(pair))[:4]))
        chosen = pl.concat([controls, *[
            treatments.filter(pl.col("Metadata_JCP2022") == identity).sort("Metadata_Well_Key").head(1)
            for identity in ids]])
        chosen = chosen.join(sites.sort("Metadata_Site_Key").unique(subset=KEYS, keep="first"),
                             on=KEYS, how="left", validate="1:1").sort("Metadata_Well_Key")
        require(len(chosen) == MAX_WELLS, "Unexpected cohort size")
        edges = []
        for query in ids:
            for reference in ids:
                shared = sorted(t for t, members in targets.items() if query in members and reference in members)
                if query != reference and shared:
                    edges.append({"query": query, "reference": reference, "shared_targets": shared})
        rows = []
        for row in chosen.to_dicts():
            rows.append({"sample_id": row["Metadata_Well_Key"], "plate": plate,
                         "identity": row["Metadata_JCP2022"], "group": "group_high",
                         "control": row["Metadata_pert_type"] == "negcon",
                         "site_key": row["Metadata_Site_Key"],
                         "channels": {c: checked_url(row["URL_Orig" + c]) for c in CHANNELS}})
        return rows, edges
    raise ValueError("No plate satisfies the frozen cohort rule; no fallback cohort")


def model_load(cache, output):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261029)
    torch.use_deterministic_algorithms(True)
    # Do not let transformers consult a shared model cache or the network.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_MODULES_CACHE"] = str(output / "hf-modules")
    package = output / "openphenom_frozen"
    package.mkdir()
    records = {}
    checkpoint = None
    for name, sha in MODEL_HASHES.items():
        size = 712434294 if name == "model.safetensors" else None
        source, records[name] = fetch(MODEL_URL + name, cache, sha, size, maximum=size or 1024**2)
        if name == "model.safetensors":
            checkpoint = source  # Large immutable weights stay in the download cache.
        else:
            import shutil
            shutil.copyfile(source, package / name)
            require(digest(package / name) == sha, "Copied model source changed")
    (package / "__init__.py").touch()
    sys.path.insert(0, str(output.resolve()))
    module = importlib.import_module("openphenom_frozen.huggingface_mae")
    config = module.MAEConfig.from_pretrained(package, local_files_only=True)
    model = module.MAEModel(config)
    # Upstream file is a torch archive despite its safetensors extension.
    archive = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(archive["state_dict"], strict=True)
    model.requires_grad_(False).eval()
    require(not any(p.requires_grad for p in model.parameters()), "Model is not frozen")
    dump(output / "model_provenance.json", {"repository": "recursionpharma/OpenPhenom",
         "revision": REVISION, "files": records, "strict_load": True, "device": "cpu",
         "channels": CHANNELS, "input": "original TIFF uint16, top-left 256x256 tile, float32",
         "normalization": "checkpoint predict(): pixels/255 then per-tile InstanceNorm2d",
         "pooling": "final mean non-CLS tokens, 384 dimensions", "historical_profile_parity": "unverified"})
    return model


def image_features(rows, cache, output, image_lock=None):
    import tifffile
    import torch
    expected = None
    if image_lock is not None:
        require(image_lock.stat().st_size <= 1024**2, "Image lock too large")
        expected = json.loads(image_lock.read_text())
        require(set(expected) == {url for row in rows for url in row["channels"].values()},
                "Image lock must describe exactly the frozen cohort URLs")
        require(all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in expected.values()),
                "Invalid image lock digest")
    model = model_load(cache, output)
    features, receipts = [], []
    for row in rows:
        channels = []
        shape = None
        for name in CHANNELS:
            url = row["channels"][name]
            path, receipt = fetch(url, cache, expected[url] if expected else None)
            with tifffile.TiffFile(path) as tiff:
                require(len(tiff.pages) == 1, "Expected one TIFF page")
                page = tiff.pages[0]
                require(len(page.shape) == 2 and min(page.shape) >= 256 and max(page.shape) <= 4096,
                        "Unexpected image dimensions")
                require(page.dtype == np.dtype("uint16"), "Expected original uint16 pixels")
                pixels = page.asarray()
            shape = shape or pixels.shape
            require(pixels.shape == shape, "Channel dimensions differ")
            channels.append(pixels[:256, :256].astype(np.float32))
            receipts.append({**receipt, "site_key": row["site_key"], "channel": name,
                             "shape": list(shape)})
        batch = torch.from_numpy(np.stack(channels)[None])
        with torch.inference_mode():
            vector = model.predict(batch).numpy()
            if not features:
                require(np.array_equal(vector, model.predict(batch).numpy()), "Inference repeat differs")
        require(vector.shape == (1, 384) and np.isfinite(vector).all(), "Invalid MAE output")
        features.append(vector[0])
        print(f"MAE {len(features)}/{len(rows)}: {row['sample_id']}", flush=True)
    dump(output / "image_receipts.json", receipts)
    dump(output / "image_lock.json", {r["url"]: r["sha256"] for r in receipts})
    return np.array(features, dtype=np.float64)


def measurement_contract(receipt):
    return {key: receipt[key] for key in ("contract_version", "channels", "codec", "schema_sha256",
            "parity_mode", "legacy_intensity_semantics")} | {"cp_measure": receipt["dependency_contract"]["cp_measure"]}


def measurement_features(manifest_path, rows):
    """Read existing cell measurements; no segmentation/extraction or relabeling.

    Explicit receipt schema makes site identity, object IDs, units and selected
    columns mandatory. One contract per run; never combine cp_measure versions.
    """
    require(manifest_path.stat().st_size <= 1024**2, "Measurement manifest too large")
    spec = json.loads(manifest_path.read_text())
    require(spec.get("schema_version") == 1 and spec.get("measurement_contract"), "Missing measurement contract")
    features = spec["features"]
    require(1 <= len(features) <= 512 and len(features) == len(set(features)), "Invalid measurement feature list")
    require(all(isinstance(f, str) and f and not f.lower().startswith("metadata") for f in features),
            "Metadata cannot be a measurement")
    require(set(spec["units"]) == set(features) and all(isinstance(v, str) and v.strip()
            for v in spec["units"].values()), "Declare units for every measurement; use pixels, not assumed microns")
    compartment = spec["object_type"]
    require(compartment["column"] == "metadata_object" and compartment["value"] in {"cell", "nuclei"},
            "Declare cell or nuclei compartment explicitly")
    tables = spec["tables"]
    require(len(tables) == len(rows) and len({t["site_key"] for t in tables}) == len(tables),
            "Need exactly one unique measurement table per selected site")
    by_site = {t["site_key"]: t for t in tables}
    require(set(by_site) == {r["site_key"] for r in rows}, "Measurement/cohort site mismatch")
    result, evidence = [], []
    for row in rows:
        table = by_site[row["site_key"]]
        path = (manifest_path.parent / table["path"]).resolve()
        require(path.is_file() and path.stat().st_size <= MAX_MEASUREMENT_BYTES, "Measurement table exceeds size bound")
        require(digest(path) == table["sha256"], "Measurement checksum mismatch")
        receipt_path = (manifest_path.parent / table["receipt"]["path"]).resolve()
        require(receipt_path.stat().st_size <= 1024**2 and digest(receipt_path) == table["receipt"]["sha256"],
                "Measurement receipt checksum mismatch")
        receipt = json.loads(receipt_path.read_text())
        require(receipt["status"] == "complete" and receipt["site_key"] == row["site_key"]
                and receipt["output_sha256"] == table["sha256"] and receipt["output_bytes"] == path.stat().st_size,
                "Measurement receipt identity mismatch")
        require(measurement_contract(receipt) == spec["measurement_contract"], "Mixed/drifting measurement contracts")
        require(table["object_id"] not in features, "Object identity is not a feature")
        columns = ["Metadata_Site_Key", table["object_id"], *features]
        df = (pl.scan_parquet(path).filter(pl.col(compartment["column"]) == compartment["value"])
              .select(columns).head(100001).collect())
        require(len(df) == receipt["object_counts"][compartment["value"]], "Compartment object count mismatch")
        require(0 < len(df) <= 100000, "Empty or oversized cell table")
        require(all(df.schema[f].is_numeric() for f in features), "Measurements must have numeric types")
        require(df["Metadata_Site_Key"].null_count() == 0 and
                df["Metadata_Site_Key"].unique().to_list() == [row["site_key"]], "Cell table site mismatch")
        require(df[table["object_id"]].null_count() == 0 and df[table["object_id"]].n_unique() == len(df),
                "Missing/duplicate cell identity")
        values = df.select(features).to_numpy().astype(np.float64)
        require(np.isfinite(values).all(), "Nonfinite measurements; no silent cell exclusion")
        result.append(np.median(values, axis=0))
        evidence.append({"site_key": row["site_key"], "sha256": table["sha256"], "objects": len(df),
                         "receipt_sha256": table["receipt"]["sha256"], "mask_sha256": receipt.get("mask_sha256"),
                         "image_root": receipt.get("image_root")})
    return np.array(result), {"manifest_sha256": digest(manifest_path), "features": features,
        "units": spec["units"], "object_type": compartment,
        "measurement_contract": spec["measurement_contract"], "tables": evidence,
        "aggregation": "median over ALL supplied full-site cells; one selected site per well",
        "comparison_warning": "Full-site cell medians and a single top-left image tile have different spatial support"}


def normalize(rows, x):
    require(x.ndim == 2 and x.shape[0] == len(rows) and np.isfinite(x).all(), "Invalid feature matrix")
    output = np.empty_like(x, dtype=np.float64)
    fits = []
    for plate in sorted({r["plate"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["plate"] == plate]
        controls = [i for i in indices if rows[i]["control"]]
        require(len(controls) >= 3, "At least three calibration controls per plate")
        median = np.median(x[controls], axis=0)
        mad = np.median(np.abs(x[controls] - median), axis=0)
        require(np.isfinite(mad).all() and (mad > 1e-18).all(), "Zero/tiny/nonfinite control MAD; no automatic feature selection")
        output[indices] = (x[indices] - median) / (mad + 1e-18)
        fits.append({"plate": plate, "fit_samples": [rows[i]["sample_id"] for i in controls],
                     "median": median.tolist(), "raw_mad": mad.tolist()})
    require(np.isfinite(output).all(), "Nonfinite normalized profiles")
    return output, fits


def evaluate(rows, x, edges):
    ids = sorted({r["identity"] for r in rows if not r["control"]})
    require(2 <= len(ids) <= 1000, "Invalid retrieval size")
    profiles = np.array([np.median(x[[i for i, r in enumerate(rows) if
                        r["identity"] == identity and not r["control"]]], axis=0) for identity in ids])
    norms = np.linalg.norm(profiles, axis=1)
    require(np.isfinite(norms).all() and (norms > 0).all(), "Invalid consensus cosine norm")
    similarities = (profiles / norms[:, None]) @ (profiles / norms[:, None]).T
    positives = {}
    seen = set()
    for edge in edges:
        q, r = edge["query"], edge["reference"]
        require(q in ids and r in ids and q != r and (q, r) not in seen, "Invalid/duplicate annotation edge")
        seen.add((q, r))
        positives.setdefault(q, set()).add(r)
    require(positives, "No annotated queries")
    per_query = []
    for q in sorted(positives):
        ranked = sorted((r for r in ids if r != q), key=lambda r: (-similarities[ids.index(q), ids.index(r)], r))
        result = {"query": q, "positives": len(positives[q]), "candidates": len(ranked), "top_reference": ranked[0]}
        for percent in (1, 5, 10):
            k = max(1, len(ranked) * percent // 100)
            result[f"k_{percent}"] = k
            result[f"recall@{percent}%"] = len(positives[q] & set(ranked[:k])) / len(positives[q])
        per_query.append(result)
    return {"metrics": {f"recall@{p}%": float(np.mean([r[f"recall@{p}%"] for r in per_query])) for p in (1, 5, 10)},
            "per_query": per_query, "queries_without_positives": sorted(set(ids) - set(positives)),
            "identities": ids, "directed_edges": len(seen),
            "interpretation": "annotation-enriched real-input software smoke test; not published benchmark scores"}


def write_profiles(path, rows, x):
    data = [{"Metadata_Sample": r["sample_id"], "Metadata_Site_Key": r["site_key"],
             "Metadata_JCP2022": r["identity"], "Metadata_Plate": r["plate"],
             "Metadata_Group": r["group"], "Metadata_negcon": r["control"],
             "Metadata_Split": "train" if r["control"] else "eval",
             **{f"Feature_{j:03d}": float(v) for j, v in enumerate(vector)}} for r, vector in zip(rows, x)]
    pl.DataFrame(data).write_csv(path, float_precision=17)


def full_readiness(output):
    """Do not launch an unbounded production run or invent missing public inputs."""
    report = {"status": "blocked", "published_score_parity": False, "benchmark_executed": False,
        "blockers": [
            "Missing public, hash-pinned production MOTIVE relationship and identifier-mapping bundle (DATA_SOURCES.md).",
            "Missing public frozen production archive: 7285 JUMP-lite, 2860 Target-2 and 1055 strict-MOTIVE metrics (REPRODUCE.md).",
            "Historical OpenPhenom feature checkpoint, preprocessing and aggregation lineage is unverified.",
            "CPU OpenPhenom recipe replay exists separately; full-grid GPU numerical parity and full MOTIVE remain unverified.",
            "Published reference cohort, recipes and expected scores must be bound together before exact comparison.",
        ], "local_recipe_replay": {
            "runner": "examples/uv_jump_benchmark.py", "targets": ["paper-v11", "strict"],
            "requires": "Explicit checksum-bound raw-profile/metadata/recipe manifest; separate references per target",
            "scope": "OpenPhenom CPU normalization and production PA/PC/NAP; not full published benchmark certification",
        }, "next_action": "Bind each target's reference/data contract; do not substitute bounded replay for full benchmark reproduction."}
    dump(output / "full_readiness.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["small", "full"], default="small")
    parser.add_argument("--output", type=Path, default=Path("jump-lite-real-output"))
    parser.add_argument("--cache", type=Path, default=Path("jump-lite-download-cache"))
    parser.add_argument("--image-lock", type=Path, help="Require exact TIFF hashes from an earlier image_lock.json")
    parser.add_argument("--cp-measure", type=Path, help="Receipt-bound existing cell-table manifest; no new extraction")
    parser.add_argument("--prepare-only", action="store_true", help="Fetch public metadata and freeze cohort; no model/image downloads")
    args = parser.parse_args(argv)
    try:
        require(not args.output.exists(), "Output already exists; choose a new directory")
        no_symlinks(args.output)
        no_symlinks(args.cache)
        require(args.output.resolve() != args.cache.resolve() and args.cache.resolve() not in args.output.resolve().parents
                and args.output.resolve() not in args.cache.resolve().parents, "Output and cache must be disjoint")
        require(args.mode != "full" or not (args.prepare_only or args.cp_measure or args.image_lock),
                "Small-mode input options do not apply to full readiness")
        args.output.mkdir(parents=True, exist_ok=False)
        if args.mode == "full":
            report = full_readiness(args.output)
            print(json.dumps(report, indent=2))
            return 3
        tables, receipts = {}, {}
        for name, (size, sha) in RELEASE.items():
            path, receipts[name] = fetch(METADATA + name, args.cache, sha, size)
            tables[name] = pl.read_parquet(path)
        rows, edges = select_cohort(tables["jump_lite_site_index.parquet"],
                                  tables["jump_lite_perturbation_metadata.parquet"],
                                  tables["jump_lite_refchem_annotations.parquet"])
        config = {"scope": "real public-input diagnostic; not published benchmark parity", "metadata": receipts,
                  "rows": rows, "edges": edges, "model_revision": REVISION,
                  "selection": "lexical first eligible group_high plate; 5 controls + 1 well for each of 6 identities; first site; first full tile",
                  "annotation_rule": "active RefChem targets with support >=5; deduplicated shared-target directed edges",
                  "tile_size": 256, "normalization": "per-plate raw MAD fit only on declared negative controls",
                  "retrieval": "median treatment consensus; cosine; exclude self before cutoff; lexical ties",
                  "calibration": "same-plate control calibration, NOT held-out-plate generalization",
                  "paper_parity": False}
        dump(args.output / "configuration.json", config)
        if args.prepare_only:
            dump(args.output / "prepared.json", {"status": "prepared_only", "wells": len(rows),
                 "configuration_sha256": digest(args.output / "configuration.json"), "inference_executed": False})
            print(f"Frozen {len(rows)} wells and {len(edges)} directed relationships; no inference executed")
            return 0
        matrices = {}
        if args.cp_measure:
            matrices["cp_measure"], evidence = measurement_features(args.cp_measure, rows)
            dump(args.output / "measurement_provenance.json", evidence)
        matrices["openphenom"] = image_features(rows, args.cache, args.output, args.image_lock)
        scores = {}
        for name, x in matrices.items():
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                normalized, fits = normalize(rows, x)
                scores[name] = evaluate(rows, normalized, edges)
            write_profiles(args.output / f"{name}_profiles.csv", rows, x)
            write_profiles(args.output / f"{name}_normalized.csv", rows, normalized)
            dump(args.output / f"{name}_fit.json", fits)
        dump(args.output / "metrics.json", scores)
        # Hash all outputs, including the exact source package executed for inference.
        hashes = {str(p.relative_to(args.output)): digest(p) for p in sorted(args.output.rglob("*"))
                  if p.is_file() and "__pycache__" not in p.parts}
        dump(args.output / "provenance.json", {"status": "small_real_run_complete", "published_score_parity": False,
             "script_sha256": digest(__file__), "output_sha256": hashes,
             "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                         **{n: importlib.metadata.version(n) for n in
                            ["numpy", "polars", "torch", "torchvision", "transformers", "timm", "tifffile", "imagecodecs"]}},
             "omissions": ["full grinder TVN/whitening/feature selection", "PA/PC/NAP/null significance",
                           "MOTIVE CG/GG", "full sites/tiles/cohort", "historical model parity", "published-score comparison"],
             "image_identity": "prior locked hashes" if args.image_lock else "first-download hashes recorded; reuse image_lock.json to enforce"})
        print(json.dumps({n: s["metrics"] for n, s in scores.items()}, indent=2))
        print(f"Completed real-data diagnostic: {args.output}. NOT published benchmark reproduction.")
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        # No completion marker on failure. Partial output is retained for diagnosis.
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
