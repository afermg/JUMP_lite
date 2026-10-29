# /// script
# requires-python = "==3.12.12"
# dependencies = ["numpy==1.26.4", "polars==1.35.2", "pandas==2.2.1",
#   "pyarrow==22.0.0", "scipy==1.16.3", "scikit-learn==1.5.2",
#   "copairs==0.5.4", "numba==0.63.1", "pyyaml==6.0.3", "certifi==2025.10.5"]
# ///
"""CPU OpenPhenom benchmark replay: separate paper-v11 and strict protocols.

Requires a checksum-bound manifest of raw well profiles, authoritative metadata,
and resolved norm_3 recipes. No implicit cohort intersection, recipe truncation,
GPU allocation or model inference. No reference => no published-parity claim.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import ssl
import sys
from urllib.request import urlopen
from unittest.mock import patch

for _name in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "POLARS_MAX_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_name] = "1"

import certifi  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
import yaml  # noqa: E402

CODE_COMMIT = "bc4f74838cad1441bd3723b37e4ad78c5cceaa9b"
METRIC_FILES = {
    "io.py": "8596149862af855a880eb8bb70d736135d126e9a230e50c1cd4df0426c1507e1",
    "metrics.py": "af91b2cf87eed387d2b23b9613d2e196ccd16ec50bd3684151017e3d21c5d1ea",
}
ID, PERT, GROUP = "Metadata_id", "Metadata_JCP2022", "Metadata_Group"
PLATE, BATCH, CONTROL, TARGET = (
    "Metadata_Plate",
    "Metadata_Batch",
    "Metadata_negcon",
    "Metadata_RefChemDB_target",
)
META = [ID, PERT, GROUP, PLATE, BATCH, CONTROL, TARGET]
SUPPORTED = {
    "clean_nans",
    "filter_features",
    "merge_metadata",
    "normalize_standardize",
    "normalize_robustmad",
    "normalize_tvn_efaar",
    "inverse_normal_transform",
    "prune_correlated",
    "evaluate_metrics",
}
TARGETS = ("paper-v11", "strict")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def dump(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def safe_output(path):
    path = Path(os.path.abspath(path))
    require(
        not path.exists() and not any(p.is_symlink() for p in (path, *path.parents)),
        "Output must be new, without symlinked parents",
    )
    return path


def artifact(record, base, limit):
    require(set(record) >= {"path", "sha256"}, "Artifact needs path and SHA256")
    require(re.fullmatch(r"[0-9a-f]{64}", record["sha256"]), "Invalid artifact digest")
    path = (base / record["path"]).resolve()
    require(
        path.is_file() and 0 < path.stat().st_size <= limit,
        f"Missing/oversized input: {path}",
    )
    require(sha(path) == record["sha256"], f"Input checksum mismatch: {path}")
    return path


def split_treatments(meta, seed=20260811):
    """Exact composite-group SHA256 split convention from run_analysis.py."""
    memberships = {}
    for identity, group in (
        meta.filter(~pl.col(CONTROL)).select(PERT, GROUP).unique().iter_rows()
    ):
        memberships.setdefault(identity, set()).add(group)
    strata = {}
    for identity, groups in memberships.items():
        strata.setdefault("|".join(sorted(groups)), []).append(identity)
    result = {}
    for stratum, ids in sorted(strata.items()):

        def key(identity):
            return hashlib.sha256(f"{seed}\0{stratum}\0{identity}".encode()).hexdigest()

        ordered = sorted(ids, key=lambda identity: (key(identity), identity))
        selection = set(ordered[: int(math.floor(len(ordered) * 0.2 + 0.5))])
        for identity in sorted(ids):
            result[identity] = {
                "stratum": stratum,
                "role": "selection" if identity in selection else "test",
                "assignment_hash": key(identity),
            }
    require(
        {r["role"] for r in result.values()} == {"selection", "test"},
        "Strict split has an empty role",
    )
    return result


def validate_metadata(meta):
    require(set(META) <= set(meta.columns), "Missing authoritative metadata columns")
    require(meta.schema[CONTROL] == pl.Boolean, "Control flags must be typed booleans")
    require(
        all(meta[c].null_count() == 0 for c in META),
        "Missing metadata; explicitly use unknown for absent targets",
    )
    for c in [ID, PERT, GROUP, PLATE, BATCH, TARGET]:
        require(
            meta.schema[c] == pl.String and all(v and v.strip() == v for v in meta[c]),
            "Invalid metadata strings",
        )
    require(meta[ID].n_unique() == len(meta), "Duplicate physical sample identity")
    require(
        meta[CONTROL].any() and (~meta[CONTROL]).any(),
        "Need treatments and calibration controls",
    )
    identities = meta.group_by(PERT).agg(pl.col(CONTROL).n_unique().alias("n"))
    require(
        identities["n"].max() == 1,
        "A perturbation cannot be both control and treatment",
    )
    plate_batches = meta.group_by(PLATE).agg(pl.col(BATCH).n_unique().alias("n"))
    require(
        plate_batches["n"].max() == 1,
        "Plate identifiers span batches; use physical plate keys",
    )
    labels = meta.group_by(PERT, GROUP).agg(pl.col(TARGET).n_unique().alias("n"))
    require(labels["n"].max() == 1, "Conflicting annotations for an identity/group")
    return meta.select(META).sort(ID)


def enabled_steps(config):
    require(isinstance(config.get("steps"), list), "Need resolved norm_3 steps")
    steps = [s for s in config["steps"] if s.get("enabled", True)]
    require(
        steps and steps[0]["name"] == "clean_nans",
        "First enabled step must be clean_nans",
    )
    require(
        sum(s["name"] == "clean_nans" for s in steps) == 1,
        "Exactly one missingness step required",
    )
    require(
        sum(s["name"] == "normalize_tvn_efaar" for s in steps) <= 1,
        "Repeated TVN is unsupported",
    )
    for step in steps:
        require(
            step["name"] in SUPPORTED, f"Unsupported enabled operation: {step['name']}"
        )
        require(
            isinstance(step.get("params", {}), dict), "Invalid operation parameters"
        )
        p = step.get("params", {})
        allowed = {
            "clean_nans": {"na_cutoff"},
            "filter_features": {"always_enabled", "filters"},
            "merge_metadata": {"metadata_path", "skip_merge", "merge_how"},
            "normalize_standardize": {
                "batch_col",
                "fit_on_controls",
                "control_col",
                "control_key",
                "use_gpu",
            },
            "normalize_robustmad": {
                "batch_col",
                "fit_on_controls",
                "control_col",
                "control_key",
                "use_gpu",
                "epsilon",
            },
            "normalize_tvn_efaar": {
                "batch_col",
                "epsilon",
                "n_components",
                "dim_ratio_threshold",
                "dim_ratio_action",
                "control_col",
                "control_key",
            },
            "inverse_normal_transform": set(),
            "prune_correlated": {"threshold"},
            "evaluate_metrics": {
                "batch_col",
                "compound_col",
                "target_col",
                "negcon_col",
                "min_compounds_per_target",
                "pc_groups",
                "skip_batch_effects",
                "skip_visualization",
                "skip_umap",
                "n_top_compounds",
            },
        }
        require(
            set(p) <= allowed[step["name"]],
            f"Unsupported parameters for {step['name']}",
        )
        if "fit_on_controls" in p:
            require(
                isinstance(p["fit_on_controls"], bool),
                "fit_on_controls must be boolean",
            )
        if "control_col" in p or "control_key" in p:
            require(
                p.get("control_col", "Metadata_control_type") == "Metadata_control_type"
                and p.get("control_key", "negcon") == "negcon",
                "Unsupported control encoding",
            )
        require("${" not in json.dumps(p), "Config must be fully resolved")
        if step["name"] in {"normalize_standardize", "normalize_robustmad"}:
            require(
                p.get("batch_col", PLATE) == PLATE and not p.get("use_gpu", False),
                "Unsupported scaler scope/backend",
            )
        if step["name"] == "normalize_tvn_efaar":
            require(
                p.get("batch_col", BATCH) == BATCH
                and p.get("dim_ratio_action", "clamp") == "clamp",
                "Only batch-level TVN with clamp is supported",
            )
        if step["name"] == "filter_features":
            require(
                p.get("filters")
                and all(
                    f["name"] in {"variance_threshold", "drop_outliers"}
                    for f in p["filters"]
                ),
                "Unsupported feature filter",
            )
            for op in p["filters"]:
                keys = (
                    {"name", "freq_cut", "unique_cut"}
                    if op["name"] == "variance_threshold"
                    else {"name", "outlier_cutoff"}
                )
                require(set(op) <= keys, "Unsupported feature filter parameters")
        if step["name"] == "evaluate_metrics":
            require(
                p.get("compound_col", PERT) == PERT
                and p.get("target_col", TARGET) == TARGET
                and p.get("negcon_col", CONTROL) == CONTROL
                and p.get("batch_col", PLATE) == PLATE
                and p.get("min_compounds_per_target", 3) == 3
                and p.get("pc_groups", ["group_high", "group_low"])
                == ["group_high", "group_low"],
                "Evaluation parameters differ from the pinned PA/PC contract",
            )
    return [{"name": s["name"], "params": s.get("params", {})} for s in steps]


def power_spd(matrix, exponent):
    values, vectors = np.linalg.eigh(np.atleast_2d(matrix))
    require(
        np.isfinite(values).all()
        and values.min() > 0
        and values.max() / values.min() <= 1e10,
        "Ill-conditioned/nonpositive CORAL covariance",
    )
    return (vectors * values**exponent) @ vectors.T


def fit_tvn(x, controls, batches, params):
    require(controls.sum() >= 2, "TVN needs controls")
    groups = sorted(set(batches))
    counts = {b: int((controls & (batches == b)).sum()) for b in groups}
    require(
        min(counts.values()) >= 2, "Every batch needs two TVN controls; no silent skip"
    )
    requested = params.get("n_components", 128)
    require(isinstance(requested, int) and requested > 0, "Invalid PCA dimension")
    components = min(requested, x.shape[1], int(controls.sum()) - 1)
    threshold = float(params.get("dim_ratio_threshold", 2.5))
    require(
        math.isfinite(threshold) and threshold >= 1,
        "Invalid dimension/control threshold",
    )
    if components / min(counts.values()) > threshold:
        components = min(counts.values()) - 1
    global_scaler = StandardScaler().fit(x[controls])
    scaled = global_scaler.transform(x)
    pca = PCA(n_components=components, svd_solver="full", whiten=False).fit(
        scaled[controls]
    )
    projected = pca.transform(scaled)
    standardized = np.empty_like(projected)
    batch_scalers = {}
    for b in groups:
        mask = batches == b
        scaler = StandardScaler().fit(projected[controls & mask])
        standardized[mask] = scaler.transform(projected[mask])
        batch_scalers[b] = [scaler.mean_, scaler.scale_]
    epsilon = float(params.get("epsilon", 0.5))
    require(math.isfinite(epsilon) and epsilon > 0, "Invalid CORAL regularization")
    target = power_spd(
        np.atleast_2d(np.cov(standardized[controls].T)) + epsilon * np.eye(components),
        0.5,
    )
    source = {}
    output = np.empty_like(standardized)
    for b in groups:
        mask = batches == b
        inv = power_spd(
            np.atleast_2d(np.cov(standardized[controls & mask].T))
            + epsilon * np.eye(components),
            -0.5,
        )
        output[mask] = standardized[mask] @ inv @ target
        source[b] = inv
    state = {
        "global_mean": global_scaler.mean_,
        "global_scale": global_scaler.scale_,
        "pca_mean": pca.mean_,
        "components": pca.components_,
        "batch_scalers": batch_scalers,
        "target": target,
        "source": source,
        "control_counts": counts,
    }
    return output, state


def transform_tvn(x, batches, state):
    projected = (
        (x - state["global_mean"]) / state["global_scale"] - state["pca_mean"]
    ) @ state["components"].T
    require(set(batches) <= set(state["batch_scalers"]), "Unseen TVN batch")
    result = np.empty_like(projected)
    for b, (center, scale) in state["batch_scalers"].items():
        mask = batches == b
        result[mask] = (
            ((projected[mask] - center) / scale) @ state["source"][b] @ state["target"]
        )
    return result


def correlation_keep(x, threshold):
    require(0 < threshold <= 1, "Invalid correlation threshold")
    adjacency = (
        np.abs(np.nan_to_num(np.atleast_2d(np.corrcoef(x, rowvar=False)))) > threshold
    )
    np.fill_diagonal(adjacency, False)
    remaining, chosen = set(range(x.shape[1])), []
    while remaining:
        degrees = adjacency.sum(axis=1)
        node = min(remaining, key=lambda i: (degrees[i], i))
        chosen.append(node)
        neighbors = set(np.flatnonzero(adjacency[node]))
        remaining -= neighbors | {node}
        for i in neighbors | {node}:
            adjacency[i, :] = False
            adjacency[:, i] = False
    return np.isin(np.arange(x.shape[1]), chosen)


def int_transform(x, references):
    result = np.empty_like(x)
    for i, reference in enumerate(references):
        left = np.searchsorted(reference, x[:, i], side="left")
        right = np.searchsorted(reference, x[:, i], side="right")
        rank = np.clip((left + right + 1.0) / 2.0, 1.0, len(reference))
        result[:, i] = norm.ppf((rank - 0.375) / (len(reference) + 0.25))
    return result


def fit_recipe(x, meta, config, target, split):
    require(target in TARGETS, "Unknown protocol")
    require(
        x.ndim == 2 and x.shape[0] == len(meta) and 1 <= x.shape[1] <= 4096,
        "Bad feature matrix",
    )
    x = np.asarray(x, dtype=np.float64).copy()
    meta = meta.clone()
    controls = meta[CONTROL].to_numpy()
    if target == "paper-v11":
        permitted = np.ones(len(meta), dtype=bool)
    else:
        require(
            set(meta.filter(~pl.col(CONTROL))[PERT]) <= set(split),
            "Incomplete treatment split",
        )
        permitted = controls | np.array(
            [split.get(i, {}).get("role") == "selection" for i in meta[PERT]]
        )
        require(
            (permitted & ~controls).any() and (~permitted).any(),
            "Empty strict fit/test role",
        )
    states, excluded = [], []
    for step in enabled_steps(config):
        name, p = step["name"], step["params"]
        if name in {"merge_metadata", "evaluate_metrics"}:
            continue  # Metadata is explicitly prejoined and hash-bound before feature loading.
        state = {"name": name}
        if name == "clean_nans":
            cutoff = float(p.get("na_cutoff", 0.3))
            require(0 <= cutoff <= 1, "Invalid missingness cutoff")
            keep = (~np.isfinite(x[permitted])).mean(axis=0) <= cutoff
            require(keep.any(), "Missingness removed every feature")
            x = x[:, keep]
            finite = np.isfinite(x).all(axis=1)
            excluded = meta.filter(pl.Series(~finite))[ID].to_list()
            x, meta = x[finite], meta.filter(pl.Series(finite))
            controls, permitted = controls[finite], permitted[finite]
            require(
                controls.any() and (permitted & ~controls).any(),
                "Cleaning emptied fit role",
            )
            if target == "strict":
                require((~permitted).any(), "Cleaning emptied test role")
            state["keep"] = keep
        elif name == "filter_features":
            masks = []
            for op in p["filters"]:
                values = x[permitted]
                if op["name"] == "variance_threshold":
                    keep = []
                    for column in values.T:
                        _, counts = np.unique(column, return_counts=True)
                        counts.sort()
                        keep.append(
                            len(counts) / len(values) >= op.get("unique_cut", 0.01)
                            and (
                                len(counts) < 2
                                or counts[-2] / counts[-1] >= op.get("freq_cut", 0.05)
                            )
                        )
                    keep = np.array(keep)
                else:
                    z = (values - values.mean(axis=0)) / (values.std(axis=0) + 1e-8)
                    keep = np.max(np.abs(z), axis=0) <= op.get("outlier_cutoff", 500)
                x = x[:, keep]
                masks.append(keep)
            state["masks"] = masks
        elif name in {"normalize_robustmad", "normalize_standardize"}:
            fitting = controls if p.get("fit_on_controls", False) else permitted
            plates = meta[PLATE].to_numpy()
            scalers = {}
            for plate in sorted(set(plates)):
                mask = plates == plate
                values = x[fitting & mask]
                require(len(values), "Plate has no permitted calibration rows")
                if name == "normalize_robustmad":
                    center = np.median(values, axis=0)
                    scale = np.median(np.abs(values - center), axis=0) + float(
                        p.get("epsilon", 1e-18)
                    )
                else:
                    center, scale = values.mean(axis=0), values.std(axis=0) + 1e-18
                x[mask] = (x[mask] - center) / scale
                scalers[plate] = [center, scale]
            state["scalers"] = scalers
        elif name == "inverse_normal_transform":
            state["references"] = [np.sort(c) for c in x[permitted].T]
            x = int_transform(x, state["references"])
        elif name == "prune_correlated":
            state["keep"] = correlation_keep(
                x[permitted], float(p.get("threshold", 0.9))
            )
            x = x[:, state["keep"]]
        elif name == "normalize_tvn_efaar":
            x, fitted = fit_tvn(x, controls, meta[BATCH].to_numpy(), p)
            state.update(fitted)
        require(x.shape[1] > 0 and np.isfinite(x).all(), f"Invalid output after {name}")
        state["fit_ids_sha256"] = digest(
            sorted(
                meta.filter(
                    pl.Series(
                        controls
                        if name == "normalize_tvn_efaar"
                        or p.get("fit_on_controls", False)
                        else permitted
                    )
                )[ID]
            )
        )
        states.append(state)
    audit = {
        "target": target,
        "fit_ids": meta.filter(pl.Series(permitted))[ID].to_list(),
        "excluded_nonfinite_ids": excluded,
        "fit_rows": int(permitted.sum()),
        "test_rows_excluded_from_fit": int((~permitted).sum()),
        "output_features": x.shape[1],
    }
    return x, meta, states, audit


def apply_recipe(x, meta, states):
    x, meta = np.asarray(x, dtype=np.float64).copy(), meta.clone()
    for state in states:
        name = state["name"]
        if name == "clean_nans":
            x = x[:, state["keep"]]
            finite = np.isfinite(x).all(axis=1)
            x, meta = x[finite], meta.filter(pl.Series(finite))
        elif name == "filter_features":
            for keep in state["masks"]:
                x = x[:, keep]
        elif name.startswith("normalize_") and "scalers" in state:
            plates = meta[PLATE].to_numpy()
            require(set(plates) <= set(state["scalers"]), "Unseen transform plate")
            for plate, (center, scale) in state["scalers"].items():
                mask = plates == plate
                x[mask] = (x[mask] - center) / scale
        elif name == "inverse_normal_transform":
            x = int_transform(x, state["references"])
        elif name == "prune_correlated":
            x = x[:, state["keep"]]
        elif name == "normalize_tvn_efaar":
            x = transform_tvn(x, meta[BATCH].to_numpy(), state)
        require(np.isfinite(x).all(), "Nonfinite transform")
    return x, meta


def encode_state(value, arrays):
    if isinstance(value, np.ndarray):
        key = f"array_{len(arrays):05d}"
        arrays[key] = value
        return {
            "array": key,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
    if isinstance(value, dict):
        return {k: encode_state(v, arrays) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [encode_state(v, arrays) for v in value]
    return value


def state_digest(states):
    return digest(encode_state(states, {}))


def save_fit(folder, x, meta, states, audit):
    folder.mkdir()
    arrays = {}
    encoded = encode_state(states, arrays)
    np.savez_compressed(folder / "state.npz", **arrays)
    dump(folder / "state.json", encoded)
    dump(folder / "fit_audit.json", audit | {"state_sha256": digest(encoded)})
    meta.with_columns(
        [pl.Series(f"Feature_{i}", x[:, i]) for i in range(x.shape[1])]
    ).write_parquet(folder / "normalized.parquet")


def load_metrics(output, source=None):
    package = output / "runtime" / "norm_3"
    package.mkdir(parents=True)
    for filename, expected in METRIC_FILES.items():
        if source is not None:
            data = (source / filename).read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/afermg/JUMP_lite/{CODE_COMMIT}/src/norm_3/{filename}"
            with urlopen(
                url,
                timeout=60,
                context=ssl.create_default_context(cafile=certifi.where()),
            ) as response:
                data = response.read(2 * 1024**2 + 1)
        require(
            len(data) <= 2 * 1024**2 and hashlib.sha256(data).hexdigest() == expected,
            "Pinned metrics source mismatch",
        )
        (package / filename).write_bytes(data)
    (package / "__init__.py").touch()
    require(
        "norm_3.metrics" not in sys.modules,
        "Refusing a previously imported metrics module",
    )
    sys.path.insert(0, str(package.parent.resolve()))
    return importlib.import_module("norm_3.metrics")


@contextmanager
def owned_metric_runtime(output):
    home = output / "runtime" / "home"
    home.mkdir(parents=True, exist_ok=True)
    # copairs chooses ThreadPool sizes using os.cpu_count and writes ~/.copairs.
    # Bound scheduling only; do not modify pairs, scores, random seeds or null sizes.
    with (
        patch.dict(
            os.environ,
            {
                "HOME": str(home.resolve()),
                "USERPROFILE": str(home.resolve()),
                "NUMBA_CACHE_DIR": str(home / "numba"),
            },
        ),
        patch("os.cpu_count", return_value=1),
    ):
        yield


def score(metrics, x, meta, split, role, output):
    if role != "all":
        keep = meta[CONTROL].to_numpy() | np.array(
            [split.get(i, {}).get("role") == role for i in meta[PERT]]
        )
        x, meta = x[keep], meta.filter(pl.Series(keep))
    require(meta.filter(~pl.col(CONTROL)).height > 0, "Empty scoring role")
    controls = meta.filter(pl.col(CONTROL)).select(PLATE, GROUP).unique()
    missing = (
        meta.filter(~pl.col(CONTROL))
        .select(PLATE, GROUP)
        .unique()
        .join(controls, on=[PLATE, GROUP], how="anti")
    )
    require(not len(missing), "Scoring partition lacks plate/group controls")
    features = [f"Feature_{i}" for i in range(x.shape[1])]
    frame = meta.with_columns([pl.Series(f, x[:, i]) for i, f in enumerate(features)])
    distance = "euclidean" if x.shape[1] <= 2 else "cosine"
    with owned_metric_runtime(output):
        from copairs import map as copairs_map
        from functools import partial

        with patch.object(
            copairs_map,
            "mean_average_precision",
            partial(copairs_map.mean_average_precision, max_workers=1),
        ):
            pa = metrics.calculate_phenotypic_activity(
                frame,
                features,
                null_size=10000,
                seed=0,
                compound_col=PERT,
                negcon_col=CONTROL,
                batch_col=PLATE,
                group_col=GROUP,
                distance=distance,
            )
            pc = metrics.calculate_phenotypic_consistency(
                frame,
                features,
                null_size=10000,
                seed=0,
                compound_col=PERT,
                target_col=TARGET,
                negcon_col=CONTROL,
                group_col=GROUP,
                pc_groups=["group_high", "group_low"],
                distance=distance,
            )
    require(
        pa.get("activity_map") is not None
        and pc.get("target_consistency") is not None
        and pa.get("n_compounds", 0) > 0
        and pc.get("n_targets_total", 0) > 0,
        "Undefined PA/PC; never substitute zero",
    )
    a, c = (
        float(pa["mean_normalized_average_precision"]),
        float(pc["mean_normalized_average_precision"]),
    )
    require(math.isfinite(a) and math.isfinite(c), "Nonfinite PA/PC")
    return (
        {
            "pa_mean_nap": a,
            "pc_mean_nap": c,
            "pa_pc_product": a * c,
            "wells": len(meta),
            "controls": int(meta[CONTROL].sum()),
            "features": x.shape[1],
            "pa_units": int(pa["n_compounds"]),
            "pc_targets": int(pc["n_targets_total"]),
        },
        pa["activity_map"],
        pc["target_consistency"],
    )


def choose_winner(candidates):
    require(candidates, "No candidates")
    pa = [c["pa_mean_nap"] for c in candidates]
    pc = [c["pc_mean_nap"] for c in candidates]
    require(all(math.isfinite(v) for v in pa + pc), "Invalid candidate score")

    def scale(v, values):
        return (
            1.0
            if max(values) == min(values)
            else (v - min(values)) / (max(values) - min(values))
        )

    ranked = [
        c
        | {"selection_score": scale(c["pa_mean_nap"], pa) * scale(c["pc_mean_nap"], pc)}
        for c in candidates
    ]
    return sorted(ranked, key=lambda c: (-c["selection_score"], c["recipe"]))


def validate_reference(reference, identity, scores):
    require(
        reference.get("reference_kind")
        in {
            "published_frozen_reference",
            "local_cpu_replay_not_published",
            "independent_implementation_validation",
        },
        "Unknown reference provenance kind",
    )
    require(
        reference["protocol"] == identity,
        "Reference protocol/cohort/recipe/split identity mismatch",
    )
    expected = reference["scores"]
    require(len(expected) == len(scores), "Reference score coverage mismatch")

    def key(row):
        return row["codec"], row["recipe"]

    wanted = {key(r): r for r in expected}
    require(
        len(wanted) == len(expected) and set(wanted) == {key(r) for r in scores},
        "Reference keys mismatch",
    )
    differences = []
    for actual in scores:
        ref = wanted[key(actual)]
        for field in ("wells", "controls", "features", "pa_units", "pc_targets"):
            require(
                ref[field] == actual[field], f"Reference population mismatch: {field}"
            )
        for field in ("pa_mean_nap", "pc_mean_nap", "pa_pc_product"):
            value = float(ref[field])
            require(math.isfinite(value), "Nonfinite reference score")
            if not np.isclose(actual[field], value, rtol=1e-5, atol=1e-6):
                differences.append(
                    {
                        "codec": actual["codec"],
                        "recipe": actual["recipe"],
                        "metric": field,
                        "actual": actual[field],
                        "expected": value,
                    }
                )
    return {
        "status": "match" if not differences else "mismatch",
        "rtol": 1e-5,
        "atol": 1e-6,
        "reference_kind": reference["reference_kind"],
        "differences": differences,
    }


def run(args):
    safe_output(args.output)
    require(args.manifest.stat().st_size <= 8 * 1024**2, "Manifest too large")
    spec = json.loads(args.manifest.read_text())
    require(
        spec.get("schema_version") == 1 and spec.get("family") == "openphenom",
        "Only schema v1 / OpenPhenom is supported",
    )
    require(
        spec.get("profile_kind") == "raw_well_features",
        "Normalized profiles cannot substitute for raw inputs",
    )
    require(
        spec.get("cohort_scope") in {"bounded-validation", "full-declared-cohort"},
        "Declare cohort scope",
    )
    require(
        spec["cohort_scope"] != "full-declared-cohort" or args.allow_large,
        "Full cohort requires --allow-large",
    )
    limit = 8 * 1024**3 if args.allow_large else 512 * 1024**2
    row_limit = 200000 if args.allow_large else 20000
    recipe_limit = 175 if args.allow_large else 4
    require(
        1 <= len(spec["recipes"]) <= recipe_limit,
        "Recipe inventory exceeds declared resource limit",
    )
    require(
        1 <= len(spec["codecs"]) <= 4 and spec["codecs"][0]["name"] == "Raw",
        "Raw must be first; at most four codecs",
    )
    require(
        len({c["name"] for c in spec["codecs"]}) == len(spec["codecs"])
        and {c["name"] for c in spec["codecs"]} <= {"Raw", "HQ", "MQ", "D20"},
        "Invalid codec inventory",
    )
    base = args.manifest.resolve().parent
    metadata_path = artifact(spec["metadata"], base, limit)
    meta = validate_metadata(
        pl.scan_parquet(metadata_path).head(row_limit + 1).collect()
    )
    require(len(meta) <= row_limit, "Metadata exceeds row budget")
    targets = TARGETS if args.target == "both" else (args.target,)
    # Freeze the treatment split before reading any profile/feature matrix.
    split = (
        split_treatments(meta, spec.get("seed", 20260811))
        if "strict" in targets
        else {}
    )
    recipes, signatures, sources = {}, set(), {"metadata": spec["metadata"]["sha256"]}
    for record in spec["recipes"]:
        name = record["id"]
        require(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,150}", name)
            and name not in recipes,
            "Invalid/duplicate recipe ID",
        )
        path = artifact(record["config"], base, 1024**2)
        config = yaml.safe_load(path.read_text())
        signature = digest(enabled_steps(config))
        require(
            signature not in signatures,
            "Duplicate effective recipes: declare aliases, do not score them twice",
        )
        signatures.add(signature)
        recipes[name] = config
        sources["recipe:" + name] = record["config"]["sha256"]
    paths = {}
    width = None
    for record in spec["codecs"]:
        columns = record["features"]
        require(
            1 <= len(columns) <= 4096
            and len(columns) == len(set(columns))
            and all(not c.lower().startswith("metadata") for c in columns),
            "Invalid ordered feature axis",
        )
        require(
            width is None or width == len(columns), "Cross-codec feature widths differ"
        )
        width = len(columns)
        paths[record["name"]] = artifact(record["profiles"], base, limit)
        sources["profiles:" + record["name"]] = record["profiles"]["sha256"]
    require(
        spec["cohort_scope"] != "full-declared-cohort" or width == 384,
        "Full OpenPhenom cohort requires all 384 declared coordinates",
    )
    output = safe_output(args.output)
    output.mkdir(parents=True)
    dump(output / "input_manifest.json", spec)
    dump(output / "treatment_split.json", split)
    source_identity = {
        "family": "openphenom",
        "cohort_scope": spec["cohort_scope"],
        "sources": sources,
        "feature_axes": {c["name"]: c["features"] for c in spec["codecs"]},
        "cohort_sha256": digest(meta.to_dicts()),
        "metric_source_sha256": METRIC_FILES,
        "metrics": {"null_size": 10000, "seed": 0, "min_compounds_per_target": 3},
    }
    identities = {
        t: source_identity
        | {"target": t, "split_sha256": digest(split) if t == "strict" else None}
        for t in targets
    }
    dump(output / "protocols.json", identities)
    references = {}
    for t in targets:
        if t in spec.get("references", {}):
            ref_path = artifact(spec["references"][t], base, 8 * 1024**2)
            references[t] = json.loads(ref_path.read_text())
            require(
                references[t].get("protocol") == identities[t],
                "Reference protocol identity mismatch",
            )
    metrics = load_metrics(output, args.metrics_source)
    results = {t: [] for t in targets}
    for t in targets:
        (output / t).mkdir()
    winner = None
    strict_evaluation_ids = None
    for record in spec["codecs"]:
        codec = record["name"]
        columns = record["features"]
        # Only identity + ordered raw features are loaded; metadata/labels are separate.
        table = pq.read_table(
            paths[codec],
            columns=[ID, *columns],
            filters=[(ID, "in", meta[ID].to_list())],
            use_threads=False,
        )
        table.validate(full=True)
        data = pl.from_arrow(table)
        require(
            len(data) == len(meta) and data[ID].n_unique() == len(meta),
            "Raw input does not cover the exact metadata cohort",
        )
        data = meta.select(ID).join(data, on=ID, how="left", validate="1:1")
        require(
            all(data.schema[c].is_numeric() for c in columns), "Non-numeric input axis"
        )
        raw = data.select(columns).to_numpy().astype(np.float64)
        if "paper-v11" in targets:
            for recipe, config in recipes.items():
                print(f"paper-v11 {codec} {recipe}", flush=True)
                x, cleaned, states, audit = fit_recipe(
                    raw, meta, config, "paper-v11", {}
                )
                directory = output / "paper-v11" / f"{codec}--{recipe}"
                save_fit(directory, x, cleaned, states, audit)
                values, pa, pc = score(metrics, x, cleaned, {}, "all", output)
                pa.to_csv(directory / "pa_units.csv", index=False)
                pc.to_csv(directory / "pc_targets.csv", index=False)
                results["paper-v11"].append(values | {"codec": codec, "recipe": recipe})
        if "strict" in targets:
            if codec == "Raw":
                candidates = []
                for recipe, config in recipes.items():
                    print(f"strict Raw selection {recipe}", flush=True)
                    x, cleaned, states, audit = fit_recipe(
                        raw, meta, config, "strict", split
                    )
                    directory = output / "strict" / f"selection--{recipe}"
                    save_fit(directory, x, cleaned, states, audit)
                    values, pa, pc = score(
                        metrics, x, cleaned, split, "selection", output
                    )
                    pa.to_csv(directory / "pa_units.csv", index=False)
                    pc.to_csv(directory / "pc_targets.csv", index=False)
                    candidates.append(values | {"codec": "Raw", "recipe": recipe})
                ranking = choose_winner(candidates)
                winner = ranking[0]["recipe"]
                dump(
                    output / "strict" / "selection.json",
                    {
                        "winner": winner,
                        "ranking": ranking,
                        "rule": "within-family minmax(PA)*minmax(PC); constant ranges=1; lexical ties",
                    },
                )
            require(winner in recipes, "No frozen Raw-selected recipe")
            print(f"strict held-out {codec} {winner}", flush=True)
            x, cleaned, states, audit = fit_recipe(
                raw, meta, recipes[winner], "strict", split
            )
            eval_mask = cleaned[CONTROL].to_numpy() | np.array(
                [split.get(i, {}).get("role") == "test" for i in cleaned[PERT]]
            )
            eval_ids = sorted(cleaned.filter(pl.Series(eval_mask))[ID])
            if strict_evaluation_ids is None:
                strict_evaluation_ids = eval_ids
            require(
                eval_ids == strict_evaluation_ids,
                "Codec held-out cohorts differ; no implicit intersection",
            )
            directory = output / "strict" / f"test--{codec}"
            save_fit(directory, x, cleaned, states, audit)
            values, pa, pc = score(metrics, x, cleaned, split, "test", output)
            pa.to_csv(directory / "pa_units.csv", index=False)
            pc.to_csv(directory / "pc_targets.csv", index=False)
            results["strict"].append(values | {"codec": codec, "recipe": winner})
    comparisons = {}
    for t in targets:
        dump(
            output / t / "scores.json",
            {
                "protocol": identities[t],
                "scores": results[t],
                "reference_kind": "local_cpu_replay_not_published",
            },
        )
        ref = spec.get("references", {}).get(t)
        if ref is None:
            comparisons[t] = {"status": "not_supplied", "published_parity": False}
        else:
            reference = references[t]
            comparisons[t] = validate_reference(reference, identities[t], results[t])
            comparisons[t]["published_parity"] = False
            comparisons[t]["publication_provenance"] = (
                "Caller-declared reference kind; whole-paper coverage/provenance is not independently certified"
            )
    # Refuse input drift during a potentially long multi-recipe run.
    artifact(spec["metadata"], base, limit)
    for record in spec["recipes"]:
        artifact(record["config"], base, 1024**2)
    for record in spec["codecs"]:
        artifact(record["profiles"], base, limit)
    for ref in spec.get("references", {}).values():
        artifact(ref, base, 8 * 1024**2)
    mismatch = any(v["status"] == "mismatch" for v in comparisons.values())
    dump(output / "comparison.json", comparisons)
    hashes = {
        str(p.relative_to(output)): sha(p)
        for p in sorted(output.rglob("*"))
        if p.is_file() and "runtime" not in p.relative_to(output).parts
    }
    dump(
        output / "provenance.json",
        {
            "status": "reference_mismatch" if mismatch else "computations_complete",
            "reference_comparison": comparisons,
            "script_sha256": sha(__file__),
            "output_sha256": hashes,
            "backend": "CPU float64; sklearn full-SVD PCA and NumPy symmetric eigendecomposition; not a CuPy/cuML bitwise claim",
            "runtime": {
                n: importlib.metadata.version(n)
                for n in [
                    "numpy",
                    "polars",
                    "pandas",
                    "scipy",
                    "scikit-learn",
                    "copairs",
                    "numba",
                ]
            },
            "scope": spec["cohort_scope"],
            "targets": list(targets),
            "limitations": [
                "raw feature checkpoint/preprocessing lineage supplied externally",
                "no CG/GG/full MOTIVE execution",
                "no automatic public production-input archive",
                "CPU numerical parity requires an independent reference",
            ],
        },
    )
    print(json.dumps(comparisons, indent=2))
    return 4 if mismatch else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--target", choices=[*TARGETS, "both"], default="both")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--allow-large",
        action="store_true",
        help="Explicitly permit up to 200000 wells / 175 effective recipes / 4 codecs",
    )
    p.add_argument(
        "--metrics-source",
        type=Path,
        help="Optional local norm_3 directory; exact pinned source hashes still required",
    )
    args = p.parse_args(argv)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return run(args)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
