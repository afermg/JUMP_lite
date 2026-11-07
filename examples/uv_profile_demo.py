# /// script
# requires-python = "==3.12.12"
# dependencies = ["numpy==2.2.6"]
# ///
"""Standalone SYNTHETIC smoke demo, NOT scientific evidence. See --help.

CPU RobustMAD formula and cosine/median retrieval follow the small primitives in
JUMP_lite at 94f19cfe1865ad9a07858908dce0caa679627204, not its full pipeline.
No repository imports, private inputs, training, or image processing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys

# Set before importing NumPy; override inherited oversized thread settings.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_variable] = "1"

import numpy as np

META = ["Metadata_Sample", "Metadata_JCP2022", "Metadata_Group", "Metadata_Plate",
        "Metadata_negcon", "Metadata_Split"]
ANN = ["task", "query_group", "query_id", "reference_group", "reference_id"]
FEATURES = ["Feature_1", "Feature_2", "Feature_3", "Feature_4"]
MAX_BYTES = 8 * 1024 * 1024
MAX_ROWS = 10000
MAX_FEATURES = 256
MAX_PAIRS = 1000000
EPSILON = 1e-18
BASE = "94f19cfe1865ad9a07858908dce0caa679627204"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def csv_bytes(columns, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def read_bounded(path):
    with Path(path).open("rb") as handle:
        data = handle.read(MAX_BYTES + 1)
    require(len(data) <= MAX_BYTES, "Input exceeds 8 MiB limit")
    return data


def parse_csv(data):
    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""))
    columns = reader.fieldnames
    require(columns and len(columns) == len(set(columns)), "Missing/duplicate CSV header")
    rows = []
    for row in reader:
        require(len(rows) < MAX_ROWS, "CSV exceeds 10000 rows")
        require(None not in row and all(v is not None for v in row.values()),
                "Misaligned CSV row")
        require(all(v.strip() and v == v.strip() for v in row.values()),
                "Empty or whitespace-padded CSV value")
        rows.append(row)
    require(rows, "Empty CSV table")
    return columns, rows


def synthetic_inputs():
    """Fixed arithmetic fixture: 10 fit controls + 12 evaluation wells, two plates.

    Paired treatment identities have deliberately similar profiles. These are
    invented relationships, not real compounds, measurements, or annotations.
    """
    rows = []
    for plate_index, plate in enumerate(["synthetic-P1", "synthetic-P2"]):
        offset = np.array([10., 20., 30., 40.]) + plate_index * 3
        scale = np.array([1., 2., 3., 4.]) * (plate_index + 1)
        for i, level in enumerate([-2., -1., 0., 1., 2.]):
            row = dict(zip(META, [f"{plate}-ctrl-{i}", "synthetic-control",
                                  "synthetic-compounds", plate, "true", "train"]))
            row.update(zip(FEATURES, offset + scale * level))
            rows.append(row)
        for i in range(6):
            base = np.eye(4)[i // 2] * 4 + 0.2
            perturbation = np.array([0.03, -0.02, 0.01, 0.02]) * (i + 1 + plate_index)
            row = dict(zip(META, [f"{plate}-treatment-{i}", f"synthetic-drug-{i}",
                                  "synthetic-compounds", plate, "false", "eval"]))
            row.update(zip(FEATURES, offset + scale * (base + perturbation)))
            rows.append(row)
    annotations = [dict(zip(ANN, ["CC", "synthetic-compounds", f"synthetic-drug-{i}",
                                 "synthetic-compounds", f"synthetic-drug-{i ^ 1}"]))
                   for i in range(6)]
    return csv_bytes(META + FEATURES, rows), csv_bytes(ANN, annotations)


def validate_profiles(columns, rows, features):
    require(features and len(features) == len(set(features)), "Features must be nonempty and unique")
    require(len(features) <= MAX_FEATURES, "At most 256 selected features")
    require(all(f.startswith("Feature_") for f in features),
            "Only Feature_ columns may enter feature arrays; metadata is forbidden")
    require(set(META + features) <= set(columns), "Missing metadata/feature columns")
    require(all(c.startswith(("Metadata_", "Feature_")) for c in columns),
            "All columns must start with Metadata_ or Feature_")
    require(len({r["Metadata_Sample"] for r in rows}) == len(rows), "Duplicate sample identity")
    require(len({r["Metadata_Group"] for r in rows}) == 1, "Demo requires exactly one group")
    identities = {}
    for row in rows:
        require(row["Metadata_negcon"] in {"true", "false"}, "Control flag must be true or false")
        require(row["Metadata_Split"] in {"train", "eval"}, "Split must be train or eval")
        require(row["Metadata_Split"] != "train" or row["Metadata_negcon"] == "true",
                "Only declared negative controls may be training rows")
        identity = row["Metadata_JCP2022"]
        flag = row["Metadata_negcon"]
        require(identity not in identities or identities[identity] == flag,
                "Conflicting control/treatment identity")
        identities[identity] = flag
    try:
        x = np.array([[float(row[f]) for f in features] for row in rows], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("Features must be numeric") from exc
    require(np.isfinite(x).all(), "Nonfinite input features")
    return x


def normalize(rows, x):
    """Per-plate raw MAD, fit exclusively on train negative controls.

    Matches RobustMAD_CPU's (x - median)/(raw MAD + 1e-18), NOT GPU MAD
    scaling or full TVN. Reject zero MAD rather than amplify constant features.
    """
    result = np.empty_like(x)
    fitted = []
    for plate in sorted({r["Metadata_Plate"] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r["Metadata_Plate"] == plate]
        fit_indices = [i for i in indices if rows[i]["Metadata_Split"] == "train"
                       and rows[i]["Metadata_negcon"] == "true"]
        require(len(fit_indices) >= 3, f"Plate {plate} needs at least 3 training controls")
        median = np.median(x[fit_indices], axis=0)
        mad = np.median(np.abs(x[fit_indices] - median), axis=0)
        require(np.isfinite(mad).all() and (mad > EPSILON).all(),
                f"Zero/tiny/nonfinite training-control MAD on plate {plate}")
        result[indices] = (x[indices] - median) / (mad + EPSILON)
        fitted.append({"plate": plate, "fit_samples": [rows[i]["Metadata_Sample"] for i in fit_indices],
                       "median": median.tolist(), "raw_mad": mad.tolist()})
    require(np.isfinite(result).all(), "Nonfinite normalized features")
    return result, fitted


def evaluate(rows, normalized, ann_columns, annotations):
    require(ann_columns == ANN, f"Annotation header must be {ANN}")
    group = rows[0]["Metadata_Group"]
    by_identity = {}
    for i, row in enumerate(rows):
        if row["Metadata_Split"] == "eval" and row["Metadata_negcon"] == "false":
            by_identity.setdefault(row["Metadata_JCP2022"], []).append(i)
    ids = sorted(by_identity)
    require(len(ids) >= 2, "Need at least two evaluation treatment identities")
    require(len(ids) ** 2 <= MAX_PAIRS, "Retrieval exceeds 1000000 pair limit")
    consensus = np.array([np.median(normalized[by_identity[key]], axis=0) for key in ids])
    norms = np.linalg.norm(consensus, axis=1)
    require(np.isfinite(norms).all() and (norms > 0).all(), "Zero/nonfinite cosine profile norm")
    unit = consensus / norms[:, None]
    similarities = unit @ unit.T
    index = {key: i for i, key in enumerate(ids)}
    positives = {}
    edges = set()
    for edge in annotations:
        require(edge["task"] == "CC", "This small demo supports CC only")
        require(edge["query_group"] == group == edge["reference_group"],
                "Annotation/profile group mismatch")
        query, reference = edge["query_id"], edge["reference_id"]
        require(query in index and reference in index,
                "Unmatched annotation identity (must be an evaluation treatment)")
        require(query != reference, "Self-edges are not valid positives")
        require((query, reference) not in edges, "Duplicate annotation edge")
        edges.add((query, reference))
        positives.setdefault(query, set()).add(reference)
    require(positives, "No positive annotations")
    per_query = []
    for query in sorted(positives):
        # Remove self entirely; deterministic lexical identity tie break.
        candidates = [key for key in ids if key != query]
        ranked = sorted(candidates, key=lambda key: (-similarities[index[query], index[key]], key))
        positive = positives[query]
        result = {"query_id": query, "n_positives": len(positive), "n_candidates": len(candidates),
                  "top_reference": ranked[0]}
        for percent in (1, 5, 10):
            k = max(1, len(candidates) * percent // 100)
            result[f"k_{percent}pct"] = k
            result[f"recall_{percent}pct"] = len(positive.intersection(ranked[:k])) / len(positive)
        per_query.append(result)
    metrics = {f"recall@{p}%": float(np.mean([r[f"recall_{p}pct"] for r in per_query]))
               for p in (1, 5, 10)}
    coverage = {"treatment_identities": len(ids), "queries_with_positives": len(positives),
                "queries_without_positives": sorted(set(ids) - set(positives)),
                "matched_directed_edges": len(edges), "unmatched_edges": 0,
                "evaluation_wells": sum(map(len, by_identity.values()))}
    return metrics, per_query, coverage


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def run(profiles_data, annotations_data, features, output, source_label):
    require(not output.exists(), f"Refusing existing output: {output}")
    columns, rows = parse_csv(profiles_data)
    ann_columns, annotations = parse_csv(annotations_data)
    rows.sort(key=lambda r: r["Metadata_Sample"])
    x = validate_profiles(columns, rows, features)
    # Floating-point overflow is a failure, never a valid score or silently dropped row.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        normalized, fitted = normalize(rows, x)
        metrics, per_query, coverage = evaluate(rows, normalized, ann_columns, annotations)
    normalized_rows = []
    for row, values in zip(rows, normalized):
        normalized_rows.append({**{c: row[c] for c in columns if c.startswith("Metadata_")},
                                **dict(zip(features, values))})
    out_columns = [c for c in columns if c.startswith("Metadata_")] + features
    config = {"features": features, "normalization": "per_plate_train_negative_controls_raw_MAD",
              "epsilon": EPSILON, "minimum_training_controls_per_plate": 3,
              "consensus": "median_eval_noncontrol_wells_per_identity", "distance": "cosine",
              "self_matches": "excluded_before_ranking_and_cutoff",
              "ties": "lexical_ascending_reference_identity", "cutoff": "max(1, floor(eligible_references * percent / 100))",
              "percentages": [1, 5, 10], "fit": fitted,
              "limits": {"bytes_per_input": MAX_BYTES, "rows_per_input": MAX_ROWS,
                         "features": MAX_FEATURES, "pairs": MAX_PAIRS}, "numeric_threads": 1}
    artifacts = {
        "input_profiles.csv": profiles_data, "input_annotations.csv": annotations_data,
        "normalized_profiles.csv": csv_bytes(out_columns, normalized_rows),
        "metrics.json": json_bytes({"task": "CC", "metrics": metrics, "coverage": coverage}),
        "per_query.csv": csv_bytes(list(per_query[0]), per_query),
        "configuration.json": json_bytes(config),
    }
    report = {"status": "complete", "source": source_label,
              "scientific_evidence": False if source_label == "SYNTHETIC_NOT_SCIENTIFIC_EVIDENCE" else "not_assessed",
              "input_sha256": {"profiles": sha256(profiles_data), "annotations": sha256(annotations_data)},
              "script_sha256": sha256(Path(__file__).read_bytes()),
              "reference_repository": "https://github.com/afermg/JUMP_lite", "reference_commit": BASE,
              "reference_primitives": ["src/norm_3/core.py:RobustMAD_CPU",
                                       "src/motive/evaluate_cross_modality_retrieval.py:cosine, median, recall"],
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "system": platform.system(), "machine": platform.machine()},
              "omissions": ["TVN/whitening/batch correction", "feature selection", "PA/PC/NAP/null inference",
                            "CG/GG retrieval", "activity/tier filtering", "global training/evaluation recipe",
                            "image processing/model inference", "biological validation"],
              "output_sha256": {name: sha256(data) for name, data in artifacts.items()}}
    output.mkdir(parents=True, exist_ok=False)
    for name, data in artifacts.items():
        (output / name).write_bytes(data)
    # Completion marker is last: I/O failures may leave partial files but never valid metrics disguised as zero.
    (output / "provenance.json").write_bytes(json_bytes(report))
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, help="Precomputed well-level CSV; requires --annotations and --features")
    parser.add_argument("--annotations", type=Path, help="Directed CC relationship CSV")
    parser.add_argument("--features", help="Comma-separated explicit Feature_ columns (required for custom inputs)")
    parser.add_argument("--output", type=Path, default=Path("uv-profile-demo-output"), help="New output directory; never overwritten")
    args = parser.parse_args(argv)
    try:
        custom = [args.profiles is not None, args.annotations is not None, args.features is not None]
        require(all(custom) or not any(custom), "Supply --profiles, --annotations and --features together")
        if all(custom):
            inputs = read_bounded(args.profiles), read_bounded(args.annotations)
            features = args.features.split(",")
            label = "USER_SUPPLIED_PRECOMPUTED_FEATURES_NOT_VALIDATED_BIOLOGICALLY"
        else:
            inputs = synthetic_inputs()
            features = FEATURES
            label = "SYNTHETIC_NOT_SCIENTIFIC_EVIDENCE"
        metrics = run(*inputs, features, args.output, label)
    except (ValueError, OSError, UnicodeError, csv.Error, FloatingPointError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"{label}: {json.dumps(metrics, sort_keys=True)}")
    print(f"Outputs: {args.output}; completion marker: provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
