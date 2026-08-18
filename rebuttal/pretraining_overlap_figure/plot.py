#!/usr/bin/env python3
"""Render and verify the MorphEM-specific pretraining-overlap sensitivity figure."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "inputs"
OUTPUT_DIR = ROOT / "outputs"
CONTRASTS = INPUT_DIR / "morphem_exclusion_contrasts.csv"
SUBSETS = INPUT_DIR / "morphem_exclusion_subsets.csv"
EXPECTED_INPUT_HASHES = {
    "morphem_exclusion_contrasts.csv": "a05ce4c88ceabb6a54c7254a406aed225d5e7cdf0dfc861bcfb82cec73e3a2a8",
    "morphem_exclusion_subsets.csv": "5e3016d0d546724447c1c205a92df1169b6d6f3b981db6c8134e2e5c4bc8b9b2",
}
SUBSET_ORDER = (
    "morphem_same_acquisition_wells_excluded",
    "morphem_same_acquisition_plates_excluded",
)
COMPARATOR_ORDER = ("DINOv2", "SubCell", "OpenPhenom")
COLORS = {"DINOv2": "#0072B2", "SubCell": "#009E73", "OpenPhenom": "#D55E00"}
ARTIFACTS = (
    "pretraining_overlap_sensitivity.pdf",
    "pretraining_overlap_sensitivity.png",
    "provenance.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs() -> pd.DataFrame:
    for name, expected in EXPECTED_INPUT_HASHES.items():
        path = INPUT_DIR / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input drift: {path}")

    frame = pd.read_csv(CONTRASTS)
    required = {
        "subset",
        "subset_label",
        "retained_wells",
        "comparator",
        "estimate",
        "ci_low",
        "ci_high",
        "holm_p",
        "bootstrap_replicates",
    }
    if set(frame.columns) != required:
        raise RuntimeError(f"unexpected contrast columns: {list(frame.columns)}")
    if len(frame) != 6:
        raise RuntimeError(f"expected six MorphEM contrasts, found {len(frame)}")
    if set(frame["subset"]) != set(SUBSET_ORDER):
        raise RuntimeError("figure data must contain only the two MorphEM-defined exclusions")
    if set(frame["comparator"]) != set(COMPARATOR_ORDER):
        raise RuntimeError("unexpected comparator set")
    if frame.groupby("subset").size().to_dict() != {name: 3 for name in SUBSET_ORDER}:
        raise RuntimeError("each exclusion must contain three learned-model contrasts")
    if not ((frame["ci_low"] > 0) & (frame["ci_high"] > frame["ci_low"])).all():
        raise RuntimeError("all frozen intervals must support MorphEM over the comparator")
    if not (frame["holm_p"] <= 0.05).all():
        raise RuntimeError("all six displayed contrasts must remain Holm-supported")
    if set(frame["bootstrap_replicates"]) != {50_000}:
        raise RuntimeError("unexpected bootstrap replicate count")

    subsets = pd.read_csv(SUBSETS)
    if set(subsets["subset"]) != set(SUBSET_ORDER) or len(subsets) != 2:
        raise RuntimeError("subset summary must contain only two MorphEM-defined exclusions")
    expected_wells = {
        "morphem_same_acquisition_wells_excluded": 109_790,
        "morphem_same_acquisition_plates_excluded": 74_137,
    }
    observed_wells = dict(zip(subsets["subset"], subsets["wells"], strict=True))
    if observed_wells != expected_wells:
        raise RuntimeError(f"unexpected retained-well counts: {observed_wells}")
    return frame


def render(frame: pd.DataFrame, pdf_path: Path, png_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.25), sharex=True, sharey=True)
    labels = {
        "morphem_same_acquisition_wells_excluded": (
            "A  Direct matches excluded",
            "109,790 wells retained",
        ),
        "morphem_same_acquisition_plates_excluded": (
            "B  MorphEM-matched plates excluded",
            "74,137 wells retained",
        ),
    }
    y_positions = {name: 2 - idx for idx, name in enumerate(COMPARATOR_ORDER)}

    for axis, subset in zip(axes, SUBSET_ORDER, strict=True):
        sub = frame[frame["subset"].eq(subset)].set_index("comparator")
        for comparator in COMPARATOR_ORDER:
            row = sub.loc[comparator]
            estimate = float(row["estimate"])
            low = float(row["ci_low"])
            high = float(row["ci_high"])
            axis.errorbar(
                estimate,
                y_positions[comparator],
                xerr=[[estimate - low], [high - estimate]],
                fmt="o",
                color=COLORS[comparator],
                ecolor=COLORS[comparator],
                markersize=6.5,
                elinewidth=2,
                capsize=3,
                markeredgecolor="white",
                markeredgewidth=0.7,
                zorder=3,
            )
        title, subtitle = labels[subset]
        axis.set_title(f"{title}\n{subtitle}", loc="left", pad=8)
        axis.axvline(0, color="#555555", linewidth=1, linestyle="--", zorder=1)
        axis.grid(axis="x", color="#dddddd", linewidth=0.7, zorder=0)
        axis.set_axisbelow(True)
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.set_xlim(-0.001, 0.021)
        axis.set_xticks([0.0, 0.005, 0.010, 0.015, 0.020])

    axes[0].set_yticks(
        [y_positions[name] for name in COMPARATOR_ORDER],
        [f"vs {name}" for name in COMPARATOR_ORDER],
    )
    fig.supxlabel("MorphEM − comparator in Raw PA × PC score", y=0.02)
    fig.suptitle(
        "MorphEM remains higher after two provenance-supported exclusions",
        x=0.10,
        ha="left",
        fontsize=12,
        fontweight="bold",
    )
    fig.subplots_adjust(left=0.18, right=0.985, top=0.72, bottom=0.24, wspace=0.16)
    fig.savefig(
        pdf_path,
        metadata={"Creator": "JUMP-lite", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        png_path,
        dpi=300,
        metadata={"Software": "JUMP-lite"},
    )
    plt.close(fig)


def generate() -> None:
    frame = validate_inputs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pretraining-overlap-figure-") as tmp_name:
        tmp = Path(tmp_name)
        pdf = tmp / ARTIFACTS[0]
        png = tmp / ARTIFACTS[1]
        render(frame, pdf, png)
        provenance = {
            "analysis": "MorphEM-specific pretraining-overlap exclusion figure",
            "bootstrap_replicates": 50_000,
            "displayed_subsets": list(SUBSET_ORDER),
            "comparators": list(COMPARATOR_ORDER),
            "exclusion_policy": (
                "Only exclusions supported by MorphEM's acquisition-level CHAMMI-75 inventory are displayed. "
                "No OpenPhenom-specific exclusion is constructed because no image- or plate-level membership manifest is available."
            ),
            "frozen_inputs": {
                name: {"sha256": digest} for name, digest in EXPECTED_INPUT_HASHES.items()
            },
            "archived_parent_sources": {
                "sensitivity_pairwise.csv": {
                    "sha256": "0f52305a2d285e4e4843f2b3d2ee6e19ee8333cab3c7f10b41d282a2fe4fbcf2"
                },
                "sensitivity_subset_summary.csv": {
                    "sha256": "07bb24483cc54745181f1851e445bc14f6461e991541c66fa502e47077f16ebe"
                },
            },
            "plot_source_sha256": sha256(Path(__file__)),
        }
        (tmp / ARTIFACTS[2]).write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for name in ARTIFACTS:
            shutil.copy2(tmp / name, OUTPUT_DIR / name)

    checksums = {name: sha256(OUTPUT_DIR / name) for name in ARTIFACTS}
    (OUTPUT_DIR / "artifact_checksums.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def verify() -> None:
    validate_inputs()
    checksum_path = OUTPUT_DIR / "artifact_checksums.json"
    if not checksum_path.is_file():
        raise RuntimeError("missing artifact_checksums.json")
    expected = json.loads(checksum_path.read_text(encoding="utf-8"))
    if set(expected) != set(ARTIFACTS):
        raise RuntimeError("unexpected output checksum inventory")
    for name, digest in expected.items():
        path = OUTPUT_DIR / name
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"output drift: {path}")
    provenance = json.loads((OUTPUT_DIR / "provenance.json").read_text(encoding="utf-8"))
    if "OpenPhenom-specific exclusion is constructed" not in provenance["exclusion_policy"]:
        raise RuntimeError("missing OpenPhenom provenance limitation")
    print("Verified two frozen inputs and three figure artifacts.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify()
    else:
        generate()
        verify()


if __name__ == "__main__":
    main()
