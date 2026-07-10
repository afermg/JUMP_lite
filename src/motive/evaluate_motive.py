#!/usr/bin/env python3
"""MOTIVE evaluation for a single profile parquet.

CLI:

    uv run python evaluation/evaluate_motive.py \
        --input  <some>/output.parquet \
        --output <some>/motive_eval/ \
        --annotations metadata/motive_annotations.parquet \
        --splits      metadata/motive_eval_compounds.parquet

Runs retrieval tasks against MOTIVE labels:

- MOTIVE-CC  : cosine recall@k over the cc compound→compound graph. (default on)
- MOTIVE-GG  : cosine recall@k over the gg gene→gene graph, per modality.
              (default on)
- MOTIVE-CG  : cosine recall@k compound→ORF/CRISPR using JCP-keyed cg edges.
              Reference pool defaults to ``full`` (every ORF/CRISPR JCP in the
              input). Pass ``--run-filtered`` to additionally evaluate against
              the ``filtered`` pool (only gene JCPs that are MOTIVE targets;
              mirrors MOTIVE's ``targets_with_links`` filter). (default on)
- MOTIVE-PC  : copairs target-level retrieval, MOTIVE cg targets (strings).
              (opt-in via ``--run-pc`` — copairs PC dominates wall time)
              Two reference-pool settings are reported per (compound_group ×
              target_modality):
                * ``filtered`` — references = gene JCPs that resolve from at
                  least one MOTIVE target (mirrors MOTIVE's ``targets_with_links``
                  filter from ``motive/jump.py``).
                * ``full`` — references = every ORF / CRISPR JCP present in
                  the input parquet.

Outputs into ``--output``:
- ``metrics.json``                 (presence triggers idempotency)
- ``motive_pc_per_target.csv``
- ``motive_cc_summary.csv``
- ``motive_cg_summary.csv``
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from evaluate_cross_modality_retrieval import (
    calculate_recall_at_k,
    compute_cosine_similarity_matrix,
    get_consensus_profiles,
)
from evaluate_phenotypic_activity import (
    calculate_phenotypic_consistency,
    get_numeric_features,
    infer_columns,
    load_profiles,
    merge_metadata,
    setup_control_columns,
)


PER_TASK_CSVS = (
    "motive_pc_per_target.csv",
    "motive_cc_summary.csv",
    "motive_cg_summary.csv",
    "motive_gg_summary.csv",
)

CG_SETTINGS_DEFAULT = ("full",)
CG_SETTINGS_WITH_FILTERED = ("full", "filtered")
COMPOUND_GROUPS = ("group_high", "group_low")
TARGET_MODALITIES = ("orf", "crispr")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _refuse_overwrite(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(
            f"refusing to overwrite existing {path}. Pass --force to opt in."
        )


def _attach_motive_target_list(
    df: pl.DataFrame, cg_ann: pl.DataFrame, compound_col: str
) -> pl.DataFrame:
    """Attach a pipe-joined ``Metadata_target_list`` column for PC.

    Aggregates MOTIVE target *symbols* per compound (one symbol can appear
    multiple times in cg_ann after JCP resolution; ``unique()`` collapses).
    """
    target_list = (
        cg_ann.group_by(compound_col)
        .agg(
            pl.col("target")
            .unique()
            .sort()
            .str.join("|")
            .alias("Metadata_target_list")
        )
    )
    return df.join(target_list, on=compound_col, how="left")


def _restrict_compounds_to_split(
    df: pl.DataFrame,
    splits: pl.DataFrame,
    split_eval: str,
    compound_col: str,
) -> pl.DataFrame:
    """Keep ORF/CRISPR rows untouched; filter compound rows to the eval split."""
    split_jcps = (
        splits.filter(pl.col("split") == split_eval)
        .select(pl.col(compound_col).cast(pl.Utf8))
        .unique()
    )
    df = df.with_columns(pl.col(compound_col).cast(pl.Utf8))
    if "Metadata_Perturbation_Type" in df.columns:
        compound_mask = pl.col("Metadata_Perturbation_Type") == "compound"
    elif "Metadata_Group" in df.columns:
        compound_mask = pl.col("Metadata_Group").is_in(list(COMPOUND_GROUPS))
    else:
        raise ValueError(
            "input is missing both Metadata_Perturbation_Type and Metadata_Group; "
            "cannot identify compound rows"
        )

    compounds = df.filter(compound_mask).join(split_jcps, on=compound_col, how="inner")
    others = df.filter(~compound_mask)
    out = pl.concat([compounds, others], how="vertical_relaxed")
    print(
        f"[split={split_eval}] kept {compounds.height:,} compound rows "
        f"({split_jcps.height:,} JCPs) + {others.height:,} ORF/CRISPR rows"
    )
    return out


# ---------------------------------------------------------------------------
# MOTIVE-CC
# ---------------------------------------------------------------------------


def run_motive_cc(
    df: pl.DataFrame,
    features: list[str],
    cc_ann: pl.DataFrame,
    output_dir: Path,
    compound_col: str,
    group_col: str,
    negcon_col: str,
    k_percentages: list[float],
) -> dict[str, Any]:
    """Cosine recall@k over the MOTIVE compound→compound graph, per group.

    Runs separately for ``group_high ↔ group_high`` and
    ``group_low ↔ group_low`` (symmetric within-group). Cross-group cc edges
    are not scored. One summary row per compound group.
    """
    print("\n=== MOTIVE-CC ===")
    df_use = df.filter(pl.col(negcon_col) == False)

    pairs = cc_ann.filter(pl.col("source") == "cc").select(
        compound_col, "partner_jcp"
    )

    summary_rows: list[dict[str, Any]] = []
    for compound_group in COMPOUND_GROUPS:
        df_group = df_use.filter(pl.col(group_col) == compound_group)
        if df_group.height == 0:
            print(f"  [{compound_group}] no rows; skipping")
            continue

        metadata_df, feat_arr = get_consensus_profiles(
            df_group, features, compound_col, group_col
        )
        jcp_list = metadata_df[compound_col].to_list()
        jcp_to_idx: dict[str, list[int]] = defaultdict(list)
        for i, jcp in enumerate(jcp_list):
            jcp_to_idx[jcp].append(i)
        n = len(jcp_list)
        if n < 2:
            print(f"  [{compound_group}] only {n} consensus compounds — skipping")
            continue

        sim = compute_cosine_similarity_matrix(feat_arr, feat_arr)
        # Defensive same-JCP mask (a JCP appearing twice in the same group is
        # not expected, but np.fill_diagonal alone wouldn't catch it).
        jcp_array = np.array(jcp_list)
        sim[jcp_array[:, None] == jcp_array[None, :]] = -np.inf

        pos = np.zeros((n, n), dtype=bool)
        for jcp_a, jcp_b in pairs.iter_rows():
            i_list = jcp_to_idx.get(jcp_a)
            j_list = jcp_to_idx.get(jcp_b)
            if not i_list or not j_list:
                continue
            for i in i_list:
                for j in j_list:
                    if i != j:
                        pos[i, j] = True

        n_pos = int(pos.sum())
        n_q_with_pos = int((pos.sum(axis=1) > 0).sum())
        n_unique = len(set(jcp_list))
        print(
            f"  [{compound_group}] {n} consensus rows ({n_unique} unique JCPs), "
            f"{n_pos} positive pairs, {n_q_with_pos} queries with ≥1 positive"
        )
        if n_pos == 0:
            continue

        recall = calculate_recall_at_k(sim, pos, k_percentages)
        row = {
            "task": "MOTIVE_CC",
            "compound_group": compound_group,
            "n_unique_compounds": n_unique,
            "n_consensus_rows": n,
            "n_positive_pairs": n_pos,
            "n_queries_with_positives": n_q_with_pos,
            **recall,
        }
        summary_rows.append(row)
        recall_str = ", ".join(f"{k}={v*100:.2f}%" for k, v in recall.items())
        print(f"  [{compound_group}] {recall_str}")

    if summary_rows:
        pl.DataFrame(summary_rows).write_csv(output_dir / "motive_cc_summary.csv")
    return {"summary": summary_rows}


# ---------------------------------------------------------------------------
# MOTIVE-GG
# ---------------------------------------------------------------------------


def run_motive_gg(
    df: pl.DataFrame,
    features: list[str],
    gg_ann: pl.DataFrame,
    output_dir: Path,
    compound_col: str,
    group_col: str,
    negcon_col: str,
    k_percentages: list[float],
    settings: tuple[str, ...] = CG_SETTINGS_DEFAULT,
) -> dict[str, Any]:
    """Cosine recall@k over the MOTIVE gene→gene graph, per modality.

    Per modality (ORF, CRISPR), two reference-pool settings:
      'filtered' — references = gene JCPs with ≥1 gg edge in this modality.
      'full'     — references = every gene JCP in this modality in the input.

    Same-JCP cross-row masking applied (defensive against a JCP appearing in
    multiple consensus rows).
    """
    print("\n=== MOTIVE-GG ===")
    df_use = df.filter(pl.col(negcon_col) == False)

    summary_rows: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}

    for modality in TARGET_MODALITIES:
        ref_group = f"group_{modality}"
        df_mod = df_use.filter(pl.col(group_col) == ref_group)
        if df_mod.height == 0:
            print(f"  no {ref_group} rows; skipping modality={modality}")
            continue

        meta_full, feat_full = get_consensus_profiles(
            df_mod, features, compound_col, group_col
        )
        jcp_full = meta_full[compound_col].to_list()
        n_full = len(jcp_full)

        edges = (
            gg_ann.filter(pl.col("source") == "gg")
            .filter(pl.col("target_modality") == modality)
            .select(compound_col, "partner_jcp")
        )
        jcps_in_edges = set(edges[compound_col].to_list()) | set(
            edges["partner_jcp"].to_list()
        )
        coverage[f"{modality}_edges_in_motive"] = edges.height
        coverage[f"{modality}_jcps_in_motive"] = len(jcps_in_edges)
        coverage[f"{modality}_jcps_in_profiles"] = n_full
        coverage[f"{modality}_jcps_in_filtered_pool"] = len(
            jcps_in_edges & set(jcp_full)
        )

        for setting in settings:
            if setting == "filtered":
                keep_idx = [i for i, j in enumerate(jcp_full) if j in jcps_in_edges]
                if not keep_idx:
                    print(
                        f"  [{setting} {modality}] no gg-graph gene JCPs in "
                        "profile data; skipping"
                    )
                    continue
                jcp_list = [jcp_full[i] for i in keep_idx]
                feat = feat_full[keep_idx]
            else:
                jcp_list = jcp_full
                feat = feat_full

            n = len(jcp_list)
            if n < 2:
                continue

            jcp_to_idx: dict[str, list[int]] = defaultdict(list)
            for i, j in enumerate(jcp_list):
                jcp_to_idx[j].append(i)

            sim = compute_cosine_similarity_matrix(feat, feat)
            jcp_array = np.array(jcp_list)
            same_jcp = jcp_array[:, None] == jcp_array[None, :]
            sim[same_jcp] = -np.inf

            pos = np.zeros((n, n), dtype=bool)
            for jcp_a, jcp_b in edges.iter_rows():
                i_list = jcp_to_idx.get(jcp_a)
                j_list = jcp_to_idx.get(jcp_b)
                if not i_list or not j_list:
                    continue
                for i in i_list:
                    for j in j_list:
                        if i != j:
                            pos[i, j] = True

            n_pos = int(pos.sum())
            n_q_with_pos = int((pos.sum(axis=1) > 0).sum())
            if n_pos == 0:
                print(f"  [{setting} {modality}] no positives; skipping")
                continue

            recall = calculate_recall_at_k(sim, pos, k_percentages)
            row = {
                "setting": setting,
                "target_modality": modality,
                "n_consensus_rows": n,
                "n_unique_jcps": len(set(jcp_list)),
                "n_queries_with_positives": n_q_with_pos,
                "n_positive_pairs": n_pos,
                **recall,
            }
            summary_rows.append(row)
            recall_str = ", ".join(f"{k}={v*100:.1f}%" for k, v in recall.items())
            print(
                f"  [{setting} {modality}] n={n}, q_with_pos={n_q_with_pos}, "
                f"pos={n_pos}, {recall_str}"
            )

    if summary_rows:
        pl.DataFrame(summary_rows).write_csv(output_dir / "motive_gg_summary.csv")
    return {"summary": summary_rows, "coverage": coverage}


# ---------------------------------------------------------------------------
# MOTIVE-CG (JCP-keyed, two settings)
# ---------------------------------------------------------------------------


def run_motive_cg(
    df: pl.DataFrame,
    features: list[str],
    cg_ann: pl.DataFrame,
    output_dir: Path,
    compound_col: str,
    group_col: str,
    negcon_col: str,
    k_percentages: list[float],
    settings: tuple[str, ...] = CG_SETTINGS_DEFAULT,
) -> dict[str, Any]:
    """JCP-space compound→gene retrieval, two reference-pool settings.

    Setting 'filtered': references = ORF/CRISPR JCPs that resolve from at least
    one MOTIVE compound→target edge. Mirrors MOTIVE's ``targets_with_links``
    filter from ``motive/jump.py``.

    Setting 'full': references = every ORF/CRISPR JCP present in the input
    profile dataframe (post split-restriction). Bigger haystack, harder task.
    """
    print("\n=== MOTIVE-CG ===")
    df_use = df.filter(pl.col(negcon_col) == False)

    cg_edges = (
        cg_ann.filter(pl.col("source") == "cg")
        .filter(pl.col("partner_jcp").is_not_null())
        .filter(pl.col("target_modality").is_not_null())
        .select(compound_col, "partner_jcp", "target_modality")
    )
    if cg_edges.height == 0:
        print("  no resolvable cg edges; skipping")
        return {"summary": [], "coverage": {}}

    # compound consensus profiles
    df_comp = df_use.filter(pl.col(group_col).is_in(list(COMPOUND_GROUPS)))
    if df_comp.height == 0:
        print("  no compound rows after filter; skipping")
        return {"summary": [], "coverage": {}}
    meta_comp, feat_comp = get_consensus_profiles(
        df_comp, features, compound_col, group_col
    )
    comp_jcp = meta_comp[compound_col].to_list()
    comp_group = meta_comp[group_col].to_list()
    comp_jcp_to_idx: dict[str, int] = {j: i for i, j in enumerate(comp_jcp)}

    # build positive map: compound_jcp → set[(gene_jcp, modality)]
    pos_map: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for c, t, m in cg_edges.iter_rows():
        pos_map[c].add((t, m))

    summary_rows: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {
        "compounds_with_resolvable_target": int(
            cg_edges[compound_col].n_unique()
        ),
        "unique_resolved_target_jcps_total": int(
            cg_edges["partner_jcp"].n_unique()
        ),
    }

    for modality in TARGET_MODALITIES:
        ref_group = f"group_{modality}"
        df_ref_full = df_use.filter(pl.col(group_col) == ref_group)
        if df_ref_full.height == 0:
            print(f"  no {ref_group} rows; skipping modality={modality}")
            continue
        meta_ref_full, feat_ref_full = get_consensus_profiles(
            df_ref_full, features, compound_col, group_col
        )
        ref_jcp_full = meta_ref_full[compound_col].to_list()

        target_jcps_for_modality = {
            t for edges in pos_map.values() for (t, m) in edges if m == modality
        }
        coverage[f"{modality}_target_jcps_in_motive"] = len(target_jcps_for_modality)
        coverage[f"{modality}_jcps_in_profiles"] = len(ref_jcp_full)
        coverage[f"{modality}_jcps_in_filtered_pool"] = len(
            target_jcps_for_modality & set(ref_jcp_full)
        )

        for setting in settings:
            if setting == "filtered":
                keep_idx = [
                    i for i, j in enumerate(ref_jcp_full)
                    if j in target_jcps_for_modality
                ]
                if not keep_idx:
                    print(
                        f"  [{setting} {modality}] no MOTIVE-target gene JCPs in "
                        "profile data; skipping"
                    )
                    continue
                ref_jcp = [ref_jcp_full[i] for i in keep_idx]
                feat_ref = feat_ref_full[keep_idx]
            else:  # full
                ref_jcp = ref_jcp_full
                feat_ref = feat_ref_full

            n_ref = len(ref_jcp)
            ref_jcp_to_idx = {j: i for i, j in enumerate(ref_jcp)}

            for compound_group in COMPOUND_GROUPS:
                q_indices = [
                    i for i, g in enumerate(comp_group) if g == compound_group
                ]
                if not q_indices:
                    continue

                pos_mask = np.zeros((len(q_indices), n_ref), dtype=bool)
                for qi, comp_idx in enumerate(q_indices):
                    c = comp_jcp[comp_idx]
                    for t, m in pos_map.get(c, ()):
                        if m != modality:
                            continue
                        ri = ref_jcp_to_idx.get(t)
                        if ri is not None:
                            pos_mask[qi, ri] = True

                n_pos = int(pos_mask.sum())
                n_q_with_pos = int((pos_mask.sum(axis=1) > 0).sum())
                if n_pos == 0:
                    print(
                        f"  [{setting} {compound_group} → {modality}] "
                        f"no positives; skipping"
                    )
                    continue

                sub_sim = compute_cosine_similarity_matrix(
                    feat_comp[q_indices], feat_ref
                )
                recall = calculate_recall_at_k(sub_sim, pos_mask, k_percentages)
                row = {
                    "setting": setting,
                    "compound_group": compound_group,
                    "target_modality": modality,
                    "n_query_compounds": len(q_indices),
                    "n_reference_genes": n_ref,
                    "n_queries_with_positives": n_q_with_pos,
                    "n_positive_pairs": n_pos,
                    **recall,
                }
                summary_rows.append(row)
                recall_str = ", ".join(f"{k}={v*100:.1f}%" for k, v in recall.items())
                print(
                    f"  [{setting} {compound_group} → {modality}] "
                    f"{n_q_with_pos}/{len(q_indices)} queries, "
                    f"{n_pos} pos pairs, {n_ref} refs, {recall_str}"
                )

    if summary_rows:
        pl.DataFrame(summary_rows).write_csv(output_dir / "motive_cg_summary.csv")
    return {"summary": summary_rows, "coverage": coverage}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, required=True)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("metadata/motive_annotations.parquet"),
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("metadata/motive_eval_compounds.parquet"),
    )
    parser.add_argument("--split-eval", type=str, default="test")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("metadata/metadata_dataset_filtered_4reps.parquet"),
        help="Used only if Metadata_JCP2022 is missing from the input parquet.",
    )
    parser.add_argument("--compound-col", type=str, default="Metadata_JCP2022")
    parser.add_argument("--group-col", type=str, default="Metadata_Group")
    parser.add_argument("--negcon-col", type=str, default="Metadata_negcon")
    parser.add_argument("--null-size", type=int, default=10_000)
    parser.add_argument("--p-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-compounds-per-target", type=int, default=3)
    parser.add_argument("--recall-k-percentages", type=str, default="1,5,10")
    parser.add_argument(
        "--run-pc",
        action="store_true",
        help="Run MOTIVE-PC (off by default — copairs PC dominates wall time).",
    )
    parser.add_argument("--skip-cc", action="store_true")
    parser.add_argument("--skip-cg", action="store_true")
    parser.add_argument("--skip-gg", action="store_true")
    parser.add_argument(
        "--run-filtered",
        action="store_true",
        help=(
            "Also evaluate the 'filtered' reference pool (gene JCPs that are "
            "MOTIVE targets) for CG and GG. By default only 'full' is reported."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing metrics.json / per-task CSVs.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.json"

    if metrics_path.exists() and not args.force:
        print(f"[skip] metrics.json already exists at {metrics_path}")
        return

    for name in PER_TASK_CSVS:
        _refuse_overwrite(args.output / name, args.force)

    k_percentages = [
        float(x) for x in args.recall_k_percentages.split(",") if x.strip()
    ]
    settings = (
        CG_SETTINGS_WITH_FILTERED if args.run_filtered else CG_SETTINGS_DEFAULT
    )

    # ---------------- load profiles ----------------
    print(f"[load] {args.input}")
    df = load_profiles(args.input)
    print(f"[load] shape={df.shape}")
    if args.compound_col not in df.columns:
        print(
            f"[load] {args.compound_col} missing — merging metadata from "
            f"{args.metadata}"
        )
        df = merge_metadata(df, args.metadata)
    df = setup_control_columns(df)
    df = df.filter(pl.col(args.compound_col).is_not_null())

    features, _ = infer_columns(df)
    features = get_numeric_features(df, features)
    if not features:
        raise ValueError("no numeric feature columns found in input parquet")
    print(f"[features] {len(features)} numeric features")

    # ---------------- load MOTIVE files ----------------
    if not args.annotations.exists():
        raise FileNotFoundError(
            f"motive annotations not found: {args.annotations}. "
            "Run `just motive-curate` first."
        )
    if not args.splits.exists():
        raise FileNotFoundError(
            f"motive splits not found: {args.splits}. "
            "Run `just motive-curate <splits-path>` first."
        )
    annotations = pl.read_parquet(args.annotations)
    splits = pl.read_parquet(args.splits)
    cg_ann = annotations.filter(pl.col("source") == "cg")
    cc_ann = annotations.filter(pl.col("source") == "cc")
    gg_ann = annotations.filter(pl.col("source") == "gg")
    print(
        f"[motive] cg rows={cg_ann.height:,}  cc rows={cc_ann.height:,}  "
        f"gg rows={gg_ann.height:,}  splits rows={splits.height:,}"
    )

    # ---------------- restrict to eval split ----------------
    df = _restrict_compounds_to_split(df, splits, args.split_eval, args.compound_col)
    if df.height == 0:
        raise ValueError("no rows remain after split restriction; aborting")

    # ---------------- run tasks ----------------
    metrics: dict[str, Any] = {
        "input_file": str(args.input),
        "split_eval": args.split_eval,
        "seed": args.seed,
        "n_features": len(features),
    }

    if not args.run_pc:
        print("[skip] MOTIVE-PC (pass --run-pc to enable)")
    else:
        print("\n=== MOTIVE-PC ===")
        df_pc = _attach_motive_target_list(df, cg_ann, args.compound_col)
        pc_results = calculate_phenotypic_consistency(
            df_pc,
            features,
            null_size=args.null_size,
            p_threshold=args.p_threshold,
            seed=args.seed,
            min_compounds_per_target=args.min_compounds_per_target,
            compound_col=args.compound_col,
            target_col="Metadata_target_list",
            negcon_col=args.negcon_col,
        )
        if pc_results["target_consistency"] is not None:
            pc_results["target_consistency"].to_csv(
                args.output / "motive_pc_per_target.csv", index=False
            )
        metrics["MOTIVE_PC"] = {
            "pct_targets_active": pc_results["pct_targets_active"],
            "n_targets_active": pc_results["n_targets_active"],
            "n_targets_total": pc_results["n_targets_total"],
            "median_n_total_pairs": pc_results["median_n_total_pairs"],
        }

    if args.skip_cc:
        print("[skip] MOTIVE-CC")
    else:
        cc_block = run_motive_cc(
            df,
            features,
            cc_ann,
            args.output,
            args.compound_col,
            args.group_col,
            args.negcon_col,
            k_percentages,
        )
        metrics["MOTIVE_CC"] = cc_block

    if args.skip_cg:
        print("[skip] MOTIVE-CG")
    else:
        cg_block = run_motive_cg(
            df,
            features,
            cg_ann,
            args.output,
            args.compound_col,
            args.group_col,
            args.negcon_col,
            k_percentages,
            settings=settings,
        )
        metrics["MOTIVE_CG"] = cg_block

    if args.skip_gg:
        print("[skip] MOTIVE-GG")
    else:
        gg_block = run_motive_gg(
            df,
            features,
            gg_ann,
            args.output,
            args.compound_col,
            args.group_col,
            args.negcon_col,
            k_percentages,
            settings=settings,
        )
        metrics["MOTIVE_GG"] = gg_block

    # ---------------- write metrics.json ----------------
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\n[write] {metrics_path}")
    for name in PER_TASK_CSVS:
        p = args.output / name
        if p.exists():
            print(f"[write] {p}")


if __name__ == "__main__":
    main()
