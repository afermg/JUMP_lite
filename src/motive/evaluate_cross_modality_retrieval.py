#!/usr/bin/env python3
"""Cross-Modality Retrieval Evaluation.

Evaluates how well compound profiles retrieve their corresponding
ORF/CRISPR genetic perturbation profiles based on shared targets.

For each compound with a target annotation:
1. Rank all ORF/CRISPR profiles by cosine similarity
2. Calculate recall at 1%, 5%, 10% based on matching targets

This module is called from evaluate_phenotypic_activity.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


def compute_cosine_similarity_matrix(
    query_features: np.ndarray,
    reference_features: np.ndarray,
) -> np.ndarray:
    """
    Compute cosine similarity between query and reference feature matrices.

    Args:
        query_features: (n_queries, n_features) array
        reference_features: (n_references, n_features) array

    Returns:
        (n_queries, n_references) cosine similarity matrix
    """
    # Normalize rows to unit vectors
    query_norms = np.linalg.norm(query_features, axis=1, keepdims=True)
    ref_norms = np.linalg.norm(reference_features, axis=1, keepdims=True)

    # Avoid division by zero
    query_norms = np.where(query_norms == 0, 1, query_norms)
    ref_norms = np.where(ref_norms == 0, 1, ref_norms)

    query_normalized = query_features / query_norms
    ref_normalized = reference_features / ref_norms

    # Cosine similarity is dot product of normalized vectors
    similarity = query_normalized @ ref_normalized.T

    return similarity


def calculate_recall_at_k(
    sim: np.ndarray,
    positive_mask: np.ndarray,
    k_percentages: list[float] = [1.0, 5.0, 10.0],
) -> dict[str, float]:
    """
    Calculate recall at k% by ranking references by similarity per query.

    For each query, counts how many of its positive references appear in the
    top-k% of references when ordered by descending ``sim``. Returns the mean
    across queries with at least one positive.

    Args:
        sim: (n_queries, n_references) similarity matrix; higher = more similar.
        positive_mask: (n_queries, n_references) bool — True where (query, ref)
            is a true positive pair.
        k_percentages: List of k values as percentages (e.g., [1, 5, 10]).

    Returns:
        Dictionary with recall@k for each percentage.
    """
    n_queries, n_references = sim.shape
    n_pos = positive_mask.sum(axis=1)
    valid = n_pos > 0
    results: dict[str, float] = {}

    for k_pct in k_percentages:
        k = max(1, int(n_references * k_pct / 100))
        # Top-k references per query (unsorted within top-k — fine for recall).
        top_k = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        rows = np.arange(n_queries)[:, None]
        hits = positive_mask[rows, top_k].sum(axis=1)
        recall_per_query = hits[valid] / n_pos[valid]
        avg_recall = float(recall_per_query.mean()) if recall_per_query.size else 0.0
        results[f"recall@{k_pct:.0f}%"] = avg_recall

    return results


def get_consensus_profiles(
    df: pl.DataFrame,
    features: list[str],
    compound_col: str = "Metadata_JCP2022",
    group_col: str = "Metadata_Group",
) -> tuple[pl.DataFrame, np.ndarray]:
    """
    Get consensus (median) profiles per compound.

    Args:
        df: DataFrame with profiles
        features: Feature column names
        compound_col: Column with compound identifier
        group_col: Column with group identifier

    Returns:
        Tuple of (metadata DataFrame, feature array)
    """
    # Group by compound and group, take median of features
    agg_exprs = [pl.col(f).median() for f in features]

    # Get all metadata columns
    meta_cols = [c for c in df.columns if c.startswith("Metadata")]

    # For metadata, take first value (should be same within group)
    meta_agg = [pl.col(c).first() for c in meta_cols if c not in [compound_col, group_col]]

    consensus = (
        df.group_by([compound_col, group_col])
        .agg(agg_exprs + meta_agg)
    )

    metadata = consensus.select([c for c in consensus.columns if c.startswith("Metadata")])
    feature_array = consensus.select(features).to_numpy()

    return metadata, feature_array


def evaluate_cross_modality_retrieval(
    df: pl.DataFrame,
    features: list[str],
    annotations: pl.DataFrame,
    output_dir: Path,
    activity_map: "pd.DataFrame | None" = None,
    compound_col: str = "Metadata_JCP2022",
    group_col: str = "Metadata_Group",
    negcon_col: str = "Metadata_negcon",
    target_col: str = "target",
    k_percentages: list[float] = [1.0, 5.0, 10.0],
    similarity_directions: list[str] = ["top", "bottom"],
    include_genetic_pairs: bool = False,
) -> dict[str, Any]:
    """
    Evaluate cross-modality retrieval for compounds vs ORF/CRISPR.

    Iterates through:
    - Similarity direction (top = most similar, bottom = least similar)
    - Activity filter modes (all compounds vs active-only)
    - Groups (HIGH, LOW, and optionally ORF, CRISPR)
    - CrossModalityTiers (Tier1, Tier2, Tier3, combinations) - for compounds only
    - Modality (all, Positive, Negative) - for compounds only

    For each compound, ranks ORF and CRISPR profiles by cosine similarity
    and calculates recall@k based on matching targets.

    When include_genetic_pairs=True, also evaluates:
    - ORF vs CRISPR retrieval (matching on gene symbol)
    - CRISPR vs ORF retrieval (matching on gene symbol)

    Args:
        df: Profiles with metadata
        features: Feature column names
        annotations: Tier annotations with CrossModalityTier
        output_dir: Directory to save results
        activity_map: Per-compound phenotypic activity results
        compound_col: Column containing compound identifier
        group_col: Column containing group identifier
        negcon_col: Column containing negative control flag
        target_col: Column containing target gene name in annotations
        k_percentages: List of k values as percentages for recall
        similarity_directions: List of directions to evaluate ("top" and/or "bottom")
            - "top": recall among most similar (highest cosine similarity)
            - "bottom": recall among least similar (lowest cosine similarity)
        include_genetic_pairs: If True, also evaluate ORF vs CRISPR retrieval

    Returns:
        Dictionary with results for each subset
    """
    import pandas as pd

    print("\n" + "=" * 60)
    print("CROSS-MODALITY RETRIEVAL EVALUATION")
    print("=" * 60)

    results = {}
    all_results = []

    # Get unique CrossModalityTiers (excluding any nulls)
    base_tiers = [
        t for t in annotations["CrossModalityTier"].unique().to_list()
        if t is not None
    ]
    base_tiers = sorted(base_tiers)  # Tier1, Tier2, Tier3
    print(f"CrossModality Tiers available: {base_tiers}")

    # Create tier options including combinations
    tier_options = []
    tier_to_filter = {}

    for tier in base_tiers:
        tier_options.append(tier)
        tier_to_filter[tier] = [tier]

    # Add cumulative combinations
    if len(base_tiers) >= 2:
        combo_name = "+".join(base_tiers[:2])
        tier_options.append(combo_name)
        tier_to_filter[combo_name] = base_tiers[:2]

    if len(base_tiers) >= 3:
        combo_name = "+".join(base_tiers[:3])
        tier_options.append(combo_name)
        tier_to_filter[combo_name] = base_tiers[:3]

    print(f"Tier options: {tier_options}")

    # Modality options
    modality_values = annotations["modality_clean"].unique().to_list()
    modality_options = ["all"] + sorted([m for m in modality_values if m is not None])
    print(f"Modality options: {modality_options}")

    # Compound groups (HIGH, LOW)
    compound_groups = ["group_high", "group_low"]

    # Reference groups (ORF, CRISPR)
    reference_groups = ["group_orf", "group_crispr"]

    # Check which groups are available
    available_groups = df[group_col].unique().to_list()
    compound_groups = [g for g in compound_groups if g in available_groups]
    reference_groups = [g for g in reference_groups if g in available_groups]

    print(f"Compound groups: {compound_groups}")
    print(f"Reference groups: {reference_groups}")

    if not compound_groups or not reference_groups:
        print("Warning: Missing required groups for cross-modality retrieval")
        return {"summary": [], "subset_results": {}}

    # Filter out negative controls
    df_filtered = df.filter(pl.col(negcon_col) == False)
    print(f"Samples after filtering negcons: {df_filtered.height}")

    # Get consensus profiles for all groups
    print("\nComputing consensus profiles...")
    metadata_all, features_all = get_consensus_profiles(
        df_filtered, features, compound_col, group_col
    )

    # Convert to lists for efficient indexing
    jcp_list = metadata_all[compound_col].to_list()
    groups_list = metadata_all[group_col].to_list()
    symbol_list = (
        metadata_all["Metadata_Symbol"].to_list()
        if "Metadata_Symbol" in metadata_all.columns
        else [None] * len(jcp_list)
    )

    # Create JCP to index mapping
    jcp_to_idx = {jcp: i for i, jcp in enumerate(jcp_list)}

    # Active filter modes
    if activity_map is not None and len(activity_map) > 0:
        active_filter_modes = [False, True]
        print("Active filter modes: [all_compounds, active_only]")

        # Pre-compute active JCPs per group
        active_jcps_by_group = {}
        for grp in activity_map["Metadata_Group"].unique():
            grp_active = activity_map[
                (activity_map["Metadata_Group"] == grp) &
                (activity_map["below_corrected_p"] == True)
            ][compound_col].tolist()
            active_jcps_by_group[grp] = set(grp_active)
            print(f"  Active JCPs in {grp}: {len(grp_active)}")
    else:
        active_filter_modes = [False]
        active_jcps_by_group = {}
        print("Active filter modes: [all_compounds] (no activity_map)")

    # Compute full similarity matrix once
    print("\nComputing cosine similarity matrix...")
    similarity_matrix = compute_cosine_similarity_matrix(features_all, features_all)
    print(f"  Similarity matrix shape: {similarity_matrix.shape}")

    # Main evaluation loop
    print(f"\nSimilarity directions to evaluate: {similarity_directions}")

    for filter_active_only in active_filter_modes:
        active_mode_name = "active_only" if filter_active_only else "all_compounds"
        print(f"\n{'='*50}")
        print(f"Active filter: {active_mode_name}")
        print(f"{'='*50}")

        for compound_group in compound_groups:
            print(f"\n--- Compound Group: {compound_group} ---")

            for tier_name in tier_options:
                print(f"  Tier: {tier_name}")

                # Get annotations for this tier
                tiers_to_include = tier_to_filter[tier_name]
                tier_ann = annotations.filter(
                    pl.col("CrossModalityTier").is_in(tiers_to_include)
                )

                if tier_ann.height == 0:
                    print(f"    No annotations for {tier_name}")
                    continue

                for modality_clean in modality_options:
                    # Filter annotations by modality
                    if modality_clean == "all":
                        subset_ann = tier_ann
                    else:
                        subset_ann = tier_ann.filter(
                            pl.col("modality_clean") == modality_clean
                        )

                    if subset_ann.height == 0:
                        continue

                    # Get compounds with targets in this subset
                    compound_jcps = subset_ann[compound_col].unique().to_list()

                    # Filter to compounds in the current group
                    compound_indices = [
                        jcp_to_idx[jcp]
                        for jcp in compound_jcps
                        if jcp in jcp_to_idx and groups_list[jcp_to_idx[jcp]] == compound_group
                    ]

                    if not compound_indices:
                        continue

                    # Apply activity filter if enabled
                    if filter_active_only:
                        active_jcps = active_jcps_by_group.get(compound_group, set())
                        compound_indices = [
                            idx for idx in compound_indices
                            if jcp_list[idx] in active_jcps
                        ]

                    if len(compound_indices) < 1:
                        continue

                    # Build target mapping for compounds
                    compound_to_targets = {}
                    for idx in compound_indices:
                        jcp = jcp_list[idx]
                        targets = subset_ann.filter(
                            pl.col(compound_col) == jcp
                        )[target_col].unique().to_list()
                        compound_to_targets[jcp] = set(targets)

                    # Evaluate against each reference group
                    for ref_group in reference_groups:
                        ref_name = ref_group.replace("group_", "").upper()

                        # Get reference indices
                        ref_indices = [
                            i for i, g in enumerate(groups_list)
                            if g == ref_group
                        ]

                        if not ref_indices:
                            continue

                        # Apply activity filter to references if enabled
                        if filter_active_only:
                            active_refs = active_jcps_by_group.get(ref_group, set())
                            ref_indices = [
                                idx for idx in ref_indices
                                if jcp_list[idx] in active_refs
                            ]

                        if len(ref_indices) < 1:
                            continue

                        # Get reference JCPs and their gene symbols
                        ref_jcps = [jcp_list[idx] for idx in ref_indices]

                        # Build reference gene symbol mapping
                        # For ORF/CRISPR, the Symbol column contains the target gene
                        ref_to_symbol = {}
                        for idx in ref_indices:
                            jcp = jcp_list[idx]
                            symbol = symbol_list[idx]
                            if symbol:
                                ref_to_symbol[jcp] = symbol

                        if not ref_to_symbol:
                            print(f"      {modality_clean} vs {ref_name}: No symbols found")
                            continue

                        # Extract similarity submatrix
                        query_indices_arr = np.array(compound_indices)
                        ref_indices_arr = np.array(ref_indices)

                        sub_similarity = similarity_matrix[np.ix_(query_indices_arr, ref_indices_arr)]

                        # Compute rankings for both directions
                        # Build positive mask
                        n_queries = len(compound_indices)
                        n_refs = len(ref_indices)
                        positive_mask = np.zeros((n_queries, n_refs), dtype=bool)

                        for qi, comp_idx in enumerate(compound_indices):
                            comp_jcp = jcp_list[comp_idx]
                            comp_targets = compound_to_targets.get(comp_jcp, set())

                            for ri, ref_idx in enumerate(ref_indices):
                                ref_jcp = jcp_list[ref_idx]
                                ref_symbol = ref_to_symbol.get(ref_jcp, "")

                                # Check if reference gene matches any compound target
                                if ref_symbol in comp_targets:
                                    positive_mask[qi, ri] = True

                        n_positives = positive_mask.sum()
                        n_queries_with_positives = (positive_mask.sum(axis=1) > 0).sum()

                        if n_positives == 0:
                            continue

                        # Loop through similarity directions.
                        # top: rank by descending sim. bottom: rank by ascending sim
                        # (== descending -sim).
                        for sim_direction in similarity_directions:
                            sim_for_recall = (
                                sub_similarity
                                if sim_direction == "top"
                                else -sub_similarity
                            )
                            recall_results = calculate_recall_at_k(
                                sim_for_recall, positive_mask, k_percentages
                            )

                            subset_name = f"{compound_group}_{tier_name}_{modality_clean}_{active_mode_name}_vs_{ref_name}_{sim_direction}"

                            result_entry = {
                                "subset": subset_name,
                                "compound_group": compound_group,
                                "tier": tier_name,
                                "modality_clean": modality_clean,
                                "active_filter": active_mode_name,
                                "reference_group": ref_name,
                                "similarity_direction": sim_direction,
                                "n_query_compounds": len(compound_indices),
                                "n_reference_perturbations": len(ref_indices),
                                "n_queries_with_positives": int(n_queries_with_positives),
                                "n_positive_pairs": int(n_positives),
                            }
                            result_entry.update(recall_results)

                            results[subset_name] = result_entry
                            all_results.append(result_entry)

                            recall_str = ", ".join(
                                f"{k}: {v*100:.1f}%"
                                for k, v in recall_results.items()
                            )
                            print(
                                f"      {modality_clean} vs {ref_name} ({sim_direction}): "
                                f"{n_queries_with_positives}/{len(compound_indices)} queries, "
                                f"{n_positives} pos pairs, {recall_str}"
                            )

    # =========================================================================
    # Genetic Perturbation Pairs (ORF vs CRISPR)
    # =========================================================================
    if include_genetic_pairs:
        print("\n" + "#" * 60)
        print("# GENETIC PERTURBATION CROSS-MODALITY (ORF vs CRISPR)")
        print("#" * 60)

        # Define genetic query/reference pairs
        genetic_pairs = [
            ("group_orf", "group_crispr"),
            ("group_crispr", "group_orf"),
        ]

        # Filter to available pairs
        genetic_pairs = [
            (q, r) for q, r in genetic_pairs
            if q in available_groups and r in available_groups
        ]

        if not genetic_pairs:
            print("Warning: ORF and/or CRISPR groups not available")
        else:
            for query_group, ref_group in genetic_pairs:
                query_name = query_group.replace("group_", "").upper()
                ref_name = ref_group.replace("group_", "").upper()
                print(f"\n--- {query_name} vs {ref_name} ---")

                for filter_active_only in active_filter_modes:
                    active_mode_name = "active_only" if filter_active_only else "all_compounds"

                    # Get query indices (genetic perturbations use Metadata_Symbol as target)
                    query_indices = [
                        i for i, g in enumerate(groups_list)
                        if g == query_group
                    ]

                    # Get reference indices
                    ref_indices = [
                        i for i, g in enumerate(groups_list)
                        if g == ref_group
                    ]

                    if not query_indices or not ref_indices:
                        continue

                    # Apply activity filter if enabled
                    if filter_active_only:
                        active_query = active_jcps_by_group.get(query_group, set())
                        active_ref = active_jcps_by_group.get(ref_group, set())

                        query_indices = [
                            idx for idx in query_indices
                            if jcp_list[idx] in active_query
                        ]
                        ref_indices = [
                            idx for idx in ref_indices
                            if jcp_list[idx] in active_ref
                        ]

                    if len(query_indices) < 1 or len(ref_indices) < 1:
                        print(f"  {active_mode_name}: Skipping (not enough samples)")
                        continue

                    # Build query symbol mapping (for genetic perturbations, symbol IS the target)
                    query_to_symbol = {}
                    for idx in query_indices:
                        jcp = jcp_list[idx]
                        symbol = symbol_list[idx]
                        if symbol:
                            query_to_symbol[jcp] = symbol

                    # Build reference symbol mapping
                    ref_to_symbol = {}
                    for idx in ref_indices:
                        jcp = jcp_list[idx]
                        symbol = symbol_list[idx]
                        if symbol:
                            ref_to_symbol[jcp] = symbol

                    if not query_to_symbol or not ref_to_symbol:
                        print(f"  {active_mode_name}: No symbols found")
                        continue

                    # Extract similarity submatrix
                    query_indices_arr = np.array(query_indices)
                    ref_indices_arr = np.array(ref_indices)

                    sub_similarity = similarity_matrix[np.ix_(query_indices_arr, ref_indices_arr)]

                    # Build positive mask (matching gene symbols)
                    n_queries = len(query_indices)
                    n_refs = len(ref_indices)
                    positive_mask = np.zeros((n_queries, n_refs), dtype=bool)

                    for qi, query_idx in enumerate(query_indices):
                        query_jcp = jcp_list[query_idx]
                        query_symbol = query_to_symbol.get(query_jcp, "")

                        for ri, ref_idx in enumerate(ref_indices):
                            ref_jcp = jcp_list[ref_idx]
                            ref_symbol = ref_to_symbol.get(ref_jcp, "")

                            # Check if gene symbols match
                            if query_symbol and ref_symbol and query_symbol == ref_symbol:
                                positive_mask[qi, ri] = True

                    n_positives = positive_mask.sum()
                    n_queries_with_positives = (positive_mask.sum(axis=1) > 0).sum()

                    if n_positives == 0:
                        print(f"  {active_mode_name}: No matching gene symbols found")
                        continue

                    # Loop through similarity directions.
                    # top: rank by descending sim. bottom: rank by ascending sim.
                    for sim_direction in similarity_directions:
                        sim_for_recall = (
                            sub_similarity
                            if sim_direction == "top"
                            else -sub_similarity
                        )
                        recall_results = calculate_recall_at_k(
                            sim_for_recall, positive_mask, k_percentages
                        )

                        subset_name = f"{query_group}_vs_{ref_group}_{active_mode_name}_{sim_direction}"

                        result_entry = {
                            "subset": subset_name,
                            "compound_group": query_group,
                            "tier": "genetic",
                            "modality_clean": "genetic",
                            "active_filter": active_mode_name,
                            "reference_group": ref_name,
                            "similarity_direction": sim_direction,
                            "n_query_compounds": len(query_indices),
                            "n_reference_perturbations": len(ref_indices),
                            "n_queries_with_positives": int(n_queries_with_positives),
                            "n_positive_pairs": int(n_positives),
                        }
                        result_entry.update(recall_results)

                        results[subset_name] = result_entry
                        all_results.append(result_entry)

                        recall_str = ", ".join(
                            f"{k}: {v*100:.1f}%"
                            for k, v in recall_results.items()
                        )
                        print(
                            f"  {active_mode_name} ({sim_direction}): "
                            f"{n_queries_with_positives}/{len(query_indices)} queries, "
                            f"{n_positives} pos pairs, {recall_str}"
                        )

    # Save summary
    if all_results:
        summary_df = pd.DataFrame(all_results)
        summary_path = output_dir / "cross_modality_retrieval_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved cross-modality retrieval summary to: {summary_path}")

    return {
        "summary": all_results,
        "subset_results": results,
    }


def main():
    """Test function for standalone execution."""
    print("Cross-modality retrieval module.")
    print("This module is intended to be called from evaluate_phenotypic_activity.py")


if __name__ == "__main__":
    main()
