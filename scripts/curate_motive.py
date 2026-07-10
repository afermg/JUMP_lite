#!/usr/bin/env python3
"""Curate MOTIVE annotations and evaluation splits.

One-shot prep with three modes:

- ``--mode full`` (default) — writes ``motive_annotations.parquet``:
  - ``cg`` rows: explode ``Metadata_MOTIVE_target`` for compound rows where
    the target is known, then resolve each gene symbol to its ORF and/or
    CRISPR JCP via the metadata parquet. All compound→gene relationship
    types are pooled (no ``rel_type`` filter).
  - ``cc`` rows: from the raw compound-compound annotation parquet, map
    each side's InChIKey to JCP2022, drop self-pairs, symmetrise.
  - ``gg`` rows: from the gene-gene annotation parquet, restrict both sides
    to JUMP gene symbols, explode into JCP×JCP within each modality.
    All ``rel_type`` values pooled.

- ``--mode strict`` — writes ``motive_annotations_strict.parquet``:
  - ``cg`` rows: derived from ``annotations_compound_gene_curated.parquet``
    filtered to direct-mechanism ``rel_type``s (drugbank target/binding/
    inhibition/activation/enzyme; biokg DRUG_BINDING_GENE / DRUG_INHIBITION
    _GENE / DRUG_ACTIVATION_GENE; "binds"/"inhibitor"/"agonist"/"antagonist").
    Drops UPREGULATES/DOWNREGULATES (transcriptional cascade), ASSOCIATES_CHaG
    (literature co-mention), DRUG_REACTION_GENE / DRUG_CATALYSIS_GENE (drug
    metabolism). Resolved to gene_jcp via metadata gene lookup.
  - ``cc`` rows: NOT derived from the cc parquet (RESEMBLES/DDI/synergy are
    too noisy). Instead, pairs of compounds that share at least one
    high-confidence target via the strict cg subset.
  - ``gg`` rows: filtered to ``ppi``/``PPI``, ``GENE_BINDING_GENE``,
    ``INTERACTS_GiG``/``interacts``, ``GENE_PTMOD_GENE``. Drops metabolic
    co-occurrence, transcriptional regulation, statistical co-variation.

- ``--mode ultra_strict`` — writes ``motive_annotations_ultra_strict.parquet``:
  - ``cg`` rows: same source as strict, but with the rel_type allowlist
    completed for direct-binding pharmacology (adds blocker, modulator,
    activator, partial agonist, inverse agonist, DRUG_BINDACT_GENE, etc.)
    and ``enzyme`` removed (drug metabolism, reversed causal arrow).
  - ``cc`` rows: bridged via shared CG target *and* matching action class
    (inhibitory / activating / binding). Drops cross-class pairs like
    "agonist of FOO + antagonist of FOO" that strict's bridge accepted.
  - ``gg`` rows: tightened to ``ppi``/``PPI``/``GENE_BINDING_GENE``/
    ``GENE_PTMOD_GENE`` only. Drops ``INTERACTS_GiG``/``interacts`` from
    strict (~294k rows of generic / text-mined edges).

  See the long comment above ``ULTRA_STRICT_CG_REL_TYPES`` for rationale.

All modes also write ``motive_eval_compounds.parquet`` (mapping the upstream MOTIVE
split file to JCP2022) — same in any mode.

The schema of the output parquet is identical between modes:
``(Metadata_JCP2022, target, partner_jcp, target_modality,
   source ∈ {cg, cc, gg})``. The downstream evaluator
``evaluation/evaluate_motive.py`` does not need any changes — pass the
strict file via ``--annotations metadata/motive_annotations_strict.parquet``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl


CG_KEY_CANDIDATES = (
    "Metadata_JCP2022",
    "JCP2022",
    "Metadata_InChIKey",
    "InChIKey",
    "inchikey",
    "compound_inchikey",
)


def build_cg_annotations(metadata_path: Path) -> pl.DataFrame:
    """Build cg edges as ``(compound_jcp, target_symbol, gene_jcp, target_modality)``.

    Each output row is one resolved (compound, gene-reagent) edge. A single
    MOTIVE compound→target_symbol annotation may produce multiple rows: one per
    matching ORF JCP and one per matching CRISPR JCP. Symbols with no
    matching ORF/CRISPR JCP in the metadata parquet are dropped.

    Output schema: ``Metadata_JCP2022, target, partner_jcp, target_modality,
    source="cg"`` where ``partner_jcp`` is the resolved gene JCP.
    """
    print(f"[cg] reading {metadata_path}")
    md = pl.read_parquet(metadata_path).with_columns(
        pl.col("Metadata_JCP2022").cast(pl.Utf8),
        pl.col("Metadata_Symbol").cast(pl.Utf8),
        pl.col("Metadata_MOTIVE_target").cast(pl.Utf8),
        pl.col("Metadata_Perturbation_Type").cast(pl.Utf8),
    )

    required = {
        "Metadata_JCP2022",
        "Metadata_Perturbation_Type",
        "Metadata_MOTIVE_target",
        "Metadata_Symbol",
    }
    missing = required - set(md.columns)
    if missing:
        raise ValueError(f"metadata parquet missing columns: {sorted(missing)}")

    compounds = (
        md.filter(pl.col("Metadata_Perturbation_Type") == "compound")
        .filter(pl.col("Metadata_MOTIVE_target").is_not_null())
        .filter(pl.col("Metadata_MOTIVE_target") != "unknown")
        .select("Metadata_JCP2022", "Metadata_MOTIVE_target")
        .unique(subset=["Metadata_JCP2022"])
    )
    print(f"[cg] {compounds.height:,} compounds with MOTIVE targets")

    edges = (
        compounds.with_columns(
            pl.col("Metadata_MOTIVE_target").str.split("|").alias("target_list")
        )
        .explode("target_list")
        .filter(pl.col("target_list").is_not_null())
        .filter(pl.col("target_list") != "")
        .rename({"target_list": "target"})
        .select("Metadata_JCP2022", "target")
        .unique()
    )
    n_total_targets = edges["target"].n_unique()
    n_total_edges = edges.height
    print(
        f"[cg] {n_total_edges:,} (compound, target) edges over "
        f"{n_total_targets:,} unique target symbols"
    )

    gene_lookup = (
        md.filter(pl.col("Metadata_Perturbation_Type").is_in(["orf", "crispr"]))
        .filter(pl.col("Metadata_Symbol").is_not_null())
        .select(
            pl.col("Metadata_Symbol").alias("target"),
            pl.col("Metadata_JCP2022").alias("gene_jcp"),
            pl.col("Metadata_Perturbation_Type").alias("target_modality"),
        )
        .unique()
    )
    print(
        f"[cg] gene lookup: {gene_lookup.height:,} (symbol, gene_jcp, modality) rows"
    )

    resolved = edges.join(gene_lookup, on="target", how="inner")
    n_resolved_targets = resolved["target"].n_unique()
    n_resolved_compounds = resolved["Metadata_JCP2022"].n_unique()
    print(
        f"[cg] resolved {n_resolved_targets:,}/{n_total_targets:,} targets "
        f"({100*n_resolved_targets/n_total_targets:.1f}%) to ≥1 gene JCP — "
        f"covers {n_resolved_compounds:,}/{compounds.height:,} compounds"
    )
    print(
        f"[cg] {resolved.height:,} resolved (compound, target, gene_jcp, modality) rows"
    )

    return resolved.select(
        pl.col("Metadata_JCP2022"),
        pl.col("target"),
        pl.col("gene_jcp").alias("partner_jcp"),
        pl.col("target_modality"),
        pl.lit("cg").alias("source"),
    )


def build_cc_annotations(
    annotations_cc_path: Path,
    inchikey_map_path: Path,
) -> pl.DataFrame:
    """Symmetrised compound→compound graph keyed by JCP2022.

    Mirrors the connectivity-key trick at ``scripts/build_metadata_dataset.py:343-360``.
    Output: ``Metadata_JCP2022, target=null, source="cc", partner_jcp``.
    """
    print(f"[cc] reading {annotations_cc_path}")
    cc = pl.read_parquet(annotations_cc_path)
    required = {"inchikey_a", "inchikey_b"}
    missing = required - set(cc.columns)
    if missing:
        raise ValueError(f"cc parquet missing columns: {sorted(missing)}")

    print(f"[cc] reading {inchikey_map_path}")
    ik_map = pl.read_csv(inchikey_map_path).select(
        "InChIKey_Connectivity", "Metadata_JCP2022"
    )
    if ik_map.height == 0:
        raise ValueError(f"empty inchikey map: {inchikey_map_path}")

    cc_pairs = (
        cc.select(
            pl.col("inchikey_a").str.split("-").list.first().alias("ik_a"),
            pl.col("inchikey_b").str.split("-").list.first().alias("ik_b"),
        )
        .filter(pl.col("ik_a").is_not_null() & pl.col("ik_b").is_not_null())
        .unique()
    )
    print(f"[cc] {cc_pairs.height:,} unique connectivity pairs")

    joined = (
        cc_pairs.join(
            ik_map.rename(
                {"InChIKey_Connectivity": "ik_a", "Metadata_JCP2022": "jcp_a"}
            ),
            on="ik_a",
            how="inner",
        )
        .join(
            ik_map.rename(
                {"InChIKey_Connectivity": "ik_b", "Metadata_JCP2022": "jcp_b"}
            ),
            on="ik_b",
            how="inner",
        )
        .filter(pl.col("jcp_a") != pl.col("jcp_b"))
        .select("jcp_a", "jcp_b")
        .unique()
    )
    print(f"[cc] {joined.height:,} pairs after JCP join + self-drop")

    symmetrised = pl.concat(
        [
            joined,
            joined.rename({"jcp_a": "jcp_b", "jcp_b": "jcp_a"}).select(
                "jcp_a", "jcp_b"
            ),
        ]
    ).unique()
    print(f"[cc] {symmetrised.height:,} pairs after symmetrise")

    return symmetrised.select(
        pl.col("jcp_a").alias("Metadata_JCP2022"),
        pl.lit(None, dtype=pl.Utf8).alias("target"),
        pl.col("jcp_b").alias("partner_jcp"),
        pl.lit(None, dtype=pl.Utf8).alias("target_modality"),
        pl.lit("cc").alias("source"),
    )


def build_gg_annotations(
    annotations_gg_path: Path,
    metadata_path: Path,
) -> pl.DataFrame:
    """Symmetrised gene→gene graph keyed by JCP2022, partitioned by modality.

    Mirrors MOTIVE's ``load_gene_gene_annotations`` (motive/jump.py): drop
    self-loops at symbol level, canonicalise pair order, dedup, restrict to
    JUMP gene symbols. Then explode each symbol pair into JCP×JCP pairs
    within each modality (ORF↔ORF and CRISPR↔CRISPR; no cross-modality),
    drop self-loops at JCP level, and symmetrise.

    Output schema: ``Metadata_JCP2022 (gene_a JCP), target=null,
    partner_jcp (gene_b JCP), target_modality ∈ {orf, crispr}, source="gg"``.
    """
    print(f"[gg] reading {annotations_gg_path}")
    gg = pl.read_parquet(annotations_gg_path)
    required = {"target_a", "target_b"}
    missing = required - set(gg.columns)
    if missing:
        raise ValueError(f"gg parquet missing columns: {sorted(missing)}")

    canon = (
        gg.select("target_a", "target_b")
        .filter(pl.col("target_a").is_not_null() & pl.col("target_b").is_not_null())
        .filter(pl.col("target_a") != pl.col("target_b"))
        .with_columns(
            pl.when(pl.col("target_a") < pl.col("target_b"))
            .then(pl.col("target_a"))
            .otherwise(pl.col("target_b"))
            .alias("sym_lo"),
            pl.when(pl.col("target_a") < pl.col("target_b"))
            .then(pl.col("target_b"))
            .otherwise(pl.col("target_a"))
            .alias("sym_hi"),
        )
        .select("sym_lo", "sym_hi")
        .unique()
    )
    print(f"[gg] {canon.height:,} canonical symbol pairs")

    md = pl.read_parquet(metadata_path).with_columns(
        pl.col("Metadata_Symbol").cast(pl.Utf8),
        pl.col("Metadata_JCP2022").cast(pl.Utf8),
        pl.col("Metadata_Perturbation_Type").cast(pl.Utf8),
    )
    gene_lookup = (
        md.filter(pl.col("Metadata_Perturbation_Type").is_in(["orf", "crispr"]))
        .filter(pl.col("Metadata_Symbol").is_not_null())
        .select(
            pl.col("Metadata_Symbol").alias("symbol"),
            pl.col("Metadata_JCP2022").alias("gene_jcp"),
            pl.col("Metadata_Perturbation_Type").alias("modality"),
        )
        .unique()
    )

    pieces = []
    for modality in ("orf", "crispr"):
        sym2jcp = gene_lookup.filter(pl.col("modality") == modality).select(
            "symbol", "gene_jcp"
        )
        joined = (
            canon.join(
                sym2jcp.rename({"symbol": "sym_lo", "gene_jcp": "jcp_lo"}),
                on="sym_lo",
                how="inner",
            )
            .join(
                sym2jcp.rename({"symbol": "sym_hi", "gene_jcp": "jcp_hi"}),
                on="sym_hi",
                how="inner",
            )
            .filter(pl.col("jcp_lo") != pl.col("jcp_hi"))
            .select("jcp_lo", "jcp_hi")
            .unique()
        )
        print(f"[gg/{modality}] {joined.height:,} JCP pairs after symbol→JCP explode")
        symmetrised = pl.concat(
            [
                joined,
                joined.rename({"jcp_lo": "jcp_hi", "jcp_hi": "jcp_lo"}).select(
                    "jcp_lo", "jcp_hi"
                ),
            ]
        ).unique()
        pieces.append(
            symmetrised.with_columns(pl.lit(modality).alias("modality_col"))
        )

    out = pl.concat(pieces, how="vertical")
    print(f"[gg] {out.height:,} total gg JCP pairs (symmetrised, all modalities)")

    return out.select(
        pl.col("jcp_lo").alias("Metadata_JCP2022"),
        pl.lit(None, dtype=pl.Utf8).alias("target"),
        pl.col("jcp_hi").alias("partner_jcp"),
        pl.col("modality_col").alias("target_modality"),
        pl.lit("gg").alias("source"),
    )


# ---------------------------------------------------------------------------
# strict-mode rel_type allowlists
# ---------------------------------------------------------------------------
#
# Why filter at all
# -----------------
# The raw biokg / openbiolink / pharmebinet annotation parquets pool many
# qualitatively different relationship types under one ``rel_type`` column.
# When every row becomes a positive in ``positive_mask``, the eval is asking
# "do these two profiles cluster because they share *any* literature edge?"
# That collapses signal (shared mechanism → shared morphology) with noise
# (transcriptional response, drug metabolism, generic co-mention, structural
# resemblance). Every false positive drags recall@k toward random.
#
# The strict allowlists below keep edges where a co-clustering of profiles
# is *causally plausible* — direct mechanism — and drop the rest.
#
# CG: compound → gene
# -------------------
# Keep:
#   target, targets                (drugbank gold standard)
#   binds, BINDS_CHbG,
#   DRUG_BINDING_GENE              (direct physical binding)
#   DRUG_INHIBITION_GENE,
#   inhibitor                      (direct inhibition)
#   DRUG_ACTIVATION_GENE,
#   agonist, antagonist            (direct mechanism)
#   enzyme                         (drugbank's curated enzyme target)
# Drop:
#   UPREGULATES_CHuG, upregulates  (transcriptional response — DOWNSTREAM of
#   DOWNREGULATES_CHdG,             the actual target; the compound may not
#   downregulates                   bind the gene whose mRNA changes)
#   ASSOCIATES_CHaG                (literature co-mention only — non-mechanistic)
#   DRUG_REACTION_GENE,            (drug metabolism — the gene's protein acts
#   DRUG_CATALYSIS_GENE             on the drug, not the other way; CYP-style
#                                   relationships, not "compound targets gene")
#   unknown                        (no claim being made)
#
# Empirically (one-config smoke on a morphem-hq run, JUMP-lite sweep),
# strict CG is ~4× tighter (75k rows vs 301k full) but recall@5% is similar
# (6.0% vs 6.0% on group_high → orf). The dropped edges aren't enriched
# for morphology *or* depleted relative to the kept ones — they're just
# less interpretable. Filtering doesn't help CG much, but doesn't hurt.
STRICT_CG_REL_TYPES = frozenset(
    {
        "target", "targets",
        "binds", "BINDS_CHbG", "DRUG_BINDING_GENE",
        "DRUG_INHIBITION_GENE", "inhibitor",
        "DRUG_ACTIVATION_GENE", "agonist", "antagonist",
        "enzyme",
    }
)

# GG: gene ↔ gene
# ---------------
# Keep:
#   ppi, PPI                       (physical protein-protein interaction —
#                                   strongest predictor of shared morphology
#                                   under perturbation; protein complexes,
#                                   direct binding partners)
#   GENE_BINDING_GENE              (binding, alternate notation)
#   INTERACTS_GiG, interacts       (generic but mechanistic interaction)
#   GENE_PTMOD_GENE                (post-translational modification — direct
#                                   one-protein-modifies-another relation)
# Drop:
#   GENE_GENE                      (too generic; openbiolink catch-all)
#   GENE_REACTION_GENE,            (co-occurrence in the same metabolic
#   GENE_CATALYSIS_GENE             reaction — same pathway often, but the
#                                   *phenotypes* of perturbing two enzymes
#                                   in a pathway are usually different)
#   regulates, REGULATES_GrG       (transcriptional regulation — the
#                                   regulator perturbation drives a cascade
#                                   that doesn't morphologically resemble
#                                   the perturbed-target cell)
#   covaries, COVARIES_GcG         (statistical co-variation across tissues —
#                                   weakest mechanistic prior we have)
#   GENE_ACTIVATION_GENE,          (could keep — they're mechanistic — but
#   GENE_INHIBITION_GENE            small in number and noisy in practice)
#
# Empirically, strict GG keeps ~40% of rows (1.13M vs 2.46M JCP-pairs). Lift
# on recall@5% is small (5.8% strict vs 5.6% full on the morphem smoke,
# crispr 6.8% vs 6.5%) — the filter helps but most of the GG signal is
# already in the PPI edges that dominate the full set anyway.
STRICT_GG_REL_TYPES = frozenset(
    {
        "ppi", "PPI",
        "GENE_BINDING_GENE",
        "INTERACTS_GiG", "interacts",
        "GENE_PTMOD_GENE",
    }
)


# ---------------------------------------------------------------------------
# ultra-strict allowlists + action-class map
# ---------------------------------------------------------------------------
#
# What "ultra strict" means here
# -------------------------------
# Three changes vs strict, all in the direction of "tighter mechanism story":
#
# 1. CG: the strict allowlist was incomplete on direct-binding actions (it
#    omitted blocker/modulator/positive modulator/allosteric modulator/
#    activator/partial agonist/inverse agonist/DRUG_BINDACT_GENE — all of
#    which are real direct-binding pharmacology). Ultra-strict adds them.
#    Ultra-strict also drops `enzyme` (biokg/primekg "gene encodes an enzyme
#    that metabolises the drug" — reversed causal arrow, same problem as
#    DRUG_REACTION_GENE that strict already drops). Net row count is
#    similar to strict (~238k vs ~240k); the rel_type distribution is
#    cleaner.
#
# 2. CC bridge requires *same action class*. Strict accepts (A,B) iff both
#    touch the same gene via any allowlisted rel_type, so an agonist of FOO
#    and an antagonist of FOO become a positive — opposite phenotypes. Ultra
#    strict tags each CG row with one of {inhibitory, activating, binding}
#    and the bridge requires both rows to share the class on the same gene.
#    "binding" = unspecified/mixed action (binds, targets, modulator,
#    DRUG_BINDACT_GENE) — pairs with itself only.
#
# 3. GG: drop INTERACTS_GiG / interacts (~294k rows of generic, often
#    text-mined edges). Keep PPI/PPI/GENE_BINDING_GENE/GENE_PTMOD_GENE only
#    (~900k physical/PTM rows). The argument is symmetric to dropping
#    upregulates/downregulates in CG: "interacts" is a low-confidence
#    catch-all and dilutes the PPI-dominated signal.
#
# Excluded from scope (deliberate, not oversights):
#   - No modality-aware (ORF vs CRISPR) policy. Adds complexity, biology
#     argument is shaky, can be added later.
#   - No database-trust filter. Rel_type-level filtering captures most of
#     the trust gradient already; per-database policies were not worth the
#     extra moving parts.
#   - No pChEMBL / IC50 / approval-status gating. Those columns aren't in
#     the curated parquet — would require an upstream pull.
ULTRA_STRICT_CG_REL_TYPES = frozenset(
    {
        # direct binding (action unspecified)
        "binds", "targets",
        # inhibitory
        "inhibitor", "antagonist", "blocker",
        "negative modulator", "inverse agonist",
        "DRUG_INHIBITION_GENE",
        "inhibitory allosteric modulator",
        # activating
        "agonist", "activator",
        "positive modulator", "partial agonist",
        "DRUG_ACTIVATION_GENE",
        # modulators (binding, action mixed/unspecified)
        "modulator", "allosteric modulator",
        "DRUG_BINDACT_GENE",
        # composite labels — rare but real
        "agonist,antagonist",
        "agonist,allosteric modulator",
        "blocker,activator",
    }
)

ULTRA_STRICT_GG_REL_TYPES = frozenset(
    {
        "ppi", "PPI",
        "GENE_BINDING_GENE",
        "GENE_PTMOD_GENE",
    }
)

# rel_type → action class. Used only by the action-aware CC bridge.
# Composite labels with mixed direction (e.g. agonist,antagonist) are tagged
# "binding" (i.e. action unknown) rather than guessed.
CG_ACTION_CLASS: dict[str, str] = {
    # inhibitory
    "inhibitor": "inhibitory",
    "antagonist": "inhibitory",
    "blocker": "inhibitory",
    "negative modulator": "inhibitory",
    "inverse agonist": "inhibitory",
    "DRUG_INHIBITION_GENE": "inhibitory",
    "inhibitory allosteric modulator": "inhibitory",
    # activating
    "agonist": "activating",
    "activator": "activating",
    "positive modulator": "activating",
    "partial agonist": "activating",
    "DRUG_ACTIVATION_GENE": "activating",
    "agonist,allosteric modulator": "activating",
    # binding (unspecified or mixed)
    "binds": "binding",
    "targets": "binding",
    "modulator": "binding",
    "allosteric modulator": "binding",
    "DRUG_BINDACT_GENE": "binding",
    "agonist,antagonist": "binding",
    "blocker,activator": "binding",
}

# CC: compound ↔ compound — bridged via shared CG target, NOT from the cc parquet
# -------------------------------------------------------------------------------
# The compound-compound annotation parquet is dominated by:
#   RESEMBLES_CrC          (~46%) — chemical-structure similarity (Tanimoto).
#                                    Same scaffold ≠ same target. High false
#                                    positive rate for phenotypic retrieval.
#   synergistic interaction (~27%) — drug combos. Two compounds working
#                                    *together* often hit *different*
#                                    pathways. Not "shared mechanism".
#   DDI, INTERACTS_CiC      (~27%) — drug-drug interactions — usually
#                                    pharmacokinetic (one inhibits the
#                                    other's CYP metabolism). Not phenotype-
#                                    driving.
#
# None of these directly captures "compounds that produce the same morphology
# because they hit the same target". So in strict mode we discard the cc
# parquet entirely and *derive* CC edges from the strict CG subset:
#
#   positive_cc(a, b) = ∃ gene g  such that  (a, g) ∈ strict_CG  AND
#                                            (b, g) ∈ strict_CG
#
# Every strict-CC positive is a pair of compounds that share a high-confidence
# direct target. Empirically this is the biggest single win in the strict
# variant: morphem-hq smoke goes from CC@5% = 7.02%/6.12% (high/low) under the
# full RESEMBLES-laden cc graph to 11.60%/9.28% under the bridged graph —
# absolute lift of +3 to +5 percentage points, ×1.6-1.7 enrichment. That's
# real signal that was being washed out by structural-resemblance noise.
#
# Implemented in build_cc_annotations_bridged() below; it never reads the
# raw compound-compound parquet in strict mode.


def build_cg_annotations_strict(
    annotations_cg_curated_path: Path,
    inchikey_map_cg_path: Path,
    metadata_path: Path,
    rel_types: frozenset[str] = STRICT_CG_REL_TYPES,
) -> pl.DataFrame:
    """Strict cg edges from the curated raw parquet, filtered by rel_type.

    Output schema matches ``build_cg_annotations``:
    ``(Metadata_JCP2022, target, partner_jcp, target_modality, source="cg")``.
    """
    print(f"[cg-strict] reading {annotations_cg_curated_path}")
    cg = pl.read_parquet(annotations_cg_curated_path)
    required = {"target", "rel_type", "inchikey"}
    missing = required - set(cg.columns)
    if missing:
        raise ValueError(f"cg curated parquet missing columns: {sorted(missing)}")

    n_total = cg.height
    cg = cg.filter(pl.col("rel_type").is_in(list(rel_types)))
    print(
        f"[cg-strict] kept {cg.height:,}/{n_total:,} rows after rel_type filter "
        f"({100*cg.height/n_total:.1f}%)"
    )
    rel_counts = cg.group_by("rel_type").len().sort("len", descending=True)
    print(f"[cg-strict] kept rel_type distribution:\n{rel_counts}")

    print(f"[cg-strict] reading {inchikey_map_cg_path}")
    ik_map = pl.read_csv(inchikey_map_cg_path).select(
        "InChIKey_Connectivity", "Metadata_JCP2022"
    )
    cg_with_jcp = (
        cg.with_columns(
            pl.col("inchikey").str.split("-").list.first().alias("InChIKey_Connectivity")
        )
        .join(ik_map, on="InChIKey_Connectivity", how="inner")
        .select(
            pl.col("Metadata_JCP2022").cast(pl.Utf8),
            pl.col("target").cast(pl.Utf8),
        )
        .filter(pl.col("target").is_not_null() & (pl.col("target") != ""))
        .unique()
    )
    print(
        f"[cg-strict] {cg_with_jcp.height:,} unique (compound_jcp, target_symbol) edges "
        f"covering {cg_with_jcp.select('Metadata_JCP2022').n_unique():,} compounds and "
        f"{cg_with_jcp.select('target').n_unique():,} target symbols"
    )

    md = pl.read_parquet(metadata_path).with_columns(
        pl.col("Metadata_Symbol").cast(pl.Utf8),
        pl.col("Metadata_JCP2022").cast(pl.Utf8),
        pl.col("Metadata_Perturbation_Type").cast(pl.Utf8),
    )
    gene_lookup = (
        md.filter(pl.col("Metadata_Perturbation_Type").is_in(["orf", "crispr"]))
        .filter(pl.col("Metadata_Symbol").is_not_null())
        .select(
            pl.col("Metadata_Symbol").alias("target"),
            pl.col("Metadata_JCP2022").alias("gene_jcp"),
            pl.col("Metadata_Perturbation_Type").alias("target_modality"),
        )
        .unique()
    )

    resolved = cg_with_jcp.join(gene_lookup, on="target", how="inner")
    n_targets = cg_with_jcp.select("target").n_unique()
    n_resolved_targets = resolved.select("target").n_unique()
    n_resolved_compounds = resolved.select("Metadata_JCP2022").n_unique()
    print(
        f"[cg-strict] resolved {n_resolved_targets:,}/{n_targets:,} targets "
        f"({100*n_resolved_targets/max(n_targets,1):.1f}%) — covers "
        f"{n_resolved_compounds:,} compounds"
    )
    print(f"[cg-strict] {resolved.height:,} (compound, target, gene_jcp, modality) rows")

    return resolved.select(
        pl.col("Metadata_JCP2022"),
        pl.col("target"),
        pl.col("gene_jcp").alias("partner_jcp"),
        pl.col("target_modality"),
        pl.lit("cg").alias("source"),
    )


def build_cc_annotations_bridged(cg_strict: pl.DataFrame) -> pl.DataFrame:
    """CC edges = compound pairs that share at least one strict cg target symbol.

    Operates at the gene-symbol level (not gene_jcp): the question is "do these
    two compounds bind the same protein?", regardless of which JUMP reagent we
    happen to have for that gene. Output is symmetric.
    """
    edges_at_symbol = cg_strict.select("Metadata_JCP2022", "target").unique()
    pairs = (
        edges_at_symbol.rename({"Metadata_JCP2022": "jcp_a"})
        .join(
            edges_at_symbol.rename({"Metadata_JCP2022": "jcp_b"}),
            on="target",
            how="inner",
        )
        .filter(pl.col("jcp_a") != pl.col("jcp_b"))
        .select("jcp_a", "jcp_b")
        .unique()
    )
    n_compounds_in_graph = (
        pl.concat([pairs.select(pl.col("jcp_a").alias("jcp")),
                   pairs.select(pl.col("jcp_b").alias("jcp"))])
        .unique().height
    )
    print(
        f"[cc-bridged] {pairs.height:,} symmetric (compound_a, compound_b) pairs "
        f"sharing ≥1 strict target — covers {n_compounds_in_graph:,} compounds"
    )
    return pairs.select(
        pl.col("jcp_a").alias("Metadata_JCP2022"),
        pl.lit(None, dtype=pl.Utf8).alias("target"),
        pl.col("jcp_b").alias("partner_jcp"),
        pl.lit(None, dtype=pl.Utf8).alias("target_modality"),
        pl.lit("cc").alias("source"),
    )


def build_gg_annotations_strict(
    annotations_gg_path: Path,
    metadata_path: Path,
    rel_types: frozenset[str] = STRICT_GG_REL_TYPES,
) -> pl.DataFrame:
    """Strict gg edges: rel_type-filtered then run through ``build_gg_annotations``."""
    gg_full = pl.read_parquet(annotations_gg_path)
    n_total = gg_full.height
    gg_filtered = gg_full.filter(pl.col("rel_type").is_in(list(rel_types)))
    print(
        f"[gg-strict] kept {gg_filtered.height:,}/{n_total:,} rows after rel_type filter "
        f"({100*gg_filtered.height/n_total:.1f}%)"
    )

    # Hand off to the existing build_gg_annotations by writing to a temp file
    # would be wasteful — instead, inline the same logic on the filtered frame.
    canon = (
        gg_filtered.select("target_a", "target_b")
        .filter(pl.col("target_a").is_not_null() & pl.col("target_b").is_not_null())
        .filter(pl.col("target_a") != pl.col("target_b"))
        .with_columns(
            pl.when(pl.col("target_a") < pl.col("target_b"))
            .then(pl.col("target_a")).otherwise(pl.col("target_b")).alias("sym_lo"),
            pl.when(pl.col("target_a") < pl.col("target_b"))
            .then(pl.col("target_b")).otherwise(pl.col("target_a")).alias("sym_hi"),
        )
        .select("sym_lo", "sym_hi")
        .unique()
    )
    print(f"[gg-strict] {canon.height:,} canonical symbol pairs")

    md = pl.read_parquet(metadata_path).with_columns(
        pl.col("Metadata_Symbol").cast(pl.Utf8),
        pl.col("Metadata_JCP2022").cast(pl.Utf8),
        pl.col("Metadata_Perturbation_Type").cast(pl.Utf8),
    )
    gene_lookup = (
        md.filter(pl.col("Metadata_Perturbation_Type").is_in(["orf", "crispr"]))
        .filter(pl.col("Metadata_Symbol").is_not_null())
        .select(
            pl.col("Metadata_Symbol").alias("symbol"),
            pl.col("Metadata_JCP2022").alias("gene_jcp"),
            pl.col("Metadata_Perturbation_Type").alias("modality"),
        )
        .unique()
    )

    pieces = []
    for modality in ("orf", "crispr"):
        sym2jcp = gene_lookup.filter(pl.col("modality") == modality).select(
            "symbol", "gene_jcp"
        )
        joined = (
            canon.join(
                sym2jcp.rename({"symbol": "sym_lo", "gene_jcp": "jcp_lo"}),
                on="sym_lo", how="inner",
            )
            .join(
                sym2jcp.rename({"symbol": "sym_hi", "gene_jcp": "jcp_hi"}),
                on="sym_hi", how="inner",
            )
            .filter(pl.col("jcp_lo") != pl.col("jcp_hi"))
            .select("jcp_lo", "jcp_hi")
            .unique()
        )
        print(f"[gg-strict/{modality}] {joined.height:,} JCP pairs after symbol→JCP explode")
        symmetrised = pl.concat(
            [joined,
             joined.rename({"jcp_lo": "jcp_hi", "jcp_hi": "jcp_lo"}).select("jcp_lo", "jcp_hi")]
        ).unique()
        pieces.append(symmetrised.with_columns(pl.lit(modality).alias("modality_col")))

    out = pl.concat(pieces, how="vertical")
    print(f"[gg-strict] {out.height:,} total gg JCP pairs (symmetrised, all modalities)")
    return out.select(
        pl.col("jcp_lo").alias("Metadata_JCP2022"),
        pl.lit(None, dtype=pl.Utf8).alias("target"),
        pl.col("jcp_hi").alias("partner_jcp"),
        pl.col("modality_col").alias("target_modality"),
        pl.lit("gg").alias("source"),
    )


def build_cg_annotations_ultra_strict(
    annotations_cg_curated_path: Path,
    inchikey_map_cg_path: Path,
    metadata_path: Path,
) -> pl.DataFrame:
    """Ultra-strict cg edges, tagged with ``action_class``.

    Same shape as ``build_cg_annotations_strict`` plus a ``rel_type`` column
    and an ``action_class`` column ∈ {inhibitory, activating, binding}. The
    ``rel_type`` and ``action_class`` columns are consumed by the action-aware
    CC bridge and stripped before the final output write.
    """
    print(f"[cg-ultra] reading {annotations_cg_curated_path}")
    cg = pl.read_parquet(annotations_cg_curated_path)
    required = {"target", "rel_type", "inchikey"}
    missing = required - set(cg.columns)
    if missing:
        raise ValueError(f"cg curated parquet missing columns: {sorted(missing)}")

    n_total = cg.height
    cg = cg.filter(pl.col("rel_type").is_in(list(ULTRA_STRICT_CG_REL_TYPES)))
    print(
        f"[cg-ultra] kept {cg.height:,}/{n_total:,} rows after rel_type filter "
        f"({100*cg.height/n_total:.1f}%)"
    )
    rel_counts = cg.group_by("rel_type").len().sort("len", descending=True)
    print(f"[cg-ultra] kept rel_type distribution:\n{rel_counts}")

    # tag action class
    action_map = pl.DataFrame(
        {
            "rel_type": list(CG_ACTION_CLASS.keys()),
            "action_class": list(CG_ACTION_CLASS.values()),
        }
    )
    cg = cg.join(action_map, on="rel_type", how="left")
    n_unmapped = cg.filter(pl.col("action_class").is_null()).height
    if n_unmapped:
        unmapped = cg.filter(pl.col("action_class").is_null())["rel_type"].unique()
        raise ValueError(
            f"[cg-ultra] {n_unmapped:,} rows have rel_types missing from "
            f"CG_ACTION_CLASS: {sorted(unmapped)}"
        )
    print(
        "[cg-ultra] action_class distribution:\n"
        f"{cg.group_by('action_class').len().sort('len', descending=True)}"
    )

    print(f"[cg-ultra] reading {inchikey_map_cg_path}")
    ik_map = pl.read_csv(inchikey_map_cg_path).select(
        "InChIKey_Connectivity", "Metadata_JCP2022"
    )
    cg_with_jcp = (
        cg.with_columns(
            pl.col("inchikey").str.split("-").list.first().alias("InChIKey_Connectivity")
        )
        .join(ik_map, on="InChIKey_Connectivity", how="inner")
        .select(
            pl.col("Metadata_JCP2022").cast(pl.Utf8),
            pl.col("target").cast(pl.Utf8),
            pl.col("rel_type"),
            pl.col("action_class"),
        )
        .filter(pl.col("target").is_not_null() & (pl.col("target") != ""))
        .unique()
    )
    print(
        f"[cg-ultra] {cg_with_jcp.height:,} unique "
        f"(compound_jcp, target, rel_type, action_class) edges covering "
        f"{cg_with_jcp.select('Metadata_JCP2022').n_unique():,} compounds and "
        f"{cg_with_jcp.select('target').n_unique():,} target symbols"
    )

    md = pl.read_parquet(metadata_path).with_columns(
        pl.col("Metadata_Symbol").cast(pl.Utf8),
        pl.col("Metadata_JCP2022").cast(pl.Utf8),
        pl.col("Metadata_Perturbation_Type").cast(pl.Utf8),
    )
    gene_lookup = (
        md.filter(pl.col("Metadata_Perturbation_Type").is_in(["orf", "crispr"]))
        .filter(pl.col("Metadata_Symbol").is_not_null())
        .select(
            pl.col("Metadata_Symbol").alias("target"),
            pl.col("Metadata_JCP2022").alias("gene_jcp"),
            pl.col("Metadata_Perturbation_Type").alias("target_modality"),
        )
        .unique()
    )

    resolved = cg_with_jcp.join(gene_lookup, on="target", how="inner")
    n_targets = cg_with_jcp.select("target").n_unique()
    n_resolved_targets = resolved.select("target").n_unique()
    n_resolved_compounds = resolved.select("Metadata_JCP2022").n_unique()
    print(
        f"[cg-ultra] resolved {n_resolved_targets:,}/{n_targets:,} targets "
        f"({100*n_resolved_targets/max(n_targets,1):.1f}%) — covers "
        f"{n_resolved_compounds:,} compounds"
    )
    print(f"[cg-ultra] {resolved.height:,} (compound, target, gene_jcp, modality) rows")

    return resolved.select(
        pl.col("Metadata_JCP2022"),
        pl.col("target"),
        pl.col("gene_jcp").alias("partner_jcp"),
        pl.col("target_modality"),
        pl.col("rel_type"),
        pl.col("action_class"),
        pl.lit("cg").alias("source"),
    )


def build_cc_annotations_bridged_action_aware(
    cg_ultra: pl.DataFrame,
) -> pl.DataFrame:
    """CC edges = compound pairs sharing a gene with the same action class.

    Stricter than ``build_cc_annotations_bridged``: a pair (A,B) is positive
    iff there is some gene g such that (A,g,c) and (B,g,c) are both in the
    ultra-strict CG subset for the same action class c. Cross-class pairs
    (e.g. agonist of g vs antagonist of g) are dropped.

    Operates at the gene-symbol level — same as the strict bridge. Output is
    symmetric.
    """
    edges = (
        cg_ultra.select("Metadata_JCP2022", "target", "action_class")
        .unique()
    )
    pairs = (
        edges.rename({"Metadata_JCP2022": "jcp_a"})
        .join(
            edges.rename({"Metadata_JCP2022": "jcp_b"}),
            on=["target", "action_class"],
            how="inner",
        )
        .filter(pl.col("jcp_a") != pl.col("jcp_b"))
        .select("jcp_a", "jcp_b")
        .unique()
    )
    n_compounds_in_graph = (
        pl.concat(
            [
                pairs.select(pl.col("jcp_a").alias("jcp")),
                pairs.select(pl.col("jcp_b").alias("jcp")),
            ]
        )
        .unique()
        .height
    )
    print(
        f"[cc-ultra] {pairs.height:,} symmetric (a, b) pairs sharing ≥1 "
        f"target+action_class — covers {n_compounds_in_graph:,} compounds"
    )
    return pairs.select(
        pl.col("jcp_a").alias("Metadata_JCP2022"),
        pl.lit(None, dtype=pl.Utf8).alias("target"),
        pl.col("jcp_b").alias("partner_jcp"),
        pl.lit(None, dtype=pl.Utf8).alias("target_modality"),
        pl.lit("cc").alias("source"),
    )


def _read_splits_file(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix in (".csv",):
        return pl.read_csv(path)
    if suffix in (".tsv", ".txt"):
        return pl.read_csv(path, separator="\t")
    raise ValueError(
        f"unrecognised splits file suffix: {suffix} (expected .parquet/.csv/.tsv)"
    )


def _detect_split_col(df: pl.DataFrame) -> str:
    for c in ("split", "Split", "fold", "subset"):
        if c in df.columns:
            return c
    raise ValueError(
        f"could not auto-detect split column in {df.columns}. "
        "Pass --splits-split-col explicitly."
    )


def _detect_key_col(df: pl.DataFrame) -> str:
    for c in CG_KEY_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"could not auto-detect key column in {df.columns}. "
        "Pass --splits-key-col explicitly."
    )


def build_splits(
    splits_path: Path,
    inchikey_map_path: Path,
    key_col: str | None,
    split_col: str | None,
) -> pl.DataFrame:
    """Map an upstream MOTIVE split file to ``Metadata_JCP2022 → split``.

    The upstream file may be keyed by JCP2022 (passes through), by InChIKey
    (mapped via connectivity), or by compound_name (currently unsupported —
    raises with an actionable error).
    """
    print(f"[splits] reading {splits_path}")
    raw = _read_splits_file(splits_path)
    print(f"[splits] columns: {raw.columns}")

    key_col = key_col or _detect_key_col(raw)
    split_col = split_col or _detect_split_col(raw)
    print(f"[splits] key_col={key_col}  split_col={split_col}")

    raw = raw.select(key_col, split_col).rename({split_col: "split"})

    if key_col in ("Metadata_JCP2022", "JCP2022"):
        return raw.rename({key_col: "Metadata_JCP2022"}).unique().select(
            "Metadata_JCP2022", "split"
        )

    if key_col in ("Metadata_InChIKey", "InChIKey", "inchikey", "compound_inchikey"):
        ik_map = pl.read_csv(inchikey_map_path).select(
            "InChIKey_Connectivity", "Metadata_JCP2022"
        )
        mapped = (
            raw.with_columns(
                pl.col(key_col)
                .str.split("-")
                .list.first()
                .alias("InChIKey_Connectivity")
            )
            .join(ik_map, on="InChIKey_Connectivity", how="inner")
            .select("Metadata_JCP2022", "split")
            .unique()
        )
        print(
            f"[splits] mapped {mapped.height:,} JCPs from "
            f"{raw.height:,} inchikey rows"
        )
        return mapped

    raise ValueError(
        f"key column '{key_col}' not supported. The MOTIVE split file is "
        "expected to be keyed by Metadata_JCP2022 or InChIKey. If yours is "
        "keyed by compound name, pre-map it to InChIKey first."
    )


def _refuse_overwrite(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(
            f"refusing to overwrite existing {path}. Pass --force to opt in."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("metadata/metadata_dataset_filtered_4reps.parquet"),
    )
    parser.add_argument(
        "--annotations-cc",
        type=Path,
        default=Path(
            "data/annotations/"
            "annotations_compound_compound.parquet"
        ),
    )
    parser.add_argument(
        "--annotations-gg",
        type=Path,
        default=Path(
            "data/annotations/"
            "annotations_gene_gene.parquet"
        ),
    )
    parser.add_argument(
        "--inchikey-map",
        type=Path,
        default=Path(
            "metadata/inchikey_to_jcp2022_mapping_compound_compound.csv"
        ),
    )
    parser.add_argument(
        "--annotations-cg-curated",
        type=Path,
        default=Path(
            "data/annotations/"
            "annotations_compound_gene_curated.parquet"
        ),
        help="Raw curated compound-gene parquet (used by --mode strict).",
    )
    parser.add_argument(
        "--inchikey-map-cg",
        type=Path,
        default=Path(
            "data/annotations/"
            "inchikey_to_jcp2022_mapping_compound_gene.csv"
        ),
        help="InChIKey→JCP map for the cg path (used by --mode strict).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "strict", "ultra_strict"],
        default="full",
        help=(
            "full = pool all rel_types (writes motive_annotations.parquet). "
            "strict = direct-mechanism only (writes motive_annotations_strict.parquet). "
            "ultra_strict = strict + complete direct-binding allowlist + "
            "action-aware CC bridge + tighter GG (writes "
            "motive_annotations_ultra_strict.parquet). See module-level "
            "comments above ULTRA_STRICT_CG_REL_TYPES for details."
        ),
    )
    parser.add_argument(
        "--motive-splits-path",
        type=Path,
        default=None,
        help=(
            "Local copy of the published MOTIVE split file (parquet/csv/tsv). "
            "Required — there is no auto-download."
        ),
    )
    parser.add_argument(
        "--splits-key-col",
        type=str,
        default=None,
        help="Column in splits file holding the join key (auto-detected if unset).",
    )
    parser.add_argument(
        "--splits-split-col",
        type=str,
        default=None,
        help="Column in splits file holding the split label (auto-detected).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("metadata"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-splits",
        action="store_true",
        help="Only write motive_annotations.parquet (use when no splits file yet).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotations_filename = {
        "full": "motive_annotations.parquet",
        "strict": "motive_annotations_strict.parquet",
        "ultra_strict": "motive_annotations_ultra_strict.parquet",
    }[args.mode]
    annotations_out = args.output_dir / annotations_filename
    splits_out = args.output_dir / "motive_eval_compounds.parquet"

    _refuse_overwrite(annotations_out, args.force)
    if not args.skip_splits:
        _refuse_overwrite(splits_out, args.force)
        if args.motive_splits_path is None:
            print(
                "ERROR: --motive-splits-path is required (or pass --skip-splits "
                "to write annotations only). Download the split file from the "
                "upstream MOTIVE repo and point this flag at the local copy.",
                file=sys.stderr,
            )
            sys.exit(2)

    if args.mode == "full":
        cg = build_cg_annotations(args.metadata)
        cc = build_cc_annotations(args.annotations_cc, args.inchikey_map)
        gg = build_gg_annotations(args.annotations_gg, args.metadata)
    elif args.mode == "strict":
        cg = build_cg_annotations_strict(
            args.annotations_cg_curated, args.inchikey_map_cg, args.metadata,
        )
        cc = build_cc_annotations_bridged(cg)
        gg = build_gg_annotations_strict(args.annotations_gg, args.metadata)
    else:  # ultra_strict
        cg_with_action = build_cg_annotations_ultra_strict(
            args.annotations_cg_curated, args.inchikey_map_cg, args.metadata,
        )
        cc = build_cc_annotations_bridged_action_aware(cg_with_action)
        gg = build_gg_annotations_strict(
            args.annotations_gg, args.metadata,
            rel_types=ULTRA_STRICT_GG_REL_TYPES,
        )
        # strip rel_type / action_class columns from cg before final concat
        # so the output schema stays identical across modes
        cg = cg_with_action.drop("rel_type", "action_class")
    annotations = pl.concat([cg, cc, gg], how="vertical_relaxed")
    print(f"[annotations] mode={args.mode}  total rows: {annotations.height:,}")
    print(annotations.group_by("source").len())
    annotations.write_parquet(annotations_out)
    print(f"[write] {annotations_out}")

    if not args.skip_splits:
        splits = build_splits(
            args.motive_splits_path,
            args.inchikey_map,
            args.splits_key_col,
            args.splits_split_col,
        )
        print(f"[splits] total rows: {splits.height:,}")
        print(splits.group_by("split").len())
        splits.write_parquet(splits_out)
        print(f"[write] {splits_out}")


if __name__ == "__main__":
    main()
