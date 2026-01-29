#!/usr/bin/env python
"""Compare compound and target overlap between metadata_dataset and refchemdb."""

import polars as pl
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3

metadata_dir = Path("/home/jfredinh/projects/JUMP_core/metadata")
output_dir = Path("/home/jfredinh/projects/JUMP_core/metadata/dataset_overlaps")
output_dir.mkdir(exist_ok=True)

# Load data
df_metadata = pl.read_parquet(metadata_dir / "metadata_dataset.parquet")
df_refchemdb = pl.read_parquet(metadata_dir / "refchemdb_conf_jump_matched.parquet")

# Get unique identifiers
metadata_JCPs = set(df_metadata["Metadata_JCP2022"].drop_nulls().unique().to_list())

df_compound_subset = df_metadata.filter(~df_metadata["Metadata_Perturbation_Type"].is_in(["orf", "crispr"]))

metadata_compounds = set(df_compound_subset["Metadata_JCP2022"].drop_nulls().unique().to_list())
refchemdb_compounds = set(df_refchemdb["Metadata_JCP2022"].drop_nulls().unique().to_list())
metadata_symbols = set(df_metadata["Metadata_Symbol"].drop_nulls().unique().to_list())
refchemdb_targets = set(df_refchemdb["target"].drop_nulls().unique().to_list())

# Compound overlap
cmpd_overlap = metadata_compounds & refchemdb_compounds
print("=== COMPOUND OVERLAP (Metadata_JCP2022) ===")
print(f"Metadata:  {len(metadata_compounds):,}")
print(f"RefChemDB: {len(refchemdb_compounds):,}")
print(f"Overlap:   {len(cmpd_overlap):,} ({len(cmpd_overlap)/len(refchemdb_compounds)*100:.1f}% of RefChemDB)")

# Symbol/Target overlap
symbol_overlap = metadata_symbols & refchemdb_targets
print("\n=== TARGET OVERLAP (Metadata_Symbol vs target) ===")
print(f"Metadata symbols:  {len(metadata_symbols):,}")
print(f"RefChemDB targets: {len(refchemdb_targets):,}")
print(f"Overlap:           {len(symbol_overlap):,} ({len(symbol_overlap)/len(refchemdb_targets)*100:.1f}% of RefChemDB targets)")

# Venn diagrams
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

venn2([metadata_compounds, refchemdb_compounds], set_labels=("Metadata", "RefChemDB"), ax=axes[0])
axes[0].set_title("Compound Overlap (Metadata_JCP2022)")

venn2([metadata_symbols, refchemdb_targets], set_labels=("Metadata Symbols", "RefChemDB Targets"), ax=axes[1])
axes[1].set_title("Target Overlap (Metadata_Symbol vs target)")

plt.tight_layout()
plt.savefig(output_dir / "compound_target_overlap_venn.png", dpi=150)
print(f"\nSaved: {output_dir / 'compound_target_overlap_venn.png'}")



# === COMPOUNDS PER SOURCE VENN ===
sources = df_metadata["Metadata_Source"].cast(pl.Utf8).unique().sort().to_list()
jcpids_per_source = {
    src: set(df_metadata.filter(pl.col("Metadata_Source").cast(pl.Utf8) == src)["Metadata_JCP2022"].drop_nulls().unique().to_list())
    for src in sources
}


print("\n=== JCPIDs PER SOURCE ===")
for src, jcpids in sorted(jcpids_per_source.items()):
    print(f"{src}: {len(jcpids):,}")


# Compare unique compounds in "source_2", "source_6", "source_8"

compound_source_labels = ["source_2", "source_6", "source_8"]
compound_sources = [jcpids_per_source[s] for s in compound_source_labels if s in jcpids_per_source]

fig, ax = plt.subplots(figsize=(8, 6))
v = venn3(compound_sources, set_labels=compound_source_labels, ax=ax, alpha=0.6)
# Set colors only for patches that exist (not None)
colors = ["red", "yellow", "blue"]
for i, patch_id in enumerate(["100", "010", "001"]):
    patch = v.get_patch_by_id(patch_id)
    if patch is not None:
        patch.set_color(colors[i])
ax.set_title("Compound Overlap by Source")
ax.annotate("(source_8 is a subset of source_2 ∩ source_6)", xy=(0.5, -0.1), xycoords="axes fraction", ha="center", fontsize=9, style="italic")
plt.tight_layout()
plt.savefig(output_dir / "compound_source_overlap_venn.png", dpi=150)
print(f"\nSaved: {output_dir / 'compound_source_overlap_venn.png'}")

# === SUPERVENN DIAGRAM ===
from supervenn import supervenn

plt.figure(figsize=(14, 6))
supervenn(
    compound_sources,
    compound_source_labels,
    side_plots="right",
    chunks_ordering="size",
    widths_minmax_ratio=0.05,
    min_width_for_annotation=1,   # show count if chunk > 50
    sets_ordering="minimize gaps",
    rotate_col_annotations=True,   # rotate numbers for readability
    col_annotations_area_height=1.5,
)
plt.suptitle("Compound Overlap by Source", fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "compound_source_overlap_supervenn.png", dpi=150)
print(f"Saved: {output_dir / 'compound_source_overlap_supervenn.png'}")





compound_source_labels = ["source_2", "source_6", "source_8", "source_7"]
compound_sources = [jcpids_per_source[s] for s in compound_source_labels if s in jcpids_per_source]


# === SUPERVENN DIAGRAM ===
from supervenn import supervenn

plt.figure(figsize=(14, 6))
supervenn(
    compound_sources,
    compound_source_labels,
    side_plots="right",
    chunks_ordering="size",
    widths_minmax_ratio=0.05,
    min_width_for_annotation=1,
    sets_ordering="minimize gaps",
    rotate_col_annotations=True,
    col_annotations_area_height=1.5,
)
plt.suptitle("Compound Overlap by Source (4 sources)", fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "compound_source_overlap_supervenn_4.png", dpi=150)
print(f"Saved: {output_dir / 'compound_source_overlap_supervenn_4.png'}")

# === TARGET ANNOTATIONS PER SOURCE (Compounds only) ===
jcp_to_targets = (
    df_refchemdb.select(["Metadata_JCP2022", "target"])
    .drop_nulls()
    .unique()
)

targets_per_source = {}
for src in compound_source_labels:
    if src in jcpids_per_source:
        src_jcps = list(jcpids_per_source[src])
        src_targets = set(
            jcp_to_targets.filter(pl.col("Metadata_JCP2022").is_in(src_jcps))["target"].to_list()
        )
        targets_per_source[src] = src_targets

print("\n=== TARGETS PER SOURCE (Compounds via RefChemDB) ===")
for src, targets in sorted(targets_per_source.items()):
    print(f"{src}: {len(targets):,} targets")

target_source_labels = list(targets_per_source.keys())
target_sources = [targets_per_source[s] for s in target_source_labels]

plt.figure(figsize=(14, 6))
supervenn(
    target_sources,
    target_source_labels,
    side_plots="right",
    chunks_ordering="size",
    widths_minmax_ratio=0.05,
    min_width_for_annotation=1,
    sets_ordering="minimize gaps",
    rotate_col_annotations=True,
    col_annotations_area_height=1.5,
)
plt.suptitle("Target Overlap by Source (via RefChemDB)", fontsize=14)
plt.tight_layout()
plt.savefig(output_dir / "target_source_overlap_supervenn.png", dpi=150)
print(f"Saved: {output_dir / 'target_source_overlap_supervenn.png'}")

# === TARGET ANNOTATIONS - ALL SOURCES (Compounds + ORF + CRISPR) ===
all_targets_per_source = {}

# Compound sources: map through refchemdb
for src in ["source_2", "source_6", "source_8", "source_7"]:
    if src in jcpids_per_source:
        src_jcps = list(jcpids_per_source[src])
        src_targets = set(
            jcp_to_targets.filter(pl.col("Metadata_JCP2022").is_in(src_jcps))["target"].to_list()
        )
        all_targets_per_source[f"{src} (compound)"] = src_targets

# ORF/CRISPR sources: use Metadata_Symbol directly
for src, pert_type in [("source_4", "ORF"), ("source_13", "CRISPR")]:
    src_symbols = set(
        df_metadata.filter(pl.col("Metadata_Source").cast(pl.Utf8) == src)["Metadata_Symbol"]
        .drop_nulls()
        .unique()
        .to_list()
    )
    all_targets_per_source[f"{src} ({pert_type})"] = src_symbols

print("\n=== TARGETS PER SOURCE (All modalities) ===")
for src, targets in sorted(all_targets_per_source.items()):
    print(f"{src}: {len(targets):,} targets")

all_target_labels = list(all_targets_per_source.keys())
all_target_sources = [all_targets_per_source[s] for s in all_target_labels]

plt.figure(figsize=(16, 8))
supervenn(
    all_target_sources,
    all_target_labels,
    side_plots="right",
    chunks_ordering="size",
    widths_minmax_ratio=0.05,
    min_width_for_annotation=1,
    sets_ordering="minimize gaps",
    rotate_col_annotations=True,
    col_annotations_area_height=1.5,
)
plt.suptitle("Target Overlap - All Sources (Compounds via RefChemDB, ORF/CRISPR via Metadata_Symbol)", fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / "target_all_sources_overlap_supervenn.png", dpi=150)
print(f"Saved: {output_dir / 'target_all_sources_overlap_supervenn.png'}")

# === TARGET ANNOTATIONS - ALL SOURCES (>=4 replicates filter) ===
MIN_REPLICATES = 4

def get_jcpids_with_min_replicates(df, sources, min_reps):
    """Get JCPIDs that have >= min_reps replicates across the given sources."""
    counts = (
        df.filter(pl.col("Metadata_Source").cast(pl.Utf8).is_in(sources))
        .group_by("Metadata_JCP2022")
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") >= min_reps)
    )
    return set(counts["Metadata_JCP2022"].drop_nulls().to_list())

def get_symbols_with_min_replicates(df, sources, min_reps):
    """Get Metadata_Symbols that have >= min_reps replicates across the given sources."""
    counts = (
        df.filter(pl.col("Metadata_Source").cast(pl.Utf8).is_in(sources))
        .group_by("Metadata_Symbol")
        .agg(pl.len().alias("count"))
        .filter(pl.col("count") >= min_reps)
    )
    return set(counts["Metadata_Symbol"].drop_nulls().to_list())

filtered_targets_per_source = {}

# Compound sources grouped: source_2, source_6, source_8 together
cpg_sources = ["source_2", "source_6", "source_8"]
cpg_jcps = get_jcpids_with_min_replicates(df_metadata, cpg_sources, MIN_REPLICATES)
cpg_targets = set(
    jcp_to_targets.filter(pl.col("Metadata_JCP2022").is_in(list(cpg_jcps)))["target"].to_list()
)
filtered_targets_per_source["source_2/6/8 (compound)"] = cpg_targets

# source_7 separately
s7_jcps = get_jcpids_with_min_replicates(df_metadata, ["source_7"], MIN_REPLICATES)
s7_targets = set(
    jcp_to_targets.filter(pl.col("Metadata_JCP2022").is_in(list(s7_jcps)))["target"].to_list()
)
filtered_targets_per_source["source_7 (compound)"] = s7_targets

# ORF/CRISPR sources: filter Symbols by replicate count (each separately)
for src, pert_type in [("source_4", "ORF"), ("source_13", "CRISPR")]:
    filtered_symbols = get_symbols_with_min_replicates(df_metadata, [src], MIN_REPLICATES)
    filtered_targets_per_source[f"{src} ({pert_type})"] = filtered_symbols

print(f"\n=== TARGETS PER SOURCE (>={MIN_REPLICATES} replicates) ===")
for src, targets in sorted(filtered_targets_per_source.items()):
    print(f"{src}: {len(targets):,} targets")

filtered_target_labels = list(filtered_targets_per_source.keys())
filtered_target_sources = [filtered_targets_per_source[s] for s in filtered_target_labels]

plt.figure(figsize=(16, 8))
supervenn(
    filtered_target_sources,
    filtered_target_labels,
    side_plots="right",
    chunks_ordering="size",
    widths_minmax_ratio=0.05,
    min_width_for_annotation=1,
    sets_ordering="minimize gaps",
    rotate_col_annotations=True,
    col_annotations_area_height=1.5,
)
plt.suptitle(f"Target Overlap - All Sources (>={MIN_REPLICATES} replicates per perturbation)", fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / "target_all_sources_filtered_supervenn.png", dpi=150)
print(f"Saved: {output_dir / 'target_all_sources_filtered_supervenn.png'}")

# === HEATMAP OF TARGET INTERSECTIONS ===
import numpy as np

n_sets = len(filtered_target_labels)
intersection_matrix = np.zeros((n_sets, n_sets), dtype=int)

for i, label_i in enumerate(filtered_target_labels):
    for j, label_j in enumerate(filtered_target_labels):
        intersection_matrix[i, j] = len(filtered_targets_per_source[label_i] & filtered_targets_per_source[label_j])

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(intersection_matrix, cmap="YlOrRd")

ax.set_xticks(range(n_sets))
ax.set_yticks(range(n_sets))
ax.set_xticklabels(filtered_target_labels, rotation=45, ha="right")
ax.set_yticklabels(filtered_target_labels)

# Add text annotations
for i in range(n_sets):
    for j in range(n_sets):
        text_color = "white" if intersection_matrix[i, j] > intersection_matrix.max() / 2 else "black"
        ax.text(j, i, f"{intersection_matrix[i, j]:,}", ha="center", va="center", color=text_color, fontsize=10)

plt.colorbar(im, ax=ax, label="Number of shared targets")
ax.set_title(f"Target Intersection Heatmap (>={MIN_REPLICATES} replicates)")
plt.tight_layout()
plt.savefig(output_dir / "target_intersection_heatmap.png", dpi=150)
print(f"Saved: {output_dir / 'target_intersection_heatmap.png'}")

# === JCP ID OVERLAP: source_2/6/8 vs source_7 (filtered) ===
fig, ax = plt.subplots(figsize=(8, 6))
venn2(
    [cpg_jcps, s7_jcps],
    set_labels=("source_2/6/8", "source_7"),
    ax=ax,
    alpha=0.6
)
ax.set_title(f"JCP ID Overlap (>={MIN_REPLICATES} replicates)")
plt.tight_layout()
plt.savefig(output_dir / "jcpid_overlap_cpg_vs_s7_venn.png", dpi=150)
print(f"Saved: {output_dir / 'jcpid_overlap_cpg_vs_s7_venn.png'}")

print(f"\n=== JCP ID OVERLAP (>={MIN_REPLICATES} replicates) ===")
print(f"source_2/6/8: {len(cpg_jcps):,} JCPIDs")
print(f"source_7:     {len(s7_jcps):,} JCPIDs")
print(f"Overlap:      {len(cpg_jcps & s7_jcps):,} JCPIDs")

# Save JCP IDs to parquet
pl.DataFrame({"Metadata_JCP2022": sorted(cpg_jcps)}).write_parquet(output_dir / "jcpids_source_2_6_8_4reps.parquet")
pl.DataFrame({"Metadata_JCP2022": sorted(s7_jcps)}).write_parquet(output_dir / "jcpids_source_7_4reps.parquet")
print(f"Saved: {output_dir / 'jcpids_source_2_6_8_4reps.parquet'}")
print(f"Saved: {output_dir / 'jcpids_source_7_4reps.parquet'}")

# === TARGETS WITH >= 3 DIFFERENT JCPIDs ===
MIN_JCPIDS_PER_TARGET = 3

def get_targets_with_min_jcpids(jcpids, jcp_to_targets_df, min_jcpids):
    """Get targets that have >= min_jcpids different JCPIDs hitting them."""
    # Filter to only the JCPIDs in our set
    filtered = jcp_to_targets_df.filter(pl.col("Metadata_JCP2022").is_in(list(jcpids)))
    # Count unique JCPIDs per target
    target_counts = (
        filtered.group_by("target")
        .agg(pl.n_unique("Metadata_JCP2022").alias("n_jcpids"))
        .filter(pl.col("n_jcpids") >= min_jcpids)
    )
    return set(target_counts["target"].to_list()), target_counts

cpg_targets_3plus, cpg_target_counts = get_targets_with_min_jcpids(cpg_jcps, jcp_to_targets, MIN_JCPIDS_PER_TARGET)
s7_targets_3plus, s7_target_counts = get_targets_with_min_jcpids(s7_jcps, jcp_to_targets, MIN_JCPIDS_PER_TARGET)

print(f"\n=== TARGETS WITH >={MIN_JCPIDS_PER_TARGET} JCPIDs (each with >={MIN_REPLICATES} replicates) ===")
print(f"source_2/6/8: {len(cpg_targets_3plus):,} targets (out of {len(cpg_targets):,} total)")
print(f"source_7:     {len(s7_targets_3plus):,} targets (out of {len(s7_targets):,} total)")
print(f"Overlap:      {len(cpg_targets_3plus & s7_targets_3plus):,} targets")

# Save targets with >=3 JCPIDs to parquet
cpg_target_counts.write_parquet(output_dir / "targets_source_2_6_8_3plus_jcpids.parquet")
s7_target_counts.write_parquet(output_dir / "targets_source_7_3plus_jcpids.parquet")
print(f"Saved: {output_dir / 'targets_source_2_6_8_3plus_jcpids.parquet'}")
print(f"Saved: {output_dir / 'targets_source_7_3plus_jcpids.parquet'}")

# Venn diagram of these filtered targets
fig, ax = plt.subplots(figsize=(8, 6))
venn2(
    [cpg_targets_3plus, s7_targets_3plus],
    set_labels=("source_2/6/8", "source_7"),
    ax=ax,
    alpha=0.6
)
ax.set_title(f"Target Overlap (>={MIN_JCPIDS_PER_TARGET} JCPIDs per target, >={MIN_REPLICATES} reps)")
plt.tight_layout()
plt.savefig(output_dir / "target_overlap_3plus_jcpids_venn.png", dpi=150)
print(f"Saved: {output_dir / 'target_overlap_3plus_jcpids_venn.png'}")
