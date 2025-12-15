import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import warnings
    from concurrent.futures import ThreadPoolExecutor
    from functools import lru_cache
    from itertools import islice
    from math import pow
    from pathlib import Path, PosixPath

    import duckdb
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from polars import selectors as cs
    from trommel.core import basic_cleanup
    from umap import UMAP
    import pandas as pd

    import os
    return np, pd, plt


@app.cell
def _():

    from copairs import map
    from copairs.matching import assign_reference_index
    from copairs.map.average_precision import p_values
    return map, p_values


@app.cell
def _(pd):
    metadata_target2  = pd.read_csv("analysis/feature_similarity/input/JUMP-Target-2_compound_metadata.tsv", sep="\t")
    plate_map = pd.read_csv("analysis/feature_similarity/input/JUMP-Target-2_compound_platemap.tsv", sep="\t")
    metadata_target2 = metadata_target2.drop_duplicates()
    return metadata_target2, plate_map


@app.cell
def _(metadata_target2, plate_map):
    plate_map_annotated = plate_map.merge(metadata_target2, on="broad_sample")

    plate_map_annotated = plate_map_annotated.rename(columns={col: f'Metadata_{col}' for col in plate_map_annotated.columns})
    return (plate_map_annotated,)


@app.cell
def _(plate_map_annotated):
    plate_map_annotated
    return


@app.cell
def _(pd):
    # clean_zarr = pd.read_parquet("analysis/feature_similarity/output/jpegxl_lossy_effort_3.parquet")
    # clean_zarr = clean_zarr.drop(columns="nr")

    clean_zarr = pd.read_parquet("analysis/feature_similarity/output/zstd.parquet")
    clean_zarr = clean_zarr.drop(columns="nr")

    return (clean_zarr,)


@app.cell
def _(clean_zarr, plate_map_annotated):
    df = clean_zarr.merge(plate_map_annotated, left_on="Metadata_Well", right_on="Metadata_well_position", how="left")
    return (df,)


@app.cell
def _(clean_zarr):
    clean_zarr[["Metadata_Plate","Metadata_Well"]].value_counts()
    return


@app.cell
def _(df):
    df.Metadata_control_type.value_counts()
    return


@app.cell
def _(df):
    df.Metadata_Plate.value_counts()
    return


@app.cell
def _(df):
    df.shape
    return


@app.cell
def _():
    # positive pairs are replicates of the same treatment
    pos_sameby = ["Metadata_pert_iname"]
    pos_diffby = []

    neg_sameby = ["Metadata_Plate"]
    # negative pairs are replicates of different treatments
    neg_diffby = ["Metadata_pert_iname", "Metadata_negcon"]
    return neg_diffby, neg_sameby, pos_diffby, pos_sameby


@app.cell
def _(df):
    df[["Metadata_Source", 
                  "Metadata_Batch",
                  "Metadata_Plate",
                  "Metadata_Site",
                "Metadata_pert_iname", 
                "Metadata_pert_type", 
               "Metadata_target_list",
               "Metadata_control_type",
               "Metadata_Well"]].isna().mean()
    return


@app.cell
def _(df):
    df["Metadata_target_list"] = df["Metadata_target_list"].fillna("unknown")
    df["Metadata_control_type"] = df["Metadata_control_type"].fillna("trt")
    return


@app.cell
def _(df):
    df_median = df.groupby(["Metadata_Source", 
                  "Metadata_Batch",
                  "Metadata_Plate",
                  "Metadata_Site",
                "Metadata_pert_iname", 
                "Metadata_pert_type", 
               "Metadata_target_list",
               "Metadata_control_type",
               "Metadata_Well"], as_index=False)[df.filter(regex="^(?!Metadata)").columns].median()#.reset_index()
    return (df_median,)


@app.cell
def _(df_median):
    df_median["Metadata_control_type"].value_counts()
    return


@app.cell
def _(df_median):
    df_median["Metadata_negcon"] = df_median["Metadata_control_type"] == "negcon"
    return


@app.cell
def _(df_median):
    df_median["Metadata_negcon"].value_counts()
    return


@app.cell
def _(df_median, map, neg_diffby, neg_sameby, pos_diffby, pos_sameby):
    metadata = df_median.filter(regex="^Metadata")
    profiles = df_median.filter(regex="^(?!Metadata)").values

    activity_ap = map.average_precision(
        metadata, profiles, pos_sameby, pos_diffby, neg_sameby, neg_diffby
    )
    activity_ap = activity_ap.query("Metadata_pert_iname != 'DMSO'")  # remove DMSO
    #activity_ap.to_csv("data/activity_ap.csv", index=False)
    activity_ap
    return (activity_ap,)


@app.cell
def _(activity_ap, np, p_values):
    # Calculate p-values using the same null_size and seed as mean_average_precision
    activity_ap["p_value"] = p_values(activity_ap, null_size=10_000, seed=0)
    activity_ap["-log10(p-value)"] = -activity_ap["p_value"].apply(np.log10)
    activity_ap["below_p"] = activity_ap["p_value"] < 0.05

    active_ratio_ap = activity_ap["below_p"].mean()
    return (active_ratio_ap,)


@app.cell
def _(active_ratio_ap, activity_ap, np, plt):
    # Plot raw and normalized AP scores vs -log10 p-values
    fig, axes = plt.subplots(1, 1, figsize=(14, 14))

    # Plot 1: Raw AP vs -log10(p-value)
    axes.scatter(
        data=activity_ap,
        x="average_precision",
        y="-log10(p-value)",
        c="below_p",
        cmap="tab10",
        s=10,
    )
    axes.axhline(-np.log10(0.05), color="black", linestyle="--")
    axes.set_xlabel("AP")
    axes.set_ylabel("-log10(p-value)")
    axes.set_title("Replicate retrieval")
    axes.text(
        0.65,
        1.5,
        f"Retrieved = {100 * active_ratio_ap:.2f}%",
        va="center",
        ha="left",
    )

    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(activity_ap, map, np, pos_sameby):
    activity_map = map.mean_average_precision(
        activity_ap, pos_sameby, null_size=10_000, threshold=0.05, seed=0
    )
    activity_map["-log10(p-value)"] = -activity_map["corrected_p_value"].apply(np.log10)
    activity_map
    return (activity_map,)


@app.cell
def _(activity_map):
    activity_map.below_corrected_p.mean()
    return


if __name__ == "__main__":
    app.run()
