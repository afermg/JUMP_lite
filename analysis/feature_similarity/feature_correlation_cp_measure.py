import marimo

__generated_with = "0.18.1"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Cleaning up the data

    Clean up and calculate statistics on-the-fly. We save intermediate results for faster reruns.
    """)
    return


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
    return Path, basic_cleanup, cs, duckdb, mo, np, os, pd, pl, warnings


@app.cell
def _():


    # from util_ap import get_phenotypic_scores
    # from util_stats import pvals_from_profile
    # from utils import add_metadata_map
    return


@app.cell
def _(Path):
    workspace_dir = Path("/work") / "datasets" /  "aliby_output"
    cache_dir = (
        Path(workspace_dir) / "db_cache"
    )  # Folder to save the intermediate databases. This greatly speeds up restarting the notebook.
    return cache_dir, workspace_dir


@app.cell
def _(fb_results):
    fb_results.value
    return


@app.cell
def _(mo, workspace_dir):
    fb_results = mo.ui.file_browser(
        workspace_dir,
        selection_mode="directory",
        multiple=True,
        label="Pick the folder with the pipeline results (the level called `profiles`)",
        restrict_navigation=True,
    )
    fb_results
    return (fb_results,)


@app.cell
def _(fb_results):
    print(fb_results.value)
    return


@app.cell
def _(Path, fb_results):
    assert len(fb_results.value), "Please select an assay."
    profiles_dir = fb_results.value[0].path
    if profiles_dir.name != "profiles":
        # Search for probile dir recursively
        top_level_dirs = list(profiles_dir.glob("*"))
        profiles_dirs = [Path(next(x.rglob("profiles"))) for x in top_level_dirs]
          # This will fail if folder is not found
        print(f"Using profile {profiles_dirs}")
        print(top_level_dirs)
        #print(profiles_dir)
    return (profiles_dirs,)


@app.cell
def _(profiles_dirs):
    len(profiles_dirs)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Once you pick your profile directory, automatically determine the file structure for images and masks.
    """)
    return


@app.cell
def _(Path, basic_cleanup, cache_dir, cs, duckdb, pl, warnings, workspace_dir):

    def get_features(profiles_dir):

        parquet_files = profiles_dir / "*.parquet"
        # Filter empty dfs
        masks_dir = profiles_dir / ".." / "steps"
        distance ="cosine"

        site_col = "site"
        db_file = Path(cache_dir) / (
            "_".join(profiles_dir.relative_to(workspace_dir).parts) + f"_{distance}" + ".db"
        )

        cache_dir.mkdir(exist_ok=True, parents=True)

        objects = tuple([
            x.name.split("_")[-1]
            for x in next(masks_dir.glob("*")).glob("*")
            if x.name.startswith("segment")
        ])

        metric_name = "metric"  # "Feature" # This was originally Feature
        branch_name = "branch"
        value_name = "values"  # "value" This used to be value. Will go back to that in the following analyses
        cc_metric = "Area"  # Feature to be used for cell count. Make sure this maps 1:1 with the segmentation masks.
        tp_name = "tp"

        overwrite_str = "TABLE IF NOT EXISTS"

        with duckdb.connect(db_file) as con:
            raw = con.sql(
                f"""
                SELECT *, parse_filename(filename,true) AS {site_col}, from read_parquet('{parquet_files}', filename=true)
                """
            )
            # Cover old datasets with column name "values" and datatype lists
            value_dtype = [x[1] for x in raw.description if x[0] == value_name]
            if len(value_dtype) and value_dtype[0] == "list":
                raw = con.sql(f"SELECT *,UNNEST({value_name}) AS value FROM raw")
                #raw_pl = raw.pl()

            con.sql(  # Create well-level dataset
                f"""
                CREATE {overwrite_str} well_level AS (SELECT {site_col},{branch_name} || {metric_name} AS full_metric_name,object,mean(value) AS cvalue FROM raw GROUP BY {tp_name},{site_col},{branch_name},{metric_name},object)
                """
            )
            # well_level_pl = well_level.pl()
            oc_df = con.sql(
                f"""
                SELECT {site_col},object,count({site_col}) AS oc FROM raw WHERE {metric_name} = '{cc_metric}' GROUP BY {site_col},{branch_name},{metric_name},object ORDER BY SITE,object
                """,
            )
            oc_piv = con.sql("PIVOT oc_df ON object USING any_value(oc)").pl()
            pivoted = con.sql(
                f"PIVOT well_level ON object,full_metric_name USING any_value(cvalue)"
            )#.pl()
            pivoted_pl = pivoted.pl() # Note that this conversion is not possible for time series data

        warnings.filterwarnings("ignore")

        # For some reason this is necessary for viability assay
        clean, ndropped = basic_cleanup(pivoted_pl, meta_selector=cs.by_dtype(pl.String))
        #meta_added = add_metadata_map(clean)
        # meta_added = meta_added.select(cs.by_dtype(pl.String), pl.col([k for k,v in meta_added.std().to_dicts()[0].items() if v and v > 1]))
        return clean, pivoted_pl, ndropped
    return (get_features,)


@app.cell
def _(profiles_dirs):
    profiles_dirs
    return


@app.cell
def _(Path):
    Path("analysis") 
    return


@app.cell
def _(os):

    print(os.getcwd())
    return


@app.cell
def _(Path, get_features, profiles_dirs):
    features_per_compression = {}
    for path in profiles_dirs:
        _clean, _pivoted_pl, _ndropped = get_features(path)

        _clean = _clean.to_pandas()
        _clean[["Metadata_Source", 
                  "Metadata_Batch",
                  "Metadata_Plate",
                  "Metadata_Well",
                  "Metadata_Site"]] = _clean.site.str.split("__",  expand=True)
        name = str(path).split("/")[-2]
        _clean = _clean.rename(columns={"site" : "Metadata_id"})
        features_per_compression[name] = {"clean" : _clean, "pivoted_pl": _pivoted_pl.to_pandas(), "ndropped": _ndropped}


    
        _clean.to_parquet(Path(f"analysis/feature_similarity/output/{name.split(".")[0]}.parquet"))
    return (features_per_compression,)


@app.cell
def _(features_per_compression):
    features_per_compression["zstd.zarr"]["clean"].isna().sum().sort_values()
    return


@app.cell
def _(features_per_compression):
    features_per_compression["zstd.zarr"]["pivoted_pl"].isna().sum(axis=0).sort_values()
    return


@app.cell
def _(features_per_compression, np):
    uncompressed_df = features_per_compression["zstd.zarr"]["pivoted_pl"].copy()
    results = []
    for key in features_per_compression.keys():
        df_compression = features_per_compression[key]["pivoted_pl"].copy()
        print("original: ", key, df_compression.shape)
        merged_df_with_non_compressed = uncompressed_df.merge(df_compression, on="site" )

        print("Merged: ", merged_df_with_non_compressed.shape)
        merged_df_with_non_compressed_nan = merged_df_with_non_compressed.dropna(axis=1, thresh=1000)
        merged_df_with_non_compressed_nan = merged_df_with_non_compressed_nan.dropna(axis=0)

        print("Nan dropped: ", merged_df_with_non_compressed_nan.shape)

        size_ = merged_df_with_non_compressed_nan.shape[0]

        count=0
        for feature in [x for x in df_compression.columns if x != "site" and ((x+"_x" in merged_df_with_non_compressed_nan.columns) and (x+"_y" in merged_df_with_non_compressed_nan.columns))]:

            corr = np.corrcoef(merged_df_with_non_compressed_nan[feature+"_x"],merged_df_with_non_compressed_nan[feature+"_y"])

            results.append([key, feature, corr[0,1], size_])
            count+=1
        print(count)



    return df_compression, results


@app.cell
def _(pd, results):
    df = pd.DataFrame(results, columns=["key", "feature", "corr", "size"])
    return (df,)


@app.cell
def _(df):
    df["key"].value_counts()
    return


@app.cell
def _(df):
    df["size"].value_counts()
    return


@app.cell
def _():
    import seaborn as sns
    return (sns,)


@app.cell
def _(df):
    df.groupby("key")["corr"].mean()
    return


@app.cell
def _(df, sns):
    sns.boxplot(data=df , x = "key", y = "corr")#, log_scale=(False, True))
    return


@app.cell
def _(df_compression):
    df_compression
    return


if __name__ == "__main__":
    app.run()
