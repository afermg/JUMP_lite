from functools import partial
from pathlib import Path
from time import perf_counter

from itertools import product, starmap

import duckdb
import polars as pl
from broad_babel.data import get_table
from joblib import Parallel, delayed
from jump_portrait.fetch import get_item_location_metadata, get_jump_image_batch
from PIL import Image
from pooch import retrieve
from tqdm import tqdm


def get_whole_plate_location_info(
    plate: str = "BR00121438",
) -> pl.DataFrame:
    """ """
    # Get plates
    meta_wells = get_table("well")
    meta_plate = get_table("plate")
    con = duckdb.connect()
    plate_info = con.sql(f"SELECT * FROM meta_plate WHERE Metadata_Plate='{plate}'")
    wells_in_plate = con.sql(f"SELECT *FROM meta_wells WHERE Metadata_Plate='{plate}'")
    whole_plate_metadata = con.sql(
        "SELECT Metadata_Source,Metadata_Batch,Metadata_Plate,Metadata_Well,Metadata_JCP2022 FROM wells_in_plate NATURAL JOIN plate_info"
    ).pl()
    con.close()
    return whole_plate_metadata


def get_metadata_batch(
    perturbations: tuple[str],
    cols=(
        "Metadata_Source",
        "Metadata_Batch",
        "Metadata_Plate",
        "Metadata_Well",
    ),
) -> pl.DataFrame:
    """
    Pull metadata tables using as many processes as possible. Maps JCP id -> address (source, plate, well, sites)
    """
    metadata = Parallel(n_jobs=-1)(
        delayed(partial(get_item_location_metadata, input_column="JCP2022"))(x)
        for x in perturbations
    )

    return [x.select((*cols, "Metadata_JCP2022")) for x in metadata]


out_path = Path("/work/datasets/jump_core/raw")
out_path.mkdir(parents=True, exist_ok=True)

meta_file = out_path.parent / "metadata.parquet"
progress_file = out_path.parent / "progress.txt"

sample = 10  # No. of CRISPR and ORF to test
seed = 1

pull_plates = True
plates_to_pull = ["JCPQC016", "BR00121438", "ACPJUM012", "110000293081"]



# Pull JCP ids
crispr = (
    pl.scan_csv(
        retrieve(
            "https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/crispr.csv.gz",
            known_hash="55e36e6802c6fc5f8e5d5258554368d64601f1847205e0fceb28a2c246c8d1ed",
        ),
    )
    .select(pl.col("Metadata_JCP2022"))
    .collect()
    .to_series()
    .sample(sample, seed=seed)
)
orf = (
    pl.scan_csv(
        retrieve(
            "https://github.com/jump-cellpainting/datasets/raw/refs/heads/main/metadata/orf.csv.gz",
            known_hash="9c7ec4b0fa460a3a30f270a15f11b5e85cef9dd105c8a0ab8ab50f6cc98894b8",
        ),
    )
    .select(pl.col("Metadata_JCP2022"))
    .collect()
    .to_series()
    # .sample(sample, seed=seed)
)

compound_selection = (
    pl.scan_csv(
        "../metadata/repurposed_compounds.tsv",
        separator="\t",
    )
    .select(pl.col("Metadata_JCP2022"))
    .collect()
    .to_series()
    .unique()
)

# %%

channels = ["DNA", "AGP", "Mito", "RNA", "ER"]
sites = [str(i) for i in range(1, 7)]  # 1->6
correction = "Orig"

# Do not pull the mapper unless explicitly told to
if not (meta_file).exists():
    # %% Download metadata tables
    print("Downloading gene metadata")
    gene_list = (*crispr, *orf)
    t_start = perf_counter()
    gene_rows = get_metadata_batch(gene_list)
    print(f"Done downloading gene metadata in {int(perf_counter() - t_start)} seconds")

    # Compounds too
    print("Downloading compound metadata")
    t_start = perf_counter()
    compound_selection = get_metadata_batch(compound_selection)

    # Add whole plates if necessary
    whole_plate = get_whole_plate_location_info("110000293081")
    compound_rows = pl.concat((whole_plate, compound_selection)).unique()
    print(
        f"Done downloading compound metadata in {int(perf_counter() - t_start)} seconds"
    )
    all_rows_data = (*gene_rows, *compound_rows)

    metadata_all = pl.concat(all_rows_data)
    metadata_all.write_parquet(meta_file)
    print("Parquet saved. Will download images now.")

else:
    metadata_all = pl.read_parquet(meta_file)



if pull_plates:
    print("Only pulling plates:", plates_to_pull)
    metadata_all = []
    for plate in plates_to_pull:
        whole_plate = get_whole_plate_location_info(plate)

        metadata_all.append(whole_plate)
    
    metadata_all = pl.concat(metadata_all)



# Convert metadata to list of single-row DataFrames for parallel processing
all_rows = [metadata_all.slice(i, 1) for i in range(len(metadata_all))]

# %%
def save_array(image, address: tuple[str]):
    # `address` is a tuple of (source, plate, well, channel, site)
    fullname = "__".join(address)
    pil_img = Image.fromarray(image)
    # check if file exists to avoid re-downloading
    if (out_path / f"{fullname}.tif").exists():
        return 
    pil_img.save(out_path / f"{fullname}.tif")

def check_all_exist(meta: pl.DataFrame, channel, site, correction) -> bool:
    iterable = list(
        starmap(
            lambda *x: (*x[0], *x[1:]),
            product(meta.rows(), channel, site, [correction]),
        )
    )
    fails = 0
    for address in iterable:
        fullname = "__".join(address)
        if not (out_path / f"{fullname}.tif").exists():
             fails += 1
    if fails == 0 or (fails == len(site)):
        return True
    return False


def download_and_save_image(meta: pl.DataFrame, channel, site, correction):
    try:
        meta_nojcp = meta.select(pl.exclude("Metadata_JCP2022"))
        if check_all_exist(meta_nojcp, channel, site, correction):
            return True
        addresses, images = get_jump_image_batch(
            meta_nojcp, channel=channel, site=site, correction=correction
        )
        for address, image in zip(addresses, images):
            save_array(image, address)
        return True
    except:
        return False

for x in tqdm(all_rows[:], total=len(all_rows)):

    download_and_save_image(x, channel=channels, site=sites, correction=correction)

        



fh = open(progress_file, "w")
results = Parallel(n_jobs=32)(
    delayed(
        partial(
            download_and_save_image, channel=channels, site=sites, correction=correction
        )
    )(x)
    for x in tqdm(all_rows[:], total=len(all_rows), file=fh)
)
fh.close()
progress_file.unlink()

# print the results
n_success = sum(results)
print(f"Successfully downloaded {n_success} out of {len(all_rows)} items.")
