from functools import partial
from pathlib import Path

import boto3
import duckdb
from joblib import Parallel, delayed

out_dir = Path("/work/datasets/jump_lite/imgs/raw")
print("Loading list of files")
with duckdb.connect() as con:
    uris_list = [
        (
            *list(x.values())[:-3],
            str(x["Metadata_Site"]),
            x["Metadata_Channel"].removeprefix("URL_Orig"),
            x["uri"].removeprefix("s3://cellpainting-gallery/"),
        )
        for x in con.sql(
            "FROM read_parquet('/work/datasets/jump_lite/misc/jl_index_tidy.parquet')"
        )
        .to_arrow_table()
        .to_pylist()
    ]
print("File list ready")


def download_uri(meta: list[str], out_dir: str):
    from botocore import UNSIGNED
    from botocore.config import Config
    from loguru import logger

    *location, key = meta
    local_name = "__".join(location) + ".tif"
    local_file = out_dir / Path(local_name)
    Path(local_file).parent.mkdir(exist_ok=True, parents=True)
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    try:
        if not local_file.exists():
            logger.add(out_dir / "../../misc" / "download_log.txt")
            text = f"Downloading {key} into {local_file}"
            logger.info(text)
            s3_client.download_file("cellpainting-gallery", key, str(local_file))
            logger.info(f"{key} was successfully downloaded")

    except Exception as e:
        logger.error(f"{key} Failed: {e}")


curried = partial(download_uri, out_dir=out_dir)


print("Downloads will start now")
result = list(Parallel(n_jobs=192)(delayed((curried))(x) for x in uris_list))
