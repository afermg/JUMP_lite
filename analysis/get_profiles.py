#!/usr/bin/env jupyter

"""Test a basic pipeline with a unique zarr directory as input."""  #

import os
import shutil
from functools import partial
from pathlib import Path
from time import perf_counter, strftime

from aliby.io.dataset import dispatch_dataset
from aliby.pipe import run_pipeline_and_post
from builder import build_pipeline
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

# Register the codecs manually


# dataset = "jump_target2_subset_BR00121438"
dataset = "jump_target2_4plate"
datasets_path = Path(f"/work/datasets/{dataset}")

compression_paths = [
    x for x in datasets_path.glob("*/") if x.name.startswith("jpegxl_lossy_d")
]
# %%
addresses = [f"ipc:///tmp/cellpose_{i}.ipc" for i in range(1, 9)]
n_devices = 4
# extract_ncores = None
# extract_ncores = os.cpu_count()
extract_ncores = 10


# %%
dsets = list(
    map(partial(dispatch_dataset, is_zarr=True, is_monozarr=True), compression_paths)
)

# %%
for compression_dir, dset in tqdm(zip(compression_paths, dsets), total=len(dsets)):
    pos_ids = dset.get_position_ids()
    input_paths = list(pos_ids.values())

    if __name__ == "__main__":  # Add logging
        timestamp = strftime("%s%m%d%H%M")
        output_path = (
            Path("/work/datasets/aliby_output")
            / "cp_measure"
            / dataset
            / compression_dir.name
        )

        logger.remove()
        logger.add(output_path / f"{timestamp}_{dataset}.log")
        # shutil.copy(__file__, output_path / f"{timestamp}_script.py")
    pipeline_recipes = dict([
        build_pipeline(input_path, n_devices, addresses, extract_ncores)
        for input_path in input_paths
    ])
    pipeline_curried = [
        partial(
            run_pipeline_and_post,
            pipeline=recipe,
            output_path=output_path,
            fov=fov,
            overwrite=False,
        )
        for fov, recipe in pipeline_recipes.items()
    ]

    if True:
        result = Parallel(5)(
            delayed(pipeline_curried[i])(img_source=x)
            for i, x in enumerate(input_paths)
        )
    else:
        from tqdm import tqdm

        t0 = perf_counter()
        result = [
            pipeline_curried[i](img_source=input_path)
            for i, input_path in tqdm(enumerate(input_paths))
        ]
        # print(f"Processing took {perf_counter() - t0} seconds")
