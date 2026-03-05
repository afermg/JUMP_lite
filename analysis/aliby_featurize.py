#!/usr/bin/env jupyter
"""CLI tool to featurize a dataset using a specific deep learning model deployed via Nahual."""  #

import shutil
from functools import partial
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from time import strftime

import numcodecs
from aliby.io.dataset import dispatch_dataset
from aliby.pipe import run_pipeline_and_post
from imagecodecs.numcodecs import Jpegxl
from loguru import logger

# Register the codecs manually
numcodecs.register_codec(Jpegxl)

threaded = True
n_devices = 4
n_addresses = 20
capture_order = "CYX"
regex = None
input_dimensions = None

# 4 plates
# dataset = "jump_target2_4plate"
# datasets_path = Path(f"/work/datasets/{dataset}")
# out_dir = Path(f"/work/datasets/aliby_output/plate4_rerun_scale_std")

# JL
# dataset = "jump_lite_updated"
# datasets_path = Path(f"/work/datasets/compressed_test/{dataset}")
# out_dir = Path("/work/datasets/aliby_output/jump_lite_rerun")

# Process raw images
dataset = "jump_lite/imgs"
datasets_path = Path(f"/work/datasets/{dataset}/")
regex = "(.*)__([A-Z][0-9]{2})__([0-9])__([A-Za-z]+).tif"  # Our format
capture_order = "PWFC"  # Plate, Well, Channel Foci
input_dimensions = "YX"
nchannels = 5
out_dir = Path("/work/datasets/aliby_output/jump_lite_raw/")


compression_paths = [
    x
    for x in datasets_path.glob("*/")
    # if x.name.endswith("mq.zarr")
    # or x.name.startswith("jpegxl_lossy_d15")
]
# %%
# %%
if capture_order == "CYX":  # Zarr
    dsets = list(
        map(
            partial(dispatch_dataset, is_zarr=True, is_monozarr=True), compression_paths
        )
    )
else:  # Raw
    dsets = []
    for compression_path in compression_paths:
        dsets.append(
            dispatch_dataset(compression_path, capture_order=capture_order, regex=regex)
        )

# %%
input_paths = []
print("Loading input paths as Image/Zarr objects")
for dset in dsets:
    # MonozarZarr dataset returns a dictionary with store->str, inputs_list -> list[str]
    input_paths.append(dset.get_position_ids())

# %%

# Parameters shared amongst all models: tile_size and which channels to use (ids)
# These tell us when to pad or select channels to match the models
# model_group -> (tile_size, channels)
# Channels are ordered alphabetically (AGP, DNA, ER, Mito, RNA)
model_groups_inputs = dict(
    dinov2=dict(
        tile_size=224,
        selected_channels=[0, 1, 2],
    ),
    openphenom=dict(
        tile_size=256,
        selected_channels=[0, 1, 2, 3, 4],
        clip_outliers=True,
        standard_scale=False,
        convert_8bit=True,
    ),  # openphenom
    subcell=dict(
        tile_size=448,
        selected_channels=[3, 2, 1, 0],
        clip_outliers=True,
        # convert_8bit=True,
        standard_scale=False,
    ),
    morphem=dict(
        tile_size=224,
        selected_channels=[0, 1, 2, 3, 4],
    ),
)

# Only models in this dictionary will be used
model_setup_params = dict(
    # morphem=dict(
    #     model_group="morphem",
    #     model_name="CaicedoLab/MorphEm",
    #     device=-1,
    # ),
    # subcell__clip01=dict(
    #     model_group="subcell",
    #     model_type="mae_contrast_supcon_model",
    #     model_channels="rybg",
    #     device=-1,
    # ),
    dinov2=dict(
        model_group="dinov2",
        repo_or_dir="facebookresearch/dinov2",
        model_name="dinov2_vits14",
        device=-1,
    ),
    openphenom=dict(
        model_group="openphenom",
        model_name="recursionpharma/OpenPhenom",
        device=-1,
        convert_8bit=True,
        standard_scale=True,
        clip_outliers=True,
    ),
    # subcell__nonstd=dict(
    #     model_group="subcell",
    #     model_type="mae_contrast_supcon_model",
    #     model_channels="rybg",
    #     device=-1,
    # ),
    dinov2_random=dict(
        model_group="dinov2",
        repo_or_dir="facebookresearch/dinov2",
        model_name="dinov2_vits14",
        pretrained=False,
        device=-1,
    ),
)
model_params = {
    model_name: {
        **model_groups_inputs[v["model_group"]],
        **{
            # "setup_params": model_setup_params.get(model_name, {}),
            "model_group": v.pop("model_group"),
            "setup_params": v,
            "address": f"ipc:///tmp/{model_name.split('__')[0]}{{}}.ipc",  # ignore stuff after `__`. Allows reusing nahual instances.
        },
    }
    for i, (model_name, v) in enumerate(model_setup_params.items())
}


# %%
def process_input_path(
    input_path: dict[str, str],  # "store_path"->store_path,"key"->group_key
    output_path: str,
    model_name: str,
    model_params: dict,
    # address: str,
    # device: int = 0,
):
    ipc_addr = model_params["address"]
    setup_params = model_params["setup_params"]
    if "{}" in ipc_addr:
        hashed_input = hash(str(input_path["key"]))
        hashed_input_int = int(hashed_input)
        address_id = hashed_input_int % n_addresses
        ipc_addr = ipc_addr.format(f"_{address_id}")

        # print(f"Formatted ipc address into {ipc_addr}")
        if setup_params.get("device") == -1:
            device_id = hashed_input_int % n_devices
            setup_params["device"] = device_id
            # print(f"Selected device {device_id}")

    embedding_step_name = f"nahual_embed_{model_name}"
    fluo_base_config = {
        "input_path": input_path,
        "image_kwargs": {
            "capture_order": capture_order,  # Only used with images
        },
        "ntps": 1,
        "tile": {
            "kind": "crop",
            "tile_size": model_params["tile_size"],
            "calculate_drift": False,
            "clip_outliers": model_params.get("clip_outliers", False),
            "convert_8bit": model_params.get("convert_8bit", False),
            "standard_scale": model_params.get("standard_scale", True),
        },
    }
    # These are conditional on whether we use zarr or a list of images
    if all((regex, input_dimensions)):
        fluo_base_config["image_kwargs"]["regex"] = regex
        fluo_base_config["image_kwargs"]["input_dimensions"] = input_dimensions

    embed_params = dict(
        address=ipc_addr,
        model_group=model_params["model_group"],
        selected_channels=model_params["selected_channels"],
        setup_params=setup_params,
    )
    base_pipeline = {
        "io": {**fluo_base_config},
        "steps": {
            "tile": dict(
                **fluo_base_config["tile"],
                **dict(
                    image_kwargs=dict(
                        source=input_path,
                        **fluo_base_config["image_kwargs"],
                    )
                ),
            ),
            embedding_step_name: embed_params,
        },
        "passed_data": {embedding_step_name: [("pixels", "tile", "data")]},
        "save": (),
        "save_interval": 1,
    }

    try:
        fov = input_path["key"]
        if isinstance(fov, tuple):  # Cover key is (str | tuple[str])
            fov = "__".join(input_path["key"])
        result, _ = run_pipeline_and_post(
            pipeline=base_pipeline,
            img_source=input_path,
            output_path=output_path,
            fov=fov,
            overwrite=False,
        )
    except Exception as e:
        logger.error(e)


# %%
# for model_name, v in model_params.items(): for compression_dir, dset in zip(compression_paths, dsets), total=len(dsets):
def process_with_timestamp(
    parameters: tuple[tuple[str, tuple[int, tuple[int], int]],],
    output_basedir: str | Path,
    dataset_name: str,
):
    (model_name, model_params), (input_paths, compression_dir) = parameters
    assert len(input_paths), "No files found in input dataset"
    if __name__ == "__main__":  # Add logging
        timestamp = strftime("%s%m%d%H%M")
        output_path = output_basedir / dataset_name / model_name / compression_dir.name

        logger.remove()
        logger.add(output_path / f"{timestamp}_{dataset_name}_{model_name}.log")
        if __file__:
            shutil.copy(__file__, output_path / f"{timestamp}_script.py")

    print(f"Output path: {output_basedir}")
    process_input_path_curried = partial(
        process_input_path,
        output_path=output_path,
        model_name=model_name,
        model_params=model_params,
    )
    # Thread or not
    if threaded:
        with Pool() as p:
            result = p.map(process_input_path_curried, input_paths)
    else:
        result = [process_input_path_curried(path) for path in input_paths]
    # print(f"Processing took {perf_counter() - t0} seconds")
    return result


process_dataset_curried = partial(
    process_with_timestamp,
    dataset_name=dataset,
    output_basedir=out_dir,
)

parameters_combinations = list(
    product(
        model_params.items(),
        list(zip(input_paths, compression_paths)),
    )
)

# result = Parallel(5)(
#     delayed(process_dataset_curried)(x) for x in parameters_combinations
# )

for paramset in parameters_combinations:
    result = process_dataset_curried(paramset)
# Clean Nahual server instances to make space for new models
# subprocess.run("screen -ls | awk -F'.' '/\\S+_[0-9]/ {print $1}' | xargs kill")
