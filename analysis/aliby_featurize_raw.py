#!/usr/bin/env jupyter
"""CLI tool to featurize a dataset using a specific deep learning model deployed via Nahual."""  #

import shutil
from functools import partial
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from time import perf_counter, strftime

import numcodecs
from aliby.io.dataset import dispatch_dataset
from aliby.pipe import run_pipeline_and_post
from imagecodecs.numcodecs import Jpegxl
from loguru import logger

# from joblib import Parallel, delayed

# Register the codecs manually
numcodecs.register_codec(Jpegxl)

dataset = "jump_core_annotated"
datasets_path = Path(f"/work/datasets/{dataset}/raw")
# regex = ".*source_4__2021_08_23_Batch12__BR00126717__B11__DNA__2__Orig.tif"
regex = "(.*)__([A-Z][0-9]{2})__([A-Za-z]+)__([0-9])__Orig.tif"
capture_order = "PWCF"  # Plate, Well, Channel Foci
input_dimensions = "YX"
nchannels = 5
addresses = [f"ipc:///tmp/cellpose{i}.ipc" for i in range(6)]
dset = dispatch_dataset(datasets_path, regex=regex, capture_order=capture_order)
output_basedir = Path("/work/datasets/aliby_output/")
n_devices = 4
n_addresses = 40

# # Parameters shared amongst all models: tile_size and which channels to use (ids)
# # These tell us when to pad or select channels to match the models
# # model_group -> (tile_size, channels)
model_groups_inputs = dict(
    dinov2=dict(
        tile_size=224,
        selected_channels=[0, 1, 2],
    ),
    openphenom=dict(
        tile_size=256,
        selected_channels=[0, 1, 2, 3, 4],
        minmax_8bit=True,
    ),  # openphenom
    subcell=dict(
        tile_size=448,
        selected_channels=[0, 1, 2, 3],
    ),
    morphem=dict(
        tile_size=224,
        selected_channels=[0, 1, 2, 3, 4],
        # minmax_8bit=True,
    ),
)

# Only models in this dictionary will be used
model_setup_params = dict(
    # dinov2=dict(
    #     model_group="dinov2",
    #     repo_or_dir="facebookresearch/dinov2",
    #     model_name="dinov2_vits14",
    #     device=-1,
    # ),
    # subcell=dict(
    #     model_group="subcell",
    #     model_type="mae_contrast_supcon_model",
    #     model_channels="rybg",
    #     device=-1,
    # ),
    # dinov2_random=dict(
    #     model_group="dinov2",
    #     repo_or_dir="facebookresearch/dinov2",
    #     model_name="dinov2_vits14",
    #     pretrained=False,
    #     device=-1,
    # ),
    openphenom=dict(
        model_group="openphenom",
        model_name="recursionpharma/OpenPhenom",
        device=-1,
    ),
    # morphem=dict(
    #     model_group="morphem",
    #     model_name="CaicedoLab/MorphEm",
    #     device=-1,
    # ),
)
model_params = {
    model_name: {
        **model_groups_inputs[v["model_group"]],
        **{
            # "setup_params": model_setup_params.get(model_name, {}),
            "model_group": v.pop("model_group"),
            "setup_params": v,
            "address": f"ipc:///tmp/{model_name}{{}}.ipc",
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
    ipc_addr = model_params.get("address")
    setup_params = model_params["setup_params"]
    if ipc_addr and "{}" in ipc_addr:
        hashed_input = hash(str(input_path))
        hashed_input_int = int(hashed_input)
        address_id = hashed_input_int % n_addresses
        ipc_addr = ipc_addr.format(f"_{address_id}")

        # print(f"Formatted ipc address into {ipc_addr}")
        if setup_params.get("device") == -1:  # TODO formalise this
            device_id = hashed_input_int % n_devices
            setup_params["device"] = device_id
            print(f"Selected device {device_id}")

    embedding_step_name = f"nahual_embed_{model_name}"
    fluo_base_config = {
        "input_path": input_path,
        "image_kwargs": {
            "regex": regex,
            "capture_order": capture_order,
            "input_dimensions": input_dimensions,
        },
        "ntps": 1,
        "tile": {
            "kind": "crop",
            "tile_size": model_params["tile_size"],
            "calculate_drift": False,
            "minmax_8bit": model_params.get("minmax_8bit", False),
        },
    }
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

    # try:
    result, _ = run_pipeline_and_post(
        pipeline=base_pipeline,
        img_source=input_path["path"],
        output_path=output_path,
        fov="__".join(input_path["key"]),  # This is expected a string
        overwrite=False,
    )
    # except Exception as e:
    #     print(f"Error: {e}")


# %%
# for model_name, v in model_params.items(): for compression_dir, dset in zip(compression_paths, dsets), total=len(dsets):
def process_with_timestamp(
    input_paths: dict[tuple[str] | str, str],
    compression_dir: str,
    model_params: dict,
    model_name: str,
    output_basedir: str | Path,
    dataset_name: str,
):
    # (input_paths, compression_dir), (model_name, model_params) = parameters
    # input_paths = list(dataset.get_position_ids().values())
    assert len(input_paths), "No files found in input dataset"
    if __name__ == "__main__":  # Add logging
        timestamp = strftime("%s%m%d%H%M")
        output_path = output_basedir / model_name / dataset_name / compression_dir.name

        logger.remove()
        logger.add(output_path / f"{timestamp}_{dataset_name}_{model_name}.log")
        # if __file__:
        #     shutil.copy(__file__, output_path / f"{timestamp}_script.py")

    print(output_path)
    process_input_path_curried = partial(
        process_input_path,
        output_path=output_path,
        model_name=model_name,
        model_params=model_params,
    )
    if True:
        with Pool() as p:
            process_input_path_curried2 = partial(
                process_input_path_curried,
                model_name=model_name,
                output_path=output_path,
                model_params=model_params,
            )
            result = p.map(process_input_path_curried2, input_paths)
    else:
        result = []
        for key_path in input_paths:
            output = process_input_path(key_path, output_path, model_name, model_params)
            result.append(output)

    # print(f"Processing took {perf_counter() - t0} seconds")
    return result


process_dataset_curried = partial(
    process_with_timestamp,
    dataset_name=dataset,
    output_basedir=output_basedir,
)
input_paths = dset.get_position_ids()
result = process_with_timestamp(  #
    input_paths=input_paths,
    compression_dir=datasets_path,
    model_params=model_params["openphenom"],
    model_name="openphenom",
    dataset_name=dataset,
    output_basedir=output_basedir,
)
# parameters_combinations = list(
#     product(list(zip(input_paths, compression_paths)), model_params.items())
# )

# for paramset in parameters_combinations:
#     result = process_dataset_curried(paramset)
#     # Clean Nahual server instances to make space for new models
#     # subprocess.run("screen -ls | awk -F'.' '/\\S+_[0-9]/ {print $1}' | xargs kill")
