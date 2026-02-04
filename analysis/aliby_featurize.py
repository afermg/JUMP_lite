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
from joblib import Parallel, delayed
from loguru import logger
from tqdm import tqdm

# Register the codecs manually
numcodecs.register_codec(Jpegxl)

# dataset = "jump_target2_subset_BR00121438"
# dataset = "jump_target2_4plate"
dataset = "jump_core_annotated"
datasets_path = Path(f"/work/datasets/{dataset}")
compression_paths = [x for x in datasets_path.glob("*/") if x.name != "raw"]
n_devices = 4
n_addresses = 48

# Parameters shared amongst all models: tile_size and which channels to use (ids)
# These tell us when to pad or select channels to match the models
# model_group -> (tile_size, channels)
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
    dinov2=dict(
        model_group="dinov2",
        repo_or_dir="facebookresearch/dinov2",
        model_name="dinov2_vits14",
        device=-1,
    ),
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
    # openphenom=dict(
    #     model_group="openphenom",
    #     model_name="recursionpharma/OpenPhenom",
    #     device=-1,
    # ),
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
    ipc_addr = model_params["address"]
    setup_params = model_params["setup_params"]
    if "{}" in ipc_addr:
        hashed_input = hash(str(input_path))
        hashed_input_int = int(hashed_input)
        address_id = hashed_input_int % n_addresses
        ipc_addr = ipc_addr.format(f"_{address_id}")

        print(f"Formatted ipc address into {ipc_addr}")
        if setup_params.get("device") == -1:  # TODO formalise this
            device_id = hashed_input_int % n_devices
            setup_params["device"] = device_id
            print(f"Selected device {device_id}")

    embedding_step_name = f"nahual_embed_{model_name}"
    fluo_base_config = {
        "input_path": input_path,
        "image_kwargs": {
            "capture_order": "CYX",
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
        img_source=input_path,
        output_path=output_path,
        fov=input_path["key"],
        overwrite=False,
    )
    # except Exception as e:
    #     print(f"Error: {e}")


# %%
dsets = list(
    map(partial(dispatch_dataset, is_zarr=True, is_monozarr=True), compression_paths)
)
input_paths = []
print("Loading input paths as Image/Zarr objects")
for dset in dsets:
    # MonozarZarr dataset returns a dictionary with store->str, inputs_list -> list[str]
    input_paths.append(dset.get_position_ids())


# %%
# for model_name, v in model_params.items(): for compression_dir, dset in zip(compression_paths, dsets), total=len(dsets):
def process_with_timestamp(
    parameters: tuple[tuple[str, tuple[int, tuple[int], int]],],
    output_basedir: str | Path,
    dataset_name: str,
):
    (input_paths, compression_dir), (model_name, model_params) = parameters
    assert len(input_paths), "No files found in input dataset"
    if __name__ == "__main__":  # Add logging
        timestamp = strftime("%s%m%d%H%M")
        output_path = output_basedir / model_name / dataset_name / compression_dir.name

        logger.remove()
        logger.add(output_path / f"{timestamp}_{dataset_name}_{model_name}.log")
        # if __file__:
        #     shutil.copy(__file__, output_path / f"{timestamp}_script.py")

        # if False:
        #     result = Parallel(30)(delayed(process_input_path)(x) for x in input_paths)
        # else:
        #     from tqdm import tqdm
        # t0 = perf_counter()
    print(output_path)
    process_input_path_curried = partial(
        process_input_path,
        output_path=output_path,
        model_name=model_name,
        model_params=model_params,
    )
    with Pool() as p:
        result = p.map(process_input_path_curried, input_paths)
    # result = [
    #     process_input_path(input_path_d, output_path, model_name, model_params)
    #     for group_key, input_path_d in tqdm(input_paths.items())
    # ]
    # print(f"Processing took {perf_counter() - t0} seconds")
    return result


process_dataset_curried = partial(
    process_with_timestamp,
    dataset_name=dataset,
    output_basedir=Path("/work/datasets/aliby_output/"),
)

parameters_combinations = list(
    product(list(zip(input_paths, compression_paths)), model_params.items())
)

# result = Parallel(5)(
#     delayed(process_dataset_curried)(x) for x in parameters_combinations
# )

for paramset in parameters_combinations:
    result = process_dataset_curried(paramset)
    # Clean Nahual server instances to make space for new models
    # subprocess.run("screen -ls | awk -F'.' '/\\S+_[0-9]/ {print $1}' | xargs kill")
