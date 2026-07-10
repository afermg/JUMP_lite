"""Aliby featurization driver — produces aliby_output/ consumed by `extract-cp-*`.

EXTERNAL DEPENDENCIES (not in pixi env by default):
- `aliby` Python package — see https://gitlab.com/aliby/aliby
- Nahual model-serving GPU servers must be launched before running this script,
  one server per model in ``model_setup_params``. A legacy launcher exists at
  ``archive/analysis/deploy_nahual_featurizers.sh``; adapt to your hardware.

Two filesystem roots are env-driven; everything else (dataset name, codec filter,
model list, device/address counts) is a hardcoded project choice — edit the
constants block below to match your run.

Env vars:
    DATA_ROOT        — base directory for input compressed datasets
                       (default: ./data)
    ALIBY_OUTPUT     — base directory for aliby_output/ outputs
                       (default: $DATA_ROOT/aliby_output)

Usage:
    DATA_ROOT=/my/data ALIBY_OUTPUT=/my/aliby_output python prep/aliby_featurize.py
"""

import os
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

numcodecs.register_codec(Jpegxl)

# ─── EDIT THIS BLOCK TO MATCH YOUR RUN ──────────────────────────────────────
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data"))
ALIBY_OUTPUT = Path(os.environ.get("ALIBY_OUTPUT", DATA_ROOT / "aliby_output"))

dataset = "jump_lite_updated"
datasets_path = DATA_ROOT / "compressed_test" / dataset
codec_glob_prefix = "jpegxl_lossy_mq"
n_devices = 4
n_addresses = 20
# ────────────────────────────────────────────────────────────────────────────

compression_paths = [
    x for x in datasets_path.glob("*/") if x.name.startswith(codec_glob_prefix)
]

# Parameters shared amongst all models: tile_size and which channels to use (ids)
model_groups_inputs = dict(
    dinov2=dict(
        tile_size=224,
        selected_channels=[0, 1, 2],
    ),
    openphenom=dict(
        tile_size=256,
        selected_channels=[0, 1, 2, 3, 4],
        minmax_8bit=True,
    ),
    subcell=dict(
        tile_size=448,
        selected_channels=[0, 1, 2, 3],
    ),
    morphem=dict(
        tile_size=224,
        selected_channels=[0, 1, 2, 3, 4],
    ),
)

model_setup_params = dict(
    openphenom=dict(
        model_group="openphenom",
        model_name="recursionpharma/OpenPhenom",
        device=-1,
    ),
    morphem=dict(
        model_group="morphem",
        model_name="CaicedoLab/MorphEm",
        device=-1,
    ),
    dinov2=dict(
        model_group="dinov2",
        repo_or_dir="facebookresearch/dinov2",
        model_name="dinov2_vits14",
        device=-1,
    ),
    subcell=dict(
        model_group="subcell",
        model_type="mae_contrast_supcon_model",
        model_channels="rybg",
        device=-1,
    ),
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
            "model_group": v.pop("model_group"),
            "setup_params": v,
            "address": f"ipc:///tmp/{model_name}{{}}.ipc",
        },
    }
    for model_name, v in model_setup_params.items()
}


def process_input_path(
    input_path: dict[str, str],
    output_path: str,
    model_name: str,
    model_params: dict,
):
    ipc_addr = model_params["address"]
    setup_params = model_params["setup_params"]
    if "{}" in ipc_addr:
        hashed_input = hash(str(input_path["key"]))
        hashed_input_int = int(hashed_input)
        address_id = hashed_input_int % n_addresses
        ipc_addr = ipc_addr.format(f"_{address_id}")
        print(f"Formatted ipc address into {ipc_addr}")
        if setup_params.get("device") == -1:
            device_id = hashed_input_int % n_devices
            setup_params["device"] = device_id
            print(f"Selected device {device_id}")

    embedding_step_name = f"nahual_embed_{model_name}"
    fluo_base_config = {
        "input_path": input_path,
        "image_kwargs": {"capture_order": "CYX"},
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

    result, _ = run_pipeline_and_post(
        pipeline=base_pipeline,
        img_source=input_path,
        output_path=output_path,
        fov=input_path["key"],
        overwrite=False,
    )


def process_with_timestamp(
    parameters,
    output_basedir: str | Path,
    dataset_name: str,
):
    (model_name, model_params), (input_paths, compression_dir) = parameters
    assert len(input_paths), "No files found in input dataset"
    if __name__ == "__main__":
        timestamp = strftime("%s%m%d%H%M")
        output_path = output_basedir / model_name / dataset_name / compression_dir.name
        logger.remove()
        logger.add(output_path / f"{timestamp}_{dataset_name}_{model_name}.log")

    print(output_path)
    process_input_path_curried = partial(
        process_input_path,
        output_path=output_path,
        model_name=model_name,
        model_params=model_params,
    )
    with Pool() as p:
        result = p.map(process_input_path_curried, input_paths.values())
    return result


if __name__ == "__main__":
    dsets = list(
        map(partial(dispatch_dataset, is_zarr=True, is_monozarr=True), compression_paths)
    )
    input_paths = []
    print("Loading input paths as Image/Zarr objects")
    for dset in dsets:
        input_paths.append(dset.get_position_ids())

    process_dataset_curried = partial(
        process_with_timestamp,
        dataset_name=dataset,
        output_basedir=ALIBY_OUTPUT,
    )

    parameters_combinations = list(
        product(
            model_params.items(),
            list(zip(input_paths, compression_paths)),
        )
    )

    for paramset in parameters_combinations:
        process_dataset_curried(paramset)
