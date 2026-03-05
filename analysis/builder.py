# %%
from itertools import combinations, product


def _create_extract_multich_tree(
    channels: list[int], extract_ncores: int | None
) -> dict:
    """Generate the extract_multich_tree dictionary for colocalization."""
    return {
        "tree": {
            pair: {
                "None": {
                    "max": ["pearson", "costes", "manders_fold", "rwc"],
                },
            }
            for pair in combinations(channels, r=2)
        },
        "kwargs": {
            "ncores": extract_ncores,
        },
    }


def build_pipeline(
    input_path: str,
    n_devices: int,
    addresses: list[str],
    extract_ncores: int | None,
):
    """Convenience function to build a pipeline definition."""
    fluo_base_config = {
        "input_path": input_path,
        "capture_order": "CYX",
        "ntps": 1,
        "segmentation_channel": {"nuclei": 1, "cell": 2},
    }
    fl_channels = range(5)

    hashed_input = hash(str(input_path))
    device_id = hashed_input % n_devices

    segmentation_channel: dict[str, int] = fluo_base_config["segmentation_channel"]
    random_hash = hash(str(input_path))
    for i, ch in enumerate(segmentation_channel):
        # segment_kwargs = pipeline["steps"][f"segment_{ch}"]["segmenter_kwargs"]
        address = addresses[random_hash % len(addresses)]
        device_id = random_hash % n_devices
    # logger.debug(f"{device_id=} {address=}")

    seg_params = {
        f"segment_{obj}": dict(
            segmenter_kwargs=dict(
                kind="nahual_cellpose",
                address=address,
                setup_params=dict(device=device_id),
                channel_to_segment=ch_id,
            ),
            img_channel=ch_id,
        )
        for i, (obj, ch_id) in enumerate(
            fluo_base_config["segmentation_channel"].items()
        )
    }

    extract_base = dict(
        channels=fl_channels,
        tree={
            **{
                i: {
                    "max": [
                        "radial_zernikes",
                        "intensity",
                        "sizeshape",
                        "ferret",
                        "texture",
                        "radial_distribution",
                        "zernike",
                        # "granularity", # Too time-consuming, deactivated for now
                    ]
                }
                for i in fl_channels
            },
        },
        kwargs=dict(
            # ncores=None,  # os.cpu_count(),
            ncores=extract_ncores,
        ),
    )

    ext_params = {
        f"extract{name}_{obj}": var
        for (name, var), obj in product(
            (
                ("", extract_base),
                # ("multi", extract_multich_tree),
            ),
            segmentation_channel,
        )
        if len(var)
    }

    base_pipeline = {
        "io": {**fluo_base_config},
        "nchannels": 5,
        "fl_channels": fl_channels,
        "extract_multich_tree": _create_extract_multich_tree(
            fl_channels, extract_ncores
        ),
        "steps": dict(
            tile=dict(
                image_kwargs=dict(
                    source=input_path,
                    # regex=regex,
                    capture_order=fluo_base_config["capture_order"],
                    # dimorder=fluo_base_config["dimorder"],
                ),
                tile_size=None,
                ref_channel=0,
                ref_z=0,
                calculate_drift=False,
            ),
            **seg_params,
            **ext_params,
        ),
        "passed_data": dict(
            **{
                f"extract_{obj}": [
                    ("masks", f"segment_{obj}"),
                    ("pixels", "tile"),
                ]
                for obj in fluo_base_config["segmentation_channel"]
            },
        ),
        "passed_methods": {
            f"segment_{obj}": ("tile", "get_tp_data", "img_channel")
            for obj in segmentation_channel
        },
        "save": (
            # "tile",
            *seg_params.keys(),
        ),
        "save_interval": 1,
    }

    # try:
    fov = input_path["key"]  # TODO Homogeneize how to name experiments
    return fov, base_pipeline

    # result, _ = run_pipeline_and_post(
    #     pipeline=base_pipeline,
    #     img_source=input_path,
    #     output_path=output_path,
    #     fov=fov,
    #     overwrite=False,
    # )
    # except Exception as e:
    #     print(f"Error: {e}")
