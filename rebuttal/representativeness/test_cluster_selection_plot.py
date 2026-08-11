from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from PIL import Image

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "outputs/profile_cluster_representativeness_v1"
SPEC = importlib.util.spec_from_file_location(
    "plot_cluster_selection_compounds", ROOT / "plot_cluster_selection_compounds.py"
)
assert SPEC and SPEC.loader
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)

EXPECTED_COORDINATE_DIGEST = "5249f5bf80e939c513b047592601b57f0dbd309c6583398b126eccf93657190e"


def test_embedding_fit_api_cannot_receive_selection_labels():
    signature = inspect.signature(PLOT.fit_label_blind_umap)
    assert list(signature.parameters) == ["pca32", "eligible"]
    source = inspect.getsource(PLOT.main)
    embedding_fixed = source.index("coordinates = fit_label_blind_umap")
    manifest_verified = source.index("verify(SELECTED)")
    manifest_parsed = source.index("pl.read_parquet(SELECTED)")
    assert embedding_fixed < manifest_verified < manifest_parsed


def test_embedding_row_order_selection_join_and_counts_are_exact():
    embedding = pl.read_parquet(RESULT / "compound_cluster_umap.parquet")
    unlabeled = pl.read_parquet(RESULT / "fit/cluster_assignments_unlabeled.parquet")
    selected = pl.read_parquet(ROOT / "outputs/profile_space/manifests/jump_lite_compounds.parquet")
    assignments = pl.read_parquet(RESULT / "compound_cluster_assignments.parquet")

    assert embedding.shape == (115_721, 6)
    assert embedding["Metadata_JCP2022"].to_list() == unlabeled["Metadata_JCP2022"].to_list()
    assert embedding["cluster_id"].to_list() == unlabeled["cluster_id"].to_list()
    assert embedding["cluster_id"].to_list() == assignments["cluster_id"].to_list()
    assert embedding["fit_eligible"].sum() == 95_426
    assert embedding["cluster_id"].n_unique() == 128
    assert embedding["selected"].sum() == 3_832
    embedded_selected = set(
        embedding.filter(pl.col("selected"))["Metadata_JCP2022"].to_list()
    )
    assert embedded_selected == set(selected["Metadata_JCP2022"].to_list())


def test_coordinate_digest_is_exact_and_finite():
    embedding = pl.read_parquet(RESULT / "compound_cluster_umap.parquet")
    coords = embedding.select("umap_1", "umap_2").to_numpy()
    assert coords.dtype == np.float32
    assert np.isfinite(coords).all()
    digest = PLOT.coordinate_digest(embedding["Metadata_JCP2022"].to_list(), coords)
    provenance = json.loads((RESULT / "cluster_selection_figure_provenance.json").read_text())
    assert digest == EXPECTED_COORDINATE_DIGEST
    assert provenance["coordinate_digest_sha256_ids_then_float32"] == digest
    assert provenance["parameters"]["umap_fit_rows"] == 95_426
    assert provenance["parameters"]["transform_ineligible_rows"] == 20_295
    assert {record["path_scope"] for record in provenance["inputs"]} == {
        PLOT.REPOSITORY_SCOPE
    }
    assert all(not Path(record["path"]).is_absolute() for record in provenance["inputs"])


def test_summary_table_values_and_qualifications_are_exact():
    summary = pd.read_csv(RESULT / "cluster_selection_summary_table.csv").set_index("metric")
    expected = {
        "total_compounds": 115_721,
        "fit_eligible_compounds": 95_426,
        "selected_compounds": 3_832,
        "all_compound_prevalence": 0.03311412794566241,
        "eligible_compound_prevalence": 0.040156770691425814,
        "occupied_clusters": 120,
        "total_clusters": 128,
        "eligible_mass_in_occupied_clusters": 0.962368746463228,
        "total_variation_selected_vs_eligible": 0.47504893669424875,
        "jensen_shannon_divergence_nats": 0.17227756705149666,
        "average_precision_constant": 0.03946976570130267,
        "average_precision_acquisition_structure": 0.5158264268349165,
        "average_precision_cluster_only": 0.27766569255144513,
        "average_precision_structure_plus_cluster": 0.5764816772139572,
        "combined_to_structure_ap_ratio": 1.117588489506476,
        "conditional_permutation_p": 0.0004997501249375,
        "mean_seed_adjusted_rand_index": 0.16241422160818178,
    }
    assert set(summary.index) == set(expected)
    for metric, value in expected.items():
        assert abs(float(summary.loc[metric, "value"]) - value) <= 1e-12
    assert "1.25x" in summary.loc["combined_to_structure_ap_ratio", "qualification"]
    assert "biological classes" in summary.loc["mean_seed_adjusted_rand_index", "qualification"]


def test_plot_outputs_are_valid_and_hashed():
    png = RESULT / "cluster_selection_compound_map.png"
    pdf = RESULT / "cluster_selection_compound_map.pdf"
    with Image.open(png) as image:
        assert image.size == (4560, 1575)
        image.verify()
    assert pdf.read_bytes().startswith(b"%PDF-")
    hashes = json.loads((RESULT / "cluster_selection_figure_hashes.json").read_text())["files"]
    assert len(hashes) == 5
    for record in hashes:
        assert record["path_scope"] == PLOT.FIGURE_OUTPUT_SCOPE
        assert not Path(record["path"]).is_absolute()
        path = RESULT / record["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert path.stat().st_size == record["bytes"]
        assert digest == record["sha256"]


def test_plot_generation_refuses_overwrite():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        existing = root / PLOT.OUTPUT_NAMES[0]
        existing.write_text("keep")
        try:
            PLOT.require_outputs_absent(root)
        except RuntimeError as error:
            assert "Refusing to overwrite" in str(error)
        else:
            raise AssertionError("existing plot output was overwritten")
        assert existing.read_text() == "keep"


def run_all():
    tests = [value for key, value in sorted(globals().items())
             if key.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"{len(tests)} cluster plot tests passed")


if __name__ == "__main__":
    run_all()
