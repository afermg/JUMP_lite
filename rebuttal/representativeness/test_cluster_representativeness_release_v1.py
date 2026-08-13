from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from PIL import Image

HERE = Path(__file__).resolve().parent
RESULT = HERE / "outputs/profile_cluster_representativeness_release_v1"
HISTORICAL = HERE / "outputs/profile_cluster_representativeness_v1"
SPEC = importlib.util.spec_from_file_location(
    "release_rescore", HERE / "rescore_cluster_representativeness_release_v1.py"
)
assert SPEC and SPEC.loader
m = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)


def test_current_metadata_and_manifest_contracts_are_exact():
    manifest, facts = m.derive_current_manifest()
    assert facts["metadata"]["bytes"] == 949_745
    assert facts["metadata"]["sha256"] == "bbedb37f12fdeb9a09e72abaa166159d286052dc1201166d811c5310db5cd7e1"
    assert facts["compound_identifiers"] == 3_776
    assert facts["treatments"] == 3_775
    assert facts["negative_controls"] == 1
    assert facts["negative_control_id"] == "JCP2022_033924"
    assert facts["historical_only_count"] == 57
    assert facts["current_only_count"] == 0
    assert facts["all_assignments_present"] and facts["all_fit_eligible"]
    assert manifest.height == manifest["Metadata_JCP2022"].n_unique() == 3_775
    assert manifest["Metadata_JCP2022"].to_list() == sorted(manifest["Metadata_JCP2022"].to_list())
    assert "JCP2022_033924" not in set(manifest["Metadata_JCP2022"].to_list())


def test_current_outputs_are_distinct_complete_and_hash_frozen():
    assert RESULT != HISTORICAL
    assert {path.name for path in RESULT.iterdir()} == m.OUTPUT_NAMES
    assert not (RESULT / "matched_retrieval_sensitivity.csv").exists()
    hashes = json.loads((RESULT / "output_hashes.json").read_text())
    assert len(hashes["files"]) == len(m.OUTPUT_NAMES) - 1
    for record in hashes["files"]:
        path = m.resolve_output_record(record, RESULT)
        assert path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    provenance = json.loads((RESULT / "provenance.json").read_text())
    reviewed = provenance["reviewed_fit_identity"]
    assert (reviewed["analysis_helper"]["bytes"], reviewed["analysis_helper"]["sha256"]) == m.EXPECTED_ANALYSIS_SOURCE
    assert (reviewed["fit_completion"]["bytes"], reviewed["fit_completion"]["sha256"]) == m.EXPECTED_FIT_COMPLETION
    assert {name: (record["bytes"], record["sha256"]) for name, record in reviewed["fit_artifacts"].items()} == m.EXPECTED_FIT_ARTIFACTS
    assert provenance["counts"] == {
        "all": 115_721, "current_only": 0, "eligible": 95_426,
        "historical_only": 57, "release_compound_identifiers": 3_776,
        "release_negative_controls_excluded": 1, "selected_treatments": 3_775,
    }
    assert provenance["matched_comparator_sensitivity"].startswith("omitted")


def test_current_assignment_scoring_and_partition_values_are_exact():
    assignments = pl.read_parquet(RESULT / "compound_cluster_assignments.parquet")
    manifest = pl.read_parquet(RESULT / "current_release_treatment_manifest.parquet")
    assert assignments.height == assignments["Metadata_JCP2022"].n_unique() == 115_721
    assert int(assignments["selected"].sum()) == 3_775
    assert assignments.filter(pl.col("selected"))["fit_eligible"].all()
    assert set(assignments.filter(pl.col("selected"))["Metadata_JCP2022"].to_list()) == set(manifest["Metadata_JCP2022"].to_list())
    table = pl.read_parquet(RESULT / "cluster_selection_table.parquet")
    assert table.height == 128 and int(table["n_compounds"].sum()) == 115_721
    assert int(table["n_selected"].sum()) == 3_775
    summary = json.loads((RESULT / "representation_summary.json").read_text())
    assert summary["selected_occupied_clusters"] == 120
    assert abs(summary["eligible_compound_mass_in_occupied_clusters"] - 0.962368746463228) < 1e-15
    assert abs(summary["total_variation_selected_vs_eligible"] - 0.4749893478709552) < 1e-15
    assert abs(summary["jensen_shannon_divergence_nats"] - 0.17226944974115826) < 1e-15
    metrics = pd.read_csv(RESULT / "retrieval_metrics.csv")
    eligible = metrics[metrics.universe == "eligible_primary"].set_index("predictor")
    expected = {"constant": 0.03888706305173615, "count_only": 0.5135366526438606,
                "cluster_only": 0.27310476907581527, "count_plus_cluster": 0.5733133252859649}
    for predictor, value in expected.items():
        assert abs(eligible.loc[predictor, "average_precision"] - value) < 1e-15
    conclusion = json.loads((RESULT / "selection_conclusion.json").read_text())
    assert conclusion["detectably_better_than_structure"] is True
    assert conclusion["materially_better_than_structure"] is False
    assert abs(conclusion["combined_to_count_ap_ratio"] - 1.1164019594986136) < 1e-15
    assert conclusion["conditional_permutation_p"] == 1 / 2001
    sensitivity = pd.read_csv(RESULT / "retrieval_partition_sensitivity.csv")
    assert len(sensitivity) == 20 and sensitivity.n_selected.eq(3_775).all()
    assert sensitivity.groupby(["preprocessing", "k"]).size().eq(5).all()


def test_current_figure_reuses_frozen_coordinates_and_current_labels():
    current = pl.read_parquet(RESULT / "compound_cluster_umap.parquet")
    frozen = pl.read_parquet(HISTORICAL / "compound_cluster_umap.parquet")
    assert current.shape == frozen.shape == (115_721, 6)
    assert current["Metadata_JCP2022"].to_list() == frozen["Metadata_JCP2022"].to_list()
    np.testing.assert_array_equal(current.select("umap_1", "umap_2").to_numpy(),
                                  frozen.select("umap_1", "umap_2").to_numpy())
    assert int(current["selected"].sum()) == 3_775
    figure_provenance = json.loads((RESULT / "cluster_selection_figure_provenance.json").read_text())
    assert figure_provenance["counts"]["selected_treatments"] == 3_775
    assert "reused byte-for-byte" in figure_provenance["label_blind_contract"]
    with Image.open(RESULT / "cluster_selection_compound_map.png") as image:
        assert image.size == (4560, 1575); image.verify()
    assert (RESULT / "cluster_selection_compound_map.pdf").read_bytes().startswith(b"%PDF-")
    summary = pd.read_csv(RESULT / "cluster_selection_summary_table.csv").set_index("metric")
    assert summary.loc["selected_compounds", "value"] == 3_775
    assert "Broad occupancy" in summary.loc["eligible_mass_in_occupied_clusters", "qualification"]
    report = (RESULT / "REPORT.md").read_text()
    assert "3,776 compound identifiers: 3,775 treatments plus one negative control" in report
    assert "no model-rank claim for broader JUMP" in report


def test_reviewed_helper_marker_and_fit_artifacts_are_independently_pinned():
    assert Path(m.analysis.__file__).resolve() == m.ANALYSIS_SOURCE.resolve()
    reviewed = m.verify_reviewed_fit()
    assert (reviewed["analysis_helper"]["bytes"], reviewed["analysis_helper"]["sha256"]) == m.EXPECTED_ANALYSIS_SOURCE
    assert (reviewed["fit_completion"]["bytes"], reviewed["fit_completion"]["sha256"]) == m.EXPECTED_FIT_COMPLETION
    assert {name: (record["bytes"], record["sha256"]) for name, record in reviewed["fit_artifacts"].items()} == m.EXPECTED_FIT_ARTIFACTS

    original = m.EXPECTED_ANALYSIS_SOURCE
    try:
        m.EXPECTED_ANALYSIS_SOURCE = (original[0], "0" * 64)
        try: m.verify_reviewed_fit()
        except RuntimeError: pass
        else: raise AssertionError("analysis helper drift was accepted")
    finally:
        m.EXPECTED_ANALYSIS_SOURCE = original


def test_changing_fit_artifact_and_marker_together_still_fails_closed():
    with tempfile.TemporaryDirectory(dir=HERE) as directory:
        repository = Path(directory)
        shutil.copy2(m.ANALYSIS_SOURCE, repository / m.ANALYSIS_SOURCE.name)
        changed_root = repository / "historical"; fit = changed_root / "fit"
        shutil.copytree(HISTORICAL / "fit", fit)
        diagnostics = fit / "clustering_diagnostics.csv"
        diagnostics.write_bytes(diagnostics.read_bytes() + b"changed\n")
        marker = json.loads((fit / "fit_complete.json").read_text())
        for record in marker["artifacts"]:
            path = fit / record["path"]
            record["bytes"] = path.stat().st_size
            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (fit / "fit_complete.json").write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
        original = (m.REPOSITORY_ROOT, m.HISTORICAL_ROOT, m.ANALYSIS_SOURCE)
        try:
            m.REPOSITORY_ROOT = repository
            m.HISTORICAL_ROOT = changed_root
            m.ANALYSIS_SOURCE = repository / "analyze_cluster_representativeness.py"
            try: m.verify_reviewed_fit()
            except RuntimeError: pass
            else: raise AssertionError("coordinated fit-artifact/marker drift was accepted")
        finally:
            m.REPOSITORY_ROOT, m.HISTORICAL_ROOT, m.ANALYSIS_SOURCE = original


def test_output_records_reject_traversal_absolute_and_resolved_escapes():
    invalid = [
        {"path": "../outside", "path_scope": m.OUTPUT_SCOPE},
        {"path": str((RESULT.parent / "outside").resolve()), "path_scope": m.OUTPUT_SCOPE},
        {"path": "REPORT.md", "path_scope": "wrong_scope"},
    ]
    for record in invalid:
        try: m.resolve_output_record(record, RESULT)
        except RuntimeError: pass
        else: raise AssertionError(f"unsafe output record was accepted: {record}")
    with tempfile.TemporaryDirectory() as directory:
        parent = Path(directory); root = parent / "result"; outside = parent / "outside"
        root.mkdir(); outside.mkdir(); (outside / "file").write_text("outside")
        (root / "escape").symlink_to(outside, target_is_directory=True)
        record = {"path": "escape/file", "path_scope": m.OUTPUT_SCOPE}
        try: m.resolve_output_record(record, root)
        except RuntimeError: pass
        else: raise AssertionError("resolved symlink escape was accepted")


def test_current_runner_directly_refuses_existing_root():
    try: m.main(["--output-dir", str(RESULT)])
    except RuntimeError as error: assert "must be absent" in str(error)
    else: raise AssertionError("existing current-release root was accepted")


def run_all():
    tests = [value for key, value in sorted(globals().items()) if key.startswith("test_") and callable(value)]
    for test in tests: test()
    print(f"{len(tests)} current-release cluster tests passed")


if __name__ == "__main__":
    run_all()
