from __future__ import annotations

import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score

HERE=Path(__file__).resolve().parent
PATH=HERE/"analyze_cluster_representativeness.py"
SPEC=importlib.util.spec_from_file_location("cluster_analysis",PATH); assert SPEC and SPEC.loader
m=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)
SENS_PATH=HERE/"score_cluster_partition_sensitivity.py"
SENS_SPEC=importlib.util.spec_from_file_location("cluster_sensitivity",SENS_PATH); assert SENS_SPEC and SENS_SPEC.loader
s=importlib.util.module_from_spec(SENS_SPEC); SENS_SPEC.loader.exec_module(s)
OUT=HERE/"outputs/profile_cluster_representativeness_v1"


def test_frozen_input_identities_and_counts():
    for p in (m.CONSENSUS,m.FEATURES,m.SELECTED): m.verify(p)
    c=pl.read_parquet(m.CONSENSUS,columns=["Metadata_JCP2022","n_wells"])
    assert c.height==115721 and c["Metadata_JCP2022"].n_unique()==115721
    assert int((c["n_wells"]>=4).sum())==95426
    s=pl.read_parquet(m.SELECTED,columns=["Metadata_JCP2022"])
    assert s.height==s["Metadata_JCP2022"].n_unique()==3832


def test_deterministic_folds_and_strata():
    assert m.stable_fold("JCP2022_000001")==m.stable_fold("JCP2022_000001")
    assert 0<=m.stable_fold("x")<5
    assert m.structural_stratum(3,1)=="ineligible_lt4"
    assert m.structural_stratum(4,1)=="w4_7_single"
    assert m.structural_stratum(7,2)=="w4_7_multi"
    assert m.structural_stratum(8,1)=="w8plus"


def test_canonical_relabel_is_deterministic():
    centers=np.array([[2.,0.],[-1.,4.],[2.,-1.]])
    labels=np.array([0,1,2,0])
    a,c,map_=m.canonical_labels(labels,centers)
    b,d,_=m.canonical_labels(labels,centers)
    np.testing.assert_array_equal(a,b); np.testing.assert_array_equal(c,d)
    assert sorted(set(a))==[0,1,2]


def test_oof_cluster_probability_has_no_self_label_leakage():
    n=20
    df=pd.DataFrame({"fold_id":[i%5 for i in range(n)],"structural_stratum":["w8plus"]*n,
        "cluster_id":np.arange(n),"n_wells":[8]*n,"n_sources":[1]*n,"n_plates":[8]*n})
    y=np.array([0,1]*10)
    scores=m.crossfit_scores(df,y)
    np.testing.assert_allclose(scores["cluster_only"],.5)
    assert np.isfinite(scores["count_plus_cluster"]).all()


def test_constant_average_precision_equals_prevalence():
    y=np.array([1,0,0,1,0,0,0,0])
    score=np.full(len(y),y.mean())
    assert abs(average_precision_score(y,score)-y.mean())<1e-12


def test_bh_adjust_is_bounded_and_monotone_by_rank():
    p=np.array([.04,.001,.2,.02]); q=m.bh_adjust(p)
    assert np.all((q>=p)&(q<=1))
    order=np.argsort(p); assert np.all(np.diff(q[order])>=-1e-12)


def test_write_new_refuses_overwrite():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); p=root/"x.json"; m.write_json_new(p,{"x":1})
        try: m.write_json_new(p,{"x":2})
        except RuntimeError: pass
        else: raise AssertionError("overwrite was accepted")
        (root/"output_hashes.json").write_text("{}")
        try: m.assert_score_outputs_absent(root)
        except RuntimeError: pass
        else: raise AssertionError("completed score output was accepted")


def make_frozen_fit(root: Path) -> list[Path]:
    fit = root / "fit"; fit.mkdir(parents=True)
    artifacts=[]
    for name in sorted(m.FIT_ARTIFACT_NAMES):
        path=fit/name; path.write_bytes((name+"\n").encode()); artifacts.append(path)
    (fit/"fit_complete.json").write_text(json.dumps({
        "status":"complete",
        "artifacts":[m.record(path, base=fit, scope=m.FIT_SCOPE) for path in artifacts],
    }))
    return artifacts


def test_score_root_requires_only_frozen_fit():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); make_frozen_fit(root)
        m.verify_frozen_fit(root); m.assert_score_root_pristine(root)
        (root/"REPORT.md").write_text("partial")
        m.verify_frozen_fit(root)
        try: m.assert_score_root_pristine(root)
        except RuntimeError: pass
        else: raise AssertionError("partial score root was accepted")


def test_frozen_fit_records_survive_relocation():
    with tempfile.TemporaryDirectory() as td:
        original=Path(td)/"original"; relocated=Path(td)/"relocated"
        make_frozen_fit(original)
        shutil.move(original,relocated)
        completion=m.verify_frozen_fit(relocated)
        assert {record["path_scope"] for record in completion["artifacts"]}=={m.FIT_SCOPE}
        assert all(not Path(record["path"]).is_absolute() for record in completion["artifacts"])


def test_frozen_fit_rejects_absolute_and_escape_records():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); artifacts=make_frozen_fit(root); marker=root/"fit/fit_complete.json"
        pristine=json.loads(marker.read_text())
        for invalid in (str(artifacts[0].resolve()), "../outside"):
            changed=json.loads(json.dumps(pristine)); changed["artifacts"][0]["path"]=invalid
            marker.write_text(json.dumps(changed))
            try: m.verify_frozen_fit(root)
            except RuntimeError: pass
            else: raise AssertionError(f"unsafe frozen-fit path was accepted: {invalid}")
        marker.write_text(json.dumps(pristine))
        changed=json.loads(marker.read_text()); changed["artifacts"][0]["path_scope"]="unknown"
        marker.write_text(json.dumps(changed))
        try: m.verify_frozen_fit(root)
        except RuntimeError: pass
        else: raise AssertionError("unknown frozen-fit path scope was accepted")


def test_sensitivity_rejects_changed_selected_manifest():
    with tempfile.TemporaryDirectory() as td:
        changed=Path(td)/"selected.parquet"
        pl.DataFrame({"Metadata_JCP2022":["JCP2022_changed"]}).write_parquet(changed)
        try: s.verify_selected_manifest(changed)
        except RuntimeError: pass
        else: raise AssertionError("changed selected manifest was accepted")


def test_frozen_fit_rejects_changed_artifact():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); artifacts=make_frozen_fit(root); m.verify_frozen_fit(root)
        artifacts[0].write_text("changed")
        try: m.verify_frozen_fit(root)
        except RuntimeError: pass
        else: raise AssertionError("changed frozen fit artifact was accepted")


def test_sensitivity_refuses_partial_output():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); root.mkdir(exist_ok=True)
        partial=(root/s.TARGET_NAME).with_name(s.TARGET_NAME+".partial"); partial.write_text("partial")
        try: s.assert_target_absent(root)
        except RuntimeError: pass
        else: raise AssertionError("partial sensitivity output was accepted")


def test_completed_outputs_and_model_portability():
    if not (OUT/"output_hashes.json").exists(): return
    a=pl.read_parquet(OUT/"compound_cluster_assignments.parquet")
    assert a.height==a["Metadata_JCP2022"].n_unique()==115721
    assert int(a["selected"].sum())==3832 and a.filter(pl.col("selected"))["fit_eligible"].all()
    t=pl.read_parquet(OUT/"cluster_selection_table.parquet")
    assert t.height==128 and int(t["n_compounds"].sum())==115721 and int(t["n_selected"].sum())==3832
    model=np.load(OUT/"fit/primary_model.npz")
    names=model["feature_names"].tolist(); source=pl.read_parquet(m.CONSENSUS).head(100)
    x=source.select(names).to_numpy().astype(float)
    x=np.where(np.isfinite(x),x,model["imputation_medians"])
    x=np.clip((x-model["scaling_medians"])/model["scaling_iqrs"],-float(model["clip_limit"][0]),float(model["clip_limit"][0]))
    coords=(x-model["pca_mean"])@model["pca_components"].T
    labels=np.argmin(np.sum((coords[:,None,:]-model["centroids"][None,:,:])**2,axis=2),axis=1)
    expected=a.head(100)["cluster_id"].to_numpy(); np.testing.assert_array_equal(labels,expected)
    sensitivity=pd.read_csv(OUT/"retrieval_partition_sensitivity.csv")
    assert len(sensitivity)==20
    assert set(map(tuple,sensitivity[["preprocessing","k"]].drop_duplicates().to_numpy()))=={
        ("clip10",64),("clip10",128),("clip10",256),("rank_gaussian",128)}
    assert sensitivity.groupby(["preprocessing","k"]).size().eq(5).all()
    metrics=pd.read_csv(OUT/"retrieval_metrics.csv")
    for u in ("eligible_primary","all_descriptive"):
        rows=metrics[metrics.universe==u]
        assert set(rows.predictor)=={"constant","count_only","cluster_only","count_plus_cluster"}
        assert rows.n.nunique()==1
    selected=set(pl.read_parquet(m.SELECTED)["Metadata_JCP2022"].to_list())
    evaluation=set(pl.read_parquet(m.EVALUATION)["Metadata_JCP2022"].to_list())
    assert not selected&evaluation
    for path in m.MANIFESTS.glob("matched_random_seed_*.parquet"):
        ids=pl.read_parquet(path)["Metadata_JCP2022"].to_list()
        assert len(ids)==len(set(ids))==3832 and not selected&set(ids) and not set(ids)&evaluation
    hashes=json.loads((OUT/"output_hashes.json").read_text())
    for rec in hashes["files"]:
        local=m.resolve_record_path(rec,bases={m.OUTPUT_SCOPE:OUT})
        actual=m.record(local,base=OUT,scope=m.OUTPUT_SCOPE)
        assert actual==rec


def run_all():
    tests=[v for k,v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests: test()
    print(f"{len(tests)} cluster tests passed")

if __name__=="__main__": run_all()
