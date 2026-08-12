#!/usr/bin/env python3
"""Score every frozen label-blind partition without refitting it."""
from __future__ import annotations
import argparse, hashlib, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

HERE=Path(__file__).resolve().parent
try:
 import analyze_cluster_representativeness as cluster_analysis
except ModuleNotFoundError:
 spec=importlib.util.spec_from_file_location(
  "analyze_cluster_representativeness", HERE/"analyze_cluster_representativeness.py"
 )
 if spec is None or spec.loader is None: raise
 cluster_analysis=importlib.util.module_from_spec(spec); spec.loader.exec_module(cluster_analysis)

DEFAULT=HERE/"outputs/profile_cluster_representativeness_v1"
SELECTED = cluster_analysis.SELECTED
EXPECTED_SELECTED_BYTES, EXPECTED_SELECTED_SHA256 = cluster_analysis.EXPECTED[SELECTED]
TARGET_NAME = "retrieval_partition_sensitivity.csv"


def verify_selected_manifest(path: Path = SELECTED) -> dict[str, object]:
 rec = cluster_analysis.record(path)
 if (rec["bytes"], rec["sha256"]) != (EXPECTED_SELECTED_BYTES, EXPECTED_SELECTED_SHA256):
  raise RuntimeError(f"Frozen selected-manifest drift: {rec}")
 ids = pl.read_parquet(path, columns=["Metadata_JCP2022"])
 if ids.height != 3832 or ids["Metadata_JCP2022"].n_unique() != 3832:
  raise RuntimeError("Frozen selected-manifest count/uniqueness drift")
 return rec


def assert_target_absent(out: Path) -> Path:
 target = out / TARGET_NAME
 partial = target.with_name(target.name + ".partial")
 if target.exists() or partial.exists():
  raise RuntimeError(f"Refusing completed/partial sensitivity output: {target}")
 return target


def main():
 p=argparse.ArgumentParser(); p.add_argument("--output-dir",type=Path,default=DEFAULT); a=p.parse_args(); out=a.output_dir.resolve(strict=True)
 cluster_analysis.verify_frozen_fit(out)
 verify_selected_manifest()
 target=assert_target_absent(out)
 base=pl.read_parquet(out/"fit/cluster_assignments_unlabeled.parquet",columns=["Metadata_JCP2022","fit_eligible","fold_id"])
 selected=set(pl.read_parquet(SELECTED,columns=["Metadata_JCP2022"])["Metadata_JCP2022"].to_list())
 base=base.with_columns(pl.col("Metadata_JCP2022").is_in(selected).cast(pl.Int8).alias("selected"))
 sens=pl.read_parquet(out/"fit/cluster_assignments_sensitivity_unlabeled.parquet",columns=["Metadata_JCP2022","preprocessing","k","seed","cluster_id"])
 rows=[]
 for key,g in sens.group_by(["preprocessing","k","seed"],maintain_order=True):
  prep,k,seed=key
  x=g.join(base,on="Metadata_JCP2022",how="inner",validate="1:1").filter(pl.col("fit_eligible"))
  if x.height!=95426 or int(x["selected"].sum())!=3832: raise RuntimeError("Sensitivity join/count drift")
  y=x["selected"].to_numpy(); c=x["cluster_id"].to_numpy(); folds=x["fold_id"].to_numpy(); score=np.empty(len(x),float)
  for fold in range(5):
   test=folds==fold; train=~test
   n=np.bincount(c[train],minlength=int(k)); pos=np.bincount(c[train],weights=y[train],minlength=int(k))
   score[test]=(pos[c[test]]+.5)/(n[c[test]]+1)
  prev=y.mean(); pscore=np.clip(score,1e-9,1-1e-9)
  total=np.bincount(c,minlength=int(k)); sel=np.bincount(c,weights=y,minlength=int(k)); ps=sel/sel.sum(); pe=total/total.sum(); mid=.5*(ps+pe)
  nzs=ps>0; nze=pe>0; js=.5*np.sum(ps[nzs]*np.log(ps[nzs]/mid[nzs]))+.5*np.sum(pe[nze]*np.log(pe[nze]/mid[nze]))
  rows.append({"preprocessing":prep,"k":int(k),"seed":int(seed),"n":len(y),"n_selected":int(y.sum()),"prevalence":prev,
   "roc_auc":roc_auc_score(y,pscore),"average_precision":average_precision_score(y,pscore),"ap_lift":average_precision_score(y,pscore)/prev,
   "log_loss":log_loss(y,pscore,labels=[0,1]),"brier":brier_score_loss(y,pscore),"selected_occupied_clusters":int((sel>0).sum()),
   "eligible_mass_in_occupied_clusters":float(total[sel>0].sum()/total.sum()),"total_variation":float(.5*np.abs(ps-pe).sum()),"jensen_shannon_nats":float(js)})
 pd.DataFrame(rows).sort_values(["preprocessing","k","seed"]).to_csv(target,index=False)
 print(pd.DataFrame(rows).groupby(["preprocessing","k"])[["average_precision","roc_auc","log_loss","total_variation"]].agg(["min","median","max"]).to_string())
 return 0
if __name__=="__main__": raise SystemExit(main())
