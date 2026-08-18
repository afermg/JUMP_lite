#!/usr/bin/env python3
"""Build the fail-closed two-panel explanation of Figure 3c."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT_ROOT = HERE.parent
DEFAULT_OUTPUT = HERE / "outputs" / "release_v1"
FAMILY_ORDER = ["cp_measure", "dinov2", "morphem", "openphenom", "subcell"]
FAMILY_LABEL = {"cp_measure": "cp_measure", "dinov2": "DINOv2", "morphem": "MorphEM", "openphenom": "OpenPhenom", "subcell": "SubCell"}

# Every byte read from a sibling release is pinned here.
INPUTS: dict[str, tuple[str, int, str]] = {
    "paired": ("paired_recipes/outputs/release_v1/results/paired_config_deltas.csv", 55488, "6490dfa4a2641a6b8e1a21230dc92df22282ab7209582b7b7caf40de0ef737be"),
    "pooled": ("paired_recipes/outputs/release_v1/results/pooled_summary.csv", 648, "092b6e87497cf3f95499dc266bbe6f41eef456442b841ed5e30fb12e1debec2b"),
    "family": ("paired_recipes/outputs/release_v1/results/family_summary.csv", 1742, "5a2088d20a0f4f53ba77de4cb1c9361bfcfef4f30dfeefc417f3bcf72993d63d"),
    "fixed": ("fixed_recipe_bootstrap/outputs/release_v1/results/mq_vs_d2e8.csv", 2590, "28019c50c85850b21a2ce162c4cfb36cf38952610db71d458442b87ad1e44612"),
    "fixed_recipes": ("fixed_recipe_bootstrap/outputs/release_v1/manifests/selected_recipes.csv", 28948, "666be4164d02158d899a9a23600fd01db0bbf84e2724b1825bbb14bd216e4b97"),
    "interaction": ("normalization_interactions/outputs/release_v1/paired_recipe_deltas.csv", 48944, "f6c710ea8c5449c37a8cb43a7d9ade2719e810012a1ec83f65e750db70ef4172"),
    "variation": ("normalization_interactions/outputs/release_v1/variation_decomposition.csv", 225, "53c9b4e1c22d90fe1755dea3faf1dfa5eeaa3f749a59fbb8de9f0d8349828ea1"),
    "signs": ("normalization_interactions/outputs/release_v1/recipe_sign_consistency.csv", 5009, "b39e32580a0a6c9aa13ae80caf738d390f26a7f45fa736cdeb02d248645f2a61"),
    "loo": ("plate_unit_influence/outputs/release_v1/leave_one_plate_out.csv", 5251, "f814a603de2726153942aa83c9ffa42955d54314715d4b3fed67d31deee08c3e"),
    "influence": ("plate_unit_influence/outputs/release_v1/influence_summary.csv", 958, "09eacf714e929fb2319e3d596c10d56dd3e7405f7053e578269e053ee95120b1"),
    "coverage": ("plate_unit_influence/outputs/release_v1/coverage_manifest.csv", 313, "93f5199b112ced5403b586cb7b666540347be5829254ebb5c23891dfd8ef293d"),
    "effort": ("effort_sensitivity/outputs/e3_vs_hq_bootstrap.csv", 2210, "35ccf05afd62c10cfc4ffa9ebb3a18045da859c38d2517da7ff58028a949cb05"),
    "quality": ("effort_sensitivity/outputs/image_quality_summary.csv", 281, "c02c6d8670f40d708dfd4169b97fe69ccb55b8c5a0ba15be00ee9483363910ea"),
    "features": ("effort_sensitivity/outputs/feature_correlation_summary.csv", 215, "a81b414db59ff99e30a3afa7877236454d3a3fe5a1b1a205ca2fdc3efee4d0b4"),
    "segmentation": ("effort_sensitivity/outputs/segmentation_common_site_summary.csv", 547, "d011ff3fa962d9f131a1b4b62af83250fb578dc411a1e59bf70e058803724f93"),
    "effort_recipes": ("effort_sensitivity/outputs/selected_recipes.csv", 22838, "b4a27d9d69e4af7ec65d5fc3326dbed7b8827045d1a06c86a9bac667a082e4db"),
}
EXPECTED_OUTPUTS = {"CAPTION.md", "REPORT.md", "figure_data.csv", "mq_d2e8_synthesis.pdf", "mq_d2e8_synthesis.png", "provenance.json"}

class AnalysisError(RuntimeError): pass

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20), b""): h.update(b)
    return h.hexdigest()

def verify_inputs(root: Path) -> tuple[dict[str, Path], list[dict[str, Any]]]:
    paths={}; records=[]
    for key,(rel,size,digest) in INPUTS.items():
        p=root/rel
        if not p.is_file() or p.is_symlink(): raise AnalysisError(f"missing or unsafe sibling input: {p}")
        got_size=p.stat().st_size; got_hash=sha256_file(p)
        if (got_size,got_hash)!=(size,digest): raise AnalysisError(f"sibling drift for {key}: {(got_size,got_hash)}")
        paths[key]=p; records.append({"key":key,"path":rel,"size_bytes":size,"sha256":digest})
    return paths,records

def load_data(paths: dict[str,Path]) -> dict[str,pd.DataFrame]:
    d={k:pd.read_csv(v) for k,v in paths.items()}
    if len(d["paired"])!=240 or d["paired"].family.nunique()!=5 or not d["paired"].groupby("family").size().eq(48).all(): raise AnalysisError("paired grid drift")
    if len(d["fixed"])!=5 or len(d["effort"])!=5: raise AnalysisError("forest coverage drift")
    if len(d["interaction"])!=240 or len(d["signs"])!=48: raise AnalysisError("interaction coverage drift")
    if len(d["loo"])!=25 or len(d["influence"])!=5: raise AnalysisError("influence coverage drift")
    if int(d["pooled"].iloc[0].n_paired_rows)!=240 or not bool(d["pooled"].iloc[0].pooled_median_inversion): raise AnalysisError("pooled summary drift")
    variation=dict(zip(d["variation"].component,d["variation"].fraction_of_total_variation))
    expected={"family":.507771,"recipe_structure":.103642,"family_by_recipe_residual":.388587}
    if any(abs(variation[k]-v)>5e-7 for k,v in expected.items()): raise AnalysisError("variation decomposition drift")
    if int(d["signs"].unanimous_sign.sum())!=1: raise AnalysisError("sign-consistency drift")
    if d["coverage"].dropped_for_common.sum()!=1: raise AnalysisError("common-population coverage drift")
    return d

def panel_label(ax,label:str)->None:
    ax.text(-.14,1.02,label,transform=ax.transAxes,fontweight="bold",fontsize=12,va="bottom")

def forest(ax:Any, frame:pd.DataFrame, point:str, low:str, high:str, support:str, xlabel:str, title:str)->None:
    frame=frame.set_index("family").loc[FAMILY_ORDER].reset_index(); y=np.arange(5)
    x=frame[point].to_numpy(); lo=x-frame[low].to_numpy(); hi=frame[high].to_numpy()-x
    supported=frame[support].astype(str).ne("unresolved").to_numpy()
    for i in range(5):
        color="#D55E00" if supported[i] else "#4C78A8"
        ax.errorbar(x[i],y[i],xerr=np.array([[lo[i]],[hi[i]]]),fmt="o",color=color,ecolor=color,capsize=2,lw=1.4,ms=4)
        if supported[i]: ax.text(frame[high].iloc[i]+.00025,y[i],"*",va="center",fontsize=9,color="#D55E00")
    ax.axvline(0,color="black",lw=.8); ax.set_yticks(y,[FAMILY_LABEL[f] for f in FAMILY_ORDER],fontsize=7); ax.invert_yaxis()
    ax.set_xlabel(xlabel,fontsize=7); ax.set_title(title,fontsize=8,fontweight="bold"); ax.tick_params(axis="x",labelsize=7); ax.grid(axis="x",alpha=.2)

def make_figure(d:dict[str,pd.DataFrame], png:Path, pdf:Path)->None:
    fig,axes=plt.subplots(1,2,figsize=(7.15,3.65),gridspec_kw={"width_ratios":[1.08,1.0]})
    fig.subplots_adjust(left=.09,right=.98,top=.76,bottom=.22,wspace=.40)

    # A: direct descriptive explanation of the pooled Figure 3c ordering.
    ax=axes[0]; p=d["paired"].copy(); order=[FAMILY_LABEL[f] for f in FAMILY_ORDER]
    canon={"cp_measure":"cp_measure","DINOv2":"DINOv2","MorphEM":"MorphEM","OpenPhenom":"OpenPhenom","SubCell":"SubCell"}
    p["display"]=p.family.map(canon)
    rng=np.random.default_rng(20260818)
    for i,family in enumerate(order):
        vals=p.loc[p.display==family,"delta_nap_product"].to_numpy(); jitter=rng.uniform(-.16,.16,len(vals))
        ax.scatter(i+jitter,vals,s=9,alpha=.40,color="#4C78A8",edgecolors="none")
        ax.plot([i-.22,i+.22],[np.median(vals)]*2,color="#D55E00",lw=2.2)
        ax.scatter(i,np.mean(vals),marker="D",s=20,color="black",zorder=3)
    pooled=d["pooled"].iloc[0]
    ax.axhline(0,color="black",lw=.8)
    ax.set_ylim(-.0215,.0092)
    ax.set_xticks(range(5),order,rotation=25,ha="right",fontsize=7.5)
    ax.tick_params(axis="y",labelsize=7.5)
    ax.set_ylabel("Matched NAP-product difference\n(MQ − D2-E8)",fontsize=8)
    ax.set_title("Matched MQ − D2-E8 contrasts",fontsize=9,fontweight="bold")
    ax.text(
        .02,.98,
        "Figure 3c pooled medians:\n"
        f"MQ {pooled.mq_marginal_product_median:.5f} > D2-E8 {pooled.d2e8_marginal_product_median:.5f}\n"
        f"Matched median {pooled.paired_product_median_delta:+.5f}; MQ higher in {pooled.paired_product_mq_greater_fraction*100:.0f}%",
        transform=ax.transAxes,va="top",fontsize=6.8,
        bbox={"facecolor":"white","edgecolor":"none","alpha":.82,"pad":1.5},
    )
    ax.text(.99,.02,"orange: median   ◆: mean",transform=ax.transAxes,ha="right",va="bottom",fontsize=6.4)
    panel_label(ax,"A")

    # B: inferential fixed-recipe sensitivity requested by reviewers.
    ax=axes[1]
    forest(
        ax,d["fixed"],"product_delta_mq_minus_d2e8","product_delta_ci_low",
        "product_delta_ci_high","product_supported_direction",
        "NAP-product difference (MQ − D2-E8)",
        "Fixed-recipe bootstrap contrasts\n50,000 draws; * = Holm-supported",
    )
    ax.text(.01,-.24,"D2-E8 higher  ←",transform=ax.transAxes,ha="left",fontsize=6.5)
    ax.text(.99,-.24,"→  MQ higher",transform=ax.transAxes,ha="right",fontsize=6.5)
    panel_label(ax,"B")

    fig.suptitle("Explaining Figure 3c",fontsize=12,fontweight="bold",y=.96)
    fig.text(.5,.875,"The pooled MQ median does not imply a consistent codec advantage",ha="center",fontsize=9)
    frozen_time=dt.datetime(2026,8,18,tzinfo=dt.timezone.utc)
    metadata={"Creator":"JUMP-lite mq_d2e8 synthesis","CreationDate":frozen_time,"ModDate":frozen_time}
    fig.savefig(pdf,metadata=metadata,bbox_inches="tight")
    fig.savefig(png,dpi=300,metadata={"Software":"JUMP-lite mq_d2e8 synthesis"},bbox_inches="tight")
    plt.close(fig)

def build_figure_data(d:dict[str,pd.DataFrame])->pd.DataFrame:
    rows=[]
    for _,r in d["paired"].iterrows():
        rows.append({"panel":"A","family":r.family,"unit":"matched_family_recipe","point":r.delta_nap_product,"low":np.nan,"high":np.nan,"detail":r.config})
    for _,r in d["fixed"].iterrows():
        rows.append({"panel":"B","family":r.family,"unit":"fixed_recipe_bootstrap","point":r.product_delta_mq_minus_d2e8,"low":r.product_delta_ci_low,"high":r.product_delta_ci_high,"detail":r.product_supported_direction})
    return pd.DataFrame(rows)

def caption_text()->str:
    return """# Supplementary Figure caption

**Explaining Figure 3c.** **(A)** MQ-minus-D2-E8 NAP-product differences for 240 exactly matched family–recipe cells (five families by 48 deterministic recipes). Orange bars show family medians and black diamonds show means; the pooled medians in Figure 3c are 0.02406 for MQ and 0.02250 for D2-E8. Recipe cells are sensitivity settings, not biological replicates. **(B)** MQ-minus-D2-E8 contrasts after selecting one recipe per family using Zstd alone and fixing it across codecs. Points and bars show estimates and pointwise 95% intervals from 50,000 shared-weight paired cluster-bootstrap draws over 306 PA compounds and 201 PC targets; the asterisk denotes support after Holm adjustment across five families. Only MorphEM supports D2-E8 over MQ. PA and PC margins were resampled independently, omitting their unknown covariance; the intervals are conditional, and non-support is not equivalence.
"""

def report_text(d:dict[str,pd.DataFrame])->str:
    p=d["pooled"].iloc[0]; fixed=d["fixed"].set_index("family")
    return f"""# Explaining Figure 3c

The apparent pooled inversion in Figure 3c is an aggregation result, not a general paired codec advantage. The pooled MQ and D2-E8 medians are {p.mq_marginal_product_median:.8f} and {p.d2e8_marginal_product_median:.8f}, while the median exact paired delta across 240 family/recipe cells is {p.paired_product_median_delta:+.8f}. Only MorphEM has a Holm-supported fixed-recipe product contrast, favoring D2-E8 ({fixed.loc['morphem','product_delta_mq_minus_d2e8']:+.5f}, pointwise 95% interval [{fixed.loc['morphem','product_delta_ci_low']:+.5f}, {fixed.loc['morphem','product_delta_ci_high']:+.5f}]).

The active figure presents the two analyses that most directly explain Figure 3c: exact matched recipe cells and fixed-recipe uncertainty. Supporting archived analyses remain part of this verified synthesis: normalization effects are family-dependent (only 1/48 aligned recipe structures has a unanimous sign), plate/laboratory omission changes several signs, and the HQ/E3 comparison cannot identify effort because the historical default is unpinned and the codec grid is not factorial. Together these results support small-cohort aggregation and analysis-pipeline sensitivity, not denoising or biological improvement.
"""

def write_json(path:Path,obj:Any)->None: path.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
def write_checksums(root:Path)->None:
    rows=[]
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.name!="artifact_checksums.json": rows.append({"path":p.relative_to(root).as_posix(),"size_bytes":p.stat().st_size,"sha256":sha256_file(p)})
    write_json(root/"artifact_checksums.json",{"artifacts":rows,"root":"."})
def verify_release(root:Path)->None:
    manifest=root/"artifact_checksums.json"
    if not manifest.is_file(): raise AnalysisError("missing artifact_checksums.json")
    obj=json.loads(manifest.read_text()); names={r["path"] for r in obj["artifacts"]}
    if names!=EXPECTED_OUTPUTS: raise AnalysisError(f"release inventory drift: {names ^ EXPECTED_OUTPUTS}")
    actual={p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual!=EXPECTED_OUTPUTS|{"artifact_checksums.json"}: raise AnalysisError("unexpected release files")
    for r in obj["artifacts"]:
        p=root/r["path"]
        if p.stat().st_size!=r["size_bytes"] or sha256_file(p)!=r["sha256"]: raise AnalysisError(f"release checksum mismatch: {p}")

def generate(input_root:Path,output:Path)->None:
    paths,records=verify_inputs(input_root); d=load_data(paths)
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mq-d2-synthesis-",dir=output.parent) as td:
        stage=Path(td)/"release_v1"; stage.mkdir()
        make_figure(d,stage/"mq_d2e8_synthesis.png",stage/"mq_d2e8_synthesis.pdf")
        build_figure_data(d).to_csv(stage/"figure_data.csv",index=False,float_format="%.12g",lineterminator="\n")
        (stage/"CAPTION.md").write_text(caption_text()); (stage/"REPORT.md").write_text(report_text(d))
        write_json(stage/"provenance.json",{"analysis":"mq_d2e8_synthesis","protocol_version":2,"active_panels":["A_matched_family_recipe","B_fixed_recipe_bootstrap"],"inputs":records,"canonical_datasets_read":False,"family_order":FAMILY_ORDER,"qualifications":["recipe cells are deterministic sensitivities, not biological replicates","plate and laboratory are confounded","PA and PC bootstrap margins use a working-independence approximation","historical default effort is unpinned and the codec grid is not factorial","no denoising or biological improvement is inferred"]})
        write_checksums(stage); verify_release(stage)
        backup=output.with_name(output.name+".backup")
        if backup.exists(): shutil.rmtree(backup)
        if output.exists(): os.replace(output,backup)
        try: os.replace(stage,output)
        except Exception:
            if backup.exists(): os.replace(backup,output)
            raise
        if backup.exists(): shutil.rmtree(backup)

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("--input-root",type=Path,default=DEFAULT_INPUT_ROOT); ap.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); ap.add_argument("--verify-only",action="store_true"); args=ap.parse_args()
    try:
        if args.verify_only: verify_inputs(args.input_root); verify_release(args.output); print("Verified pinned sibling inputs and synthesis release checksums.")
        else: generate(args.input_root,args.output); print(f"Generated {args.output}")
    except (AnalysisError,OSError,ValueError,KeyError) as e: print(f"ERROR: {e}"); return 2
    return 0
if __name__=="__main__": raise SystemExit(main())
