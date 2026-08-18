#!/usr/bin/env python3
"""Approximate Target-2 JPEG XL effort sensitivity from archived outputs only.

HQ and E3 both use distance=1 in the archived producer source, while HQ omits
``effort`` and E3 explicitly sets effort=3.  The numeric encoder default used
when the images were produced is not frozen, so this is an approximate effort
sensitivity, not a controlled distance-by-effort factorial.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SWEEP = Path("/work/datasets/JUMP-lite-wacv/sweeps/MAIN_RESULTS__figure_4_variance_first_v11")
SWEEP_CSV = SWEEP / "sweep_results.csv"
IMAGE_QUALITY = Path("/work/users/jfredinh/projects/cleaning-JUMP_CORE/analysis/image_quality/output/quality_metrics.csv")
FEATURE_CORR = Path("/work/users/jfredinh/projects/cleaning-JUMP_CORE/analysis/output/codec_feature_correlation.csv")
SEGMENTATION = Path("/work/users/jfredinh/projects/cleaning-JUMP_CORE/analysis/segmentation/output/segmentation_comparison/detailed_results")
COMPRESS_SOURCE = REPO / "src/compress_tif.py"

EXPECTED = {
    str(SWEEP_CSV): (1_096_114, "08923c7bd27bca54c0a3f484429ced31d1b48ad097c974773591a89ac63eb53a"),
    str(IMAGE_QUALITY): (224_818, "1bac02ffc9190dc8dab4ef2ec6a01dc2b02240b03211bfc7560bc1ab583483eb"),
    str(FEATURE_CORR): (120_194, "c9c433d9b01fb89d32a91bd86452af400775e3f5b7674deb49df2067ef489a62"),
}
FAMILIES = ("cp_measure", "dinov2", "morphem", "openphenom", "subcell")
PREFIX = {
    "cp_measure": "cp_measure", "dinov2": "dinov2", "morphem": "morphem",
    "openphenom": "openphenom", "subcell": "subcell__clip01",
}
MODEL = {
    "cp_measure": "", "dinov2": "dinov2_", "morphem": "morphem_",
    "openphenom": "openphenom_", "subcell": "subcell__clip01_",
}
DISPLAY = {
    "cp_measure": "cp_measure", "dinov2": "DINOv2", "morphem": "MorphEM",
    "openphenom": "OpenPhenom", "subcell": "SubCell",
}
CODECS = (("Zstd", "zstd"), ("HQ", "jpegxl_lossy_hq"), ("E3", "jpegxl_lossy_effort_3"))
SEED = 20_260_818
REPLICATES = 50_000

class AnalysisError(RuntimeError):
    pass

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def verify(path: Path, expected: tuple[int, str] | None = None) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise AnalysisError(f"missing/empty input: {path}")
    record = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
    if expected and (record["size_bytes"], record["sha256"]) != expected:
        raise AnalysisError(f"input drift: {path}")
    return record

def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)

def write_csv(path: Path, df: pd.DataFrame) -> None:
    if df.empty: raise AnalysisError(f"refusing empty output: {path}")
    atomic_text(path, df.to_csv(index=False, lineterminator="\n"))

def write_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")

def model_name(family: str, codec_folder: str) -> str:
    return f"{MODEL[family]}{codec_folder}_raw"

def select_recipes(sweep: pd.DataFrame) -> dict[str, str]:
    required = {"model", "config", "PA", "PC", "PA_mean_nap", "PC_mean_nap"}
    if missing := required - set(sweep): raise AnalysisError(f"missing sweep columns: {sorted(missing)}")
    out: dict[str, str] = {}
    for family in FAMILIES:
        rows = sweep[sweep.model == model_name(family, "zstd")].copy()
        if len(rows) != 48 or rows.config.duplicated().any():
            raise AnalysisError(f"expected 48 unique Zstd recipes for {family}, got {len(rows)}")
        rows["metric"] = rows.PA * rows.PC / 100.0
        best = rows.metric.max()
        out[family] = sorted(rows.loc[np.isclose(rows.metric, best, rtol=0, atol=1e-15), "config"].astype(str))[0]
    return out

def align(tables: list[pd.DataFrame], key: str, expected_n: int) -> pd.DataFrame:
    if not tables: raise AnalysisError("no alignment tables")
    result = tables[0]
    expected = set(result[key].astype(str))
    if len(result) != expected_n or result[key].duplicated().any(): raise AnalysisError(f"bad initial {key}")
    for table in tables[1:]:
        if len(table) != expected_n or table[key].duplicated().any() or set(table[key].astype(str)) != expected:
            raise AnalysisError(f"{key} key-set/duplicate drift")
        result = result.merge(table, on=key, how="inner", validate="one_to_one")
    return result.sort_values(key).reset_index(drop=True)

def load_fixed_inputs(sweep: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    recipes = select_recipes(sweep)
    selected: list[dict[str, Any]] = []
    pa_tables: list[pd.DataFrame] = []
    pc_tables: list[pd.DataFrame] = []
    inputs: list[dict[str, Any]] = []
    for family in FAMILIES:
        config = recipes[family]
        for label, folder_codec in CODECS:
            folder = SWEEP / f"{PREFIX[family]}_jump_target2_4plate_{folder_codec}_raw_features" / config
            metrics_p = folder / "results/metrics.json"
            pa_p = folder / "results/phenotypic_activity_map.csv"
            pc_p = folder / "results/phenotypic_consistency_per_target.csv"
            cfg_p = folder / "pipeline_config.yaml"
            out_p = folder / "output.parquet"
            recs = [verify(p) for p in (metrics_p, pa_p, pc_p, cfg_p, out_p)]
            inputs.extend(recs)
            metrics = json.loads(metrics_p.read_text())
            pa = pd.read_csv(pa_p)
            pc = pd.read_csv(pc_p)
            pa_key, pc_key = "Metadata_broad_sample", "Metadata_target"
            if pa_key not in pa or pc_key not in pc: raise AnalysisError(f"per-unit key absent: {folder}")
            if len(pa) != 306 or len(pc) != 201: raise AnalysisError(f"per-unit denominator drift: {folder}")
            pa_point = float(pa.mean_normalized_average_precision.mean())
            pc_point = float(pc.mean_normalized_average_precision.mean())
            if not math.isclose(pa_point, float(metrics["PA_mean_nap"]), abs_tol=1e-12): raise AnalysisError(f"PA point drift: {folder}")
            if not math.isclose(pc_point, float(metrics["PC_mean_nap"]), abs_tol=1e-12): raise AnalysisError(f"PC point drift: {folder}")
            row = sweep[(sweep.model == model_name(family, folder_codec)) & (sweep.config == config)]
            if len(row) != 1: raise AnalysisError(f"sweep row coverage drift: {folder}")
            col = f"{family}__{label}"
            pa_tables.append(pa[[pa_key, "mean_normalized_average_precision"]].rename(columns={"mean_normalized_average_precision": col}))
            pc_tables.append(pc[[pc_key, "mean_normalized_average_precision"]].rename(columns={"mean_normalized_average_precision": col}))
            selected.append({"family": family, "display_family": DISPLAY[family], "codec": label, "config": config,
                             "selection_metric": "PA*PC/100 on Zstd only", "pa_point": pa_point,
                             "pc_point": pc_point, "product_point": pa_point * pc_point,
                             "metrics_path": str(metrics_p), "metrics_sha256": recs[0]["sha256"],
                             "pa_path": str(pa_p), "pa_sha256": recs[1]["sha256"],
                             "pc_path": str(pc_p), "pc_sha256": recs[2]["sha256"],
                             "config_path": str(cfg_p), "config_sha256": recs[3]["sha256"],
                             "output_path": str(out_p), "output_sha256": recs[4]["sha256"]})
    return align(pa_tables, "Metadata_broad_sample", 306), align(pc_tables, "Metadata_target", 201), pd.DataFrame(selected), inputs

def interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(np.quantile(values, [0.025, 0.975]).tolist())

def centered_pvalue(values: np.ndarray, point: float) -> float:
    centered = values - point
    return min(1.0, 2 * min((np.count_nonzero(centered <= -abs(point)) + 1) / (len(values) + 1),
                            (np.count_nonzero(centered >= abs(point)) + 1) / (len(values) + 1)))

def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p); result = np.empty(len(p)); running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[idx])); result[idx] = running
    return result

def bootstrap(pa: pd.DataFrame, pc: pd.DataFrame, replicates: int = REPLICATES, seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    pcols = [c for c in pa if c != "Metadata_broad_sample"]
    if pcols != [c for c in pc if c != "Metadata_target"]: raise AnalysisError("PA/PC columns differ")
    a, c = pa[pcols].to_numpy(float), pc[pcols].to_numpy(float)
    if not np.isfinite(a).all() or not np.isfinite(c).all() or replicates <= 0: raise AnalysisError("invalid bootstrap input")
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    ab = np.empty((replicates, len(pcols))); cb = np.empty_like(ab)
    for start in range(0, replicates, 250):
        n = min(250, replicates-start)
        aw = rng.multinomial(len(a), np.full(len(a), 1/len(a)), size=n)
        cw = rng.multinomial(len(c), np.full(len(c), 1/len(c)), size=n)
        ab[start:start+n] = aw @ a / len(a); cb[start:start+n] = cw @ c / len(c)
    pb = ab * cb; ap = a.mean(0); cp = c.mean(0); pp = ap * cp
    summaries=[]; contrasts=[]
    for i,col in enumerate(pcols):
        family,codec=col.split("__"); al,ah=interval(ab[:,i]); cl,ch=interval(cb[:,i]); pl,ph=interval(pb[:,i])
        summaries.append({"family":family,"codec":codec,"pa_point":ap[i],"pa_ci_low":al,"pa_ci_high":ah,
                          "pc_point":cp[i],"pc_ci_low":cl,"pc_ci_high":ch,
                          "product_point":pp[i],"product_ci_low":pl,"product_ci_high":ph,
                          "replicates":replicates,"seed":seed})
    for family in FAMILIES:
        h=pcols.index(f"{family}__HQ"); e=pcols.index(f"{family}__E3")
        diff=pb[:,e]-pb[:,h]; lo,hi=interval(diff)
        row={"family":family,"display_family":DISPLAY[family],"hq_product":pp[h],"e3_product":pp[e],
             "product_delta_e3_minus_hq":pp[e]-pp[h],"product_delta_ci_low":lo,"product_delta_ci_high":hi,
             "product_centered_bootstrap_p":centered_pvalue(diff,pp[e]-pp[h])}
        for name,points,boots in (("pa",ap,ab),("pc",cp,cb)):
            d=boots[:,e]-boots[:,h]; dl,dh=interval(d)
            row.update({f"{name}_hq":points[h],f"{name}_e3":points[e],f"{name}_delta_e3_minus_hq":points[e]-points[h],f"{name}_delta_ci_low":dl,f"{name}_delta_ci_high":dh})
        contrasts.append(row)
    con=pd.DataFrame(contrasts); con["product_holm_p"]=holm(con.product_centered_bootstrap_p.to_numpy())
    con["supported_direction"]=np.where(con.product_holm_p<0.05,np.where(con.product_delta_e3_minus_hq>0,"E3>HQ","HQ>E3"),"unresolved")
    return pd.DataFrame(summaries),con

def evidence_summaries() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    records=[]
    quality=pd.read_csv(IMAGE_QUALITY); quality=quality[quality.codec.isin(["jpegxl_lossy_hq","jpegxl_lossy_effort_3"])]
    if len(quality)!=200 or quality.groupby("codec").site_name.nunique().ne(100).any(): raise AnalysisError("image-quality coverage drift")
    q=quality.groupby("codec").agg(n_sites=("site_name","nunique"),ssim_mean=("ssim","mean"),ssim_median=("ssim","median"),psnr_mean=("psnr","mean"),psnr_median=("psnr","median"),lpips_mean=("lpips","mean"),lpips_median=("lpips","median")).reset_index()
    fc=pd.read_csv(FEATURE_CORR)
    if len(fc)!=790 or fc.feature.duplicated().any() or fc[["hq","effort_3"]].isna().any().any(): raise AnalysisError("feature-correlation coverage drift")
    delta=fc.effort_3-fc.hq
    f=pd.DataFrame([{"n_features":len(fc),"hq_median_correlation":fc.hq.median(),"e3_median_correlation":fc.effort_3.median(),"e3_minus_hq_mean":delta.mean(),"e3_minus_hq_median":delta.median(),"fraction_e3_gt_hq":(delta>0).mean()}])
    segrows=[]
    for compartment in ("segment_cell","segment_nuclei"):
        tables={}
        for label,codec in (("HQ","jpegxl_lossy_hq"),("E3","jpegxl_lossy_effort_3")):
            p=SEGMENTATION/f"{compartment}_{codec}.csv"; records.append(verify(p)); d=pd.read_csv(p)
            if len(d)!=9216 or d.source_id.duplicated().any(): raise AnalysisError(f"segmentation coverage drift: {p}")
            tables[label]=d.set_index("source_id")
        if set(tables["HQ"].index)!=set(tables["E3"].index): raise AnalysisError("segmentation common-site mismatch")
        for metric in ("inst_f1_50","iou"):
            delta=tables["E3"].loc[sorted(tables["HQ"].index),metric]-tables["HQ"].loc[sorted(tables["HQ"].index),metric]
            segrows.append({"compartment":compartment.removeprefix("segment_"),"metric":metric,"n_common_sites":len(delta),"hq_mean":tables["HQ"][metric].mean(),"e3_mean":tables["E3"][metric].mean(),"e3_minus_hq_mean":delta.mean(),"e3_minus_hq_median":delta.median(),"fraction_e3_gt_hq":(delta>0).mean()})
    return q,f,pd.DataFrame(segrows),records

def plot_panel(contrasts: pd.DataFrame, q: pd.DataFrame, f: pd.DataFrame, seg: pd.DataFrame, out: Path) -> None:
    frame=contrasts.set_index("family").loc[list(FAMILIES)].reset_index(); y=np.arange(len(frame))
    fig,ax=plt.subplots(figsize=(7.2,4.6)); x=frame.product_delta_e3_minus_hq.to_numpy(); lo=x-frame.product_delta_ci_low.to_numpy(); hi=frame.product_delta_ci_high.to_numpy()-x
    colors=np.where(frame.product_holm_p<0.05,"#D55E00","#666666")
    for i in range(len(frame)):
        ax.errorbar(x[i],y[i],xerr=np.array([[lo[i]],[hi[i]]]),fmt="o",color=colors[i],
                    ecolor=colors[i],elinewidth=2,capsize=3,markersize=6,zorder=2)
    ax.axvline(0,color="black",lw=.9)
    ax.set_yticks(y,frame.display_family); ax.invert_yaxis(); ax.set_xlabel("E3 − HQ PA–PC NAP product")
    ax.set_title("Approximate fixed-distance effort sensitivity",fontweight="bold")
    for i,row in frame.iterrows(): ax.text(max(frame.product_delta_ci_high)*1.04,i,f"$p_{{Holm}}$={row.product_holm_p:.3g}",va="center",fontsize=8)
    ax.grid(axis="x",alpha=.2)
    fig.subplots_adjust(left=.20,right=.84,top=.88,bottom=.28)
    fig.text(.20,.035,f"Pixel fidelity (median): HQ SSIM {q.loc[q.codec=='jpegxl_lossy_hq','ssim_median'].iloc[0]:.6f}; E3 {q.loc[q.codec=='jpegxl_lossy_effort_3','ssim_median'].iloc[0]:.6f}\ncp_measure features: E3 > HQ for {100*f.fraction_e3_gt_hq.iloc[0]:.1f}% of 790; segmentation uses 9,216 common sites per compartment",fontsize=8,va="bottom")
    fig.savefig(out.with_suffix(".png"),dpi=220,bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"),bbox_inches="tight",metadata={"CreationDate":None,"ModDate":None}); plt.close(fig)

def main() -> None:
    out=HERE/"outputs"; out.mkdir(parents=True,exist_ok=True)
    base_inputs=[verify(p,EXPECTED[str(p)]) for p in (SWEEP_CSV,IMAGE_QUALITY,FEATURE_CORR)]
    source_record=verify(COMPRESS_SOURCE)
    source=COMPRESS_SOURCE.read_text()
    if '"jpegxl_lossy_hq": Jpegxl(lossless=False, distance=1.0)' not in source or '"jpegxl_lossy_effort_3": Jpegxl(lossless=False, distance=1.0, effort=3)' not in source:
        raise AnalysisError("producer codec declarations drift")
    sweep=pd.read_csv(SWEEP_CSV); pa,pc,selected,fixed_inputs=load_fixed_inputs(sweep)
    summary,contrast=bootstrap(pa,pc); quality,features,seg,extra_inputs=evidence_summaries()
    write_csv(out/"selected_recipes.csv",selected); write_csv(out/"score_intervals.csv",summary); write_csv(out/"e3_vs_hq_bootstrap.csv",contrast)
    write_csv(out/"image_quality_summary.csv",quality); write_csv(out/"feature_correlation_summary.csv",features); write_csv(out/"segmentation_common_site_summary.csv",seg)
    plot_panel(contrast,quality,features,seg,out/"effort_sensitivity")
    provenance={"protocol":"effort_sensitivity_v1","seed":SEED,"bootstrap_replicates":REPLICATES,
                "qualification":"Approximate fixed-distance sensitivity: HQ omitted effort in producer source; E3 set effort=3. Numeric historical default and encoder build are not frozen. PA and PC margins are independently resampled under a working-independence approximation.",
                "codec_grid":"Not a distance-by-effort factorial; cannot isolate effort for D2-E8 versus MQ.",
                "producer_source":source_record,"current_lock_imagecodecs":"2026.1.14 (current lock only; not historical producer proof)",
                "software":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__},
                "inputs":base_inputs+fixed_inputs+extra_inputs}
    write_json(out/"provenance.json",provenance)
    lines=["# Approximate effort sensitivity","","HQ and E3 both use JPEG XL distance 1 in the archived producer source. HQ omitted the effort argument while E3 set effort 3. The historical numeric default and encoder build are not frozen, so this is an approximate sensitivity—not a controlled effort experiment.","","## Fixed-recipe paired bootstrap","","| Family | E3 − HQ product (95% interval) | Holm result |","|---|---:|---|"]
    for _,r in contrast.iterrows(): lines.append(f"| {r.display_family} | {r.product_delta_e3_minus_hq:+.5f} [{r.product_delta_ci_low:+.5f}, {r.product_delta_ci_high:+.5f}] | {r.supported_direction} ($p_{{Holm}}={r.product_holm_p:.4g}$) |")
    lines += ["","One Zstd-selected recipe was frozen across codecs. PA (306 compound clusters) and PC (201 target clusters) were independently resampled 50,000 times with shared weights across model/codec columns. Product intervals omit unknown PA–PC covariance.","","## Supporting archived evidence",f"","- Pixel metrics use 100 matched sites per codec.",f"- HQ/E3 median SSIM: {quality.loc[quality.codec=='jpegxl_lossy_hq','ssim_median'].iloc[0]:.6f}/{quality.loc[quality.codec=='jpegxl_lossy_effort_3','ssim_median'].iloc[0]:.6f}.",f"- E3 exceeds HQ correlation for {100*features.fraction_e3_gt_hq.iloc[0]:.1f}% of 790 cp_measure features.","- Cell and nuclei segmentation comparisons use the exact 9,216-site common set.","","## Limitation","","The available codecs are not a distance-by-effort factorial: HQ=(D1, default/omitted effort), E3=(D1,E3), D2-E8=(D2,E8), and MQ=(D3, default/omitted effort). This analysis cannot attribute D2-E8 versus MQ behavior to effort.",""]
    atomic_text(out/"REPORT.md","\n".join(lines))
    artifacts=[]
    for p in sorted(out.iterdir()):
        if p.name=="artifact_checksums.json": continue
        artifacts.append({"path":p.name,"size_bytes":p.stat().st_size,"sha256":sha256(p)})
    write_json(out/"artifact_checksums.json",{"artifacts":artifacts})
    print(contrast[["display_family","product_delta_e3_minus_hq","product_delta_ci_low","product_delta_ci_high","product_holm_p","supported_direction"]].to_string(index=False))

if __name__=="__main__": main()
