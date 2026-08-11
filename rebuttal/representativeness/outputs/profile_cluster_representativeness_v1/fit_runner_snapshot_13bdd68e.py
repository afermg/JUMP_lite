#!/usr/bin/env python3
"""Label-blind clustering followed by frozen JUMP-lite selection scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from itertools import combinations
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import scipy
from scipy import sparse
from scipy.stats import beta, rankdata
from sklearn import __version__ as sklearn_version
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             average_precision_score, brier_score_loss,
                             log_loss, roc_auc_score, silhouette_score)
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

ROOT = Path(__file__).resolve().parent
PROFILE_ROOT = ROOT / "outputs/profile_space"
CONSENSUS = PROFILE_ROOT / "compound_consensus_plate_robust.parquet"
FEATURES = PROFILE_ROOT / "selected_features.txt"
MANIFESTS = PROFILE_ROOT / "manifests"
SELECTED = MANIFESTS / "jump_lite_compounds.parquet"
EVALUATION = MANIFESTS / "fixed_evaluation_compounds.parquet"
DESIGN = ROOT / "CLUSTER_SELECTION_DESIGN.md"
DEFAULT_OUT = ROOT / "outputs/profile_cluster_representativeness_v1"
EXPECTED = {
    CONSENSUS: (40_554_151, "dc2f84178a15f2e18177d4475b094af0da8fab10b1856bd3d1e4f6521d6c9d06"),
    FEATURES: (3_847, "fb05b7454feca3afe6677c73465adcb596481e7b32707eb52d9e21f6d6d42601"),
    SELECTED: (33_252, "a0671dcaae029a2c32ac58fdaf09178806b495d33a5ea439ff859b3c0fbe74de"),
}
SEEDS = (13, 42, 2026, 31415, 65537)
PCA_SEED = 2026
PCA_COMPONENTS = 32
PRIMARY_K = 128
BATCH_SIZE = 4096
N_INIT = 5
SILHOUETTE_N = 5000
PERMUTATIONS = 2000


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(block_size): h.update(block)
    return h.hexdigest()


def record(path: Path) -> dict[str, object]:
    p = path.resolve(strict=True)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}


def verify(path: Path) -> dict[str, object]:
    expected = EXPECTED.get(path)
    rec = record(path)
    if expected and (rec["bytes"], rec["sha256"]) != expected:
        raise RuntimeError(f"Input identity drift: {path}: {rec}")
    return rec


def write_json_new(path: Path, value: object) -> None:
    if path.exists(): raise RuntimeError(f"Refusing overwrite: {path}")
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def stable_fold(value: str) -> int:
    return int(hashlib.sha256(f"cluster-selection-fold-v1::{value}".encode()).hexdigest()[:16], 16) % 5


def structural_stratum(n_wells: int, n_sources: int) -> str:
    if n_wells < 4: return "ineligible_lt4"
    if n_wells >= 8: return "w8plus"
    return "w4_7_single" if n_sources == 1 else "w4_7_multi"


def stable_preprocess(values: np.ndarray, eligible: np.ndarray, clip: float = 10.0):
    x = np.asarray(values, dtype=np.float64).copy(); x[~np.isfinite(x)] = np.nan
    fit = x[eligible]
    finite = np.isfinite(fit).mean(axis=0)
    med = np.nanmedian(fit, axis=0)
    q25, q75 = np.nanquantile(fit, .25, axis=0), np.nanquantile(fit, .75, axis=0)
    iqr = q75 - q25
    keep = (finite >= .95) & np.isfinite(med) & np.isfinite(iqr) & (iqr > 0)
    if keep.sum() != 94: raise RuntimeError(f"Expected 94 stable features, got {keep.sum()}")
    raw = np.where(np.isfinite(x[:, keep]), x[:, keep], med[keep])
    scaled = np.clip((raw - med[keep]) / iqr[keep], -clip, clip)
    return scaled.astype(np.float32), keep, med[keep], iqr[keep]


def canonical_labels(labels: np.ndarray, centers: np.ndarray):
    order = np.lexsort(tuple(centers[:, i] for i in reversed(range(centers.shape[1]))))
    mapping = np.empty(len(order), dtype=np.int32); mapping[order] = np.arange(len(order), dtype=np.int32)
    return mapping[labels], centers[order], mapping


def distances_margin(x: np.ndarray, centers: np.ndarray, chunk: int = 4096):
    labels = np.empty(len(x), np.int32); nearest = np.empty(len(x), np.float32); margin = np.empty(len(x), np.float32)
    for start in range(0, len(x), chunk):
        part = x[start:start+chunk]
        d2 = np.sum((part[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        two = np.partition(d2, 1, axis=1)[:, :2]; two.sort(axis=1)
        labels[start:start+len(part)] = np.argmin(d2, axis=1)
        nearest[start:start+len(part)] = np.sqrt(two[:, 0])
        margin[start:start+len(part)] = np.sqrt(two[:, 1]) - np.sqrt(two[:, 0])
    return labels, nearest, margin


def fit_partition(coords: np.ndarray, eligible: np.ndarray, k: int, seed: int):
    model = MiniBatchKMeans(n_clusters=k, batch_size=BATCH_SIZE, n_init=N_INIT,
                            random_state=seed, max_iter=200, reassignment_ratio=.01)
    model.fit(coords[eligible])
    raw, dist, margin = distances_margin(coords, model.cluster_centers_.astype(np.float32))
    labels, centers, mapping = canonical_labels(raw, model.cluster_centers_.astype(np.float32))
    return labels, dist, margin, centers, float(model.inertia_), int(model.n_iter_)


def fit_clusters(args: argparse.Namespace) -> int:
    started = time.perf_counter(); out = args.output_dir.resolve(strict=False)
    if out.exists(): raise RuntimeError(f"Output root must be absent: {out}")
    verify(CONSENSUS); verify(FEATURES)
    if not DESIGN.is_file(): raise RuntimeError("Preregistered design is required")
    out.mkdir(parents=True); fitdir = out / "fit"; fitdir.mkdir()
    feature_names = [x for x in FEATURES.read_text().splitlines() if x]
    if len(feature_names) != 96: raise RuntimeError("Expected 96 requested features")
    frame = pl.read_parquet(CONSENSUS)
    if frame.height != 115_721 or frame["Metadata_JCP2022"].n_unique() != 115_721:
        raise RuntimeError("Consensus count/uniqueness drift")
    eligible = frame["n_wells"].to_numpy() >= 4
    if eligible.sum() != 95_426 or (~eligible).sum() != 20_295: raise RuntimeError("Eligible count drift")
    values = frame.select(feature_names).to_numpy()
    clipped, keep, impute, iqr = stable_preprocess(values, eligible)
    kept = np.asarray(feature_names, dtype="U")[keep]
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", random_state=PCA_SEED)
    coords = pca.fit_transform(clipped[eligible])
    all_coords = pca.transform(clipped).astype(np.float32)
    hash_order = np.argsort([hashlib.sha256(f"silhouette-v1::{v}".encode()).hexdigest()
                             for v in frame["Metadata_JCP2022"].to_list()])
    silhouette_rows = hash_order[:SILHOUETTE_N]

    diagnostics=[]; sens_frames=[]; primary_runs=[]
    clip_coords = all_coords
    configs=[("clip10",64,clip_coords),("clip10",128,clip_coords),("clip10",256,clip_coords)]
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=None, random_state=PCA_SEED)
    raw_kept = np.where(np.isfinite(values[:, keep]), values[:, keep], impute)
    rank_eligible = qt.fit_transform(raw_kept[eligible])
    rank_pca = PCA(n_components=PCA_COMPONENTS, svd_solver="randomized", random_state=PCA_SEED)
    rank_pca.fit(rank_eligible); rank_coords = rank_pca.transform(qt.transform(raw_kept)).astype(np.float32)
    configs.append(("rank_gaussian",128,rank_coords))
    labels_by_config: dict[tuple[str,int], list[np.ndarray]] = {}
    for prep,k,coord in configs:
        labels_by_config[(prep,k)] = []
        for seed in SEEDS:
            labels,dist,margin,centers,inertia,n_iter = fit_partition(coord, eligible, k, seed)
            labels_by_config[(prep,k)].append(labels)
            sizes=np.bincount(labels[eligible],minlength=k)
            sil=float(silhouette_score(coord[silhouette_rows],labels[silhouette_rows],metric="euclidean"))
            diagnostics.append({"preprocessing":prep,"k":k,"seed":seed,"inertia":inertia,"n_iter":n_iter,
                "silhouette_n":SILHOUETTE_N,"silhouette":sil,"cluster_min":int(sizes.min()),
                "cluster_median":float(np.median(sizes)),"cluster_max":int(sizes.max())})
            sens_frames.append(pl.DataFrame({"Metadata_JCP2022":frame["Metadata_JCP2022"],
                "preprocessing":prep,"k":k,"seed":seed,"cluster_id":labels,"cluster_distance":dist}))
            if prep=="clip10" and k==PRIMARY_K:
                primary_runs.append((inertia,seed,labels,dist,margin,centers))
    stability=[]
    for (prep,k), arrays in labels_by_config.items():
        for (i,a),(j,b) in combinations(enumerate(arrays),2):
            stability.append({"preprocessing":prep,"k":k,"seed_a":SEEDS[i],"seed_b":SEEDS[j],
                "ari":adjusted_rand_score(a[eligible],b[eligible]),
                "ami":adjusted_mutual_info_score(a[eligible],b[eligible])})
    diagnostics.extend(stability)
    best=min(primary_runs,key=lambda x:(x[0],x[1]))
    _,best_seed,labels,dist,margin,centers=best
    strata=[structural_stratum(int(w),int(s)) for w,s in zip(frame["n_wells"],frame["n_sources"],strict=True)]
    unlabeled=pl.DataFrame({"Metadata_JCP2022":frame["Metadata_JCP2022"],"n_wells":frame["n_wells"],
        "n_sources":frame["n_sources"],"n_plates":frame["n_plates"],"fit_eligible":eligible,
        "structural_stratum":strata,"cluster_id":labels,"cluster_distance":dist,"cluster_margin":margin,
        "fold_id":[stable_fold(v) for v in frame["Metadata_JCP2022"].to_list()]})
    unlabeled.write_parquet(fitdir/"cluster_assignments_unlabeled.parquet",compression="zstd")
    pl.concat(sens_frames).write_parquet(fitdir/"cluster_assignments_sensitivity_unlabeled.parquet",compression="zstd")
    pd.DataFrame(diagnostics).to_csv(fitdir/"clustering_diagnostics.csv",index=False)
    np.savez_compressed(fitdir/"primary_model.npz",feature_names=kept,imputation_medians=impute,
        scaling_medians=np.nanmedian(values[eligible][:,keep],axis=0),scaling_iqrs=iqr,
        clip_limit=np.asarray([10.0]),pca_mean=pca.mean_,pca_components=pca.components_,
        centroids=centers,best_seed=np.asarray([best_seed]),k=np.asarray([PRIMARY_K]))
    identity={"version":"profile-cluster-fit-v1","design":record(DESIGN),"runner":record(Path(__file__)),
        "inputs":[verify(CONSENSUS),verify(FEATURES)],"parameters":{"eligible":"n_wells>=4","seeds":list(SEEDS),
        "primary":{"preprocessing":"clip10","pca":32,"k":128,"best_seed":best_seed},
        "sensitivities":["clip10-k64","clip10-k256","rank-gaussian-k128"]},
        "counts":{"all":frame.height,"eligible":int(eligible.sum()),"retained_features":int(keep.sum())},
        "pca_explained_variance":float(pca.explained_variance_ratio_.sum()),"runtime_seconds":time.perf_counter()-started}
    write_json_new(fitdir/"computation_identity.json",identity)
    frozen=[fitdir/"cluster_assignments_unlabeled.parquet",fitdir/"cluster_assignments_sensitivity_unlabeled.parquet",
            fitdir/"clustering_diagnostics.csv",fitdir/"primary_model.npz",fitdir/"computation_identity.json"]
    write_json_new(fitdir/"fit_complete.json",{"status":"complete","artifacts":[record(p) for p in frozen]})
    print(json.dumps(identity,indent=2,sort_keys=True)); return 0


def bh_adjust(p: np.ndarray) -> np.ndarray:
    order=np.argsort(p); ranked=p[order]; n=len(p)
    adj=np.minimum.accumulate((ranked*n/np.arange(1,n+1))[::-1])[::-1]
    out=np.empty(n); out[order]=np.minimum(adj,1); return out


def tie_metrics(y: np.ndarray, score: np.ndarray, cutoff: int):
    threshold=np.sort(score)[::-1][min(cutoff,len(score))-1]
    chosen=score>=threshold
    return float(y[chosen].mean()), float(y[chosen].sum()/y.sum()), int(chosen.sum())


def metric_row(y: np.ndarray, score: np.ndarray, predictor: str, universe: str, cutoff: int):
    p=np.clip(score,1e-9,1-1e-9); prec,rec,nret=tie_metrics(y,p,cutoff)
    prev=float(y.mean())
    return {"universe":universe,"predictor":predictor,"n":len(y),"n_selected":int(y.sum()),"prevalence":prev,
        "roc_auc":roc_auc_score(y,p),"average_precision":average_precision_score(y,p),
        "ap_lift":average_precision_score(y,p)/prev,"log_loss":log_loss(y,p,labels=[0,1]),
        "brier":brier_score_loss(y,p),"precision_at_selected_count":prec,
        "recall_at_selected_count":rec,"retrieved_with_ties":nret}


def crossfit_scores(df: pd.DataFrame, y: np.ndarray):
    folds=df.fold_id.to_numpy(); strata=df.structural_stratum.to_numpy().reshape(-1,1)
    enc_s=OneHotEncoder(handle_unknown="ignore",sparse_output=True).fit(strata)
    s_all=enc_s.transform(strata)
    enc_c=OneHotEncoder(handle_unknown="ignore",sparse_output=True).fit(df.cluster_id.to_numpy().reshape(-1,1))
    c_all=enc_c.transform(df.cluster_id.to_numpy().reshape(-1,1))
    numeric=np.log1p(df[["n_wells","n_sources","n_plates"]].to_numpy(float))
    scores={k:np.empty(len(df),float) for k in ("constant","count_only","cluster_only","count_plus_cluster")}
    for fold in range(5):
        test=folds==fold; train=~test; prev=(y[train].sum()+.5)/(train.sum()+1)
        scores["constant"][test]=prev
        counts=np.bincount(df.cluster_id.to_numpy()[train],minlength=PRIMARY_K)
        pos=np.bincount(df.cluster_id.to_numpy()[train],weights=y[train],minlength=PRIMARY_K)
        scores["cluster_only"][test]=(pos[df.cluster_id.to_numpy()[test]]+.5)/(counts[df.cluster_id.to_numpy()[test]]+1)
        mean=numeric[train].mean(0); sd=numeric[train].std(0); sd[sd==0]=1
        z=sparse.csr_matrix((numeric-mean)/sd)
        x_count=sparse.hstack([z,s_all],format="csr")
        x_comb=sparse.hstack([z,s_all,c_all],format="csr")
        for name,x in (("count_only",x_count),("count_plus_cluster",x_comb)):
            model=LogisticRegression(C=1.0,max_iter=500,solver="lbfgs")
            model.fit(x[train],y[train]); scores[name][test]=model.predict_proba(x[test])[:,1]
    return scores


def score_selection(args: argparse.Namespace) -> int:
    started=time.perf_counter(); out=args.output_dir.resolve(strict=True); fitdir=out/"fit"
    completion=json.loads((fitdir/"fit_complete.json").read_text())
    for rec in completion["artifacts"]:
        p=Path(rec["path"])
        if record(p)!={"path":str(p.resolve()),"bytes":rec["bytes"],"sha256":rec["sha256"]}: raise RuntimeError("Frozen fit drift")
    verify(SELECTED)
    assignments=pl.read_parquet(fitdir/"cluster_assignments_unlabeled.parquet")
    ids=pl.read_parquet(SELECTED,columns=["Metadata_JCP2022"])
    if ids.height!=3832 or ids["Metadata_JCP2022"].n_unique()!=3832: raise RuntimeError("Selected manifest drift")
    selected_set=set(ids["Metadata_JCP2022"].to_list())
    df=assignments.with_columns(pl.col("Metadata_JCP2022").is_in(selected_set).alias("selected"))
    if df.filter(pl.col("selected")).height!=3832 or not df.filter(pl.col("selected"))["fit_eligible"].all():
        raise RuntimeError("Selected membership/eligibility drift")
    pdf=df.to_pandas(); y=pdf.selected.astype(int).to_numpy(); eligible=pdf.fit_eligible.to_numpy(bool)
    metrics=[]; score_sets={}
    for name,mask in (("eligible_primary",eligible),("all_descriptive",np.ones(len(pdf),bool))):
        local=pdf.loc[mask].reset_index(drop=True); yy=y[mask]; scores=crossfit_scores(local,yy); score_sets[name]=scores
        for pred,score in scores.items(): metrics.append(metric_row(yy,score,pred,name,int(yy.sum())))
    # Full-data cluster fractions and OOF score for the assignment table.
    totals=np.bincount(pdf.cluster_id,minlength=PRIMARY_K); positives=np.bincount(pdf.cluster_id,weights=y,minlength=PRIMARY_K)
    frac=positives/totals; global_prev=y.mean(); oof_all=score_sets["all_descriptive"]["cluster_only"]
    pdf["cluster_selected_fraction"]=frac[pdf.cluster_id]
    pdf["cluster_selected_lift"]=pdf.cluster_selected_fraction/global_prev
    pdf["selection_probability_oof"]=oof_all
    pdf["selection_score_rank"]=rankdata(-oof_all,method="average")
    pdf.to_parquet(out/"compound_cluster_assignments.parquet",index=False)

    # Structure-conditional expectations and deterministic permutation null.
    strata=pdf.structural_stratum.to_numpy(); clusters=pdf.cluster_id.to_numpy(); unique_s=sorted(set(strata))
    expected=np.zeros(PRIMARY_K); variance=np.zeros(PRIMARY_K)
    for s in unique_s:
        idx=strata==s; p=y[idx].mean(); n=np.bincount(clusters[idx],minlength=PRIMARY_K)
        expected+=n*p; variance+=n*p*(1-p)
    residual=(positives-expected)/np.sqrt(np.maximum(variance,1e-12))
    observed_stat=float(np.sum((positives[variance>0]-expected[variance>0])**2/variance[variance>0]))
    rng=np.random.default_rng(20260811); exceed=np.zeros(PRIMARY_K,int); global_exceed=0
    for _ in range(PERMUTATIONS):
        perm=np.empty_like(y)
        for s in unique_s:
            idx=np.flatnonzero(strata==s); perm[idx]=rng.permutation(y[idx])
        cnt=np.bincount(clusters,weights=perm,minlength=PRIMARY_K)
        exceed += (np.abs(cnt-expected)>=np.abs(positives-expected)-1e-12)
        stat=float(np.sum((cnt[variance>0]-expected[variance>0])**2/variance[variance>0]))
        global_exceed += stat>=observed_stat-1e-12
    p_cluster=(exceed+1)/(PERMUTATIONS+1); q_cluster=bh_adjust(p_cluster)
    rows=[]
    for c in range(PRIMARY_K):
        idx=clusters==c; n=int(idx.sum()); k=int(y[idx].sum()); sm=(k+.5)/(n+1)
        lo,hi=beta.ppf([.025,.975],k+.5,n-k+.5)
        rows.append({"cluster_id":c,"n_compounds":n,"n_selected":k,"n_nonselected":n-k,
            "selected_fraction":k/n,"jeffreys_probability":sm,"jeffreys_working_low":lo,
            "jeffreys_working_high":hi,"overall_selected_fraction":global_prev,"selection_lift":k/n/global_prev,
            "log2_selection_lift":math.log2(max(k/n/global_prev,1e-12)),"conditional_expected_selected":expected[c],
            "conditional_lift":k/expected[c] if expected[c]>0 else np.nan,"conditional_standardized_residual":residual[c],
            "conditional_permutation_p":p_cluster[c],"conditional_bh_q":q_cluster[c],
            "cluster_distance_median":float(np.median(pdf.cluster_distance.to_numpy()[idx])),
            "cluster_distance_p90":float(np.quantile(pdf.cluster_distance.to_numpy()[idx],.9)),
            "n_wells_median":float(np.median(pdf.n_wells.to_numpy()[idx])),
            "n_sources_median":float(np.median(pdf.n_sources.to_numpy()[idx])),
            "n_plates_median":float(np.median(pdf.n_plates.to_numpy()[idx]))})
    table=pd.DataFrame(rows); table.to_csv(out/"cluster_selection_table.csv",index=False); table.to_parquet(out/"cluster_selection_table.parquet",index=False)
    pd.DataFrame(metrics).to_csv(out/"retrieval_metrics.csv",index=False)

    # Ten separate matched comparisons and leakage contracts.
    eval_ids=set(pl.read_parquet(EVALUATION,columns=["Metadata_JCP2022"])["Metadata_JCP2022"].to_list())
    if selected_set & eval_ids: raise RuntimeError("Selected/evaluation leakage")
    matched_rows=[]
    for path in sorted(MANIFESTS.glob("matched_random_seed_*.parquet")):
        comp=pl.read_parquet(path,columns=["Metadata_JCP2022"])["Metadata_JCP2022"].to_list()
        if len(comp)!=3832 or len(set(comp))!=3832 or selected_set&set(comp) or not set(comp)<=eval_ids:
            raise RuntimeError(f"Matched manifest contract failure: {path}")
        mask=pdf.Metadata_JCP2022.isin(selected_set|set(comp)).to_numpy(); local=pdf.loc[mask].reset_index(drop=True); yy=local.selected.astype(int).to_numpy()
        scores=crossfit_scores(local,yy)
        seed=int(path.stem.rsplit("_",1)[1])
        for pred,score in scores.items(): matched_rows.append({"seed":seed,**metric_row(yy,score,pred,f"matched_{seed}",3832)})
    matched=pd.DataFrame(matched_rows); matched.to_csv(out/"matched_retrieval_sensitivity.csv",index=False)

    # Representation summaries on eligible universe.
    ec=clusters[eligible]; ey=y[eligible]; n_e=np.bincount(ec,minlength=PRIMARY_K); n_s=np.bincount(ec,weights=ey,minlength=PRIMARY_K)
    p=n_s/n_s.sum(); q=n_e/n_e.sum(); occupied=n_s>0
    m=.5*(p+q); js=.5*np.sum(np.where(p>0,p*np.log(p/m),0))+.5*np.sum(np.where(q>0,q*np.log(q/m),0))
    rep={"clusters":PRIMARY_K,"selected_occupied_clusters":int(occupied.sum()),"occupied_fraction":float(occupied.mean()),
         "eligible_compound_mass_in_occupied_clusters":float(n_e[occupied].sum()/n_e.sum()),
         "total_variation_selected_vs_eligible":float(.5*np.abs(p-q).sum()),"jensen_shannon_divergence_nats":float(js),
         "selection_lift_median":float(np.median(table.selection_lift)),"selection_lift_iqr":[float(table.selection_lift.quantile(.25)),float(table.selection_lift.quantile(.75))]}
    write_json_new(out/"representation_summary.json",rep)
    perm={"permutations":PERMUTATIONS,"shuffle":"within structural_stratum","observed_statistic":observed_stat,
          "one_sided_p":(global_exceed+1)/(PERMUTATIONS+1),"interpretation":"finite-cohort design-null; not population inference"}
    write_json_new(out/"permutation_summary.json",perm)

    # All-cluster plot.
    fig,ax=plt.subplots(2,1,figsize=(12,7),sharex=True)
    ax[0].bar(table.cluster_id,table.n_compounds,color="#777777"); ax[0].set_ylabel("Compounds")
    ax[1].bar(table.cluster_id,table.selected_fraction,color="#2b8cbe"); ax[1].axhline(global_prev,color="black",ls="--",lw=1,label="overall")
    ax[1].set_ylabel("Selected fraction"); ax[1].set_xlabel("Canonical cluster ID"); ax[1].legend(); fig.tight_layout()
    fig.savefig(out/"cluster_selection_all_clusters.png",dpi=180); plt.close(fig)

    actual=pd.DataFrame(metrics); e=actual[actual.universe=="eligible_primary"].set_index("predictor")
    ap_ratio=float(e.loc["count_plus_cluster","average_precision"]/e.loc["count_only","average_precision"])
    detectable=perm["one_sided_p"]<=.05 and e.loc["count_plus_cluster","log_loss"]<e.loc["count_only","log_loss"]
    material=detectable and ap_ratio>=1.25
    mean_ari=pd.read_csv(fitdir/"clustering_diagnostics.csv").query("preprocessing=='clip10' and k==128 and ari==ari")["ari"].mean()
    conclusion={"detectably_better_than_structure":bool(detectable),"materially_better_than_structure":bool(material),
                "combined_to_count_ap_ratio":ap_ratio,"conditional_permutation_p":perm["one_sided_p"],
                "primary_seed_mean_ari":float(mean_ari),"unique_cluster_biological_claims_allowed":bool(mean_ari>=.8)}
    write_json_new(out/"selection_conclusion.json",conclusion)
    report=f"""# Cluster-selection representativeness report\n\n## Scope\n\nThis CPU-only analysis partitions the frozen 115,721-compound CellProfiler consensus without selection labels, then scores the frozen 3,832-ID JUMP-lite membership. Selected prevalence is {3832/115721:.6%} overall and {3832/95426:.6%} among the 95,426 fit-eligible compounds. Results describe this finite cohort.\n\n## Partition diagnostics\n\nPrimary clip-10/PCA32/K=128 clusters were selected by minimum inertia across five fixed seeds before labels. Mean pairwise seed ARI was {mean_ari:.3f}; therefore unique cluster-specific biological claims are {'permitted by the preregistered gate' if mean_ari>=.8 else 'prohibited'}. Clusters are resolution-dependent partitions of a continuum.\n\n## Representation and retrieval\n\nSelected compounds occupied {rep['selected_occupied_clusters']}/128 clusters covering {rep['eligible_compound_mass_in_occupied_clusters']:.2%} of eligible compounds. TV distance was {rep['total_variation_selected_vs_eligible']:.4f}; Jensen-Shannon divergence was {rep['jensen_shannon_divergence_nats']:.4f} nats.\n\nEligible OOF count-only AP was {e.loc['count_only','average_precision']:.6f}; adding cluster identity gave {e.loc['count_plus_cluster','average_precision']:.6f} (ratio {ap_ratio:.3f}). OOF log loss changed from {e.loc['count_only','log_loss']:.6f} to {e.loc['count_plus_cluster','log_loss']:.6f}. The 2,000-shuffle structure-conditional design-null p was {perm['one_sided_p']:.6f}. Under preregistered gates, cluster identity was {'detectably' if detectable else 'not detectably'} better than acquisition structure and {'materially' if material else 'not materially'} better.\n\n## Limitations\n\nSelection is strongly confounded with replication structure, clusters are unstable/resolution-dependent, the feature projection is fixed rather than biology-optimized, and low retrieval supports mixing but does not prove biological representativeness. Matched-set results are ten separate descriptive sensitivities, not p-values or population percentiles.\n"""
    (out/"REPORT.md").write_text(report)
    provenance={"version":"profile-cluster-selection-v1","design":record(DESIGN),"runner":record(Path(__file__)),
        "fit_completion":record(fitdir/"fit_complete.json"),"scoring_inputs":[verify(SELECTED),record(EVALUATION)]+[record(p) for p in sorted(MANIFESTS.glob("matched_random_seed_*.parquet"))],
        "counts":{"all":len(pdf),"selected":int(y.sum()),"eligible":int(eligible.sum())},"conclusion":conclusion,
        "python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"polars":pl.__version__,"scipy":scipy.__version__,"sklearn":sklearn_version,
        "runtime_seconds":time.perf_counter()-started}
    write_json_new(out/"provenance.json",provenance)
    outputs=[p for p in sorted(out.iterdir()) if p.is_file() and p.name!="output_hashes.json"]+[p for p in sorted(fitdir.iterdir()) if p.is_file()]
    write_json_new(out/"output_hashes.json",{"files":[record(p) for p in outputs]})
    print(json.dumps({"representation":rep,"permutation":perm,"conclusion":conclusion},indent=2,sort_keys=True)); return 0


def main() -> int:
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="action",required=True)
    fit=sub.add_parser("fit-clusters"); fit.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    score=sub.add_parser("score-selection"); score.add_argument("--output-dir",type=Path,default=DEFAULT_OUT)
    args=p.parse_args(); return fit_clusters(args) if args.action=="fit-clusters" else score_selection(args)

if __name__=="__main__": raise SystemExit(main())
