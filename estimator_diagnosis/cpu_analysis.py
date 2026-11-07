#!/usr/bin/env python3
"""CPU-only diagnostics for the SAO-50 D-TRAK / EKFAC attribution study.

Reads existing outputs only; writes nothing outside estimator_diagnosis/.
"""
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, rankdata

ROOT = Path("/ocean/projects/mth260005p/xjiang6/stable-audio-tools")
OUT = ROOT / "sao_ip_100/outputs"
DIAG = ROOT / "estimator_diagnosis"

sellers = pd.read_csv(OUT / "selection/seller_manifest.csv")
seller_ids = list(sellers["seller_id"])
S = len(seller_ids)
manifest = json.load(open(OUT / "generated_queries/prompt_manifest.json"))["items"]
prompt_ids = sorted(int(it["prompt_id"]) for it in manifest)
src_seller = {int(it["prompt_id"]): it["source_seller_id"] for it in manifest}
Q = len(prompt_ids)


def to_matrix(df, col):
    df = df.copy()
    df["prompt_id"] = df["prompt_id"].astype(int)
    piv = df.pivot(index="seller_id", columns="prompt_id", values=col)
    return piv.loc[seller_ids, prompt_ids].to_numpy(dtype=np.float64)


def per_seller_spearman(est, gt):
    vals = []
    for j in range(gt.shape[0]):
        r = spearmanr(est[j], gt[j]).correlation
        vals.append(r)
    return np.array(vals)


A = to_matrix(pd.read_csv(OUT / "a_star.csv"), "a_star")
res = {}

# ---------------- ground truth structure ----------------
mu_j = A.mean(1, keepdims=True)
c_q = A.mean(0, keepdims=True)
grand = A.mean()
resid = A - mu_j - c_q + grand
tot = ((A - grand) ** 2).sum()
res["gt_var_fraction_seller_main"] = float(((mu_j - grand) ** 2).sum() * Q / tot)
res["gt_var_fraction_query_common"] = float(((c_q - grand) ** 2).sum() * S / tot)
res["gt_var_fraction_interaction"] = float((resid ** 2).sum() / tot)
# within-seller (what per-seller Spearman sees): fraction of each row's variance explained by c_q
row_var = A.var(1)
row_r2 = np.array([np.corrcoef(A[j], c_q[0])[0, 1] ** 2 for j in range(S)])
res["gt_mean_within_seller_R2_of_common_query_component"] = float(row_r2.mean())
res["gt_per_seller_spearman_if_predict_common_component_only"] = float(
    np.nanmean(per_seller_spearman(np.tile(c_q, (S, 1)), A)))
# leave-one-seller-out common component (avoid self-inclusion)
loo_c = (A.sum(0, keepdims=True) - A) / (S - 1)
res["gt_per_seller_spearman_if_predict_LOO_common_component"] = float(
    np.nanmean(per_seller_spearman(loo_c, A)))
res["gt_frac_entries_positive"] = float((A > 0).mean())
res["gt_mean_a_star"] = float(A.mean())
res["gt_std_a_star"] = float(A.std())
res["gt_mean_abs_resid_over_std_c"] = float(np.abs(resid).mean() / c_q.std())

# Source-seller signal: is a*[source, q] larger than other sellers?
sid_index = {s: i for i, s in enumerate(seller_ids)}
ranks = []
z_src = []
for qi, pid in enumerate(prompt_ids):
    s = sid_index[src_seller[pid]]
    col = A[:, qi]
    r = rankdata(-col)[s]  # 1 = largest a*
    ranks.append(r)
    others = np.delete(col, s)
    z_src.append((col[s] - others.mean()) / others.std())
ranks = np.array(ranks)
res["gt_source_seller_mean_rank_of_50"] = float(ranks.mean())
res["gt_source_seller_frac_top1"] = float((ranks <= 1).mean())
res["gt_source_seller_frac_top5"] = float((ranks <= 5).mean())
res["gt_source_seller_median_z"] = float(np.median(z_src))
# seller-specific oracle: indicator of being the source seller of the query
IND = np.zeros_like(A)
for qi, pid in enumerate(prompt_ids):
    IND[sid_index[src_seller[pid]], qi] = 1.0
res["gt_per_seller_spearman_source_indicator_oracle"] = float(
    np.nanmean(per_seller_spearman(IND + 1e-9 * np.random.default_rng(0).standard_normal(A.shape), A)))
# same but after removing the common query component from a*
A_dc = A - loo_c
res["gt_per_seller_spearman_source_indicator_vs_decommoned_astar"] = float(
    np.nanmean(per_seller_spearman(IND + 1e-9 * np.random.default_rng(0).standard_normal(A.shape), A_dc)))

# Paired difference noise sanity: loss_full spread vs a* spread
L = torch.load(OUT / "losses_model_x_query.pt").double().numpy()
res["loss_full_mean"] = float(L[0].mean())
res["loss_full_std_over_queries"] = float(L[0].std())
res["model0_minus_mean_loo_loss_mean"] = float((L[0] - L[1:].mean(0)).mean())
res["frac_queries_where_model0_is_best_of_51"] = float((L.argmin(0) == 0).mean())
rank0 = np.array([rankdata(L[:, q])[0] for q in range(L.shape[1])])
res["model0_mean_rank_of_51_by_loss"] = float(rank0.mean())
# correlation of LOO models' loss deviations across sellers (shared seed-independent structure)
Dev = L[1:] - L[1:].mean(0, keepdims=True)
cc = np.corrcoef(Dev)
res["mean_offdiag_corr_between_LOO_model_deviation_profiles"] = float(cc[~np.eye(S, dtype=bool)].mean())

# ---------------- estimator scores ----------------
methods = {
    "dtrak_v100_chunked": (OUT / "a_hat_dtrak_v100_chunked.csv", "a_hat_dtrak"),
    "dtrak_6h_fullproj": (OUT / "a_hat_dtrak_6h.csv", "a_hat_dtrak"),
    "ekfac": (OUT / "a_hat_ekfac_v100_layer_sharded.csv", "a_hat_ekfac"),
    "clap": (OUT / "baselines/clap/a_hat_clap.csv", "a_hat_clap"),
    "fingerprint": (OUT / "baselines/fingerprint/a_hat_fingerprint.csv", "a_hat_fingerprint"),
    "random": (OUT / "baselines/random/a_hat_random.csv", "a_hat_random"),
}
Ahat = {}
for name, (path, col) in methods.items():
    M = to_matrix(pd.read_csv(path), col)
    Ahat[name] = M
    sp = per_seller_spearman(M, A)
    sp_dc = per_seller_spearman(M, A_dc)
    # how much of each estimator's row variance is its own common query component
    cq_hat = M.mean(0, keepdims=True)
    r2 = np.array([np.corrcoef(M[j], cq_hat[0])[0, 1] ** 2 for j in range(S)])
    # source-seller hit-rate of the estimator (does it rank the prompt's source seller first?)
    hits = []
    for qi, pid in enumerate(prompt_ids):
        hits.append(rankdata(-M[:, qi])[sid_index[src_seller[pid]]])
    hits = np.array(hits)
    res[f"{name}_mean_per_seller_spearman"] = float(np.nanmean(sp))
    res[f"{name}_mean_per_seller_spearman_vs_decommoned_astar"] = float(np.nanmean(sp_dc))
    res[f"{name}_within_seller_R2_of_own_common_query_component"] = float(np.nanmean(r2))
    res[f"{name}_corr_own_common_component_with_gt_common"] = float(
        spearmanr(cq_hat[0], c_q[0]).correlation)
    res[f"{name}_source_seller_mean_rank"] = float(hits.mean())
    res[f"{name}_source_seller_top1"] = float((hits <= 1).mean())
    # per-query Spearman across sellers (alternative metric)
    pq = [spearmanr(M[:, q], A[:, q]).correlation for q in range(Q)]
    res[f"{name}_mean_per_query_spearman_across_sellers"] = float(np.nanmean(pq))

# agreement between the two D-TRAK runs (same f/K/lambda, different JL draw + caption draw)
a1, a2 = Ahat["dtrak_v100_chunked"], Ahat["dtrak_6h_fullproj"]
res["dtrak_run_agreement_seller_level_pearson"] = float(np.corrcoef(a1.ravel(), a2.ravel())[0, 1])
res["dtrak_run_agreement_seller_level_mean_row_spearman"] = float(
    np.nanmean([spearmanr(a1[j], a2[j]).correlation for j in range(S)]))

json.dump(res, open(DIAG / "cpu_analysis_results.json", "w"), indent=2)
for k, v in res.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
