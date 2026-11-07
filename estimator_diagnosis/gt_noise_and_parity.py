#!/usr/bin/env python3
"""Ground-truth noise ceiling + batch-parity artefact at seller level (CPU only)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

ROOT = Path("/ocean/projects/mth260005p/xjiang6/stable-audio-tools")
OUT = ROOT / "sao_ip_100/outputs"
DIAG = ROOT / "estimator_diagnosis"
res = {}

sellers = pd.read_csv(OUT / "selection/seller_manifest.csv")
seller_ids = list(sellers["seller_id"])
genre = dict(zip(sellers["seller_id"], sellers["primary_genre"]))
ntracks = dict(zip(sellers["seller_id"], sellers["selected_tracks"]))
S = len(seller_ids)
sidx = {s: i for i, s in enumerate(seller_ids)}
items = json.load(open(OUT / "generated_queries/prompt_manifest.json"))["items"]
prompt_ids = sorted(int(it["prompt_id"]) for it in items)
src = {int(it["prompt_id"]): it["source_seller_id"] for it in items}
Q = len(prompt_ids)

astar = pd.read_csv(OUT / "a_star.csv")
astar["prompt_id"] = astar["prompt_id"].astype(int)
A = astar.pivot(index="seller_id", columns="prompt_id", values="a_star").loc[seller_ids, prompt_ids].to_numpy()
c = A.mean(0, keepdims=True)
mu = A.mean(1, keepdims=True)
R = A - c - mu + A.mean()  # interaction residual

src_mask = np.zeros_like(A, dtype=bool)
same_genre = np.zeros_like(A, dtype=bool)
for qi, p in enumerate(prompt_ids):
    s = src[p]
    src_mask[sidx[s], qi] = True
    for j, sj in enumerate(seller_ids):
        if sj != s and genre[sj] == genre[s]:
            same_genre[j, qi] = True
other = ~src_mask & ~same_genre
sd_other = R[other].std()
res["resid_std_all"] = float(R.std())
res["resid_std_diffgenre_nonsource"] = float(sd_other)
res["resid_mean_source_in_sd_units"] = float(R[src_mask].mean() / sd_other)
res["resid_mean_samegenre_nonsource_in_sd_units"] = float(R[same_genre].mean() / sd_other)
res["resid_mean_diffgenre_nonsource_in_sd_units"] = float(R[other].mean() / sd_other)
res["resid_var_share_of_source_cells"] = float((R[src_mask] ** 2).sum() / (R ** 2).sum())
res["resid_var_share_of_samegenre_cells"] = float((R[same_genre] ** 2).sum() / (R ** 2).sum())
res["frac_cells_source"] = float(src_mask.mean())
res["frac_cells_samegenre"] = float(same_genre.mean())

# Seller main effect vs catalogue size (fewer tracks -> fewer optimizer steps at fixed 50 epochs)
sizes = np.array([ntracks[s] for s in seller_ids])
res["seller_main_effect_vs_ntracks_spearman"] = float(spearmanr(mu[:, 0], sizes).correlation)

# Noise-ceiling model: within-seller row of a* = c_q + d_jq + n_jq.
# If diff-genre non-source residuals are pure retraining (seed) noise, estimate the share of
# within-seller variance that is seller-specific *signal*.
row_var = A.var(1).mean()
c_var = c.var()
noise_var = sd_other ** 2
signal_var = max(R.var() - noise_var, 0.0)
res["within_seller_var_share_common_c"] = float(c_var / row_var)
res["within_seller_var_share_seller_signal_est"] = float(signal_var / row_var)
res["within_seller_var_share_seed_noise_est"] = float(noise_var / row_var)
res["pearson_ceiling_perfect_data_only_predictor"] = float(np.sqrt(signal_var / row_var))
res["pearson_ceiling_predictor_knowing_c_plus_signal"] = float(np.sqrt((signal_var + c_var) / row_var))
# Empirical: 'oracle' that knows the residual on source + same-genre cells only
orc = np.where(src_mask | same_genre, R, 0.0)
rng = np.random.default_rng(0)
res["per_seller_spearman_oracle_source_and_samegenre_resid"] = float(np.nanmean(
    [spearmanr(orc[j] + 1e-12 * rng.standard_normal(Q), A[j]).correlation for j in range(S)]))
orc2 = orc + c
res["per_seller_spearman_oracle_c_plus_source_samegenre_resid"] = float(np.nanmean(
    [spearmanr(orc2[j], A[j]).correlation for j in range(S)]))

# Batch-parity artefact in estimator seller matrices
def to_mat(path, col):
    df = pd.read_csv(path)
    df["prompt_id"] = df["prompt_id"].astype(int)
    return df.pivot(index="seller_id", columns="prompt_id", values=col).loc[seller_ids, prompt_ids].to_numpy()

for name, run, path, col in [
    ("dtrak_chunked", "dtrak_attribution_v100_chunked", OUT / "a_hat_dtrak_v100_chunked.csv", "a_hat_dtrak"),
    ("dtrak_6h", "dtrak_attribution_6h", OUT / "a_hat_dtrak_6h.csv", "a_hat_dtrak"),
]:
    M = to_mat(path, col)
    qids = (OUT / run / "members/single/query_features.memmap.ids.txt").read_text().splitlines()
    pos = {int(Path(q).stem.split("_")[1]): i % 2 for i, q in enumerate(qids)}
    par = np.array([pos[p] for p in prompt_ids])
    r = [abs(pearsonr(M[j], par)[0]) for j in range(S)]
    res[f"{name}_mean_abs_corr_seller_row_with_query_batch_parity"] = float(np.mean(r))
    # fraction of within-seller variance explained by parity
    r2 = [pearsonr(M[j], par)[0] ** 2 for j in range(S)]
    res[f"{name}_mean_within_seller_R2_query_parity"] = float(np.mean(r2))
    # remove parity means within each seller and rescore
    Mp = M.copy()
    for j in range(S):
        for b in (0, 1):
            Mp[j, par == b] -= M[j, par == b].mean()
    res[f"{name}_per_seller_spearman_after_removing_parity_means"] = float(np.nanmean(
        [spearmanr(Mp[j], A[j]).correlation for j in range(S)]))
    res[f"{name}_per_seller_spearman_raw"] = float(np.nanmean(
        [spearmanr(M[j], A[j]).correlation for j in range(S)]))
# GT parity (ground-truth run batch position) sanity
gq = (OUT / "losses_model_x_query.pt.query_ids.txt").read_text().splitlines()
gpos = {int(Path(q).stem.split("_")[1]): i % 2 for i, q in enumerate(gq)}
gpar = np.array([gpos[p] for p in prompt_ids])
res["gt_mean_abs_corr_seller_row_with_gt_batch_parity"] = float(np.mean([abs(pearsonr(A[j], gpar)[0]) for j in range(S)]))
res["null_expected_abs_corr_Q200"] = float(np.sqrt(2 / np.pi) / np.sqrt(Q))

json.dump(res, open(DIAG / "gt_noise_and_parity_results.json", "w"), indent=2)
for k, v in res.items():
    print(f"{k}: {v:.4f}")
