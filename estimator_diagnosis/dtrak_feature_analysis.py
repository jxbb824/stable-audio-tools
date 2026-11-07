#!/usr/bin/env python3
"""CPU diagnostics on the saved D-TRAK projected features and EKFAC score matrices.

- eigen-spectrum of Phi^T Phi vs lambda
- lambda sweep / raw inner-product / cosine rescoring, evaluated against a*
- agreement between the two production D-TRAK runs (chunked JL vs full JL)
- batch-position ("shared noise") artefact
- EKFAC score heavy-tailedness
Nothing outside estimator_diagnosis/ is written.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

ROOT = Path("/ocean/projects/mth260005p/xjiang6/stable-audio-tools")
OUT = ROOT / "sao_ip_100/outputs"
DIAG = ROOT / "estimator_diagnosis"

sellers = pd.read_csv(OUT / "selection/seller_manifest.csv")
seller_ids = list(sellers["seller_id"])
S = len(seller_ids)
cat = pd.read_csv(OUT / "selection/category_manifest.csv")
relpath_to_seller = dict(zip(cat["relpath"], cat["seller_id"]))
manifest = json.load(open(OUT / "generated_queries/prompt_manifest.json"))["items"]
prompt_ids = sorted(int(it["prompt_id"]) for it in manifest)

astar = pd.read_csv(OUT / "a_star.csv")
astar["prompt_id"] = astar["prompt_id"].astype(int)
A = astar.pivot(index="seller_id", columns="prompt_id", values="a_star").loc[seller_ids, prompt_ids].to_numpy()
loo_c = (A.sum(0, keepdims=True) - A) / (S - 1)
A_dc = A - loo_c


def per_seller_spearman(M, G):
    return float(np.nanmean([spearmanr(M[j], G[j]).correlation for j in range(G.shape[0])]))


def load_run(run):
    d = OUT / run / "members/single"
    tm = json.load(open(d / "train_features_raw.memmap.meta.json"))
    qm = json.load(open(d / "query_features.memmap.meta.json"))
    X = np.array(np.memmap(d / "train_features_raw.memmap", dtype=np.float32, mode="r",
                           shape=(tm["num_rows_allocated"], tm["proj_dim"])))[: tm["valid_rows"]].astype(np.float64)
    Xq = np.array(np.memmap(d / "query_features.memmap", dtype=np.float32, mode="r",
                            shape=(qm["num_rows_allocated"], qm["proj_dim"])))[: qm["valid_rows"]].astype(np.float64)
    tid = (d / "train_features_raw.memmap.ids.txt").read_text().splitlines()
    qid = (d / "query_features.memmap.ids.txt").read_text().splitlines()
    return X, Xq, tid, qid


def seller_matrix(scores_q_x_t, tids, qids):
    """Aggregate per-track scores [Q_raw, N_raw] into seller x prompt matrix (sum)."""
    t_seller = [relpath_to_seller["/".join(t.split("/")[-2:])] for t in tids]
    q_pid = [int(Path(q).stem.split("_")[1]) for q in qids]
    M = np.zeros((S, len(prompt_ids)))
    col = {p: i for i, p in enumerate(prompt_ids)}
    sidx = {s: i for i, s in enumerate(seller_ids)}
    for n, s in enumerate(t_seller):
        for qi, p in enumerate(q_pid):
            pass
    tsel = np.array([sidx[s] for s in t_seller])
    qcol = np.array([col[p] for p in q_pid])
    agg = np.zeros((S, scores_q_x_t.shape[0]))
    np.add.at(agg, tsel, scores_q_x_t.T)
    M[:, qcol] = agg
    return M


res = {}
runs = {"chunked": "dtrak_attribution_v100_chunked", "full6h": "dtrak_attribution_6h"}
data = {k: load_run(v) for k, v in runs.items()}

lams = [1e-1, 1e1, 1e3, 1e5, 1e7, 1e9, 1e11]
for name, (X, Xq, tid, qid) in data.items():
    N, k = X.shape
    K = X.T @ X
    ev = np.linalg.eigvalsh(K)[::-1]
    res[f"{name}_N_train"] = int(N)
    res[f"{name}_proj_dim"] = int(k)
    res[f"{name}_rank_PhiTPhi"] = int((ev > ev[0] * 1e-10).sum())
    res[f"{name}_eig_max"] = float(ev[0])
    res[f"{name}_eig_at_rank_10"] = float(ev[9])
    res[f"{name}_eig_at_rank_100"] = float(ev[99])
    res[f"{name}_eig_at_rank_N"] = float(ev[N - 1])
    res[f"{name}_eig_median_nonzero"] = float(np.median(ev[:N]))
    res[f"{name}_eff_rank_participation"] = float(ev[:N].sum() ** 2 / (ev[:N] ** 2).sum())
    tn = np.linalg.norm(X, axis=1)
    qn = np.linalg.norm(Xq, axis=1)
    res[f"{name}_train_feat_norm_median"] = float(np.median(tn))
    res[f"{name}_train_feat_norm_max_over_median"] = float(tn.max() / np.median(tn))
    res[f"{name}_query_feat_norm_median"] = float(np.median(qn))
    # mean pairwise cosine among train features (shared direction?)
    Xn = X / tn[:, None]
    Qn = Xq / qn[:, None]
    C = Xn @ Xn.T
    res[f"{name}_mean_offdiag_cos_train_train"] = float((C.sum() - N) / (N * (N - 1)))
    res[f"{name}_mean_cos_query_train"] = float((Qn @ Xn.T).mean())
    top_dir = np.linalg.svd(Xn, full_matrices=False)[2][0]
    res[f"{name}_frac_train_energy_in_top_singular_dir"] = float(((Xn @ top_dir) ** 2).mean())

    # batch-position artefact: rows at the same batch position (index parity, batch_size=2,
    # shards are even-aligned) received identical diffusion noise at every timestep.
    tpar = np.arange(N) % 2
    qpar = np.arange(Xq.shape[0]) % 2
    cos_qt = Qn @ Xn.T
    same = (qpar[:, None] == tpar[None, :])
    res[f"{name}_cos_query_train_same_noise"] = float(cos_qt[same].mean())
    res[f"{name}_cos_query_train_diff_noise"] = float(cos_qt[~same].mean())
    cos_tt = C.copy()
    samett = (tpar[:, None] == tpar[None, :]) & ~np.eye(N, dtype=bool)
    difftt = (tpar[:, None] != tpar[None, :])
    res[f"{name}_cos_train_train_same_noise"] = float(cos_tt[samett].mean())
    res[f"{name}_cos_train_train_diff_noise"] = float(cos_tt[difftt].mean())

    # rescoring variants
    for lam in lams:
        Kinv = np.linalg.inv(K + lam * np.eye(k))
        sc = Xq @ Kinv @ X.T
        M = seller_matrix(sc, tid, qid)
        res[f"{name}_rescore_lambda_{lam:.0e}_per_seller_spearman"] = per_seller_spearman(M, A)
        res[f"{name}_rescore_lambda_{lam:.0e}_per_seller_spearman_decommoned"] = per_seller_spearman(M, A_dc)
    sc = Xq @ X.T
    M = seller_matrix(sc, tid, qid)
    res[f"{name}_rescore_raw_dot_per_seller_spearman"] = per_seller_spearman(M, A)
    res[f"{name}_rescore_raw_dot_per_seller_spearman_decommoned"] = per_seller_spearman(M, A_dc)
    sc = Qn @ Xn.T
    M = seller_matrix(sc, tid, qid)
    res[f"{name}_rescore_cosine_per_seller_spearman"] = per_seller_spearman(M, A)
    res[f"{name}_rescore_cosine_per_seller_spearman_decommoned"] = per_seller_spearman(M, A_dc)
    # also check the production scores are reproduced by the lambda=0.1 formula
    prod = np.array(np.memmap(OUT / runs[name] / "members/single/scores_query_x_train_raw.memmap",
                              dtype=np.float32, mode="r", shape=(Xq.shape[0], N)))
    Kinv = np.linalg.inv(K + 0.1 * np.eye(k))
    sc = Xq @ Kinv @ X.T
    res[f"{name}_prod_scores_reproduced_corr"] = float(np.corrcoef(prod.ravel(), sc.ravel())[0, 1])

# cross-run agreement at the track level (requires same train/query order)
(X1, Q1, t1, q1), (X2, Q2, t2, q2) = data["chunked"], data["full6h"]
res["crossrun_train_order_identical"] = bool(t1 == t2)
res["crossrun_query_order_identical"] = bool(q1 == q2)
tperm = [t2.index(t) for t in t1]
qperm = [q2.index(q) for q in q1]
X2, Q2 = X2[tperm], Q2[qperm]
t1par = np.arange(len(t1)) % 2
t2par = np.array(tperm) % 2
res["crossrun_frac_train_rows_same_batch_position"] = float((t1par == t2par).mean())
G1, G2 = X1 @ X1.T, X2 @ X2.T
off = ~np.eye(len(t1), dtype=bool)
res["crossrun_train_gram_offdiag_corr"] = float(np.corrcoef(G1[off], G2[off])[0, 1])
res["crossrun_train_sqnorm_corr"] = float(np.corrcoef(np.diag(G1), np.diag(G2))[0, 1])
H1, H2 = Q1 @ Q1.T, Q2 @ Q2.T
offq = ~np.eye(len(q1), dtype=bool)
res["crossrun_query_gram_offdiag_corr"] = float(np.corrcoef(H1[offq], H2[offq])[0, 1])
res["crossrun_query_sqnorm_corr"] = float(np.corrcoef(np.diag(H1), np.diag(H2))[0, 1])
R1, R2 = Q1 @ X1.T, Q2 @ X2.T
res["crossrun_query_train_rawdot_corr"] = float(np.corrcoef(R1.ravel(), R2.ravel())[0, 1])
for lam in [1e-1, 1e3, 1e7]:
    s1 = Q1 @ np.linalg.inv(X1.T @ X1 + lam * np.eye(X1.shape[1])) @ X1.T
    s2 = Q2 @ np.linalg.inv(X2.T @ X2 + lam * np.eye(X2.shape[1])) @ X2.T
    res[f"crossrun_trak_score_corr_lambda_{lam:.0e}"] = float(np.corrcoef(s1.ravel(), s2.ravel())[0, 1])

# EKFAC per-track score tails
E = torch.load(OUT / "ekfac_attribution_v100_layer_sharded/scores_train_x_query.pt").double().numpy()  # [N, Q]
absE = np.abs(E)
flat = np.sort(absE.ravel())[::-1]
res["ekfac_frac_abs_mass_top_0.1pct_entries"] = float(flat[: max(1, flat.size // 1000)].sum() / flat.sum())
res["ekfac_frac_abs_mass_top_1pct_entries"] = float(flat[: flat.size // 100].sum() / flat.sum())
row_mass = absE.sum(1)
srt = np.sort(row_mass)[::-1]
res["ekfac_frac_abs_mass_top_10_train_tracks"] = float(srt[:10].sum() / srt.sum())
res["ekfac_kurtosis_entries"] = float(((E - E.mean()) ** 4).mean() / E.var() ** 2)
# same statistic for D-TRAK production scores
D = np.array(np.memmap(OUT / "dtrak_attribution_v100_chunked/scores_query_x_train.memmap", dtype=np.float32,
                       mode="r", shape=(200, 1117))).T.astype(np.float64)
absD = np.abs(D)
flatD = np.sort(absD.ravel())[::-1]
res["dtrak_frac_abs_mass_top_1pct_entries"] = float(flatD[: flatD.size // 100].sum() / flatD.sum())
srtD = np.sort(absD.sum(1))[::-1]
res["dtrak_frac_abs_mass_top_10_train_tracks"] = float(srtD[:10].sum() / srtD.sum())
res["dtrak_kurtosis_entries"] = float(((D - D.mean()) ** 4).mean() / D.var() ** 2)
# EKFAC vs D-TRAK per-track agreement (same train axis order: candidate pool order)
eq = pd.read_csv(OUT / "ekfac_attribution_v100_layer_sharded/query_axis_manifest.csv")["filename"].tolist()
dq = pd.read_csv(OUT / "dtrak_attribution_v100_chunked/query_axis_manifest.csv")["filename"].tolist()
et = pd.read_csv(OUT / "ekfac_attribution_v100_layer_sharded/train_axis_manifest.csv")["relpath"].tolist()
dt = pd.read_csv(OUT / "dtrak_attribution_v100_chunked/train_axis_manifest.csv")["relpath"].tolist()
assert et == dt
perm = [eq.index(f) for f in dq]
E_al = E[:, perm]
res["ekfac_vs_dtrak_track_score_spearman"] = float(spearmanr(E_al.ravel(), D.ravel()).correlation)
res["ekfac_vs_dtrak_track_score_mean_per_query_spearman"] = float(np.nanmean(
    [spearmanr(E_al[:, q], D[:, q]).correlation for q in range(D.shape[1])]))

json.dump(res, open(DIAG / "dtrak_feature_analysis_results.json", "w"), indent=2)
for k_, v in res.items():
    print(f"{k_}: {v:.5g}" if isinstance(v, float) else f"{k_}: {v}")
