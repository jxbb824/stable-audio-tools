#!/usr/bin/env python3
"""Tiny D-TRAK feature-stability validation on ONE GPU (a few examples only).

Re-uses the production feature code (scripts/dtrak_extract_features.py) and the production
chunked Gaussian JL projection (seed 42, 32 chunks of 33,554,432), so features live in the same
2048-d space as sao_ip_100/outputs/dtrak_attribution_v100_chunked.

For a handful of train tracks / queries, computes K=10 D-TRAK features under controlled changes
(caption draw, noise draw, crop window, measurement f) and reports cosines.  Writes only to
estimator_diagnosis/.  Stops early at a wall-clock budget and saves results incrementally.
"""
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

T0 = time.time()
BUDGET_S = float(sys.argv[1]) if len(sys.argv) > 1 else 45 * 60
import os as _os
if _os.environ.get("DEADLINE_EPOCH"):
    BUDGET_S = min(BUDGET_S, float(_os.environ["DEADLINE_EPOCH"]) - T0)

ROOT = Path("/ocean/projects/mth260005p/xjiang6/stable-audio-tools")
DIAG = ROOT / "estimator_diagnosis"
OUT = ROOT / "sao_ip_100/outputs"
PROD = OUT / "dtrak_attribution_v100_chunked/members/single"
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("dtx", ROOT / "scripts/dtrak_extract_features.py")
dtx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dtx)

from stable_audio_tools.data.dataset import create_dataloader_from_config  # noqa: E402
from stable_audio_tools.inference.sampling import get_alphas_sigmas  # noqa: E402
from stable_audio_tools.models import create_model_from_config  # noqa: E402
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict  # noqa: E402

import importlib  # noqa: E402
md_fma = importlib.import_module("stable_audio_tools.configs.dataset_configs.custom_metadata.custom_md_fma")


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


device = torch.device("cuda:0")
model_config = json.load(open(ROOT / "model_config_freeze_vae.json"))
K = 10
TIMESTEPS = dtx.select_timesteps(K, 1000, "uniform")

# ---------------- examples ----------------
items = {it["prompt_id"]: it for it in json.load(open(OUT / "generated_queries/prompt_manifest.json"))["items"]}
sel = pd.read_csv(OUT / "selection/selected_tracks.csv")
canon = dict(zip(sel["relpath"], sel["prompt"]))
QUERIES = [156, 154]
TRACKS = {"t_a": items[156]["source_relpath"], "t_b": items[154]["source_relpath"], "t_c": "05/91905.mp3"}

prod_tids = ["/".join(t.split("/")[-2:]) for t in (PROD / "train_features_raw.memmap.ids.txt").read_text().splitlines()]
prod_qids = [Path(q).name for q in (PROD / "query_features.memmap.ids.txt").read_text().splitlines()]
prod_X = np.array(np.memmap(PROD / "train_features_raw.memmap", dtype=np.float32, mode="r", shape=(1117, 2048)))
prod_Q = np.array(np.memmap(PROD / "query_features.memmap", dtype=np.float32, mode="r", shape=(200, 2048)))
# production batch position (batch_size=2, shards even-aligned) -> identical noise slice
prod_pos_track = {k: prod_tids.index(r) % 2 for k, r in TRACKS.items()}
prod_pos_query = {q: prod_qids.index(f"query_{q:04d}.mp3") % 2 for q in QUERIES}

# ---------------- data ----------------
train_cfg = json.load(open(OUT / "dtrak_attribution_v100_chunked/resolved_train_dataset_config.json"))
inc = DIAG / "gpu_validation_include_paths.txt"
inc.write_text("\n".join(str(ROOT / "dataset/small-20000/audio" / r) for r in TRACKS.values()) + "\n")
train_cfg["include_paths_file"] = str(inc)


def make_ds(cfg, random_crop):
    cfg = dict(cfg)
    cfg["random_crop"] = random_crop
    dl = create_dataloader_from_config(cfg, batch_size=1, num_workers=1, sample_rate=model_config["sample_rate"],
                                       sample_size=model_config["sample_size"], audio_channels=2, shuffle=False)
    return dl.dataset


ds_first = make_ds(train_cfg, False)
ds_rand = make_ds(train_cfg, True)
q_cfg = json.load(open(ROOT / "sao_ip_100/configs/query_dataset.json"))
ds_q = make_ds(q_cfg, False)


def idx_of(ds, suffix):
    return [i for i, f in enumerate(ds.filenames) if f.endswith(suffix)][0]


# ---------------- model ----------------
log("loading model")
diffusion = create_model_from_config(model_config)
copy_state_dict(diffusion, load_ckpt_state_dict(str(OUT / "models/0/model.ckpt")))
diffusion = diffusion.to(device).eval()
diffusion.pretransform.eval()
model = diffusion.model
param_list = [p for n, p in model.named_parameters() if p.requires_grad]
grad_dim = sum(p.numel() for p in param_list)
for p in model.parameters():
    p.requires_grad_(False)
for p in param_list:
    p.requires_grad_(True)
chunks = dtx.build_gradient_chunks(param_list, 33554432)
projectors, chunk_seeds = dtx.create_chunked_projectors(chunks, 2048, 42, device, 8)
prod_meta = json.load(open(PROD / "train_features_raw.memmap.meta.json"))
log(f"grad_dim={grad_dim} (prod {prod_meta['grad_dim']}), chunks={len(chunks)}, "
    f"chunk_seeds_match_prod={chunk_seeds == prod_meta.get('projection_chunk_seeds', chunk_seeds)}")
log(f"dist_shift={diffusion.dist_shift}, objective={diffusion.diffusion_objective}")

latent_cache = {}


vae_diag = {}


def encode_like_prod(audio, pos):
    """Reproduce production latent: VAE encode samples from the posterior with CUDA RNG, which in
    production (batch b>=1) sits right after set_seeds(900) + randn([2,64,L])."""
    x = audio[None].to(device).repeat(2, 1, 1)
    with torch.no_grad():
        dtx.set_seeds(0 * 1000 + 900)
        _ = torch.randn((2, 64, x.shape[-1] // 2048), device=device)
        z = diffusion.pretransform.encode(x)[pos:pos + 1]
        pre = diffusion.pretransform.model.encoder(x[:1])
        mean, scale = pre.chunk(2, dim=1)
        std = torch.nn.functional.softplus(scale) + 1e-4
        vae_diag.setdefault("posterior_std_over_latent_std", []).append(float(std.mean() / mean.std()))
        vae_diag.setdefault("posterior_std_mean", []).append(float(std.mean()))
    return z


def get_example(kind, key, crop_seed=None, pos=0):
    if kind == "query":
        audio, md = ds_q[idx_of(ds_q, f"query_{key:04d}.mp3")]
    else:
        rel = TRACKS[key]
        if crop_seed is None:
            audio, md = ds_first[idx_of(ds_first, rel)]
        else:
            random.seed(crop_seed)
            audio, md = ds_rand[idx_of(ds_rand, rel)]
    ck = (kind, key, crop_seed, pos)
    if ck not in latent_cache:
        latent_cache[ck] = encode_like_prod(audio, pos)
    return latent_cache[ck], md


def feature(latent, md, f_name, e_seed, pos):
    cond = dtx.get_condition_kwargs(diffusion, [md], device)
    per_t = torch.zeros(K, 2048, device=device)
    gnorm, fvals = [], []
    for ti, t_int in enumerate(TIMESTEPS):
        dtx.set_seeds(e_seed * 1000 + t_int)
        t = torch.full((1,), t_int / 1000.0, device=device)
        # production draws randn_like on the [batch=2, 64, L] latent and slices the batch position
        noise = torch.randn((2,) + tuple(latent.shape[1:]), device=device)[pos:pos + 1]
        a, s = get_alphas_sigmas(t)
        a, s = a[:, None, None], s[:, None, None]
        noised = latent * a + noise * s
        target = noise * a - latent * s
        kw = {"cfg_dropout_prob": 0.0}
        kw.update({k: v for k, v in cond.items() if v is not None})
        pred = model(noised, t, **kw)
        val = dtx.loss_from_prediction(pred, target, f_name)
        grads = torch.autograd.grad(val, param_list)
        gnorm.append(float(torch.sqrt(sum((g.float() ** 2).sum() for g in grads))))
        fvals.append(float(val))
        per_t[ti] = dtx.project_gradient_chunks(grads, chunks, projectors, 2048, 0, device, 8)
        del grads, pred, val
    return per_t.cpu().numpy(), gnorm, fvals


def canon_md(md, rel):
    md = dict(md)
    md["prompt"] = canon[rel]
    return md


def random_caption_md(md, seed):
    random.seed(seed)
    md = dict(md)
    md["prompt"] = md_fma.get_custom_metadata({"path": md["path"]}, None)["prompt"]
    return md


# job list in priority order: (name, kind, key, f, e_seed, caption_mode, crop_seed)
jobs = [
    ("q156_loss_e0", "query", 156, "loss", 0, None, None),
    ("t_a_loss_cap1_e0", "train", "t_a", "loss", 0, ("rand", 1), None),
    ("t_a_loss_cap2_e0", "train", "t_a", "loss", 0, ("rand", 2), None),
    ("t_a_loss_canon_e0", "train", "t_a", "loss", 0, ("canon",), None),
    ("t_a_loss_canon_e1", "train", "t_a", "loss", 1, ("canon",), None),
    ("t_a_loss_canon_e0_crop", "train", "t_a", "loss", 0, ("canon",), 7),
    ("q156_loss_e1", "query", 156, "loss", 1, None, None),
    ("t_a_msl2_canon_e0", "train", "t_a", "mean-squared-l2-norm", 0, ("canon",), None),
    ("t_a_msl2_canon_e1", "train", "t_a", "mean-squared-l2-norm", 1, ("canon",), None),
    ("q156_msl2_e0", "query", 156, "mean-squared-l2-norm", 0, None, None),
    ("q156_msl2_e1", "query", 156, "mean-squared-l2-norm", 1, None, None),
    ("t_c_loss_canon_e0", "train", "t_c", "loss", 0, ("canon",), None),
    ("t_c_msl2_canon_e0", "train", "t_c", "mean-squared-l2-norm", 0, ("canon",), None),
    ("q154_loss_e0", "query", 154, "loss", 0, None, None),
    ("t_b_loss_canon_e0", "train", "t_b", "loss", 0, ("canon",), None),
    ("t_b_msl2_canon_e0", "train", "t_b", "mean-squared-l2-norm", 0, ("canon",), None),
    ("q154_msl2_e0", "query", 154, "mean-squared-l2-norm", 0, None, None),
    ("t_b_loss_cap1_e0", "train", "t_b", "loss", 0, ("rand", 1), None),
    ("t_b_loss_canon_e1", "train", "t_b", "loss", 1, ("canon",), None),
    ("t_b_msl2_canon_e1", "train", "t_b", "mean-squared-l2-norm", 1, ("canon",), None),
    ("t_c_loss_canon_e1", "train", "t_c", "loss", 1, ("canon",), None),
    ("t_c_msl2_canon_e1", "train", "t_c", "mean-squared-l2-norm", 1, ("canon",), None),
]

feats, info = {}, {}
per_job_s = None
for name, kind, key, f_name, e_seed, cap, crop in jobs:
    elapsed = time.time() - T0
    if per_job_s is not None and elapsed + 1.2 * per_job_s > BUDGET_S:
        log(f"budget reached, stopping before {name}")
        break
    t_start = time.time()
    pos = prod_pos_track[key] if kind == "train" else prod_pos_query[key]
    latent, md = get_example(kind, key, crop, pos)
    if kind == "train":
        rel = TRACKS[key]
        md = canon_md(md, rel) if cap[0] == "canon" else random_caption_md(md, cap[1])
    pt, gn, fv = feature(latent, md, f_name, e_seed, pos)
    feats[name] = pt
    info[name] = {"prompt": md["prompt"], "seconds_start": md.get("seconds_start"),
                  "seconds_total": md.get("seconds_total"), "pos": pos, "grad_norm_per_t": gn,
                  "f_per_t": fv, "sec": time.time() - t_start}
    per_job_s = time.time() - t_start if per_job_s is None else 0.5 * per_job_s + 0.5 * (time.time() - t_start)
    log(f"done {name} in {time.time() - t_start:.1f}s | prompt={md['prompt'][:80]!r}")
    np.savez(DIAG / "gpu_validation_features.npz", **feats)
    json.dump(info, open(DIAG / "gpu_validation_info.json", "w"), indent=1)


# ---------------- comparisons ----------------
def fv(n):
    return feats[n].mean(0)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


res = {"prod_pos_track": prod_pos_track, "prod_pos_query": {str(k): v for k, v in prod_pos_query.items()},
       "timesteps": TIMESTEPS}
pairs = [
    ("pipeline_repro: q156 loss e0 vs production query feature", "q156_loss_e0", ("prodQ", 156)),
    ("pipeline_repro: q154 loss e0 vs production query feature", "q154_loss_e0", ("prodQ", 154)),
    ("caption: t_a randcap1 vs randcap2 (same noise, same crop)", "t_a_loss_cap1_e0", "t_a_loss_cap2_e0"),
    ("caption: t_a randcap1 vs production feature", "t_a_loss_cap1_e0", ("prodT", "t_a")),
    ("caption: t_a randcap2 vs production feature", "t_a_loss_cap2_e0", ("prodT", "t_a")),
    ("caption: t_a canonical vs production feature", "t_a_loss_canon_e0", ("prodT", "t_a")),
    ("caption: t_a canonical vs randcap1", "t_a_loss_canon_e0", "t_a_loss_cap1_e0"),
    ("caption: t_b randcap1 vs production feature", "t_b_loss_cap1_e0", ("prodT", "t_b")),
    ("caption: t_b canonical vs production feature", "t_b_loss_canon_e0", ("prodT", "t_b")),
    ("noise[loss]: t_a canon e0 vs e1", "t_a_loss_canon_e0", "t_a_loss_canon_e1"),
    ("noise[loss]: t_b canon e0 vs e1", "t_b_loss_canon_e0", "t_b_loss_canon_e1"),
    ("noise[loss]: t_c canon e0 vs e1", "t_c_loss_canon_e0", "t_c_loss_canon_e1"),
    ("noise[loss]: q156 e0 vs e1", "q156_loss_e0", "q156_loss_e1"),
    ("crop[loss]: t_a canon e0 first-window vs random-window", "t_a_loss_canon_e0", "t_a_loss_canon_e0_crop"),
    ("noise[msl2]: t_a canon e0 vs e1", "t_a_msl2_canon_e0", "t_a_msl2_canon_e1"),
    ("noise[msl2]: t_b canon e0 vs e1", "t_b_msl2_canon_e0", "t_b_msl2_canon_e1"),
    ("noise[msl2]: t_c canon e0 vs e1", "t_c_msl2_canon_e0", "t_c_msl2_canon_e1"),
    ("noise[msl2]: q156 e0 vs e1", "q156_msl2_e0", "q156_msl2_e1"),
    ("align[loss]: q156 vs its source t_a (canon, e0)", "q156_loss_e0", "t_a_loss_canon_e0"),
    ("align[loss]: q156 vs its source t_a (canon, noise e1 vs e0)", "q156_loss_e1", "t_a_loss_canon_e0"),
    ("align[loss]: q156 vs its source t_a (random caption, e0)", "q156_loss_e0", "t_a_loss_cap1_e0"),
    ("align[loss]: q156 vs control t_c (canon, e0)", "q156_loss_e0", "t_c_loss_canon_e0"),
    ("align[loss]: q156 vs other-source t_b (canon, e0)", "q156_loss_e0", "t_b_loss_canon_e0"),
    ("align[msl2]: q156 vs its source t_a (canon, e0)", "q156_msl2_e0", "t_a_msl2_canon_e0"),
    ("align[msl2]: q156 vs its source t_a (canon, noise e1 vs e0)", "q156_msl2_e1", "t_a_msl2_canon_e0"),
    ("align[msl2]: q156 vs control t_c (canon, e0)", "q156_msl2_e0", "t_c_msl2_canon_e0"),
    ("align[msl2]: q156 vs other-source t_b (canon, e0)", "q156_msl2_e0", "t_b_msl2_canon_e0"),
    ("align[loss]: q154 vs its source t_b (canon, e0)", "q154_loss_e0", "t_b_loss_canon_e0"),
    ("align[loss]: q154 vs control t_c (canon, e0)", "q154_loss_e0", "t_c_loss_canon_e0"),
    ("align[loss]: q154 vs other-source t_a (canon, e0)", "q154_loss_e0", "t_a_loss_canon_e0"),
    ("align[msl2]: q154 vs its source t_b (canon, e0)", "q154_msl2_e0", "t_b_msl2_canon_e0"),
    ("align[msl2]: q154 vs control t_c (canon, e0)", "q154_msl2_e0", "t_c_msl2_canon_e0"),
    ("align[msl2]: q154 vs other-source t_a (canon, e0)", "q154_msl2_e0", "t_a_msl2_canon_e0"),
]


def resolve(x):
    if isinstance(x, tuple) and x[0] == "prodQ":
        return prod_Q[prod_qids.index(f"query_{x[1]:04d}.mp3")].astype(np.float64), None
    if isinstance(x, tuple) and x[0] == "prodT":
        return prod_X[prod_tids.index(TRACKS[x[1]])].astype(np.float64), None
    if x not in feats:
        return None, None
    return fv(x), feats[x]


cmp = {}
for label, a, b in pairs:
    va, pa = resolve(a)
    vb, pb = resolve(b)
    if va is None or vb is None:
        continue
    entry = {"cos_K_avg": cos(va, vb)}
    if pa is not None and pb is not None:
        entry["cos_per_t"] = [round(cos(pa[i], pb[i]), 3) for i in range(K)]
    cmp[label] = entry
res["comparisons"] = cmp

# rank of the source track among the 1117 production train features (loss f, same JL space)
Xn = prod_X / np.linalg.norm(prod_X, axis=1, keepdims=True)
lam = 0.1
Kinv = np.linalg.inv(prod_X.astype(np.float64).T @ prod_X.astype(np.float64) + lam * np.eye(2048))
for q, tk in [(156, "t_a"), (154, "t_b")]:
    name = f"q{q}_loss_e0"
    if name in feats:
        v = fv(name)
        c = Xn @ (v / np.linalg.norm(v))
        trak = (prod_X.astype(np.float64) @ Kinv) @ v
        src = prod_tids.index(TRACKS[tk])
        res[f"{name}: source-track rank among 1117 prod train feats (cosine)"] = int((c > c[src]).sum() + 1)
        res[f"{name}: source-track rank among 1117 prod train feats (TRAK lambda=0.1)"] = int((trak > trak[src]).sum() + 1)
        same_pos = np.arange(1117) % 2 == prod_pos_query[q]
        res[f"{name}: mean cos to prod train feats, same-noise rows"] = float(c[same_pos].mean())
        res[f"{name}: mean cos to prod train feats, diff-noise rows"] = float(c[~same_pos].mean())
res["vae_posterior"] = vae_diag
res["prod_row_index_track"] = {k: prod_tids.index(r) for k, r in TRACKS.items()}
res["timing_s_per_feature"] = {k: round(v["sec"], 1) for k, v in info.items()}
res["total_wall_s"] = time.time() - T0
json.dump(res, open(DIAG / "gpu_validation_results.json", "w"), indent=1)
log(json.dumps(res, indent=1))
