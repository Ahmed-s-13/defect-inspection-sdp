"""Shared pieces for the architecture B experiments (archB_choose.py, archB_report.py).

Architecture B = enhanced_res gc7 (7 classes, with no-defect) + architecture A's decision layer, quality gate off
(legacy), calibrated on d1_gc7_fix crops (calibration_gc7.json). Architecture A = v2 + calibration.json.
Model outputs (768-d features, probabilities, kNN distance) are cached per model and per set under
data/work/enhanced_res/archB/ (git-ignored); in-memory images (wallpaper patches, static) are recomputed.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch
from PIL import Image

import archA_d5 as A5
import archA_gates as G
import archA_static as ST
from arch_a import calibrate as C
from arch_a import runtime as R
from klshift.common import eval_transform
from model import EWORK, MAIN, MODELS, EnhancedRes, load_split

CACHE = EWORK / "archB"
OFF = MAIN / "data/work/enhanced_res/offtarget"
ARCH = {"A": ("v2", R.CALIBRATIONS["v2"]), "B": ("gc7", R.CALIBRATIONS["gc7"])}
_MODELS = {}


def model(name):
    if name not in _MODELS:
        _MODELS[name] = EnhancedRes(MODELS[name])
    return _MODELS[name]


def outputs(name, key, items):
    """(features, probs, knn distance) of crop paths (cached under CACHE/<model>_<key>.npz) or PIL images (not cached)."""
    m = model(name)
    if all(isinstance(x, (str, Path)) for x in items):
        CACHE.mkdir(parents=True, exist_ok=True)
        return C.outputs(m, [str(x) for x in items], CACHE / f"{name}_{key}.npz")
    tf, F, P, D = eval_transform(), [], [], []
    with torch.no_grad():
        for i in range(0, len(items), 64):
            x = torch.stack([tf(im.convert("RGB")) for im in items[i:i + 64]])
            f = m.features(x)
            F.append(f.cpu())
            P.append(m.proba_from_features(f).cpu())
            D.append(m.knn_distance(f, C.K).cpu())
    return torch.cat(F).numpy(), torch.cat(P).numpy().astype(float), torch.cat(D).numpy().astype(float)


def decide(arch, items, F, P, D):
    """Decisions of an architecture on items (paths or PIL images); the quality gate runs only if it is on."""
    crops = None
    if arch.use_quality:
        crops = [Image.open(x).convert("RGB") if isinstance(x, (str, Path)) else x for x in items]
    return arch.decide(P, crops=crops, knn_dist=D, feats=F)


def halves(data):
    """val / cal half / test half of a split, as in calibrate.py."""
    crops = load_split(data)
    va = crops[crops.split == "val"].reset_index(drop=True)
    te = crops[crops.split == "test"].reset_index(drop=True)
    te["half"] = C.halves(te)
    return va, te[te.half == "cal"].reset_index(drop=True), te[te.half == "test"].reset_index(drop=True)


def extra_sets(plain=False):
    """Sets outside D1: the off-target benchmark (with FGVC no-damage and the user photo), D5, wallpapers, static.
    plain: also the 28 Oct plain-surface negatives (Places365 plain scenes, DTD), tried for B and left out."""
    man = pd.read_csv(OFF / "manifest.csv")
    if not plain:
        man = man[~man.source.isin(["places365-plain", "dtd"])].reset_index(drop=True)
    man["path"] = [str(MAIN / p) for p in man.crop_path]
    d5, _ = A5.d5_crops()
    rng = np.random.default_rng(0)
    wall = G.offtarget_patches(G.N_OFF, rng)
    static = [im for ims in ST.nonsense(np.random.default_rng(0)).values() for im in ims]
    return man, d5, wall, static


def fam_mean(v, g):
    return float(pd.Series(np.asarray(v, float)).groupby(np.asarray(g)).mean().mean())


def family_ci(v, g, n=1000, seed=0):
    """95% interval of a crop-level rate, resampling whole families."""
    v, g = np.asarray(v, float), np.asarray(g)
    fams, inv = np.unique(g, return_inverse=True)
    s, c = np.bincount(inv, v), np.bincount(inv)
    rng = np.random.default_rng(seed)
    b = [s[k].sum() / c[k].sum() for k in (rng.integers(0, len(fams), len(fams)) for _ in range(n))]
    return float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))
