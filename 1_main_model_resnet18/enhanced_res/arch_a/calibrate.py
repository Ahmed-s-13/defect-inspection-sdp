"""Fit architecture A's numbers for enhanced_res and write calibration.json (read by runtime.py at prediction time).

From 1_main_model_resnet18/:
  python enhanced_res/arch_a/calibrate.py            # ~10 min on an M-series Mac (8-view features, cached after)

Nothing is trained; the enhanced_res weights are unchanged. Data (all d1_grouped, see cal_split.csv):
  val       (4,203 crops)   temperature T (minimum NLL), and the quality gate's reference statistics
  cal half  (1,838 crops,   CP thresholds q (family-weighted LAC at alpha 0.05 / 0.10 / 0.20), and the calibration
             88 families)   scores of both gates
  test half (2,417 crops,   only read to store the expected results, which enhanced_res/evaluate.py checks against
             87 families)
Re-run it when the model, the labels or the split change. The research behind every choice is in experiments/ and
results/ (README.md in this folder).
"""
import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))                # 1_main_model_resnet18

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader

from enhanced_res.arch_a import runtime as R
from enhanced_res.model import EWORK, MAIN, EnhancedRes, load_split
from klshift.common import CropDataset

SPLIT = HERE / "cal_split.csv"
CACHE = EWORK / "arch_a_outputs.npz"                    # git-ignored
ALPHA, ALPHAS = 0.10, (0.05, 0.10, 0.20)
ALPHA_Q, ALPHA_OOD, K = 0.10, 0.10, 5


def outputs(m, paths):
    """Probabilities and kNN distances for every crop, cached by crop list."""
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        if list(z["crop_path"]) == list(paths):
            return z["probs"], z["knn_dist"]
    P, D = [], []
    for i, (x, _) in enumerate(DataLoader(CropDataset(list(paths)), batch_size=64, num_workers=4)):
        f = m.features(x)
        P.append(m.proba_from_features(f).cpu())
        D.append(m.knn_distance(f, K).cpu())
        print(f"\r  features {min((i + 1) * 64, len(paths))}/{len(paths)}", end="", flush=True)
    print()
    probs, dist = torch.cat(P).numpy().astype(float), torch.cat(D).numpy().astype(float)
    np.savez(CACHE, crop_path=np.array(paths), probs=probs, knn_dist=dist)
    return probs, dist


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def g9(a):
    return [float(f"{v:.9g}") for v in np.asarray(a, float)]


def main():
    crops = load_split()
    va = crops[crops.split == "val"].reset_index(drop=True)
    sp = pd.read_csv(SPLIT)
    sp["crop_path"] = [str(MAIN / p) for p in sp.crop_path]
    te = crops[crops.split == "test"].merge(sp[["crop_path", "split"]].rename(columns={"split": "half"}),
                                            on="crop_path", validate="1:1").reset_index(drop=True)
    cal, tst = te[te.half == "cal"].reset_index(drop=True), te[te.half == "test"].reset_index(drop=True)
    print(f"val {len(va)} crops | cal half {len(cal)} crops, {cal.src_group.nunique()} families | "
          f"test half {len(tst)} crops, {tst.src_group.nunique()} families")

    m = EnhancedRes()
    classes = m.classes
    cid = {c: i for i, c in enumerate(classes)}
    paths = list(va.crop_path) + list(cal.crop_path) + list(tst.crop_path)
    P, D = outputs(m, paths)
    nv, nc = len(va), len(cal)
    sl = {"val": slice(0, nv), "cal": slice(nv, nv + nc), "test": slice(nv + nc, None)}
    y = {"val": va.cls.map(cid).to_numpy(), "cal": cal.cls.map(cid).to_numpy(), "test": tst.cls.map(cid).to_numpy()}
    g = {"cal": cal.src_group.to_numpy(), "test": tst.src_group.to_numpy()}

    # temperature on val
    r = minimize_scalar(lambda lt: nll(R.temper(P[sl["val"]], np.exp(lt)), y["val"]),
                        bounds=(np.log(0.05), np.log(20)), method="bounded")
    T = float(np.exp(r.x))
    Pc = R.temper(P[sl["cal"]], T)
    print(f"T = {T:.3f}  (val NLL {nll(P[sl['val']], y['val']):.3f} -> {r.fun:.3f})")

    # CP thresholds on the cal half
    q = {f"{a:.2f}": R.family_threshold(Pc, y["cal"], g["cal"], a) for a in ALPHAS}

    # quality gate: reference = val crops, calibration scores = cal half
    def stats(df):
        st = [R.quality_stats(Image.open(p).convert("RGB")) for p in df.crop_path]
        return {s: np.array([d[s] for d in st]) for s in R.STATS}
    print("  quality statistics ...", flush=True)
    ref = {s: np.sort(v) for s, v in stats(va).items()}
    ref = {s: np.asarray(g9(v)) for s, v in ref.items()}         # exactly what the JSON will hold
    q_cal, _ = R.quality_score(ref, stats(cal))

    cfg = {
        "about": "architecture A for enhanced_res: quality gate -> OOD gate -> temperature -> LAC CP set. Written by "
                 "enhanced_res/arch_a/calibrate.py; read by enhanced_res/arch_a/runtime.py.",
        "created": str(date.today()),
        "model": {"backbone": "enhanced_res_backbone.pt", "head": "enhanced_res_head.pt", "release": "v2.0-enhanced-res",
                  "knn_alpha": m.alpha, "tau": m.tau},
        "data": {"split": "d1_grouped", "cal_split": "enhanced_res/arch_a/cal_split.csv",
                 "val_crops": nv, "cal_crops": nc, "cal_families": int(cal.src_group.nunique()),
                 "test_half_crops": len(tst), "test_half_families": int(tst.src_group.nunique())},
        "classes": classes,
        "temperature": {"kind": "probability temperature: p ** (1/T), renormalised; fitted on val by NLL", "T": T},
        "cp": {"score": "LAC: 1 - tempered probability of the true class",
               "weighting": "every calibration family carries weight 1", "alpha": ALPHA, "q": q,
               "rule": "set = every class with tempered probability >= 1 - q[alpha]"},
        "quality_gate": {"alpha": ALPHA_Q, "stats": list(R.STATS), "sides": R.SIDES,
                         "score": "-log(min over stats of the tail probability vs the val reference)",
                         "reference": {s: list(v) for s, v in ref.items()},
                         "cal_scores": g9(q_cal), "cal_groups": [int(v) for v in g["cal"]]},
        "ood_gate": {"alpha": ALPHA_OOD, "k": K, "score": "1 - mean cosine similarity to the k nearest train features",
                     "cal_scores": g9(D[sl["cal"]]), "cal_groups": [int(v) for v in g["cal"]]},
    }
    path = HERE / "calibration.json"
    path.write_text(json.dumps(cfg))

    # expected results on the test half (enhanced_res/evaluate.py compares against these)
    a = R.ArchA(path)
    print("  checking the test half ...", flush=True)
    dec = a.decide(P[sl["test"]], crops=[Image.open(p).convert("RGB") for p in tst.crop_path], knn_dist=D[sl["test"]])
    exp = R.summarise(dec, a, P[sl["test"]], y["test"], g["test"])
    cfg["expected_test_half"] = exp
    path.write_text(json.dumps(cfg))
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB)")
    print(json.dumps({"T": T, "q": q, **exp}, indent=2))


if __name__ == "__main__":
    main()
