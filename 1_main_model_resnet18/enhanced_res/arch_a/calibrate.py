"""Fit architecture A's numbers for enhanced_res and write calibration_gc7.json / calibration.json (read by runtime.py).

From 1_main_model_resnet18/:
  python enhanced_res/arch_a/calibrate.py            # the main model gc7 -> calibration_gc7.json (~10 min, cached after)
  python enhanced_res/arch_a/calibrate.py --model v2 # the 6-class v2 -> calibration.json (what trt_enhanced_res reads)

gc7 uses d1_gc7: its no-defect crops join the val set (temperature, quality reference) and the test halves, each crop
in its photo family's half of cal_split.csv. The OOD gate's bank is gc7's own train features, no-defect included.

Nothing is trained; the enhanced_res weights are unchanged. Data (all d1_grouped, see cal_split.csv):
  val       (4,203 crops)   temperature T (minimum NLL), and the quality gate's reference statistics
  cal half  (1,838 crops,   CP thresholds q (family-weighted LAC at alpha 0.05 / 0.10 / 0.20), and the calibration
             88 families)   scores of both gates
  (Pooling val into the calibration was tried on 27 October and rolled back: experiments/archA_pooled_cal.py.)
  off-target (local only)   the AoN logistic regression: D1 train crops vs people, faces, helmets,
                            hands, gloves, tools, vehicles and industrial / hangar / airport scenes from
                            experiments/offtarget_data.py (Open Images + Places365, not in git). Without that data
                            the regression stored in calibration.json is reused.
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
from enhanced_res.model import EWORK, MAIN, MODELS, EnhancedRes, load_split
from klshift.common import CropDataset

SPLIT = HERE / "cal_split.csv"
CACHE = {"v2": EWORK / "arch_a_outputs.npz", "gc7": EWORK / "arch_a_outputs_gc7.npz"}   # git-ignored
ALPHA, ALPHAS = 0.10, (0.05, 0.10, 0.20)
ALPHA_Q, ALPHA_OOD, K = 0.05, 0.08, 5      # quality 0.05 (28 Oct; experiments/archA_quality_alpha.py). OOD: kNN + AoN, Bonferroni (0.04 each). Checked without the test
                                           # half: 500 halvings of the cal half stop 12.9% of families (budget 15%;
                                           # experiments/archA_alpha_check.py). Worst case by the union bound: 0.05 + 0.08 = 13%.
QUALITY_SIDES = {**R.SIDES, "sharpness": "both"}     # also flag far-too-sharp crops (noise), 2026-10-27
OFFTARGET = EWORK / "offtarget/manifest.csv"
OFF_CACHE = {"v2": EWORK / "arch_a_offtarget_feats.npz", "gc7": EWORK / "arch_a_offtarget_feats_gc7.npz"}


def outputs(m, paths, cache):
    """Features, probabilities and kNN distances for every crop, cached by crop list."""
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        if list(z["crop_path"]) == list(paths) and "feats" in z:
            return z["feats"], z["probs"], z["knn_dist"]
    F, P, D = [], [], []
    for i, (x, _) in enumerate(DataLoader(CropDataset(list(paths)), batch_size=64, num_workers=4)):
        f = m.features(x)
        F.append(f.cpu())
        P.append(m.proba_from_features(f).cpu())
        D.append(m.knn_distance(f, K).cpu())
        print(f"\r  features {min((i + 1) * 64, len(paths))}/{len(paths)}", end="", flush=True)
    print()
    feats = torch.cat(F).numpy().astype(np.float32)
    probs, dist = torch.cat(P).numpy().astype(float), torch.cat(D).numpy().astype(float)
    np.savez(cache, crop_path=np.array(paths), feats=feats, probs=probs, knn_dist=dist)
    return feats, probs, dist


def fit_aon(m, old, model):
    """Logistic regression on the 768-d features: D1 train crops (0) vs off-target crops (1)."""
    from sklearn.linear_model import LogisticRegression
    if not OFFTARGET.exists():
        if old is None:
            raise SystemExit(f"{OFFTARGET} missing and no previous regression: run experiments/offtarget_data.py")
        print("  off-target data not found: reusing the AoN regression from calibration.json")
        return {k: v for k, v in old.items() if k != "cal_scores"}
    man = pd.read_csv(OFFTARGET)
    man = man[man.role.isin(["fit", "test"]) & (man.source != "user")]
    neg, _, _ = outputs(m, [str(MAIN / p) for p in man.crop_path], OFF_CACHE[model])
    head = torch.load(MODELS[model] / "enhanced_res_head.pt", map_location="cpu")
    pos = head["knn_bank"].float().numpy()                      # the model's train crops' features
    X = np.concatenate([pos, neg])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000)
    clf.fit((X - mu) / sd, np.r_[np.zeros(len(pos)), np.ones(len(neg))])
    print(f"  AoN: {len(pos)} D1 train crops vs {len(neg)} off-target crops "
          f"({man.category.nunique()} categories)")
    return {"about": "P(not aircraft skin) = sigmoid(((f - mu) / sd) @ coef + intercept); f = EnhancedRes.features",
            "negatives": sorted(man.category.unique().tolist()),
            "mu": g9(mu), "sd": g9(sd), "coef": g9(clf.coef_[0]), "intercept": float(clf.intercept_[0])}


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def g9(a):
    return [float(f"{v:.9g}") for v in np.asarray(a, float)]


def halves(test_crops):
    """cal / test half of every test crop: cal_split.csv's row, else (no-defect crops) its photo family's half."""
    sp = pd.read_csv(SPLIT)
    sp["crop_path"] = [str(MAIN / p) for p in sp.crop_path]
    fam = sp.drop_duplicates("src_group").set_index("src_group").split
    h = test_crops[["crop_path"]].merge(sp[["crop_path", "split"]], on="crop_path", how="left", validate="1:1").split
    h = h.fillna(test_crops.src_group.map(fam).reset_index(drop=True))
    assert h.notna().all(), "a test crop whose photo family is not in cal_split.csv"
    return h.to_numpy()


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["gc7", "v2"], default="gc7")
    model = ap.parse_args().model
    crops = load_split("d1_gc7" if model == "gc7" else "d1_grouped")
    va = crops[crops.split == "val"].reset_index(drop=True)
    te = crops[crops.split == "test"].reset_index(drop=True)
    te["half"] = halves(te)
    cal, tst = te[te.half == "cal"].reset_index(drop=True), te[te.half == "test"].reset_index(drop=True)
    print(f"val {len(va)} crops | cal half {len(cal)} crops, {cal.src_group.nunique()} families | "
          f"test half {len(tst)} crops, {tst.src_group.nunique()} families")

    m = EnhancedRes(MODELS[model])
    classes = m.classes
    cid = {c: i for i, c in enumerate(classes)}
    paths = list(va.crop_path) + list(cal.crop_path) + list(tst.crop_path)
    Fe, P, D = outputs(m, paths, CACHE[model])
    nv, nc = len(va), len(cal)
    sl = {"val": slice(0, nv), "cal": slice(nv, nv + nc), "test": slice(nv + nc, None)}
    y = {"val": va.cls.map(cid).to_numpy(), "cal": cal.cls.map(cid).to_numpy(), "test": tst.cls.map(cid).to_numpy()}
    g = {"val": va.src_group.to_numpy(), "cal": cal.src_group.to_numpy(), "test": tst.src_group.to_numpy()}

    # temperature on val
    r = minimize_scalar(lambda lt: nll(R.temper(P[sl["val"]], np.exp(lt)), y["val"]),
                        bounds=(np.log(0.05), np.log(20)), method="bounded")
    T = float(np.exp(r.x))
    print(f"T = {T:.3f}  (val NLL {nll(P[sl['val']], y['val']):.3f} -> {r.fun:.3f})")

    # CP thresholds on the cal half
    Pc = R.temper(P[sl["cal"]], T)
    q = {f"{a:.2f}": R.family_threshold(Pc, y["cal"], g["cal"], a) for a in ALPHAS}

    # quality gate: reference = val crops, calibration scores = cal half
    def stats(df):
        st = [R.quality_stats(Image.open(p).convert("RGB")) for p in df.crop_path]
        return {s: np.array([d[s] for d in st]) for s in R.STATS}
    print("  quality statistics ...", flush=True)
    ref = {s: np.sort(v) for s, v in stats(va).items()}
    ref = {s: np.asarray(g9(v)) for s, v in ref.items()}         # exactly what the JSON will hold
    q_cal, _ = R.quality_score(ref, stats(cal), QUALITY_SIDES)

    # unfamiliar-input gate, second part: AoN
    path = R.CALIBRATIONS[model]
    og = json.loads(path.read_text())["ood_gate"] if path.exists() else {}
    old = og.get("aon", og.get("aircraft_or_not"))                 # "aircraft_or_not": the key before 28 Oct
    lr = fit_aon(m, old, model)
    z = (Fe[sl["cal"]] - np.asarray(lr["mu"])) / np.asarray(lr["sd"])
    lr["cal_scores"] = g9(1 / (1 + np.exp(-(z @ np.asarray(lr["coef"]) + lr["intercept"]))))

    cfg = {
        "about": "architecture A for enhanced_res: quality gate and OOD gate (in parallel; re-aim wins if both fire) -> temperature -> LAC CP set. Written by "
                 "enhanced_res/arch_a/calibrate.py; read by enhanced_res/arch_a/runtime.py.",
        "created": str(date.today()),
        "model": {"version": model,
                  "backbone": ("gc7/" if model == "gc7" else "") + "enhanced_res_backbone.pt",
                  "head": ("gc7/" if model == "gc7" else "") + "enhanced_res_head.pt",
                  "release": "v3.0-gc7" if model == "gc7" else "v2.0-enhanced-res",
                  "knn_alpha": m.alpha, "tau": m.tau},
        "data": {"split": "d1_gc7" if model == "gc7" else "d1_grouped", "cal_split": "enhanced_res/arch_a/cal_split.csv",
                 "val_crops": nv, "cal_crops": nc, "cal_families": int(cal.src_group.nunique()),
                 "test_half_crops": len(tst), "test_half_families": int(tst.src_group.nunique())},
        "classes": classes,
        "temperature": {"kind": "probability temperature: p ** (1/T), renormalised; fitted on val by NLL", "T": T},
        "cp": {"score": "LAC: 1 - tempered probability of the true class",
               "weighting": "every calibration family carries weight 1", "alpha": ALPHA, "q": q,
               "rule": "set = every class with tempered probability >= 1 - q[alpha]"},
        "quality_gate": {"alpha": ALPHA_Q, "stats": list(R.STATS), "sides": QUALITY_SIDES,
                         "score": "-log(min over stats of the tail probability vs the val reference)",
                         "reference": {s: list(v) for s, v in ref.items()},
                         "cal_scores": g9(q_cal), "cal_groups": [int(v) for v in g["cal"]]},
        "ood_gate": {"alpha": ALPHA_OOD, "combine": "Bonferroni: p = 2 x min(p_knn, p_aon)",
                     "k": K, "score": "1 - mean cosine similarity to the k nearest train features",
                     "cal_scores": g9(D[sl["cal"]]), "cal_groups": [int(v) for v in g["cal"]],
                     "aon": lr},
    }
    path.write_text(json.dumps(cfg))

    # expected results on the test half (enhanced_res/evaluate.py compares against these)
    a = R.ArchA(path)
    print("  checking the test half ...", flush=True)
    dec = a.decide(P[sl["test"]], crops=[Image.open(p).convert("RGB") for p in tst.crop_path], knn_dist=D[sl["test"]],
                   feats=Fe[sl["test"]])
    exp = R.summarise(dec, a, P[sl["test"]], y["test"], g["test"])
    cfg["expected_test_half"] = exp
    path.write_text(json.dumps(cfg))
    print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} kB)")
    print(json.dumps({"T": T, "q": q, **exp}, indent=2))


if __name__ == "__main__":
    main()
