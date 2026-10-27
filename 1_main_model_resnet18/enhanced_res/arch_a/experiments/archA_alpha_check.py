"""OOD alpha for architecture A without the test half (enhanced_res v2).

From 1_main_model_resnet18/ (after calibrate.py --model v2 and archA_offtarget_check.py, whose caches it reuses):
  python enhanced_res/arch_a/experiments/archA_alpha_check.py

The test half (87 families) is never read here: it is kept for the final report only.
The rule: alpha_q + alpha_ood = 0.15, the false-reject budget. Each gate's conformal p-value is valid, so by the union
bound the share of normal families stopped is at most alpha_q + alpha_ood in expectation, with no tuning. Checked two
ways that don't use the test half:
  1. inside the cal half: 500 random halvings of its 88 families; calibrate on one half, measure the other
  2. what each alpha lets through on the off-target sets (benchmark photos, wallpapers, static) and stops on D5
Writes enhanced_res/arch_a/results/archA/alpha_check/README.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from PIL import Image

import archA_pooled_cal as PC
from arch_a import calibrate as C
from arch_a import runtime as R
from model import load_split

ALPHAS_OOD = (0.03, 0.04, 0.05, 0.06, 0.07, 0.08)
N, BUDGET = 500, 0.15
OUT = HERE.parent / "results" / "archA" / "alpha_check"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = R.ArchA(R.CALIBRATIONS["v2"])
    crops = load_split("d1_grouped")
    va = crops[crops.split == "val"].reset_index(drop=True)
    te = crops[crops.split == "test"].reset_index(drop=True)
    te["half"] = C.halves(te)
    cal = te[te.half == "cal"].reset_index(drop=True)                 # the test half is not loaded
    nv, nc = len(va), len(cal)
    z = np.load(C.CACHE["v2"], allow_pickle=True)                       # rows: val, cal half, test half
    assert list(z["crop_path"][:nv + nc]) == list(va.crop_path) + list(cal.crop_path)
    Fe, P, D = z["feats"][nv:nv + nc], z["probs"][nv:nv + nc], z["knn_dist"][nv:nv + nc]
    cid = {c: i for i, c in enumerate(a.classes)}
    y, g = cal.cls.map(cid).to_numpy(), cal.src_group.to_numpy()
    air = a.aon_score(Fe)
    ref = {s: np.asarray(v) for s, v in a.ref.items()}
    st = [R.quality_stats(Image.open(p).convert("RGB")) for p in cal.crop_path]
    st = {s: np.array([d[s] for d in st]) for s in R.STATS}
    qs = R.quality_score(ref, st, a.sides)[0]

    rng = np.random.default_rng(0)
    fams = np.unique(g)
    rows = {ao: [] for ao in ALPHAS_OOD}
    for _ in range(N):
        h = np.isin(g, rng.choice(fams, len(fams) // 2, replace=False))
        c, m = h, ~h
        for ao in ALPHAS_OOD:
            G = PC.Gates(a, P[c], y[c], g[c], D[c], air[c], g[c], ref, qs[c], g[c], ao)
            act, _ = G.actions(P[m], D[m], air[m], {s: st[s][m] for s in R.STATS})
            rows[ao].append(PC.fam_mean(np.isin(act, (R.RETAKE, R.REAIM)), g[m]))

    off = PC.off_target_sets()
    md = ["# OOD α without the test half (enhanced_res v2)", "",
          "The test half is not read. Rule: α_quality + α_OOD = the 15% false-reject budget (union bound, no tuning). "
          f"Check 1: {N} random halvings of the 88 cal-half families (calibrate on 44, measure on the other 44). "
          "Check 2: the off-target sets and D5, gates calibrated on the whole cal half.", "",
          "| OOD α | α_q + α_OOD | normal families stopped, cal-half halvings: mean (5th–95th pct) | "
          "off-target photos pass | wallpapers pass | static pass | D5 stopped |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for ao in ALPHAS_OOD:
        G = PC.Gates(a, P, y, g, D, air, g, ref, qs, g, ao)
        r = {}
        for k, (P_, D_, A_, S_) in off.items():
            act, _ = G.actions(P_, D_, A_, S_)
            stop = np.isin(act, (R.RETAKE, R.REAIM)).mean()
            r[k] = stop if k == "D5" else 1 - stop
        v = np.array(rows[ao])
        md.append(f"| {ao:.2f} | {a.alpha_q + ao:.2f} | {v.mean() * 100:.1f}% ({np.quantile(v, .05) * 100:.1f}–"
                  f"{np.quantile(v, .95) * 100:.1f}%) | {r['off-target photos'] * 100:.1f}% | "
                  f"{r['wallpapers'] * 100:.1f}% | {r['static'] * 100:.1f}% | {r['D5'] * 100:.1f}% |")
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
