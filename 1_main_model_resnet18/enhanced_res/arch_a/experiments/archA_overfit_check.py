"""Overfitting check for the 27 October changes: was anything chosen on the data it is judged on?

From 1_main_model_resnet18/ (after archA_pooled_cal.py, whose caches it reuses):
  python enhanced_res/arch_a/experiments/archA_overfit_check.py

1. OOD alpha chosen WITHOUT the test families: the 266 val + cal-half families are split 200 times into two halves;
   one half calibrates, the other measures the gates' false reject; the largest alpha <= 15% on average is picked.
   Then that alpha is judged once on the 87 test-half families.
2. Per-class coverage on the test half, cal half alone vs pooled (pooling was judged on overall coverage only).
3. How many crops sit right at the CP threshold (q moved from 0.958 to 0.952).
Writes enhanced_res/arch_a/results/archA/pooled_cal/overfit_check.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

import archA_pooled_cal as PC
from arch_a import calibrate as C
from arch_a import runtime as R
from model import load_split

ALPHAS_OOD = (0.04, 0.05, 0.06, 0.07, 0.08)
BUDGET, N = 0.15, 200


def main():
    a = R.ArchA(R.CALIBRATIONS["v2"])
    crops = load_split("d1_grouped")
    va = crops[crops.split == "val"].reset_index(drop=True)
    tr = crops[crops.split == "train"].reset_index(drop=True)
    te = crops[crops.split == "test"].reset_index(drop=True)
    te["half"] = C.halves(te)
    cal, tst = te[te.half == "cal"].reset_index(drop=True), te[te.half == "test"].reset_index(drop=True)
    paths = list(va.crop_path) + list(cal.crop_path) + list(tst.crop_path)
    Fe, P, D = C.outputs(None, paths, C.CACHE["v2"])
    cid = {c: i for i, c in enumerate(a.classes)}
    nv, nc = len(va), len(cal)
    y = np.r_[va.cls.map(cid), cal.cls.map(cid), tst.cls.map(cid)].astype(int)
    g = np.r_[va.src_group, cal.src_group, tst.src_group]
    air = a.aon_score(Fe)
    st_all = PC.quality_stats(paths + list(tr.crop_path))
    n3 = len(paths)
    st = {s: st_all[s][:n3] for s in R.STATS}
    ref = {s: np.sort(st[s][:nv]) for s in R.STATS}
    qs = R.quality_score(ref, st, a.sides)[0]
    qs_tr = R.quality_score(ref, {s: st_all[s][n3:] for s in R.STATS}, a.sides)[0]
    g_tr = tr.src_group.to_numpy()
    pool = np.arange(nv + nc)
    test = np.arange(nv + nc, n3)
    md = ["# Overfitting check (enhanced_res v2, 27 October 2026)", ""]

    # 1. alpha chosen on val + cal only
    rng = np.random.default_rng(1)
    fams = np.unique(g[pool])
    fr = {ao: [] for ao in ALPHAS_OOD}
    for _ in range(N):
        half = rng.choice(fams, len(fams) // 2, replace=False)
        c_i, h_i = pool[np.isin(g[pool], half)], pool[~np.isin(g[pool], half)]
        # quality calibration: the calibration half's cal-half crops + train (val crops are the reference)
        qc = c_i[c_i >= nv]
        for ao in ALPHAS_OOD:
            G = PC.Gates(a, P[c_i], y[c_i], g[c_i], D[c_i], air[c_i], g[c_i], ref,
                         np.r_[qs[qc], qs_tr], np.r_[g[qc], g_tr], ao)
            act, _ = G.actions(P[h_i], D[h_i], air[h_i], {s: st[s][h_i] for s in R.STATS})
            fr[ao].append(PC.fam_mean(np.isin(act, (R.RETAKE, R.REAIM)), g[h_i]))
    md += ["## 1. OOD α chosen without looking at the test families", "",
           f"{N} random halvings of the {len(fams)} val + cal-half families: calibrate on one half, measure the gates' "
           "false reject on the other. (Val crops can't be judged against a quality reference built from themselves "
           "without bias: they look slightly more typical, so this check slightly understates quality false rejects on the val families.)", "",
           "| OOD α | families stopped, val + cal halves (mean) | test half (judged once) |", "|---:|---:|---:|"]
    G_final = {}
    for ao in ALPHAS_OOD:
        G = PC.Gates(a, P[pool], y[pool], g[pool], D[pool], air[pool], g[pool], ref,
                     np.r_[qs[nv:nv + nc], qs_tr], np.r_[g[nv:nv + nc], g_tr], ao)
        act, S = G.actions(P[test], D[test], air[test], {s: st[s][test] for s in R.STATS})
        G_final[ao] = (G, act, S)
        md.append(f"| {ao:.2f} | {np.mean(fr[ao]) * 100:.1f}% | "
                  f"{PC.fam_mean(np.isin(act, (R.RETAKE, R.REAIM)), g[test]) * 100:.1f}% |")
    ok = [ao for ao in ALPHAS_OOD if np.mean(fr[ao]) <= BUDGET]
    pick = max(ok) if ok else min(ALPHAS_OOD)
    md += ["", f"Largest α within the {BUDGET:.0%} budget using val + cal only: **{pick:.2f}** "
           f"(deployed: {a.alpha_ood:.2f}).", ""]

    # 2. per-class coverage
    today_q = R.family_threshold(R.temper(P[nv:nv + nc], a.T), y[nv:nv + nc], g[nv:nv + nc], a.alpha)
    Pt = R.temper(P[test], a.T)
    yt = y[test]
    md += ["## 2. Coverage per class on the test half (target 90%; not guaranteed per class)", "",
           "| class | crops | cal half alone (q {:.3f}) | pooled (q {:.3f}) |".format(today_q, a.q), "|---|---:|---:|---:|"]
    for k, c in enumerate(a.classes):
        m = yt == k
        cov = lambda q: (Pt[m, k] >= 1 - q).mean() * 100
        md.append(f"| {c} | {m.sum()} | {cov(today_q):.1f}% | {cov(a.q):.1f}% |")

    # 3. sensitivity at the threshold
    s_true = 1 - Pt[np.arange(len(yt)), yt]
    between = ((s_true > a.q) & (s_true <= today_q)).mean() * 100
    md += ["", "## 3. How much moved at the threshold", "",
           f"Test-half crops whose true class was inside the set with q {today_q:.3f} but drops out with q {a.q:.3f}: "
           f"**{between:.1f}%** of crops. That is the whole coverage change from pooling.", ""]
    out = HERE.parent / "results" / "archA" / "pooled_cal" / "overfit_check.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
