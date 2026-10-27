"""Quality gate alpha 0.10 vs 0.05: false reject vs false accept, without the test half (enhanced_res v2).

From 1_main_model_resnet18/ (after calibrate.py --model v2 and archA_offtarget_check.py, whose caches it reuses):
  python enhanced_res/arch_a/experiments/archA_quality_alpha.py

The test half is never read. Everything is measured inside the cal half (88 families):
  - up to PER_FAMILY crops per family, plus 14 synthetic corruptions of each (blur, motion blur, dark, overexposed,
    glare, noise, JPEG; archA_gates.CORRUPTIONS)
  - N random halvings: both gates are recalibrated on 44 families and measured on the other 44
    false reject = clean crops stopped by either gate; false accept = corrupted crops that pass both gates
  - plus D5 (false reject) and the off-target sets (false accept), gates calibrated on the whole cal half
The OOD gate stays at its deployed alpha; only the quality alpha changes.
Writes enhanced_res/arch_a/results/archA/quality_alpha/README.md.
"""
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

import archA_gates as G
import archA_offtarget_check as OC
import archA_pooled_cal as PC
from arch_a import calibrate as C
from arch_a import runtime as R
from model import MODELS, EnhancedRes, load_split

ALPHAS_Q = (0.10, 0.05)
PER_FAMILY, N, SEED = 6, 200, 0
OUT = HERE.parent / "results" / "archA" / "quality_alpha"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    a = R.ArchA(R.CALIBRATIONS["v2"])
    m = EnhancedRes(MODELS["v2"])
    ext = OC.ER(m)
    head = torch.load(MODELS["v2"] / "enhanced_res_head.pt", map_location="cpu")
    bank = head["knn_bank"].float().numpy()
    bank /= np.linalg.norm(bank, axis=1, keepdims=True)

    crops = load_split("d1_grouped")
    va = crops[crops.split == "val"].reset_index(drop=True)
    te = crops[crops.split == "test"].reset_index(drop=True)
    te["half"] = C.halves(te)
    cal = te[te.half == "cal"].reset_index(drop=True)                 # the test half is not loaded
    nv, nc = len(va), len(cal)
    z = np.load(C.CACHE["v2"], allow_pickle=True)
    assert list(z["crop_path"][nv:nv + nc]) == list(cal.crop_path)
    D_all, F_all = z["knn_dist"][nv:nv + nc], z["feats"][nv:nv + nc]
    g_all = cal.src_group.to_numpy()

    # calibration scores of every cal-half crop
    st_all = [R.quality_stats(Image.open(p).convert("RGB")) for p in cal.crop_path]
    st_all = {s: np.array([d[s] for d in st_all]) for s in R.STATS}
    qs_all = R.quality_score(a.ref, st_all, a.sides)[0]
    air_all = a.aon_score(F_all)

    # evaluation sample: up to PER_FAMILY crops per family, clean + corrupted
    idx = np.concatenate([rng.permutation(np.flatnonzero(g_all == f))[:PER_FAMILY] for f in np.unique(g_all)])
    ims = [Image.open(cal.crop_path[i]).convert("RGB") for i in idx]
    sets = {"clean": (D_all[idx], air_all[idx], qs_all[idx])}
    for name, fn in G.CORRUPTIONS.items():
        cor = [fn(im, np.random.default_rng(SEED + k)) for k, im in enumerate(ims)]
        x = OC.features(ext, f"calq_{name}", cor)
        f = x[:, :768]
        st = [R.quality_stats(c) for c in cor]
        st = {s: np.array([d[s] for d in st]) for s in R.STATS}
        sets[name] = (R.knn_distance(f, bank), a.aon_score(f), R.quality_score(a.ref, st, a.sides)[0])
        print(f"  {name} done", flush=True)
    g_idx = g_all[idx]

    def stopped(c_mask, aq, d, air, qs):
        """Gates calibrated on the cal-half crops in c_mask; True where a crop is stopped."""
        gc = g_all[c_mask]
        p_q = R.pvalue(qs_all[c_mask], gc, qs)
        p_o = np.minimum(1, 2 * np.minimum(R.pvalue(D_all[c_mask], gc, d), R.pvalue(air_all[c_mask], gc, air)))
        return (p_q <= aq) | (p_o <= a.alpha_ood)

    fams = np.unique(g_all)
    res = {aq: {k: [] for k in sets} for aq in ALPHAS_Q}
    for _ in range(N):
        cal_f = rng.choice(fams, len(fams) // 2, replace=False)
        c_mask = np.isin(g_all, cal_f)
        e_mask = ~np.isin(g_idx, cal_f)
        for aq in ALPHAS_Q:
            for k, (d, air, qs) in sets.items():
                s = stopped(c_mask, aq, d[e_mask], air[e_mask], qs[e_mask])
                res[aq][k].append(PC.fam_mean(s, g_idx[e_mask]) if k == "clean" else 1 - s.mean())

    # D5 and the off-target sets, gates on the whole cal half
    off = PC.off_target_sets()
    full = np.ones(nc, bool)
    ext_rows = {}
    for aq in ALPHAS_Q:
        for k, (P_, D_, A_, S_) in off.items():
            s = stopped(full, aq, D_, A_, R.quality_score(a.ref, S_, a.sides)[0])
            ext_rows.setdefault(k, {})[aq] = s.mean() if k == "D5" else 1 - s.mean()

    pct = lambda v: f"{v * 100:.1f}%"
    md = ["# Quality gate α 0.10 vs 0.05 (enhanced_res v2, test half not used)", "",
          f"OOD gate unchanged (α {a.alpha_ood}). Inside the cal half: {len(idx)} crops (≤ {PER_FAMILY} per family, "
          f"{len(fams)} families) and their 14 synthetic corruptions; {N} random halvings, gates recalibrated on 44 "
          "families and measured on the other 44 (mean, 5th–95th percentile).", "",
          "## False reject (good photos stopped)", "", "| set | α 0.10 | α 0.05 |", "|---|---:|---:|"]
    q = lambda v: f"{pct(np.mean(v))} ({pct(np.quantile(v, .05))}–{pct(np.quantile(v, .95))})"
    md.append(f"| normal cal-half families (halvings) | {q(res[0.10]['clean'])} | {q(res[0.05]['clean'])} |")
    md.append(f"| D5, other photos of normal defects | {pct(ext_rows['D5'][0.10])} | {pct(ext_rows['D5'][0.05])} |")
    md += ["", "## False accept (bad or off-target images that get through)", "", "| set | α 0.10 | α 0.05 |",
           "|---|---:|---:|"]
    for k in G.CORRUPTIONS:
        md.append(f"| {k} | {pct(np.mean(res[0.10][k]))} | {pct(np.mean(res[0.05][k]))} |")
    allc = [np.mean(res[aq][k]) for aq in ALPHAS_Q for k in G.CORRUPTIONS]
    md.append(f"| **all corruptions (mean)** | **{pct(np.mean(allc[:14]))}** | **{pct(np.mean(allc[14:]))}** |")
    for k in ("off-target photos", "wallpapers", "static"):
        md.append(f"| {k} | {pct(ext_rows[k][0.10])} | {pct(ext_rows[k][0.05])} |")
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
