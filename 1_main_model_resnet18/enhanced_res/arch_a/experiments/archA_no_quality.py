"""What if architecture A dropped the quality gate and relied on the OOD gate alone? (enhanced_res v2)

From 1_main_model_resnet18/ (after calibrate.py --model v2 and archA_offtarget_check.py, whose caches it reuses):
  python enhanced_res/arch_a/experiments/archA_no_quality.py

The test half is never read. Configurations:
  deployed       quality α 0.05 + OOD α 0.08 (kNN + AoN, Bonferroni), in parallel
  OOD only 0.08  no quality gate; OOD gate unchanged
  OOD only 0.13  no quality gate; OOD gate given the budget the quality gate used (0.05 + 0.08)
Inside the cal half (88 families): up to PER_FAMILY crops per family and their 14 synthetic corruptions; N random
halvings, gates recalibrated on 44 families and measured on the other 44. For the corrupted crops that get through,
it also checks what the visor shows (CP set at the deployed threshold): does the set still hold the true class, and how
often is a single wrong class shown with no warning? Plus D5 and the off-target sets, gates on the whole cal half.
Writes enhanced_res/arch_a/results/archA/no_quality/README.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

import numpy as np
import torch
from PIL import Image

import archA_gates as G
import archA_offtarget_check as OC
import archA_pooled_cal as PC
from arch_a import calibrate as C
from arch_a import runtime as R
from model import MODELS, EnhancedRes, load_split

CONFIGS = {"deployed (quality 0.05 + OOD 0.08)": (0.05, 0.08), "OOD only, α 0.08": (None, 0.08),
           "OOD only, α 0.13": (None, 0.13)}
PER_FAMILY, N, SEED = 6, 200, 0
OUT = HERE.parent / "results" / "archA" / "no_quality"


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
    D_all, F_all, P_all = z["knn_dist"][nv:nv + nc], z["feats"][nv:nv + nc], z["probs"][nv:nv + nc]
    g_all = cal.src_group.to_numpy()
    cid = {c: i for i, c in enumerate(a.classes)}
    y_all = cal.cls.map(cid).to_numpy()
    st_all = [R.quality_stats(Image.open(p).convert("RGB")) for p in cal.crop_path]
    st_all = {s: np.array([d[s] for d in st_all]) for s in R.STATS}
    qs_all = R.quality_score(a.ref, st_all, a.sides)[0]
    air_all = a.aon_score(F_all)

    idx = np.concatenate([rng.permutation(np.flatnonzero(g_all == f))[:PER_FAMILY] for f in np.unique(g_all)])
    ims = [Image.open(cal.crop_path[i]).convert("RGB") for i in idx]
    sets = {"clean": (D_all[idx], air_all[idx], qs_all[idx], P_all[idx])}
    for name, fn in G.CORRUPTIONS.items():
        cor = [fn(im, np.random.default_rng(SEED + k)) for k, im in enumerate(ims)]
        x = OC.features(ext, f"calq_{name}", cor)
        st = [R.quality_stats(c) for c in cor]
        st = {s: np.array([d[s] for d in st]) for s in R.STATS}
        sets[name] = (R.knn_distance(x[:, :768], bank), a.aon_score(x[:, :768]),
                      R.quality_score(a.ref, st, a.sides)[0], x[:, 768:])
        print(f"  {name} done", flush=True)
    g_idx, y_idx = g_all[idx], y_all[idx]

    def stopped(c_mask, aq, ao, d, air, qs):
        gc = g_all[c_mask]
        p_o = np.minimum(1, 2 * np.minimum(R.pvalue(D_all[c_mask], gc, d), R.pvalue(air_all[c_mask], gc, air)))
        s = p_o <= ao
        return s if aq is None else s | (R.pvalue(qs_all[c_mask], gc, qs) <= aq)

    S_cp = {k: a.sets(v[3])[1] for k, v in sets.items()}                 # CP sets at the deployed threshold
    fams = np.unique(g_all)
    res = {c: {k: {"pass": [], "cov": [], "wrong": []} for k in sets} for c in CONFIGS}
    for _ in range(N):
        cal_f = rng.choice(fams, len(fams) // 2, replace=False)
        c_mask, e = np.isin(g_all, cal_f), ~np.isin(g_idx, cal_f)
        for c, (aq, ao) in CONFIGS.items():
            for k, (d, air, qs, _) in sets.items():
                s = stopped(c_mask, aq, ao, d[e], air[e], qs[e])
                ps = ~s
                S = S_cp[k][e]
                hit = S[np.arange(e.sum()), y_idx[e]]
                single = S.sum(1) == 1
                r = res[c][k]
                r["pass"].append(PC.fam_mean(s, g_idx[e]) if k == "clean" else ps.mean())
                r["cov"].append(hit[ps].mean() if ps.any() else np.nan)
                r["wrong"].append((ps & single & ~hit).mean())

    off = PC.off_target_sets()
    full = np.ones(nc, bool)
    ext_rows = {}
    for c, (aq, ao) in CONFIGS.items():
        for k, (P_, D_, A_, S_) in off.items():
            s = stopped(full, aq, ao, D_, A_, R.quality_score(a.ref, S_, a.sides)[0])
            ext_rows.setdefault(k, {})[c] = s.mean() if k == "D5" else 1 - s.mean()

    names = list(CONFIGS)
    pct = lambda v: "–" if np.all(np.isnan(np.asarray(v, float))) else f"{np.nanmean(v) * 100:.1f}%"
    hdr = ["| set | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    md = ["# Dropping the quality gate (enhanced_res v2, test half not used)", "",
          f"Inside the cal half: {len(idx)} crops (≤ {PER_FAMILY} per family, {len(fams)} families) and their 14 "
          f"synthetic corruptions; {N} random halvings (gates recalibrated on 44 families, measured on the other 44). "
          "D5 and the off-target sets use gates calibrated on the whole cal half.", "",
          "## False reject (good photos stopped)", ""] + hdr
    md.append("| normal cal-half families | " + " | ".join(pct(res[c]["clean"]["pass"]) for c in names) + " |")
    md.append("| D5, other photos of normal defects | " + " | ".join(pct(ext_rows["D5"][c]) for c in names) + " |")
    md += ["", "## False accept (bad images that get through)", ""] + hdr
    for k in G.CORRUPTIONS:
        md.append(f"| {k} | " + " | ".join(pct(res[c][k]["pass"]) for c in names) + " |")
    md.append("| **all corruptions (mean)** | " + " | ".join(
        "**" + pct([np.mean(res[c][k]["pass"]) for k in G.CORRUPTIONS]) + "**" for c in names) + " |")
    for k in ("off-target photos", "wallpapers", "static"):
        md.append(f"| {k} | " + " | ".join(pct(ext_rows[k][c]) for c in names) + " |")
    md += ["", "## Does it matter? What the visor shows for bad images that get through", "",
           "Coverage = the CP set holds the true class (target 90%). Silent wrong = a single wrong class shown "
           "with no warning, as a share of all corrupted crops.", "",
           "| set | " + " | ".join(f"{c}: coverage / silent wrong" for c in names) + " |", "|---|" + "---:|" * len(names)]
    for k in ["clean"] + list(G.CORRUPTIONS):
        md.append(f"| {k} | " + " | ".join(f"{pct(res[c][k]['cov'])} / {pct(res[c][k]['wrong'])}" for c in names) + " |")
    md.append("| **all corruptions (mean)** | " + " | ".join(
        f"**{pct([np.nanmean(res[c][k]['cov']) for k in G.CORRUPTIONS])} / "
        f"{pct([np.mean(res[c][k]['wrong']) for k in G.CORRUPTIONS])}**" for c in names) + " |")
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
