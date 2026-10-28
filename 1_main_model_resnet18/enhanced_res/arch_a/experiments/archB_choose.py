"""Architecture B's choices, made inside the cal half only (the test half is not read).

From 1_main_model_resnet18/ (after calibrate.py --model gc7 --data d1_gc7_fix):
  python enhanced_res/arch_a/experiments/archB_choose.py

1. OOD alpha (0.05 / 0.08 / 0.10 / 0.13 / 0.15). B has one gate, so false reject <= alpha directly. N random halvings
   of the 88 cal-half families: calibrate the gate on 44, measure on the other 44 (defect crops and no-defect crops
   separately, per family). Plus what each alpha lets through: off-target photos (the AoN saw these categories;
   in-sample), wallpaper patches, static.
2. Miss guard (none / alpha_miss 0.02 / 0.05 / 0.10): the same halvings; CP threshold and guard fitted on 44 families,
   measured on the other 44: silent misses (a defect shown as nothing) and no-defect crops shown as nothing.
3. Blur / darkness on the silent path: synthetic corruptions of cal-half defect crops; how often the deployed B
   shows nothing on them, vs the same crops clean.
Writes enhanced_res/arch_a/results/archB/choices.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import archA_gates as G
import archB_common as BC
from archB_common import R

ALPHAS_OOD = (0.05, 0.08, 0.10, 0.13, 0.15)
GUARDS = (None, 0.02, 0.05, 0.10)
N, PER_FAMILY, SEED = 200, 6, 0
OUT = BC.HERE.parent / "results" / "archB"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = R.ArchA(BC.ARCH["B"][1])
    nd = a.nd
    va, cal, _ = BC.halves(a.cal["data"]["split"])
    z = np.load(BC.C.CACHE["gc7"], allow_pickle=True)                  # written by calibrate.py
    nv, nc = len(va), len(cal)
    assert list(z["crop_path"][nv:nv + nc]) == list(cal.crop_path), "re-run calibrate.py --model gc7 --data d1_gc7_fix"
    F, P, D = z["feats"][nv:nv + nc], z["probs"][nv:nv + nc], z["knn_dist"][nv:nv + nc]
    cid = {c: i for i, c in enumerate(a.classes)}
    y, g = cal.cls.map(cid).to_numpy(), cal.src_group.to_numpy()
    air = a.aon_score(F)
    Pt = R.temper(P, a.T)
    fams = np.unique(g)
    rng = np.random.default_rng(SEED)

    def p_ood(c, d, ai):
        return np.minimum(1, 2 * np.minimum(R.pvalue(D[c], g[c], d), R.pvalue(air[c], g[c], ai)))

    # 1 + 2: halvings
    fr = {ao: {"defect": [], "no-defect": []} for ao in ALPHAS_OOD}
    gd = {mg: {"silent miss": [], "no-defect shown nothing": []} for mg in GUARDS}
    for _ in range(N):
        c = np.isin(g, rng.choice(fams, len(fams) // 2, replace=False))
        e = ~c
        po_e = p_ood(c, D[e], air[e])
        for ao in ALPHAS_OOD:
            stop = po_e <= ao
            for lbl, m in (("defect", y[e] != nd), ("no-defect", y[e] == nd)):
                fr[ao][lbl].append(BC.fam_mean(stop[m], g[e][m]))
        po_c = p_ood(c, D[c], air[c])
        pc = c.copy()
        pc[c] = po_c > a.alpha_ood                                        # gate-passed calibration crops
        q = R.family_threshold(Pt[pc], y[pc], g[pc], a.alpha)
        passed_e = po_e > a.alpha_ood
        S = Pt[e] >= 1 - q
        lone = S[:, nd] & (S.sum(1) == 1) & passed_e
        dm = pc & (y != nd)
        for mg in GUARDS:
            ok = np.ones(e.sum(), bool) if mg is None else R.pvalue(Pt[dm, nd], g[dm], Pt[e][:, nd]) <= mg
            nothing = lone & ok
            gd[mg]["silent miss"].append(nothing[y[e] != nd].mean())
            gd[mg]["no-defect shown nothing"].append(nothing[y[e] == nd].mean())

    # off-target sets at each alpha (gate on the whole cal half)
    man, _, wall, static = BC.extra_sets()
    ot = man[man.role.isin(["fit", "test"])].reset_index(drop=True)
    sets = {"off-target photos (AoN trained on these)": BC.outputs("gc7", "offtarget_fit_test", list(ot.path)),
            "wallpaper patches": BC.outputs("gc7", "wall", wall), "static": BC.outputs("gc7", "static", static)}
    full = np.ones(nc, bool)
    ext = {k: {ao: float((p_ood(full, v[2], a.aon_score(v[0])) > ao).mean()) for ao in ALPHAS_OOD} for k, v in sets.items()}

    # 3: corruptions of defect crops on the silent path (deployed B)
    di = np.flatnonzero(y != nd)
    idx = np.concatenate([rng.permutation(di[g[di] == f])[:PER_FAMILY] for f in np.unique(g[di])])
    ims = [BC.Image.open(cal.crop_path[i]).convert("RGB") for i in idx]
    clean = a.decide(P[idx], knn_dist=D[idx], feats=F[idx])
    silent = {"clean": np.mean([d.action == R.NOTHING for d in clean])}
    for name, fn in G.CORRUPTIONS.items():
        cor = [fn(im, np.random.default_rng(SEED + k)) for k, im in enumerate(ims)]
        f_, p_, d_ = BC.outputs("gc7", f"cal_{name}", cor)
        silent[name] = np.mean([d.action == R.NOTHING for d in a.decide(p_, knn_dist=d_, feats=f_)])
        print(f"  {name}: {silent[name] * 100:.1f}% shown nothing", flush=True)

    pct = lambda v: f"{np.mean(v) * 100:.1f}%"
    q5 = lambda v: f"{pct(v)} ({np.quantile(v, .05) * 100:.1f}–{np.quantile(v, .95) * 100:.1f}%)"
    md = ["# Architecture B: choices made inside the cal half (test half not used)", "",
          f"Model gc7, crops {a.cal['data']['split']}, quality gate off. {N} random halvings of the {len(fams)} cal-half "
          "families (fit on 44, measure on the other 44); mean (5th–95th percentile).", "",
          "## 1. OOD α", "", "| OOD α | defect crops stopped (per family) | no-defect crops stopped (per family) | "
          + " | ".join(f"{k} passing" for k in sets) + " |", "|---:|---:|---:|" + "---:|" * len(sets)]
    for ao in ALPHAS_OOD:
        md.append(f"| {ao:.2f} | {q5(fr[ao]['defect'])} | {q5(fr[ao]['no-defect'])} | "
                  + " | ".join(f"{ext[k][ao] * 100:.1f}%" for k in sets) + " |")
    md += ["", f"## 2. Miss guard (OOD α {a.alpha_ood}, CP α {a.alpha})", "",
           "Silent miss = a defect crop shown as nothing. A lone no-defect with the guard failing is shown as "
           "\"POSSIBLE …: check\" instead. With 44 calibration families, α_miss below about 1/45 can't be certified.", "",
           "| miss guard | silent miss | no-defect shown nothing |", "|---|---:|---:|"]
    for mg in GUARDS:
        md.append(f"| {'none (plain LAC)' if mg is None else f'α_miss {mg:.2f}'} | {q5(gd[mg]['silent miss'])} | "
                  f"{q5(gd[mg]['no-defect shown nothing'])} |")
    md += ["", "## 3. Damaged images on the silent path (deployed B, no quality gate)", "",
           f"{len(idx)} cal-half defect crops (≤ {PER_FAMILY} per family) and 14 synthetic corruptions of them: share "
           "shown as nothing (a silent miss).", "", "| set | shown nothing |", "|---|---:|"]
    md += [f"| {k} | {v * 100:.1f}% |" for k, v in silent.items()]
    (OUT / "choices.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
