"""Can architecture A calibrate on more photo families without new data? Pooling val with the cal half.

From 1_main_model_resnet18/ (after calibrate.py --model v2, whose output cache it reuses):
  python enhanced_res/arch_a/experiments/archA_pooled_cal.py

Today every conformal threshold (CP set, quality gate, OOD gate) is fitted on the cal half alone: 88 photo families.
Candidates, all on enhanced_res v2 (d1_grouped), nothing retrained:
  today     CP + OOD on the cal half (88 families); quality: reference = val, calibration = cal half
  pooled    CP + OOD on val + cal half (266 families); quality: reference = val, calibration = cal half + train
            (quality uses pixel statistics only, so train crops are as valid as any; OOD can't use train: those
            crops are in the kNN bank and the regression's positives)
  val only  CP + OOD on val alone (178 families): isolates the bias from val having picked the model and fitted T
Two checks:
  1. the fixed test half (87 families): coverage, set size, gate false reject, visor actions, plus off-target sets
     (benchmark photos, wallpapers, static) and D5
  2. 200 random re-splits of the 175 test families into cal / test halves: how much coverage and the gates' false
     reject move from one draw of calibration families to the next, with and without val pooled in
Writes enhanced_res/arch_a/results/archA/pooled_cal/README.md and results.json.
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

from arch_a import calibrate as C
from arch_a import runtime as R
from model import EWORK, MAIN, MODELS, EnhancedRes, load_split

OUT = HERE.parent / "results" / "archA" / "pooled_cal"
QCACHE = EWORK / "arch_a_quality_stats_v2.npz"
N_SPLITS, SEED = 200, 0
SWEEP = (0.04, 0.05, 0.06, 0.07)


def quality_stats(paths):
    """The 4 quality statistics of every crop, cached by crop list."""
    if QCACHE.exists():
        z = np.load(QCACHE, allow_pickle=True)
        if list(z["paths"]) == list(paths):
            return {s: z[s] for s in R.STATS}
    out = {s: [] for s in R.STATS}
    for i, p in enumerate(paths):
        d = R.quality_stats(Image.open(p).convert("RGB"))
        for s in R.STATS:
            out[s].append(d[s])
        if i % 500 == 0:
            print(f"\r  quality statistics {i}/{len(paths)}", end="", flush=True)
    print()
    out = {s: np.asarray(v) for s, v in out.items()}
    np.savez(QCACHE, paths=np.array(paths), **out)
    return out


def fam_mean(v, g):
    return float(pd.Series(np.asarray(v, float)).groupby(np.asarray(g)).mean().mean())


class Gates:
    """Conformal thresholds of one configuration (same rules as runtime.ArchA, other calibration data)."""

    def __init__(self, a, P_cal, y_cal, g_cal, o_dist, o_air, o_g, q_ref, q_scores, q_g, alpha_ood=None):
        self.a = a
        self.alpha_ood = a.alpha_ood if alpha_ood is None else alpha_ood
        self.q = R.family_threshold(R.temper(P_cal, a.T), y_cal, g_cal, a.alpha)
        self.o = (o_dist, o_air, o_g)
        self.ref, self.qs = q_ref, (q_scores, q_g)

    def sets(self, P):
        return R.temper(P, self.a.T) >= 1 - self.q

    def p_ood(self, dist, air):
        d, ai, g = self.o
        return np.minimum(1, 2 * np.minimum(R.pvalue(d, g, dist), R.pvalue(ai, g, air)))

    def p_quality(self, stats):
        sc, _ = R.quality_score(self.ref, stats, self.a.sides)
        return R.pvalue(*self.qs, sc)

    def actions(self, P, dist, air, stats):
        po, pq = self.p_ood(dist, air), self.p_quality(stats)
        S = self.sets(P)
        n = S.sum(1)
        act = np.where(po <= self.alpha_ood, R.REAIM, np.where(pq <= self.a.alpha_q, R.RETAKE,
                       np.where(n == 1, R.SHOW, np.where(n > 1, R.AMBIGUOUS, R.UNCLEAR))))
        return act, S


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = R.ArchA(R.CALIBRATIONS["v2"])
    crops = load_split("d1_grouped")
    va = crops[crops.split == "val"].reset_index(drop=True)
    tr = crops[crops.split == "train"].reset_index(drop=True)
    te = crops[crops.split == "test"].reset_index(drop=True)
    te["half"] = C.halves(te)
    cal, tst = te[te.half == "cal"].reset_index(drop=True), te[te.half == "test"].reset_index(drop=True)
    paths = list(va.crop_path) + list(cal.crop_path) + list(tst.crop_path)
    Fe, P, D = C.outputs(None, paths, C.CACHE["v2"])             # cache written by calibrate.py --model v2
    cid = {c: i for i, c in enumerate(a.classes)}
    nv, nc = len(va), len(cal)
    part = {"val": slice(0, nv), "cal": slice(nv, nv + nc), "test": slice(nv + nc, None)}
    df = {"val": va, "cal": cal, "test": tst}
    y = {k: df[k].cls.map(cid).to_numpy() for k in df}
    g = {k: df[k].src_group.to_numpy() for k in df}
    air = a.aon_score(Fe)
    st_all = quality_stats(paths + list(tr.crop_path))
    n3 = len(paths)
    st = {k: {s: st_all[s][:n3][part[k]] for s in R.STATS} for k in part}
    st["train"] = {s: st_all[s][n3:] for s in R.STATS}
    g["train"] = tr.src_group.to_numpy()
    ref = {s: np.sort(st["val"][s]) for s in R.STATS}
    qscore = {k: R.quality_score(ref, st[k], a.sides)[0] for k in ("val", "cal", "test", "train")}
    print(f"families: train {len(set(g['train']))}, val {len(set(g['val']))}, cal half {len(set(g['cal']))}, "
          f"test half {len(set(g['test']))}")

    def cfg(cp_parts, q_parts, alpha_ood=None):
        cat = lambda d, ks: np.concatenate([d[k] for k in ks])
        Pk = np.concatenate([P[part[k]] for k in cp_parts])
        return Gates(a, Pk, cat(y, cp_parts), cat(g, cp_parts),
                     np.concatenate([D[part[k]] for k in cp_parts]), np.concatenate([air[part[k]] for k in cp_parts]),
                     cat(g, cp_parts), ref, cat(qscore, q_parts), cat(g, q_parts), alpha_ood)

    configs = {"today": cfg(["cal"], ["cal"]), "pooled": cfg(["val", "cal"], ["cal", "train"]),
               "val only": cfg(["val"], ["cal"])}
    configs.update({f"pooled, OOD α {ao:.2f}": cfg(["val", "cal"], ["cal", "train"], ao) for ao in SWEEP})

    # ---------------------------------------------------------------- 1. the fixed test half
    off = off_target_sets()
    res = {}
    for name, G in configs.items():
        t = part["test"]
        act, S = G.actions(P[t], D[t], air[t], st["test"])
        hit = S[np.arange(len(y["test"])), y["test"]]
        shown = act == R.SHOW
        right = S[np.arange(len(y["test"])), y["test"]]
        stop = np.isin(act, (R.RETAKE, R.REAIM))
        r = {"q": G.q, "cp_coverage": float(hit.mean()), "cp_family_coverage": fam_mean(hit, g["test"]),
             "cp_mean_set_size": float(S.sum(1).mean()), "cp_single_class": float((S.sum(1) == 1).mean()),
             "gates_false_reject_family": fam_mean(stop, g["test"]),
             "re-take": float((act == R.RETAKE).mean()), "re-aim": float((act == R.REAIM).mean()),
             "human_attention": float(1 - shown.mean()), "single_class_wrong": float((shown & ~right).mean())}
        for k, (P_, D_, A_, S_) in off.items():
            a_, _ = G.actions(P_, D_, A_, S_)
            r[f"{k} stopped" if k == "D5" else f"{k} pass"] = float(
                np.isin(a_, (R.RETAKE, R.REAIM)).mean() if k == "D5" else (~np.isin(a_, (R.RETAKE, R.REAIM))).mean())
        res[name] = r
        print(name, {k: round(v, 4) for k, v in r.items()})

    # ---------------------------------------------------------------- 2. random re-splits of the test families
    rng = np.random.default_rng(SEED)
    te_idx = np.r_[np.arange(nv, nv + nc), np.arange(nv + nc, n3)]       # every test-split crop
    te_y, te_g = np.r_[y["cal"], y["test"]], np.r_[g["cal"], g["test"]]
    te_qs = np.r_[qscore["cal"], qscore["test"]]
    te_st = {s: np.r_[st["cal"][s], st["test"][s]] for s in R.STATS}
    fams = np.unique(te_g)
    rows = []
    for i in range(N_SPLITS):
        cal_f = rng.choice(fams, 88, replace=False)
        c_m = np.isin(te_g, cal_f)
        t_m = ~c_m
        for name, ao in [("today", None), ("pooled", None)] + [(f"pooled, OOD α {x:.2f}", x) for x in SWEEP]:
            ci = te_idx[c_m]
            if name == "today":
                G = Gates(a, P[ci], te_y[c_m], te_g[c_m], D[ci], air[ci], te_g[c_m], ref, te_qs[c_m], te_g[c_m])
            else:
                vi = np.arange(nv)
                G = Gates(a, np.r_[P[vi], P[ci]], np.r_[y["val"], te_y[c_m]], np.r_[g["val"], te_g[c_m]],
                          np.r_[D[vi], D[ci]], np.r_[air[vi], air[ci]], np.r_[g["val"], te_g[c_m]], ref,
                          np.r_[te_qs[c_m], qscore["train"]], np.r_[te_g[c_m], g["train"]], ao)
            ti = te_idx[t_m]
            act, S = G.actions(P[ti], D[ti], air[ti], {s: te_st[s][t_m] for s in R.STATS})
            hit = S[np.arange(t_m.sum()), te_y[t_m]]
            rows.append({"split": i, "config": name, "coverage": hit.mean(),
                         "family_coverage": fam_mean(hit, te_g[t_m]), "set_size": S.sum(1).mean(),
                         "gates_false_reject_family": fam_mean(np.isin(act, (R.RETAKE, R.REAIM)), te_g[t_m])})
        print(f"\r  random splits {i + 1}/{N_SPLITS}", end="", flush=True)
    print()
    sp = pd.DataFrame(rows)
    spread = {}
    for name, grp in sp.groupby("config"):
        spread[name] = {c: {"mean": float(grp[c].mean()), "sd": float(grp[c].std()),
                            "p5": float(grp[c].quantile(0.05)), "p95": float(grp[c].quantile(0.95)),
                            "below_90": float((grp[c] < 0.90).mean()) if "coverage" in c else None}
                        for c in ("coverage", "family_coverage", "set_size", "gates_false_reject_family")}
    (OUT / "results.json").write_text(json.dumps({"test_half": res, "random_splits": spread}, indent=1))
    write_readme(res, spread, {k: len(set(v)) for k, v in g.items()})
    print("wrote", OUT)


def off_target_sets():
    """Model outputs + quality statistics of D5 and the off-target sets (cached features from archA_offtarget_check)."""
    import archA_d5 as A5
    import archA_gates as G
    import archA_offtarget_check as OC
    import archA_static as ST
    m = EnhancedRes(MODELS["v2"])
    ext = OC.ER(m)
    head = torch.load(MODELS["v2"] / "enhanced_res_head.pt", map_location="cpu")
    bank = head["knn_bank"].float().numpy()
    bank /= np.linalg.norm(bank, axis=1, keepdims=True)
    man = pd.read_csv(OC.OFF / "manifest.csv")
    man = man[man.role.isin(["fit", "test"])]
    d5, _ = A5.d5_crops()
    Dg = G.prepare()
    sets = {"off-target photos": [str(MAIN / p) for p in man.crop_path], "D5": list(d5.crop_path),
            "wallpapers": Dg["sets"]["off-target"],
            "static": [im for ims in ST.nonsense(np.random.default_rng(0)).values() for im in ims]}
    key = {"off-target photos": None, "D5": "d5", "wallpapers": "wallpapers", "static": "static"}
    a = R.ArchA(R.CALIBRATIONS["v2"])
    out = {}
    for name, items in sets.items():
        if name == "off-target photos":                  # the check script cached all benchmark crops as "bench"
            allp = [str(MAIN / p) for p in pd.read_csv(OC.OFF / "manifest.csv").crop_path]
            x = OC.features(ext, "bench", allp)
            x = x[np.isin(allp, items)]
        else:
            x = OC.features(ext, key[name], items)
        f, p = x[:, :768], x[:, 768:]
        ims = [Image.open(i).convert("RGB") if isinstance(i, str) else i for i in items]
        st = [R.quality_stats(im) for im in ims]
        out[name] = (p, R.knn_distance(f, bank), a.aon_score(f), {s: np.array([d[s] for d in st]) for s in R.STATS})
    return out


def write_readme(res, spread, nfam):
    pct = lambda v: f"{v * 100:.1f}%"
    names = list(res)
    rows = [("CP threshold q", lambda r: f"{r['q']:.3f}"),
            ("CP coverage per crop / per family (target 90%)",
             lambda r: f"{pct(r['cp_coverage'])} / {pct(r['cp_family_coverage'])}"),
            ("mean set size / single-class answers", lambda r: f"{r['cp_mean_set_size']:.2f} / {pct(r['cp_single_class'])}"),
            ("normal families stopped by the gates", lambda r: pct(r["gates_false_reject_family"])),
            ("re-take / re-aim (crops)", lambda r: f"{pct(r['re-take'])} / {pct(r['re-aim'])}"),
            ("inspector needed", lambda r: pct(r["human_attention"])),
            ("single wrong class shown without warning", lambda r: pct(r["single_class_wrong"])),
            ("D5 stopped (normal defects, other photos)", lambda r: pct(r["D5 stopped"])),
            ("off-target photos passing", lambda r: pct(r["off-target photos pass"])),
            ("wallpapers / static passing", lambda r: f"{pct(r['wallpapers pass'])} / {pct(r['static pass'])}")]
    md = ["# Pooling val into architecture A's calibration (enhanced_res v2)", "",
          f"Families: cal half {nfam['cal']}, val {nfam['val']}, train {nfam['train']}; test half {nfam['test']}. "
          "Nothing retrained. `today`: everything calibrated on the cal half. `pooled`: CP and the OOD gate on val + cal "
          "half, the quality gate on cal half + train (reference: val). `val only`: CP and OOD on val alone, to see the "
          "bias from val having picked the model and fitted T.", "",
          "## 1. The fixed test half (87 families)", "", "| | " + " | ".join(names) + " |",
          "|---|" + "---:|" * len(names)]
    md += [f"| {lbl} | " + " | ".join(f(res[n]) for n in names) + " |" for lbl, f in rows]
    md += ["", f"## 2. {N_SPLITS} random re-splits of the 175 test families (88 cal / 87 test)", "",
           "How much the numbers move from one draw of calibration families to the next. `pooled` adds val (and train, "
           "for the quality gate) to every draw.", "",
           "| | today: mean (5th–95th pct) | pooled: mean (5th–95th pct) |", "|---|---:|---:|"]
    for c, lbl in (("coverage", "CP coverage per crop"), ("family_coverage", "CP coverage per family"),
                   ("set_size", "mean set size"), ("gates_false_reject_family", "normal families stopped")):
        f = (lambda v: f"{v:.2f}") if c == "set_size" else pct
        md.append(f"| {lbl} | " + " | ".join(
            f"{f(spread[n][c]['mean'])} ({f(spread[n][c]['p5'])}–{f(spread[n][c]['p95'])})" for n in ("today", "pooled")) + " |")
    md.append(f"| draws with per-crop coverage below 90% | {pct(spread['today']['coverage']['below_90'])} | "
              f"{pct(spread['pooled']['coverage']['below_90'])} |")
    md += ["", "### OOD α sweep with pooling: normal families stopped by the gates, mean (5th–95th pct) over the draws", "",
           "| OOD α | normal families stopped | CP coverage per family |", "|---:|---:|---:|"]
    for x in (*SWEEP, None):
        n = "pooled" if x is None else f"pooled, OOD α {x:.2f}"
        r = spread[n]
        md.append(f"| {'0.08' if x is None else f'{x:.2f}'} | {pct(r['gates_false_reject_family']['mean'])} "
                  f"({pct(r['gates_false_reject_family']['p5'])}–{pct(r['gates_false_reject_family']['p95'])}) | "
                  f"{pct(r['family_coverage']['mean'])} |")
    (OUT / "README.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
