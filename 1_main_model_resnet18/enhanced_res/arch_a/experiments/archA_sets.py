"""Architecture A, step 3: which prediction-set method. LAC vs APS vs RAPS, and global vs per-class vs pooled thresholds.

From 1_main_model_resnet18/:
  python enhanced_res/arch_a/experiments/archA_sets.py          # uses the cached val/test features from archA_cp.py (no model run)

Probabilities: enhanced_res tempered with prob-T (A step 1 default). Thresholds: family-weighted (every calibration
photo family counts once), fitted on the cal half, measured on the test half; 100 random group halvings for stability.

Scores (smaller = the class fits better; a class goes in the set when its score <= the threshold):
  LAC   1 - p(class)                                   (Sadinle, Lei & Wasserman 2019)
  APS   total probability of all classes at least as likely as this one   (Romano, Sesia & Candes 2020; deterministic)
  RAPS  APS + lambda * max(0, rank - k_reg): penalises long sets  (Angelopoulos, Bates, Malik & Jordan 2021)
        k_reg and lambda chosen on val (smallest mean set size at alpha 0.10), never on cal or test.

Thresholds:
  global    one threshold for all classes (architecture 0's way)
  per-class Mondrian CP: one threshold per true class, from that class's calibration crops (Vovk et al. 2005)
  pooled    two groups sharing a threshold: {crack, dent} ("deformation") and the other four
A per-class threshold needs at least (1 - alpha) / alpha calibration families of that class (9 at alpha 0.10, 19 at
0.05); with fewer, the rule cannot pick a finite threshold and the class is always included.

Per-class coverage gets a 95% interval from resampling whole test families (2,000 draws).
Writes enhanced_res/arch_a/results/archA/step3/.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import arch0_cp as A0
import archA_cp as A1
from model import MAIN, load_split

CACHE = A0.CACHE
OUT = HERE.parent / "results" / "archA" / "step3"
ALPHAS = (0.05, 0.10)
REPEATS, BOOT = 100, 2000
POOL = {"crack": 0, "dent": 0}          # everything else -> group 1


# ------------------------------------------------------------------ scores for every (crop, class)
def score_matrix(p, kind, lam=0.0, kreg=0):
    if kind == "LAC":
        return 1 - p
    order = np.argsort(-p, 1)
    sorted_p = np.take_along_axis(p, order, 1)
    cum = np.cumsum(sorted_p, 1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(p.shape[1])[None].repeat(len(p), 0), 1)
    s = np.take_along_axis(cum, rank, 1)                      # APS: mass of classes at least as likely
    if kind == "RAPS":
        s = s + lam * np.maximum(0, rank + 1 - kreg)
    return s


# ------------------------------------------------------------------ family-weighted threshold
def threshold(s, g, alpha):
    """Smallest t with family-weighted share of calibration scores <= t at least (1-alpha)(1 + 1/G); inf if impossible."""
    if len(s) == 0:
        return np.inf
    _, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    G = len(cnt)
    order = np.argsort(s)
    cum = np.cumsum(w[order])
    k = np.searchsorted(cum, (1 - alpha) * (G + 1) - 1e-9)
    return np.inf if k >= len(s) else float(s[order][k])


def fit(S_cal, y_cal, g_cal, alpha, mode, classes):
    """Per-class threshold vector (length K) for 'global', 'per-class' or 'pooled'."""
    K = len(classes)
    true = S_cal[np.arange(len(y_cal)), y_cal]
    if mode == "global":
        return np.full(K, threshold(true, g_cal, alpha))
    if mode == "per-class":
        return np.array([threshold(true[y_cal == k], g_cal[y_cal == k], alpha) for k in range(K)])
    grp = np.array([POOL.get(c, 1) for c in classes])
    t = {gr: threshold(true[grp[y_cal] == gr], g_cal[grp[y_cal] == gr], alpha) for gr in (0, 1)}
    return np.array([t[grp[k]] for k in range(K)])


def sets_from(S, thr):
    return S <= thr[None, :]


def stats(sets, y, g, K):
    size = sets.sum(1)
    hit = sets[np.arange(len(y)), y]
    per = [hit[y == k].mean() if (y == k).any() else np.nan for k in range(K)]
    return {"coverage": hit.mean(), "family_coverage": pd.Series(hit).groupby(g).mean().mean(),
            "mean_size": size.mean(), "singleton": (size == 1).mean(), "all_classes": (size == K).mean(),
            "empty": (size == 0).mean(), "worst_class": np.nanmin(per), **{f"cov_{k}": v for k, v in enumerate(per)}}


def class_ci(sets, y, g, K, rng):
    hit = sets[np.arange(len(y)), y].astype(float)
    ug, inv = np.unique(g, return_inverse=True)
    out = []
    for k in range(K):
        m = y == k
        h = np.bincount(inv[m], hit[m], len(ug))
        n = np.bincount(inv[m], minlength=len(ug)).astype(float)
        d = rng.integers(0, len(ug), (BOOT, len(ug)))
        cov = h[d].sum(1) / np.maximum(n[d].sum(1), 1)
        out.append(np.quantile(cov[n[d].sum(1) > 0], [0.025, 0.975]))
    return out


def tune_raps(P_val, y_val, g_val, rng):
    """k_reg and lambda with the smallest mean set size at alpha 0.10, by calibrating on half of val's families and
    measuring on the other half (20 random halvings)."""
    fams = np.unique(g_val)
    best = None
    for kreg in (1, 2, 3):
        for lam in (0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5):
            S = score_matrix(P_val, "RAPS", lam, kreg)
            sizes = []
            for i in range(20):
                a = np.isin(g_val, rng.choice(fams, len(fams) // 2, replace=False))
                thr = fit(S[a], y_val[a], g_val[a], 0.10, "global", range(P_val.shape[1]))
                sizes.append(sets_from(S[~a], thr).sum(1).mean())
            if best is None or np.mean(sizes) < best[0]:
                best = (float(np.mean(sizes)), lam, kreg)
    return best[1], best[2]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    crops = load_split()
    va, _, Lv, Kv = A1.model_parts(crops, "val")
    te, _, Lt, Kt = A1.model_parts(crops, "test")
    classes = list(np.load(CACHE / "feats_test.npz", allow_pickle=True)["classes"])
    K, cid = len(classes), {c: i for i, c in enumerate(classes)}
    T = json.loads((CACHE / "archA_step1.json").read_text())["T_prob"]
    Pv, Pt = A1.temper(A1.blend(Lv, Kv), T), A1.temper(A1.blend(Lt, Kt), T)
    yv, yt = va.cls.map(cid).to_numpy(), te.cls.map(cid).to_numpy()
    gv, gt = va.src_group.to_numpy(), te.src_group.to_numpy()

    sp = pd.read_csv(HERE.parent / "cal_split.csv")
    sp["crop_path"] = [str(MAIN / p) for p in sp.crop_path]
    te = te.merge(sp[["crop_path", "split"]].rename(columns={"split": "half"}), on="crop_path", validate="1:1")
    is_cal = (te.half == "cal").to_numpy()

    lam, kreg = tune_raps(Pv, yv, gv, rng)
    print(f"RAPS tuned on val: lambda = {lam}, k_reg = {kreg}")
    S = {"LAC": score_matrix(Pt, "LAC"), "APS": score_matrix(Pt, "APS"),
         f"RAPS (lambda={lam}, k_reg={kreg})": score_matrix(Pt, "RAPS", lam, kreg)}
    combos = [(sc, mode) for sc in S for mode in ("global", "per-class", "pooled")]

    fam_per_class = pd.Series(gt[is_cal]).groupby(yt[is_cal]).nunique()

    # fixed split
    rows, sets_keep, thr_keep = [], {}, {}
    for sc, mode in combos:
        for a in ALPHAS:
            thr = fit(S[sc][is_cal], yt[is_cal], gt[is_cal], a, mode, classes)
            st = sets_from(S[sc][~is_cal], thr)
            sets_keep[(sc, mode, a)], thr_keep[(sc, mode, a)] = st, thr
            rows.append({"score": sc, "thresholds": mode, "alpha": a, "always_included": int(np.isinf(thr).sum()),
                         **stats(st, yt[~is_cal], gt[~is_cal], K)})
    fixed = pd.DataFrame(rows)

    # stability
    rep = []
    for i in range(REPEATS):
        m = A0.halve_groups(te, 1000 + i)
        for sc, mode in combos:
            for a in ALPHAS:
                thr = fit(S[sc][m], yt[m], gt[m], a, mode, classes)
                rep.append({"score": sc, "thresholds": mode, "alpha": a,
                            **stats(sets_from(S[sc][~m], thr), yt[~m], gt[~m], K)})
    rep = pd.DataFrame(rep)
    rsum = rep.groupby(["score", "thresholds", "alpha"], sort=False).agg(
        coverage=("coverage", "mean"), cov_min=("coverage", "min"), cov_max=("coverage", "max"),
        family_coverage=("family_coverage", "mean"), size=("mean_size", "mean"), singleton=("singleton", "mean"),
        worst_class=("worst_class", "mean"), **{f"cov_{k}": (f"cov_{k}", "mean") for k in range(K)})

    # per-class intervals at alpha 0.10 for LAC
    ci = {mode: class_ci(sets_keep[("LAC", mode, 0.10)], yt[~is_cal], gt[~is_cal], K, rng)
          for mode in ("global", "per-class", "pooled")}

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for (sc, mode), d in rep[rep.alpha == 0.10].groupby(["score", "thresholds"], sort=False):
        ax[0].scatter(d.mean_size.mean(), d.coverage.mean(), s=60,
                      marker={"global": "o", "per-class": "s", "pooled": "^"}[mode], label=f"{sc.split(' (')[0]}, {mode}")
    ax[0].axhline(0.90, color="grey", ls=":")
    ax[0].set_xlabel("mean set size"); ax[0].set_ylabel("coverage"); ax[0].legend(fontsize=7)
    ax[0].set_title("alpha 0.10, average of 100 group halvings")
    x = np.arange(K)
    for j, mode in enumerate(("global", "per-class", "pooled")):
        st = fixed[(fixed.score == "LAC") & (fixed.thresholds == mode) & (fixed.alpha == 0.10)].iloc[0]
        v = np.array([st[f"cov_{k}"] for k in range(K)])
        lo = np.array([c[0] for c in ci[mode]]); hi = np.array([c[1] for c in ci[mode]])
        ax[1].errorbar(x + (j - 1) * 0.25, v, yerr=[v - lo, hi - v], fmt="o", capsize=3, label=f"LAC {mode}")
    ax[1].axhline(0.90, color="grey", ls=":")
    ax[1].set_xticks(x, classes, rotation=20); ax[1].set_ylabel("coverage (95% family interval)")
    ax[1].set_title("Per-class coverage, alpha 0.10, fixed split"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "step3.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------------------ report
    pct = lambda v: "-" if pd.isna(v) else f"{v * 100:.1f}%"
    md = ["# Architecture A, step 3: LAC vs APS vs RAPS; global vs per-class vs pooled thresholds", "",
          f"prob-T probabilities (T = {T:.3f}); family-weighted thresholds fitted on the cal half; measured on the test half "
          f"(2,417 crops, 87 families). RAPS tuned on val: lambda = {lam}, k_reg = {kreg}. No retraining.", "",
          "Calibration families per class in the cal half (a per-class threshold needs >= 9 at alpha 0.10 and >= 19 at "
          "0.05; below that the class is always included):", "",
          "| " + " | ".join(classes) + " |", "|" + "---:|" * K,
          "| " + " | ".join(str(int(fam_per_class.get(k, 0))) for k in range(K)) + " |", "",
          "## Fixed split", "",
          "| score | thresholds | alpha | coverage | family coverage | worst class | mean set size | single class | all 6 | empty | classes always included |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in fixed.itertuples():
        md.append(f"| {r.score} | {r.thresholds} | {r.alpha:.2f} | {pct(r.coverage)} | {pct(r.family_coverage)} | "
                  f"{pct(r.worst_class)} | {r.mean_size:.2f} | {pct(r.singleton)} | {pct(r.all_classes)} | "
                  f"{pct(r.empty)} | {r.always_included} |")
    md += ["", f"## Stability: {REPEATS} random group halvings (averages)", "",
           "| score | thresholds | alpha | coverage (min-max) | family coverage | worst class | mean set size | single class |",
           "|---|---|---:|---|---:|---:|---:|---:|"]
    for (sc, mode, a), r in rsum.iterrows():
        md.append(f"| {sc} | {mode} | {a:.2f} | {pct(r.coverage)} ({pct(r.cov_min)}-{pct(r.cov_max)}) | "
                  f"{pct(r.family_coverage)} | {pct(r.worst_class)} | {r['size']:.2f} | {pct(r.singleton)} |")
    md += ["", "## Per-class coverage, LAC, alpha 0.10 (fixed split, 95% interval resampling whole test families)", "",
           "| class | test crops | global | per-class | pooled |", "|---|---:|---|---|---|"]
    for k, c in enumerate(classes):
        cells = []
        for mode in ("global", "per-class", "pooled"):
            st = fixed[(fixed.score == "LAC") & (fixed.thresholds == mode) & (fixed.alpha == 0.10)].iloc[0]
            lo, hi = ci[mode][k]
            cells.append(f"{pct(st[f'cov_{k}'])} ({pct(lo)}-{pct(hi)})")
        md.append(f"| {c} | {int((yt[~is_cal] == k).sum())} | " + " | ".join(cells) + " |")
    md += ["", "Per-class coverage averaged over the 100 halvings, alpha 0.10:", "",
           "| score | thresholds | " + " | ".join(classes) + " |", "|---|---|" + "---:|" * K]
    for (sc, mode, a), r in rsum.iterrows():
        if a == 0.10:
            md.append(f"| {sc} | {mode} | " + " | ".join(pct(r[f'cov_{k}']) for k in range(K)) + " |")
    md += ["", "![step 3](step3.png)", ""]
    (OUT / "README.md").write_text("\n".join(md))
    fixed.to_csv(OUT / "fixed_split.csv", index=False)
    rep.to_csv(OUT / "repeats.csv", index=False)
    (CACHE / "archA_step3.json").write_text(json.dumps({"T_prob": T, "raps_lambda": lam, "raps_kreg": kreg,
        "thresholds": {f"{sc} | {mode} | {a:.2f}": [None if np.isinf(v) else float(v) for v in t]
                       for (sc, mode, a), t in thr_keep.items()}}, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
