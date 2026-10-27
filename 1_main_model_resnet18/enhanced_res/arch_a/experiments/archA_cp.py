"""Architecture A, step 1: temperature scaling + family-weighted CP, compared against architecture 0.

From 1_main_model_resnet18/:
  python enhanced_res/arch_a/experiments/archA_cp.py              # features (cached after the first run), T on val, CP, report

What it does
  1. For every grouped val and test crop: the 768 features, the 5 MLP heads' logits and the 5-NN vote
     (cached in data/work/enhanced_res/calibration/feats_<split>.npz). Checked: re-blending them reproduces the
     probabilities architecture 0 used.
  2. Temperature scaling, fitted on val by minimising the negative log-likelihood (Guo et al. 2017):
       (a) "logit-T":  divide each head's logits by T, softmax, average, blend 50/50 with the kNN vote as before.
       (b) "prob-T":   temper the final blended probabilities, p_i^(1/T) renormalised. Runs after the model's
           output (no ONNX change) and never changes the top class.
  3. CP on the cal half, measured on the test half (enhanced_res/arch_a/cal_split.csv, as architecture 0), for each
     probability version x two calibration rules:
       crop    plain LAC, every calibration crop counts once (architecture 0's rule)
       family  each calibration crop weighted 1/(its family's crop count), so every photo family counts once
     Coverage is reported per crop and per family (each test family's own coverage, averaged).
  4. Stability over the same 100 random group halvings as architecture 0.

Writes enhanced_res/arch_a/results/archA/step1/ and data/work/enhanced_res/calibration/archA_step1.json (T values and
thresholds). Val was used to tune enhanced_res (kNN weight, epochs); using it again for T is standard practice
and keeps cal and test untouched.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader

import arch0_cp as A0
from model import MAIN, EnhancedRes, load_split
from klshift.common import CropDataset
from klshift.mlp import knn_proba, l2n

CACHE = A0.CACHE
OUT = HERE.parent / "results" / "archA" / "step1"
ALPHAS = (0.05, 0.10, 0.20)
REPEATS = 100


# ------------------------------------------------------------------ features, logits, kNN vote
def model_parts(crops, split, batch=64, workers=4):
    f = CACHE / f"feats_{split}.npz"
    d = crops[crops.split == split].reset_index(drop=True)
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if list(z["crop_path"]) == list(d.crop_path):
            return d, z["feats"], z["logits"], z["knn"]
    m = EnhancedRes()
    F = []
    with torch.no_grad():
        for x, _ in DataLoader(CropDataset(list(d.crop_path)), batch_size=batch, num_workers=workers):
            F.append(m.features(x))
        feats = torch.cat(F)
        logits = torch.stack([h(feats) for h in m.heads])                  # 5 x N x 6
        knn = knn_proba(m.bank, m.bank_y, l2n(feats), 5, len(m.classes))
    assert m.alpha == 0.5 and m.tau == 0, "the blend recipe changed: update blend()"
    out = dict(feats=feats.cpu().numpy(), logits=logits.cpu().numpy(), knn=knn.cpu().numpy())
    np.savez(f, crop_path=np.array(d.crop_path), classes=np.array(m.classes), **out)
    return d, out["feats"], out["logits"], out["knn"]


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def blend(logits, knn, T=1.0):
    """enhanced_res's output with the heads' logits divided by T (T = 1: the model as shipped)."""
    return 0.5 * softmax(logits / T).mean(0) + 0.5 * knn


def temper(p, T):
    q = np.clip(p, 1e-12, None) ** (1.0 / T)
    return q / q.sum(1, keepdims=True)


def nll(p, y):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def fit_T(fn, y):
    r = minimize_scalar(lambda lt: nll(fn(np.exp(lt)), y), bounds=(np.log(0.05), np.log(20)), method="bounded")
    return float(np.exp(r.x))


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(conf[m].mean() - (pred[m] == y[m]).mean())
    return float(e)


def reliability(p, y):
    conf, pred = p.max(1), p.argmax(1)
    rows = []
    for lo, hi in [(0, .5), (.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, .95), (.95, 1.0001)]:
        m = (conf >= lo) & (conf < hi)
        rows.append({"bin": f"{lo:.2f}-{min(hi, 1):.2f}", "crops": int(m.sum()),
                     "stated": float(conf[m].mean()) if m.any() else np.nan,
                     "right": float((pred[m] == y[m]).mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ family-weighted CP
def family_threshold(p_cal, y_cal, g_cal, alpha):
    """LAC threshold where every calibration family carries total weight 1 (its crops share it)."""
    s = 1 - p_cal[np.arange(len(y_cal)), y_cal]
    _, inv, cnt = np.unique(g_cal, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    G = len(cnt)
    order = np.argsort(s)
    cum = np.cumsum(w[order])
    need = (1 - alpha) * (G + 1)
    k = np.searchsorted(cum, need - 1e-9)
    return 1.0 if k >= len(s) else float(s[order][k])


def coverage_by_family(sets, y, g):
    hit = sets[np.arange(len(y)), y]
    return float(pd.Series(hit).groupby(g).mean().mean())


def evaluate(p, y, g, is_cal, rule, alpha):
    """Thresholds on the cal half, measured on the test half."""
    pc, yc, gc = p[is_cal], y[is_cal], g[is_cal]
    q = A0.lac_threshold(pc, yc, alpha) if rule == "crop" else family_threshold(pc, yc, gc, alpha)
    S = A0.lac_sets(p[~is_cal], q)
    yt, gt = y[~is_cal], g[~is_cal]
    out = A0.summary(S, yt)
    out["family_coverage"] = coverage_by_family(S, yt, gt)
    out["q"] = q
    return out, S


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    crops = load_split()
    va, fv, Lv, Kv = model_parts(crops, "val")
    te, ft, Lt, Kt = model_parts(crops, "test")
    classes = list(np.load(CACHE / "feats_test.npz", allow_pickle=True)["classes"])
    cid = {c: i for i, c in enumerate(classes)}
    yv, yt = va.cls.map(cid).to_numpy(), te.cls.map(cid).to_numpy()

    # sanity: re-blending reproduces architecture 0's cached probabilities
    p0 = np.load(CACHE / "probs_test.npz", allow_pickle=True)["p"]
    base_t = blend(Lt, Kt)
    drift = float(np.abs(base_t - p0).max())
    print(f"re-blended vs cached probabilities: max difference {drift:.2e}, same top class "
          f"{(base_t.argmax(1) == p0.argmax(1)).mean() * 100:.2f}%")

    # temperature on val
    base_v = blend(Lv, Kv)
    T_logit = fit_T(lambda T: blend(Lv, Kv, T), yv)
    T_prob = fit_T(lambda T: temper(base_v, T), yv)
    print(f"fitted on val: logit-T = {T_logit:.3f}, prob-T = {T_prob:.3f}")

    versions = {"as shipped (architecture 0)": base_t,
                f"logit-T (T={T_logit:.2f})": blend(Lt, Kt, T_logit),
                f"prob-T (T={T_prob:.2f})": temper(base_t, T_prob)}

    sp = pd.read_csv(HERE.parent / "cal_split.csv")
    sp["crop_path"] = [str(MAIN / p) for p in sp.crop_path]
    te = te.merge(sp[["crop_path", "split"]].rename(columns={"split": "half"}), on="crop_path", validate="1:1")
    is_cal = (te.half == "cal").to_numpy()
    g = te.src_group.to_numpy()

    # calibration quality on the test half
    calib = []
    for name, p in versions.items():
        pt, yy = p[~is_cal], yt[~is_cal]
        calib.append({"version": name, "accuracy": float((pt.argmax(1) == yy).mean()),
                      "mean_confidence": float(pt.max(1).mean()), "ECE": ece(pt, yy), "NLL": nll(pt, yy)})
    calib = pd.DataFrame(calib).set_index("version")
    rel = {name: reliability(p[~is_cal], yt[~is_cal]) for name, p in versions.items()}

    # CP on the fixed split
    rows, sets = [], {}
    for name, p in versions.items():
        for rule in ("crop", "family"):
            for a in ALPHAS:
                r, S = evaluate(p, yt, g, is_cal, rule, a)
                sets[(name, rule, a)] = S
                rows.append({"version": name, "calibration": rule, "alpha": a, **r})
    fixed = pd.DataFrame(rows)

    per_class = pd.DataFrame({f"{n.split(' (')[0]} | {r} | a={a:.2f}": [S[yt[~is_cal] == c, c].mean()
                                                                         for c in range(len(classes))]
                              for (n, r, a), S in sets.items() if a == 0.10}, index=classes)

    # stability over random group halvings
    rep = []
    for i in range(REPEATS):
        m = A0.halve_groups(te, 1000 + i)
        for name, p in versions.items():
            for rule in ("crop", "family"):
                for a in ALPHAS:
                    r, _ = evaluate(p, yt, g, m, rule, a)
                    rep.append({"repeat": i, "version": name, "calibration": rule, "alpha": a, **r})
    rep = pd.DataFrame(rep)
    rep_sum = rep.groupby(["version", "calibration", "alpha"], sort=False).agg(
        coverage=("coverage", "mean"), coverage_min=("coverage", "min"), coverage_max=("coverage", "max"),
        family_coverage=("family_coverage", "mean"), size=("mean_size", "mean"),
        singleton=("singleton", "mean"), full=("mean_size", lambda s: np.nan))

    full6 = {k: float((S.sum(1) == len(classes)).mean()) for k, S in sets.items()}
    fixed["all_6_classes"] = [full6[(r.version, r.calibration, r.alpha)] for r in fixed.itertuples()]

    (CACHE / "archA_step1.json").write_text(json.dumps(
        {"T_logit": T_logit, "T_prob": T_prob, "fitted_on": "val (NLL)",
         "thresholds": {f"{r.version} | {r.calibration} | {r.alpha:.2f}": r.q for r in fixed.itertuples()}},
        indent=2))

    # ------------------------------------------------------------------ figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, p in versions.items():
        r = rel[name]
        ax[0].plot(r.stated, r.right, "o-", label=name)
    ax[0].plot([0, 1], [0, 1], "k:", lw=1)
    ax[0].set_xlabel("stated confidence (bin mean)"); ax[0].set_ylabel("actually right")
    ax[0].set_title("Reliability on the test half"); ax[0].legend(fontsize=7)
    for name in versions:
        s = rep[(rep.version == name) & (rep.calibration == "crop")].groupby("alpha")[["mean_size", "coverage"]].mean()
        ax[1].plot(s.mean_size, s.coverage, "o-", label=f"{name}, crop")
        s = rep[(rep.version == name) & (rep.calibration == "family")].groupby("alpha")[["mean_size", "coverage"]].mean()
        ax[1].plot(s.mean_size, s.coverage, "s--", label=f"{name}, family")
    for a in ALPHAS:
        ax[1].axhline(1 - a, color="grey", lw=.5, ls=":")
    ax[1].set_xlabel("mean set size"); ax[1].set_ylabel("coverage")
    ax[1].set_title(f"CP, average of {REPEATS} group halvings (alpha 0.20 / 0.10 / 0.05)"); ax[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(OUT / "step1.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------------------ report
    pct = lambda v: f"{v * 100:.1f}%"
    md = ["# Architecture A, step 1: temperature scaling + family-weighted CP", "",
          "Compared against architecture 0 (model as shipped, crop-weighted LAC). T fitted on val by NLL; CP "
          "thresholds on the cal half; everything below measured on the test half (2,417 crops, 87 families). "
          "No retraining.", "",
          f"Re-blending the cached heads' logits and kNN vote reproduces architecture 0's probabilities "
          f"(max difference {drift:.1e}).", "",
          f"Fitted temperatures: logit-T = **{T_logit:.3f}**, prob-T = **{T_prob:.3f}** (T > 1 softens).", "",
          "## Calibration of the stated confidence (test half)", "",
          "| version | accuracy | mean stated confidence | ECE | NLL |", "|---|---:|---:|---:|---:|"]
    for n, r in calib.iterrows():
        md.append(f"| {n} | {pct(r.accuracy)} | {pct(r.mean_confidence)} | {r.ECE:.3f} | {r.NLL:.3f} |")
    md += ["", "Reliability (stated vs actually right, per confidence bin):", ""]
    hdr = "| bin | " + " | ".join(f"{n}: crops / stated / right" for n in versions) + " |"
    md += [hdr, "|---|" + "---|" * len(versions)]
    for i in range(len(next(iter(rel.values())))):
        cells = []
        for n in versions:
            r = rel[n].iloc[i]
            cells.append(f"{r.crops} / {pct(r.stated) if r.crops else '-'} / {pct(r.right) if r.crops else '-'}")
        md.append(f"| {rel[next(iter(versions))].iloc[i].bin} | " + " | ".join(cells) + " |")
    md += ["", "## CP on the fixed split", "",
           "Coverage per crop (as architecture 0) and per family (each test family's coverage, averaged).", "",
           "| version | calibration | alpha | coverage (crop) | coverage (family) | mean set size | single class | all 6 | empty |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in fixed.itertuples():
        md.append(f"| {r.version} | {r.calibration} | {r.alpha:.2f} | {pct(r.coverage)} | {pct(r.family_coverage)} | "
                  f"{r.mean_size:.2f} | {pct(r.singleton)} | {pct(r.all_6_classes)} | {pct(r.empty)} |")
    md += ["", f"## Stability: {REPEATS} random group halvings", "",
           "| version | calibration | alpha | coverage mean (min-max) | family coverage | mean set size | single class |",
           "|---|---|---:|---|---:|---:|---:|"]
    for (v, c, a), r in rep_sum.iterrows():
        md.append(f"| {v} | {c} | {a:.2f} | {pct(r.coverage)} ({pct(r.coverage_min)}-{pct(r.coverage_max)}) | "
                  f"{pct(r.family_coverage)} | {r['size']:.2f} | {pct(r.singleton)} |")
    md += ["", "## Per-class coverage at alpha 0.10 (fixed split)", "",
           "| class | " + " | ".join(per_class.columns) + " |", "|---|" + "---:|" * len(per_class.columns)]
    for c, r in per_class.iterrows():
        md.append(f"| {c} | " + " | ".join(pct(v) for v in r) + " |")
    md += ["", "![step 1](step1.png)", ""]
    (OUT / "README.md").write_text("\n".join(md))
    fixed.to_csv(OUT / "fixed_split.csv", index=False)
    rep.to_csv(OUT / "repeats.csv", index=False)
    calib.to_csv(OUT / "calibration.csv")
    print("\n".join(md))


if __name__ == "__main__":
    main()
