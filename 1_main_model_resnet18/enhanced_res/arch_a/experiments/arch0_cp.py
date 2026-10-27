"""Architecture 0: plain split conformal prediction (LAC) on enhanced_res, the baseline for every later method.

From 1_main_model_resnet18/:
  python enhanced_res/arch_a/experiments/arch0_cp.py              # probabilities (cached after the first run), split, CP, report
  python enhanced_res/arch_a/experiments/arch0_cp.py --mapie      # also cross-check the CP sets against the MAPIE library

What it does
  1. enhanced_res probabilities for every grouped val and test crop (cached in data/work/enhanced_res/calibration/).
  2. cal split: d1_grouped's test groups are halved by source group into "cal" and "test" (seed 0, balanced by each
     group's main class). Train and val are unchanged, so enhanced_res is not retrained. Written to
     enhanced_res/arch_a/cal_split.csv, one row per test crop.
  3. On the new test half: coverage and set size of the fixed top-1/2/3 lists that predict.py used to print, and of LAC
     split CP at alpha 0.05 / 0.10 / 0.20. LAC: score = 1 - probability of the true class, q = the
     ceil((n+1)(1-alpha))-th smallest cal score, set = every class with probability >= 1 - q.
  4. Stability: the same over 100 random group halvings of the original test groups, plus 95% intervals that
     resample whole source groups on the fixed split. Secondary check: calibrate on val, evaluate on all of test.

Plain means: the model's output as it is (no temperature scaling), one threshold for all classes, no unknown checks.
Crops of one source group are correlated, so the guarantee holds at the group level only approximately; the
repeated halvings show how much it moves. Writes enhanced_res/arch_a/results/arch0/ and
data/work/enhanced_res/calibration/arch0_thresholds.json (the q values for predict-time use).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from model import EWORK, MAIN, EnhancedRes, load_split
from klshift.common import CropDataset

ALPHAS = (0.05, 0.10, 0.20)
TOPK = (1, 2, 3)
CACHE = EWORK / "calibration"
OUT = HERE.parent / "results" / "arch0"
SPLIT_CSV = HERE.parent / "cal_split.csv"
REPEATS, SEED = 100, 0


# ------------------------------------------------------------------ probabilities
def probabilities(crops, split, batch=64, workers=4):
    """enhanced_res class probabilities for one split, cached as .npz next to the weights (git-ignored)."""
    f = CACHE / f"probs_{split}.npz"
    d = crops[crops.split == split].reset_index(drop=True)
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if list(z["crop_path"]) == list(d.crop_path):
            return d, z["p"]
    model = EnhancedRes()
    out = []
    with torch.no_grad():
        for x, _ in DataLoader(CropDataset(list(d.crop_path)), batch_size=batch, num_workers=workers):
            out.append(model(x).cpu().numpy())
    p = np.concatenate(out)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, p=p, crop_path=np.array(d.crop_path), classes=np.array(model.classes))
    return d, p


# ------------------------------------------------------------------ the cal split
def halve_groups(d, seed):
    """Split test groups in two halves by source group, balanced by each group's main class. Returns a bool mask
    (True = cal) over the crops of d."""
    rng = np.random.default_rng(seed)
    main_cls = d.groupby("src_group").cls.agg(lambda s: s.value_counts().index[0])
    cal = set()
    for _, gs in main_cls.groupby(main_cls):
        g = gs.index.to_numpy().copy()
        rng.shuffle(g)
        cal.update(g[: (len(g) + rng.integers(0, 2)) // 2])   # odd counts: the extra group goes either way
    return d.src_group.isin(cal).to_numpy()


# ------------------------------------------------------------------ methods
def lac_threshold(p_cal, y_cal, alpha):
    """LAC: the ceil((n+1)(1-alpha))-th smallest score 1 - p(true class)."""
    s = np.sort(1 - p_cal[np.arange(len(y_cal)), y_cal])
    k = int(np.ceil((len(s) + 1) * (1 - alpha)))
    return 1.0 if k > len(s) else float(s[k - 1])


def lac_sets(p, q):
    return p >= 1 - q


def topk_sets(p, k):
    s = np.zeros_like(p, dtype=bool)
    np.put_along_axis(s, np.argsort(-p, 1)[:, :k], True, 1)
    return s


def summary(sets, y):
    size = sets.sum(1)
    return {"coverage": float(sets[np.arange(len(y)), y].mean()), "mean_size": float(size.mean()),
            "singleton": float((size == 1).mean()), "empty": float((size == 0).mean())}


def group_ci(sets, y, groups, n=2000, seed=0):
    """95% interval of coverage and mean set size, resampling whole source groups."""
    rng = np.random.default_rng(seed)
    hit, size = sets[np.arange(len(y)), y].astype(float), sets.sum(1).astype(float)
    ug, inv = np.unique(groups, return_inverse=True)
    gh, gs, gn = (np.bincount(inv, w, len(ug)) for w in (hit, size, np.ones_like(hit)))
    draws = rng.integers(0, len(ug), (n, len(ug)))
    cov = gh[draws].sum(1) / gn[draws].sum(1)
    siz = gs[draws].sum(1) / gn[draws].sum(1)
    return np.quantile(cov, [0.025, 0.975]), np.quantile(siz, [0.025, 0.975])


def methods(p_cal, y_cal):
    """name -> function(p) giving prediction sets; the LAC thresholds are fitted on (p_cal, y_cal)."""
    m = {f"top-{k}": (lambda p, k=k: topk_sets(p, k)) for k in TOPK}
    for a in ALPHAS:
        q = lac_threshold(p_cal, y_cal, a)
        m[f"LAC a={a:.2f}"] = (lambda p, q=q: lac_sets(p, q))
    return m


# ------------------------------------------------------------------ MAPIE cross-check
def mapie_check(p_cal, y_cal, p_te, alpha):
    """Same LAC sets from the MAPIE library (prefit classifier = the stored probabilities)."""
    from mapie.classification import SplitConformalClassifier
    from sklearn.base import BaseEstimator, ClassifierMixin

    class Probs(ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            self.classes_ = np.arange(X.shape[1])
            return self

        def predict_proba(self, X):
            return X

        def predict(self, X):
            return X.argmax(1)

    est = Probs().fit(p_cal, y_cal)
    m = SplitConformalClassifier(est, confidence_level=1 - alpha, conformity_score="lac", prefit=True)
    m.conformalize(p_cal, y_cal)
    return m.predict_set(p_te)[1][:, :, 0]


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mapie", action="store_true", help="cross-check LAC against the MAPIE library")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    crops = load_split()
    te, p_te_all = probabilities(crops, "test", workers=a.workers)
    va, p_va = probabilities(crops, "val", workers=a.workers)
    classes = list(np.load(CACHE / "probs_test.npz", allow_pickle=True)["classes"])
    cidx = {c: i for i, c in enumerate(classes)}
    y_te_all, y_va = te.cls.map(cidx).to_numpy(), va.cls.map(cidx).to_numpy()
    acc = float((p_te_all.argmax(1) == y_te_all).mean())
    print(f"enhanced_res test accuracy from cached probabilities: {acc:.4f} (reference 0.8045)")

    # 2. the fixed cal split, written as a manifest over d1_grouped's test crops (train and val are unchanged)
    is_cal = halve_groups(te, SEED)
    man = te[["crop_path", "cls", "src_group"]].copy()
    man["crop_path"] = [str(Path(p).relative_to(MAIN)) for p in man.crop_path]
    man["split"] = np.where(is_cal, "cal", "test")
    man.to_csv(SPLIT_CSV, index=False)
    assert man.groupby("src_group").split.nunique().max() == 1
    counts = pd.concat([crops[crops.split.isin(["train", "val"])][["cls", "src_group", "split"]], man]).groupby(
        "split").agg(crops=("cls", "size"), groups=("src_group", "nunique")).reindex(["train", "val", "cal", "test"])
    print(counts.to_string())

    p_cal, y_cal = p_te_all[is_cal], y_te_all[is_cal]
    p_ev, y_ev, g_ev = p_te_all[~is_cal], y_te_all[~is_cal], te.src_group.to_numpy()[~is_cal]

    # 3. fixed split results
    rows, sets_by = [], {}
    for name, f in methods(p_cal, y_cal).items():
        s = f(p_ev)
        sets_by[name] = s
        (clo, chi), (slo, shi) = group_ci(s, y_ev, g_ev)
        rows.append({"method": name, **summary(s, y_ev), "coverage_lo": clo, "coverage_hi": chi,
                     "size_lo": slo, "size_hi": shi})
    fixed = pd.DataFrame(rows).set_index("method")
    q = {f"{al:.2f}": lac_threshold(p_cal, y_cal, al) for al in ALPHAS}

    per_class = pd.DataFrame({name: [s[y_ev == c, c].mean() for c in range(len(classes))]
                              for name, s in sets_by.items()}, index=classes)
    per_class["test crops"] = np.bincount(y_ev, minlength=len(classes))
    size_hist = pd.DataFrame({name: np.bincount(s.sum(1), minlength=len(classes) + 1) for name, s in sets_by.items()
                              if name.startswith("LAC")})

    # 4a. repeated group halvings
    rep = []
    for r in range(REPEATS):
        m = halve_groups(te, 1000 + r)
        for name, f in methods(p_te_all[m], y_te_all[m]).items():
            rep.append({"repeat": r, "method": name, **summary(f(p_te_all[~m]), y_te_all[~m])})
    rep = pd.DataFrame(rep)
    rep_sum = rep.groupby("method", sort=False).agg(
        coverage_mean=("coverage", "mean"), coverage_min=("coverage", "min"), coverage_max=("coverage", "max"),
        size_mean=("mean_size", "mean"), singleton_mean=("singleton", "mean"))

    # 4b. calibrate on val, evaluate on all of test
    val_rows = [{"method": n, **summary(f(p_te_all), y_te_all)} for n, f in methods(p_va, y_va).items()]
    val_cal = pd.DataFrame(val_rows).set_index("method")

    # optional MAPIE cross-check
    mapie_note = "not run (use --mapie)"
    if a.mapie:
        agree = []
        for al in ALPHAS:
            ms = mapie_check(p_cal, y_cal, p_ev, al)
            agree.append(f"alpha {al:.2f}: {np.mean(np.all(ms == sets_by[f'LAC a={al:.2f}'], 1)) * 100:.2f}% identical sets")
        mapie_note = "; ".join(agree)
        print("MAPIE cross-check:", mapie_note)

    # thresholds for later predict-time use (git-ignored, next to the weights)
    (CACHE / "arch0_thresholds.json").write_text(json.dumps(
        {"method": "LAC split CP, plain enhanced_res probabilities", "split_file": str(SPLIT_CSV.relative_to(MAIN)),
         "split_seed": SEED, "classes": classes, "q": q,
         "rule": "prediction set = classes with probability >= 1 - q[alpha]"}, indent=2))

    # ------------------------------------------------------------------ figures
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, g in rep.groupby("method", sort=False):
        ax[0].scatter(g.mean_size, g.coverage, s=8, alpha=.35, label=name,
                      marker="o" if name.startswith("LAC") else "s")
    for al in ALPHAS:
        ax[0].axhline(1 - al, color="grey", lw=.6, ls=":")
    ax[0].set_xlabel("mean prediction-set size"); ax[0].set_ylabel("coverage (true class in the set)")
    ax[0].set_title(f"{REPEATS} random group halvings of the test groups"); ax[0].legend(fontsize=7)
    x = np.arange(len(classes) + 1)
    for i, col in enumerate(size_hist.columns):
        ax[1].bar(x + (i - 1) * 0.27, size_hist[col] / size_hist[col].sum(), 0.27, label=col)
    ax[1].set_xticks(x); ax[1].set_xlabel("prediction-set size"); ax[1].set_ylabel("share of test crops")
    ax[1].set_title("LAC set sizes (fixed split)"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "coverage_vs_size.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------------------ report
    def pct(v):
        return f"{v * 100:.1f}%"

    md = ["# Architecture 0: plain LAC split conformal prediction on enhanced_res", "",
          "Baseline for every later calibration method. The model's probabilities as they are (no temperature "
          "scaling), one threshold for all classes, no unknown checks.", "",
          "## Data", "",
          f"d1_grouped's test groups are halved by source group into `cal` and `test` (seed {SEED}, balanced by each "
          "group's main class); train and val are unchanged, so enhanced_res is not retrained. "
          f"Manifest: `enhanced_res/arch_a/cal_split.csv`.", "",
          "| split | crops | source groups |", "|---|---:|---:|"]
    md += [f"| {s} | {r.crops:,} | {r.groups} |" for s, r in counts.iterrows()]
    md += ["", f"enhanced_res accuracy on all original test crops (from the cached probabilities): {acc:.4f}.", "",
           "## Fixed split: coverage and set size on the new test half", "",
           "Coverage = share of crops whose true class is in the set. 95% intervals resample whole source groups.", "",
           "| method | coverage (95% CI) | mean set size (95% CI) | single class | empty set |",
           "|---|---|---|---:|---:|"]
    for name, r in fixed.iterrows():
        md.append(f"| {name} | {pct(r.coverage)} ({pct(r.coverage_lo)}-{pct(r.coverage_hi)}) | "
                  f"{r.mean_size:.2f} ({r.size_lo:.2f}-{r.size_hi:.2f}) | {pct(r.singleton)} | {pct(r['empty'])} |")
    md += ["", "LAC thresholds (probability cut-off = 1 - q): " +
           ", ".join(f"alpha {k}: q = {v:.4f} (keep classes with p >= {1 - v:.4f})" for k, v in q.items()), "",
           "## Per-class coverage (fixed split)", "",
           "| class | " + " | ".join(per_class.columns) + " |", "|---|" + "---:|" * len(per_class.columns)]
    for c, r in per_class.iterrows():
        md.append(f"| {c} | " + " | ".join(pct(v) for v in r.iloc[:-1]) + f" | {int(r.iloc[-1])} |")
    md += ["", "Plain LAC only targets coverage over all classes together; rare classes can fall below 1 - alpha. "
           "Per-class (Mondrian) CP in architecture A addresses that.", "",
           "## Set-size distribution (LAC, fixed split, crops)", "",
           "| set size | " + " | ".join(size_hist.columns) + " |", "|---|" + "---:|" * len(size_hist.columns)]
    for k, r in size_hist.iterrows():
        md.append(f"| {k} | " + " | ".join(f"{int(v):,}" for v in r) + " |")
    md += ["", f"## Stability: {REPEATS} random group halvings", "",
           "| method | coverage mean (min-max) | mean set size | single class |", "|---|---|---:|---:|"]
    for name, r in rep_sum.iterrows():
        md.append(f"| {name} | {pct(r.coverage_mean)} ({pct(r.coverage_min)}-{pct(r.coverage_max)}) | "
                  f"{r.size_mean:.2f} | {pct(r.singleton_mean)} |")
    md += ["", "## Secondary check: calibrate on val, evaluate on all test crops", "",
           "val was already used to choose the kNN weight and MLP epochs, so this is slightly optimistic.", "",
           "| method | coverage | mean set size | single class |", "|---|---:|---:|---:|"]
    for name, r in val_cal.iterrows():
        md.append(f"| {name} | {pct(r.coverage)} | {r.mean_size:.2f} | {pct(r.singleton)} |")
    md += ["", f"MAPIE cross-check of the LAC sets: {mapie_note}.", "",
           "![coverage vs size](coverage_vs_size.png)", ""]
    (OUT / "README.md").write_text("\n".join(md))
    fixed.to_csv(OUT / "fixed_split.csv")
    rep.to_csv(OUT / "repeats.csv", index=False)
    per_class.to_csv(OUT / "per_class_coverage.csv")
    print("\n".join(md))


if __name__ == "__main__":
    main()
