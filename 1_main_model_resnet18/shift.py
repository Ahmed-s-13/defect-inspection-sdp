"""Step 6: KL divergence between dataset embedding distributions, figures, summary.

Every number is computed for two versions:
  all    - every crop
  dedup  - D2-D5 crops whose original image does NOT match a D1 image (in_d1 == False)
D1's sample is always D1-held-out (val+test). PCA is fitted on D1-held-out only.

Outputs (in <results>): kl_matrix.csv, kl_per_class.csv, pca_variance.csv,
  pca_scatter_all.png, pca_scatter_dedup.png, umap.png, kl_heatmap_dedup.png,
  kl_corruption.png, summary.md
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors

from klshift.common import ALL_SETS, CLASS_MAP, DEFECT, DERIVED, NO_DAMAGE, SEPARATE, class_key, source_of

# Reference categorical palette (light); markers give a second, colour-free encoding.
COLORS = {"D1": "#2a78d6", "D2": "#eb6834", "D3": "#1baf7a", "D4": "#eda100", "D5": "#e87ba4",
          "D6": "#6250d6", "D7": "#e34948", "D8": "#1baf7a", "D9": "#008300", "D1nd": "#52514e"}
MARKERS = {"D1": "o", "D2": "s", "D3": "^", "D4": "D", "D5": "v",
           "D6": "P", "D7": "X", "D8": "^", "D9": "*", "D1nd": "h"}
for _d, (_src, _) in DERIVED.items():  # patch sets share their source's colour and marker
    COLORS.setdefault(_d, COLORS[_src]); MARKERS.setdefault(_d, MARKERS[_src])
COLORS.update({"D10": "#6250d6", "D10n": "#1baf7a", "D10a": "#e34948"})
MARKERS.update({"D10": "p", "D10n": "d", "D10a": "8"})
NAMES = {"D1nd": "D1 no-damage patches", **{f"D{i}p": f"D{i} skin patches" for i in range(6, 10)},
         "D10": "fuselage mock-up, all frames", "D10n": "mock-up, normal frames", "D10a": "mock-up, anomaly frames"}
CORR_COLORS = {"gaussian_noise": "#2a78d6", "blur": "#eb6834", "low_contrast": "#1baf7a", "jpeg": "#6250d6"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b"})


# ---------------------------------------------------------------- estimators

def gauss_kl(P, Q):
    """Closed-form KL(P||Q) between Gaussians fitted with Ledoit-Wolf shrinkage."""
    k = P.shape[1]
    lp, lq = LedoitWolf().fit(P), LedoitWolf().fit(Q)
    Sp, Sq = lp.covariance_, lq.covariance_
    dm = lq.location_ - lp.location_
    tr = np.trace(np.linalg.solve(Sq, Sp))
    maha = dm @ np.linalg.solve(Sq, dm)
    ld = np.linalg.slogdet(Sq)[1] - np.linalg.slogdet(Sp)[1]
    return 0.5 * (tr + maha - k + ld)


def knn_kl(P, Q, k=5):
    """Wang, Kulkarni & Verdu (2009) k-NN estimate of KL(P||Q)."""
    n, d = P.shape
    m = len(Q)
    rho = NearestNeighbors(n_neighbors=k + 1).fit(P).kneighbors(P)[0][:, k]  # excludes self
    nu = NearestNeighbors(n_neighbors=k).fit(Q).kneighbors(P)[0][:, k - 1]
    eps = 1e-12
    return d / n * np.sum(np.log(np.maximum(nu, eps) / np.maximum(rho, eps))) + np.log(m / (n - 1))


ESTIMATORS = {"gaussian": gauss_kl, "knn": knn_kl}


def both_dirs(P, Q):
    out = {}
    for name, f in ESTIMATORS.items():
        pq, qp = f(P, Q), f(Q, P)
        out[name] = (pq, qp, (pq + qp) / 2)
    return out


# ---------------------------------------------------------------- sampling helpers

def group_halves(groups, rng):
    """Split row indices into two halves with no group on both sides."""
    ug = rng.permutation(np.unique(groups))
    sizes = pd.Series(groups).value_counts()
    a, b, na, nb = [], [], 0, 0
    for g in ug:
        if na <= nb:
            a.append(g); na += sizes[g]
        else:
            b.append(g); nb += sizes[g]
    return np.flatnonzero(np.isin(groups, a)), np.flatnonzero(np.isin(groups, b))


def pick(idx, n, rng):
    return rng.choice(idx, n, replace=False)


def summarize(records):
    """records: list of dicts with keys incl. pq, qp, sym per seed -> mean/std rows."""
    df = pd.DataFrame(records)
    keys = [c for c in df.columns if c not in ("seed", "pq", "qp", "sym")]
    g = df.groupby(keys, sort=False, dropna=False)
    out = g.agg(kl_pq_mean=("pq", "mean"), kl_pq_std=("pq", "std"),
                kl_qp_mean=("qp", "mean"), kl_qp_std=("qp", "std"),
                sym_mean=("sym", "mean"), sym_std=("sym", "std"), n_seeds=("sym", "size")).reset_index()
    return out


# ---------------------------------------------------------------- figures

def ellipse(ax, xy, color, nsig=2):
    if len(xy) < 3:
        return
    cov = np.cov(xy.T)
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
    w, h = 2 * nsig * np.sqrt(np.maximum(vals[::-1], 0))
    ax.add_patch(Ellipse(xy.mean(0), w, h, angle=ang, fill=False, lw=2, color=color))


def circles_fig(Zk, samples, skipped, path, var2, k, dedup):
    """Points per dataset with dashed 2-sigma circles, in D1's PCA plane and in an LDA plane."""
    names = list(samples)
    lab = np.concatenate([[d] * len(samples[d]) for d in names])
    allpts = np.concatenate([Zk[samples[d]] for d in names])
    lda = LinearDiscriminantAnalysis(n_components=2).fit(allpts, lab) if len(names) >= 3 else None
    panels = [("D1's PCA plane (unsupervised)", allpts[:, :2],
               f"PC1 ({var2[0] * 100:.0f}% of D1 variance)", f"PC2 ({var2[1] * 100:.0f}%)")]
    if lda is not None:
        panels.append((f"LDA plane (directions that best separate the datasets, from PCA-{k})",
                       lda.transform(allpts), "LD1", "LD2"))
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2 * len(panels), 6.4), squeeze=False)
    for ax, (title, P2, xl, yl) in zip(axes[0], panels):
        span = np.ptp(P2, axis=0)
        placed = []
        for d in names:
            xy = P2[lab == d]
            c = COLORS[d]
            ax.scatter(*xy.T, s=16, marker=MARKERS[d], color=c, alpha=0.55, lw=0, label=f"{d} (n={len(xy)})")
            cov = np.cov(xy.T)
            vals, vecs = np.linalg.eigh(cov)
            ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
            w, h = 4 * np.sqrt(np.maximum(vals[::-1], 0))
            ctr = xy.mean(0)
            ax.add_patch(Ellipse(ctr, w, h, angle=ang, facecolor=c, alpha=0.07, lw=0))
            ax.add_patch(Ellipse(ctr, w, h, angle=ang, fill=False, ls=(0, (6, 4)), lw=2.2, color=c))
            th = np.radians(ang)  # label sits on top of its circle so nested circles keep their labels apart
            top = np.sqrt((w / 2 * np.sin(th)) ** 2 + (h / 2 * np.cos(th)) ** 2)
            pos = np.array([ctr[0], ctr[1] + top])
            if any(np.all(np.abs(pos - q) < 0.08 * span) for q in placed):  # would cover another label
                pos = np.array([ctr[0], ctr[1] - top])
            placed.append(pos)
            ax.text(*pos, d, fontsize=12, fontweight="bold", color="#0b0b0b", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, lw=1.2, alpha=0.9))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(color="#e4e3df", lw=0.6)
        ax.legend(fontsize=9, markerscale=1.6, frameon=False, loc="best")
    note = ("Dashed circles: 2-sigma region of each dataset; label on top of its circle. One random crop per image. "
            + ("Images found in D1 removed from the other datasets. " if dedup else "All images, including those also in D1. ")
            + "Left: the first two directions of D1's own variation (unsupervised). Right: LDA picks the two directions "
            "that best separate the datasets, so it shows them more separated than an unsupervised view would.")
    if skipped:
        note += f" Not shown (too few images): {', '.join(skipped)}."
    fig.text(0.01, 0.01, note, fontsize=8.5, color="#52514e", wrap=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def scatter_fig(Z2, sets, dup_mask, title, path, var2, plot_n, rng):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for ds, idx in sets.items():
        sub = idx if len(idx) <= plot_n else pick(idx, plot_n, rng)
        dup = dup_mask[sub]
        c = COLORS[ds]
        ax.scatter(*Z2[sub[~dup]].T, s=14, marker=MARKERS[ds], color=c, alpha=0.45, lw=0,
                   label=f"{ds} (n={len(idx)})")
        if dup.any():
            ax.scatter(*Z2[sub[dup]].T, s=16, marker=MARKERS[ds], facecolors="none", edgecolors=c,
                       alpha=0.6, lw=0.8)
        ellipse(ax, Z2[idx], c)
    if dup_mask.any():
        ax.scatter([], [], s=16, marker="o", facecolors="none", edgecolors="#52514e",
                   label="hollow = image also in D1")
    ax.set_xlabel(f"PC1 of D1-held-out ({var2[0] * 100:.1f}% of D1 variance)")
    ax.set_ylabel(f"PC2 ({var2[1] * 100:.1f}%)")
    ax.set_title(title, fontsize=11)
    ax.grid(color="#e4e3df", lw=0.6)
    ax.legend(fontsize=9, markerscale=1.6, frameon=False, loc="best")
    fig.text(0.01, 0.01, "Ellipses: 2-sigma of each dataset's points in this plane. "
             "Up to %d points per dataset drawn." % plot_n, fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def heatmap_fig(tabs, dss, caption, path):
    sz = max(5.6, 0.62 * len(dss) + 2.5)
    fs = 9 if len(dss) <= 6 else 6.5
    fig, axes = plt.subplots(1, len(tabs), figsize=(sz * 1.1 * len(tabs), sz))
    for ax, (name, M, S) in zip(np.atleast_1d(axes), tabs):
        vmax = np.nanmax(M) if np.isfinite(M).any() else 1
        ax.imshow(np.where(np.isnan(M), 0, M), cmap="Blues", vmin=0, vmax=vmax)
        for i in range(len(dss)):
            for j in range(len(dss)):
                if i == j:
                    t = "-"
                elif np.isnan(M[i, j]):
                    t = "too few"
                else:
                    t = f"{M[i, j]:.2f}\n±{S[i, j]:.2f}"
                v = 0 if np.isnan(M[i, j]) else M[i, j]
                ax.text(j, i, t, ha="center", va="center", fontsize=fs,
                        color="white" if v > 0.6 * vmax else "#0b0b0b")
        ax.set_xticks(range(len(dss)), dss)
        ax.set_yticks(range(len(dss)), dss)
        ax.set_title(name, fontsize=11)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.text(0.01, 0.01, caption, fontsize=8.5, color="#52514e", wrap=True)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def corruption_fig(C, ref, path, k, n):
    ests = ["gaussian", "knn"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, est in zip(axes, ests):
        sub = C[C.estimator == est]
        for kind, g in sub.groupby("corruption", sort=False):
            g = g.sort_values("severity")
            ax.errorbar(g.severity, g.kl_pq_mean, yerr=g.kl_pq_std, marker="o", ms=6, lw=2,
                        capsize=3, color=CORR_COLORS.get(kind), label=kind.replace("_", " "))
        ax.axhline(ref[est]["in_distribution"], color="#52514e", ls="--", lw=1,
                   label="D1-train vs D1-held-out")
        ax.set_xticks([0, 1, 2, 3], ["0\n(noise floor)", "1", "2", "3"])
        ax.set_xlabel("corruption severity")
        ax.set_ylabel("KL(corrupted half A || clean half B)")
        ax.set_title(f"{'Gaussian (Ledoit-Wolf)' if est == 'gaussian' else 'k-NN (k=5)'}", fontsize=11)
        ax.grid(color="#e4e3df", lw=0.6)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Sanity check: KL must rise with corruption severity (PCA k={k}, n={n}, mean ± sd over seeds)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="data/work")
    ap.add_argument("--results", default="results")
    ap.add_argument("--k", type=int, default=32, help="main PCA dimension")
    ap.add_argument("--ks", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--n-cap", type=int, default=2000, help="max common sample size")
    ap.add_argument("--min-n", type=int, default=100, help="datasets with fewer crops get no KL value")
    ap.add_argument("--class-k", type=int, default=16, help="PCA dimension for per-class KL")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--plot-n", type=int, default=1500)
    ap.add_argument("--circle-n", type=int, default=300, help="points per dataset in overlap_circles_*.png")
    ap.add_argument("--all-copies", action="store_true",
                    help="keep every near-copy inside a dataset (default: one image per cluster)")
    ap.add_argument("--findings", default="findings.md", help="hand-written findings inserted into summary.md")
    ap.add_argument("--sets", nargs="+", default=None,
                    help=f"crop sets to compare (D1 always included); default: all except {', '.join(SEPARATE)}")
    ap.add_argument("--out", default=None, help="output folder (default: --results)")
    ap.add_argument("--per-crop", action="store_true",
                    help="sample crops freely (default: at most one crop per original image per sample)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    work, res = Path(a.work), Path(a.results)
    E = np.load(res / "embeddings.npz", allow_pickle=False)
    if a.out:  # write everything below to another folder; inputs stay in --results
        res = Path(a.out)
        res.mkdir(parents=True, exist_ok=True)
    X = E["emb"].astype(np.float64)
    meta = pd.DataFrame({k: E[k] for k in E.files if k != "emb"})
    # identical crops give zero k-NN distances; keep one copy per dataset
    _, first = np.unique(np.column_stack([pd.factorize(meta.dataset)[0], X]), axis=0, return_index=True)
    n_exact = len(X) - len(first)
    keep = np.zeros(len(X), bool)
    keep[first] = True
    n_all = len(X)
    meta_all = meta.copy()
    if not a.all_copies:  # one image per within-dataset near-copy cluster
        keep &= meta.rep.to_numpy()
    X, meta = X[keep], meta[keep].reset_index(drop=True)
    print(f"{len(X)} of {n_all} embeddings kept ({n_exact} exact duplicates; "
          f"{'all copies' if a.all_copies else 'one image per near-copy cluster'})")

    wanted = ["D1"] + [d for d in (a.sets or [x for x in ALL_SETS if x not in SEPARATE]) if d != "D1"]
    dss = [d for d in ALL_SETS if d in wanted and (meta.dataset == d).any()]
    is_held = (meta.dataset == "D1") & meta.split.isin(["val", "test"])
    held = np.flatnonzero(is_held)
    d1_train = np.flatnonzero((meta.dataset == "D1") & (meta.split == "train"))
    groups = meta.group.to_numpy()
    img_id = pd.factorize(meta.dataset + "|" + meta.image)[0]

    def thin(idx, rng, keys=None):
        """One random crop per original image (sibling crops are strongly correlated)."""
        if a.per_crop:
            return idx
        keys = img_id[idx] if keys is None else keys
        perm = rng.permutation(len(idx))
        _, first = np.unique(keys[perm], return_index=True)
        return idx[perm[first]]

    def size(idx):  # sampling units available: images (default) or crops
        return len(idx) if a.per_crop else len(np.unique(img_id[idx]))

    # PCA on D1-held-out only
    kmax = min(max(a.ks), len(held) - 1)
    ks = [k for k in a.ks if k <= kmax]
    pca = PCA(n_components=kmax, random_state=a.seed).fit(X[held])
    Z = pca.transform(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    pvar = pd.DataFrame({"k": ks, "d1_variance_kept": [cum[k - 1] for k in ks]})
    pvar.to_csv(res / "pca_variance.csv", index=False)
    print("PCA variance kept:", dict(zip(pvar.k, pvar.d1_variance_kept.round(3))))

    # sets per version
    versions = {}
    for ver in ["all", "dedup"]:
        sets = {"D1": held}
        for ds in dss[1:]:
            m = (meta.dataset == ds).to_numpy()
            if source_of(ds) == "D1":  # D1nd: only from D1-held-out images, never deduplicated away
                m = m & meta.split.isin(["val", "test"]).to_numpy()
            elif ver == "dedup":
                m = m & ~meta.in_d1.to_numpy()
            sets[ds] = np.flatnonzero(m)
        versions[ver] = sets
    defect_versions = {v: {d: i for d, i in s.items() if d in DEFECT} for v, s in versions.items()}

    # corrupted embeddings, keyed by original row (rows refer to the un-filtered array)
    Cz = np.load(work / "embeddings_corrupt.npz", allow_pickle=False)
    old_to_new = -np.ones(len(keep), int)
    old_to_new[np.flatnonzero(keep)] = np.arange(keep.sum())
    c_row = old_to_new[Cz["row"]]
    c_ok = c_row >= 0
    CZ = pca.transform(Cz["emb"][c_ok].astype(np.float64))
    c_row, c_kind, c_sev = c_row[c_ok], Cz["corruption"][c_ok], Cz["severity"][c_ok]

    rows, crows, too_few, ns, sizes_tab = [], [], [], {}, []
    halves_n = None
    for ver, sets in versions.items():
        counts = {d: size(i) for d, i in sets.items()}
        ok = [d for d in dss if counts[d] >= a.min_n]
        too_few += [dict(version=ver, dataset=d, units=counts[d]) for d in dss if d not in ok]
        # group-disjoint halves of D1-held-out, one pair per seed; n must fit in every half
        held_halves = [group_halves(groups[held], np.random.default_rng(a.seed + 1000 * sd + 2))
                       for sd in range(a.seeds)]
        halves_n = min(min(size(held[x]), size(held[y])) for x, y in held_halves)
        n = int(min([a.n_cap, halves_n, size(d1_train)] + [counts[d] for d in ok]))
        ns[ver] = n
        sizes_tab.append({"version": ver, **{d: f"{len(i)} / {size(i)}" for d, i in sets.items()}, "common n": n})
        print(f"[{ver}] n = {n}; eligible: {ok}; too few: {[d for d in dss if d not in ok]}")
        for seed in range(a.seeds):
            rng = np.random.default_rng(a.seed + 1000 * seed + 1)
            samp = {d: pick(thin(sets[d], rng), n, rng) for d in ok}
            hA, hB = held[held_halves[seed][0]], held[held_halves[seed][1]]
            a_idx, b_idx = pick(thin(hA, rng), n, rng), pick(thin(hB, rng), n, rng)
            tr_idx = pick(thin(d1_train, rng), n, rng)
            for k in ks:
                def add(P, Q, kind, name_p, name_q, extra=None):
                    for est, (pq, qp, sym) in both_dirs(P, Q).items():
                        rows.append(dict(version=ver, estimator=est, k=k, n=n, kind=kind, P=name_p, Q=name_q,
                                         seed=seed, pq=pq, qp=qp, sym=sym, **(extra or {})))
                for i, p in enumerate(ok):
                    for q in ok[i + 1:]:
                        add(Z[samp[p], :k], Z[samp[q], :k], "pair", p, q)
                add(Z[a_idx, :k], Z[b_idx, :k], "noise_floor", "D1-half-A", "D1-half-B")
                add(Z[tr_idx, :k], Z[samp["D1"], :k], "in_distribution", "D1-train", "D1")
            # corruption sanity check at the main k: corrupted half A vs clean half B
            k = a.k if a.k in ks else ks[-1]
            in_a = np.isin(c_row, hA)
            for kind in pd.unique(c_kind):
                for sev in [1, 2, 3]:
                    cand = np.flatnonzero(in_a & (c_kind == kind) & (c_sev == sev))
                    cand = thin(cand, rng, keys=img_id[c_row[cand]])
                    if len(cand) < n:
                        continue
                    P = CZ[pick(cand, n, rng), :k]
                    for est, (pq, qp, sym) in both_dirs(P, Z[b_idx, :k]).items():
                        crows.append(dict(version=ver, estimator=est, k=k, n=n, kind="corruption",
                                          P=f"D1-half-A+{kind}@{sev}", Q="D1-half-B",
                                          corruption=kind, severity=sev, seed=seed, pq=pq, qp=qp, sym=sym))
    R = summarize(rows)
    C = summarize(crows) if crows else pd.DataFrame()
    out = pd.concat([R, C, pd.DataFrame([dict(kind="too_few", **t) for t in too_few])], ignore_index=True)
    out.to_csv(res / "kl_matrix.csv", index=False)

    # per-class KL vs D1-held-out of the mapped class (both versions, class-k dims)
    d1c = meta.cls.map(class_key)
    prow = []
    kc = min(a.class_k, kmax)
    for ver, sets in versions.items():
        for ds in dss[1:]:
            idx = sets[ds]
            cmap = {class_key(s): t for s, t in CLASS_MAP.get(ds, {}).items()}
            mapped = meta.cls.iloc[idx].map(lambda c: cmap.get(class_key(c)))
            for target in sorted(set(v for v in cmap.values() if v)):
                p_idx = idx[(mapped == target).to_numpy()]
                q_all = held[(d1c.iloc[held] == class_key(target)).to_numpy()]
                if len(q_all) == 0:
                    continue
                halves = [group_halves(groups[q_all], np.random.default_rng(a.seed + 1000 * sd + 7))
                          for sd in range(a.seeds)]
                n_c = min([size(p_idx), a.n_cap] + [min(size(q_all[x]), size(q_all[y])) for x, y in halves])
                if n_c < a.min_n:
                    d1_half = min(min(size(q_all[x]), size(q_all[y])) for x, y in halves)
                    prow.append(dict(version=ver, dataset=ds, d1_class=target, crops=len(p_idx),
                                     d1_crops=len(q_all), note=f"{ds}: {size(p_idx)}, D1 half: {d1_half}"))
                    continue
                for seed in range(a.seeds):
                    rng = np.random.default_rng(a.seed + 1000 * seed + 8)
                    qa, qb = halves[seed]
                    P, Q = Z[pick(thin(p_idx, rng), n_c, rng), :kc], Z[pick(thin(q_all[qb], rng), n_c, rng), :kc]
                    F = Z[pick(thin(q_all[qa], rng), n_c, rng), :kc]
                    for est, f in ESTIMATORS.items():
                        prow.append(dict(version=ver, dataset=ds, d1_class=target, crops=len(p_idx),
                                         d1_crops=len(q_all), n=n_c, k=kc, estimator=est, seed=seed,
                                         sym=(f(P, Q) + f(Q, P)) / 2, floor=(f(F, Q) + f(Q, F)) / 2, note=""))
    PC = pd.DataFrame(prow, columns=["version", "dataset", "d1_class", "crops", "d1_crops", "n", "k",
                                     "estimator", "seed", "sym", "floor", "note"])
    done = PC[PC.note == ""]
    keys = ["version", "dataset", "d1_class", "estimator", "crops", "d1_crops", "n", "k"]
    done = done.groupby(keys).agg(sym_mean=("sym", "mean"), sym_std=("sym", "std"),
                                  class_noise_floor=("floor", "mean")).reset_index()
    PC = pd.concat([done, PC[PC.note != ""][["version", "dataset", "d1_class", "crops", "d1_crops", "note"]]],
                   ignore_index=True)
    PC.to_csv(res / "kl_per_class.csv", index=False)

    # linear separability: how often a logistic regression on the same PCA-k features guesses the
    # source set right (5-fold CV, balanced classes; 50% = indistinguishable)
    sep_rows = []
    srng = np.random.default_rng(a.seed + 99)
    dd = versions["dedup"]

    def sep(P, Q):
        n = min(size(P), size(Q), a.n_cap)
        P, Q = pick(thin(P, srng), n, srng), pick(thin(Q, srng), n, srng)
        Xs, y = np.vstack([Z[P, :a.k], Z[Q, :a.k]]), np.r_[np.zeros(n), np.ones(n)]
        return cross_val_score(LogisticRegression(max_iter=4000), Xs, y, cv=5).mean(), n

    hA, hB = held[held_halves[0][0]], held[held_halves[0][1]]
    sep_rows.append(dict(P="D1-half-A", Q="D1-half-B", **dict(zip(["accuracy", "n"], sep(hA, hB)))))
    for d in [x for x in dss[1:] if size(dd[x]) >= a.min_n]:
        for ref_set in ["D1", "D1nd"]:
            if ref_set in dd and d != ref_set:
                sep_rows.append(dict(P=d, Q=ref_set, **dict(zip(["accuracy", "n"], sep(dd[d], dd[ref_set])))))
    SEP = pd.DataFrame(sep_rows)
    SEP.to_csv(res / "separability.csv", index=False)

    # ---------------------------------------------------------------- figures
    rng = np.random.default_rng(a.seed)
    var2 = pca.explained_variance_ratio_[:2]
    in_d1_other = (meta.in_d1 & (meta.dataset != "D1")).to_numpy()
    scatter_fig(Z[:, :2], defect_versions["all"], in_d1_other,
                "All images: embeddings in D1's PCA plane", res / "pca_scatter_all.png", var2, a.plot_n, rng)
    scatter_fig(Z[:, :2], defect_versions["dedup"], np.zeros(len(Z), bool),
                "Deduplicated (images found in D1 removed from D2-D5)", res / "pca_scatter_dedup.png",
                var2, a.plot_n, rng)

    def circle_samples(sets):
        samples, skipped = {}, []
        for d, idx in sets.items():
            if size(idx) < a.min_n:
                skipped.append(f"{d} ({size(idx)})")
                continue
            t = thin(idx, rng)
            samples[d] = t if len(t) <= a.circle_n else pick(t, a.circle_n, rng)
        return samples, skipped

    for ver in ["dedup", "all"]:
        circles_fig(Z[:, :a.k], *circle_samples(defect_versions[ver]), res / f"overlap_circles_{ver}.png",
                    var2, a.k, ver == "dedup")
    if a.sets:  # a custom comparison: one circles figure with exactly the chosen sets
        circles_fig(Z[:, :a.k], *circle_samples(versions["dedup"]), res / "overlap_circles_selected.png",
                    var2, a.k, True)
    for kind, group in [("whole", NO_DAMAGE), ("patch", [f"{d}p" for d in NO_DAMAGE])]:
        sel = [d for d in ["D1", "D1nd"] + group if d in versions["dedup"]]
        if len(sel) > 2:
            circles_fig(Z[:, :a.k], *circle_samples({d: versions["dedup"][d] for d in sel}),
                        res / f"overlap_circles_nodamage_{kind}.png", var2, a.k, True)

    import umap  # slow import; only needed here
    sets = defect_versions["dedup"]
    sub = np.concatenate([i if len(i) <= a.plot_n else pick(i, a.plot_n, rng) for i in sets.values()])
    U = umap.UMAP(n_neighbors=30, min_dist=0.2, metric="cosine", random_state=a.seed).fit_transform(X[sub])
    lab = meta.dataset.to_numpy()[sub]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for ds in sets:
        m = lab == ds
        ax.scatter(*U[m].T, s=10, marker=MARKERS[ds], color=COLORS[ds], alpha=0.5, lw=0, label=f"{ds} (n={m.sum()})")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("UMAP of pooled 512-d embeddings (deduplicated)", fontsize=11)
    ax.legend(fontsize=9, markerscale=2, frameon=False)
    fig.text(0.01, 0.01, "UMAP keeps local neighbourhoods only: distances between clusters and cluster sizes\n"
             "are NOT meaningful. Use the KL table for distances.", fontsize=8.5, color="#52514e")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(res / "umap.png", dpi=170)
    plt.close(fig)

    def ref_val(ver, est, kind, k=a.k):
        r = R[(R.version == ver) & (R.estimator == est) & (R.kind == kind) & (R.k == k)]
        return float(r.sym_mean.iloc[0]) if len(r) else np.nan

    def sym_matrix(ver, est, k=a.k):
        M, S = np.full((len(dss), len(dss)), np.nan), np.full((len(dss), len(dss)), np.nan)
        sub = R[(R.version == ver) & (R.estimator == est) & (R.k == k) & (R.kind == "pair")]
        for r in sub.itertuples():
            i, j = dss.index(r.P), dss.index(r.Q)
            M[i, j] = M[j, i] = r.sym_mean
            S[i, j] = S[j, i] = r.sym_std
        return M, S

    tabs = []
    for est, nm in [("gaussian", "Gaussian (Ledoit-Wolf)"), ("knn", "k-NN (k=5)")]:
        M, S = sym_matrix("dedup", est)
        tabs.append((f"Symmetric KL, deduplicated - {nm}", M, S))
    cap = (f"PCA k={a.k} fitted on D1-held-out, n={ns['dedup']} {'crops' if a.per_crop else 'images'} per dataset, mean ± sd over {a.seeds} seeds. "
           f"Reference (Gaussian / k-NN): noise floor (two halves of D1-held-out) = "
           f"{ref_val('dedup', 'gaussian', 'noise_floor'):.3f} / {ref_val('dedup', 'knn', 'noise_floor'):.3f}; "
           f"in-distribution (D1-train vs D1-held-out) = {ref_val('dedup', 'gaussian', 'in_distribution'):.3f} / "
           f"{ref_val('dedup', 'knn', 'in_distribution'):.3f}.\nD1 = D1-held-out. n counts {'crops' if a.per_crop else 'images (one random crop each)'}; 'too few' = under {a.min_n} left after dedup.")
    dsel = [i for i, d in enumerate(dss) if d in DEFECT]
    heatmap_fig([(t, M[np.ix_(dsel, dsel)], S[np.ix_(dsel, dsel)]) for t, M, S in tabs],
                [dss[i] for i in dsel], cap, res / "kl_heatmap_dedup.png")
    if len(dsel) < len(dss):
        M0 = tabs[0][1]  # drop sets with no value at all (e.g. D2 after dedup)
        asel = [i for i in range(len(dss)) if np.isfinite(np.delete(M0[i], i)).any()]
        gone = [d for i, d in enumerate(dss) if i not in asel]
        heatmap_fig([(t, M[np.ix_(asel, asel)], S[np.ix_(asel, asel)]) for t, M, S in tabs], [dss[i] for i in asel],
                    cap + " p = skin patches; D1nd = D1 no-damage patches."
                    + (f" Not shown (too few): {', '.join(gone)}." if gone else ""), res / "kl_heatmap_all_sets.png")

    if len(C):
        ref = {e: {kd: ref_val("dedup", e, kd) for kd in ["noise_floor", "in_distribution"]} for e in ESTIMATORS}
        Cd = C[C.version == "dedup"]
        zero = R[(R.version == "dedup") & (R.kind == "noise_floor") & (R.k == Cd.k.iloc[0])]
        z = [dict(estimator=r.estimator, corruption=c, severity=0, kl_pq_mean=r.kl_pq_mean, kl_pq_std=r.kl_pq_std)
             for r in zero.itertuples() for c in pd.unique(Cd.corruption)]
        Cplot = pd.concat([pd.DataFrame(z), Cd], ignore_index=True)
        corruption_fig(Cplot, ref, res / "kl_corruption.png", int(Cd.k.iloc[0]), ns["dedup"])

    write_summary(a, res, work, R, C, PC, pvar, ns, dss, sizes_tab, meta_all, too_few, ref_val, sym_matrix, SEP)
    print("wrote", res / "summary.md")


# ---------------------------------------------------------------- summary.md

def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_summary(a, res, work, R, C, PC, pvar, ns, dss, sizes_tab, meta, too_few, ref_val, sym_matrix, SEP):
    L = ["# Distribution shift between D1-D5 (KL divergence on ResNet-18 embeddings)\n",
         "_Generated by `shift.py`. Numbers are relative: compare them only against the reference "
         "values at the same k and n._\n",
         ("Every dataset keeps all its near-copies in the KL samples.\n" if a.all_copies else
          "**Within-dataset near-copies:** KL samples keep one image (all its crops) per near-copy cluster "
          "(same photo stem, phash ≤ threshold, or same original file name), so copies of one photo "
          "do not pull the sample together. The classifier was trained on all of D1-train.\n")]

    L.append("## 1. Dataset sizes\n")
    st = pd.read_csv(work / "dataset_stats.csv") if (work / "dataset_stats.csv").exists() else None
    cnt = meta.groupby("dataset").agg(crops_in_d1=("in_d1", "sum"), clusters=("group", "nunique"),
                                      crops_1_per_cluster=("rep", "sum")).reset_index()
    if st is not None:
        t = st[["dataset", "images", "crops", "median_w", "median_h", "most_common_size", "median_aspect",
                "square_share"]].merge(cnt, on="dataset", how="left")
        for col in ["crops_in_d1", "clusters", "crops_1_per_cluster"]:
            t[col] = ["-" if pd.isna(v) or (col == "crops_in_d1" and d == "D1") else int(v)
                      for d, v in zip(t.dataset, t[col])]
        t["dataset"] = [d if d in dss else f"{d} (overlap only)" for d in t.dataset]
        L += [md_table(t), ""]
    L.append("Available for the KL analysis, as crops / distinct images (D1 = D1-held-out). "
             + ("Samples draw crops freely." if a.per_crop else
                "Each sample takes at most one random crop per image, because crops of one image are strongly "
                "correlated; n is therefore a number of images.") + "\n")
    L += [md_table(pd.DataFrame(sizes_tab)), ""]
    if too_few:
        L.append(f"**Too few for a KL value (under {a.min_n}):** " + ", ".join(
            f"{t['dataset']} ({t['version']}: {t['units']})" for t in too_few) + "\n")

    L.append("## 2. Image overlap between datasets\n")
    inp = Path(a.results)  # overlap and classifier outputs live in the main results folder
    up = "" if inp.resolve() == res.resolve() else os.path.relpath(inp, res) + "/"
    ov = inp / "overlap_matrix.csv"
    if ov.exists():
        o = pd.read_csv(ov, index_col=0) * 100
        L += ["Share (%) of images in the row dataset that also appear in the column dataset "
              "(phash Hamming ≤ threshold or same original file name). Diagonal = near-copies inside the dataset.\n",
              md_table(o.round(1).reset_index().rename(columns={"index": "row \\ col"}), "{:.1f}"), "",
              f"![overlap]({up}overlap_heatmap.png)\n", f"Pairs sheet for checking the threshold: `{up}duplicate_pairs_sheet.png`.\n"]

    L.append("## 3. D1 classifier (feature extractor)\n")
    rp = inp / "d1_report.json"
    if rp.exists():
        r = json.loads(rp.read_text())
        L += [f"Test accuracy **{r['test_accuracy']:.3f}**, macro-F1 **{r['test_macro_f1']:.3f}**, "
              f"balanced accuracy **{r['test_balanced_accuracy']:.3f}**. "
              f"Train macro-F1 (no augmentation) {r['train_macro_f1_no_aug']:.3f} vs test {r['test_macro_f1']:.3f} "
              f"(gap {r['train_macro_f1_no_aug'] - r['test_macro_f1']:+.3f}).\n",
              md_table(pd.DataFrame({"class": list(r["test_per_class_recall"]),
                                     "test recall": list(r["test_per_class_recall"].values()),
                                     "test crops": list(r["test_support"].values())})), "",
              f"Details: `{up}d1_report.md`, `{up}d1_confusion.png`, `{up}d1_training_curves.png`.\n"]

    L.append(f"## 4. KL divergence (PCA k = {a.k}, fitted on D1-held-out)\n")
    L.append("D1 variance kept by PCA: " + ", ".join(f"k={int(r.k)}: {r.d1_variance_kept * 100:.1f}%"
                                                    for r in pvar.itertuples()) + "\n")
    for ver in ["dedup", "all"]:
        L.append(f"### {'Deduplicated (quote these)' if ver == 'dedup' else 'All images'} — n = {ns[ver]}\n")
        sub = R[(R.version == ver) & (R.k == a.k)].copy()
        tab = []
        for (p, q, kind), g in sub.groupby(["P", "Q", "kind"], sort=False):
            row = dict(P=p, Q=q, kind=kind)
            for e in ESTIMATORS:
                x = g[g.estimator == e].iloc[0]
                row[f"{e} KL(P‖Q)"] = f"{x.kl_pq_mean:.3f}"
                row[f"{e} KL(Q‖P)"] = f"{x.kl_qp_mean:.3f}"
                row[f"{e} sym"] = f"{x.sym_mean:.3f} ± {x.sym_std:.3f}"
            tab.append(row)
        tab = pd.DataFrame(tab)
        tab["o"] = tab.kind.map({"noise_floor": 0, "in_distribution": 1, "pair": 2})
        L += [md_table(tab.sort_values("o", kind="stable").drop(columns="o")), ""]
    L.append("![heatmap](kl_heatmap_dedup.png)\n")

    L.append("### Sensitivity to k (deduplicated, symmetric KL vs D1, Gaussian / k-NN)\n")
    sk = []
    for k in sorted(R.k.unique()):
        row = {"k": int(k), "noise floor": f"{ref_val('dedup', 'gaussian', 'noise_floor', k):.3f} / "
                                           f"{ref_val('dedup', 'knn', 'noise_floor', k):.3f}",
               "D1-train": f"{ref_val('dedup', 'gaussian', 'in_distribution', k):.3f} / "
                           f"{ref_val('dedup', 'knn', 'in_distribution', k):.3f}"}
        for d in [x for x in dss[1:] if x in DEFECT]:
            vals = []
            for e in ESTIMATORS:
                M, _ = sym_matrix("dedup", e, k)
                vals.append(M[0, dss.index(d)])
            row[d] = " / ".join("—" if np.isnan(v) else f"{v:.3f}" for v in vals)
        sk.append(row)
    L += [md_table(pd.DataFrame(sk)), ""]

    L.append("### Known-shift sanity check\n")
    if len(C):
        Cd = C[C.version == "dedup"]
        ok = []
        for (e, kind), g in Cd.groupby(["estimator", "corruption"]):
            v = g.sort_values("severity").kl_pq_mean.to_numpy()
            ok.append(dict(estimator=e, corruption=kind,
                           **{f"sev {s}": f"{x:.3f}" for s, x in zip(g.sort_values("severity").severity, v)},
                           noise_floor=f"{ref_val('dedup', e, 'noise_floor', int(Cd.k.iloc[0])):.3f}",
                           rises=("yes" if np.all(np.diff(v) > 0) and v[0] > ref_val('dedup', e, 'noise_floor', int(Cd.k.iloc[0])) else "**NO**")))
        L += [md_table(pd.DataFrame(ok)), "", "![corruption](kl_corruption.png)\n"]
        if all(o["rises"] == "yes" for o in ok):
            L.append("KL rises with severity for every corruption and both estimators, so the setup responds to known shift.\n")
        else:
            L.append("**Warning: KL does not rise monotonically for every corruption - check the setup before trusting the table.**\n")

    pc = PC[PC.sym_mean.notna()]
    if len(pc):
        L.append(f"### Per-class KL vs the mapped D1 class (k = {a.class_k}, symmetric, mean over seeds)\n")
        L += [md_table(pc[["version", "dataset", "d1_class", "estimator", "n", "sym_mean", "class_noise_floor"]]), ""]
        few = PC[PC.note.fillna("") != ""]
        if len(few):
            L.append(f"No per-class value (fewer than {a.min_n} {'crops' if a.per_crop else 'images'} on one side; "
                     "the D1 side is a group-split half of the D1-held-out class): " + ", ".join(
                f"{r.dataset}→{r.d1_class} ({r.version}; {r.note})" for r in few.itertuples()) + "\n")
        L.append("`class_noise_floor` = two halves of the D1-held-out class at the same n. "
                 "D4 is wind-turbine blades (different material); D5 Fastener Damage → Missing-Head is a loose match.\n")

    nd = [d for d in ["D1nd"] + NO_DAMAGE + [f"{x}p" for x in NO_DAMAGE] if d in dss]
    if nd:
        Mg, Sg = sym_matrix("dedup", "gaussian")
        Mk, Sk = sym_matrix("dedup", "knn")
        i1 = dss.index("D1")
        j_nd = dss.index("D1nd") if "D1nd" in dss else None
        L.append(f"### No-damage datasets (deduplicated, k = {a.k}, n = {ns['dedup']}, symmetric KL, Gaussian / k-NN)\n")
        L.append("D6-D9 are whole-aircraft photos; `p` = skin patches cut from the middle of each aircraft box; "
                 "D1nd = patches of D1-held-out photos away from every defect box (same cameras as D1, no labelled damage). "
                 "If a no-damage set is far from D1nd, using it as the 'no damage' class would let a model separate "
                 "classes by where the photo came from.\n")
        fmt = lambda M, S, i, j: "—" if j is None or np.isnan(M[i, j]) else f"{M[i, j]:.2f}"
        rows = [{"set": "noise floor (two halves of D1-held-out)",
                 "vs D1 (defects)": f"{ref_val('dedup', 'gaussian', 'noise_floor'):.2f} / {ref_val('dedup', 'knn', 'noise_floor'):.2f}",
                 "vs D1nd (D1 no-damage)": ""}]
        for d in nd:
            j = dss.index(d)
            rows.append({"set": f"{d} ({NAMES.get(d, 'whole aircraft')})",
                         "vs D1 (defects)": f"{fmt(Mg, Sg, i1, j)} / {fmt(Mk, Sk, i1, j)}",
                         "vs D1nd (D1 no-damage)": "—" if d == "D1nd" else f"{fmt(Mg, Sg, j_nd, j)} / {fmt(Mk, Sk, j_nd, j)}"})
        L += [md_table(pd.DataFrame(rows)), ""]
        acc = {(r.P, r.Q): r.accuracy for r in SEP.itertuples()}
        floor = acc.get(("D1-half-A", "D1-half-B"), np.nan)
        L.append(f"Linear separability (logistic regression on the same {a.k} features, 5-fold CV, balanced; "
                 f"50% = cannot tell apart). Two halves of D1-held-out: {floor:.0%}.\n")
        L += [md_table(pd.DataFrame([{"set": d, "vs D1 (defects)": f"{acc.get((d, 'D1'), np.nan):.0%}",
                                      "vs D1nd (D1 no-damage)": "—" if d == "D1nd" else f"{acc.get((d, 'D1nd'), np.nan):.0%}"}
                                     for d in nd])), "",
              "![no-damage whole](overlap_circles_nodamage_whole.png)\n",
              "![no-damage patches](overlap_circles_nodamage_patch.png)\n",
              "Full pairwise table for every set: `kl_heatmap_all_sets.png` and `kl_matrix.csv`.\n"]

    if a.sets:
        L.append(f"![selected sets](overlap_circles_selected.png)\n\n**Dashed circles for the compared sets** "
                 f"({', '.join(dss)}), deduplicated, one crop per image.\n")
    L.append("## 5. Figures\n")
    L += ["![PCA all](pca_scatter_all.png)\n",
          "**Figure 1a.** D1-held-out and D2–D5 in the first two principal components of D1-held-out, "
          "with 2-σ ellipses. Hollow markers are crops whose original image also appears in D1.\n",
          "![PCA dedup](pca_scatter_dedup.png)\n",
          "**Figure 1b.** Same, with images found in D1 removed from D2–D5.\n",
          "![overlap circles dedup](overlap_circles_dedup.png)\n",
          "**Figure 1c.** Dashed 2-σ circles per dataset (deduplicated, one crop per image), in D1's PCA plane (left) "
          "and in an LDA plane that best separates the datasets (right). `overlap_circles_all.png` is the same with all images.\n",
          "![UMAP](umap.png)\n",
          "**Figure 2.** UMAP of the pooled 512-d embeddings (deduplicated). Distances between clusters "
          "are not meaningful in UMAP; only local neighbourhoods are.\n",
          "**Figure 3.** `kl_heatmap_dedup.png` (above). **Figure 4.** `kl_corruption.png` (above).\n"]

    fp = Path(a.findings)
    if fp.exists():  # hand-written interpretation, kept outside the generated file so re-runs keep it
        L += [fp.read_text().strip(), ""]
    L.append("## 6. What the numbers say (auto-generated)\n")
    nf = ref_val("dedup", "gaussian", "noise_floor")
    ind = ref_val("dedup", "gaussian", "in_distribution")
    Mg, _ = sym_matrix("dedup", "gaussian")
    Mk, _ = sym_matrix("dedup", "knn")
    d1row = {d: (Mg[0, dss.index(d)], Mk[0, dss.index(d)]) for d in dss[1:]
             if d in DEFECT and not np.isnan(Mg[0, dss.index(d)])}
    if d1row:
        order = sorted(d1row, key=lambda d: d1row[d][0])
        rank_k = sorted(d1row, key=lambda d: d1row[d][1])
        L.append(f"- Closest to D1 (deduplicated, Gaussian symmetric KL): **{order[0]}** "
                 f"({d1row[order[0]][0]:.2f}, {d1row[order[0]][0] / nf:.0f}× the noise floor of {nf:.3f}). "
                 f"Farthest: **{order[-1]}** ({d1row[order[-1]][0]:.2f}, {d1row[order[-1]][0] / nf:.0f}×).")
        L.append(f"- Ranking by Gaussian KL: {' < '.join(order)}; by k-NN KL: {' < '.join(rank_k)} "
                 f"({'same order' if order == rank_k else 'the estimators disagree on the order'}).")
        L.append(f"- For scale: D1-train vs D1-held-out = {ind:.3f}. Every dataset is "
                 f"{'far above' if min(v[0] for v in d1row.values()) > 3 * max(ind, nf) else 'comparable to'} "
                 "this in-distribution reference.\n")
    L.append("_These sentences are generated from the table; read them together with the figures._\n")

    L.append("## 7. Caveats\n")
    L += ["- **Viewpoint.** KL is measured through a network trained on D1, so it measures shift *as this model sees it*.",
          "- **Duplicates.** Quote the deduplicated numbers. The all-images numbers mainly show how much the datasets share. "
          "Deduplication only removes images found in D1; D2–D5 can still share images with each other (see the overlap table), "
          "which lowers their mutual KL.",
          "- **Gaussian assumption.** Gaussian KL treats each dataset as one blob, which is why the k-NN estimator is reported too.",
          "- **Relative numbers only.** KL values depend on k and n. Compare them at fixed k and n against the reference values, "
          "never as absolute numbers.",
          ("- **Correlated crops.** Samples draw crops freely, so sibling crops of one image sit close together and push "
           "the k-NN estimates (and the noise floor) up." if a.per_crop else
           "- **One crop per image.** Crops of one image are strongly correlated (with free crop sampling the k-NN noise "
           "floor was about 100× higher), so each sample takes at most one random crop per image. "
           "This limits n to the smallest dataset's number of distinct images."),
          "- **Hash limits.** Hash matching misses heavily cropped or edited copies, so \"deduplicated\" means \"no near-identical copies found\". "
          "It also over-matches low-detail images: plain grey or white blade tiles in D4, and photos of a single plane on plain sky "
          "(D6 and D8 match each other and themselves heavily, mostly falsely). So the overlap table overstates D6-D9's internal "
          "and D6-D8 overlap, and inside D6-D9 copies are grouped by file name only.",
          "- **No-damage sets.** D6-D9 were measured through a network trained only on D1 defects, which never saw a "
          "'no damage' example. D1nd patches are random areas away from labelled defects, so they include background "
          "(ground, sky, shadow) and possibly unlabelled damage. D6 comes from a Kaggle copy of the FGVC-Aircraft 2013b "
          "archive (complete: 10,000 images and boxes) because the official server was too slow; its 20 px copyright banner "
          "was cut off. D9 uses only Open Images' validation and test images. Licences: FGVC is for non-commercial research, "
          "D7/D8 are CC BY 4.0, Open Images images are CC BY 2.0.",
          "- **D10 (fuselage mock-up).** Video frames, so near-copies were found without chaining (a frame is kept only if it "
          "differs from every kept frame). Each frame gives two 720×720 squares, a wider view than D1's defect crops. Its anomalies "
          "are placed objects, and normal and anomaly frames were filmed separately, so part of their difference is the recording "
          "session. Analysed in its own run (`results/d10/`). Licence: CC BY 4.0.",
          "- **k-NN values near zero.** The k-NN estimator is unbiased only on average, so values for identical distributions "
          "scatter around 0 and can be slightly negative. Read them as \"no measurable shift\".",
          "- **Feature extractor.** The D1 classifier reaches test macro-F1 of about 0.8 and is weakest on scratch, corrosion and "
          "paint-peel-off. Its features are good enough to rank datasets, not to judge fine differences within those classes.",
          "- **Dataset versions.** D2 = Roboflow v22 (raw), D4 = v2, D5 = v3 (no augmentation listed, yet it holds many same-photo "
          "copies, which the near-copy clustering collapses). D3 was dropped because it exists only as a 3× augmented export "
          "and 98% of it is D2; it appears in the overlap table only.",
          "- **Licensing.** D1's license is unknown and its sources carry their own licenses. That is fine for this analysis, "
          "but worth noting before D1 is used in anything published.\n"]
    (res / "summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
