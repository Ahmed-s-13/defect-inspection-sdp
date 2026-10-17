"""Step 4: did the healing work? All numbers on D1-test unless stated (selection on D1-val).

  1. Teacher-student score: original vs healed, healed vs real clean (AUROC).
  2. Visual grids: original | hole | healed | TS map before | TS map after
     (10 random per class + the 10 healed crops with the highest TS score).
  3. D1 ResNet-18 embeddings: KL(healed||clean) vs KL(defective||clean) (Gaussian + k-NN,
     PCA k=32 fitted on D1-held-out as in the shift task), PCA scatter, and the drop in
     the D1 classifier's confidence in the original defect class.
  4. Artifact probe: healed vs real clean, and clean-with-random-region-inpainted vs real
     clean. Fine-tuned stock ResNet-18 (same recipe as Stage 1) + linear probe on frozen
     D1 embeddings. AUROC near 1 = detectable inpainting fingerprint.

Outputs: results/healing/step4_checks.md/.json, step4_*.png
"""
import argparse
import json

import hcommon as H

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms as T

import bincls
import shift  # shift task: gauss_kl, knn_kl, group_halves, ellipse
import ts
from klshift.common import CropDataset, build_model, embed_batch

# n is capped below the size of a clean half (~940 images on D1-test) so it is equal across seeds.
K_PCA, N_CAP, SEEDS = 32, 900, range(5)


def load_sets():
    """D1 rows for the four groups, each with path, image, group_id, split (+cls, box)."""
    gen = pd.read_csv(H.WORK / "generated.csv")
    gen["gpath"] = [str(H.HERE / p) for p in gen.path]
    d = H.d1_defects()
    heal = gen[(gen.kind == "healed") & (gen.dataset == "D1")][["src_crop_path", "gpath"]]
    healed = d.merge(heal, left_on="crop_path", right_on="src_crop_path")
    healed["orig_path"], healed["path"] = healed.path, healed.gpath
    c = H.d1_clean()
    cc = gen[gen.kind == "ctrl_clean"][["src_crop_path", "gpath", "hx1", "hy1", "hx2", "hy2"]]
    ctrl = c.merge(cc, left_on="crop_path", right_on="src_crop_path")
    ctrl["orig_path"], ctrl["path"] = ctrl.path, ctrl.gpath
    return {"defective": d, "healed": healed, "clean": c, "clean_inpainted": ctrl}


class D1Net:
    def __init__(self, device):
        ck = torch.load(H.KL / "data/work/model_best.pt", map_location=device)
        self.classes = ck["classes"]
        self.m = build_model(len(self.classes)).to(device).eval()
        self.m.load_state_dict(ck["state"])
        self.device = device

    @torch.no_grad()
    def run(self, paths, batch=128, workers=4):
        E, P = [], []
        for x, _ in DataLoader(CropDataset(list(paths)), batch_size=batch, num_workers=workers):
            e = embed_batch(self.m, x.to(self.device))
            E.append(e.cpu().numpy())
            P.append(torch.softmax(self.m.fc(e), 1).cpu().numpy())
        return np.concatenate(E), np.concatenate(P)


def one_per_image(df, rng):
    return df.groupby("image", sort=False).sample(1, random_state=int(rng.integers(1 << 31))).index


# ------------------------------------------------------------------ 1+2: teacher-student
def ts_checks(sets, device, a, R):
    model = ts.load_ts(ts.TS_DIR / "msrd_resnet18_best.pt", device)
    sc = {}
    for k, df in sets.items():
        t = df[df.split == "test"]
        sc[k] = pd.Series(ts.score(model, ts.PatchDS(t.path), device, workers=a.workers), index=t.index)
    auc = lambda p, n: roc_auc_score(np.r_[np.ones(len(p)), np.zeros(len(n))], np.r_[p, n])
    filt = pd.read_csv(H.WORK / "clean_D1_scores.csv").set_index("crop_path").dropped
    keep = ~sets["clean"].loc[sc["clean"].index, "crop_path"].map(filt).values
    R["ts"] = {
        "mean_score": {k: float(v.mean()) for k, v in sc.items()},
        "AUROC_original_vs_healed": auc(sc["defective"], sc["healed"]),
        "paired_share_score_lower_after_healing": float((sc["healed"].values < pd.Series(
            sc["defective"].values, index=sets["defective"].loc[sc["defective"].index, "crop_path"])[
            sets["healed"].loc[sc["healed"].index, "crop_path"]].values).mean()),
        "AUROC_healed_vs_clean": auc(sc["healed"], sc["clean"]),
        "AUROC_defective_vs_clean": auc(sc["defective"], sc["clean"]),
        "AUROC_clean_inpainted_vs_clean": auc(sc["clean_inpainted"], sc["clean"]),
        "AUROC_healed_vs_clean_filtered (sensitivity)": auc(sc["healed"], sc["clean"][keep]),
    }
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bins = np.linspace(0, np.percentile(np.concatenate(list(sc.values())), 99.5), 50)
    for k in ["clean", "defective", "healed", "clean_inpainted"]:
        ax.hist(sc[k], bins, histtype="step", lw=1.8, density=True, label=k.replace("_", " "))
    ax.set(xlabel="teacher-student score (max of anomaly map)", ylabel="density", title="D1-test")
    ax.legend()
    fig.tight_layout()
    fig.savefig(H.RES / "step4_ts_scores.png", dpi=150)
    plt.close(fig)

    h = sets["healed"].loc[sc["healed"].index].assign(score=sc["healed"].values)
    rng = np.random.default_rng(a.seed)
    for cls, g in h.groupby("cls"):
        visual(model, device, g.sample(min(10, len(g)), random_state=a.seed),
               H.RES / f"step4_grid_{cls}.png", f"D1-test {cls}: 10 random")
    visual(model, device, h.nlargest(10, "score"), H.RES / "step4_grid_worst10.png",
           "D1-test: the 10 healed crops with the highest teacher-student score")


def visual(model, device, rows, path, title):
    tf = T.ToTensor()
    orig = torch.stack([tf(Image.open(p).convert("RGB")) for p in rows.orig_path])
    heal = torch.stack([tf(Image.open(p).convert("RGB")) for p in rows.path])
    mb, ma = ts.amaps(model, orig.to(device)), ts.amaps(model, heal.to(device))
    vmax = max(np.percentile(mb, 99.5), np.percentile(ma, 99.5))
    from healer import hole_for_box, rect_mask
    n = len(rows)
    fig, ax = plt.subplots(n, 5, figsize=(9.5, 1.95 * n + 0.6))
    ax = np.atleast_2d(ax)
    for i, r in enumerate(rows.itertuples()):
        hole = rect_mask(hole_for_box((r.bx1, r.by1, r.bx2, r.by2)))
        cols = [orig[i].permute(1, 2, 0), hole, heal[i].permute(1, 2, 0), mb[i], ma[i]]
        for j, x in enumerate(cols):
            kw = dict(cmap="jet", vmin=0, vmax=vmax) if j >= 3 else dict(cmap="gray") if j == 1 else {}
            ax[i, j].imshow(x, **kw)
            ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        ax[i, 0].add_patch(plt.Rectangle((r.bx1, r.by1), r.bx2 - r.bx1, r.by2 - r.by1, fill=False, ec="lime", lw=1))
        ax[i, 3].set_xlabel(f"{mb[i].max():.3f}", fontsize=7, labelpad=1)
        ax[i, 4].set_xlabel(f"{ma[i].max():.3f}", fontsize=7, labelpad=1)
        ax[i, 0].set_ylabel(r.cls, fontsize=7)
    for j, t in enumerate(["original (box)", "hole", "healed", "TS map before", "TS map after"]):
        ax[0, j].set_title(t, fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ 3: embeddings
def embedding_checks(sets, net, a, R):
    e = np.load(H.KL / "results/embeddings.npz", allow_pickle=True)
    held = (e["dataset"] == "D1") & (e["split"] != "train")
    pca = PCA(K_PCA, random_state=a.seed).fit(e["emb"][held])
    T_ = {k: v[v.split == "test"] for k, v in sets.items()}
    Z, P = {}, {}
    for k, df in T_.items():
        emb, prob = net.run(df.path, workers=a.workers)
        Z[k], P[k] = pd.DataFrame(pca.transform(emb), index=df.index), prob

    # KL, one random crop per image, equal n, 5 seeds; noise floor = two group-halves of clean.
    recs = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = {k: one_per_image(T_[k], rng) for k in T_}
        ca, cb = shift.group_halves(T_["clean"].loc[idx["clean"], "group_id"].values, rng)
        n = min(N_CAP, len(ca), len(cb), *(len(v) for v in idx.values()))
        Q = Z["clean"].loc[idx["clean"]].values
        pairs = {"noise floor (clean half vs clean half)": (Q[rng.choice(ca, n, False)], Q[rng.choice(cb, n, False)])}
        for k in ["defective", "healed", "clean_inpainted"]:
            Pk = Z[k].loc[rng.choice(idx[k], n, False)].values
            pairs[f"{k} || clean"] = (Pk, Q[rng.choice(len(Q), n, False)])
        for name, (Pm, Qm) in pairs.items():
            for est, (pq, qp, sym) in shift.both_dirs(Pm, Qm).items():
                recs.append(dict(pair=name, estimator=est, n=n, seed=seed, pq=pq, qp=qp, sym=sym))
    kl = shift.summarize(recs)
    kl.to_csv(H.RES / "step4_kl.csv", index=False)
    R["kl"] = kl.to_dict("records")

    # PCA scatter: defective, healed, real clean.
    fig, ax = plt.subplots(figsize=(7, 6))
    rng = np.random.default_rng(a.seed)
    for k, col in [("clean", "#1baf7a"), ("defective", "#e34948"), ("healed", "#2a78d6")]:
        z = Z[k].values
        s = z[rng.choice(len(z), min(1500, len(z)), False)]
        ax.scatter(s[:, 0], s[:, 1], s=4, alpha=.35, color=col, label=f"{k} (n={len(z)})")
        shift.ellipse(ax, z[:, :2], col)
    ax.set(xlabel="PC1", ylabel="PC2",
           title="D1-test in D1's PCA space (D1 ResNet-18 embeddings), 2-sigma ellipses")
    ax.legend(markerscale=3)
    fig.tight_layout()
    fig.savefig(H.RES / "step4_pca_scatter.png", dpi=150)
    plt.close(fig)

    # D1 classifier confidence in the original class, original vs healed (paired).
    d, h = T_["defective"], T_["healed"]
    ci = {c: i for i, c in enumerate(net.classes)}
    pos = pd.Series(np.arange(len(d)), index=d.crop_path)
    po = P["defective"][pos[h.crop_path].values, h.cls.map(ci).values]
    ph = P["healed"][np.arange(len(h)), h.cls.map(ci).values]
    pred_o = P["defective"][pos[h.crop_path].values].argmax(1)
    pred_h = P["healed"].argmax(1)
    y = h.cls.map(ci).values
    conf = {"all": dict(mean_p_orig=float(po.mean()), mean_p_healed=float(ph.mean()),
                        mean_drop=float((po - ph).mean()),
                        correct_before=float((pred_o == y).mean()), correct_after=float((pred_h == y).mean()))}
    for c in net.classes:
        m = (h.cls == c).values
        conf[c] = dict(mean_p_orig=float(po[m].mean()), mean_p_healed=float(ph[m].mean()),
                       mean_drop=float((po[m] - ph[m]).mean()),
                       correct_before=float((pred_o[m] == y[m]).mean()), correct_after=float((pred_h[m] == y[m]).mean()))
    R["d1_confidence"] = conf


# ------------------------------------------------------------------ 4: artifact probe
def probe_sets(sets, a):
    """(a) healed vs real clean. (b) clean-with-random-region-inpainted vs real clean, using
    disjoint image-group halves so no clean patch appears in both classes."""
    out = {}
    h, c, ci = sets["healed"], sets["clean"], sets["clean_inpainted"]
    out["healed_vs_clean"] = {s: (list(c[c.split == s].path) + list(h[h.split == s].path),
                                  [0] * (c.split == s).sum() + [1] * (h.split == s).sum()) for s in ["train", "val", "test"]}
    half = H.group_half(c, a.seed)
    out["clean_inpainted_vs_clean"] = {}
    for s in ["train", "val", "test"]:
        real = c[(c.split == s) & ~c.group_id.isin(half)]
        inp = ci[(ci.split == s) & ci.group_id.isin(half)]
        out["clean_inpainted_vs_clean"][s] = (list(real.path) + list(inp.path), [0] * len(real) + [1] * len(inp))
    return out


def artifact_probe(sets, net, device, a, R):
    R["probe"] = {}
    cache = {}
    for name, S in probe_sets(sets, a).items():
        r = {"n": {s: [int(np.sum(np.array(S[s][1]) == v)) for v in (0, 1)] for s in S}}
        # Linear probe on frozen D1-ResNet embeddings (C picked on val).
        E = {}
        for s in S:
            miss = [p for p in S[s][0] if p not in cache]
            if miss:
                cache.update(zip(miss, net.run(miss, workers=a.workers)[0]))
            E[s] = np.stack([cache[p] for p in S[s][0]])
        best = max(((roc_auc_score(S["val"][1], LogisticRegression(C=C, max_iter=3000).fit(E["train"], S["train"][1])
                                   .decision_function(E["val"])), C) for C in (0.01, 0.1, 1.0)))
        lr = LogisticRegression(C=best[1], max_iter=3000).fit(E["train"], S["train"][1])
        r["linear_probe_test_AUROC"] = roc_auc_score(S["test"][1], lr.decision_function(E["test"]))
        r["linear_probe_C"] = best[1]
        # Fine-tuned stock ResNet-18, the same recipe Stage 1 uses.
        print(f"[probe {name}] fine-tuning ResNet-18", flush=True)
        m, vauc, hist = bincls.train_binary(*S["train"], *S["val"], device, epochs=a.probe_epochs,
                                            workers=a.workers, seed=a.seed)
        r["finetuned_val_AUROC"] = vauc
        r["finetuned_test_AUROC"] = roc_auc_score(S["test"][1], bincls.predict(m, S["test"][0], device, workers=a.workers))
        r["finetuned_epochs"] = len(hist)
        R["probe"][name] = r
        print(f"[probe {name}] {r}", flush=True)


def write_md(R):
    t, p, c = R["ts"], R["probe"], R["d1_confidence"]
    kl = pd.DataFrame(R["kl"])
    L = ["# Step 4: did the healing work?\n",
         "All numbers on D1-test (4,293 defect crops, their healed versions, 4,120 real clean patches, "
         "unfiltered). Selection, where any, on D1-val.\n",
         "## Artifact probe (the critical check)\n",
         "| probe task | fine-tuned ResNet-18 test AUROC | linear probe on D1 embeddings, test AUROC |",
         "|---|---|---|"]
    for k, lab in [("healed_vs_clean", "healed vs real clean"),
                   ("clean_inpainted_vs_clean", "clean with random region inpainted vs real clean")]:
        L.append(f"| {lab} | **{p[k]['finetuned_test_AUROC']:.3f}** | {p[k]['linear_probe_test_AUROC']:.3f} |")
    L += ["\nAUROC near 1 means the inpainting leaves a fingerprint a Stage 1 network can learn; "
          "0.5 means it cannot tell.\n",
          "## Teacher-student score\n", "| comparison | AUROC | target |", "|---|---|---|",
          f"| original vs healed | {t['AUROC_original_vs_healed']:.3f} | high |",
          f"| healed vs real clean | {t['AUROC_healed_vs_clean']:.3f} | about 0.5 |",
          f"| (reference) defective vs real clean | {t['AUROC_defective_vs_clean']:.3f} | step-2 baseline |",
          f"| (reference) clean with random region inpainted vs real clean | {t['AUROC_clean_inpainted_vs_clean']:.3f} | about 0.5 |",
          f"| (sensitivity) healed vs filtered real clean | {t['AUROC_healed_vs_clean_filtered (sensitivity)']:.3f} | |",
          f"\nShare of crops whose score went down after healing: {t['paired_share_score_lower_after_healing']:.1%}. "
          "Mean score: " + ", ".join(f"{k} {v:.3f}" for k, v in t["mean_score"].items()) + ".\n",
          "## KL divergence in D1's PCA space (k = 32, one crop per image, mean ± std over 5 seeds)\n",
          "| pair | estimator | n | KL(P‖Q) | KL(Q‖P) | symmetric |", "|---|---|---|---|---|---|"]
    for r in kl.itertuples():
        L.append(f"| {r.pair} | {r.estimator} | {r.n} | {r.kl_pq_mean:.2f} ± {r.kl_pq_std:.2f} | "
                 f"{r.kl_qp_mean:.2f} ± {r.kl_qp_std:.2f} | {r.sym_mean:.2f} ± {r.sym_std:.2f} |")
    L += ["\n## D1 classifier on healed crops (paired with their originals)\n",
          "| class | P(original class) before | after | correct before | correct after |", "|---|---|---|---|---|"]
    for k, v in c.items():
        L.append(f"| {k} | {v['mean_p_orig']:.3f} | {v['mean_p_healed']:.3f} | {v['correct_before']:.3f} | {v['correct_after']:.3f} |")
    L += ["\n![ts](step4_ts_scores.png)\n", "![pca](step4_pca_scatter.png)\n",
          "Visual grids: `step4_grid_<class>.png` (10 random each) and `step4_grid_worst10.png`.\n"]
    (H.RES / "step4_checks.md").write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe-epochs", type=int, default=6)
    ap.add_argument("--skip-probe", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    H.seed_everything(a.seed)
    device = H.pick_device()
    sets = load_sets()
    net = D1Net(device)
    R = {}
    ts_checks(sets, device, a, R)
    embedding_checks(sets, net, a, R)
    if not a.skip_probe:
        artifact_probe(sets, net, device, a, R)
    (H.RES / "step4_checks.json").write_text(json.dumps(R, indent=2, default=float))
    if not a.skip_probe:
        write_md(R)


if __name__ == "__main__":
    main()
