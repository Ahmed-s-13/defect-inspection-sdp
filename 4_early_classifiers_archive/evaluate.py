"""Evaluate one trained model against the four problems, not just test accuracy.

    python evaluate.py --run model_1_plain_cnn/results/aircraft --data data/aircraft

Writes report.md, report.json and five figures into the run folder:
    1 imbalance      macro-F1 / balanced accuracy / per-class recall (accuracy alone hides minority classes)
    2 overfitting    train vs val vs test gap, training curves
    3 shift          8 camera/lighting corruptions x 3 severities; per-source scores;
                     a held-out source ("ood" split) and a second dataset if you give one
    4 shortcuts      hide the defect / hide the context / where Grad-CAM looks
    5 PCA            how many dimensions the features really use, and what they cluster by
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#3987e5", "#0d366b"])
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
                     "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.titlecolor": INK, "axes.grid": True, "grid.color": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "figure.dpi": 130})


def run(model, df, data_dir, size, device, pre=None, emb=False, workers=2):
    ds = C.CropDataset(df, data_dir, C.eval_transform(size), pre=pre)
    return C.predict(model, C.make_loader(ds, 128, workers=workers), device, return_embeddings=emb)


def cluster_bootstrap(y, probs, groups, fn, n=1000, seed=0):
    """95% interval that resamples whole photos, because several crops from one
    photo are not independent test cases."""
    rng = np.random.default_rng(seed)
    idx_by_group = [np.nonzero(groups == g)[0] for g in np.unique(groups)]
    vals = []
    for _ in range(n):
        idx = np.concatenate([idx_by_group[i] for i in rng.integers(0, len(idx_by_group), len(idx_by_group))])
        vals.append(fn(y[idx], probs[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def defect_bal_acc(y, probs):
    """Mean recall over the classes present (the defect classes)."""
    pred = probs.argmax(1)
    return float(np.mean([(pred[y == c] == c).mean() for c in np.unique(y)]))


def gradcam(model, x, target):
    """Grad-CAM (Selvaraju et al., ICCV 2017) on the last conv feature map."""
    fmap = model.body(x)
    fmap.retain_grad()
    logits = model.fc(model.pool(fmap).flatten(1))
    logits.gather(1, target[:, None]).sum().backward()
    w = fmap.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * fmap).sum(1, keepdim=True))
    cam = F.interpolate(cam, x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return (cam / cam.flatten(1).amax(1).clamp(min=1e-8)[:, None, None]).detach()


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def f3(v):
    return f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="folder containing model.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--shift-data", help="a second prepared dataset to test on without training on it")
    ap.add_argument("--class-map", default="", help='for --shift-data: "their class=our class,..." (unmapped classes are dropped)')
    ap.add_argument("--cam-samples", type=int, default=300)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    device = C.get_device()
    out = Path(a.run)
    model, ck = C.load_checkpoint(out / "model.pt", device)
    classes, size = ck["classes"], ck["img_size"]
    df, data_classes = C.load_metadata(a.data)
    assert data_classes == classes, "model and dataset have different classes"
    K = len(classes)
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    R = {"method": ck["method"], "data": a.data, "classes": classes}
    md = [f"# Evaluation: {ck['method']} on {Path(a.data).name}", ""]

    # ------------------------------------------------------------ 1 + 2: clean scores, imbalance, overfitting
    p_tr, y_tr, e_tr = run(model, tr, a.data, size, device, emb=True, workers=a.workers)
    p_va, y_va = run(model, va, a.data, size, device, workers=a.workers)
    p_te, y_te, e_te = run(model, te, a.data, size, device, emb=True, workers=a.workers)
    R["train"], R["val"], R["test"] = (C.compute_metrics(y, p, classes) for y, p in ((y_tr, p_tr), (y_va, p_va), (y_te, p_te)))
    g_te = te["group"].to_numpy()
    R["test"]["macro_f1_ci95"] = cluster_bootstrap(y_te, p_te, g_te, C.macro_f1)
    R["test"]["balanced_accuracy_ci95"] = cluster_bootstrap(y_te, p_te, g_te, defect_bal_acc)
    R["overfit_gap_macro_f1"] = R["train"]["macro_f1"] - R["test"]["macro_f1"]
    majority = float(np.bincount(y_te, minlength=K).max() / len(y_te))
    R["test"]["majority_class_accuracy"] = majority
    t = R["test"]
    md += ["## 1. Test scores (imbalance-aware)", "",
           f"{t['n']} test crops from {len(np.unique(g_te))} photos that the model never saw in any form.", "",
           md_table(["metric", "value", "95% interval"], [
               ["macro-F1", f3(t["macro_f1"]), f"{t['macro_f1_ci95'][0]:.3f} to {t['macro_f1_ci95'][1]:.3f}"],
               ["balanced accuracy", f3(t["balanced_accuracy"]), f"{t['balanced_accuracy_ci95'][0]:.3f} to {t['balanced_accuracy_ci95'][1]:.3f}"],
               ["accuracy", f3(t["accuracy"]), f"always guessing the largest class scores {majority:.3f}"],
               ["calibration error (ECE)", f3(t["ece"]), "0 = the stated confidence is trustworthy"]]), "",
           md_table(["class", "precision", "recall", "F1", "test crops"],
                    [[c, f3(v["precision"]), f3(v["recall"]), f3(v["f1"]), v["support"]] for c, v in t["per_class"].items()]), "",
           "![confusion](confusion.png)", "",
           "## 2. Overfitting", "",
           md_table(["split", "macro-F1", "balanced accuracy"],
                    [[s, f3(R[s]["macro_f1"]), f3(R[s]["balanced_accuracy"])] for s in ("train", "val", "test")]), "",
           f"Train minus test macro-F1: **{R['overfit_gap_macro_f1']:+.3f}**. "
           "A large gap means the model memorised training crops. Val is not a clean number: it chose the checkpoint.", "",
           "![curves](curves.png)", ""]

    cm = np.array(t["confusion"], dtype=float)
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(2.4 + 0.8 * K, 1.8 + 0.8 * K))
    ax.imshow(cmn, cmap=BLUES, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center", color="white" if cmn[i, j] > 0.55 else INK)
    ax.set_xticks(range(K), classes, rotation=30, ha="right")
    ax.set_yticks(range(K), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Test confusion (counts)")
    fig.tight_layout()
    fig.savefig(out / "confusion.png")
    plt.close(fig)

    hist_file = out / "history.json"
    hist = json.loads(hist_file.read_text()) if hist_file.exists() else []
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
    axes[0].plot([h["epoch"] for h in hist], [h["train_loss"] for h in hist], color=SERIES[0], lw=2)
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    for key, label, col in (("train_macro_f1", "train", SERIES[0]), ("val_macro_f1", "validation", SERIES[1])):
        pts = [(h["epoch"], h[key]) for h in hist if key in h]
        if pts:
            axes[1].plot(*zip(*pts), color=col, lw=2, marker="o", ms=3, label=label)
    axes[1].set_title("Macro-F1 during training")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "curves.png")
    plt.close(fig)

    # ------------------------------------------------------------ 3: distribution shift
    rob = {}
    for name in C.CORRUPTIONS:
        rob[name] = []
        for s in (1, 2, 3):
            p, y = run(model, te, a.data, size, device, pre=C.corruption(name, s), workers=a.workers)
            rob[name].append(C.macro_f1(y, p))
    allv = [v for vs in rob.values() for v in vs]
    worst = min(rob, key=lambda k: rob[k][-1])
    R["corruptions_macro_f1"] = rob
    R["corruption_mean_macro_f1"] = float(np.mean(allv))
    R["corruption_retained"] = float(np.mean(allv) / max(t["macro_f1"], 1e-8))
    md += ["## 3. Distribution shift", "",
           "Same test crops, degraded the way a helmet camera in a hangar would degrade them. Macro-F1:", "",
           md_table(["condition", "mild", "medium", "severe"], [[k] + [f3(v) for v in vs] for k, vs in rob.items()]), "",
           f"Clean {t['macro_f1']:.3f} -> mean under corruption **{R['corruption_mean_macro_f1']:.3f}** "
           f"({R['corruption_retained']:.0%} retained). Weakest condition: {worst}.", "",
           "![robustness](robustness.png)", ""]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    for i, (k, vs) in enumerate(rob.items()):
        ax.plot([0, 1, 2, 3], [t["macro_f1"]] + vs, color=SERIES[i], lw=2, marker="o", ms=4, label=k)
    ax.set_xticks([0, 1, 2, 3], ["clean", "mild", "medium", "severe"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("test macro-F1")
    ax.set_title("Robustness to camera and lighting conditions")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(out / "robustness.png")
    plt.close(fig)

    src = te["source"].to_numpy()
    if len(np.unique(src)) > 1:
        rows = []
        R["per_source"] = {}
        for s in np.unique(src):
            m = src == s
            R["per_source"][s] = dict(n=int(m.sum()), accuracy=float((p_te[m].argmax(1) == y_te[m]).mean()),
                                      balanced_accuracy=defect_bal_acc(y_te[m], p_te[m]))
            rows.append([s, int(m.sum()), f3(R["per_source"][s]["accuracy"]), f3(R["per_source"][s]["balanced_accuracy"])])
        md += ["Score by photo source (guessed from filename prefixes). A source that scores far below the rest is a shift the model has not absorbed:", "",
               md_table(["source", "test crops", "accuracy", "balanced accuracy"], rows), ""]

    extra = []
    ood = df[df.split == "ood"].reset_index(drop=True)
    if len(ood):
        extra.append((f"held-out source ({', '.join(sorted(ood['source'].unique()))})", ood, a.data))
    if a.shift_data:
        sdf, _ = C.load_metadata(a.shift_data)
        mapping = dict(pair.split("=") for pair in a.class_map.split(",") if "=" in pair)
        mapping = {k.strip(): v.strip() for k, v in mapping.items()} | {c: c for c in classes}
        sdf["label"] = sdf["label"].map(mapping)
        sdf = sdf[sdf["label"].isin(classes)].reset_index(drop=True)
        sdf["y"] = sdf["label"].map({c: i for i, c in enumerate(classes)}).astype(int)
        if len(sdf):
            extra.append((f"other dataset ({Path(a.shift_data).name})", sdf, a.shift_data))
    R["shift_sets"] = {}
    for label, d, root in extra:
        p, y = run(model, d, root, size, device, workers=a.workers)
        m = C.compute_metrics(y, p, classes)
        R["shift_sets"][label] = m
        md += [f"**Never-trained-on data: {label}**, {m['n']} crops: macro-F1 {m['macro_f1']:.3f}, "
               f"balanced accuracy {m['balanced_accuracy']:.3f} (in-distribution test: {t['macro_f1']:.3f} / {t['balanced_accuracy']:.3f}).", "",
               md_table(["class", "recall", "crops"], [[c, f3(v["recall"]), v["support"]] for c, v in m["per_class"].items() if v["support"]]), ""]

    # ------------------------------------------------------------ 4: shortcut learning
    has_box = ~te[["bx0"]].isna().to_numpy().ravel()
    dt = te[has_box].reset_index(drop=True)
    yd, pd_clean = y_te[has_box], p_te[has_box]
    p_hid, _ = run(model, dt, a.data, size, device, pre=C.hide_defect, workers=a.workers)
    p_ctx, _ = run(model, dt, a.data, size, device, pre=C.hide_context, workers=a.workers)
    chance = 1.0 / K
    clean, hid, ctx = defect_bal_acc(yd, pd_clean), defect_bal_acc(yd, p_hid), defect_bal_acc(yd, p_ctx)
    reliance = float(np.clip((hid - chance) / max(clean - chance, 1e-8), 0, 1))
    S = dict(clean=clean, defect_hidden=hid, context_hidden=ctx, chance=chance, context_reliance=reliance)
    if C.BACKGROUND in classes:
        S["defect_hidden_called_no_damage"] = float((p_hid.argmax(1) == classes.index(C.BACKGROUND)).mean())

    correct = np.nonzero(pd_clean.argmax(1) == yd)[0]
    pick = np.random.default_rng(0).permutation(correct)[: a.cam_samples]
    tf = C.eval_transform(size)
    inside, area, shown = [], [], []
    for start in range(0, len(pick), 32):
        idx = pick[start:start + 32]
        imgs = [Image.open(Path(a.data) / dt.loc[i, "path"]).convert("RGB") for i in idx]
        x = torch.stack([tf(im) for im in imgs]).to(device).requires_grad_(True)
        cam = gradcam(model, x, torch.as_tensor(yd[idx], device=device)).cpu().numpy()
        for i, im, c in zip(idx, imgs, cam):
            b = dt.loc[i, ["bx0", "by0", "bx1", "by1"]].to_numpy(dtype=float)
            x0, y0, x1, y1 = (np.round(b * size)).astype(int)
            if c.sum() <= 0 or x1 <= x0 or y1 <= y0:
                continue
            inside.append(float(c[y0:y1, x0:x1].sum() / c.sum()))
            area.append(float((x1 - x0) * (y1 - y0) / size ** 2))
            if len(shown) < 8:
                shown.append((im.resize((size, size)), c, (x0, y0, x1, y1), classes[yd[i]]))
    if inside:
        S["gradcam_mass_inside_box"] = float(np.mean(inside))
        S["box_area_fraction"] = float(np.mean(area))
        S["gradcam_focus_ratio"] = float(np.mean(inside) / np.mean(area))
    R["shortcut"] = S
    rows = [["defect and context both visible", f3(clean), "reference"],
            ["defect painted over (context only)", f3(hid), f"should fall towards chance ({chance:.3f})"],
            ["context painted over (defect only)", f3(ctx), "should stay near the reference (necessary, not sufficient)"]]
    md += ["## 4. Shortcut learning", "",
           "Each test crop comes with the annotator's box around the defect, so we can ask what the model actually used. Balanced accuracy over the defect classes:", "",
           md_table(["what the model can see", "balanced accuracy", "what we want"], rows), "",
           f"**Context reliance: {reliance:.0%}** of the model's above-chance skill survives with the defect hidden "
           "(0% = it needs to see the defect; 100% = it never needed to)."]
    if "defect_hidden_called_no_damage" in S:
        md += [f"With the defect hidden, {S['defect_hidden_called_no_damage']:.0%} of crops are called `{C.BACKGROUND}` (the correct answer for a hidden defect)."]
    if inside:
        md += ["", f"Grad-CAM: {S['gradcam_mass_inside_box']:.0%} of the heat lands inside the defect box, which covers "
               f"{S['box_area_fraction']:.0%} of the crop on average: **{S['gradcam_focus_ratio']:.2f}x** what uniform attention would give "
               "(above 1 = looks at the defect; at or below 1 = looks elsewhere).", "", "![gradcam](gradcam.png)"]
        fig, axes = plt.subplots(2, len(shown), figsize=(1.7 * len(shown), 3.7), squeeze=False)
        for j, (im, c, (x0, y0, x1, y1), lab) in enumerate(shown):
            axes[0, j].imshow(im)
            axes[1, j].imshow(im)
            axes[1, j].imshow(c, cmap="magma", alpha=0.55, vmin=0, vmax=1)
            for ax in axes[:, j]:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="white", lw=1.2))
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
            axes[0, j].set_title(lab, fontsize=8, loc="center")
        fig.suptitle("Where the model looks (white box = annotated defect, bright = high Grad-CAM)", fontsize=9, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(out / "gradcam.png")
        plt.close(fig)
    md += ["", "Caveats: the defect-hidden row is the decisive one. The context-hidden row cannot rule out a colour shortcut, "
           "because the surface colour is still visible inside the box. A painted rectangle also reveals the box's shape "
           "(a long thin box hints at a crack), so read these as a comparison between models, not as absolute proof.", ""]

    # ------------------------------------------------------------ 5: PCA of the learned features
    pca = PCA().fit(e_tr)
    cum = np.cumsum(pca.explained_variance_ratio_)
    need = {f"{int(q * 100)}%": int(np.searchsorted(cum, q) + 1) for q in (0.90, 0.95, 0.99)}
    ks = [k for k in (2, 4, 8, 16, 32, 64) if k < min(e_tr.shape)] + [min(e_tr.shape)]
    sweep = {}
    for k in ks:
        clf = make_pipeline(StandardScaler(), PCA(k), LogisticRegression(max_iter=3000, class_weight="balanced"))
        sweep[k] = float(f1_score(y_te, clf.fit(e_tr, y_tr).predict(e_te), average="macro", zero_division=0))
    P = dict(feature_dim=int(e_tr.shape[1]), components_for_variance=need, macro_f1_by_components=sweep)
    s_tr = tr["source"].to_numpy()
    shared = [s for s in np.unique(s_tr) if (s_tr == s).sum() >= 10 and (src == s).sum() >= 5]
    if len(shared) > 1:  # can a linear probe read the photo source off the features?
        mtr, mte = np.isin(s_tr, shared), np.isin(src, shared)
        probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")).fit(e_tr[mtr], s_tr[mtr])
        P["source_probe_balanced_accuracy"] = float(balanced_accuracy_score(src[mte], probe.predict(e_te[mte])))
        P["source_probe_chance"] = 1.0 / len(shared)
    R["pca"] = P
    md += ["## 5. PCA of the learned features", "",
           f"The model's feature vector has {P['feature_dim']} dimensions. PCA on the training features: "
           f"{need['90%']} components hold 90% of the variance, {need['95%']} hold 95%, {need['99%']} hold 99%.", "",
           "Test macro-F1 of a plain linear classifier on only the first k principal components:", "",
           md_table(["components kept"] + [str(k) for k in ks], [["macro-F1"] + [f3(sweep[k]) for k in ks]]), "",
           "If the score saturates at a small k, the rest of the feature space is spare capacity that can only memorise."]
    if "source_probe_balanced_accuracy" in P:
        md += ["", f"Source probe: a linear classifier predicts which photo source a test crop came from with balanced accuracy "
               f"{P['source_probe_balanced_accuracy']:.3f} (chance {P['source_probe_chance']:.3f}). High means the features carry "
               "\"which camera/collection\" information; combined with a class-source imbalance in `data_audit.py`, that is the shortcut to worry about."]
    md += ["", "![pca](pca.png)", ""]
    z = pca.transform(e_te)[:, :2]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].plot(range(1, len(cum) + 1), cum, color=SERIES[0], lw=2)
    axes[0].set_xscale("log")
    axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("principal components kept")
    axes[0].set_title("Cumulative variance explained")
    for ax, key, title in ((axes[1], "label", "Test features by class"), (axes[2], "source", "Same points by photo source")):
        vals = te[key].to_numpy()
        order = classes if key == "label" else sorted(np.unique(vals))
        for i, v in enumerate(order[:8]):
            m = vals == v
            ax.scatter(z[m, 0], z[m, 1], s=14, color=SERIES[i], edgecolor="#fcfcfb", lw=0.4, label=v)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out / "pca.png")
    plt.close(fig)

    (out / "report.json").write_text(json.dumps(R, indent=1))
    (out / "report.md").write_text("\n".join(md))
    print(f"test macro-F1 {t['macro_f1']:.3f}  bal-acc {t['balanced_accuracy']:.3f}  train-test gap {R['overfit_gap_macro_f1']:+.3f}  "
          f"under corruption {R['corruption_mean_macro_f1']:.3f}  context reliance {reliance:.0%}")
    print(f"report: {out / 'report.md'}")


if __name__ == "__main__":
    main()
