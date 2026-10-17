"""Step 7: train the MLP head on the D1 embeddings of the trained ResNet-18 (Apple GPU / MPS).

The ResNet-18's own last layer (fc) is a single linear layer from the 512-d embedding to the 6 classes.
The MLP head replaces it with an ensemble of 5 small MLPs, averaged with a 5-nearest-neighbour vote
against the D1-train embeddings. The ResNet-18 itself is not changed or retrained.

Input: <results>/embeddings.npz from embed.py, D1 rows only. The saved D1 split is kept as it is:
  train (20,051 crops)  fits the MLPs and is the kNN bank
  val   ( 4,293 crops)  picks each MLP's best epoch and the kNN blend weight
  test  ( 4,293 crops)  is never touched until the final report

Optional extra training data: --aug-views N re-embeds every D1-train crop N times through the ResNet-18
with its training augmentation (cached in <work>/aug_views.npz, about 1 min per view on MPS). It gave
no gain in our run, so it is off by default.

Outputs:
  <work>/mlp_head.pt   the trained head (MLP ensemble + kNN bank), used by predict.py
  <out>/report.md, report.json, confusion.png, confusion_resnet_fc.png, curves.png, history.csv, test_predictions.csv
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch import nn

from klshift.common import pick_device
from klshift.mlp import HEAD_FILE, Ensemble, MLPHead, knn_proba, l2n, load_d1_embeddings, sync


def augmented_views(meta, n_views, work, device, seed, batch=128, workers=4):
    """N augmented embeddings per D1-train crop, cached in <work>/aug_views.npz. Returns (n_views, N_train, 512)."""
    paths = meta["crop_path"][meta["split"] == "train"]
    cache = work / "aug_views.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if len(z["crop_path"]) == len(paths) and (z["crop_path"] == paths).all() and len(z["emb"]) >= n_views:
            print(f"augmented views: using {n_views} of {len(z['emb'])} cached in {cache}")
            return z["emb"][:n_views]
    from torch.utils.data import DataLoader
    from evaluate import load_model
    from klshift.common import CropDataset, embed_batch, seed_everything, train_transform
    seed_everything(seed)
    model, _ = load_model(work, device)
    views = []
    for v in range(n_views):
        t0 = time.time()
        dl = DataLoader(CropDataset(paths, transform=train_transform()),
                        batch_size=batch, num_workers=workers)
        with torch.no_grad():
            views.append(np.concatenate([embed_batch(model, x.to(device)).cpu().numpy() for x, _ in dl]))
        sync(device)
        print(f"  augmented view {v + 1}/{n_views}: {len(paths)} crops on {device} in {time.time() - t0:.0f}s", flush=True)
    E = np.stack(views).astype(np.float32)
    np.savez(cache, emb=E, crop_path=paths)
    return E


def macro_f1(y, p):
    return f1_score(y, p, average="macro")


def train_one(Xtr, ytr, Xva, yva, n_classes, a, seed, device):
    """One MLP. All tensors live on `device`; returns (best state, history)."""
    torch.manual_seed(seed)
    head = MLPHead(n_classes, tuple(a.hidden), a.dropout, a.in_dropout)
    head.mu.copy_(Xtr.mean(0).cpu())
    head.sd.copy_(Xtr.std(0).cpu() + 1e-6)
    head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    g = torch.Generator(device=device).manual_seed(seed)
    noise_scale = a.noise * head.sd  # noise in standardised units, applied to raw embeddings
    best, best_state, hist = -1.0, None, []
    yva_np = yva.cpu().numpy()
    for ep in range(1, a.epochs + 1):
        head.train()
        perm = torch.randperm(len(Xtr), device=device, generator=g)
        tot = 0.0
        for i in range(0, len(Xtr), a.batch):
            b = perm[i:i + a.batch]
            xb = Xtr[b] + noise_scale * torch.randn(len(b), Xtr.shape[1], device=device, generator=g)
            loss = loss_fn(head(xb), ytr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.detach() * len(b)
        sched.step()
        head.eval()
        with torch.no_grad():
            pv = head(Xva).argmax(1).cpu().numpy()
        f = macro_f1(yva_np, pv)
        hist.append(dict(seed=seed, epoch=ep, train_loss=float(tot) / len(Xtr), val_macro_f1=f,
                         val_acc=accuracy_score(yva_np, pv)))
        if f > best:
            best, best_state = f, {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    return best_state, hist


def metrics(y, p, classes):
    lab = list(range(len(classes)))
    return dict(accuracy=accuracy_score(y, p), macro_f1=macro_f1(y, p), balanced_accuracy=balanced_accuracy_score(y, p),
                per_class_recall=dict(zip(classes, recall_score(y, p, labels=lab, average=None, zero_division=0).round(4).tolist())),
                per_class_precision=dict(zip(classes, precision_score(y, p, labels=lab, average=None, zero_division=0).round(4).tolist())),
                confusion_matrix=confusion_matrix(y, p, labels=lab).tolist())


def plot_confusion(cm, classes, title, path):
    cm = np.array(cm)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    lab = range(len(classes))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in lab:
        for j in lab:
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j] * 100:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xticks(list(lab), classes, rotation=35, ha="right")
    ax.set_yticks(list(lab), classes)
    ax.set(xlabel="predicted", ylabel="true", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="data/work", help="folder with model_best.pt; mlp_head.pt is saved here")
    ap.add_argument("--results", default="results", help="folder with embeddings.npz")
    ap.add_argument("--out", default="results/mlp", help="where the report and figures go")
    ap.add_argument("--aug-views", type=int, default=0, help="augmented embeddings per train crop (0 = plain embeddings only)")
    ap.add_argument("--seeds", type=int, default=5, help="MLPs in the ensemble")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, nargs="+", default=[1024, 512, 256])
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--in-dropout", type=float, default=0.2)
    ap.add_argument("--noise", type=float, default=0.5, help="Gaussian noise on standardised embeddings during training")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--knn", type=int, default=5, help="k for the kNN part of the blend (0 = MLP only)")
    ap.add_argument("--cpu", action="store_true", help="force CPU (default: Apple GPU / MPS)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    RESNET, HEAD, RESULTS = work / "model_best.pt", work / HEAD_FILE, out
    device = torch.device("cpu") if a.cpu else pick_device()  # Apple GPU (mps) on a Mac
    print(f"device: {device}" + ("" if device.type != "cpu" or a.cpu else "  (WARNING: no Apple GPU / CUDA found)"))

    X, meta, classes = load_d1_embeddings(Path(a.results) / "embeddings.npz")
    y = np.array([classes.index(c) for c in meta["cls"]])
    sp = meta["split"]
    tr, va, te = (sp == "train"), (sp == "val"), (sp == "test")
    print(f"D1 embeddings: {len(X)} crops ({tr.sum()} train / {va.sum()} val / {te.sum()} test), classes: {classes}")

    Xtr_np, ytr_np = X[tr], y[tr]
    if a.aug_views > 0:
        A = augmented_views(meta, a.aug_views, work, device, a.seed)
        Xtr_np = np.concatenate([Xtr_np, *A])
        ytr_np = np.tile(y[tr], 1 + a.aug_views)
        print(f"train set: {tr.sum()} plain + {a.aug_views} x {tr.sum()} augmented = {len(Xtr_np)} embeddings")

    T = lambda v, dt=torch.float32: torch.as_tensor(v, dtype=dt, device=device)  # noqa: E731
    Xtr, ytr = T(Xtr_np), T(ytr_np, torch.long)
    Xva, yva, Xte = T(X[va]), T(y[va], torch.long), T(X[te])

    # ---- MLP ensemble
    states, hist = [], []
    t0 = time.time()
    for s in range(a.seeds):
        t1 = time.time()
        st, h = train_one(Xtr, ytr, Xva, yva, len(classes), a, a.seed + s, device)
        states.append(st)
        hist += h
        bh = max(h, key=lambda r: r["val_macro_f1"])
        print(f"  MLP seed {a.seed + s}: best val macro-F1 {bh['val_macro_f1']:.4f} at epoch {bh['epoch']}  "
              f"({time.time() - t1:.0f}s on {device})", flush=True)
    sync(device)
    print(f"trained {a.seeds} MLPs in {time.time() - t0:.0f}s")
    arch = dict(hidden=tuple(a.hidden), dropout=a.dropout, in_dropout=a.in_dropout)
    ens = Ensemble([MLPHead(len(classes), **arch) for _ in states])
    for hd, st in zip(ens.heads, states):
        hd.load_state_dict(st)
    ens.to(device).eval()
    with torch.no_grad():
        mlp_va, mlp_te = ens(Xva), ens(Xte)

    # ---- kNN on the plain train embeddings, blend weight picked on val (test is not looked at)
    alpha, knn_va, knn_te = 0.0, None, None
    if a.knn > 0:
        bank, bank_y = l2n(T(X[tr])), T(y[tr], torch.long)
        knn_va = knn_proba(bank, bank_y, l2n(Xva), a.knn, len(classes))
        knn_te = knn_proba(bank, bank_y, l2n(Xte), a.knn, len(classes))
        grid = [0.0, 0.25, 0.5, 0.75]
        scores = {w: macro_f1(y[va], ((1 - w) * mlp_va + w * knn_va).argmax(1).cpu().numpy()) for w in grid}
        alpha = max(scores, key=scores.get)
        print("kNN blend weight on val: " + ", ".join(f"{w}: {s:.4f}" for w, s in scores.items()) + f"  -> {alpha}")

    def final(mlp, knn):
        return mlp if alpha == 0 else (1 - alpha) * mlp + alpha * knn

    # ---- baseline: the ResNet-18's own fc layer applied to the same embeddings
    ck = torch.load(RESNET, map_location="cpu") if RESNET.exists() else None
    if ck is None:
        print(f"note: {RESNET} not found, baseline taken from the main model's d1_report.json values only")

    def fc_pred(Xs):
        return (Xs @ ck["state"]["fc.weight"].numpy().T + ck["state"]["fc.bias"].numpy()).argmax(1)

    rows, rep = [], dict(classes=classes, n_train_crops=int(tr.sum()), n_train_embeddings=len(Xtr_np),
                         n_val=int(va.sum()), n_test=int(te.sum()), knn_k=a.knn, knn_weight=alpha,
                         args=vars(a), device=str(device))
    sets = {"val": (va, final(mlp_va, knn_va)), "test": (te, final(mlp_te, knn_te))}
    for name, (mask, prob) in sets.items():
        p = prob.argmax(1).cpu().numpy()
        rep[name] = metrics(y[mask], p, classes)
        rep[name + "_mlp_only"] = metrics(y[mask], (mlp_va if name == "val" else mlp_te).argmax(1).cpu().numpy(), classes)
        if ck is not None:
            rep[name + "_resnet_fc"] = metrics(y[mask], fc_pred(X[mask]), classes)
    test_prob = sets["test"][1].cpu().numpy()
    if ck is not None:  # exact McNemar test: is the MLP head really better than the ResNet fc on the same test crops?
        ok5, okfc = test_prob.argmax(1) == y[te], fc_pred(X[te]) == y[te]
        b, c = int((ok5 & ~okfc).sum()), int((~ok5 & okfc).sum())
        rep["mcnemar_test"] = dict(only_mlp_right=b, only_resnet_right=c,
                                   p_value=binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)
    pd.DataFrame({"crop_path": meta["crop_path"][te], "true": meta["cls"][te],
                  "pred": [classes[i] for i in test_prob.argmax(1)], "confidence": test_prob.max(1).round(4),
                  **{f"p_{c}": test_prob[:, i].round(4) for i, c in enumerate(classes)}}).to_csv(
        RESULTS / "test_predictions.csv", index=False)

    torch.save(dict(states=states, arch=arch, classes=classes, knn_k=a.knn, knn_weight=alpha,
                    knn_bank=torch.as_tensor(X[tr]) if alpha > 0 else None,
                    knn_labels=torch.as_tensor(y[tr]) if alpha > 0 else None,
                    embeddings=str(Path(a.results) / "embeddings.npz"), args=vars(a)), HEAD)
    (RESULTS / "report.json").write_text(json.dumps(rep, indent=2))
    H = pd.DataFrame(hist)
    H.to_csv(RESULTS / "history.csv", index=False)

    plot_confusion(rep["test"]["confusion_matrix"], classes, "MLP head: D1-test confusion", RESULTS / "confusion.png")
    if "test_resnet_fc" in rep:
        plot_confusion(rep["test_resnet_fc"]["confusion_matrix"], classes, "ResNet-18 fc (baseline): D1-test confusion",
                       RESULTS / "confusion_resnet_fc.png")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for s, g in H.groupby("seed"):
        ax[0].plot(g.epoch, g.train_loss, lw=1, label=f"seed {s}")
        ax[1].plot(g.epoch, g.val_macro_f1, lw=1, label=f"seed {s}")
    if "val_resnet_fc" in rep:
        ax[1].axhline(rep["val_resnet_fc"]["macro_f1"], color="gray", ls="--", lw=1, label="ResNet fc")
    ax[0].set(xlabel="epoch", ylabel="CE loss (label smoothing)", title="MLP training loss")
    ax[1].set(xlabel="epoch", ylabel="val macro-F1", title="Val macro-F1 per MLP")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "curves.png", dpi=150)
    plt.close(fig)

    # ---- report
    def row(name, m):
        return f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['balanced_accuracy']:.4f} |"
    md = ["# MLP head on the ResNet-18 embeddings: D1 report\n",
          f"Trained on {rep['n_train_embeddings']} embeddings ({rep['n_train_crops']} D1-train crops"
          + (f" + {a.aug_views} augmented views each" if a.aug_views else "") + f"); {a.seeds} MLPs "
          f"{'x'.join(map(str, a.hidden))}, " + (f"blended with {a.knn}-NN at weight {alpha} (picked on val)" if alpha else "no kNN")
          + f". Device: {device}.\n",
          "## D1-test (4,293 crops, never used for training or model selection)\n",
          "| model | accuracy | macro-F1 | balanced accuracy |", "|---|---|---|---|"]
    if "test_resnet_fc" in rep:
        md.append(row("ResNet-18 with its own fc layer", rep["test_resnet_fc"]))
    md.append(row("MLP ensemble only", rep["test_mlp_only"]))
    md.append(row("**MLP head (final: MLPs + kNN)**", rep["test"]))
    if "mcnemar_test" in rep:
        mc = rep["mcnemar_test"]
        md.append(f"\nOn the same test crops, the MLP head is right where the ResNet fc is wrong {mc['only_mlp_right']} times, "
                  f"and the other way round {mc['only_resnet_right']} times (exact McNemar test p = {mc['p_value']:.2g}).")
    md += ["\n## D1-val (used to pick epochs and the blend weight)\n",
           "| model | accuracy | macro-F1 | balanced accuracy |", "|---|---|---|---|"]
    if "val_resnet_fc" in rep:
        md.append(row("ResNet-18 fc", rep["val_resnet_fc"]))
    md.append(row("MLP head", rep["val"]))
    md += ["\n## Per-class recall on D1-test\n", "| class | ResNet-18 fc | MLP head | test crops |", "|---|---|---|---|"]
    sup = np.bincount(y[te], minlength=len(classes))
    for i, c in enumerate(classes):
        base = rep["test_resnet_fc"]["per_class_recall"][c] if "test_resnet_fc" in rep else float("nan")
        md.append(f"| {c} | {base:.4f} | {rep['test']['per_class_recall'][c]:.4f} | {sup[i]} |")
    md += ["\n![confusion](confusion.png)\n", "![baseline confusion](confusion_resnet_fc.png)\n", "![curves](curves.png)\n"]
    (RESULTS / "report.md").write_text("\n".join(md))
    print("\n".join(md[3:md.index("\n## D1-val (used to pick epochs and the blend weight)\n")]))
    print(f"\nsaved {HEAD} and {RESULTS / 'report.md'}")


if __name__ == "__main__":
    main()
