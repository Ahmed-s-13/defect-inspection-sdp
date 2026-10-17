"""Steps 3+4: leak-free 70/15/15 re-split of D1, then train ResNet-18 on D1 crops.

Groups kept on one side of the split: all crops of one original image, tiles of
one photo (shared stem), and near-duplicate images (D1-D1 pairs in dup_pairs.csv).

Outputs:
  <work>/d1_split.csv        crop -> train/val/test, group id
  <work>/model_best.pt       best checkpoint (by val macro-F1)
  <results>/d1_report.md, d1_report.json, d1_confusion.png, d1_training_curves.png, d1_history.csv
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
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from klshift.common import (CropDataset, UnionFind, build_model, load_crops, pick_device,
                            seed_everything, train_transform)


def make_split(work, seed):
    crops = load_crops(work, "D1")
    uf = UnionFind()
    for r in crops[["image", "group"]].drop_duplicates().itertuples():
        uf.union(("img", r.image), ("stem", r.group))           # tiles / same photo stem
    pairs = pd.read_csv(work / "dup_pairs.csv")
    pairs = pairs[pairs.dup & (pairs.ds_a == "D1") & (pairs.ds_b == "D1")]
    for r in pairs.itertuples():
        uf.union(("img", r.image_a), ("img", r.image_b))         # near-duplicates
    roots = {}
    crops["group_id"] = [roots.setdefault(uf.find(("img", im)), len(roots)) for im in crops.image]

    # 20 stratified group folds: 14 train / 3 val / 3 test = 70/15/15
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    fold = np.zeros(len(crops), int)
    for k, (_, idx) in enumerate(sgkf.split(crops, crops.cls, crops.group_id)):
        fold[idx] = k
    crops["split"] = np.where(fold < 14, "train", np.where(fold < 17, "val", "test"))
    assert crops.groupby("group_id").split.nunique().max() == 1
    crops[["crop_path", "cls", "image", "group_id", "split"]].to_csv(work / "d1_split.csv", index=False)

    print("D1 split (groups / images / crops):")
    for s in ["train", "val", "test"]:
        c = crops[crops.split == s]
        print(f"  {s:5s} {c.group_id.nunique():6d} groups {c.image.nunique():6d} images {len(c):6d} crops")
    print(f"  multi-image groups: {(crops.groupby('group_id').image.nunique() > 1).sum()}")
    print(pd.crosstab(crops.cls, crops.split).to_string())
    return crops


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(model(x.to(device)).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="data/work")
    ap.add_argument("--results", default="results")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--train-frac", type=float, default=1.0, help="subsample train groups to speed up")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    work, res = Path(a.work), Path(a.results)
    res.mkdir(parents=True, exist_ok=True)
    seed_everything(a.seed)
    device = pick_device()

    crops = make_split(work, a.seed)
    classes = sorted(crops.cls.unique())
    cidx = {c: i for i, c in enumerate(classes)}
    crops["y"] = crops.cls.map(cidx)
    tr, va, te = (crops[crops.split == s] for s in ["train", "val", "test"])
    if a.train_frac < 1:
        g = tr.group_id.unique()
        keep = np.random.default_rng(a.seed).choice(g, int(len(g) * a.train_frac), replace=False)
        tr = tr[tr.group_id.isin(keep)]
        print(f"subsampled train to {len(tr)} crops")

    counts = np.bincount(tr.y, minlength=len(classes))
    weights = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32)
    print("class weights:", {c: round(float(w), 2) for c, w in zip(classes, weights)})

    kw = dict(batch_size=a.batch, num_workers=a.workers, persistent_workers=a.workers > 0)
    dl_tr = DataLoader(CropDataset(tr.crop_path, tr.y, train_transform()), shuffle=True, drop_last=True, **kw)
    dl_va = DataLoader(CropDataset(va.crop_path, va.y), **kw)
    dl_te = DataLoader(CropDataset(te.crop_path, te.y), **kw)
    tr_eval = tr.sample(min(len(tr), 5000), random_state=a.seed)
    dl_tr_eval = DataLoader(CropDataset(tr_eval.crop_path, tr_eval.y), **kw)

    model = build_model(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights.to(device))
    hist, best, bad = [], -1, 0
    print(f"training on {device}: {len(tr)} train / {len(va)} val crops")
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n, correct = 0.0, 0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
            correct += (out.argmax(1) == y).sum().item()
        yv, pv = predict(model, dl_va, device)
        f1v = f1_score(yv, pv, average="macro")
        dt = time.time() - t0
        hist.append(dict(epoch=ep, train_loss=tot / n, train_acc_aug=correct / n,
                         val_acc=accuracy_score(yv, pv), val_macro_f1=f1v, seconds=dt))
        print(f"epoch {ep:2d}  loss {tot / n:.4f}  train-acc(aug) {correct / n:.3f}  "
              f"val macro-F1 {f1v:.4f}  {dt:.0f}s", flush=True)
        if ep == 1:
            print(f"  -> about {dt / 60:.1f} min per epoch; max {a.epochs} epochs "
                  f"= up to {dt * a.epochs / 3600:.1f} h (early stopping usually ends sooner)", flush=True)
        if f1v > best:
            best, bad = f1v, 0
            torch.save(dict(state=model.state_dict(), classes=classes, epoch=ep), work / "model_best.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stop after epoch {ep}")
                break
    H = pd.DataFrame(hist)
    H.to_csv(res / "d1_history.csv", index=False)

    ck = torch.load(work / "model_best.pt", map_location=device)
    model.load_state_dict(ck["state"])
    yt, pt = predict(model, dl_te, device)
    ytr, ptr = predict(model, dl_tr_eval, device)
    yv, pv = predict(model, dl_va, device)
    lab = list(range(len(classes)))
    rep = dict(
        best_epoch=int(ck["epoch"]), classes=classes,
        n_train=len(tr), n_val=len(va), n_test=len(te),
        test_accuracy=accuracy_score(yt, pt), test_macro_f1=f1_score(yt, pt, average="macro"),
        test_balanced_accuracy=balanced_accuracy_score(yt, pt),
        test_per_class_recall=dict(zip(classes, recall_score(yt, pt, labels=lab, average=None, zero_division=0).round(4).tolist())),
        test_support=dict(zip(classes, np.bincount(yt, minlength=len(classes)).tolist())),
        train_macro_f1_no_aug=f1_score(ytr, ptr, average="macro"),
        val_macro_f1=f1_score(yv, pv, average="macro"),
        confusion_matrix=confusion_matrix(yt, pt, labels=lab).tolist())
    (res / "d1_report.json").write_text(json.dumps(rep, indent=2))

    cm = np.array(rep["confusion_matrix"])
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j] * 100:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.6 else "black")
    ax.set_xticks(lab, classes, rotation=35, ha="right")
    ax.set_yticks(lab, classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("D1-test confusion matrix (count, row %)")
    fig.tight_layout()
    fig.savefig(res / "d1_confusion.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(H.epoch, H.train_loss, "o-")
    ax[0].set(xlabel="epoch", ylabel="weighted CE loss", title="Training loss")
    ax[1].plot(H.epoch, H.train_acc_aug, "o-", label="train accuracy (augmented)")
    ax[1].plot(H.epoch, H.val_acc, "o-", label="val accuracy")
    ax[1].plot(H.epoch, H.val_macro_f1, "o-", label="val macro-F1")
    ax[1].axvline(rep["best_epoch"], color="gray", ls="--", lw=1, label="best epoch")
    ax[1].set(xlabel="epoch", title="Accuracy / F1", ylim=(0, 1))
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(res / "d1_training_curves.png", dpi=150)
    plt.close(fig)

    md = [f"# D1 classifier report\n",
          f"Best epoch {rep['best_epoch']} of {len(H)}; {len(tr)} train / {len(va)} val / {len(te)} test crops.\n",
          "| metric | value |", "|---|---|",
          f"| test accuracy | {rep['test_accuracy']:.4f} |",
          f"| test macro-F1 | {rep['test_macro_f1']:.4f} |",
          f"| test balanced accuracy | {rep['test_balanced_accuracy']:.4f} |",
          f"| val macro-F1 | {rep['val_macro_f1']:.4f} |",
          f"| train macro-F1 (no augmentation, {len(tr_eval)} crops) | {rep['train_macro_f1_no_aug']:.4f} |",
          "\n| class | test recall | test crops |", "|---|---|---|"]
    md += [f"| {c} | {rep['test_per_class_recall'][c]:.4f} | {rep['test_support'][c]} |" for c in classes]
    md += ["\n![confusion](d1_confusion.png)\n", "![curves](d1_training_curves.png)\n"]
    (res / "d1_report.md").write_text("\n".join(md))
    print("\n".join(md[2:9 + len(classes) + 2]))


if __name__ == "__main__":
    main()
