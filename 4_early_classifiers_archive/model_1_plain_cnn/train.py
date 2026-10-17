"""Model 1: a plain CNN trained with cross-entropy. Nothing clever.

Five conv blocks (conv-BN-ReLU x2, max-pool), global average pooling, dropout,
one linear layer. Trained from scratch. The only concessions to the four
problems are the standard ones every supervised baseline should have:
    overfitting   -> augmentation, dropout, weight decay, early stopping
    imbalance     -> class-weighted loss (or --balance sampler / none)

    python model_1_plain_cnn/train.py --data data/aircraft --out model_1_plain_cnn/results/aircraft
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # shared code in 4_early_classifiers_archive/

import common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=12, help="stop after this many epochs without a better val macro-F1")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--balance", default="weights", choices=["weights", "sampler", "none"])
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    C.set_seed(a.seed)
    device = C.get_device()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df, classes = C.load_metadata(a.data)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    print(f"device={device}  train={len(tr)}  val={len(va)}  classes={classes}")

    sampler = C.balanced_sampler(tr["y"].to_numpy()) if a.balance == "sampler" else None
    train_loader = C.make_loader(C.CropDataset(tr, a.data, C.train_transform(a.img_size)), a.batch_size,
                                 shuffle=True, sampler=sampler, workers=a.workers, drop_last=True)
    # train set again without augmentation: the honest train score for the overfitting gap
    train_eval = C.make_loader(C.CropDataset(tr, a.data, C.eval_transform(a.img_size)), 128, workers=a.workers)
    val_loader = C.make_loader(C.CropDataset(va, a.data, C.eval_transform(a.img_size)), 128, workers=a.workers)

    model = C.Classifier("simple", len(classes), pretrained=False, dropout=a.dropout).to(device)
    weight = C.class_weights(tr["y"].to_numpy(), len(classes)).to(device) if a.balance == "weights" else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    history, best, bad = [], -1.0, 0
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []
        for i, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            print(f"\repoch {epoch}/{a.epochs}  batch {i}/{len(train_loader)}   ", end="", flush=True)
        sched.step()
        vp, vy = C.predict(model, val_loader, device)
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)), val_macro_f1=C.macro_f1(vy, vp),
                   val_bal_acc=C.bal_acc(vy, vp))
        if epoch % 5 == 0 or epoch == 1:
            tp, ty = C.predict(model, train_eval, device)
            row["train_macro_f1"] = C.macro_f1(ty, tp)
        history.append(row)
        flag = ""
        if row["val_macro_f1"] > best:
            best, bad, flag = row["val_macro_f1"], 0, "  *"
            C.save_checkpoint(out / "model.pt", model, classes, a.img_size, "baseline-simple", vars(a))
        else:
            bad += 1
        print(f"\repoch {epoch:3d}  loss {row['train_loss']:.3f}  val macro-F1 {row['val_macro_f1']:.3f}"
              f"  val bal-acc {row['val_bal_acc']:.3f}  {time.time() - t0:.0f}s{flag}")
        if bad >= a.patience:
            print(f"early stop: no improvement for {a.patience} epochs")
            break
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"best val macro-F1 {best:.3f} -> {out / 'model.pt'}")


if __name__ == "__main__":
    main()
