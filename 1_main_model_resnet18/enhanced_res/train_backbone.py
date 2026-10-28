"""enhanced_res, step 1: train the ResNet-18 backbone on the leak-free D1 split (../d1_grouped), or on d1_gc7.

Same ResNet-18, crops and augmentation as ../train.py. What can change, by flag:
  --sampler base         shuffle every train crop once per epoch, class-weighted loss (as ../train.py)
  --sampler groupclass   group- and class-balanced sampling, plain loss:
                           within a class, a crop's weight is 1 / sqrt(crops of that class in its source group),
                           so a scene shot 500 times counts like ~22 crops, not 500;
                           each class's total weight is proportional to sqrt(its crop count), a softened class
                           balance (full balancing would show the 940 scratch crops ~9x as often as dents)
  --recipe base          AdamW lr 1e-4 everywhere, constant lr, early stop on val macro-F1 (as ../train.py)
  --recipe enhanced      lr 1e-4 for the pretrained layers, 1e-3 for the new fc layer; 1 warm-up epoch then
                         cosine decay; label smoothing 0.1; EMA of the weights (decay 0.998 per step), evaluated
                         and saved instead of the raw weights

The three backbones build_head.py compares (run from 1_main_model_resnet18/, ~30-60 min each on an M-series Mac):
  python enhanced_res/train_backbone.py --sampler base       --recipe base     --out r0_plain
  python enhanced_res/train_backbone.py --sampler groupclass --recipe base     --out r1_sampler
  python enhanced_res/train_backbone.py --sampler groupclass --recipe enhanced --out r2_sampler_recipe   # enhanced_res

The main model, gc7 (7 classes: the six defects + no-defect, data ../d1_gc7):
  python enhanced_res/train_backbone.py --data d1_gc7 --out gc7          # ~25 min, best epoch 5 of 10
  python enhanced_res/train_backbone.py --data d1_gc7_fix --out gc7_fix  # the same on D1 with the 27 Oct box corrections
  --data d1_gc7: classes = the six defects in alphabetical order, then "no-defect" (index 6); the same sampler weights
  over all 7 classes (no-defect gets ~14% of the draws); len(D1 train) / 0.7 draws per epoch, so the six defect
  classes get about as many draws as before; best epoch by val 7-class macro-F1. Then build_head_gc7.py.

Output: data/work/enhanced_res/runs/<out>/backbone.pt (best epoch by val macro-F1) and history.csv.
Test is not touched here.
"""
import argparse
import copy
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

from model import EWORK, load_split
from klshift.common import CropDataset, build_model, pick_device, seed_everything, train_transform


def sample_weights(tr):
    n_cg = tr.groupby(["cls", "src_group"]).cls.transform("size")
    w = 1 / np.sqrt(n_cg)
    total = w.groupby(tr.cls).transform("sum")
    n_c = tr.groupby("cls").cls.transform("size")
    return (w * np.sqrt(n_c) / total).values


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
    ap.add_argument("--sampler", choices=["base", "groupclass"], default="groupclass")
    ap.add_argument("--recipe", choices=["base", "enhanced"], default="enhanced")
    ap.add_argument("--data", choices=["d1_grouped", "d1_gc7", "d1_grouped_fix", "d1_gc7_fix"], default="d1_grouped")
    ap.add_argument("--out", required=True, help="run name under data/work/enhanced_res/runs/")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--smoothing", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = EWORK / "runs" / a.out
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(a.seed)
    device = pick_device()

    crops = load_split(a.data)
    classes = sorted(set(crops.cls) - {"no-defect"}) + (["no-defect"] if a.data.startswith("d1_gc7") else [])
    cidx = {c: i for i, c in enumerate(classes)}
    tr = crops[crops.split == "train"].reset_index(drop=True)
    va = crops[crops.split == "val"].reset_index(drop=True)
    ytr, yva = tr.cls.map(cidx).values, va.cls.map(cidx).values
    kw = dict(batch_size=a.batch, num_workers=a.workers, persistent_workers=a.workers > 0)
    ds_tr = CropDataset(tr.crop_path, ytr, train_transform())
    if a.sampler == "groupclass":
        w = sample_weights(tr)
        n_draw = len(tr) if a.data.startswith("d1_grouped") else int(round((tr.cls != "no-defect").sum() / 0.7))
        sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), n_draw, replacement=True,
                                        generator=torch.Generator().manual_seed(a.seed))
        dl_tr = DataLoader(ds_tr, sampler=sampler, drop_last=True, **kw)
        eff = pd.Series(w).groupby(tr.cls).sum() / w.sum()
        print("expected share of each class per epoch:", eff.round(3).to_dict())
        loss_w = None
    else:
        dl_tr = DataLoader(ds_tr, shuffle=True, drop_last=True, **kw)
        counts = np.bincount(ytr, minlength=len(classes))
        loss_w = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32).to(device)
    dl_va = DataLoader(CropDataset(va.crop_path, yva), **kw)

    model = build_model(len(classes)).to(device)
    if a.recipe == "enhanced":
        fc = [p for n, p in model.named_parameters() if n.startswith("fc.")]
        rest = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
        opt = torch.optim.AdamW([dict(params=rest, lr=a.lr), dict(params=fc, lr=a.head_lr)], weight_decay=a.wd)
        steps = a.epochs * len(dl_tr)
        warm = len(dl_tr)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))
        loss_fn = torch.nn.CrossEntropyLoss(weight=loss_w, label_smoothing=a.smoothing)
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        sched, ema = None, None
        loss_fn = torch.nn.CrossEntropyLoss(weight=loss_w)

    print(f"{a.sampler} sampler, {a.recipe} recipe, {device}: {len(tr)} train / {len(va)} val crops", flush=True)
    hist, best, bad = [], -1, 0
    for ep in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.state_dict().values(), model.state_dict().values()):
                        if pe.dtype.is_floating_point:
                            pe.mul_(a.ema).add_(pm.detach(), alpha=1 - a.ema)
                        else:
                            pe.copy_(pm)
            tot += loss.item() * len(y)
            n += len(y)
        evm = ema if ema is not None else model
        yv, pv = predict(evm, dl_va, device)
        f1v = f1_score(yv, pv, average="macro")
        dt = time.time() - t0
        hist.append(dict(epoch=ep, train_loss=tot / n, val_acc=float((yv == pv).mean()), val_macro_f1=f1v, seconds=dt))
        pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
        print(f"epoch {ep:2d}  loss {tot / n:.4f}  val acc {(yv == pv).mean():.4f}  val macro-F1 {f1v:.4f}  {dt:.0f}s",
              flush=True)
        if f1v > best:
            best, bad = f1v, 0
            torch.save(dict(state=evm.state_dict(), classes=classes, epoch=ep, args=vars(a)), out / "backbone.tmp.pt")
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stop after epoch {ep}; best val macro-F1 {best:.4f}", flush=True)
                break
    (out / "backbone.tmp.pt").rename(out / "backbone.pt")


if __name__ == "__main__":
    main()
