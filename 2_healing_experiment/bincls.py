"""Stock torchvision ResNet-18 binary classifier (ImageNet weights, only fc replaced).
Used by the artifact probe (step 4) and Stage 1 (step 5). Same training recipe as the
D1 classifier of the shift task: AdamW 1e-4, wd 0.05, class-weighted CE, its train
augmentation, early stopping on val AUROC."""
import time

import hcommon as H

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from klshift.common import CropDataset, build_model, train_transform


@torch.no_grad()
def predict(model, paths, device, batch=128, workers=4):
    """P(class 1) for every path (eval transform)."""
    model.eval()
    dl = DataLoader(CropDataset(paths), batch_size=batch, num_workers=workers)
    return np.concatenate([torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy() for x, _ in dl])


def train_binary(tr_paths, tr_y, va_paths, va_y, device, epochs=10, patience=2, batch=64,
                 lr=1e-4, wd=0.05, workers=4, seed=0, log=print, ckpt=None):
    H.seed_everything(seed)
    tr_y, va_y = np.asarray(tr_y), np.asarray(va_y)
    counts = np.bincount(tr_y, minlength=2)
    w = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(CropDataset(list(tr_paths), list(tr_y), train_transform()), batch_size=batch,
                    shuffle=True, drop_last=True, num_workers=workers, persistent_workers=True, generator=g)
    model = build_model(2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.CrossEntropyLoss(weight=w)
    best, bad, best_state, hist = -1, 0, None, []
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        auc = roc_auc_score(va_y, predict(model, va_paths, device, workers=workers))
        hist.append(dict(epoch=ep, train_loss=tot / n, val_auroc=auc, seconds=time.time() - t0))
        log(f"  epoch {ep:2d} loss {tot / n:.4f} val AUROC {auc:.4f} {time.time() - t0:.0f}s")
        if auc > best:
            best, bad = auc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    if ckpt:
        torch.save({"state": best_state, "val_auroc": best, "history": hist}, ckpt)
    return model, best, hist
