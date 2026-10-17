"""Model 2: supervised contrastive learning, then a class-balanced classifier.

Still fully supervised (it uses the same labels as Model 1), but the labels are
used differently, in two stages:

Stage 1 - representation (Khosla et al., "Supervised Contrastive Learning",
    NeurIPS 2020, arXiv:2004.11362). Every crop is augmented twice. The network
    is trained so that all views of the same class sit close together in
    embedding space and views of different classes sit far apart. The loss
    compares a crop against every other crop of its class under different
    lighting / framing, which is what pushes it towards the defect itself
    rather than one photo's incidental background.

Stage 2 - classifier (Kang et al., "Decoupling Representation and Classifier
    for Long-Tailed Recognition", ICLR 2020, arXiv:1910.09217, the "cRT"
    recipe). The encoder is frozen and only a linear layer is trained on top,
    with every class sampled equally often. Their finding: imbalance damages
    the classifier far more than the features, so rebalancing only this last
    step fixes most of it without distorting the representation.

    python model_2_supcon_resnet18/train.py --data data/aircraft --out model_2_supcon_resnet18/results/aircraft
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # shared code in 4_early_classifiers_archive/

import common as C  # noqa: E402


class SupConLoss(nn.Module):
    """Eq. 2 of Khosla et al. (the L_out form). z: [B, views, D], L2-normalised."""

    def __init__(self, temperature=0.1):
        super().__init__()
        self.t = temperature

    def forward(self, z, y):
        v = z.shape[1]
        z = torch.cat(z.unbind(1), 0).float()            # [B*views, D]
        y = y.repeat(v)
        eye = torch.eye(len(y), dtype=torch.bool, device=z.device)
        sim = (z @ z.T / self.t).masked_fill(eye, -1e9)   # an anchor is never its own negative
        log_prob = sim - torch.logsumexp(sim, 1, keepdim=True)
        pos = (y[:, None] == y[None, :]) & ~eye           # same class, different view/crop
        return (-(log_prob.masked_fill(~pos, 0)).sum(1) / pos.sum(1).clamp(min=1)).mean()


@torch.no_grad()
def embed_all(model, loader, device):
    model.eval()
    out, ys = [], []
    for x, y in loader:
        out.append(model.embed(x.to(device)).float().cpu())
        ys.append(torch.as_tensor(y))
    return torch.cat(out), torch.cat(ys)


def knn_val_f1(model, train_eval, val_loader, device, k=10):
    """Label-free-head check of the embedding: k-NN from train to val."""
    tx, ty = embed_all(model, train_eval, device)
    vx, vy = embed_all(model, val_loader, device)
    knn = KNeighborsClassifier(min(k, len(ty)), metric="cosine", weights="distance")
    knn.fit(tx.numpy(), ty.numpy())
    return float(f1_score(vy.numpy(), knn.predict(vx.numpy()), average="macro", zero_division=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50", "simple"])
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=60, help="stage 1 epochs")
    ap.add_argument("--batch-size", type=int, default=64, help="crops per batch (x2 views). Bigger is better for SupCon")
    ap.add_argument("--lr", type=float, default=1e-4, help="stage 1; use ~1e-3 with --no-pretrained")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--proj-dim", type=int, default=128)
    ap.add_argument("--stage1-sampling", default="balanced", choices=["balanced", "instance"],
                    help="balanced: every class equally often in stage 1 (default). instance: natural frequencies")
    ap.add_argument("--eval-every", type=int, default=5, help="k-NN validation check every N stage-1 epochs")
    ap.add_argument("--clf-epochs", type=int, default=100, help="stage 2 epochs (cheap: features are cached)")
    ap.add_argument("--clf-lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    C.set_seed(a.seed)
    device = C.get_device()
    amp = device.type == "cuda"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df, classes = C.load_metadata(a.data)
    tr, va = df[df.split == "train"], df[df.split == "val"]
    ytr = tr["y"].to_numpy()
    print(f"device={device}  train={len(tr)}  val={len(va)}  classes={classes}")

    # Class-balanced batches by default, so a rare class always has positives to pull
    # together. This is our choice, not part of either paper: see the README.
    two_view = C.CropDataset(tr, a.data, C.train_transform(a.img_size, strong=True), two_views=True)
    sampler = C.balanced_sampler(ytr) if a.stage1_sampling == "balanced" else None
    train_loader = C.make_loader(two_view, a.batch_size, shuffle=True, sampler=sampler,
                                 workers=a.workers, drop_last=True)
    train_eval = C.make_loader(C.CropDataset(tr, a.data, C.eval_transform(a.img_size)), 128, workers=a.workers)
    val_loader = C.make_loader(C.CropDataset(va, a.data, C.eval_transform(a.img_size)), 128, workers=a.workers)

    model = C.Classifier(a.arch, len(classes), pretrained=not a.no_pretrained, dropout=0.0).to(device)
    head = nn.Sequential(nn.Linear(model.feat_dim, model.feat_dim), nn.ReLU(inplace=True),
                         nn.Linear(model.feat_dim, a.proj_dim)).to(device)  # thrown away after stage 1
    params = list(model.body.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    warm = max(1, min(5, a.epochs // 10))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [
        torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, warm),
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, a.epochs - warm))], [warm])
    scaler = torch.amp.GradScaler(enabled=amp)
    loss_fn = SupConLoss(a.temperature)

    # ---------------------------------------------------------------- stage 1
    print(f"training: {len(train_loader)} batches per epoch, {a.epochs} epochs (a batch counter appears below)")
    history, best, best_state = [], -1.0, None
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        head.train()
        losses = []
        for i, (x, y) in enumerate(train_loader, 1):
            b, v = x.shape[:2]
            x, y = x.flatten(0, 1).to(device), y.to(device)
            with torch.autocast("cuda" if amp else "cpu", enabled=amp):  # mixed precision on CUDA only
                z = head(model.embed(x))
            z = F.normalize(z.float(), dim=1).view(b, v, -1)
            loss = loss_fn(z, y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            print(f"\r[stage 1] epoch {epoch}/{a.epochs}  batch {i}/{len(train_loader)}  "
                  f"{(time.time() - t0) / i:.1f}s per batch   ", end="", flush=True)
        sched.step()
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)))
        msg = f"[stage 1] epoch {epoch:3d}  supcon loss {row['train_loss']:.3f}"
        if epoch % a.eval_every == 0 or epoch == a.epochs:
            row["val_macro_f1"] = knn_val_f1(model, train_eval, val_loader, device)
            msg += f"  k-NN val macro-F1 {row['val_macro_f1']:.3f}"
            if row["val_macro_f1"] > best:  # keep the encoder that generalises best, not the last one
                best, best_state = row["val_macro_f1"], copy.deepcopy(model.body.state_dict())
                msg += "  *"
        history.append(row)
        print(f"\r{msg}  {time.time() - t0:.0f}s" + " " * 20)
    model.body.load_state_dict(best_state)

    # ---------------------------------------------------------------- stage 2 (cRT)
    tx, ty = embed_all(model, train_eval, device)
    vx, vy = embed_all(model, val_loader, device)
    fc = nn.Linear(model.feat_dim, len(classes)).to(device)
    opt2 = torch.optim.AdamW(fc.parameters(), lr=a.clf_lr, weight_decay=1e-3)
    feats = torch.utils.data.TensorDataset(tx, ty)
    feat_loader = torch.utils.data.DataLoader(feats, batch_size=128, sampler=C.balanced_sampler(ty.numpy()))
    best2, best_fc = -1.0, None
    for epoch in range(1, a.clf_epochs + 1):
        fc.train()
        for x, y in feat_loader:
            loss = F.cross_entropy(fc(x.to(device)), y.to(device))
            opt2.zero_grad()
            loss.backward()
            opt2.step()
        fc.eval()
        with torch.no_grad():
            pred = fc(vx.to(device)).argmax(1).cpu().numpy()
        f1 = float(f1_score(vy.numpy(), pred, average="macro", zero_division=0))
        if f1 > best2:
            best2, best_fc = f1, copy.deepcopy(fc.state_dict())
    model.fc.load_state_dict(best_fc)
    print(f"[stage 2] balanced linear classifier: best val macro-F1 {best2:.3f}")

    C.save_checkpoint(out / "model.pt", model, classes, a.img_size, f"supcon-{a.arch}", vars(a))
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print(f"saved {out / 'model.pt'}")


if __name__ == "__main__":
    main()
