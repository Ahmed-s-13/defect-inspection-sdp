"""Siamese copy-detector trained on D1 itself, then used to audit train/val/test splits.

Shared-weight ResNet-18 encoder (ImageNet init) -> 128-d embedding. Training is supervised
contrastive: two images are a positive pair if they share an SSCD strict group (plus two
augmented views of one image); everything else in the batch is a negative. 15% of strict groups
are held out to measure whether the model finds copies it was never trained on.

  train:  python siamese_audit.py train
  audit:  python siamese_audit.py audit
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision.transforms import v2

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT / "data" / "siamese"
OUT = ROOT / "reports" / "siamese_audit"
SIZE, EMB, EPOCHS, GROUPS_PER_BATCH, SEED = 160, 128, 12, 64, 0
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def manifest():
    m = pd.read_csv(ROOT / "manifests" / "union_split.csv")
    m["file"] = m["path"].map(lambda p: ROOT / p)
    return m


def load_images(m):
    cache = WORK / f"images_{SIZE}.npy"
    if cache.exists():
        return np.load(cache)
    WORK.mkdir(parents=True, exist_ok=True)
    arr = np.stack([np.asarray(Image.open(f).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)) for f in m.file])
    np.save(cache, arr)
    return arr


class Siamese(nn.Module):
    def __init__(self):
        super().__init__()
        net = torchvision.models.resnet18(weights=None)
        wts = next((ROOT / "models").rglob("resnet18*.pth"), None)
        if wts is not None:
            net.load_state_dict(torch.load(wts, map_location="cpu"))
        net.fc = nn.Identity()
        self.enc = net
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, EMB))

    def forward(self, x):
        return F.normalize(self.head(self.enc(x)), dim=1)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
AUG = v2.Compose([
    v2.RandomResizedCrop(SIZE, scale=(0.5, 1.0), antialias=True),
    v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(),
    v2.RandomApply([v2.RandomRotation(90)], p=0.5),
    v2.ColorJitter(0.4, 0.4, 0.4, 0.05), v2.RandomGrayscale(0.2),
])


def to_batch(arr, idx, aug):
    x = torch.from_numpy(arr[idx]).permute(0, 3, 1, 2)
    if aug:
        x = torch.stack([AUG(i) for i in x])
    return ((x.float() / 255 - MEAN) / STD)


def supcon(z, y, t=0.1):
    sim = z @ z.T / t
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None, :]) & ~eye
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    n = pos.sum(1)
    ok = n > 0
    return -((logp * pos).sum(1)[ok] / n[ok]).mean()


@torch.no_grad()
def embed(model, arr, bs=256):
    model.eval()
    out = []
    for i in range(0, len(arr), bs):
        out.append(model(to_batch(arr, np.arange(i, min(i + bs, len(arr))), False).to(dev)).cpu())
    return torch.cat(out).numpy()


def held_out_groups(m):
    rng = np.random.default_rng(SEED)
    g = m.sscd_group.unique()
    return set(rng.choice(g, int(len(g) * 0.15), replace=False))


def train():
    torch.manual_seed(SEED)
    m = manifest()
    arr = load_images(m)
    hold = held_out_groups(m)
    is_hold = m.sscd_group.isin(hold).values
    tr = m[~is_hold].reset_index()
    by_group = tr.groupby("sscd_group")["index"].apply(np.array)
    gids = by_group.index.values
    rng = np.random.default_rng(SEED)
    model = Siamese().to(dev)
    steps = len(tr) // (GROUPS_PER_BATCH * 2)
    opt = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=EPOCHS * steps)
    for ep in range(EPOCHS):
        model.train(); tot = 0
        for _ in range(steps):
            gs = rng.choice(gids, GROUPS_PER_BATCH, replace=False)
            idx = np.concatenate([rng.choice(by_group[g], 2) for g in gs])
            y = torch.tensor(np.repeat(np.arange(GROUPS_PER_BATCH), 2), device=dev)
            x = to_batch(arr, idx, True).to(dev)
            loss = supcon(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        print(f"epoch {ep + 1}/{EPOCHS} loss {tot / steps:.4f}", flush=True)
        torch.save(model.state_dict(), WORK / "siamese.pt")


def audit():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    m = manifest()
    arr = load_images(m)
    model = Siamese().to(dev)
    model.load_state_dict(torch.load(WORK / "siamese.pt", map_location=dev))
    z = embed(model, arr)
    np.save(WORK / "embeddings.npy", z)
    hold = m.sscd_group.isin(held_out_groups(m)).values
    md = ["# Siamese audit of the splits\n"]

    # held-out check: does it find copies from groups it never trained on?
    zh, mh = z[hold], m[hold].reset_index(drop=True)
    S = zh @ zh.T
    np.fill_diagonal(S, -2)
    same = (mh.sscd_group.values[:, None] == mh.sscd_group.values[None, :])
    neither = (mh.dg_group.values[:, None] != mh.dg_group.values[None, :]) & \
              (mh.sscd_block.values[:, None] != mh.sscd_block.values[None, :])
    np.fill_diagonal(same, False)
    has_mate = same.any(1)
    nn1 = S.argmax(1)
    recall1 = (mh.sscd_group.values[nn1] == mh.sscd_group.values)[has_mate].mean()
    iu = np.triu_indices(len(mh), 1)
    s_same, s_neg = S[iu][same[iu]], S[iu][neither[iu]]
    tau = float(np.quantile(s_neg, 0.999))
    tpr = float((s_same >= tau).mean())
    auc = float((np.sort(s_neg).searchsorted(s_same) / len(s_neg)).mean())  # P(same-pair score > unrelated-pair score)
    md.append(f"## Held-out check ({hold.sum():,} images from {mh.sscd_group.nunique():,} SSCD strict groups unseen in training)\n")
    md.append(f"- Nearest neighbour is in the same strict group for **{recall1:.1%}** of held-out images that have a group-mate.")
    md.append(f"- Pair AUROC (same strict group vs different group in *both* other groupings): **{auc:.3f}**.")
    md.append(f"- Alert threshold tau = {tau:.3f} (99.9th percentile of unrelated pairs). It flags {tpr:.1%} of known same-group pairs.\n")

    # audit each split
    cls = {"original D1 shipped split": "original_split", "SSCD split": "sscd_split",
           "d1_grouped split": "dg_split", "union split": "union_split"}
    THR = (0.80, 0.90, 0.95)
    md.append("## Val+test images whose best train match scores at least t\n")
    md.append("The original shipped D1 split is the reference: it is known to be full of copies. "
              f"The held-out alert threshold tau = {tau:.3f} is too loose to read as 'copy' here, "
              "because the Siamese also scores same-hardware, different-scene crops highly, so a sweep is shown instead.\n")
    md.append("| split | " + " | ".join(f">= {x:.2f}" for x in THR) + " | val-test pairs >= 0.90 |\n|---|" + "---:|" * (len(THR) + 1))
    fig, ax = plt.subplots(1, len(cls), figsize=(4 * len(cls), 3.4), sharey=True)
    zt = torch.from_numpy(z)
    worst = {}
    for k, (name, col) in enumerate(cls.items()):
        sp = m[col].values
        tr = np.flatnonzero(sp == "train")
        row = []
        for s in ["val", "test"]:
            q = np.flatnonzero(sp == s)
            sim = (zt[q] @ zt[tr].T)
            mx, am = sim.max(1)
            row.append((mx.numpy(), q, tr[am.numpy()]))
        vt = (zt[np.flatnonzero(sp == "val")] @ zt[np.flatnonzero(sp == "test")].T)
        both = np.concatenate([row[0][0], row[1][0]])
        md.append(f"| {name} | " + " | ".join(f"{(both >= x).mean():.1%} ({int((both >= x).sum()):,})" for x in THR)
                  + f" | {int((vt >= 0.90).sum()):,} |")
        ax[k].hist(row[0][0], bins=40, alpha=.6, label="val->train"); ax[k].hist(row[1][0], bins=40, alpha=.6, label="test->train")
        ax[k].axvline(0.90, color="k", ls="--", lw=1); ax[k].set_title(name, fontsize=9); ax[k].set_xlabel("max cosine to train")
        worst[name] = row
    ax[0].legend(fontsize=7); ax[0].set_ylabel("images")
    fig.tight_layout(); fig.savefig(OUT / "fig_max_cosine.png", dpi=150); plt.close(fig)

    # contact sheet: strongest val/test -> train pairs in the union split
    sp = m["union_split"].values
    pairs_ = []
    for mx, q, nb in worst["union split"]:
        pairs_ += list(zip(mx, q, nb))
    pairs_ = sorted(pairs_, key=lambda p: -p[0])[:16]
    fig, ax = plt.subplots(4, 8, figsize=(16, 8.4))
    for a in ax.ravel():
        a.axis("off")
    for i, (s, q, n) in enumerate(pairs_):
        r, c = divmod(i, 4)
        ax[r, c * 2].imshow(arr[q]); ax[r, c * 2].set_title(f"{sp[q]} #{q}\ncos {s:.2f}", fontsize=7)
        ax[r, c * 2 + 1].imshow(arr[n]); ax[r, c * 2 + 1].set_title(f"train #{n}", fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "sheet_union_top_cross_split.jpg", dpi=110); plt.close(fig)
    md.append("\n![max cosine](fig_max_cosine.png)\n\nStrongest val/test -> train pairs in the union split: `sheet_union_top_cross_split.jpg`. Dashed line in the histogram: 0.90.")
    (OUT / "README.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    {"train": train, "audit": audit}[sys.argv[1]]()
