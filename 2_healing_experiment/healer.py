"""Step 3: box-guided inpainting healer.

A small U-Net takes (patch with a hole, hole mask) and fills the hole. It is trained
on D1-train real clean patches with random holes whose geometry is taken from real
D1-train defect boxes (rectangles, or irregular strokes inside such a box). Loss on
the hole plus a border band: L1 + teacher feature loss (frozen ImageNet ResNet-18 =
the MSRD teacher, 1 - cosine on layer1-3) + small total variation. No adversarial loss.

Healing (offline): hole = annotated box + MARGIN px, plus a FEATHER px band; the
inpainted pixels are pasted back with alpha 1 on box+margin, ramping to 0 across the
band, so pixels outside the hole are bit-identical to the original.

  python healer.py --train            train, pick on D1-val clean, curves + grids
  python healer.py --heal             write healed / random-region-inpainted crops
"""
import argparse
import time

import hcommon as H

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import msrd_anomaly as M

S = H.CROP_SIZE
MARGIN, FEATHER = 8, 6  # px at 224 scale
HEAL_DIR = H.WORK / "healer"
NORM = (torch.tensor(H.MEAN).view(1, 3, 1, 1), torch.tensor(H.STD).view(1, 3, 1, 1))


# ------------------------------------------------------------------ masks
def enlarge(box, px):
    x1, y1, x2, y2 = box
    return (max(0, x1 - px), max(0, y1 - px), min(S, x2 + px), min(S, y2 + px))


def rect_mask(box):
    m = np.zeros((S, S), np.float32)
    x1, y1, x2, y2 = box
    m[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] = 1
    return m


def feather_alpha(box):
    """1 on box+MARGIN, linear ramp to 0 over FEATHER px (Euclidean), 0 beyond."""
    x1, y1, x2, y2 = enlarge(box, MARGIN)
    yy, xx = np.mgrid[0:S, 0:S] + 0.5
    dx = np.maximum.reduce([x1 - xx, np.zeros_like(xx), xx - x2])
    dy = np.maximum.reduce([y1 - yy, np.zeros_like(yy), yy - y2])
    return np.clip(1 - np.hypot(dx, dy) / FEATHER, 0, 1).astype(np.float32)


def irregular_mask(box, rng):
    """Random thick strokes and ellipses, clipped to the box: same footprint, not a rectangle."""
    x1, y1, x2, y2 = box
    im = Image.new("L", (S, S), 0)
    d = ImageDraw.Draw(im)
    w, h = max(x2 - x1, 4), max(y2 - y1, 4)
    for _ in range(rng.integers(3, 8)):
        px, py = rng.uniform(x1, x2), rng.uniform(y1, y2)
        if rng.random() < 0.5:
            rx, ry = rng.uniform(0.15, 0.45) * w, rng.uniform(0.15, 0.45) * h
            d.ellipse((px - rx, py - ry, px + rx, py + ry), fill=255)
        else:
            qx, qy = rng.uniform(x1, x2), rng.uniform(y1, y2)
            d.line((px, py, qx, qy), fill=255, width=int(max(3, rng.uniform(0.1, 0.35) * min(w, h))))
    return (np.asarray(im) > 0).astype(np.float32) * rect_mask(box)


def hole_for_box(box):
    """Hole used for a real defect box at healing time (box + margin + feather band)."""
    return enlarge(box, MARGIN + FEATHER)


# ------------------------------------------------------------------ data
class HoleDS(Dataset):
    """Clean patch + random hole. Hole geometry = a real D1-train defect box, randomly shifted."""

    def __init__(self, paths, boxes, train, seed, p_irregular=0.5):
        self.paths, self.boxes, self.train = list(paths), np.asarray(boxes), train
        self.seed, self.p_irr = seed, p_irregular
        self.tf = T.ToTensor()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # Fixed holes for validation; fresh ones every epoch for training.
        # Training: seeded from the DataLoader worker's torch RNG (reproducible via its generator).
        rng = np.random.default_rng([self.seed, i, int(torch.randint(1 << 30, (1,)))] if self.train
                                    else [self.seed, i])
        img = Image.open(self.paths[i]).convert("RGB")
        if self.train:
            if rng.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if rng.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
        x1, y1, x2, y2 = self.boxes[rng.integers(len(self.boxes))]
        w, h = x2 - x1, y2 - y1
        dx, dy = rng.uniform(-x1, S - x2), rng.uniform(-y1, S - y2)  # any position inside the crop
        if rng.random() < 0.5:                                        # or keep the real position
            dx = dy = 0
        box = hole_for_box((x1 + dx, y1 + dy, x1 + dx + w, y1 + dy + h))
        m = irregular_mask(box, rng) if rng.random() < self.p_irr else rect_mask(box)
        if m.sum() < 16:
            m = rect_mask(box)
        return self.tf(img), torch.from_numpy(m)[None]


# ------------------------------------------------------------------ network
def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, 1, 1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Small U-Net: 4 -> 3 channels, 224 -> 14 -> 224."""

    def __init__(self, ch=(32, 64, 128, 256, 256)):
        super().__init__()
        self.down = nn.ModuleList()
        i = 4
        for c in ch:
            self.down.append(nn.Sequential(cbr(i, c), cbr(c, c)))
            i = c
        self.up = nn.ModuleList()
        for c_skip in reversed(ch[:-1]):
            self.up.append(nn.Sequential(cbr(i + c_skip, c_skip), cbr(c_skip, c_skip)))
            i = c_skip
        self.out = nn.Conv2d(i, 3, 1)

    def forward(self, img, mask):
        x = torch.cat([img * (1 - mask) + 0.5 * mask, mask], 1)  # hole filled with grey
        skips = []
        for k, d in enumerate(self.down):
            x = d(x if k == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for u, s in zip(self.up, reversed(skips[:-1])):
            x = u(torch.cat([F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1))
        return torch.sigmoid(self.out(x))


class TeacherLoss(nn.Module):
    """Frozen MSRD teacher (ImageNet ResNet-18, layer1-3). 1 - cosine, weighted by the loss region."""

    def __init__(self):
        super().__init__()
        self.t = M.Teacher("resnet18").eval()
        self.register_buffer("mean", NORM[0])
        self.register_buffer("std", NORM[1])

    def forward(self, x, y, w):
        fx = self.t((x - self.mean) / self.std)
        with torch.no_grad():
            fy = self.t((y - self.mean) / self.std)
        loss = 0
        for a, b in zip(fx, fy):
            ws = F.adaptive_max_pool2d(w, a.shape[-2:])
            loss = loss + ((1 - F.cosine_similarity(a, b, dim=1, eps=1e-6))[:, None] * ws).sum() / ws.sum().clamp_min(1)
        return loss


def loss_region(mask, border=8):
    return F.max_pool2d(mask, 2 * border + 1, 1, border)  # hole + border band


def losses(net, tl, img, mask, a):
    out = net(img, mask)
    comp = out * mask + img * (1 - mask)
    w = loss_region(mask)
    l1 = ((comp - img).abs() * w).sum() / (3 * w.sum()).clamp_min(1)
    feat = tl(comp, img, w)
    tv = (((comp[..., 1:, :] - comp[..., :-1, :]).abs() * w[..., 1:, :]).sum() +
          ((comp[..., :, 1:] - comp[..., :, :-1]).abs() * w[..., :, 1:]).sum()) / (3 * w.sum()).clamp_min(1)
    total = a.w_l1 * l1 + a.w_feat * feat + a.w_tv * tv
    return total, dict(l1=l1.item(), feat=feat.item(), tv=tv.item(), total=total.item()), comp


# ------------------------------------------------------------------ healing
@torch.no_grad()
def heal_batch(net, img, boxes, device):
    """img (B,3,S,S) in [0,1]; boxes (B,4) defect boxes at 224 scale. Returns healed images."""
    holes = torch.from_numpy(np.stack([rect_mask(hole_for_box(b)) for b in boxes]))[:, None]
    alpha = torch.from_numpy(np.stack([feather_alpha(b) for b in boxes]))[:, None]
    out = net(img.to(device), holes.to(device)).cpu()
    return alpha * out + (1 - alpha) * img, holes


def random_region(defect_box, boxes_pool, rng, tries=50):
    """A hole geometry from the defect-box pool placed so its hole does not touch the
    (enlarged) defect hole. Used for 'random region inpainted' controls. None if no room."""
    avoid = None if defect_box is None else hole_for_box(defect_box)
    for _ in range(tries):
        x1, y1, x2, y2 = boxes_pool[rng.integers(len(boxes_pool))]
        w, h = x2 - x1, y2 - y1
        if w + 2 * (MARGIN + FEATHER) >= S or h + 2 * (MARGIN + FEATHER) >= S:
            continue
        nx = rng.uniform(MARGIN + FEATHER, S - w - MARGIN - FEATHER)
        ny = rng.uniform(MARGIN + FEATHER, S - h - MARGIN - FEATHER)
        b = (nx, ny, nx + w, ny + h)
        hb = hole_for_box(b)
        if avoid is None or (hb[2] <= avoid[0] or hb[0] >= avoid[2] or hb[3] <= avoid[1] or hb[1] >= avoid[3]):
            return b
    return None


def load_healer(device):
    st = torch.load(HEAL_DIR / "healer_best.pt", map_location="cpu")
    net = UNet().to(device)
    net.load_state_dict(st["model"])
    return net.eval()


# ------------------------------------------------------------------ training
def train(a, device):
    clean, dfx = H.d1_clean(), H.d1_defects()
    box_pool = dfx.loc[dfx.split == "train", ["bx1", "by1", "bx2", "by2"]].values
    tr, va = clean[clean.split == "train"], clean[clean.split == "val"].sample(a.val_n, random_state=a.seed)
    g = torch.Generator().manual_seed(a.seed)
    dl = DataLoader(HoleDS(tr.path, box_pool, True, a.seed), batch_size=a.batch, shuffle=True,
                    drop_last=True, num_workers=a.workers, persistent_workers=True, generator=g)
    dv = DataLoader(HoleDS(va.path, box_pool, False, a.seed), batch_size=a.batch, num_workers=a.workers)
    net, tl = UNet().to(device), TeacherLoss().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, betas=(0.5, 0.999))
    print(f"U-Net {sum(p.numel() for p in net.parameters()) / 1e6:.1f} M params, {len(tr)} train patches, "
          f"{len(dl)} steps/epoch", flush=True)
    hist, best = [], 1e9
    for ep in range(1, a.epochs + 1):
        net.train()
        t0, acc, n = time.time(), {}, 0
        for img, mask in dl:
            loss, parts, _ = losses(net, tl, img.to(device), mask.to(device), a)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for k, v in parts.items():
                acc[k] = acc.get(k, 0) + v
            n += 1
            if a.max_steps and n >= a.max_steps:
                break
        net.eval()
        vacc, vn = {}, 0
        with torch.no_grad():
            for img, mask in dv:
                _, parts, _ = losses(net, tl, img.to(device), mask.to(device), a)
                for k, v in parts.items():
                    vacc[k] = vacc.get(k, 0) + v
                vn += 1
        row = dict(epoch=ep, seconds=time.time() - t0, **{f"train_{k}": v / n for k, v in acc.items()},
                   **{f"val_{k}": v / vn for k, v in vacc.items()})
        hist.append(row)
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
        if row["val_total"] < best:
            best = row["val_total"]
            torch.save({"model": net.state_dict(), "args": vars(a), "epoch": ep}, HEAL_DIR / "healer_best.pt")
        pd.DataFrame(hist).to_csv(HEAL_DIR / "history.csv", index=False)


def figures(a, device):
    net = load_healer(device)
    st = torch.load(HEAL_DIR / "healer_best.pt", map_location="cpu")
    hist = pd.read_csv(HEAL_DIR / "history.csv")
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for axx, k in zip(ax, ["l1", "feat", "tv"]):
        axx.plot(hist.epoch, hist[f"train_{k}"], "o-", label="train")
        axx.plot(hist.epoch, hist[f"val_{k}"], "o-", label="val (fixed holes)")
        axx.axvline(st["epoch"], color="gray", ls="--", lw=1)
        axx.set(xlabel="epoch", title={"l1": "masked L1", "feat": "teacher feature loss (1-cos)",
                                        "tv": "total variation"}[k])
        axx.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(H.RES / "step3_healer_curves.png", dpi=150)
    plt.close(fig)

    # (a) val clean patches with a hole: ground truth is known.
    clean, dfx = H.d1_clean(), H.d1_defects()
    pool = dfx.loc[dfx.split == "train", ["bx1", "by1", "bx2", "by2"]].values
    va = clean[clean.split == "val"].sample(8, random_state=a.seed + 1)
    ds = HoleDS(va.path, pool, False, a.seed + 1)
    img, mask = map(torch.stack, zip(*[ds[i] for i in range(len(ds))]))
    with torch.no_grad():
        out = net(img.to(device), mask.to(device)).cpu()
    comp = out * mask + img * (1 - mask)
    grid(H.RES / "step3_val_clean_inpaint.png", [img, mask, comp, (comp - img).abs() * 3],
         ["real clean", "hole", "inpainted", "|error| x3"],
         "Healer on D1-val clean patches with a random hole (ground truth known)")

    # (b) val defect crops: the real use.
    rng = np.random.default_rng(a.seed)
    rows = pd.concat([g.sample(2, random_state=a.seed) for _, g in dfx[dfx.split == "val"].groupby("cls")])
    img = torch.stack([T.ToTensor()(Image.open(p).convert("RGB")) for p in rows.path])
    healed, holes = heal_batch(net, img, rows[["bx1", "by1", "bx2", "by2"]].values, device)
    grid(H.RES / "step3_val_defect_heal.png", [img, holes, healed],
         ["original", "hole (box+margin+feather)", "healed"],
         "Healer on D1-val defect crops (2 per class)", names=list(rows.cls))


def grid(path, cols, titles, sup, names=None):
    n = cols[0].shape[0]
    fig, ax = plt.subplots(len(cols), n, figsize=(1.9 * n, 1.95 * len(cols) + 0.4))
    for r, (c, t) in enumerate(zip(cols, titles)):
        for j in range(n):
            x = c[j]
            ax[r, j].imshow(x[0] if x.shape[0] == 1 else x.permute(1, 2, 0).clamp(0, 1),
                            cmap="gray" if x.shape[0] == 1 else None, vmin=0, vmax=1)
            ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
        ax[r, 0].set_ylabel(t, fontsize=8)
    if names:
        for j, nm in enumerate(names):
            ax[0, j].set_title(nm, fontsize=8)
    fig.suptitle(sup)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------ generation
def save_jpg(t, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(path, quality=95)


def heal_all(a, device):
    """Writes, with the same JPEG q95 as every other crop:
       healed/<DS>/   defect crop with its box healed (D1 all splits, D2, D4, D5)
       ctrl_defect/D1 defect crop with a random NON-defect region inpainted (C-artifact-matched)
       ctrl_clean/D1  clean patch with a random region inpainted (artifact probe)"""
    net = load_healer(device)
    rng = np.random.default_rng(a.seed)
    d1 = H.d1_defects()
    pool = d1.loc[d1.split == "train", ["bx1", "by1", "bx2", "by2"]].values
    tf = T.ToTensor()
    jobs = []
    for ds in ["D1", "D2", "D4", "D5"]:
        d = d1 if ds == "D1" else H.defects(ds)
        jobs.append(("healed", ds, d.crop_path, d.path, d[["bx1", "by1", "bx2", "by2"]].values))
    ctrl = [random_region(b, pool, rng) for b in d1[["bx1", "by1", "bx2", "by2"]].values]
    ok = np.array([c is not None for c in ctrl])
    jobs.append(("ctrl_defect", "D1", d1.crop_path[ok], d1.path[ok], np.array([c for c in ctrl if c is not None])))
    cl = H.d1_clean()
    jobs.append(("ctrl_clean", "D1", cl.crop_path, cl.path,
                 np.array([random_region(None, pool, rng) for _ in range(len(cl))])))
    index = []
    for kind, ds, keys, paths, boxes in jobs:
        keys, paths = list(keys), list(paths)
        t0 = time.time()
        for s in range(0, len(paths), a.batch):
            img = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[s:s + a.batch]])
            healed, _ = heal_batch(net, img, boxes[s:s + a.batch], device)
            for k, p, b, h in zip(keys[s:s + a.batch], paths[s:s + a.batch], boxes[s:s + a.batch], healed):
                out = H.WORK / kind / ds / (str(p).split("/")[-1])
                save_jpg(h, out)
                index.append(dict(kind=kind, dataset=ds, src_crop_path=k, path=str(out.relative_to(H.HERE)),
                                  hx1=b[0], hy1=b[1], hx2=b[2], hy2=b[3]))
        print(f"{kind}/{ds}: {len(paths)} crops in {time.time() - t0:.0f}s"
              + ("" if kind != "ctrl_defect" else f" ({(~ok).sum()} defect crops had no room for a "
                 "non-overlapping region and were skipped)"), flush=True)
    pd.DataFrame(index).to_csv(H.WORK / "generated.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--heal", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--w-l1", type=float, default=1.0)
    ap.add_argument("--w-feat", type=float, default=0.5)
    ap.add_argument("--w-tv", type=float, default=0.05)
    ap.add_argument("--val-n", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: steps per epoch")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    H.seed_everything(a.seed)
    device = H.pick_device()
    HEAL_DIR.mkdir(parents=True, exist_ok=True)
    H.RES.mkdir(parents=True, exist_ok=True)
    if a.train:
        train(a, device)
    if a.figures or a.train:
        figures(a, device)
    if a.heal:
        heal_all(a, device)


if __name__ == "__main__":
    main()
