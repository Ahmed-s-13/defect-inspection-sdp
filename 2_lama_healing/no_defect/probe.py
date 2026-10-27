"""LaMa-shortcut probe: can a small model tell LaMa-filled patches from untouched patches of the same healed crops?

  python probe.py                 build patches, train the probes, write probe/results.json + probe/*.jpg

If it can, a classifier trained with healed crops as "no defect" can learn "LaMa texture = no defect" instead of
"no damage = no defect". Everything here is trained from scratch on these crops: no enhanced_res.

Patches are cut from the kept healed crops (no_defect_healed.csv), i.e. the 224 px JPEG crops exactly as a
classifier would see them. Per crop: 32 px patches on a 16 px grid, "fill" = fully inside the LaMa mask (eroded 4 px,
so no seam), "real" = fully outside it (dilated 4 px) and within NEAR px of it (same skin, same light), with the
same number of each taken per crop. Probes are trained on train-split crops and scored on val and test.

  colour only   logistic regression on mean / std Lab: does the content differ? (control)
  texture       logistic regression + gradient boosting on hand-made detail / noise / JPEG-grid statistics
  cnn           small CNN from scratch, 3 seeds; once without augmentation, once with online augmentation
                (JPEG 50-95, blur, noise, colour jitter, flips) scored on clean and on augmented patches
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from make_crops import photo_mask, SIZE

HERE = Path(__file__).resolve().parent
OUT = HERE / "probe"
P, STRIDE, EDGE, NEAR, PER_CROP = 32, 16, 4, 48, 8
EPOCHS, SEEDS = 20, (0, 1, 2)
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def crop_mask(c, boxes, shape):
    m = photo_mask(shape, boxes)
    l, t, s = c.crop_left, c.crop_top, c.crop_side
    sub = np.zeros((s, s), np.uint8)                          # the square can run past the photo edge
    h, w = m[t:t + s, l:l + s].shape
    sub[:h, :w] = m[t:t + s, l:l + s]
    return cv2.resize(sub, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0


def patches(rng):
    """-> X uint8 (n, P, P, 3), y (1 = fill), crop id, split, positions."""
    kept = pd.read_csv(HERE / "no_defect_healed.csv")
    crops = pd.read_csv(HERE / "crops.csv").set_index("id")
    bx = pd.read_csv(HERE / "boxes.csv")
    bx["xy"] = bx.box.map(eval)
    hw = pd.read_csv(HERE / "manifest.csv").set_index("key")[["h", "w"]]
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    X, y, ids, sp, pos = [], [], [], [], []
    for cid in kept.crop_path.map(lambda p: Path(p).stem):
        c = crops.loc[cid]
        m = crop_mask(c, list(bx[bx.key == c.key].xy), tuple(hw.loc[c.key])).astype(np.uint8)
        fill = cv2.erode(m, k(EDGE)).astype(bool)
        real = ~cv2.dilate(m, k(EDGE)).astype(bool)
        near = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5) <= NEAR
        img = np.asarray(Image.open(HERE / c.crop).convert("RGB"))
        f, r = [], []
        for yy in range(0, SIZE - P + 1, STRIDE):
            for xx in range(0, SIZE - P + 1, STRIDE):
                win = np.s_[yy:yy + P, xx:xx + P]
                if fill[win].all():
                    f.append((yy, xx))
                elif real[win].all() and near[win].any():
                    r.append((yy, xx))
        n = min(len(f), len(r), PER_CROP)
        for lab, cand in ((1, f), (0, r)):
            for i in rng.choice(len(cand), n, replace=False) if n else []:
                yy, xx = cand[i]
                X.append(img[yy:yy + P, xx:xx + P]); y.append(lab); ids.append(cid); sp.append(c.split); pos.append((yy, xx))
    return np.stack(X), np.array(y), np.array(ids), np.array(sp), np.array(pos)


def colour_feats(X):
    lab = np.stack([cv2.cvtColor(x, cv2.COLOR_RGB2LAB) for x in X]).reshape(len(X), -1, 3).astype(np.float32)
    return np.concatenate([lab.mean(1), lab.std(1)], 1)


def texture_feats(X):
    out = []
    for x in X:
        g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32)
        f = []
        for s in (0.7, 1.5, 3.0):                                   # detail at three scales
            hp = g - cv2.GaussianBlur(g, (0, 0), s)
            f += [np.abs(hp).mean(), hp.std(), np.median(np.abs(hp)), float(((hp - hp.mean()) ** 4).mean() / (hp.var() ** 2 + 1e-6))]
        lap = cv2.Laplacian(g, cv2.CV_32F)
        gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
        dx, dy = np.abs(np.diff(g, axis=1)), np.abs(np.diff(g, axis=0))
        blk = (dx[:, 7::8].mean() + dy[7::8].mean()) / (dx.mean() + dy.mean() + 1e-6)   # 8x8 JPEG grid
        f += [lap.var(), np.hypot(gx, gy).mean(), blk, g.std()]
        out.append(f)
    return np.array(out, np.float32)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        c = lambda i, o: [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU()]
        self.f = nn.Sequential(*c(3, 32), *c(32, 32), nn.MaxPool2d(2), *c(32, 64), *c(64, 64), nn.MaxPool2d(2),
                               *c(64, 128), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 1))

    def forward(self, x):
        return self.f(x).squeeze(1)


def augment(x, rng):
    """The kind of online augmentation a classifier would get; identical for both classes."""
    if rng.random() < 0.5:
        x = x[:, ::-1]
    if rng.random() < 0.5:
        x = x[::-1]
    x = np.ascontiguousarray(x).astype(np.float32)
    x = x * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20) + rng.uniform(-8, 8, 3)
    if rng.random() < 0.5:
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.2))
    if rng.random() < 0.5:
        x = x + rng.normal(0, rng.uniform(1, 6), x.shape)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.7:
        x = cv2.imdecode(cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(50, 96))])[1], cv2.IMREAD_UNCHANGED)
    return x


def tensor(X):
    return torch.from_numpy(np.stack(X)).permute(0, 3, 1, 2).float().div(255).sub(0.5).div(0.25)


def predict(net, X):
    net.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(net(tensor(X[i:i + 1024]).to(DEV))).cpu().numpy() for i in range(0, len(X), 1024)])


def train_cnn(Xtr, ytr, seed, aug):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=EPOCHS * int(np.ceil(len(Xtr) / 128)))
    for _ in range(EPOCHS):
        net.train()
        for b in np.array_split(rng.permutation(len(Xtr)), int(np.ceil(len(Xtr) / 128))):
            xb = [augment(Xtr[i], rng) for i in b] if aug else list(Xtr[b])
            loss = nn.functional.binary_cross_entropy_with_logits(net(tensor(xb).to(DEV)), torch.tensor(ytr[b], dtype=torch.float32, device=DEV))
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net


def heatmaps(net, kept_ids, crops, bx, hw, path):
    """Sliding-window P(fill) over whole test crops, next to the crop with the LaMa area outlined."""
    rows = []
    for cid in kept_ids:
        c = crops.loc[cid]
        img = np.asarray(Image.open(HERE / c.crop).convert("RGB"))
        m = crop_mask(c, list(bx[bx.key == c.key].xy), tuple(hw.loc[c.key])).astype(np.uint8)
        grid = [(yy, xx) for yy in range(0, SIZE - P + 1, 8) for xx in range(0, SIZE - P + 1, 8)]
        p = predict(net, np.stack([img[yy:yy + P, xx:xx + P] for yy, xx in grid]))
        acc, cnt = np.zeros((SIZE, SIZE)), np.zeros((SIZE, SIZE))
        for (yy, xx), v in zip(grid, p):
            acc[yy:yy + P, xx:xx + P] += v; cnt[yy:yy + P, xx:xx + P] += 1
        heat = cv2.applyColorMap((acc / cnt * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1]
        o = img.copy()
        cnt_, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(o, cnt_, -1, (255, 255, 0), 1)
        h2 = (0.45 * img + 0.55 * heat).astype(np.uint8)
        cv2.drawContours(h2, cnt_, -1, (255, 255, 255), 1)
        rows.append(np.concatenate([o, np.full((SIZE, 4, 3), 255, np.uint8), h2, np.full((SIZE, 12, 3), 255, np.uint8)], 1))
    tiles = [np.concatenate(rows[i:i + 3], 1) for i in range(0, len(rows) - len(rows) % 3, 3)]
    Image.fromarray(np.concatenate(tiles, 0)).save(path, quality=88)


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)
    X, y, ids, sp, _ = patches(rng)
    tr, va, te = sp == "train", sp == "val", sp == "test"
    print(f"patches: train {tr.sum()} val {va.sum()} test {te.sum()} (half fill, half real) from {len(set(ids))} crops")
    res = dict(patches=dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum()), crops=len(set(ids))))
    auc = lambda m, p: dict(val=round(roc_auc_score(y[va], p[va]), 3), test=round(roc_auc_score(y[te], p[te]), 3))

    for name, feats, model in (("colour only", colour_feats, lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                               ("texture, logistic", texture_feats, lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                               ("texture, boosting", texture_feats, lambda: HistGradientBoostingClassifier(random_state=0))):
        F = feats(X)
        p = model().fit(F[tr], y[tr]).predict_proba(F)[:, 1]
        res[name] = auc(None, p)
        print(name, res[name])
    F = texture_feats(X)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(F[tr], y[tr])
    names = [f"{s}@{sig}" for sig in (0.7, 1.5, 3.0) for s in ("hp_mean", "hp_std", "hp_median", "hp_kurt")] + ["lap_var", "grad", "jpeg_grid", "std"]
    res["texture means (fill vs real, train)"] = {n: [round(float(F[tr & (y == 1), i].mean()), 3), round(float(F[tr & (y == 0), i].mean()), 3)]
                                                  for i, n in enumerate(names)}
    res["texture, logistic coef"] = dict(zip(names, np.round(lr[-1].coef_[0], 2).tolist()))

    rngA = np.random.default_rng(123)
    Xaug = np.stack([augment(x, rngA) for x in X])                  # one fixed augmented copy, for scoring only
    for aug in (False, True):
        key = "cnn, augmented training" if aug else "cnn, no augmentation"
        runs = []
        for s in SEEDS:
            net = train_cnn(X[tr], y[tr], s, aug)
            p, pa = predict(net, X), predict(net, Xaug)
            runs.append(dict(clean=auc(None, p), augmented=auc(None, pa)))
            print(key, s, runs[-1])
        res[key] = dict(runs=runs, **{f"{e} {k}": round(float(np.mean([r[e][k] for r in runs])), 3)
                                       for e in ("clean", "augmented") for k in ("val", "test")})
        # per-crop view: share of test crops whose fill patches all score above all their real patches
        pc = pd.DataFrame(dict(id=ids, y=y, p=p, sp=sp))[lambda d: d.sp == "test"]
        sep = pc.groupby("id").apply(lambda d: d[d.y == 1].p.min() > d[d.y == 0].p.max(), include_groups=False)
        res[key]["test crops fully separated (last seed)"] = round(float(sep.mean()), 3)
        crops = pd.read_csv(HERE / "crops.csv").set_index("id")
        bx = pd.read_csv(HERE / "boxes.csv"); bx["xy"] = bx.box.map(eval)
        pick = np.random.default_rng(7).choice(sorted(set(ids[te])), 12, replace=False)
        heatmaps(net, pick, crops, bx, pd.read_csv(HERE / "manifest.csv").set_index("key")[["h", "w"]], OUT / f"heatmaps_{'aug' if aug else 'noaug'}.jpg")

    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if "runs" not in str(v)[:8]}, indent=1))


if __name__ == "__main__":
    main()
