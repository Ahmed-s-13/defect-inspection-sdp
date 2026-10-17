"""Step 2: train the MSRD teacher-student model (3_teacher_student_msrd/msrd_anomaly.py, architecture
unchanged) on D1-train clean patches, evaluate it as a Stage 1 baseline, and filter
the clean set (drop the most anomalous 5% per split).

Only data loading and wrappers are added here; model, losses and anomaly map come
from msrd_anomaly.py. Patches are fed at 224 px (the shift task's crop size).

Outputs:
  work/ts/msrd_<backbone>_best.pt, work/ts/history.csv
  work/clean_D1_scores.csv                       ts_score + dropped flag for every clean patch
  results/healing/step2_ts_baseline.md/.json, step2_*.png
"""
import argparse
import json
import time

import hcommon as H

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

import msrd_anomaly as M

TS_DIR = H.WORK / "ts"


class PatchDS(Dataset):
    """Returns (image in [0,1], box mask, label, name), the tuple msrd_anomaly expects."""

    def __init__(self, paths, boxes=None, labels=None, names=None):
        self.paths = list(paths)
        self.boxes = None if boxes is None else np.asarray(boxes, float)
        self.labels = np.zeros(len(self.paths), int) if labels is None else np.asarray(labels)
        self.names = [""] * len(self.paths) if names is None else list(names)
        self.tf = T.ToTensor()  # crops are already 224x224; MSRD normalizes inside

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.tf(Image.open(self.paths[i]).convert("RGB"))
        mask = torch.zeros(1, *img.shape[-2:])
        if self.boxes is not None and not np.isnan(self.boxes[i]).any():
            x1, y1, x2, y2 = self.boxes[i]
            mask[:, int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] = 1
        return img, mask, int(self.labels[i]), self.names[i]


def load_ts(ckpt, device):
    state = torch.load(ckpt, map_location="cpu")
    model = M.MSRD(state["args"]["backbone"]).to(device)
    model.load_state_dict(state["model"], strict=False)  # teacher comes from torchvision
    return model.eval()


@torch.no_grad()
def amaps(model, img, sigma=4.0):
    """Smoothed anomaly maps (B,H,W) exactly as msrd_anomaly.evaluate builds them; img in [0,1]."""
    t, s, _ = model(img)
    m = M.pyramid_anomaly_map(t, s, img.shape[-2:]).squeeze(1).cpu().numpy()
    return np.stack([gaussian_filter(x, sigma=sigma) for x in m])


@torch.no_grad()
def score(model, ds, device, batch=64, workers=4, map_size=None):
    """Image scores (max of map); optionally maps average-pooled to map_size and masks."""
    model.eval()
    dl = DataLoader(ds, batch_size=batch, num_workers=workers)
    sc, mp, mk = [], [], []
    for img, mask, _, _ in dl:
        m = amaps(model, img.to(device))
        sc.append(m.reshape(len(m), -1).max(1))
        if map_size:
            mp.append(F.adaptive_avg_pool2d(torch.from_numpy(m)[:, None], map_size)[:, 0].numpy())
            mk.append(F.adaptive_avg_pool2d(mask, map_size)[:, 0].numpy())
    sc = np.concatenate(sc)
    return (sc, np.concatenate(mp), np.concatenate(mk)) if map_size else sc


def spec_at_recall(y, s, r):
    """Specificity on clean patches when the threshold catches share r of defects."""
    return 1 - M.fpr_at_recall(np.asarray(y), np.asarray(s), r)[0]


def train(a, device, clean, defects):
    tr = clean[clean.split == "train"]
    rng = np.random.default_rng(a.seed)
    vc = clean[clean.split == "val"].sample(a.val_n, random_state=a.seed)
    vd = defects[defects.split == "val"].sample(a.val_n, random_state=a.seed)
    val_ds = PatchDS(list(vc.path) + list(vd.path), labels=[0] * len(vc) + [1] * len(vd))
    g = torch.Generator().manual_seed(a.seed)
    dl = DataLoader(PatchDS(tr.path), batch_size=a.batch, shuffle=True, drop_last=True,
                    num_workers=a.workers, persistent_workers=True, generator=g)
    model = M.MSRD(a.backbone).to(device)
    n_par = sum(p.numel() for m in model.trainable_modules() for p in m.parameters())
    print(f"teacher {a.backbone}, trainable {n_par / 1e6:.1f} M, {len(tr)} train clean patches, "
          f"{len(dl)} steps/epoch", flush=True)
    params = [p for m in model.trainable_modules() for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=a.lr, betas=(0.5, 0.999))
    hist, best = [], -1
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, skd, srec, n = time.time(), 0.0, 0.0, 0
        for img, _, _, _ in dl:
            img = img.to(device)
            t, s, rec = model(img)
            l_kd, l_rec = M.kd_loss(t, s), 1 - M.ssim(rec, img)
            loss = l_kd + a.lambda_recon * l_rec
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            skd += l_kd.item(); srec += l_rec.item(); n += 1
            if a.max_steps and n >= a.max_steps:
                break
        dt = time.time() - t0
        sc = score(model, val_ds, device, workers=a.workers)
        auc = roc_auc_score(val_ds.labels, sc)
        hist.append(dict(epoch=ep, l_kd=skd / n, l_recon=srec / n, val_auroc=auc, seconds=dt))
        print(f"epoch {ep:2d} | L_KD {skd / n:.4f} | L_Recon {srec / n:.4f} | val AUROC {auc:.4f} "
              f"| {dt:.0f}s", flush=True)
        state = {"model": {k: v for k, v in model.state_dict().items() if not k.startswith("teacher.")},
                 "args": {**vars(a), "img_size": H.CROP_SIZE}, "epoch": ep, "val_auroc": auc}
        if auc > best:
            best = auc
            torch.save(state, TS_DIR / f"msrd_{a.backbone}_best.pt")
        pd.DataFrame(hist).to_csv(TS_DIR / "history.csv", index=False)
    return pd.DataFrame(hist)


def evaluate_split(model, device, clean, defects, split, a):
    c, d = clean[clean.split == split], defects[defects.split == split]
    boxes = np.vstack([np.full((len(c), 4), np.nan), d[["bx1", "by1", "bx2", "by2"]].values])
    ds = PatchDS(list(c.path) + list(d.path), boxes, [0] * len(c) + [1] * len(d),
                 ["clean"] * len(c) + list(d.cls))
    sc, mp, mk = score(model, ds, device, workers=a.workers, map_size=56)
    y = ds.labels
    r = {"n_clean": len(c), "n_defect": len(d), "image_AUROC": roc_auc_score(y, sc),
         "spec_at_99_recall": spec_at_recall(y, sc, 0.99),
         "spec_at_95_recall": spec_at_recall(y, sc, 0.95)}
    r["per_class_AUROC"] = {k: roc_auc_score(np.r_[np.zeros(len(c)), np.ones((d.cls == k).sum())],
                                             np.r_[sc[:len(c)], sc[len(c):][(d.cls == k).values]])
                            for k in sorted(d.cls.unique())}
    # Pixel level, rough (boxes are not masks). Maps and masks average-pooled to 56x56.
    inbox = mk > 0.5
    r["pixel_AUROC_all"] = roc_auc_score(inbox.ravel(), mp.ravel())
    dm, dk = mp[len(c):], inbox[len(c):]
    has = dk.any((1, 2)) & ~dk.all((1, 2))
    r["pixel_AUROC_within_defect_crops"] = roc_auc_score(dk[has].ravel(), dm[has].ravel())
    flat = dm.reshape(len(dm), -1)
    r["pointing_game"] = float(dk.reshape(len(dk), -1)[np.arange(len(dk)), flat.argmax(1)].mean())
    r["map_mass_in_box"] = float(((dm * dk).sum((1, 2)) / dm.sum((1, 2))).mean())
    r["box_area_share"] = float(dk.mean((1, 2)).mean())
    return r, sc, ds


def vis(model, device, ds, sc, path, n_each=6, seed=0):
    rng = np.random.default_rng(seed)
    y = ds.labels
    idx = np.r_[rng.choice(np.flatnonzero(y == 0), n_each, replace=False),
                rng.choice(np.flatnonzero(y == 1), n_each, replace=False)]
    img = torch.stack([ds[i][0] for i in idx])
    m = amaps(model, img.to(device))
    vmax = np.percentile(sc, 99)
    fig, ax = plt.subplots(2, len(idx), figsize=(1.9 * len(idx), 4.2))
    for j, i in enumerate(idx):
        ax[0, j].imshow(img[j].permute(1, 2, 0))
        if y[i]:
            x1, y1, x2, y2 = ds.boxes[i]
            ax[0, j].add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="lime", lw=1.5))
        ax[0, j].set_title(f"{ds.names[i]}\n{sc[i]:.3f}", fontsize=8)
        ax[1, j].imshow(m[j], cmap="jet", vmin=0, vmax=vmax)
        for k in (0, 1):
            ax[k, j].axis("off")
    fig.suptitle("Teacher-student anomaly maps on D1-test (score = max of map; green = annotated box)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def filter_clean(model, device, a):
    clean = H.d1_clean(filtered=False)
    sc = score(model, PatchDS(clean.path), device, workers=a.workers)
    clean["ts_score"] = sc
    clean["dropped"] = False
    for s, g in clean.groupby("split"):
        thr = np.quantile(g.ts_score, 1 - a.drop)
        clean.loc[g.index, "dropped"] = g.ts_score > thr
    clean[["crop_path", "split", "ts_score", "dropped"]].to_csv(H.WORK / "clean_D1_scores.csv", index=False)
    dr = clean[clean.dropped].sort_values("ts_score", ascending=False)
    pick = dr.iloc[np.linspace(0, len(dr) - 1, 64).round().astype(int)]
    fig, ax = plt.subplots(8, 8, figsize=(14, 15))
    for axx, r in zip(ax.ravel(), pick.itertuples()):
        axx.imshow(Image.open(r.path))
        axx.set_title(f"{r.split} {r.side}px  {r.ts_score:.3f}", fontsize=7)
        axx.axis("off")
    fig.suptitle(f"Dropped clean patches (top {a.drop:.0%} TS score per split): 64 evenly spaced "
                 f"from most to least anomalous, of {len(dr)}")
    fig.tight_layout()
    fig.savefig(H.RES / "step2_dropped_clean.png", dpi=110)
    plt.close(fig)
    return clean


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="resnet18", choices=list(M.BACKBONES))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16, help="msrd_anomaly default")
    ap.add_argument("--lr", type=float, default=0.005, help="msrd_anomaly default")
    ap.add_argument("--lambda-recon", type=float, default=0.9, help="msrd_anomaly default")
    ap.add_argument("--val-n", type=int, default=1000, help="clean + defect val patches each, for picking")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: steps per epoch")
    ap.add_argument("--drop", type=float, default=0.05, help="share of clean patches dropped per split")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    M.set_seed(a.seed)
    device = H.pick_device()
    TS_DIR.mkdir(parents=True, exist_ok=True)
    H.RES.mkdir(parents=True, exist_ok=True)
    clean, defects = H.d1_clean(filtered=False), H.d1_defects()
    if not a.eval_only:
        train(a, device, clean, defects)
    if a.max_steps:
        return
    ck = TS_DIR / f"msrd_{a.backbone}_best.pt"
    model = load_ts(ck, device)
    st = torch.load(ck, map_location="cpu")
    hist = pd.read_csv(TS_DIR / "history.csv")

    # Baseline on the UNFILTERED clean set (the filter uses this model's own scores).
    res = {"checkpoint": str(ck.relative_to(H.HERE)), "best_epoch": st["epoch"], "backbone": a.backbone}
    for s in ["val", "test"]:
        res[s], sc, ds = evaluate_split(model, device, clean, defects, s, a)
        if s == "test":
            vis(model, device, ds, sc, H.RES / "step2_anomaly_maps.png", seed=a.seed)
            fig, ax = plt.subplots(figsize=(7, 4))
            bins = np.linspace(sc.min(), np.percentile(sc, 99.5), 50)
            ax.hist(sc[ds.labels == 0], bins, alpha=.6, density=True, label="real clean")
            ax.hist(sc[ds.labels == 1], bins, alpha=.6, density=True, label="defect crops")
            ax.set(xlabel="teacher-student score (max of anomaly map)", ylabel="density",
                   title="D1-test image scores")
            ax.legend()
            fig.tight_layout()
            fig.savefig(H.RES / "step2_score_hist.png", dpi=150)
            plt.close(fig)
    # Then filter the clean set and re-score test on the filtered clean set (for reference).
    filt = filter_clean(model, device, a)
    res["test_filtered_clean"], _, _ = evaluate_split(model, device, filt[~filt.dropped], defects, "test", a)
    res["dropped_per_split"] = filt[filt.dropped].split.value_counts().to_dict()
    (H.RES / "step2_ts_baseline.json").write_text(json.dumps(res, indent=2, default=float))

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(hist.epoch, hist.l_kd, "o-", label="L_KD")
    ax[0].plot(hist.epoch, hist.l_recon, "o-", label="L_Recon (1-SSIM)")
    ax[0].set(xlabel="epoch", title="Training loss")
    ax[0].legend()
    ax[1].plot(hist.epoch, hist.val_auroc, "o-")
    ax[1].axvline(st["epoch"], color="gray", ls="--", lw=1)
    ax[1].set(xlabel="epoch", title=f"D1-val image AUROC ({a.val_n}+{a.val_n} patches)")
    fig.tight_layout()
    fig.savefig(H.RES / "step2_training_curves.png", dpi=150)
    plt.close(fig)
    write_md(res, hist, a)


def write_md(res, hist, a):
    v, t, tf = res["val"], res["test"], res["test_filtered_clean"]
    rows = [("image AUROC (clean vs defect)", "image_AUROC"),
            ("specificity @ 99% defect recall", "spec_at_99_recall"),
            ("specificity @ 95% defect recall", "spec_at_95_recall"),
            ("pixel AUROC, box vs rest (all patches)", "pixel_AUROC_all"),
            ("pixel AUROC, box vs rest (inside defect crops)", "pixel_AUROC_within_defect_crops"),
            ("pointing game (map max inside box)", "pointing_game"),
            ("share of map mass inside box", "map_mass_in_box"),
            ("share of crop area that is box (chance level)", "box_area_share")]
    L = ["# Step 2: teacher-student (MSRD) baseline\n",
         f"`msrd_anomaly.py` unchanged; teacher `{a.backbone}` (ImageNet, frozen), 224 px, batch {a.batch}, "
         f"Adam lr {a.lr}, c = {a.lambda_recon}, seed {a.seed}. Trained on "
         f"D1-train clean patches only (unfiltered), {len(hist)} epochs at about {hist.seconds.median() / 60:.1f} min/epoch; "
         f"checkpoint picked by D1-val image AUROC (epoch {res['best_epoch']}).\n",
         "| metric | D1-val | D1-test | D1-test, filtered clean* |", "|---|---|---|---|"]
    L += [f"| {n} | {v[k]:.3f} | {t[k]:.3f} | {tf[k]:.3f} |" for n, k in rows]
    L += [f"| n clean / n defect | {v['n_clean']} / {v['n_defect']} | {t['n_clean']} / {t['n_defect']} "
          f"| {tf['n_clean']} / {tf['n_defect']} |",
          "\n*The filter drops the clean patches this same model scores highest, so this column "
          "flatters the baseline. Use the unfiltered column as the baseline.\n",
          "| defect class | D1-test AUROC vs clean |", "|---|---|"]
    L += [f"| {k} | {x:.3f} |" for k, x in t["per_class_AUROC"].items()]
    L += ["\nPixel metrics are rough: boxes are not masks, and maps/masks are average-pooled to 56x56.\n",
          f"Clean-set filter: dropped the top {a.drop:.0%} per split by TS score "
          f"({res['dropped_per_split']}); see `step2_dropped_clean.png`.\n",
          "![curves](step2_training_curves.png)\n", "![hist](step2_score_hist.png)\n",
          "![maps](step2_anomaly_maps.png)\n", "![dropped](step2_dropped_clean.png)\n"]
    (H.RES / "step2_ts_baseline.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
