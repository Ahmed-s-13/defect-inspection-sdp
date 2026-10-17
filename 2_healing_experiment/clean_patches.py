"""Step 1: sample real clean patches from D1 regions that touch no annotated box.

Patch sizes follow the defect-crop sizes of the same split (one target per defect
crop), so patch scale is not a class cue. Each patch is cut exactly like a defect
crop (square, inside the image, resized to 224 BICUBIC, JPEG q95).

Outputs:
  work/clean/D1/*.jpg                  224x224 clean patches
  work/clean_D1.csv                    crop_path, image, group_id, split, left, top, side, img_w, img_h
  results/healing/step1_clean_stats.md, step1_sizes.png, step1_examples.png
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import hcommon as H  # noqa: F401  (sets sys.path)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.stats import ks_2samp

from prepare import label_path, parse_labels  # shift task's YOLO parser (keeps <4 px boxes here)


def forbidden_rects(path, w, h, margin, pad):
    """Every annotated box (including tiny ones), enlarged by margin*size + pad px."""
    r = []
    for _, x1, y1, x2, y2 in parse_labels(label_path(Path(path)), w, h):
        mx, my = margin * (x2 - x1) + pad, margin * (y2 - y1) + pad
        r.append((x1 - mx, y1 - my, x2 + mx, y2 + my))
    return np.array(r, float).reshape(-1, 4)


def try_place(s, w, h, rects, rng, tries):
    """Random top-left positions for an s x s square; return the first that hits no rect."""
    x = rng.integers(0, w - s + 1, tries)
    y = rng.integers(0, h - s + 1, tries)
    if len(rects):
        hit = ((x[:, None] < rects[:, 2]) & (x[:, None] + s > rects[:, 0]) &
               (y[:, None] < rects[:, 3]) & (y[:, None] + s > rects[:, 1])).any(1)
        free = np.flatnonzero(~hit)
        if not len(free):
            return None
        x, y = x[free], y[free]
    return int(x[0]), int(y[0])


def plan_split(imgs, targets, rng, a):
    """Assign each target side to an image + position. Largest sides first (hardest to fit)."""
    rects = {r.image: r.rects for r in imgs.itertuples()}
    count = dict.fromkeys(imgs.image, 0)
    short = np.minimum(imgs.img_w.values, imgs.img_h.values)
    names, W, Hh = imgs.image.values, imgs.img_w.values, imgs.img_h.values
    out, missed = [], []
    for s in sorted(targets, reverse=True):
        cand = np.flatnonzero(short >= s)
        rng.shuffle(cand)
        placed = False
        for i in cand[:a.max_images]:
            im = names[i]
            if count[im] >= a.per_image:
                continue
            pos = try_place(s, W[i], Hh[i], rects[im], rng, a.tries)
            if pos is None:
                continue
            x, y = pos
            out.append(dict(image=im, left=x, top=y, side=int(s), img_w=int(W[i]), img_h=int(Hh[i])))
            rects[im] = np.vstack([rects[im], [x, y, x + s, y + s]])  # patches of one image don't overlap
            count[im] += 1
            placed = True
            break
        if not placed:
            missed.append(int(s))
    return pd.DataFrame(out), missed


def cut(job):
    path, rows, out_dir = job
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    paths = []
    for k, (left, top, side, stem) in enumerate(rows):
        out = out_dir / f"{stem}__c{k}.jpg"
        img.crop((left, top, left + side, top + side)).resize(
            (H.CROP_SIZE, H.CROP_SIZE), Image.BICUBIC).save(out, quality=95)
        paths.append(str(out.relative_to(H.HERE)))
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--margin", type=float, default=0.25, help="box enlargement, fraction of box size")
    ap.add_argument("--pad", type=float, default=8, help="extra px around every box")
    ap.add_argument("--per-image", type=int, default=3)
    ap.add_argument("--tries", type=int, default=30, help="positions tried per image")
    ap.add_argument("--max-images", type=int, default=400, help="images tried per target before giving up")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    H.seed_everything(a.seed)
    rng = np.random.default_rng(a.seed)
    H.RES.mkdir(parents=True, exist_ok=True)
    out_dir = H.WORK / "clean" / "D1"
    out_dir.mkdir(parents=True, exist_ok=True)

    crops = pd.read_csv(H.KL / "data/work/crops/D1.csv")
    split = pd.read_csv(H.KL / "data/work/d1_split.csv")[["crop_path", "group_id", "split"]]
    crops = crops.merge(split, on="crop_path")
    img_split = crops.drop_duplicates("image").set_index("image")[["group_id", "split"]]
    imgs = pd.read_csv(H.KL / "data/work/images/D1.csv")
    imgs = imgs[imgs.image.isin(img_split.index)].reset_index(drop=True)  # 145 box-less images have no split
    imgs = imgs.join(img_split, on="image")
    imgs["rects"] = [forbidden_rects(H.kl_path(p), w, h, a.margin, a.pad)
                     for p, w, h in zip(imgs.path, imgs.img_w, imgs.img_h)]

    plans, missed = [], {}
    for s in ["train", "val", "test"]:
        P, m = plan_split(imgs[imgs.split == s], crops.loc[crops.split == s, "crop_side"].values, rng, a)
        P["split"] = s
        plans.append(P)
        missed[s] = m
        print(f"{s}: placed {len(P)} / {len(P) + len(m)} target sizes", flush=True)
    P = pd.concat(plans, ignore_index=True)
    P = P.join(img_split[["group_id"]], on="image")
    P["stem"] = [Path(i).with_suffix("").as_posix().replace("/", "__") for i in P.image]

    paths = imgs.set_index("image").path
    jobs, order = [], []
    for im, g in P.groupby("image", sort=False):
        jobs.append((H.kl_path(paths[im]), list(zip(g.left, g.top, g.side, g.stem)), out_dir))
        order.append(g.index)
    P["crop_path"] = ""
    with ProcessPoolExecutor(a.workers) as ex:
        for idx, ps in zip(order, ex.map(cut, jobs, chunksize=16)):
            P.loc[idx, "crop_path"] = ps
    cols = ["crop_path", "image", "group_id", "split", "left", "top", "side", "img_w", "img_h"]
    P[cols].to_csv(H.WORK / "clean_D1.csv", index=False)
    report(P, crops, missed, imgs, a, rng)


def report(P, crops, missed, imgs, a, rng):
    lines = ["# Step 1: real clean-patch set from D1\n",
             f"Patches avoid every annotated box (incl. boxes < 4 px) enlarged by {a.margin:.0%} of its "
             f"size + {a.pad:.0f} px; at most {a.per_image} non-overlapping patches per image; "
             f"seed {a.seed}. Sizes are drawn from the same split's defect-crop sizes.\n",
             "| split | defect crops | clean patches | shortfall | images used / available | "
             "median side clean / defect | KS stat (side) |", "|---|---|---|---|---|---|---|"]
    for s in ["train", "val", "test"]:
        c, d = P[P.split == s], crops[crops.split == s]
        ks = ks_2samp(c.side, d.crop_side).statistic
        lines.append(f"| {s} | {len(d)} | {len(c)} | {len(missed[s])} | {c.image.nunique()} / "
                     f"{(imgs.split == s).sum()} | {c.side.median():.0f} / {d.crop_side.median():.0f} | {ks:.3f} |")
    allm = np.concatenate([np.array(m, int) for m in missed.values()]) if any(missed.values()) else np.array([], int)
    if len(allm):
        lines.append(f"\nTargets that could not be placed: {len(allm)}, side px min {allm.min()}, "
                     f"median {np.median(allm):.0f}, max {allm.max()}. These are large crops "
                     "(often the full short side of the image), where no box-free square exists.")
    q = np.percentile(crops.crop_side, [10, 25, 50, 75, 90, 95, 99]).round()
    qc = np.percentile(P.side, [10, 25, 50, 75, 90, 95, 99]).round()
    lines += ["\n| percentile | 10 | 25 | 50 | 75 | 90 | 95 | 99 |", "|---|---|---|---|---|---|---|---|",
              "| defect crop side (px) | " + " | ".join(f"{v:.0f}" for v in q) + " |",
              "| clean patch side (px) | " + " | ".join(f"{v:.0f}" for v in qc) + " |",
              "\n![sizes](step1_sizes.png)\n", "![examples](step1_examples.png)\n",
              "Label noise: background regions can still hold unannotated damage. These patches are "
              "scored by the teacher-student model after Step 2 and the top 5% are dropped "
              "(grid saved then)."]
    (H.RES / "step1_clean_stats.md").write_text("\n".join(lines))
    print("\n".join(lines[2:]))

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.geomspace(6, 1100, 40)
    ax.hist(crops.crop_side, bins, alpha=.55, label=f"defect crops (n={len(crops)})", density=True)
    ax.hist(P.side, bins, alpha=.55, label=f"clean patches (n={len(P)})", density=True)
    ax.set_xscale("log")
    ax.set(xlabel="square side in the original image (px, before resize to 224)", ylabel="density",
           title="D1 patch-size distribution: defect crops vs clean patches")
    ax.legend()
    fig.tight_layout()
    fig.savefig(H.RES / "step1_sizes.png", dpi=150)
    plt.close(fig)

    ex = P.sample(48, random_state=a.seed)
    fig, axs = plt.subplots(6, 8, figsize=(14, 11))
    for axx, r in zip(axs.ravel(), ex.itertuples()):
        axx.imshow(Image.open(H.HERE / r.crop_path))
        axx.set_title(f"{r.split} {r.side}px", fontsize=8)
        axx.axis("off")
    fig.suptitle("48 random clean patches (before the teacher-student filter)")
    fig.tight_layout()
    fig.savefig(H.RES / "step1_examples.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
