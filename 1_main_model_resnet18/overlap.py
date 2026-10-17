"""Step 2: find near-duplicate original images within and across datasets.

Two images are duplicates if phash Hamming distance <= --thr, or if their
original (de-Roboflowed) file stems match and the stem is distinctive.

Outputs:
  <work>/hashes.csv              phash of every original image
  <work>/dup_pairs.csv           every duplicate pair (plus near misses up to thr+3)
  <results>/overlap_matrix.csv   share of images in row X that also appear in column Y
  <results>/overlap_matrix_hash_only.csv, overlap_counts.csv
  <results>/overlap_heatmap.png
  <results>/duplicate_pairs_sheet.png   sample pairs by distance, to check the threshold
"""
import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import imagehash
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from klshift.common import DATASETS, distinctive_stem


def phash_file(path):
    try:
        img = Image.open(path)
        img.draft("RGB", (512, 512))  # fast JPEG decode at reduced size; phash works on 32x32 anyway
        img = ImageOps.exif_transpose(img)
        return int(str(imagehash.phash(img)), 16)
    except Exception:
        return -1


def hamming_block(a, b):
    return np.bitwise_count(a[:, None] ^ b[None, :]).astype(np.uint8)


def find_pairs(h, thr, keep_upto, block=2048):
    """All pairs i<j with Hamming distance <= keep_upto."""
    out = []
    for s in range(0, len(h), block):
        d = hamming_block(h[s:s + block], h)
        i, j = np.nonzero(d <= keep_upto)
        i = i + s
        m = j > i
        out.append(np.stack([i[m], j[m], d[i[m] - s, j[m]]], 1))
    return np.concatenate(out) if out else np.zeros((0, 3), int)


def pair_sheet(pairs, imgs, thr, path, per_bucket=6, seed=0):
    rng = np.random.default_rng(seed)
    buckets = [("hamming 0-1", (pairs.hamming <= 1)),
               ("hamming 2-3", pairs.hamming.between(2, 3)),
               (f"hamming 4-{thr}", pairs.hamming.between(4, thr)),
               (f"near miss {thr + 1}-{thr + 3} (NOT counted)", (pairs.hamming > thr) & ~pairs.name_match),
               ("file-name match only", pairs.name_match & (pairs.hamming > thr))]
    # prefer cross-dataset pairs: they matter most for the table
    rows = []
    for title, m in buckets:
        sub = pairs[m]
        sub = pd.concat([sub[sub.ds_a != sub.ds_b], sub[sub.ds_a == sub.ds_b]])
        if len(sub) > per_bucket:
            cross = sub[sub.ds_a != sub.ds_b]
            pick = cross if len(cross) >= per_bucket else sub
            sub = pick.iloc[rng.choice(len(pick), per_bucket, replace=False)]
        rows += [(title, r) for r in sub.itertuples()]
    if not rows:
        return
    cols = 3  # pairs per row
    nrow = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(nrow, cols * 2, figsize=(cols * 2 * 2.4, nrow * 3.0 + 0.6), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for k, (title, r) in enumerate(rows):
        for side, (ds, im) in enumerate([(r.ds_a, r.image_a), (r.ds_b, r.image_b)]):
            ax = axes[k // cols, (k % cols) * 2 + side]
            try:
                img = Image.open(imgs[(ds, im)])
                img.draft("RGB", (300, 300))
                ax.imshow(ImageOps.exif_transpose(img).convert("RGB"))
            except Exception:
                pass
            ax.set_title(f"{ds}: {Path(im).name[:26]}", fontsize=6)
            ax.set_box_aspect(0.75)
        axes[k // cols, (k % cols) * 2].text(
            0, -0.10, f"{title} | d={r.hamming}{' | name' if r.name_match else ''}",
            transform=axes[k // cols, (k % cols) * 2].transAxes, fontsize=7, color="darkred")
    fig.suptitle(f"Sample image pairs by phash distance (threshold = {thr})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=2.5)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def heatmap(mat, path, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat.values * 100, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(mat.columns)), mat.columns)
    ax.set_yticks(range(len(mat.index)), mat.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j] * 100
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center", fontsize=10,
                    color="white" if v > 55 else "black")
    ax.set_xlabel("...also appears in column dataset")
    ax.set_ylabel("Share of images in row dataset...")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, label="%")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="data/work")
    ap.add_argument("--results", default="results")
    ap.add_argument("--thr", type=int, default=6, help="max phash Hamming distance for a duplicate")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    work, res = Path(a.work), Path(a.results)
    res.mkdir(parents=True, exist_ok=True)

    dss = [d for d in DATASETS if (work / "images" / f"{d}.csv").exists()]
    im = pd.concat([pd.read_csv(work / "images" / f"{d}.csv") for d in dss], ignore_index=True)
    print(f"hashing {len(im)} original images ...")
    with ProcessPoolExecutor(a.workers) as ex:
        hashes = list(ex.map(phash_file, im.path, chunksize=64))
    im["phash"] = [f"{h:016x}" if h >= 0 else "" for h in hashes]
    bad = im.phash == ""
    if bad.any():
        print(f"  {bad.sum()} images could not be hashed; dropped")
    im = im[~bad].reset_index(drop=True)
    im[["dataset", "image", "orig_stem", "phash"]].to_csv(work / "hashes.csv", index=False)
    h = np.array([int(x, 16) for x in im.phash], dtype=np.uint64)

    # --- pairs: hash distance, plus file-name matches
    hp = find_pairs(h, a.thr, a.thr + 3)
    pairs = {(i, j): d for i, j, d in hp}
    by_stem = defaultdict(list)
    for idx, s in enumerate(im.orig_stem.astype(str)):
        if distinctive_stem(s):
            by_stem[s].append(idx)
    name_pairs = set()
    for idx in by_stem.values():
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                name_pairs.add((idx[x], idx[y]))
    keys = set(pairs) | name_pairs
    rows = []
    for i, j in keys:
        d = pairs.get((i, j))
        if d is None:
            d = int(np.bitwise_count(h[i] ^ h[j]))
        rows.append((i, j, int(d), (i, j) in name_pairs))
    P = pd.DataFrame(rows, columns=["i", "j", "hamming", "name_match"])
    P["dup"] = (P.hamming <= a.thr) | P.name_match
    for side in "ab":
        col = P.i if side == "a" else P.j
        P[f"ds_{side}"] = im.dataset.values[col]
        P[f"image_{side}"] = im.image.values[col]
    P = P.sort_values(["hamming", "ds_a", "ds_b"]).reset_index(drop=True)
    cols = ["ds_a", "image_a", "ds_b", "image_b", "hamming", "name_match", "dup"]
    P[cols].to_csv(work / "dup_pairs.csv", index=False)
    D = P[P.dup]
    print(f"duplicate pairs: {len(D)} (hash {int((D.hamming <= a.thr).sum())}, "
          f"name only {int((D.hamming > a.thr).sum())})")

    # --- overlap matrix: share of images in X with at least one duplicate in Y
    def matrix(dpairs):
        hit = {(x, y): set() for x in dss for y in dss}
        for r in dpairs.itertuples():
            hit[(r.ds_a, r.ds_b)].add(r.i)
            hit[(r.ds_b, r.ds_a)].add(r.j)
        n = im.dataset.value_counts()
        cnt = pd.DataFrame([[len(hit[(x, y)]) for y in dss] for x in dss], index=dss, columns=dss)
        return cnt.div(n[dss], axis=0), cnt

    share, counts = matrix(D)
    share_h, _ = matrix(D[D.hamming <= a.thr])
    share.round(4).to_csv(res / "overlap_matrix.csv")
    share_h.round(4).to_csv(res / "overlap_matrix_hash_only.csv")
    counts.assign(n_images=im.dataset.value_counts()[dss]).to_csv(res / "overlap_counts.csv")
    heatmap(share, res / "overlap_heatmap.png",
            f"Image overlap (phash d<={a.thr} or same file name)\n"
            "diagonal = images with a near-copy inside the same dataset")
    imgs = {(r.dataset, r.image): r.path for r in im.itertuples()}
    pair_sheet(P, imgs, a.thr, res / "duplicate_pairs_sheet.png")

    pd.set_option("display.width", 200)
    print("\nOverlap: share of images in ROW dataset that also appear in COLUMN dataset")
    print((share * 100).round(1).to_string())
    print("\nCounts (images):")
    print(counts.assign(n_images=im.dataset.value_counts()[dss]).to_string())


if __name__ == "__main__":
    main()
