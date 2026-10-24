"""Step 1: convert each YOLO dataset in <raw>/<DS>/ into square crops + metadata CSVs.

Crop modes:
  box         (D1-D9)  one square per labelled box, box + 25% margin, shifted inward
  patch       (D6p-D9p) no-damage skin patches: random squares from the middle of each aircraft box,
                         side = half the box's shorter side
  background  (D1nd)   D1 areas away from every defect box (box + 25% kept clear), sized like
                         that image's own defect crops

Outputs (in <work>):
  crops/<DS>/*.jpg     one 224x224 crop per box (box + 25% margin, shifted inward)
  crops/<DS>.csv       one row per crop
  images/<DS>.csv      one row per original image (used by overlap.py)
  dataset_stats.csv    sizes, resolution and aspect ratio per dataset
"""
import argparse
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageOps

from klshift.common import (CROP_SIZE, DATASETS, DERIVED, IMG_EXTS, MARGIN, original_stem,
                            photo_stem, square_crop_box)

N_PATCHES = 2      # patches per aircraft box (patch) or per image (background)
MIN_PATCH = 32     # px; smaller patches are skipped

SPLIT_NAMES = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def read_class_names(root):
    ymls = sorted(root.rglob("data.yaml")) + sorted(root.rglob("*.yaml"))
    for y in ymls:
        names = (yaml.safe_load(y.read_text()) or {}).get("names")
        if names:
            return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    txt = sorted(root.rglob("classes.txt"))
    if txt:
        return [l.strip() for l in txt[0].read_text().splitlines() if l.strip()]
    raise SystemExit(f"No class names (data.yaml / classes.txt) found under {root}")


def label_path(img_path):
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
    return img_path.with_suffix(".txt")


def parse_labels(path, w, h):
    """YOLO boxes (or polygons) -> list of (cls, x1, y1, x2, y2) in pixels."""
    if not path.exists():
        return []
    boxes = []
    for line in path.read_text().splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        c, nums = int(float(v[0])), np.array(v[1:], dtype=float)
        if len(nums) == 4:
            cx, cy, bw, bh = nums
            x1, y1, x2, y2 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
        else:  # polygon x1 y1 x2 y2 ...
            xs, ys = nums[0::2], nums[1::2]
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        x1, x2 = np.clip([x1, x2], 0, 1) * w
        y1, y2 = np.clip([y1, y2], 0, 1) * h
        boxes.append((c, x1, y1, x2, y2))
    return boxes


def clamp_square(cx, cy, side, w, h):
    side = int(round(min(side, w, h)))
    left = min(max(int(round(cx - side / 2)), 0), w - side)
    top = min(max(int(round(cy - side / 2)), 0), h - side)
    return left, top, side


def regions(mode, boxes, w, h, rng):
    """-> list of (class index or None, reference box, (left, top, side))."""
    if mode == "box":
        return [(c, (x1, y1, x2, y2), square_crop_box(x1, y1, x2, y2, w, h)) for c, x1, y1, x2, y2 in boxes]
    out = []
    if mode == "patch":
        for c, x1, y1, x2, y2 in boxes:
            bw, bh = x2 - x1, y2 - y1
            side = 0.5 * min(bw, bh)
            if side < MIN_PATCH:
                continue
            for _ in range(N_PATCHES):  # middle of the box, where the fuselage usually is
                cx = rng.uniform(x1 + 0.2 * bw, x2 - 0.2 * bw)
                cy = rng.uniform(y1 + 0.3 * bh, y2 - 0.3 * bh)
                out.append((None, (x1, y1, x2, y2), clamp_square(cx, cy, side, w, h)))
        return out
    if mode == "background":
        if not boxes:
            return []
        keep_out = [(x1 - MARGIN * (x2 - x1), y1 - MARGIN * (y2 - y1), x2 + MARGIN * (x2 - x1), y2 + MARGIN * (y2 - y1))
                    for _, x1, y1, x2, y2 in boxes]
        sides = [square_crop_box(x1, y1, x2, y2, w, h)[2] for _, x1, y1, x2, y2 in boxes]
        for _ in range(50):
            if len(out) == N_PATCHES:
                break
            side = int(min(rng.choice(sides), w, h))
            if side < MIN_PATCH:
                continue
            left, top = int(rng.integers(0, w - side + 1)), int(rng.integers(0, h - side + 1))
            r = (left, top, left + side, top + side)
            if any(r[0] < b[2] and b[0] < r[2] and r[1] < b[3] and b[1] < r[3]
                   for b in keep_out + [o[1] for o in out]):
                continue
            out.append((None, r, (left, top, side)))
        return out
    raise ValueError(mode)


def process_image(args):
    ds, root, img_path, out_dir, names, min_box, mode = args
    rel = str(img_path.relative_to(root))
    split = next((SPLIT_NAMES[p.lower()] for p in img_path.parts if p.lower() in SPLIT_NAMES), "")
    try:
        img = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
    except Exception as e:  # unreadable file
        return None, [], f"{rel}: {e}"
    w, h = img.size
    img_row = dict(dataset=ds, image=rel, path=str(img_path), name=img_path.name,
                   orig_stem=original_stem(img_path.name), photo_stem=photo_stem(img_path.name),
                   img_w=w, img_h=h, shipped_split=split, n_boxes=0, n_small=0)
    crops, boxes = [], []
    for c, x1, y1, x2, y2 in parse_labels(label_path(img_path), w, h):
        if x2 - x1 < min_box or y2 - y1 < min_box or c >= len(names):
            img_row["n_small"] += 1
        else:
            boxes.append((c, x1, y1, x2, y2))
    rng = np.random.default_rng(zlib.crc32(rel.encode()))
    src = root.name
    for j, (c, (x1, y1, x2, y2), (left, top, side)) in enumerate(regions(mode, boxes, w, h, rng)):
        crop = img.crop((left, top, left + side, top + side)).resize(
            (CROP_SIZE, CROP_SIZE), Image.BICUBIC)
        out = out_dir / f"{Path(rel).with_suffix('').as_posix().replace('/', '__')}__{j}.jpg"
        crop.save(out, quality=95)
        crops.append(dict(crop_path=str(out), dataset=ds, source=src,
                          cls=names[c] if c is not None else "no-damage", image=rel,
                          src_name=img_path.name, orig_stem=img_row["orig_stem"],
                          group=f"{src}:{img_row['photo_stem']}",
                          x1=round(x1, 1), y1=round(y1, 1), x2=round(x2, 1), y2=round(y2, 1),
                          crop_left=left, crop_top=top, crop_side=side,
                          img_w=w, img_h=h, shipped_split=split))
        img_row["n_boxes"] += 1
    return img_row, crops, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--work", default="data/work")
    ap.add_argument("--datasets", nargs="+", default=DATASETS + list(DERIVED),
                    help=f"raw datasets and/or derived crop sets ({', '.join(DERIVED)})")
    ap.add_argument("--min-box", type=float, default=4, help="skip boxes smaller than this (px)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    work = Path(a.work)
    if (work / "d1_split.csv").exists():
        raise SystemExit("This work directory already has a frozen split. Prepare crops in a new version directory.")
    (work / "images").mkdir(parents=True, exist_ok=True)
    stats = []
    for ds in a.datasets:
        src, mode = DERIVED.get(ds, (ds, "box"))
        if mode == "cropper":
            print(f"[{ds}] made by run_cropper.py (the Cropper), skipped here")
            continue
        root = Path(a.raw) / src
        if not root.exists():
            print(f"[{ds}] missing {root}, skipped")
            continue
        names = read_class_names(root)
        out_dir = work / "crops" / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        imgs = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS
                      and "labels" not in p.parts)
        jobs = [(ds, root, p, out_dir, names, a.min_box, mode) for p in imgs]
        img_rows, crop_rows, errors = [], [], []
        with ProcessPoolExecutor(a.workers) as ex:
            for r, c, err in ex.map(process_image, jobs, chunksize=32):
                if err:
                    errors.append(err)
                else:
                    img_rows.append(r)
                    crop_rows.extend(c)
        im, cr = pd.DataFrame(img_rows), pd.DataFrame(crop_rows)
        if mode == "box":  # derived sets reuse their source's images (and its overlap results)
            im.to_csv(work / "images" / f"{ds}.csv", index=False)
        cr.to_csv(work / "crops" / f"{ds}.csv", index=False)
        aspect = im.img_w / im.img_h
        stats.append(dict(
            dataset=ds, classes="|".join(names) if mode == "box" else f"{mode} crops from {src}",
            images=len(im), crops=len(cr),
            images_without_boxes=int((im.n_boxes == 0).sum()), skipped_small_boxes=int(im.n_small.sum()),
            unreadable=len(errors), median_w=int(im.img_w.median()), median_h=int(im.img_h.median()),
            min_side=int(np.minimum(im.img_w, im.img_h).min()), max_side=int(np.maximum(im.img_w, im.img_h).max()),
            distinct_sizes=int(im.groupby(["img_w", "img_h"]).ngroups),
            most_common_size="{}x{}".format(*im.groupby(["img_w", "img_h"]).size().idxmax()),
            median_aspect=round(float(aspect.median()), 3),
            square_share=round(float((im.img_w == im.img_h).mean()), 3),
            median_crop_side=int(cr.crop_side.median()) if len(cr) else 0,
            shipped_splits=im.shipped_split.value_counts().to_dict()))
        print(f"[{ds}] {len(im)} images, {len(cr)} crops, {len(errors)} unreadable, "
              f"classes: {cr.cls.value_counts().to_dict() if len(cr) else {}}")
        for e in errors[:5]:
            print("   unreadable:", e)
    if stats:
        out = work / "dataset_stats.csv"
        old = pd.read_csv(out) if out.exists() else pd.DataFrame()
        new = pd.DataFrame(stats)
        if len(old):
            new = pd.concat([old[~old.dataset.isin(new.dataset)], new]).sort_values("dataset")
        new.to_csv(out, index=False)
        print(new.drop(columns=["classes"]).to_string(index=False))


if __name__ == "__main__":
    main()
