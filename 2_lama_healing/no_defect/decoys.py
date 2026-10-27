"""Decoy inpainting: LaMa-fill a clean part of every D1 defect crop, so LaMa fill is not a "no defect" cue.

  python decoys.py                 every D1 photo (resumes: photos already in decoys.csv are skipped)
  python decoys.py --limit 20      first 20 photos only (test)

The probe (probe.py) showed a model can tell LaMa fill from real skin (it is smoother). The healed "no defect" crops
are mostly LaMa fill, so a classifier could learn "smooth LaMa texture = no defect". A decoy crop is a real defect crop
with a rectangle of clean skin LaMa-filled the same way: the defect is untouched and the label stays the same. With
decoys in the defect classes, LaMa fill shows up in every class and stops predicting one.

Per photo: every labelled box grown by 12% (the healer's mask) is off limits. For each defect crop of the photo,
1 to MAX_RECTS rectangles (uniform) are picked inside the crop square, fully outside the off-limits area and each
outside the ones already picked: rectangles on a 32 x 32 grid of the square with sides of at least 4 cells (1/8 of
the crop), drawn with probability ~ area^2, so mostly large. Several rectangles push the fill towards the healed
crops' share (median 52% of the crop), though a defect crop can't go far past ~30%: the defect sits in its middle.
All the photo's rectangles are filled in one LaMa pass on the full photo (exactly as the healed photos were made),
then every crop is cut at its D1.csv square (bicubic to 224, JPEG 95). A crop whose square has no room gets no decoy.

Outputs (this folder): decoys/crops/<split>/<id>.jpg, decoys.csv (one row per crop, `has_decoy`, `decoy_frac` =
share of the crop that is LaMa fill), decoys/run.log. Split = the photo's d1_grouped split. No trained model is used
(no enhanced_res).
"""
import argparse
import csv
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
HEALER = HERE.parent                                             # 2_lama_healing
sys.path.insert(0, str(HEALER))
from hcore import RAW, d1_label_file, read_yolo, defect_mask, load_lama, lama   # noqa: E402

MAIN = HERE.parents[1] / "1_main_model_resnet18"
GROUPS = MAIN / "d1_grouped/groups.csv"
D1_CROPS = MAIN / "data/work/crops/D1.csv"
OUT = HERE / "work/decoys"                                       # git-ignored
CSV = HERE / "decoys.csv"
COLS = ["id", "split", "group", "key", "cls", "crop", "source_crop", "has_decoy", "decoy_frac", "rect", "secs"]
SIZE, GRID, MIN_CELLS, MAX_RECTS = 224, 32, 4, 6


def pick_rect(free, rng):
    """free: GRID x GRID bool (cell fully clean) -> (r0, r1, c0, c1) cells, end-exclusive, or None."""
    ii = np.pad(np.cumsum(np.cumsum(~free, 0), 1), ((1, 0), (1, 0)))          # integral image of blocked cells
    a = np.arange(GRID + 1)
    lo, hi = np.meshgrid(a, a, indexing="ij")
    ok = hi - lo >= MIN_CELLS
    r0, r1 = lo[ok], hi[ok]
    R0, C0 = np.meshgrid(r0, r0, indexing="ij")
    R1, C1 = np.meshgrid(r1, r1, indexing="ij")
    blocked = ii[R1, C1] - ii[R0, C1] - ii[R1, C0] + ii[R0, C0]
    area = ((R1 - R0) * (C1 - C0)).astype(float)
    w = np.where(blocked == 0, area ** 2, 0).ravel()
    if w.sum() == 0:
        return None
    i = rng.choice(len(w), p=w / w.sum())
    return R0.flat[i], R1.flat[i], C0.flat[i], C1.flat[i]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="only the first N photos (test)")
    a = ap.parse_args()

    d1 = pd.read_csv(D1_CROPS)
    d1["key"] = d1.image.str.replace("content/Aircraft_dataset/images/", "", regex=False)
    g = pd.read_csv(GROUPS).set_index("key")
    d1 = d1.join(g[["new_split", "group"]].rename(columns={"group": "grp"}), on="key")
    assert d1.new_split.notna().all()
    done = set(pd.read_csv(CSV).key) if CSV.exists() else set()
    keys = [k for k in sorted(d1.key.unique()) if k not in done][:a.limit]

    OUT.mkdir(exist_ok=True)
    live = open(OUT / "run.log", "a", encoding="utf-8", buffering=1)

    def say(s):
        print(s, flush=True)
        live.write(s + "\n")

    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = load_lama(dev)
    say(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}: {d1.key.nunique()} photos, {len(done)} done, {len(keys)} to do, {dev}")
    new = not CSV.exists()
    f = open(CSV, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, COLS)
    if new:
        w.writeheader()
    t_all = time.time()
    for n, key in enumerate(keys, 1):
        t0 = time.time()
        rng = np.random.default_rng(zlib.crc32(key.encode()))          # same rectangles on a re-run
        img = np.asarray(ImageOps.exif_transpose(Image.open(RAW / "images" / key)).convert("RGB")).copy()
        H, W = img.shape[:2]
        boxes = read_yolo(d1_label_file(key), W, H)
        blocked = defect_mask(img.shape, boxes) > 0
        cs = d1[d1.key == key]
        rects, fill = {}, np.zeros((H, W), bool)
        for c in cs.itertuples():
            l, t, s = int(c.crop_left), int(c.crop_top), int(c.crop_side)
            edges = np.linspace(0, s, GRID + 1).round().astype(int)
            sq = np.ones((s, s), bool)                                   # outside the photo counts as blocked
            sub = blocked[max(t, 0):t + s, max(l, 0):l + s]
            sq[max(-t, 0):max(-t, 0) + sub.shape[0], max(-l, 0):max(-l, 0) + sub.shape[1]] = sub
            ii = np.pad(np.cumsum(np.cumsum(sq, 0), 1), ((1, 0), (1, 0)))[edges][:, edges]   # blocked px per cell
            cells = ii[1:, 1:] - ii[:-1, 1:] - ii[1:, :-1] + ii[:-1, :-1]
            free, mine = cells == 0, []
            for _ in range(rng.integers(1, MAX_RECTS + 1)):
                r = pick_rect(free, rng)
                if r is None:
                    break
                free[r[0]:r[1], r[2]:r[3]] = False
                y0, y1, x0, x1 = t + edges[r[0]], t + edges[r[1]], l + edges[r[2]], l + edges[r[3]]
                mine.append((int(x0), int(y0), int(x1), int(y1)))
                fill[y0:y1, x0:x1] = True
            if mine:
                rects[c.Index] = mine
        assert not (fill & blocked).any(), key
        out = lama(model, img, fill.astype(np.uint8) * 255, dev) if fill.any() else img
        out = np.where(fill[..., None], out, img)                       # only the decoy pixels change
        secs = time.time() - t0
        for c in cs.itertuples():
            l, t, s = int(c.crop_left), int(c.crop_top), int(c.crop_side)
            cid = Path(c.crop_path).stem.replace("content__Aircraft_dataset__images__", "")
            frac = float(fill[max(t, 0):t + s, max(l, 0):l + s].sum() / (s * s))
            dst = OUT / "crops" / c.new_split / f"{cid}.jpg"
            if frac > 0:
                dst.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(out).crop((l, t, l + s, t + s)).resize((SIZE, SIZE), Image.BICUBIC).save(dst, quality=95)
            w.writerow(dict(id=cid, split=c.new_split, group=c.grp, key=key, cls=c.cls,
                            crop=str(dst.relative_to(HERE)) if frac > 0 else "", source_crop=c.crop_path,
                            has_decoy=frac > 0, decoy_frac=round(frac, 3), rect=rects.get(c.Index, ""),
                            secs=round(secs, 2)))
        f.flush()
        if n % 50 == 0 or n == len(keys):
            eta = (time.time() - t_all) / n * (len(keys) - n)
            say(f"[{n}/{len(keys)}] {key}: {len(cs)} crop(s), {len(rects)} decoy(s) | {secs:.2f}s | ETA {eta / 60:.0f} min")
    f.close()
    say(f"done: {len(keys)} photos in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
