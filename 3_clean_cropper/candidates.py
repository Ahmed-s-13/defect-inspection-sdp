"""Step 1 of clean_cropper: cut candidate "no defect" squares from the parts of D1 photos outside every labelled box.

  python candidates.py            -> candidates/<split>/<id>.jpg, candidates.csv

Per labelled D1 photo, up to N_PER_PHOTO random squares that stay clear of every labelled box (each box grown by
KEEP_OUT of its size on every side, boxes too small for a defect crop included) and of each other. A square's side
is drawn from that photo's own defect crop sides (box + 25% margin, as the defect crops are cut), so candidates have
the defect crops' scale; it is resized to 224 px exactly as they are (bicubic, JPEG 95). Squares with any real
amount of black (rotation corners of D1's augmented copies) are skipped.

These are only candidates: many are background or hold damage nobody boxed. score.py and the review page sort them
out. Every candidate keeps its photo's d1_grouped split and group.
"""
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MAIN = REPO / "1_main_model_resnet18"
WORK = HERE / "work"                                             # git-ignored: candidate crops, caches, thumbnails
sys.path.insert(0, str(REPO / "2_lama_healing"))
from hcore import RAW, d1_label_file, read_yolo                   # noqa: E402

GROUPS = MAIN / "d1_grouped/groups.csv"
N_PER_PHOTO = 3
KEEP_OUT = 0.5     # box grown by 50% of its size per side
MARGIN = 0.25      # defect crop margin (klshift.common.MARGIN)
MIN_SIDE = 32
MAX_BLACK = 0.02   # share of near-black pixels allowed
SIZE = 224


def one(job):
    key, split, group = job
    f = d1_label_file(key)
    if not f.exists():
        return []
    img = ImageOps.exif_transpose(Image.open(RAW / "images" / key)).convert("RGB")
    w, h = img.size
    boxes = [b[1:] for b in read_yolo(f, w, h)]
    if not boxes:
        return []
    keep = [(x1 - KEEP_OUT * (x2 - x1), y1 - KEEP_OUT * (y2 - y1), x2 + KEEP_OUT * (x2 - x1), y2 + KEEP_OUT * (y2 - y1))
            for x1, y1, x2, y2 in boxes]
    sides = [int(round(min(max(x2 - x1, y2 - y1) * (1 + 2 * MARGIN), w, h))) for x1, y1, x2, y2 in boxes]
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    out, taken = [], []
    for _ in range(200):
        if len(out) == N_PER_PHOTO:
            break
        side = int(rng.choice(sides))
        if side < MIN_SIDE:
            continue
        left, top = int(rng.integers(0, w - side + 1)), int(rng.integers(0, h - side + 1))
        r = (left, top, left + side, top + side)
        if any(r[0] < b[2] and b[0] < r[2] and r[1] < b[3] and b[1] < r[3] for b in keep + taken):
            continue
        crop = img.crop(r)
        if (np.asarray(crop).max(2) < 12).mean() > MAX_BLACK:
            continue
        taken.append(r)
        cid = f"{key.replace('/', '__')[:-4]}__c{len(out)}"
        dst = WORK / "candidates" / split / f"{cid}.jpg"
        crop.resize((SIZE, SIZE), Image.BICUBIC).save(dst, quality=95)
        out.append(dict(id=cid, split=split, group=group, key=key, left=left, top=top, side=side,
                        crop=str(dst.relative_to(HERE))))
    return out


def main():
    g = pd.read_csv(GROUPS)
    for s in ("train", "val", "test"):
        (WORK / "candidates" / s).mkdir(parents=True, exist_ok=True)
    jobs = list(zip(g.key, g.new_split, g.group))
    with ProcessPoolExecutor(8) as ex:
        df = pd.DataFrame([r for rs in ex.map(one, jobs, chunksize=64) for r in rs])
    df.to_csv(HERE / "candidates.csv", index=False)
    print(f"{len(jobs)} photos -> {len(df)} candidates")
    print(df.groupby("split").agg(candidates=("id", "size"), photos=("key", "nunique"), groups=("group", "nunique")).to_string())


if __name__ == "__main__":
    main()
