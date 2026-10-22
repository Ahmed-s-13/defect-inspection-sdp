"""Draw the D1 label boxes on the full photos, sorted by split and class, for browsing by eye.

  python visualize_d1.py              -> data/d1_visual/<train|val|test>/<class>/<photo>.jpg

The split is the model's own leak-free split (data/work/d1_split.csv), not the folders D1 ships with.
Boxes come from D1's YOLO label files, read with the same code as prepare.py.

A photo goes into the folder of every class it has a box of, so a photo with a crack and a dent
appears under both. In each copy, boxes of that folder's class are drawn thick and the photo's
other boxes thin, each in its class colour with its name. Boxes under 4 px (skipped by prepare.py,
not in training) are drawn dashed in grey. Photos without any usable box (145) have no split and
are left out. counts.csv lists photos and boxes per split and class.
"""
import argparse
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps

from prepare import label_path, parse_labels, read_class_names

COLORS = {"crack": (230, 25, 75), "dent": (0, 130, 200), "corrosion": (245, 130, 48),
          "scratch": (145, 30, 180), "missing-head": (60, 180, 75), "paint-peel-off": (240, 50, 230)}
SKIPPED = (160, 160, 160)


def dashed_rect(d, box, color, width=1, dash=4):
    x1, y1, x2, y2 = box
    for a, b, horiz in [((x1, y1), x2, True), ((x1, y2), x2, True), ((x1, y1), y2, False), ((x2, y1), y2, False)]:
        start, end = (a[0], b) if horiz else (a[1], b)
        for s in range(int(start), int(end), dash * 2):
            e = min(s + dash, end)
            d.line([(s, a[1]), (e, a[1])] if horiz else [(a[0], s), (a[0], e)], fill=color, width=width)


def draw(job):
    """job = (photo path, out paths per class, class names, min box). Writes one annotated copy per class."""
    path, outs, names, min_box = job
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")  # same orientation as prepare.py
    w, h = img.size
    boxes = parse_labels(label_path(Path(path)), w, h)
    scale = max(w, h) / 640
    font = ImageFont.load_default(size=max(12, int(14 * scale)))
    lw_main, lw_other = max(3, int(3 * scale)), max(1, int(1.5 * scale))
    for cls, out in outs.items():
        im = img.copy()
        d = ImageDraw.Draw(im)
        # other classes first, this folder's class on top
        for c, x1, y1, x2, y2 in sorted(boxes, key=lambda b: names[b[0]] == cls if b[0] < len(names) else False):
            if c >= len(names) or x2 - x1 < min_box or y2 - y1 < min_box:
                dashed_rect(d, (x1, y1, x2, y2), SKIPPED)
                continue
            name = names[c]
            col, main = COLORS.get(name, (255, 255, 0)), name == cls
            d.rectangle((x1, y1, x2, y2), outline=col, width=lw_main if main else lw_other)
            tx, ty = x1, max(0, y1 - font.size - 4)
            tb = d.textbbox((tx, ty), name, font=font)
            d.rectangle((tb[0] - 2, tb[1] - 1, tb[2] + 2, tb[3] + 1), fill=col if main else (40, 40, 40))
            d.text((tx, ty), name, fill="white" if main else col, font=font)
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out, quality=90)
    return len(outs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw/D1")
    ap.add_argument("--work", default="data/work")
    ap.add_argument("--out", default="data/d1_visual")
    ap.add_argument("--min-box", type=float, default=4, help="same as prepare.py: smaller boxes were not used")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    names = read_class_names(raw)
    split = pd.read_csv(Path(a.work) / "d1_split.csv")
    photos = split.groupby("image").agg(split=("split", "first"), classes=("cls", lambda c: sorted(set(c))))
    assert split.groupby("image").split.nunique().max() == 1

    jobs, counts = [], Counter()
    for img, r in photos.iterrows():
        src = raw / img
        if not src.exists():
            raise SystemExit(f"missing {src}: unzip D1_raw.zip in the repo root")
        stem = Path(img).stem
        jobs.append((str(src), {c: out / r.split / c / f"{stem}.jpg" for c in r.classes}, names, a.min_box))
        for c in r.classes:
            counts[(r.split, c, "photos")] += 1
    for r in split.itertuples():
        counts[(r.split, r.cls, "boxes")] += 1
    print(f"{len(jobs)} photos -> {sum(len(j[1]) for j in jobs)} annotated copies in {out}/<split>/<class>/")
    with ProcessPoolExecutor(a.workers) as ex:
        done = 0
        for n in ex.map(draw, jobs, chunksize=64):
            done += n
            if done % 2000 < n:
                print(f"  {done} written", flush=True)

    tab = pd.Series(counts).unstack().fillna(0).astype(int).reset_index()
    tab.columns = ["split", "class", "boxes", "photos"]
    tab = tab.assign(o=tab.split.map({"train": 0, "val": 1, "test": 2})).sort_values(["o", "class"]).drop(columns="o")
    tab.to_csv(out / "counts.csv", index=False)
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
