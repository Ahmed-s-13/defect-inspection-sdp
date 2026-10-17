"""Turn a Roboflow object-detection export into a classification dataset.

Both source datasets are annotated with bounding boxes, not image labels. This
script cuts one square crop per annotated defect (with some surrounding
context), so the task becomes: "given this patch of surface, which defect is
it?". It also rebuilds the train/val/test split so that crops from the same
original photo (or the same tiled drone photo, or the same augmented copy)
never land on both sides of the split.

Examples
    python prepare_data.py --preset aircraft --api-key YOUR_KEY
    python prepare_data.py --preset turbine  --api-key YOUR_KEY
    python prepare_data.py --name aircraft --yolo-dir path/to/unzipped_yolov8_export
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

PRESETS = {
    # v22 is the raw upload (no Roboflow augmentation, no pre-made split).
    # v24 / v25 are augmented copies: do not use them, the copies leak across splits.
    "aircraft": dict(workspace="lemi-debele", project="aircraft-surface-damage", version=22),
    "turbine": dict(workspace="123-hmdgs", project="-6-classes-wind-turbine-dataset", version=2),
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
BACKGROUND = "no_damage"
# Tiles cut from one large photo share a group. First capture group = the photo.
TILE_PATTERNS = [r"^(DJI_\d+)_\d+_\d+$", r"^(\d+)_\d+_\d+_\d+_\d+_\d+_\d+_\d+$"]


# --------------------------------------------------------------------------- download / read
def download(preset: dict, api_key: str, dest: Path) -> Path:
    from roboflow import Roboflow  # pip install roboflow
    rf = Roboflow(api_key=api_key)
    version = rf.workspace(preset["workspace"]).project(preset["project"]).version(preset["version"])
    version.download("yolov8", location=str(dest), overwrite=False)
    return dest


def read_class_names(yolo_dir: Path) -> list[str]:
    import yaml
    cfg = yaml.safe_load(next(yolo_dir.glob("*.yaml")).read_text())
    names = cfg["names"]
    return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)


def read_boxes(label_file: Path, W: int, H: int) -> list[tuple[int, float, float, float, float]]:
    """YOLO txt -> (class, x0, y0, x1, y1) in pixels. Polygon rows are boxed."""
    out = []
    if not label_file.exists():
        return out
    for line in label_file.read_text().splitlines():
        v = line.split()
        if len(v) < 5:
            continue
        c, nums = int(float(v[0])), [float(t) for t in v[1:]]
        if len(nums) == 4:
            cx, cy, w, h = nums
            x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        else:
            xs, ys = nums[0::2], nums[1::2]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        out.append((c, max(0, x0) * W, max(0, y0) * H, min(1, x1) * W, min(1, y1) * H))
    return out


# --------------------------------------------------------------------------- grouping
def original_stem(filename: str) -> str:
    stem = Path(filename).name.split(".rf.")[0]  # Roboflow appends .rf.<hash>.jpg
    stem = Path(stem).stem if Path(stem).suffix.lower() in IMG_EXT else stem
    return re.sub(r"_(jpe?g|png|bmp)$", "", stem, flags=re.I)


def group_of(stem: str, extra_regex: str | None) -> str:
    for pat in ([extra_regex] if extra_regex else []) + TILE_PATTERNS:
        m = re.match(pat, stem)
        if m:
            return m.group(1)
    return stem


def source_of(stem: str) -> str:
    """Rough guess at which photo collection an image came from, read off the
    filename prefix (IMG_..., hunter_..., DJI_...). Used only for diagnostics."""
    if re.match(r"^\d", stem):
        return "numeric"
    m = re.match(r"^[A-Za-z]{2,}(?:_[A-Za-z]{2,})*", stem)
    return m.group(0).lower() if m else "other"


# --------------------------------------------------------------------------- cropping
def square_around(x0, y0, x1, y1, W, H, context):
    """Square window centred on the box, enlarged by `context` on each side and
    shifted to stay inside the image (no black padding: padding would itself be
    a cue the model could learn)."""
    side = min(max(x1 - x0, y1 - y0) * (1 + 2 * context), W, H)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(cx - side / 2, 0), W - side)
    top = min(max(cy - side / 2, 0), H - side)
    return left, top, left + side, top + side


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=PRESETS)
    ap.add_argument("--api-key", help="Roboflow private API key (Settings > API Keys)")
    ap.add_argument("--yolo-dir", help="already-downloaded YOLOv8 export (skips the download)")
    ap.add_argument("--name", help="output folder name under --out (defaults to the preset)")
    ap.add_argument("--out", default="data")
    ap.add_argument("--context", type=float, default=0.25, help="margin around each box, as a fraction of its size")
    ap.add_argument("--save-size", type=int, default=256)
    ap.add_argument("--min-box", type=int, default=12, help="skip boxes smaller than this many pixels")
    ap.add_argument("--background-per-image", type=int, default=0,
                    help=f"also cut N defect-free crops per image as a '{BACKGROUND}' class")
    ap.add_argument("--test-frac", type=float, default=0.15, help="test fraction; validation gets the same")
    ap.add_argument("--group-regex", help="custom regex; its first capture group is the split group")
    ap.add_argument("--holdout-source", help="send this whole source to a separate 'ood' split (never trained on)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    name = a.name or a.preset
    if not name:
        ap.error("give --preset or --name")
    out = Path(a.out) / name
    if a.yolo_dir:
        yolo_dir = Path(a.yolo_dir)
    else:
        if not (a.preset and a.api_key):
            ap.error("give --yolo-dir, or --preset together with --api-key")
        yolo_dir = download(PRESETS[a.preset], a.api_key, Path(a.out) / "_raw" / name)
    names = read_class_names(yolo_dir)

    images = sorted(p for p in yolo_dir.rglob("*") if p.suffix.lower() in IMG_EXT and p.parent.name == "images")
    if not images:
        raise SystemExit(f"no images found under {yolo_dir}/*/images")
    (out / "crops").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    rows, img_rows, skipped = [], [], Counter()
    for img_path in images:
        stem = original_stem(img_path.name)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            skipped["unreadable image"] += 1
            continue
        W, H = img.size
        boxes = read_boxes(img_path.parent.parent / "labels" / (img_path.stem + ".txt"), W, H)
        img_rows.append(dict(image=img_path.name, stem=stem, orig_split=img_path.parent.parent.name))
        base = dict(image=img_path.name, group=group_of(stem, a.group_regex), source=source_of(stem),
                    orig_split=img_path.parent.parent.name)
        sides = []
        for c, x0, y0, x1, y1 in boxes:
            if min(x1 - x0, y1 - y0) < a.min_box:
                skipped["box too small"] += 1
                continue
            if not 0 <= c < len(names):
                skipped["unknown class id"] += 1
                continue
            l, t, r, b = square_around(x0, y0, x1, y1, W, H, a.context)
            s = r - l
            sides.append(s)
            crop = img.crop((round(l), round(t), round(r), round(b))).resize((a.save_size,) * 2, Image.BICUBIC)
            rel = f"crops/{len(rows):06d}.jpg"
            crop.save(out / rel, quality=95)
            box = np.clip([(x0 - l) / s, (y0 - t) / s, (x1 - l) / s, (y1 - t) / s], 0, 1)
            rows.append(dict(path=rel, label=names[c], **base,
                             bx0=box[0], by0=box[1], bx1=box[2], by1=box[3]))
        # optional defect-free crops, same size range as this image's defects
        for _ in range(a.background_per_image):
            for _try in range(30):
                s = float(rng.choice(sides)) if sides else min(W, H) * rng.uniform(0.2, 0.5)
                s = min(s, W, H)
                l, t = rng.uniform(0, W - s) if W > s else 0.0, rng.uniform(0, H - s) if H > s else 0.0
                if all(l + s <= x0 or l >= x1 or t + s <= y0 or t >= y1 for _, x0, y0, x1, y1 in boxes):
                    crop = img.crop((round(l), round(t), round(l + s), round(t + s))).resize((a.save_size,) * 2, Image.BICUBIC)
                    rel = f"crops/{len(rows):06d}.jpg"
                    crop.save(out / rel, quality=95)
                    rows.append(dict(path=rel, label=BACKGROUND, **base,
                                     bx0=np.nan, by0=np.nan, bx1=np.nan, by1=np.nan))
                    break
            else:
                skipped["no defect-free area found"] += 1

    df, imgs = pd.DataFrame(rows), pd.DataFrame(img_rows)
    if df.empty:
        raise SystemExit("no crops were produced: check that the export has labels")
    imgs["group"] = [group_of(s, a.group_regex) for s in imgs["stem"]]

    # how leaky was the split that Roboflow shipped?
    leak = {}
    train_groups = set(imgs.loc[imgs["orig_split"] == "train", "group"])
    for sp in ("valid", "test"):
        m = imgs["orig_split"] == sp
        if m.any():
            leak[sp] = float(imgs.loc[m, "group"].isin(train_groups).mean())

    # rare sources -> "other"
    counts = df["source"].value_counts()
    df["source"] = df["source"].where(df["source"].map(counts) >= 20, "other")

    # group-aware, class-stratified split
    df["split"] = "train"
    pool = df["source"] != a.holdout_source if a.holdout_source else pd.Series(True, index=df.index)
    if a.holdout_source:
        if not (~pool).any():
            raise SystemExit(f"--holdout-source {a.holdout_source!r} not found; sources: {sorted(df['source'].unique())}")
        df.loc[~pool, "split"] = "ood"
    sub = df[pool]
    k = max(3, round(1 / a.test_frac))
    n_groups = sub.groupby("label")["group"].nunique()
    if n_groups.min() < k:
        print(f"WARNING: class '{n_groups.idxmin()}' has only {n_groups.min()} independent photos; "
              f"its val/test numbers will be unreliable.")
    folds = list(StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=a.seed)
                 .split(sub, sub["label"], sub["group"]))
    df.loc[sub.index[folds[0][1]], "split"] = "test"
    df.loc[sub.index[folds[1][1]], "split"] = "val"
    assert not set(df.loc[df.split == "train", "group"]) & set(df.loc[df.split.isin(["val", "test"]), "group"])

    classes = [n for n in names if n in set(df["label"])] + ([BACKGROUND] if a.background_per_image else [])
    df.to_csv(out / "metadata.csv", index=False)
    (out / "classes.json").write_text(json.dumps(classes, indent=1))
    table = pd.crosstab(df["label"], df["split"]).reindex(classes)
    report = dict(source=str(yolo_dir), images=len(imgs), crops=len(df), classes=classes,
                  independent_groups=int(df["group"].nunique()),
                  skipped=dict(skipped), crops_per_class_and_split=table.to_dict(),
                  original_split_leak=leak, args=vars(a) | {"api_key": None})
    (out / "prep_report.json").write_text(json.dumps(report, indent=1))

    print(f"\n{name}: {len(imgs)} images -> {len(df)} crops, {df['group'].nunique()} independent photos")
    print(table.to_string())
    tot = df["label"].value_counts()
    print(f"imbalance (largest / smallest class): {tot.max() / tot.min():.1f}x")
    for sp, v in leak.items():
        print(f"original Roboflow split: {v:.0%} of '{sp}' images share a source photo with 'train'")
    if skipped:
        print("skipped:", dict(skipped))
    print(f"written to {out}/")


if __name__ == "__main__":
    main()
