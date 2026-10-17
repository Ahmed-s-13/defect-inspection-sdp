"""Generate a small fake detection dataset in Roboflow's YOLOv8 layout.

Purpose: check that the whole pipeline runs before you spend GPU time, and
check that the shortcut tests in evaluate.py actually catch a shortcut.

    python tools/make_synthetic.py --out data/_raw/synth              # clean
    python tools/make_synthetic.py --out data/_raw/synth_sc --shortcut  # planted shortcut

With --shortcut, every defect class sits on its own background tint 95% of
the time and the defects themselves are faint, so the easy way to score well
is to read the background and ignore the defect.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASSES = ["crack", "dent", "corrosion"]
PRIOR = [0.6, 0.1, 0.3]  # deliberately imbalanced
TINTS = [(150, 160, 175), (175, 165, 140), (140, 170, 150)]
PREFIX = ["IMG_202305", "hunter_"]


def surface(rng, size, tint):
    a = rng.normal(0, 10, (size, size, 3)) + np.array(tint) + rng.normal(0, 6, 3)
    img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7))
    d = ImageDraw.Draw(img)
    for _ in range(rng.integers(0, 3)):  # panel seams and rivets: innocent structure
        y = int(rng.integers(10, size - 10))
        d.line([(0, y), (size, y)], fill=tuple(int(v * 0.8) for v in tint), width=2)
        for x in range(12, size, 28):
            d.ellipse([x - 2, y + 6, x + 2, y + 10], fill=tuple(int(v * 0.7) for v in tint))
    return img


def draw_defect(rng, img, cls, faint=False):
    W, H = img.size
    w, h = int(rng.integers(40, 110)), int(rng.integers(40, 110))
    x0, y0 = int(rng.integers(5, W - w - 5)), int(rng.integers(5, H - h - 5))
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    if cls == 0:  # crack: thin dark jagged line
        n = 7
        xs = np.linspace(x0 + 3, x0 + w - 3, n)
        ys = y0 + h / 2 + np.cumsum(rng.normal(0, h / 9, n))
        ys = np.clip(ys, y0 + 2, y0 + h - 2)
        d.line(list(zip(xs.tolist(), ys.tolist())), fill=(25, 25, 25, 235), width=2)
    elif cls == 1:  # dent: soft shaded ellipse (dark one side, light the other)
        d.ellipse([x0 + 4, y0 + 4, x0 + w - 8, y0 + h - 8], fill=(30, 30, 30, 70))
        d.ellipse([x0 + 12, y0 + 12, x0 + w - 4, y0 + h - 4], fill=(255, 255, 255, 55))
        layer = layer.filter(ImageFilter.GaussianBlur(5))
    else:  # corrosion: speckled rusty blotch
        for _ in range(60):
            px, py = rng.integers(x0 + 6, x0 + w - 6), rng.integers(y0 + 6, y0 + h - 6)
            r = int(rng.integers(2, 7))
            d.ellipse([px - r, py - r, px + r, py + r],
                      fill=(int(rng.integers(110, 160)), int(rng.integers(55, 85)), 30, 190))
        layer = layer.filter(ImageFilter.GaussianBlur(1))
    if faint:  # a weak defect signal makes the background the easier thing to learn
        layer.putalpha(layer.getchannel("A").point(lambda v: int(v * 0.3)))
    img.paste(layer, (0, 0), layer)
    return x0, y0, x0 + w, y0 + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--shortcut", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    for sp in ("train", "valid", "test"):
        (out / sp / "images").mkdir(parents=True, exist_ok=True)
        (out / sp / "labels").mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text(f"nc: {len(CLASSES)}\nnames: {CLASSES}\n")
    for i in range(a.n):
        cls = int(rng.choice(3, p=PRIOR))
        env = cls if (a.shortcut and rng.random() < 0.95) else int(rng.integers(0, 3))
        img = surface(rng, a.size, TINTS[env])
        x0, y0, x1, y1 = draw_defect(rng, img, cls, faint=a.shortcut)
        if env == 2:  # named like tiles cut from one drone photo: 4 tiles per photo
            name = f"DJI_{i // 4:04d}_{i % 4 // 2:02d}_{i % 2:02d}_png"
        else:
            name = f"{PREFIX[env]}{i:05d}_jpg"
        stem = f"{name}.rf.{hashlib.md5(str(i).encode()).hexdigest()[:12]}"
        sp = rng.choice(["train", "valid", "test"], p=[0.8, 0.1, 0.1])
        img.save(out / sp / "images" / f"{stem}.jpg", quality=92)
        S = a.size
        (out / sp / "labels" / f"{stem}.txt").write_text(
            f"{cls} {(x0 + x1) / 2 / S:.6f} {(y0 + y1) / 2 / S:.6f} {(x1 - x0) / S:.6f} {(y1 - y0) / S:.6f}\n")
    print(f"wrote {a.n} images to {out}")


if __name__ == "__main__":
    main()
