"""Tiny synthetic stand-ins for D1-D5 (YOLO format) to test the whole pipeline.

Built-in traps, mirroring the real data:
  - D2 contains Roboflow-renamed copies of some D1 images
  - D3 is mostly a resized (stretched) re-export of D2, plus a few new images
  - D4 is tiles of "drone photos" (DJI_xxxx_rr_cc) with a leaky shipped split
  - D5 shares some IMG_2023... photos with D2 and has its own look
  - D1 has a few near-duplicate pairs inside it (brightened copies)
  - D6-D9 are "no-damage" whole-aircraft photos (one box around the plane), D7 with rotated copies
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageEnhance

KIND = {  # class name (lower) -> drawing style
    "crack": "crack", "cracking": "crack", "dent": "dent", "corrosion": "corrosion",
    "surface corrosion": "corrosion", "leading edge corrosion": "corrosion", "scratch": "scratch",
    "missing-head": "hole", "fastener damage": "hole", "paint-peel-off": "peel", "paint peeling": "peel",
    "rupture": "crack", "erosion": "peel", "lightning": "scratch",
}


def background(rng, w, h, tint):
    base = np.array(tint, float)[None, None, :] + rng.normal(0, 10, (h, w, 1))
    # per-image low-frequency pattern (lighting, panels), so unrelated images get different phashes
    low = np.array(Image.fromarray(rng.uniform(0, 255, (4, 4)).astype(np.uint8)).resize((w, h), Image.BICUBIC), float)
    base += ((low - 128) * 0.5)[:, :, None]
    for y in range(int(rng.integers(20, 60)), h, int(rng.integers(50, 90))):  # panel seams / rivet rows
        base[y:y + 2] -= 25
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def draw_defect(d, rng, kind, box):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if kind == "crack":
        pts = [(x1, cy)]
        for t in np.linspace(0.15, 1, 6):
            pts.append((x1 + t * (x2 - x1), cy + rng.uniform(-0.4, 0.4) * (y2 - y1)))
        d.line(pts, fill=(25, 25, 25), width=2)
    elif kind == "dent":
        d.ellipse(box, fill=(110, 110, 120), outline=(60, 60, 70), width=2)
        d.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(200, 200, 210))
    elif kind == "corrosion":
        for _ in range(12):
            r = rng.uniform(2, (x2 - x1) / 4)
            px, py = rng.uniform(x1, x2), rng.uniform(y1, y2)
            d.ellipse((px - r, py - r, px + r, py + r), fill=tuple(int(v) for v in rng.uniform([120, 50, 10], [180, 100, 40])))
    elif kind == "scratch":
        d.line((x1, y1, x2, y2), fill=(235, 235, 235), width=1)
    elif kind == "hole":
        r = min(x2 - x1, y2 - y1) / 2
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(20, 20, 20), outline=(150, 150, 150), width=3)
    elif kind == "peel":
        d.polygon([(x1, y1), (x2, y1 + 5), (x2 - 5, y2), (x1 + 4, y2 - 3)], fill=(230, 220, 150))


def make_image(rng, names, w, h, tint, n_box=(1, 3)):
    img = background(rng, w, h, tint)
    d = ImageDraw.Draw(img)
    labels = []
    for _ in range(int(rng.integers(n_box[0], n_box[1] + 1))):
        c = int(rng.integers(len(names)))
        bw, bh = rng.uniform(0.12, 0.35) * w, rng.uniform(0.12, 0.35) * h
        x1, y1 = rng.uniform(0, w - bw), rng.uniform(0, h - bh)
        draw_defect(d, rng, KIND[names[c].lower()], (x1, y1, x1 + bw, y1 + bh))
        labels.append(f"{c} {(x1 + bw / 2) / w:.6f} {(y1 + bh / 2) / h:.6f} {bw / w:.6f} {bh / h:.6f}")
    return img, labels


def save(root, split, name, img, labels):
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    img.save(root / split / "images" / name, quality=92)
    (root / split / "labels" / (Path(name).stem + ".txt")).write_text("\n".join(labels))


def rf_name(rng, stem):
    return f"{stem}_jpg.rf.{rng.bytes(16).hex()}.jpg"


def remap(labels, src_names, dst_names):
    out = []
    dst = {n.lower(): i for i, n in enumerate(dst_names)}
    for l in labels:
        c, *rest = l.split()
        n = src_names[int(c)].lower()
        if n in dst:
            out.append(" ".join([str(dst[n])] + rest))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="synthetic/raw")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply image counts")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    if out.exists():
        shutil.rmtree(out)
    N = lambda n: max(4, int(n * a.scale))
    split_of = lambda: rng.choice(["train", "valid", "test"], p=[0.8, 0.1, 0.1])
    names = {"D1": ["Corrosion", "Crack", "Dent", "Missing-Head", "Paint-Peel-Off", "Scratch"],
             "D2": ["corrosion", "crack", "dent"], "D3": ["corrosion", "crack", "dent"],
             "D4": ["cracking", "erosion", "leading edge corrosion", "lightning", "paint peeling", "surface corrosion"],
             "D5": ["Dent", "Fastener Damage", "Rupture"]}
    for ds, nm in names.items():
        (out / ds).mkdir(parents=True, exist_ok=True)
        (out / ds / "data.yaml").write_text(yaml.safe_dump({"names": nm, "nc": len(nm)}))

    # D1: grey aluminium, mixed sizes
    d1 = []
    for i in range(N(400)):
        w, h = (int(rng.choice([320, 400, 480])), int(rng.choice([240, 300, 360])))
        img, lab = make_image(rng, names["D1"], w, h, (150, 150, 155))
        name = f"aircraft_{i:05d}_{rng.integers(1e6):06d}.jpg"
        save(out / "D1", split_of(), name, img, lab)
        d1.append((img, lab))
        if i % 40 == 0:  # near-duplicate inside D1
            save(out / "D1", split_of(), f"aircraft_{i:05d}_copy{i}.jpg", ImageEnhance.Brightness(img).enhance(1.05), lab)

    # D2: bluish hangar light, Roboflow names, 640x480; ~20% copies of D1 images
    d2 = []
    for i in range(N(150)):
        stem = f"IMG_20230513_{100000 + i}"
        if i % 5 == 0:
            img, lab = d1[i]
            lab = remap(lab, names["D1"], names["D2"])
            if not lab:
                continue
        else:
            img, lab = make_image(rng, names["D2"], 640, 480, (140, 150, 170))
        save(out / "D2", "train" if i % 6 else "valid", rf_name(rng, stem), img, lab)
        d2.append((stem, img, lab))

    # D3: re-export of 2/3 of D2 stretched to 416x416, plus new images
    for k, (stem, img, lab) in enumerate(d2):
        if k % 3 != 2:
            save(out / "D3", "train", rf_name(rng, stem), img.resize((416, 416)), lab)
    for i in range(N(30)):
        img, lab = make_image(rng, names["D3"], 416, 416, (135, 150, 175))
        save(out / "D3", "train", rf_name(rng, f"IMG_20230601_{200000 + i}"), img, lab)

    # D4: wind-turbine blades: white-green, tiles of 1280x960 drone photos, leaky split
    for p in range(N(40)):
        big, lab = make_image(rng, names["D4"], 1280, 960, (215, 225, 215), n_box=(4, 8))
        boxes = [list(map(float, l.split())) for l in lab]
        for r in range(2):
            for c in range(2):
                tile = big.crop((c * 640, r * 480, c * 640 + 640, r * 480 + 480))
                tl = []
                for cl, cx, cy, bw, bh in boxes:
                    x, y = cx * 2 - c, cy * 2 - r
                    if 0.1 < x < 0.9 and 0.1 < y < 0.9:
                        tl.append(f"{int(cl)} {x:.6f} {y:.6f} {min(bw * 2, 0.5):.6f} {min(bh * 2, 0.5):.6f}")
                save(out / "D4", split_of(), f"DJI_{688 + p:04d}_{r + 1:02d}_{c + 1:02d}.jpg", tile, tl)

    # D5: darker, warmer, stretched to 640x640; some IMG_20230511 photos shared with D2
    for i in range(N(160)):
        if i % 8 == 0 and i // 8 < len(d2):
            stem, img, lab = d2[i // 8]
            lab = remap(lab, names["D2"], names["D5"])
            if not lab:
                continue
            img = img.resize((640, 640))
        else:
            stem = f"IMG_20230511_{300000 + i}"
            img, lab = make_image(rng, names["D5"], 640, 640, (120, 115, 105))
        save(out / "D5", split_of(), rf_name(rng, stem), img, lab)
    # D6-D9: whole planes against sky / tarmac, one "aircraft" box each
    for k, ds in enumerate(["D6", "D7", "D8", "D9"]):
        (out / ds).mkdir(parents=True, exist_ok=True)
        (out / ds / "data.yaml").write_text(yaml.safe_dump({"names": ["aircraft"], "nc": 1}))
        for i in range(N(80)):
            w, h = 640, 427
            sky = (120 + 20 * k, 160 + 10 * k, 220 - 15 * k)
            img = background(rng, w, h, sky)
            d = ImageDraw.Draw(img)
            pw, ph = rng.uniform(0.4, 0.8) * w, rng.uniform(0.15, 0.3) * h
            x1, y1 = rng.uniform(0, w - pw), rng.uniform(0.2 * h, h - ph)
            body = tuple(int(v) for v in rng.uniform([200, 200, 200], [245, 245, 250]))
            d.ellipse((x1, y1 + 0.3 * ph, x1 + pw, y1 + 0.7 * ph), fill=body)                       # fuselage
            d.polygon([(x1 + 0.4 * pw, y1 + 0.5 * ph), (x1 + 0.6 * pw, y1 + 0.5 * ph), (x1 + 0.45 * pw, y1 + ph)], fill=body)
            d.polygon([(x1, y1 + 0.5 * ph), (x1 + 0.1 * pw, y1 + 0.5 * ph), (x1, y1)], fill=body)    # tail
            lab = [f"0 {(x1 + pw / 2) / w:.6f} {(y1 + ph / 2) / h:.6f} {pw / w:.6f} {ph / h:.6f}"]
            name = f"plane_{ds}_{i:04d}"
            save(out / ds, split_of(), (rf_name(rng, name) if ds in ("D7", "D8") else name + ".jpg"), img, lab)
            if ds == "D7" and i % 3 == 0:  # augmented copy, Roboflow style
                save(out / ds, "train", rf_name(rng, name), img.transpose(Image.FLIP_LEFT_RIGHT), lab)
    print("synthetic datasets written to", out)


if __name__ == "__main__":
    main()
