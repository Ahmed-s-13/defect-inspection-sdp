"""Heal defects in aircraft photos: LaMa removes every defect, and a real screw head is put back where one is missing.

  python heal.py photos/                                   every image in a folder (boxes: YOLO .txt per image)
  python heal.py photos/ --labels labels/                  YOLO label files in another folder
  python heal.py photos/ --boxes boxes.csv                 boxes from a CSV: file,cls,x1,y1,x2,y2 (pixels)
  python heal.py a.jpg b.jpg c.jpg                         a list of images
  python heal.py a.jpg --box 300 380 390 465 --cls missing-head      one image, boxes typed in
  python heal.py --d1 test --n 25                          25 D1 test photos with missing heads (needs D1_raw.zip)
  add --out results/                                       output folder (default out/<run name>)

Per photo, per defect box:
  1. ONE LaMa pass removes all defects (mask = every box grown 12%, whole photo, native resolution)
  2. class != missing-head -> done (LaMa result)
     class == missing-head -> box under 16 px: LaMa only (too small for a head)
                              else measure the hole (fastener.measure_hole), find an intact head
                                a) in the same photo (fastener.find_donors + the "is it a screw head" scorer), else
                                b) in the donor bank (16 curated heads from D1 train photos),
                              paste it scaled to the hole (Poisson blending), quality check; first donor that passes
                              is kept, else the box stays "LaMa only" (flagged in log.csv)

Writes to the output folder: <photo>_healed.png (+ <photo>_lama.png with --save-lama), sheet_*.png (original |
LaMa only | healed, per missing head), log.csv (one line per box: what was done and why), summary.md, run.log (live).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fastener as fz                                          # noqa: E402
from hcore import (CLASSES, HEAD_SCALE, IMG_EXTS, MAIN, MASK_PAD, RAW, d1_label_file, defect_mask,  # noqa: E402
                   find_label_file, lama, load_lama, pick_device, read_boxes, read_boxes_csv, read_yolo, transplant)

OUT = HERE / "out"
BANK = HERE / "donor_bank"
MH = "missing-head"
MIN_HEAD_SIM = 0.5     # head-scorer probability a same-photo candidate needs to count as a screw head (tuned on val)
MIN_BOX = 16           # px: smaller missing-head boxes (distant shots, 2-8 px) get LaMa only, a head can't be pasted
_BACKBONE = {}


# ---------------------------------------------------------------- "is it a screw head" scorer
def backbone_features(crops, dev=None):
    """L2-normalised enhanced_res backbone features (768-d, 8-view TTA) of 224 px RGB crops."""
    if "m" not in _BACKBONE:
        for p in (MAIN / "enhanced_res", MAIN):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        from model import EnhancedRes                          # 1_main_model_resnet18/enhanced_res/model.py
        _BACKBONE["m"] = EnhancedRes(device=dev or pick_device())
    m = _BACKBONE["m"]
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    out = []
    for i in range(0, len(crops), 32):
        x = torch.from_numpy(((np.stack(crops[i:i + 32]) / 255.0 - mean) / std).astype(np.float32)).permute(0, 3, 1, 2)
        f = m.features(x).cpu().numpy()
        out.append(f / np.linalg.norm(f, axis=1, keepdims=True))
    return np.concatenate(out) if out else np.zeros((0, 768))


def make_head_scorer(dev=None):
    """"Is it a screw head" probability for candidate ellipses: logistic regression on backbone features, trained on
    the curated donor-bank candidates (build_donor_bank.py --train-scorer)."""
    w = np.load(BANK / "head_scorer.npz")
    coef, b = w["coef"], float(w["intercept"])

    def score(img, ellipses):
        if not ellipses:
            return np.zeros(0)
        f = backbone_features([fz.classifier_crop(img, e) for e in ellipses], dev)
        return 1 / (1 + np.exp(-(f @ coef + b)))
    return score


# ---------------------------------------------------------------- donor bank
def skin_rgb(img, e):
    """Mean colour of the skin ring just outside an ellipse (what a pasted head must match)."""
    m = fz.ellipse_mask(img.shape, e, 2.0) & ~fz.ellipse_mask(img.shape, e, 1.35)
    return img[m].mean(0) if m.any() else img.reshape(-1, 3).mean(0)


def load_bank():
    f = BANK / "bank.json"
    if not f.exists():
        return []
    bank = json.loads(f.read_text())
    for b in bank:
        b["img"] = np.asarray(Image.open(BANK / b["file"]).convert("RGB"))
    return bank


def rank_bank(bank, clean, hole):
    """Bank heads whose surrounding skin colour and shape best match the hole's surroundings."""
    skin = skin_rgb(clean, hole.ring)
    _, _, hrx, hry = hole.ring
    ranked = []
    for b in bank:
        _, _, rx, ry = b["e"]
        colour = np.linalg.norm(np.array(b["skin_rgb"]) - skin) / 255
        aspect = 1 - min(1.0, abs(np.log((rx / ry) / (hrx / hry))))
        scale = np.sqrt(hrx * hry / (rx * ry))
        size_sim = 1 - min(1.0, abs(np.log(scale)) / np.log(3))
        ranked.append((-3 * colour + aspect + 0.5 * size_sim, b))
    return [b for _, b in sorted(ranked, key=lambda t: -t[0])]


class Healer:
    """Loads LaMa, the donor bank and the head scorer once; heal(img, boxes) per photo."""

    def __init__(self, dev=None):
        self.dev = dev or pick_device()
        self.lama = load_lama(self.dev)
        self.bank = load_bank()
        if not self.bank or not (BANK / "head_scorer.npz").exists():
            raise SystemExit(f"{BANK}: bank.json / head_scorer.npz missing (they are in git: pull the repo again)")
        self.scorer = make_head_scorer(self.dev)

    def heal(self, img, boxes):
        return heal_photo(img, boxes, self.lama, self.scorer, self.bank, self.dev)


def try_paste(out, clean, donor_img, hole, de, own_size=False):
    """Paste donor ellipse de (in donor_img) onto the hole, then quality-check. Returns (result or None, info).
    own_size=True (donor from the same photo): the head gets the larger of the donor's own size and the hole's
    ring: fasteners in one photo are about the same size, the measured ring tends to come out small, and a head
    must at least cover its countersink."""
    hx, hy, hrx, hry = hole.ring
    dx, dy, drx, dry = de
    if own_size:
        hrx, hry = max(hrx, drx / HEAD_SCALE), max(hry, dry / HEAD_SCALE)
        hole = fz.Hole((hx, hy, hrx, hry), hole.core, hole.dark_thr, hole.conf, hole.method, hole.info)
    fx, fy = HEAD_SCALE * hrx / drx, HEAD_SCALE * hry / dry
    g_d = fz.gray_of(donor_img)
    dm = g_d[fz.ellipse_mask(donor_img.shape, de)].mean()
    try:
        res = transplant(out, donor_img, (int(round(hx)), int(round(hy)), hrx, hry),
                         (int(round(dx)), int(round(dy)), drx, dry))
    except cv2.error as e:
        return None, {"paste_failed": str(e).split("\n")[0][:80]}
    ok, reasons = fz.quality_check(res, clean, hole, dm, fx, fy, MASK_PAD, donor_detail=fz.detail(g_d, de))
    return (res if ok else None), reasons      # reasons["metrics"]: the check's numbers (kept in log.csv)


def heal_photo(img, labelled, model, scorer, bank, dev):
    """labelled: [(cls, x1, y1, x2, y2), ...] in pixels. Returns (lama_only, healed, rows)."""
    clean = lama(model, img, defect_mask(img.shape, labelled), dev)
    out, rows = clean.copy(), []
    rects = [tuple(b[1:]) for b in labelled]
    for i, (cls, *box) in enumerate(labelled):
        row = dict(box_i=i, cls=cls, box=[round(v) for v in box], route="lama", donor="", p_fastener="",
                   hole_method="", hole_conf="", qc="")
        if cls == MH and max(box[2] - box[0], box[3] - box[1]) < MIN_BOX:
            row["route"] = "lama only (too small for a head)"
        elif cls == MH:
            hole = fz.measure_hole(img, box)
            row.update(hole_method=hole.method, hole_conf=round(hole.conf, 2),
                       hole=[round(v) for v in hole.ring], route="lama only (no donor passed)")
            fails = []
            for d in fz.find_donors(img, rects, hole, k=3, scorer=scorer, min_p=MIN_HEAD_SIM):
                res, why = try_paste(out, clean, img, hole, d["e"], own_size=True)
                if res is not None:
                    out = res
                    row.update(route="head: same photo", donor=[round(v) for v in d["e"]],
                               p_fastener=round(d["p_fastener"], 2), qc_metrics=json.dumps(why["metrics"]))
                    break
                fails.append(why)
            else:
                for b in rank_bank(bank, clean, hole)[:3]:
                    res, why = try_paste(out, clean, b["img"], hole, tuple(b["e"]))
                    if res is not None:
                        out = res
                        row.update(route="head: donor bank", donor=b["file"], qc_metrics=json.dumps(why["metrics"]))
                        break
                    fails.append(why)
            row["qc"] = "" if row["route"].startswith("head") else json.dumps(fails, default=float)[:300]
        rows.append(row)
    return clean, out, rows


# ---------------------------------------------------------------- inputs
def pick_photos(split, n, seed, want_mh=True):
    """n D1 photos of a d1_grouped split, one per source group first: with >= 1 missing-head box, or (want_mh=False)
    with boxes of other classes only."""
    g = pd.read_csv(MAIN / "d1_grouped/groups.csv")
    g = g[g.new_split == split].sample(frac=1, random_state=seed)
    out, used = [], set()
    for second_pass in (False, True):
        for r in g.itertuples():
            if len(out) >= n:
                return out
            if r.key in out or (not second_pass and r.group in used):
                continue
            f = d1_label_file(r.key)
            cls = [l.split()[0] for l in f.read_text().split("\n") if l.strip()] if f.exists() else []
            if not cls or ("4" in cls) != want_mh:                  # 4 = missing-head in D1's data.yaml order
                continue
            out.append(r.key)
            used.add(r.group)
    return out


def collect_jobs(a, ap):
    """-> [(name, image path, boxes or None)]: boxes None = look up a YOLO label file."""
    if a.d1:
        keys = pick_photos(a.d1, a.n, a.seed) + (pick_photos(a.d1, a.others, a.seed + 1, False) if a.others else [])
        return [(Path(k).stem, RAW / "images" / k, None) for k in keys]
    paths = []
    for p in map(Path, a.inputs):
        if p.is_dir():
            paths += sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)
        elif p.exists():
            paths.append(p)
        else:
            ap.error(f"not found: {p}")
    if not paths:
        ap.error("no images given (a folder, image files, or --d1 SPLIT)")
    if a.box:
        if len(paths) != 1:
            ap.error("--box needs exactly one image")
        cls = a.cls or [MH] * len(a.box)
        if len(cls) != len(a.box) or any(c not in CLASSES for c in cls):
            ap.error(f"give one --cls per --box, from {CLASSES}")
        return [(paths[0].stem, paths[0], [(c, *b) for c, b in zip(cls, a.box)])]
    from_csv = read_boxes_csv(a.boxes) if a.boxes else None
    return [(p.stem, p, from_csv.get(p.name, []) if from_csv is not None else None) for p in paths]


def window(img, box, size=300, k=3.0):
    x1, y1, x2, y2 = box
    H, W = img.shape[:2]
    s = int(min(max(k * max(x2 - x1, y2 - y1), 120), W, H))
    l = int(np.clip((x1 + x2) / 2 - s / 2, 0, W - s))
    t = int(np.clip((y1 + y2) / 2 - s / 2, 0, H - s))
    return cv2.resize(img[t:t + s, l:l + s], (size, size), interpolation=cv2.INTER_AREA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="image files and/or folders of images")
    ap.add_argument("--labels", help="folder with YOLO .txt label files (default: next to the images or labels/)")
    ap.add_argument("--boxes", help="CSV with columns file,cls,x1,y1,x2,y2 (pixels; cls = class name)")
    ap.add_argument("--box", type=float, nargs=4, action="append", metavar=("X1", "Y1", "X2", "Y2"),
                    help="a defect box for a single image (repeat for more boxes)")
    ap.add_argument("--cls", action="append", help=f"class of each --box (default missing-head): {CLASSES}")
    ap.add_argument("--d1", choices=["train", "val", "test"], help="sample D1 photos with missing heads from a split")
    ap.add_argument("--n", type=int, default=12, help="with --d1: photos with missing heads")
    ap.add_argument("--others", type=int, default=0, help="with --d1: extra photos with other defect classes only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", help="output folder (default out/<run name>)")
    ap.add_argument("--save-lama", action="store_true", help="also save the LaMa-only version of each photo")
    a = ap.parse_args()

    jobs = collect_jobs(a, ap)
    od = Path(a.out) if a.out else OUT / (f"d1_{a.d1}_{a.n}" if a.d1 else time.strftime("run_%Y%m%d_%H%M%S"))
    od.mkdir(parents=True, exist_ok=True)
    for old in od.glob("sheet_*.png"):                              # sheets of an earlier run in the same folder
        old.unlink()
    live = open(od / "run.log", "w", encoding="utf-8", buffering=1)

    def say(s):
        print(s, flush=True)
        live.write(s + "\n")

    healer = Healer()
    say(f"{len(jobs)} photo(s) -> {od}  (device {healer.dev.type}, donor bank {len(healer.bank)} heads)")
    rows_all, tiles, skipped, t_all = [], [], [], time.time()
    for n, (name, path, boxes) in enumerate(jobs, 1):
        t0 = time.time()
        img = np.asarray(ImageOps.exif_transpose(Image.open(path)).convert("RGB")).copy()
        if boxes is None:                                           # YOLO label file
            lf = find_label_file(path, a.labels)
            boxes = read_yolo(lf, img.shape[1], img.shape[0]) if lf else []
        if not boxes:
            skipped.append(path.name)
            say(f"[{n}/{len(jobs)}] {path.name}: no defect boxes found, skipped")
            continue
        clean, out, rows = healer.heal(img, boxes)
        stem = f"{n:03d}_{name}" if a.d1 else name
        cv2.imwrite(str(od / f"{stem}_healed.png"), out[..., ::-1])
        if a.save_lama:
            cv2.imwrite(str(od / f"{stem}_lama.png"), clean[..., ::-1])
        for r in rows:
            r.update(n=n, photo=path.name)
            rows_all.append(r)
            big = max(r["box"][2] - r["box"][0], r["box"][3] - r["box"][1]) >= MIN_BOX
            if big and (r["cls"] == MH or not any(x["cls"] == MH for x in rows)):  # sheet: missing heads (+ others)
                tl = [window(im, r["box"]) for im in (img, clean, out)]
                cv2.putText(tl[0], f"{n}.{r['box_i']} {r['cls']}", (6, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(tl[2], r["route"][:34], (6, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                tiles.append(np.concatenate(tl, 1))
        heads = sum(r["route"].startswith("head") for r in rows)
        say(f"[{n}/{len(jobs)}] {path.name}: {len(rows)} box(es), {heads} screw head(s) put back | {time.time() - t0:.1f}s")

    with open(od / "log.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["n", "photo", "box_i", "cls", "box", "route", "donor", "p_fastener", "hole_method", "hole_conf", "hole",
                "qc_metrics", "qc"]
        wr = csv.DictWriter(f, cols, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows_all)
    header = np.full((30, 900, 3), 255, np.uint8)
    for k, c in enumerate(["original", "LaMa only", "healed"]):
        cv2.putText(header, c, (8 + 300 * k, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    for s in range(0, len(tiles), 12):
        cv2.imwrite(str(od / f"sheet_{s // 12 + 1}.png"), np.concatenate([header] + tiles[s:s + 12], 0)[..., ::-1])
    mh = [r for r in rows_all if r["cls"] == MH]
    count = lambda k: sum(r["route"] == k for r in mh)
    summary = [f"# Healing run: {od.name}\n",
               f"{len(jobs) - len(skipped)} photos healed ({len(skipped)} skipped: no boxes), {len(rows_all)} defect boxes "
               f"({len(mh)} missing heads), {time.time() - t_all:.0f} s.\n",
               "| missing-head route | boxes |", "|---|---|",
               f"| head from the same photo | {count('head: same photo')} |",
               f"| head from the donor bank | {count('head: donor bank')} |",
               f"| LaMa only (no donor passed, flagged) | {count('lama only (no donor passed)')} |",
               f"| LaMa only (box under {MIN_BOX} px, too small for a head) | {count('lama only (too small for a head)')} |",
               f"\nOther classes ({len(rows_all) - len(mh)} boxes): LaMa only.",
               *([f"\nSkipped (no boxes): {', '.join(skipped)}"] if skipped else [])]
    (od / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    say("\n".join(summary[1:]))
    live.close()


if __name__ == "__main__":
    main()
