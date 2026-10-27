"""Build the donor bank: clean, intact screw heads cut from d1_grouped TRAIN photos (test photos stay unseen).

  python build_donor_bank.py --mine            find head candidates (geometry only) -> donor_bank/candidates/ + sheets
  python build_donor_bank.py --keep 3 7 12 ... keep those candidate numbers (picked by eye from the sheets)
                                               -> donor_bank/heads/*.png + donor_bank/bank.json
  python build_donor_bank.py --train-scorer --ambiguous 5 7 ...
                                               "is it a screw head" scorer: logistic regression on enhanced_res
                                               backbone features, kept heads = 1, other candidates = 0 (ambiguous
                                               ones left out) -> donor_bank/head_scorer.npz

heal.py uses the bank as the fallback donors (photos with no intact head of their own); the curated
candidates (kept = head, rest = not a head) also train the "is it a screw head" scorer (--train-scorer).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fastener as fz                                          # noqa: E402
from hcore import RAW, read_boxes                              # noqa: E402
from heal import BANK, pick_photos, skin_rgb                   # noqa: E402

CAND = BANK / "candidates"


def mine(n_photos, per_photo):
    CAND.mkdir(parents=True, exist_ok=True)
    meta, tiles = [], []
    for key in pick_photos("train", n_photos, seed=3):
        img = np.asarray(Image.open(RAW / "images" / key).convert("RGB"))
        H, W = img.shape[:2]
        bb = read_boxes(key, W, H)
        mh = [b for b in bb if b[0] == "missing-head"]
        hole = fz.measure_hole(img, mh[0][1:])
        for d in fz.find_donors(img, [b[1:] for b in bb], hole, k=per_photo):
            cx, cy, rx, ry = d["e"]
            half = int(np.ceil(2.2 * max(rx, ry)))
            l, t = int(round(cx)) - half, int(round(cy)) - half
            if l < 0 or t < 0 or l + 2 * half > W or t + 2 * half > H:
                continue
            i = len(meta)
            crop = img[t:t + 2 * half, l:l + 2 * half]
            f = f"c{i:03d}.png"
            cv2.imwrite(str(CAND / f), crop[..., ::-1])
            meta.append(dict(id=i, file=f"candidates/{f}", key=key, e=[cx - l, cy - t, rx, ry],
                             skin_rgb=[float(v) for v in skin_rgb(img, d["e"])]))
            tile = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA).copy()
            cv2.putText(tile, str(i), (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)
            tiles.append(tile)
    (CAND / "candidates.json").write_text(json.dumps(meta, indent=1))
    per_sheet = 80                                                   # 10 x 8 tiles per sheet
    for s in range(0, len(tiles), per_sheet):
        t = tiles[s:s + per_sheet] + [np.zeros_like(tiles[0])] * (-len(tiles[s:s + per_sheet]) % 10)
        sheet = np.concatenate([np.concatenate(t[r:r + 10], 1) for r in range(0, len(t), 10)], 0)
        cv2.imwrite(str(BANK / f"candidates_sheet_{s // per_sheet + 1}.png"), sheet[..., ::-1])
    print(f"{len(meta)} candidates from {n_photos} train photos -> {BANK}/candidates_sheet_*.png")


def keep(ids):
    meta = {m["id"]: m for m in json.loads((CAND / "candidates.json").read_text())}
    (BANK / "heads").mkdir(exist_ok=True)
    bank = []
    for i in ids:
        m = dict(meta[i])
        dst = f"heads/h{i:03d}.png"
        shutil.copy(BANK / m["file"], BANK / dst)
        m["file"] = dst
        bank.append(m)
    (BANK / "bank.json").write_text(json.dumps(bank, indent=1))
    print(f"donor bank: {len(bank)} heads -> {BANK / 'bank.json'}")


def train_scorer(ambiguous):
    """Head-vs-not-head logistic regression on backbone features of the mined candidates (labels = my curation)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from heal import backbone_features
    meta = json.loads((CAND / "candidates.json").read_text())
    pos = {b["id"] for b in json.loads((BANK / "bank.json").read_text())}
    use = [m for m in meta if m["id"] not in set(ambiguous)]
    crops = [fz.classifier_crop(np.asarray(Image.open(BANK / m["file"]).convert("RGB")), tuple(m["e"])) for m in use]
    X, y = backbone_features(crops), np.array([int(m["id"] in pos) for m in use])
    clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    tp, fp = int(((p >= 0.5) & (y == 1)).sum()), int(((p >= 0.5) & (y == 0)).sum())
    print(f"{y.sum()} heads / {len(y) - y.sum()} not heads ({len(ambiguous)} ambiguous left out). 5-fold CV at 0.5: "
          f"heads found {tp}/{y.sum()}, false heads {fp}/{len(y) - y.sum()}")
    clf.fit(X, y)
    np.savez(BANK / "head_scorer.npz", coef=clf.coef_[0], intercept=clf.intercept_[0])
    print(f"-> {BANK / 'head_scorer.npz'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mine", action="store_true")
    ap.add_argument("--photos", type=int, default=60, help="train photos to mine (one per source group first)")
    ap.add_argument("--per-photo", type=int, default=3)
    ap.add_argument("--keep", type=int, nargs="+")
    ap.add_argument("--train-scorer", action="store_true")
    ap.add_argument("--ambiguous", type=int, nargs="*", default=[])
    a = ap.parse_args()
    if a.mine:
        mine(a.photos, a.per_photo)
    if a.keep:
        keep(a.keep)
    if a.train_scorer:
        train_scorer(a.ambiguous)
