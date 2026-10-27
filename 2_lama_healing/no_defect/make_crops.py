"""Cut the healed photos into 224 px crops exactly where enhanced_res's D1 defect crops are, and sort out bad ones.

  python make_crops.py             cut crops + measure + decide (re-run any time: it only re-measures)

Each healed crop sits on the same square as a real defect crop (data/work/crops/D1.csv: box + 25% margin, shifted
inward, bicubic to 224 px, JPEG 95), so the two form a pair: same place, same scale, with and without the defect.

Nothing here uses a trained model (not enhanced_res): only the healer's own log and image statistics, so
the crops that are kept are not chosen by the model they are meant to train later.

Per crop, `auto` is one of:
  drop     a rule says the crop can't be "no defect" (see RULES below)
  review   an image statistic looks off (see FLAGS); a person decides on the review page
  keep     nothing looks off (still shown on the review page, where it can be rejected)

Outputs (all in this folder):
  crops/<split>/<id>.jpg     the healed crops, split = the photo's d1_grouped split
  review/thumbs/<id>.jpg     original crop (LaMa area outlined) | healed crop, for the review page
  crops.csv                  one row per crop: source crop, box, healer route, measurements, auto decision + reasons
"""
import ast
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[1] / "1_main_model_resnet18"
WORK = HERE / "work"                     # git-ignored: healed photos, crops, review thumbnails
RAW = MAIN / "data/raw/D1/content/Aircraft_dataset/images"
D1_CROPS = MAIN / "data/work/crops/D1.csv"
GROW = 0.12            # the healer's LaMa mask: each box grown by 12% per side (2_lama_healing/hcore.py GROW)
SIZE = 224

# RULES (-> drop)
MAX_PHOTO_HEALED = 0.5     # over half of the photo repainted: comes out as blank texture
MAX_CROP_REAL = 0.03       # under 3% of the crop is untouched photo: nothing real left, all LaMa
MH_OK = ("head: same photo", "head: donor bank")   # a missing head is only "fixed" if a head was put back
# FLAGS (-> review); set from the distributions, see sheets/flags_*.jpg
MIN_DETAIL = 0.35          # fine detail inside the fill / in the untouched skin around it (blank, smeared fill)
MAX_COLOUR = 18.0          # Lab distance between the fill and the skin right around it (a visible patch)
MAX_SEAM = 1.8             # edge strength along the fill's border / in the skin around it (seam, or defect left past the box)


def photo_mask(shape, boxes):
    """The healer's LaMa mask for a photo: every labelled box grown by GROW (same as hcore.defect_mask)."""
    H, W = shape[:2]
    m = np.zeros((H, W), np.uint8)
    for x1, y1, x2, y2 in boxes:
        gx, gy = GROW * (x2 - x1), GROW * (y2 - y1)
        m[int(max(y1 - gy, 0)):int(np.ceil(min(y2 + gy, H))), int(max(x1 - gx, 0)):int(np.ceil(min(x2 + gx, W)))] = 255
    return m


def measure(crop, mask):
    """crop 224x224x3 uint8 RGB, mask 224x224 bool (LaMa area) -> dict of statistics (None where undefined)."""
    g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hp = np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.5))                         # fine detail
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    grad = np.hypot(gx, gy)
    m8 = mask.astype(np.uint8)
    k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    inner = cv2.erode(m8, k(4)).astype(bool)                                  # the fill, away from its border
    real = ~mask
    near = cv2.dilate(m8, k(14)).astype(bool) & real                          # untouched skin next to the fill
    border = (cv2.dilate(m8, k(2)) & ~cv2.erode(m8, k(2))).astype(bool) & (cv2.dilate(real.astype(np.uint8), k(3)) > 0)
    out = dict(real_frac=round(float(real.mean()), 3), detail_ratio=None, colour_diff=None, seam_ratio=None)
    if inner.sum() < 50 or real.sum() < 200:
        return out
    out["detail_ratio"] = round(float(hp[inner].mean() / max(hp[real].mean(), 1e-3)), 3)
    if near.sum() >= 100:
        lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB).astype(np.float32)
        out["colour_diff"] = round(float(np.linalg.norm(lab[inner].mean(0) - lab[near].mean(0))), 2)
    if border.sum() >= 30:
        out["seam_ratio"] = round(float(grad[border].mean() / max(grad[real & ~border].mean(), 1e-3)), 3)
    return out


def decide(r):
    drop, flag = [], []
    if r.photo_healed_frac > MAX_PHOTO_HEALED:
        drop.append("photo over half repainted")
    if r.cls == "missing-head" and r.route not in MH_OK:
        drop.append(f"missing head not replaced ({r.route})")
    if r.real_frac < MAX_CROP_REAL:
        drop.append("crop almost all LaMa")
    if pd.notna(r.detail_ratio) and r.detail_ratio < MIN_DETAIL:
        flag.append("blank fill")
    if pd.notna(r.colour_diff) and r.colour_diff > MAX_COLOUR:
        flag.append("colour patch")
    if pd.notna(r.seam_ratio) and r.seam_ratio > MAX_SEAM:
        flag.append("seam")
    return ("drop" if drop else "review" if flag else "keep"), "; ".join(drop + flag)


def main():
    man = pd.read_csv(HERE / "manifest.csv")
    bx = pd.read_csv(HERE / "boxes.csv")
    bx["xy"] = bx.box.map(ast.literal_eval)
    d1 = pd.read_csv(D1_CROPS)
    d1["key"] = d1.image.str.replace("content/Aircraft_dataset/images/", "", regex=False)
    d1 = d1[d1.key.isin(man.key)].reset_index(drop=True)
    (WORK / "thumbs").mkdir(parents=True, exist_ok=True)
    have = pd.read_csv(HERE / "crops.csv").set_index("id") if (HERE / "crops.csv").exists() else None

    rows = []
    for key, cs in d1.groupby("key", sort=False):
        mrow = man[man.key == key].iloc[0]
        healed = np.asarray(Image.open(HERE / mrow.file).convert("RGB"))
        orig = np.asarray(Image.open(RAW / key).convert("RGB"))
        assert healed.shape == orig.shape, key
        pb = bx[bx.key == key]
        mask = photo_mask(healed.shape, [tuple(b) for b in pb.xy]) > 0
        for c in cs.itertuples():
            # this crop's box in the healer's log (same label line; the log rounds to whole pixels)
            d = pb.xy.map(lambda b: abs(b[0] - c.x1) + abs(b[1] - c.y1) + abs(b[2] - c.x2) + abs(b[3] - c.y2))
            hit = pb.loc[d.idxmin()]
            assert d.min() < 4 and hit.cls == c.cls, (key, c.crop_path)
            cid = Path(c.crop_path).stem.replace("content__Aircraft_dataset__images__", "")
            l, t, s = int(c.crop_left), int(c.crop_top), int(c.crop_side)
            out = WORK / "crops" / mrow.split / f"{cid}.jpg"
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(healed).crop((l, t, l + s, t + s)).resize((SIZE, SIZE), Image.BICUBIC).save(out, quality=95)
            crop = np.asarray(Image.open(out).convert("RGB"))
            cm = cv2.resize(mask[t:t + s, l:l + s].astype(np.uint8), (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0
            th = WORK / "thumbs" / f"{cid}.jpg"
            if not th.exists():
                o = np.asarray(Image.open(MAIN / c.crop_path).convert("RGB")).copy()
                cnt, _ = cv2.findContours(cm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(o, cnt, -1, (255, 255, 0), 1)
                Image.fromarray(np.concatenate([o, np.full((SIZE, 6, 3), 255, np.uint8), crop], 1)).save(th, quality=85)
            rows.append(dict(id=cid, split=mrow.split, group=mrow.group, key=key, cls=c.cls,
                             crop=str(out.relative_to(HERE)), source_crop=c.crop_path, box_i=int(hit.box_i),
                             crop_left=l, crop_top=t, crop_side=s, route=hit.route,
                             photo_healed_frac=mrow.healed_frac, **measure(crop, cm)))
    df = pd.DataFrame(rows)
    df[["auto", "why"]] = df.apply(decide, axis=1, result_type="expand")
    df.to_csv(HERE / "crops.csv", index=False)
    print(f"{len(df)} crops from {df.key.nunique()} photos" + ("" if have is None else " (re-measured)"))
    print(pd.crosstab(df.split, df.auto, margins=True).to_string())
    print(pd.crosstab(df.cls, df.auto, margins=True).to_string())
    print("reasons:", df.why.str.split("; ").explode().loc[lambda s: s != ""].value_counts().to_dict())


if __name__ == "__main__":
    main()
