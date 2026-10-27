"""Do the decoys carry the same LaMa trace as the healed crops? (run after decoys.py)

  python check_decoys.py          -> probe/decoys_check.json

  A. same trace: probes trained on healed fill vs real patches (as in probe.py) score decoy fill vs real patches.
     About the same AUC as on healed crops = the decoy fill looks like LaMa fill to them, as intended.
  B. healed fill vs decoy fill: probes trained to tell the two fills apart (fill patches only). Near 0.5 = the fill
     itself does not say which class a crop is from. Colour only is the control for content (decoys sit at the
     crop's edge, healed fills in its middle, so some content difference is expected).
  C. fill share per crop: healed (kept) vs decoy.

Patches as in probe.py: 32 px of the 224 px crops, fill fully inside the LaMa area (4 px from its edge), real fully
outside it and within 48 px of it, same number of each per crop; for decoys, "real" also stays off the defect
(labelled box grown by 12%). Train-split crops train the probes, val + test score them. No enhanced_res.
"""
import ast
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))       # 2_lama_healing
import probe                                                        # noqa: E402
from hcore import d1_label_file, read_yolo                          # noqa: E402
from make_crops import photo_mask                                   # noqa: E402
from probe import P, STRIDE, EDGE, NEAR, PER_CROP, SIZE, colour_feats, texture_feats, train_cnn, predict  # noqa: E402

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[1] / "1_main_model_resnet18"
N_TRAIN = 3000            # decoy crops sampled from train (val and test: all)
K = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def to_crop(m, l, t, s):
    sub = np.zeros((s, s), np.uint8)
    part = m[max(t, 0):t + s, max(l, 0):l + s]
    sub[max(-t, 0):max(-t, 0) + part.shape[0], max(-l, 0):max(-l, 0) + part.shape[1]] = part
    return cv2.resize(sub, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)


def decoy_patches(rng):
    d = pd.read_csv(HERE / "decoys.csv")
    d1 = pd.read_csv(MAIN / "data/work/crops/D1.csv")
    d1 = d1.set_index(d1.crop_path)
    rects = d[d.rect.notna()].groupby("key").rect.apply(lambda s: [r for x in s for r in ast.literal_eval(x)])
    d = d[d.has_decoy]
    d = pd.concat([d[d.split == "train"].sample(min(N_TRAIN, (d.split == "train").sum()), random_state=0), d[d.split != "train"]])
    X, y, ids, sp = [], [], [], []
    for c in d.itertuples():
        src = d1.loc[c.source_crop]
        W, H = int(src.img_w), int(src.img_h)
        l, t, s = int(src.crop_left), int(src.crop_top), int(src.crop_side)
        fm = np.zeros((H, W), np.uint8)
        for x0, y0, x1, y1 in rects[c.key]:
            fm[y0:y1, x0:x1] = 255
        boxes = [b[1:] for b in read_yolo(d1_label_file(c.key), W, H)]
        m, bl = to_crop(fm, l, t, s), to_crop(photo_mask((H, W), boxes), l, t, s)
        fill = cv2.erode(m, K(EDGE)).astype(bool)
        real = ~cv2.dilate(m | bl, K(EDGE)).astype(bool)
        near = cv2.distanceTransform(255 - m, cv2.DIST_L2, 5) <= NEAR
        img = np.asarray(Image.open(HERE / c.crop).convert("RGB"))
        f, r = [], []
        for yy in range(0, SIZE - P + 1, STRIDE):
            for xx in range(0, SIZE - P + 1, STRIDE):
                win = np.s_[yy:yy + P, xx:xx + P]
                if fill[win].all():
                    f.append((yy, xx))
                elif real[win].all() and near[win].any():
                    r.append((yy, xx))
        n = min(len(f), len(r), PER_CROP)
        for lab, cand in ((1, f), (0, r)):
            for i in (rng.choice(len(cand), n, replace=False) if n else []):
                yy, xx = cand[i]
                X.append(img[yy:yy + P, xx:xx + P]); y.append(lab); ids.append(c.id); sp.append(c.split)
    return np.stack(X), np.array(y), np.array(ids), np.array(sp)


def main():
    rng = np.random.default_rng(0)
    Xh, yh, ih, sh, _ = probe.patches(rng)
    Xd, yd, idd, sd = decoy_patches(rng)
    print(f"healed patches {len(Xh)} from {len(set(ih))} crops, decoy patches {len(Xd)} from {len(set(idd))} crops")
    res = dict(patches=dict(healed=len(Xh), healed_crops=len(set(ih)), decoy=len(Xd), decoy_crops=len(set(idd))))
    trh, evh, evd = sh == "train", sh != "train", sd != "train"
    auc = lambda y, p: round(float(roc_auc_score(y, p)), 3)

    # A. trained on healed fill vs real; scored on healed (val+test) and decoy (val+test)
    A = {}
    Fh, Fd = texture_feats(Xh), texture_feats(Xd)
    gb = HistGradientBoostingClassifier(random_state=0).fit(Fh[trh], yh[trh])
    A["texture, boosting"] = dict(healed=auc(yh[evh], gb.predict_proba(Fh[evh])[:, 1]), decoy=auc(yd[evd], gb.predict_proba(Fd[evd])[:, 1]))
    net = train_cnn(Xh[trh], yh[trh], 0, False)
    A["cnn, no augmentation"] = dict(healed=auc(yh[evh], predict(net, Xh[evh])), decoy=auc(yd[evd], predict(net, Xd[evd])))
    res["A: healed-trained probe, fill vs real"] = A
    print("A", A)

    # B. healed fill vs decoy fill (label 1 = healed)
    fh, fd = yh == 1, yd == 1
    Xb = np.concatenate([Xh[fh], Xd[fd]]); yb = np.r_[np.ones(fh.sum()), np.zeros(fd.sum())].astype(int)
    sb = np.r_[sh[fh], sd[fd]]
    tr, ev = sb == "train", sb != "train"
    B = {}
    for name, F in (("colour only", colour_feats(Xb)), ("texture, boosting", texture_feats(Xb))):
        mdl = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)) if name == "colour only"
               else HistGradientBoostingClassifier(random_state=0)).fit(F[tr], yb[tr])
        B[name] = auc(yb[ev], mdl.predict_proba(F[ev])[:, 1])
    B["cnn, no augmentation"] = auc(yb[ev], predict(train_cnn(Xb[tr], yb[tr], 0, False), Xb[ev]))
    # same question for the untouched patches next to each fill: how much is just where the patches sit?
    rh, rd = yh == 0, yd == 0
    Xr = np.concatenate([Xh[rh], Xd[rd]]); yr = np.r_[np.ones(rh.sum()), np.zeros(rd.sum())].astype(int)
    sr = np.r_[sh[rh], sd[rd]]
    Fr = texture_feats(Xr)
    B["texture, boosting, untouched patches (baseline)"] = auc(yr[sr != "train"], HistGradientBoostingClassifier(random_state=0)
                                                              .fit(Fr[sr == "train"], yr[sr == "train"]).predict_proba(Fr[sr != "train"])[:, 1])
    res["B: healed fill vs decoy fill"] = B
    print("B", B)

    # C. fill share
    crops = pd.read_csv(HERE / "crops.csv")
    kept = pd.read_csv(HERE / "no_defect_healed.csv").crop_path.map(lambda p: Path(p).stem)
    hf = 1 - crops[crops.id.isin(kept)].real_frac
    d = pd.read_csv(HERE / "decoys.csv")
    q = [.1, .25, .5, .75, .9]
    res["C: fill share per crop"] = dict(healed_kept={str(k): round(float(v), 3) for k, v in hf.quantile(q).items()},
                                         decoy={str(k): round(float(v), 3) for k, v in d[d.has_decoy].decoy_frac.quantile(q).items()},
                                         decoy_coverage=round(float(d.has_decoy.mean()), 3),
                                         decoy_coverage_by_class=d.groupby("cls").has_decoy.mean().round(3).to_dict())
    print("C", res["C: fill share per crop"])
    (HERE / "probe/decoys_check.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
