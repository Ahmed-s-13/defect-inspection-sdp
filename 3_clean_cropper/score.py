"""Step 2 of clean_cropper: score every candidate for "background?" and "hidden damage?" -> scores.csv

  python score.py                 (caches CLIP features and YOLO detections in work/cache/; delete to redo)

No enhanced_res is used. Three scores per candidate:

  p_skin     CLIP (ViT-B/32, LAION-2B, zero-shot): probability the crop is aircraft surface rather than sky, plants,
             buildings, ground, hangar or people. Background is easy for a classifier ("not a plane = no defect")
             and says nothing about the hard case, clean skin next to a faint defect.
  p_damage   positive-unlabelled classifier: logistic regression on CLIP features, D1 defect crops (1) vs candidates
             (0). Each candidate is scored by a model trained without its group (5-fold GroupKFold), so a high score
             means "looks like the labelled defects", not "was memorised". Unboxed damage scores high.
  yolo_conf  optional: highest confidence of a D1 defect detector (a YOLO trained on d1_grouped train, placed at
             work/yolo_d1_grouped.pt; it is not in this repo, and `pip install ultralytics` is needed) among boxes at
             conf >= 0.05 lying mostly (>= 50% of their area) inside the square; 0 if none, empty without the detector.
             It fired on only 2.7% of candidates and its audit AUC was ~0.5, so pick.py does not use it.
"""
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")                     # CLIP weights are in the local HF cache

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent / "1_main_model_resnet18"
RAW = MAIN / "data/raw/D1/content/Aircraft_dataset/images"
GROUPS = MAIN / "d1_grouped/groups.csv"
YOLO = HERE / "work/yolo_d1_grouped.pt"     # optional (see README): a YOLO detector trained on d1_grouped train
CACHE = HERE / "work/cache"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

SKIN = ["a close-up photo of an aircraft's painted metal skin", "a close-up photo of an airplane fuselage panel with rivets",
        "a photo of the metal surface of an airplane wing", "a close-up photo of aircraft aluminium sheet metal",
        "a photo of an aircraft engine or landing gear part"]
BACKGROUND = ["a photo of the sky", "a photo of clouds", "a photo of grass, trees and bushes", "a photo of a building",
              "a photo of the ground, concrete or asphalt", "a photo of the inside of a hangar", "a photo of a person",
              "a photo of a car or vehicle", "a photo of a wall or fence"]


def clip_features(paths, name):
    f = CACHE / f"clip_{name}.npy"
    if f.exists():
        return np.load(f)
    import open_clip
    model, _, pre = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=DEV)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), 256):
            x = torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[i:i + 256]]).to(DEV)
            out.append(torch.nn.functional.normalize(model.encode_image(x), dim=-1).float().cpu().numpy())
            if i % 5120 == 0:
                print(f"  clip {name}: {i}/{len(paths)}", flush=True)
    out = np.concatenate(out)
    np.save(f, out)
    return out


def clip_text(prompts):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k", device=DEV)
    tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        t = model.encode_text(tok(prompts).to(DEV))
        return torch.nn.functional.normalize(t, dim=-1).float().cpu().numpy(), float(model.logit_scale.exp())


def yolo_detections(keys):
    f = CACHE / "yolo_dets.csv"
    if f.exists():
        return pd.read_csv(f)
    from ultralytics import YOLO as Y
    model = Y(str(YOLO))
    rows = []
    for i in range(0, len(keys), 64):
        batch = keys[i:i + 64]
        for key, r in zip(batch, model.predict([str(RAW / k) for k in batch], conf=0.05, imgsz=640, device=DEV, verbose=False)):
            for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
                rows.append(dict(key=key, x1=x1, y1=y1, x2=x2, y2=y2, conf=c, cls=r.names[int(k)]))
        if i % 1280 == 0:
            print(f"  yolo: {i}/{len(keys)} photos", flush=True)
    d = pd.DataFrame(rows, columns=["key", "x1", "y1", "x2", "y2", "conf", "cls"])
    d.to_csv(f, index=False)
    return d


def main():
    CACHE.mkdir(exist_ok=True)
    cand = pd.read_csv(HERE / "candidates.csv")
    d1 = pd.read_csv(MAIN / "data/work/crops/D1.csv")
    d1["key"] = d1.image.str.replace("content/Aircraft_dataset/images/", "", regex=False)
    d1 = d1.merge(pd.read_csv(GROUPS)[["key", "group"]].rename(columns={"group": "grp"}), on="key")

    Fc = clip_features([str(HERE / p) for p in cand.crop], "candidates")
    Fd = clip_features([str(MAIN / p) for p in d1.crop_path], "defects")

    # p_skin: zero-shot, all prompts in one softmax
    T, scale = clip_text(SKIN + BACKGROUND)
    p = torch.softmax(torch.from_numpy(scale * Fc @ T.T), 1).numpy()
    cand["p_skin"] = p[:, :len(SKIN)].sum(1).round(3)
    cand["clip_top"] = [(SKIN + BACKGROUND)[i] for i in p.argmax(1)]

    # p_damage: defect crops vs candidates, out-of-fold by group
    X = np.concatenate([Fd, Fc])
    y = np.r_[np.ones(len(Fd)), np.zeros(len(Fc))]
    g = np.r_[d1.grp.values, cand.group.values]
    oof = np.zeros(len(X))
    for tr, te in GroupKFold(5).split(X, y, g):
        lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(X[tr], y[tr])
        oof[te] = lr.predict_proba(X[te])[:, 1]
    cand["p_damage"] = oof[len(Fd):].round(3)
    print(f"defect crops vs candidates, out-of-fold AUC {roc_auc_score(y, oof):.3f}")

    # yolo_conf: detector boxes mostly inside the candidate square (optional, see the docstring)
    if not YOLO.exists() and not (CACHE / "yolo_dets.csv").exists():
        print(f"no detector at {YOLO}: yolo_conf left empty")
        cand["yolo_conf"], cand["yolo_cls"] = np.nan, ""
        return finish(cand)
    det = yolo_detections(sorted(cand.key.unique()))
    by_key = {k: v[["x1", "y1", "x2", "y2", "conf"]].values for k, v in det.groupby("key")}
    yc, ycls = [], []
    dcls = {k: v.cls.values for k, v in det.groupby("key")}
    for c in cand.itertuples():
        b = by_key.get(c.key)
        best, bc = 0.0, ""
        if b is not None:
            l, t, r_, bt = c.left, c.top, c.left + c.side, c.top + c.side
            iw = np.clip(np.minimum(b[:, 2], r_) - np.maximum(b[:, 0], l), 0, None)
            ih = np.clip(np.minimum(b[:, 3], bt) - np.maximum(b[:, 1], t), 0, None)
            inside = iw * ih / np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-6) >= 0.5
            if inside.any():
                j = np.argmax(np.where(inside, b[:, 4], -1))
                best, bc = float(b[j, 4]), dcls[c.key][j]
        yc.append(round(best, 3)); ycls.append(bc)
    cand["yolo_conf"], cand["yolo_cls"] = yc, ycls
    finish(cand)


def finish(cand):
    cand.to_csv(HERE / "scores.csv", index=False)
    print(f"{len(cand)} candidates scored")
    print("p_skin quantiles", cand.p_skin.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
    print("p_damage quantiles", cand.p_damage.quantile([.1, .25, .5, .75, .9]).round(3).to_dict())
    print("yolo hit share at 0.05 / 0.15 / 0.25:", [round(float((cand.yolo_conf >= t).mean()), 3) for t in (0.05, 0.15, 0.25)])
    print("top CLIP prompt:", cand.clip_top.value_counts().to_dict())


if __name__ == "__main__":
    main()
