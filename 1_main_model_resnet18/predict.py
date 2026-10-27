"""Classify the defect in one or more images with the main architecture: enhanced_res + architecture A.

  python predict.py photo.jpg                          whole photo (centre square)
  python predict.py photo.jpg --box 120 80 260 190     defect box in pixels: cut exactly like training
  python predict.py data/work/crops/D1/                every image in a folder (e.g. ready-made crops)
  python predict.py samples/ --compare                 test samples, with the old model's answer next to it
  python predict.py photo.jpg --details                also print the gate p-values and the full ranking
  python predict.py photo.jpg --raw                    the model's raw top-2 ranking, no architecture A (--top N)
  python predict.py photo.jpg --legacy                 the old ResNet-18 + MLP head instead (raw ranking)

The model is enhanced_res (enhanced_res/README.md): a ResNet-18 trained on the leak-free d1_grouped split, whose
layer3 + layer4 features, averaged over the crop's 8 flips/rotations, go to an MLP-ensemble + kNN head. Grouped
D1-test accuracy 0.804 (macro-F1 0.738). Weights: data/work/enhanced_res/ (enhanced_res_model.zip, GitHub Release
v2.0-enhanced-res).

Architecture A (enhanced_res/arch_a/README.md) turns its probabilities into one of five answers, with numbers fitted
on d1_grouped (enhanced_res/arch_a/calibration.json); the model itself is unchanged:
  BAD IMAGE: re-take          quality gate: blurry, too dark / bright, glare (about 10% of normal photo families)
  CAN'T ASSESS: re-aim        unfamiliar-input gate: unlike any training crop (about 10% of normal families)
  CRACK 88.0%                 one class in the conformal set; the % is temperature-calibrated
  AMBIGUOUS: crack 52% dent 31%   2+ classes in the set: inspector reviews
  UNCLEAR: re-inspect         empty set
The set contains the true class about 90% of the time on normal crops (alpha 0.10; --alpha 0.05 / 0.20).

The model was trained on square crops around a defect (box + 25% margin, 224 px), so --box gives the most faithful
result. Without it the centre square of the image is used, which works for crops and close-ups but not for wide
shots. There is no "no damage" class: on a clean surface the answer means "if this is a defect, which one".

If a folder contains labels.csv (file,cls), accuracy and how often the answer holds the true class are printed.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

from evaluate import load_model
from enhanced_res.arch_a.runtime import ACTIONS, AMBIGUOUS, SHOW, ArchA
from enhanced_res.model import BACKBONE_FILE, EWORK, HEAD_FILE as ENH_HEAD, EnhancedRes
from klshift.common import CROP_SIZE, IMG_EXTS, embed_batch, eval_transform, pick_device, square_crop_box
from klshift.mlp import HEAD_FILE, MLPClassifier, sync


def to_crop(path, box=None):
    """Same steps as prepare.py: EXIF orientation, square crop, 224 px BICUBIC."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    if box:
        left, top, side = square_crop_box(*box, w, h)
    else:
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC)


class Legacy:
    """The earlier main model: ResNet-18 (model_best.pt) + MLP head (mlp_head.pt), or its own fc layer."""

    def __init__(self, work, device, resnet_only=False):
        if not (work / "model_best.pt").exists():
            raise SystemExit(f"missing {work / 'model_best.pt'}: unzip trained_models.zip in the repo root")
        self.model, self.classes = load_model(work, device)
        self.head, self.device = None, device
        if not resnet_only:
            if (work / HEAD_FILE).exists():
                self.head = MLPClassifier(work / HEAD_FILE, device)
                assert self.head.classes == self.classes, "class order of the ResNet and the MLP head differ"
            else:
                print(f"note: {work / HEAD_FILE} not found, using the ResNet fc layer (run train_mlp.py for the MLP head)")
        self.name = "legacy ResNet-18 + MLP head" if self.head else "legacy ResNet-18 fc"

    @torch.no_grad()
    def __call__(self, x):
        emb = embed_batch(self.model, x.to(self.device))
        return self.head(emb) if self.head else torch.softmax(self.model.fc(emb), 1)


def ranked(prob, classes, top):
    order = np.argsort(-prob)[:top]
    return "   ".join(f"{r}. {classes[k].upper() if r == 1 else classes[k]} {prob[k] * 100:.1f}%"
                       for r, k in enumerate(order, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="image files or folders")
    ap.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                    help="defect box in pixels of the original image (single image only)")
    ap.add_argument("--work", default="data/work", help="folder with enhanced_res/ and the legacy model files")
    ap.add_argument("--alpha", type=float, default=None,
                    help="architecture A error rate: 0.05, 0.10 (default) or 0.20; lower = larger sets, fewer misses")
    ap.add_argument("--no-gates", action="store_true", help="skip the quality and unfamiliar-input gates (CP set only)")
    ap.add_argument("--details", action="store_true", help="also print gate p-values and the full calibrated ranking")
    ap.add_argument("--raw", action="store_true", help="the model's raw ranking instead of architecture A")
    ap.add_argument("--top", type=int, default=2, help="with --raw / --legacy: how many classes to list")
    ap.add_argument("--legacy", action="store_true", help="use the earlier ResNet-18 + MLP head (original D1 split)")
    ap.add_argument("--resnet-only", action="store_true", help="with --legacy: the ResNet's own fc layer, no MLP head")
    ap.add_argument("--compare", action="store_true", help="also print the legacy model's answer")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cpu", action="store_true", help="force CPU (default: CUDA / Apple GPU)")
    a = ap.parse_args()
    paths, labels = [], {}
    for p in map(Path, a.images):
        if p.is_dir():
            paths += sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)
            if (p / "labels.csv").exists():
                lab = pd.read_csv(p / "labels.csv")
                labels.update({str(p / f): c for f, c in zip(lab.file, lab.cls)})
        else:
            paths.append(p)
    if a.box and len(paths) != 1:
        ap.error("--box needs exactly one image")
    if not paths:
        ap.error("no images found")
    work = Path(a.work)
    device = torch.device("cpu") if a.cpu else pick_device()
    ework = work / EWORK.name
    if a.legacy or a.resnet_only:
        model = Legacy(work, device, a.resnet_only)
        name = model.name
    elif (ework / BACKBONE_FILE).exists() and (ework / ENH_HEAD).exists():
        model, name = EnhancedRes(ework, device), "enhanced_res"
    else:
        print(f"note: {ework} has no enhanced_res weights (enhanced_res_model.zip), using the legacy model")
        model = Legacy(work, device)
        name = model.name
    arch = None
    if name == "enhanced_res" and not a.raw:
        arch = ArchA(alpha=a.alpha)
        assert arch.classes == model.classes, "calibration.json was made for another class order: re-run calibrate.py"
        name += (f" + architecture A (alpha {arch.alpha:.2f}"
                 + (", no gates)" if a.no_gates else f"; gates {arch.alpha_q} / {arch.alpha_ood})"))
    other = Legacy(work, device) if a.compare and model.__class__ is EnhancedRes else None
    classes = model.classes
    print(f"device: {device}  |  classifier: {name}  |  {len(paths)} image(s)\n")

    tf = eval_transform()
    right = right_o = n_lab = 0
    acts, holds = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(paths), a.batch):
            chunk = paths[i:i + a.batch]
            crops = [to_crop(p, a.box) for p in chunk]
            x = torch.stack([tf(c) for c in crops]).to(device)
            if arch:
                f = model.features(x)
                prob = model.proba_from_features(f).cpu().numpy()
                dec = arch.decide(prob, crops=None if a.no_gates else crops,
                                  knn_dist=None if a.no_gates else model.knn_distance(f).cpu().numpy())
            else:
                prob = model(x).cpu().numpy()
            prob_o = other(x).cpu().numpy() if other else None
            for j, p in enumerate(chunk):
                best = int(prob[j].argmax())
                if arch:
                    d = dec[j]
                    line = f"{p.name}:  {d.text()}"
                    if a.details:
                        line += (f"   | quality p {d.p_quality:.3f}, unfamiliar p {d.p_ood:.3f} | calibrated: "
                                 + ranked(arch.sets(prob[j:j + 1])[0][0], classes, len(classes)))
                else:
                    line = f"{p.name}:  {ranked(prob[j], classes, a.top)}"
                ob = int(prob_o[j].argmax()) if other else None
                if other:
                    line += f"   | legacy: {classes[ob]} {prob_o[j, ob] * 100:.1f}%"
                truth = labels.get(str(p))
                if truth is not None:
                    n_lab += 1
                    right += classes[best] == truth
                    right_o += other is not None and classes[ob] == truth
                    if arch:
                        acts.append(d.action)
                        holds.append(truth in d.classes)
                        ok = truth in d.classes if d.classes else None
                        mark = "" if ok is None else "" if ok else "  <- not in the answer"
                    else:
                        mark = "" if classes[best] == truth else "  <- wrong"
                    line += f"   [true: {truth}{mark}]"
                print(line)
    sync(device)
    if len(paths) > 1:
        print(f"\n{len(paths)} images in {time.time() - t0:.1f}s on {device}")
    if n_lab:
        print(f"{name.split(' + ')[0]} top-1 accuracy on the {n_lab} labelled images: {right / n_lab:.3f}"
              + (f"   (legacy: {right_o / n_lab:.3f})" if other else ""))
        if arch:
            acts, holds = np.array(acts), np.array(holds)
            share = {k: (acts == k).mean() for k in ACTIONS}
            amb = acts == AMBIGUOUS
            print("architecture A: " + ", ".join(f"{k} {v * 100:.0f}%" for k, v in share.items() if v)
                  + f" | single class right {(holds & (acts == SHOW)).mean() * 100:.0f}%, "
                  f"wrong {(~holds & (acts == SHOW)).mean() * 100:.0f}%"
                  + (f" | ambiguous sets holding the truth {holds[amb].mean() * 100:.0f}%" if amb.any() else ""))


if __name__ == "__main__":
    main()
