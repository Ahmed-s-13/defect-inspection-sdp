"""Classify the defect in one or more images with the main model (enhanced_res).

  python predict.py photo.jpg                          whole photo (centre square)
  python predict.py photo.jpg --box 120 80 260 190     defect box in pixels: cut exactly like training
  python predict.py data/work/crops/D1/                every image in a folder (e.g. ready-made crops)
  python predict.py samples/ --compare                 test samples, with the old model's answer next to it
  python predict.py photo.jpg --legacy                 the old ResNet-18 + MLP head instead

The main model is enhanced_res (enhanced_res/README.md): a ResNet-18 trained on the leak-free d1_grouped split,
whose layer3 + layer4 features, averaged over the crop's 8 flips/rotations, go to an MLP-ensemble + kNN head.
Grouped D1-test accuracy 0.804 (macro-F1 0.738). Weights: data/work/enhanced_res/ (enhanced_res_model.zip).

--legacy uses the earlier model (data/work/model_best.pt + mlp_head.pt), trained on D1's original split. Its
0.901 test accuracy is inflated by copies of training photos in that test set; on d1_grouped the same recipe
scores 0.781. If the enhanced_res weights are missing, the legacy model is used with a note.

The model was trained on square crops around a defect (box + 25% margin, 224 px), so
--box gives the most faithful result. Without it the centre square of the image is used,
which works for crops and close-ups but not for wide shots.

The model has no "no damage" class: it always names one of the 6 defect types, even
for a clean surface. Read the output as "if this is a defect, which one".

If a folder contains labels.csv (file,cls), the accuracy on that folder is printed at the end.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch
from PIL import Image, ImageOps

from evaluate import load_model
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="image files or folders")
    ap.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                    help="defect box in pixels of the original image (single image only)")
    ap.add_argument("--work", default="data/work", help="folder with enhanced_res/ and the legacy model files")
    ap.add_argument("--top", type=int, default=3, help="how many classes to list per image")
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
    other = Legacy(work, device) if a.compare and name == "enhanced_res" else None
    classes = model.classes
    print(f"device: {device}  |  classifier: {name}  |  {len(paths)} image(s)\n")

    tf = eval_transform()
    right = right_o = n_lab = 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(paths), a.batch):
            chunk = paths[i:i + a.batch]
            x = torch.stack([tf(to_crop(p, a.box)) for p in chunk]).to(device)
            prob = model(x).cpu()
            prob_o = other(x).cpu() if other else None
            for j, p in enumerate(chunk):
                top = prob[j].argsort(descending=True)[:a.top].tolist()
                best = top[0]
                others = ", ".join(f"{classes[k]} {prob[j, k] * 100:.1f}%" for k in top[1:])
                line = f"{p.name}: {classes[best].upper()}  {prob[j, best] * 100:.1f}%" + (f"   ({others})" if others else "")
                ob = int(prob_o[j].argmax()) if other else None
                if other:
                    line += f"   | legacy: {classes[ob]} {prob_o[j, ob] * 100:.1f}%"
                truth = labels.get(str(p))
                if truth is not None:
                    n_lab += 1
                    right += classes[best] == truth
                    right_o += other is not None and classes[ob] == truth
                    line += f"   [true: {truth}{'' if classes[best] == truth else '  <- wrong'}]"
                print(line)
    sync(device)
    if len(paths) > 1:
        print(f"\n{len(paths)} images in {time.time() - t0:.1f}s on {device}")
    if n_lab:
        print(f"{name} accuracy on the {n_lab} labelled images: {right / n_lab:.3f}"
              + (f"   (legacy: {right_o / n_lab:.3f})" if other else ""))


if __name__ == "__main__":
    main()
