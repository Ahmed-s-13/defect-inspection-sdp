"""Classify the defect in one or more images with the trained main model.

  python predict.py photo.jpg                          whole photo (centre square)
  python predict.py photo.jpg --box 120 80 260 190     defect box in pixels: cut exactly like training
  python predict.py data/work/crops/D1/                every image in a folder (e.g. ready-made crops)
  python predict.py samples/ --compare                 held-out test samples, with the ResNet fc answer next to it

The image goes through the ResNet-18 up to its 512-d embedding, then through the MLP head
(<work>/mlp_head.pt from train_mlp.py), which is more accurate than the ResNet's own fc layer
(D1-test accuracy 0.901 vs 0.880). If mlp_head.pt is missing, or with --resnet-only, the fc layer is used.
Runs on the Apple GPU (MPS) when there is one.

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+", help="image files or folders")
    ap.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                    help="defect box in pixels of the original image (single image only)")
    ap.add_argument("--work", default=str(Path(__file__).resolve().parents[1] / "data/d1_sscd_grouped/work"), help="folder with model_best.pt and mlp_head.pt")
    ap.add_argument("--top", type=int, default=3, help="how many classes to list per image")
    ap.add_argument("--resnet-only", action="store_true", help="use the ResNet's own fc layer instead of the MLP head")
    ap.add_argument("--compare", action="store_true", help="also print the ResNet fc answer next to the MLP head's")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--cpu", action="store_true", help="force CPU (default: Apple GPU / MPS)")
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
    if not (work / "model_best.pt").exists():
        raise SystemExit(f"missing {work / 'model_best.pt'}: unzip trained_models.zip in the repo root")

    device = torch.device("cpu") if a.cpu else pick_device()
    model, classes = load_model(work, device)
    head = None
    if not a.resnet_only:
        if (work / HEAD_FILE).exists():
            head = MLPClassifier(work / HEAD_FILE, device)
            assert head.classes == classes, "class order of the ResNet and the MLP head differ"
        else:
            print(f"note: {work / HEAD_FILE} not found, using the ResNet fc layer (run train_mlp.py for the MLP head)")
    name = "MLP head" if head else "ResNet fc"
    compare = a.compare and head is not None
    print(f"device: {device}  |  classifier: {name}  |  {len(paths)} image(s)\n")

    tf = eval_transform()
    right = right_fc = n_lab = 0
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(paths), a.batch):
            chunk = paths[i:i + a.batch]
            x = torch.stack([tf(to_crop(p, a.box)) for p in chunk]).to(device)
            emb = embed_batch(model, x)                                   # 512-d embedding
            prob_fc = torch.softmax(model.fc(emb), 1).cpu()               # the ResNet's own last layer
            prob = head(emb).cpu() if head else prob_fc                   # the MLP head
            for j, p in enumerate(chunk):
                top = prob[j].argsort(descending=True)[:a.top].tolist()
                best = top[0]
                others = ", ".join(f"{classes[k]} {prob[j, k] * 100:.1f}%" for k in top[1:])
                line = f"{p.name}: {classes[best].upper()}  {prob[j, best] * 100:.1f}%" + (f"   ({others})" if others else "")
                fb = int(prob_fc[j].argmax())
                if compare:
                    line += f"   | ResNet fc: {classes[fb]} {prob_fc[j, fb] * 100:.1f}%"
                truth = labels.get(str(p))
                if truth is not None:
                    n_lab += 1
                    right += classes[best] == truth
                    right_fc += classes[fb] == truth
                    line += f"   [true: {truth}{'' if classes[best] == truth else '  <- wrong'}]"
                print(line)
    sync(device)
    if len(paths) > 1:
        print(f"\n{len(paths)} images in {time.time() - t0:.1f}s on {device}")
    if n_lab:
        print(f"{name} accuracy on the {n_lab} labelled images: {right / n_lab:.3f}"
              + (f"   (ResNet fc: {right_fc / n_lab:.3f})" if compare else ""))


if __name__ == "__main__":
    main()
