"""Classify the defect in one or more images with the trained main model.

  python predict.py photo.jpg                          whole photo (centre square)
  python predict.py photo.jpg --box 120 80 260 190     defect box in pixels: cut exactly like training
  python predict.py data/work/crops/D1/                every image in a folder (e.g. ready-made crops)

The model was trained on square crops around a defect (box + 25% margin, 224 px), so
--box gives the most faithful result. Without it the centre square of the image is used,
which works for crops and close-ups but not for wide shots.

The model has no "no damage" class: it always names one of the 6 defect types, even
for a clean surface. Read the output as "if this is a defect, which one".
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from PIL import Image, ImageOps

from evaluate import load_model
from klshift.common import CROP_SIZE, IMG_EXTS, eval_transform, pick_device, square_crop_box


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
    ap.add_argument("--work", default="data/work", help="folder with model_best.pt")
    ap.add_argument("--top", type=int, default=3, help="how many classes to list per image")
    a = ap.parse_args()
    paths = []
    for p in map(Path, a.images):
        paths += sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS) if p.is_dir() else [p]
    if a.box and len(paths) != 1:
        ap.error("--box needs exactly one image")
    if not (Path(a.work) / "model_best.pt").exists():
        raise SystemExit(f"missing {Path(a.work) / 'model_best.pt'}: unzip trained_models.zip in the repo root")

    device = pick_device()
    model, classes = load_model(a.work, device)
    tf = eval_transform()
    with torch.no_grad():
        for p in paths:
            x = tf(to_crop(p, a.box))[None].to(device)
            prob = torch.softmax(model(x), 1)[0].cpu()
            top = prob.argsort(descending=True)[:a.top]
            best = top[0].item()
            others = ", ".join(f"{classes[i]} {prob[i] * 100:.1f}%" for i in top[1:].tolist())
            print(f"{p.name}: {classes[best].upper()}  {prob[best] * 100:.1f}%" + (f"   ({others})" if others else ""))


if __name__ == "__main__":
    main()
