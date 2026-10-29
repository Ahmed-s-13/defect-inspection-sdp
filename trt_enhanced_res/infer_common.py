"""What predict_trt.py (TensorRT, NVIDIA GPU) and predict_onnx.py (onnxruntime, any CPU, e.g. Raspberry Pi) share:
preprocessing, loading the decision layer, and the printing. numpy + Pillow only: no PyTorch, no OpenCV, no pandas.

A runtime is anything that turns a batch of normalised crops [N,3,224,224] into the model's three outputs
  probs [N,7]   class probabilities (corrosion, crack, dent, missing-head, paint-peel-off, scratch, no-defect)
  feats [N,768] the 8-view features the head works on
  knn   [N]     1 - mean cosine similarity to the 5 nearest training crops
and this file turns those into the visor's answer with architecture B (enhanced_res/arch_a/runtime.py).
"""
import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from trt_paths import HERE

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}      # = klshift.common
CROP_SIZE, MARGIN = 224, 0.25                                               # = klshift.common
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
SAMPLES = [HERE.parent / "1_main_model_resnet18" / "samples", HERE / "samples"]   # repo layout, Pi bundle layout
REFERENCE = HERE / "samples_reference.json"      # the PyTorch answers on those 30 samples (make_reference.py)


# ---------------------------------------------------------------- preprocessing (same as 1_main_model_resnet18/predict.py)
def square_crop_box(x1, y1, x2, y2, img_w, img_h, margin=MARGIN):
    """Copy of klshift.common.square_crop_box: square box + margin, shifted inward (never padded)."""
    side = int(round(min(max(x2 - x1, y2 - y1) * (1 + 2 * margin), img_w, img_h)))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(int(round(cx - side / 2)), 0), img_w - side)
    top = min(max(int(round(cy - side / 2)), 0), img_h - side)
    return left, top, side


def to_crop(path, box=None):
    """EXIF orientation, square crop (around --box, else the centre square), 224 px BICUBIC."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    if box:
        left, top, side = square_crop_box(*box, w, h)
    else:
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC)


def to_input(img):
    """PIL 224x224 RGB -> float32 [3,224,224], same as torchvision ToTensor + Normalize(ImageNet)."""
    return (np.asarray(img, np.float32).transpose(2, 0, 1) / 255.0 - MEAN) / STD


# ---------------------------------------------------------------- the decision layer
def load_arch(model="gc7", alpha=None):
    """(runtime module, ArchA) from the repo's enhanced_res/arch_a/ or, in the Pi bundle, from ./arch_a/.
    runtime.py is numpy-only; it is loaded by path so it works without the rest of the repo."""
    for d in (HERE.parent / "1_main_model_resnet18" / "enhanced_res" / "arch_a", HERE / "arch_a"):
        if (d / "runtime.py").exists():
            break
    else:
        raise SystemExit("arch_a/runtime.py not found: run from the repo (1_main_model_resnet18/enhanced_res/arch_a/) "
                         "or from the Pi bundle (arch_a/ next to this file)")
    spec = importlib.util.spec_from_file_location("arch_runtime", d / "runtime.py")
    rt = importlib.util.module_from_spec(spec)
    sys.modules["arch_runtime"] = rt
    spec.loader.exec_module(rt)
    return rt, rt.ArchA(rt.CALIBRATIONS[model], alpha=alpha)


def read_labels(folder):
    """{image path: class} from <folder>/labels.csv (columns file,cls), if there is one."""
    f = Path(folder) / "labels.csv"
    if not f.exists():
        return {}
    with open(f, newline="", encoding="utf-8-sig") as fh:
        return {str(Path(folder) / r["file"]): r["cls"] for r in csv.DictReader(fh)}


def collect(inputs):
    """Image files and folders -> (paths, labels)."""
    paths, labels = [], {}
    for p in map(Path, inputs):
        if p.is_dir():
            paths += sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXTS)
            labels.update(read_labels(p))
        else:
            paths.append(p)
    return paths, labels


def samples_dir():
    return next((d for d in SAMPLES if (d / "labels.csv").exists()), None)


def ranked(prob, classes, top):
    order = np.argsort(-prob)[:top]
    return "   ".join(f"{r}. {classes[k].upper() if r == 1 else classes[k]} {prob[k] * 100:.1f}%"
                       for r, k in enumerate(order, 1))


# ---------------------------------------------------------------- the prediction loop
def predict(infer, arch, rt, paths, labels, header, device_label, box=None, batch=16, raw=False, top=2,
            no_gates=False, details=False):
    """infer(x [N,3,224,224] float32) -> (probs, feats, knn). Prints like 1_main_model_resnet18/predict.py.
    Returns ({file name: Decision}, seconds)."""
    classes = arch.classes
    print(header + "\n")
    right = n_lab = 0
    acts, holds, results = [], [], {}
    t0 = time.time()
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        prob, feats, knn = infer(np.stack([to_input(to_crop(p, box)) for p in chunk]))
        dec = None if raw else arch.decide(prob, knn_dist=None if no_gates else knn, feats=None if no_gates else feats)
        for j, p in enumerate(chunk):
            best = int(prob[j].argmax())
            if dec:
                d = dec[j]
                results[p.name] = d
                line = f"{p.name}:  {d.text() or '(nothing shown: no damage)'}"
                if details:
                    line += (f"   | unfamiliar p {d.p_ood:.3f} | calibrated: "
                             + ranked(arch.sets(prob[j:j + 1])[0][0], classes, len(classes)))
            else:
                line = f"{p.name}:  {ranked(prob[j], classes, top)}"
            truth = labels.get(str(p))
            if truth is not None:
                n_lab += 1
                right += classes[best] == truth
                if dec:
                    acts.append(d.action)
                    holds.append(truth in d.classes)
                    line += f"   [true: {truth}{'' if (not d.classes or truth in d.classes) else '  <- not in the answer'}]"
                else:
                    line += f"   [true: {truth}{'' if classes[best] == truth else '  <- wrong'}]"
            print(line)
    secs = time.time() - t0
    if len(paths) > 1:
        print(f"\n{len(paths)} images in {secs:.1f}s on {device_label}")
    if n_lab:
        print(f"enhanced_res top-1 accuracy on the {n_lab} labelled images: {right / n_lab:.3f}")
        if acts:
            acts, holds = np.array(acts), np.array(holds)
            amb = acts == rt.AMBIGUOUS
            print(f"architecture {'B' if arch.nd is not None else 'A'}: "
                  + ", ".join(f"{k} {(acts == k).mean() * 100:.0f}%" for k in rt.ACTIONS if (acts == k).any())
                  + f" | single class right {(holds & (acts == rt.SHOW)).mean() * 100:.0f}%, "
                  f"wrong {(~holds & (acts == rt.SHOW)).mean() * 100:.0f}%"
                  + (f" | ambiguous sets holding the truth {holds[amb].mean() * 100:.0f}%" if amb.any() else ""))
    return results, secs


def selftest(results, min_same=29):
    """Compare the answers on the 30 samples with the PyTorch reference (samples_reference.json). Returns True if OK."""
    if not REFERENCE.exists():
        raise SystemExit(f"{REFERENCE.name} not found")
    ref = json.loads(REFERENCE.read_text())
    bad = [f for f, r in ref.items() if f not in results
           or results[f].action != r["action"] or sorted(results[f].classes) != sorted(r["classes"])]
    same = len(ref) - len(bad)
    print(f"\nSELFTEST: {same}/{len(ref)} samples give the same answer (action and classes) as the PyTorch reference"
          + (f"; different: {', '.join(bad)}" if bad else ""))
    ok = same >= min_same
    print("OK: this runtime matches the PyTorch model." if ok else
          "WARNING: this runtime differs from the PyTorch reference. Check the ONNX file and the versions.")
    return ok
