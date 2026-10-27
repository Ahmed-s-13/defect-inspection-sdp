"""predict.py on TensorRT: classify the defect in one or more images with the enhanced_res engine. No PyTorch needed.

  python predict_trt.py photo.jpg                          whole photo (centre square)
  python predict_trt.py photo.jpg --box 120 80 260 190     defect box in pixels: cut exactly like training
  python predict_trt.py ../1_main_model_resnet18/samples/  every image in a folder (+ accuracy if it has labels.csv)
  python predict_trt.py --grouped-test                     all 4,255 grouped D1-test crops: accuracy / macro-F1
  python predict_trt.py photo.jpg --precision fp16         the FP16 engine (faster)
  python predict_trt.py photo.jpg --raw                    the raw top-2 ranking, no architecture A (--top N)

Same steps and output as 1_main_model_resnet18/predict.py (enhanced_res + architecture A): EXIF orientation, square
crop (box + 25% margin, shifted inside the image), 224 px BICUBIC, ImageNet normalisation, then the model, then
architecture A (1_main_model_resnet18/enhanced_res/arch_a/: quality gate, temperature, conformal set; numpy only).
The model (backbone + 8-view TTA + MLP heads + 5-NN) runs entirely inside the TensorRT engine
(engines/enhanced_res_full_<precision>.engine, built by trt_runner.py from export_onnx.py). GPU memory and copies use
cuda-python. Needs: tensorrt, cuda-python, numpy, pillow, pandas.

Architecture A's unfamiliar-input (OOD) gate needs the model's 768-d features, which the current ONNX/engine does
not output ("probs" only), so it is skipped here; everything else matches predict.py. Adding a "features" output to
the export would enable it (runtime.knn_distance takes the features and the kNN bank).

The model has no "no damage" class: it always names one of the 6 defect types, even for a clean surface.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorrt as trt
from cuda.bindings import runtime as rt
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent / "1_main_model_resnet18"
sys.path.insert(0, str(MAIN))
from enhanced_res.arch_a.runtime import ACTIONS, ArchA  # noqa: E402  (numpy + PIL only)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}      # = klshift.common
CROP_SIZE, MARGIN = 224, 0.25                                               # = klshift.common
MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)
EXPECTED = {"accuracy": 0.8045, "macro_f1": 0.7376, "balanced_accuracy": 0.7367}  # repo evaluate.py, grouped test


def square_crop_box(x1, y1, x2, y2, img_w, img_h, margin=MARGIN):
    """Copy of klshift.common.square_crop_box: square box + margin, shifted inward (never padded)."""
    side = int(round(min(max(x2 - x1, y2 - y1) * (1 + 2 * margin), img_w, img_h)))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = min(max(int(round(cx - side / 2)), 0), img_w - side)
    top = min(max(int(round(cy - side / 2)), 0), img_h - side)
    return left, top, side


def to_crop(path, box=None):
    """Same as predict.py: EXIF orientation, square crop, 224 px BICUBIC."""
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


def check(res):
    err = res[0] if isinstance(res, tuple) else res
    if err != rt.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    return res[1] if isinstance(res, tuple) and len(res) == 2 else res[1:] if isinstance(res, tuple) else None


class TRTEngine:
    """Built enhanced_res engine on the GPU: x float32 [N,3,224,224] (numpy) -> probs float32 [N,6] (numpy).
    Profile 0 for single crops, profile 1 for batches of up to max_batch crops; bigger N is split into chunks."""

    def __init__(self, path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise SystemExit(f"could not load {path}: built with another TensorRT version or GPU? "
                             "Run `python trt_runner.py --rebuild`.")
        self.stream = check(rt.cudaStreamCreate())
        self.max_batch = self.engine.get_tensor_profile_shape("input", 1)[2][0]
        self.d_in = check(rt.cudaMalloc(self.max_batch * 3 * CROP_SIZE * CROP_SIZE * 4))
        self.d_out = check(rt.cudaMalloc(self.max_batch * 6 * 4))
        self.ctx = []
        for prof in (0, 1):
            ctx = self.engine.create_execution_context()
            ctx.set_optimization_profile_async(prof, int(self.stream))
            ctx.set_tensor_address("input", int(self.d_in))
            ctx.set_tensor_address("probs", int(self.d_out))
            self.ctx.append(ctx)
        check(rt.cudaStreamSynchronize(self.stream))

    def __call__(self, x):
        x = np.ascontiguousarray(x, np.float32)
        out = np.empty((len(x), 6), np.float32)
        for i in range(0, len(x), self.max_batch):
            xb, ob = x[i:i + self.max_batch], out[i:i + self.max_batch]
            ctx = self.ctx[0 if len(xb) == 1 else 1]
            ctx.set_input_shape("input", xb.shape)
            check(rt.cudaMemcpyAsync(self.d_in, xb.ctypes.data, xb.nbytes, rt.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
            if not ctx.execute_async_v3(int(self.stream)):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            check(rt.cudaMemcpyAsync(ob.ctypes.data, self.d_out, ob.nbytes, rt.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
            check(rt.cudaStreamSynchronize(self.stream))
        return out

    def close(self):
        for p in (self.d_in, self.d_out):
            rt.cudaFree(p)
        rt.cudaStreamDestroy(self.stream)


def grouped_test(model, classes, batch, arch=None):
    """All grouped D1-test crops (d1_grouped split, same list as the repo's evaluate.py), vs the PyTorch results."""
    f = MAIN / "data/work/enhanced_res/split.csv"
    if not f.exists():
        raise SystemExit(f"{f} not found: run the repo's enhanced_res/evaluate.py once (it writes it)")
    d = pd.read_csv(f)
    d = d[d.split == "test"]
    y = d.cls.map({c: i for i, c in enumerate(classes)}).values
    P, t0 = [], time.time()
    paths = [MAIN / p for p in d.crop_path]
    for i in range(0, len(paths), batch):
        P.append(model(np.stack([to_input(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])))
    prob = np.concatenate(P)
    p = prob.argmax(1)
    cm = np.zeros((6, 6), int)
    np.add.at(cm, (y, p), 1)
    tp = np.diag(cm).astype(float)
    f1 = np.where(cm.sum(0) + cm.sum(1) > 0, 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1), 0)
    m = {"accuracy": (p == y).mean(), "macro_f1": f1.mean(), "balanced_accuracy": (tp / cm.sum(1)).mean()}
    print(f"grouped D1-test: {len(y)} crops in {time.time() - t0:.0f}s")
    ok = True
    for k, v in m.items():
        same = abs(v - EXPECTED[k]) <= 0.003
        ok &= same
        print(f"  {k:20s} {v:.4f}   PyTorch {EXPECTED[k]:.4f}   {'match' if same else 'DIFFERENT'}")
    print("  per-class recall: " + ", ".join(f"{c} {r:.3f}" for c, r in zip(classes, tp / cm.sum(1))))
    print("\nOK: TensorRT matches the PyTorch enhanced_res results." if ok else "\nWARNING: differs from PyTorch.")
    if arch is not None:   # architecture A's CP set on the test half (no gates), vs calibration.json
        sp = pd.read_csv(MAIN / "enhanced_res/arch_a/cal_split.csv")
        half = d.merge(sp[["crop_path", "split"]].rename(columns={"split": "half"}), on="crop_path", how="left").half
        t = (half == "test").to_numpy()
        _, S = arch.sets(prob[t])
        exp = arch.cal.get("expected_test_half", {})
        cov, size = S[np.arange(t.sum()), y[t]].mean(), S.sum(1).mean()
        print(f"\narchitecture A, alpha {arch.alpha:.2f}, test half ({t.sum()} crops):")
        for k, v, e in [("CP coverage", cov, exp.get("cp_coverage")), ("mean set size", size, exp.get("cp_mean_set_size"))]:
            print(f"  {k:20s} {v:.4f}   PyTorch {e:.4f}   {'match' if abs(v - e) <= 0.005 else 'DIFFERENT'}"
                  if e is not None else f"  {k:20s} {v:.4f}")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="image files or folders")
    ap.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                    help="defect box in pixels of the original image (single image only)")
    ap.add_argument("--alpha", type=float, default=None, help="architecture A error rate: 0.05, 0.10 (default) or 0.20")
    ap.add_argument("--no-gates", action="store_true", help="skip the quality gate (CP set only)")
    ap.add_argument("--raw", action="store_true", help="the model's raw ranking instead of architecture A")
    ap.add_argument("--top", type=int, default=2, help="with --raw: how many classes to list")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--engine", help="engine file (default: engines/enhanced_res_full_<precision>.engine)")
    ap.add_argument("--grouped-test", action="store_true", help="evaluate on all grouped D1-test crops instead")
    ap.add_argument("--batch", type=int, default=64)
    a = ap.parse_args()
    engine = Path(a.engine) if a.engine else HERE / "engines" / f"enhanced_res_full_{a.precision}.engine"
    if not engine.exists():
        raise SystemExit(f"{engine} missing: run `python export_onnx.py` then `python trt_runner.py`")
    classes = json.loads((HERE / "onnx/classes.json").read_text())
    model = TRTEngine(engine)
    name = f"enhanced_res (TensorRT {trt.__version__}, {a.precision.upper()} engine)"
    arch = None if a.raw else ArchA(alpha=a.alpha)
    if arch is not None:
        assert arch.classes == classes, "calibration.json and onnx/classes.json have different class orders"
        name += f" + architecture A (alpha {arch.alpha:.2f}" + (", no gates)" if a.no_gates else
                                                                 f"; quality gate {arch.alpha_q}, no OOD gate)")
    if a.grouped_test:
        print(f"classifier: {name}\n")
        grouped_test(model, classes, a.batch, arch)
        return

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
        ap.error("no images found (or use --grouped-test)")
    print(f"device: cuda  |  classifier: {name}  |  {len(paths)} image(s)\n")

    right = n_lab = 0
    acts = []
    t0 = time.time()
    for i in range(0, len(paths), a.batch):
        chunk = paths[i:i + a.batch]
        crops = [to_crop(p, a.box) for p in chunk]
        prob = model(np.stack([to_input(c) for c in crops]))
        dec = arch.decide(prob, crops=None if a.no_gates else crops) if arch else None
        for j, p in enumerate(chunk):
            top = prob[j].argsort()[::-1][:a.top].tolist()
            best = top[0]
            if arch:
                line = f"{p.name}:  {dec[j].text()}"
            else:
                ranked = "   ".join(f"{r}. {classes[k].upper() if r == 1 else classes[k]} {prob[j, k] * 100:.1f}%"
                                    for r, k in enumerate(top, 1))
                line = f"{p.name}:  {ranked}"
            truth = labels.get(str(p))
            if truth is not None:
                n_lab += 1
                right += classes[best] == truth
                if arch:
                    acts.append(dec[j].action)
                    miss = dec[j].classes and truth not in dec[j].classes
                    line += f"   [true: {truth}{'  <- not in the answer' if miss else ''}]"
                else:
                    line += f"   [true: {truth}{'' if classes[best] == truth else '  <- wrong'}]"
            print(line)
    if len(paths) > 1:
        print(f"\n{len(paths)} images in {time.time() - t0:.1f}s on cuda (TensorRT)")
    if n_lab:
        print(f"enhanced_res top-1 accuracy on the {n_lab} labelled images: {right / n_lab:.3f}")
        if acts:
            acts = np.array(acts)
            print("architecture A: " + ", ".join(f"{k} {(acts == k).mean() * 100:.0f}%" for k in ACTIONS
                                                 if (acts == k).any()))
    model.close()


if __name__ == "__main__":
    main()
