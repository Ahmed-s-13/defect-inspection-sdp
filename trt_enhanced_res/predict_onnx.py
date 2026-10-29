"""predict.py with onnxruntime: classify the defect in images on any CPU (your laptop, a Raspberry Pi 5). No PyTorch,
no TensorRT, no GPU: onnxruntime + numpy + Pillow. It reads the same ONNX file TensorRT is built from.

  python predict_onnx.py photo.jpg                         whole photo (centre square)
  python predict_onnx.py photo.jpg --box 120 80 260 190    defect box in pixels: cut exactly like training
  python predict_onnx.py samples/                          every image in a folder (+ accuracy if it has labels.csv)
  python predict_onnx.py --selftest                        the 30 samples vs the PyTorch answers (should be OK)
  python predict_onnx.py --bench 20                        speed: 20 single-crop runs (median ms per crop of 8 views)
  python predict_onnx.py photo.jpg --threads 4             onnxruntime threads (default: all cores)
  python predict_onnx.py photo.jpg --details | --raw       see predict_trt.py

Same steps and output as 1_main_model_resnet18/predict.py and predict_trt.py (enhanced_res gc7 + architecture B). Use the
FP32 file (default): the FP16 one is for TensorRT and is not faster on a CPU.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from infer_common import collect, load_arch, predict, samples_dir, selftest
from trt_paths import MODEL, onnx_path


class ORTModel:
    """onnxruntime session: x float32 [N,3,224,224] (numpy) -> (probs [N,C], feats [N,768], knn [N])."""

    def __init__(self, path, threads=0):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads                  # 0 = onnxruntime's default (all cores)
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.version, self.threads = ort.__version__, threads

    def __call__(self, x):
        return self.sess.run(["probs", "feats", "knn_dist"], {"input": np.ascontiguousarray(x, np.float32)})


def bench(model, n):
    x = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)
    for _ in range(3):
        model(x)
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        model(x)
        t.append((time.perf_counter() - t0) * 1000)
    t = np.array(t)
    print(f"speed, model only, 1 crop = 8 views: median {np.median(t):.0f} ms, min {t.min():.0f}, "
          f"90th percentile {np.percentile(t, 90):.0f}  ({1000 / np.median(t):.1f} crops/s, {n} runs)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="image files or folders")
    ap.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                    help="defect box in pixels of the original image (single image only)")
    ap.add_argument("--onnx", help="ONNX file (default: onnx/enhanced_res_gc7_full_fp32.onnx)")
    ap.add_argument("--threads", type=int, default=0, help="onnxruntime threads (default 0 = all cores)")
    ap.add_argument("--alpha", type=float, default=None, help="architecture B error rate: 0.05, 0.10 (default) or 0.20")
    ap.add_argument("--no-gates", action="store_true", help="skip the unfamiliar-input gate (conformal set only)")
    ap.add_argument("--details", action="store_true", help="also print the gate p-value and the full calibrated ranking")
    ap.add_argument("--raw", action="store_true", help="the model's raw ranking instead of architecture B")
    ap.add_argument("--top", type=int, default=2, help="with --raw: how many classes to list")
    ap.add_argument("--selftest", action="store_true", help="run the 30 samples and compare with the PyTorch answers")
    ap.add_argument("--bench", type=int, metavar="N", help="time N single-crop runs")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    path = Path(a.onnx) if a.onnx else onnx_path("fp32")
    if not path.exists():
        raise SystemExit(f"{path} missing: see README.md (download enhanced_res_gc7_onnx.zip, or run export_onnx.py)")
    model = ORTModel(path, a.threads)
    if a.bench:
        bench(model, a.bench)
        if not (a.images or a.selftest):
            return
    if a.selftest:
        d = samples_dir()
        if d is None:
            raise SystemExit("samples/ not found")
        a.images = [str(d)]
    paths, labels = collect(a.images)
    if a.box and len(paths) != 1:
        ap.error("--box needs exactly one image")
    if not paths:
        ap.error("no images found (or use --selftest / --bench)")
    rt_arch, arch = load_arch(MODEL, alpha=a.alpha)
    name = f"enhanced_res {MODEL} (onnxruntime {model.version}, {path.stem.rsplit('_', 1)[-1].upper()})"
    if not a.raw:
        name += (f" + architecture {'B' if arch.nd is not None else 'A'} (alpha {arch.alpha:.2f}"
                 + (", no gates)" if a.no_gates else f"; unfamiliar-input gate {arch.alpha_ood})"))
    dev = f"cpu (onnxruntime, {a.threads or 'all'} threads)"
    results, _ = predict(model, arch, rt_arch, paths, labels, f"device: {dev}  |  classifier: {name}  |  {len(paths)} image(s)",
                         dev, a.box, a.batch, a.raw, a.top, a.no_gates, a.details)
    if a.selftest and not selftest(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
