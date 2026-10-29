"""predict.py on TensorRT: classify the defect in one or more images with the enhanced_res engine on an NVIDIA GPU.
No PyTorch needed (tensorrt + cuda-python + numpy + Pillow).

  python predict_trt.py photo.jpg                          whole photo (centre square)
  python predict_trt.py photo.jpg --box 120 80 260 190     defect box in pixels: cut exactly like training
  python predict_trt.py ../1_main_model_resnet18/samples/  every image in a folder (+ accuracy if it has labels.csv)
  python predict_trt.py --selftest                         the 30 samples vs the PyTorch answers (should be OK)
  python predict_trt.py photo.jpg --precision fp16         the FP16 engine (faster)
  python predict_trt.py photo.jpg --details                also the unfamiliar-input p-value and the full ranking
  python predict_trt.py photo.jpg --raw                    the raw top-2 ranking, no architecture B (--top N)

Same steps and output as 1_main_model_resnet18/predict.py (enhanced_res gc7 + architecture B): EXIF orientation, square
crop (box + 25% margin, shifted inside the image), 224 px BICUBIC, ImageNet normalisation, the model, then architecture B
(enhanced_res/arch_a/: unfamiliar-input gate, temperature, conformal set; numpy only). The whole model (backbone +
8-view TTA + MLP heads + 5-NN) runs inside the TensorRT engine (engines/enhanced_res_gc7_full_<precision>.engine, built
by trt_runner.py from the ONNX file); it also returns the features and kNN distance the unfamiliar-input gate needs.

What the answers mean (details: 1_main_model_resnet18/enhanced_res/arch_a/README.md):
  (nothing shown: no damage)      the set is only no-defect
  POSSIBLE DENT: check            the set holds a defect and no-defect
  CAN'T ASSESS: re-aim            unfamiliar input: not aircraft skin, or unlike any training crop
  CRACK 88.0%                     one class in the conformal set; the % is temperature-calibrated
  AMBIGUOUS: crack 52% dent 31%   2+ classes in the set: inspector reviews
  UNCLEAR: re-inspect             empty set
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as rt

from infer_common import CROP_SIZE, collect, load_arch, predict, samples_dir, selftest
from trt_paths import MODEL, engine_path

N_FEATS = 768


def check(res):
    err = res[0] if isinstance(res, tuple) else res
    if err != rt.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")
    return res[1] if isinstance(res, tuple) and len(res) == 2 else res[1:] if isinstance(res, tuple) else None


class TRTEngine:
    """Built enhanced_res engine on the GPU: x float32 [N,3,224,224] (numpy) -> (probs [N,C], feats [N,768], knn [N]).
    Profile 0 for single crops, profile 1 for batches of up to max_batch crops; bigger N is split into chunks."""

    def __init__(self, path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise SystemExit(f"could not load {path}: built with another TensorRT version or GPU? "
                             "Run `python trt_runner.py --rebuild`.")
        self.stream = check(rt.cudaStreamCreate())
        self.max_batch = self.engine.get_tensor_profile_shape("input", 1)[2][0]
        self.sizes = {"probs": self.engine.get_tensor_shape("probs")[1], "feats": N_FEATS, "knn_dist": 1}
        self.d_in = check(rt.cudaMalloc(self.max_batch * 3 * CROP_SIZE * CROP_SIZE * 4))
        self.d_out = {k: check(rt.cudaMalloc(self.max_batch * n * 4)) for k, n in self.sizes.items()}
        self.ctx = []
        for prof in (0, 1):
            ctx = self.engine.create_execution_context()
            ctx.set_optimization_profile_async(prof, int(self.stream))
            ctx.set_tensor_address("input", int(self.d_in))
            for k, ptr in self.d_out.items():
                ctx.set_tensor_address(k, int(ptr))
            self.ctx.append(ctx)
        check(rt.cudaStreamSynchronize(self.stream))

    def __call__(self, x):
        x = np.ascontiguousarray(x, np.float32)
        out = {k: np.empty((len(x), n), np.float32) for k, n in self.sizes.items()}
        for i in range(0, len(x), self.max_batch):
            xb = x[i:i + self.max_batch]
            ctx = self.ctx[0 if len(xb) == 1 else 1]
            ctx.set_input_shape("input", xb.shape)
            check(rt.cudaMemcpyAsync(self.d_in, xb.ctypes.data, xb.nbytes, rt.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
            if not ctx.execute_async_v3(int(self.stream)):
                raise RuntimeError("TensorRT execute_async_v3 failed")
            for k in self.sizes:
                ob = out[k][i:i + len(xb)]
                check(rt.cudaMemcpyAsync(ob.ctypes.data, self.d_out[k], ob.nbytes, rt.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
            check(rt.cudaStreamSynchronize(self.stream))
        return out["probs"], out["feats"], out["knn_dist"][:, 0]

    def close(self):
        for p in (self.d_in, *self.d_out.values()):
            rt.cudaFree(p)
        rt.cudaStreamDestroy(self.stream)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="*", help="image files or folders")
    ap.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"),
                    help="defect box in pixels of the original image (single image only)")
    ap.add_argument("--alpha", type=float, default=None, help="architecture B error rate: 0.05, 0.10 (default) or 0.20")
    ap.add_argument("--no-gates", action="store_true", help="skip the unfamiliar-input gate (conformal set only)")
    ap.add_argument("--details", action="store_true", help="also print the gate p-value and the full calibrated ranking")
    ap.add_argument("--raw", action="store_true", help="the model's raw ranking instead of architecture B")
    ap.add_argument("--top", type=int, default=2, help="with --raw: how many classes to list")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--engine", help="engine file (default: engines/enhanced_res_gc7_full_<precision>.engine)")
    ap.add_argument("--selftest", action="store_true", help="run the 30 samples and compare with the PyTorch answers")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    engine = Path(a.engine) if a.engine else engine_path(a.precision)
    if not engine.exists():
        raise SystemExit(f"{engine} missing: run `python trt_runner.py` (needs the ONNX files: see README.md)")
    if a.selftest:
        d = samples_dir()
        if d is None:
            raise SystemExit("samples/ not found")
        a.images = [str(d)]
    paths, labels = collect(a.images)
    if a.box and len(paths) != 1:
        ap.error("--box needs exactly one image")
    if not paths:
        ap.error("no images found (or use --selftest)")
    rt_arch, arch = load_arch(MODEL, alpha=a.alpha)
    model = TRTEngine(engine)
    name = f"enhanced_res {MODEL} (TensorRT {trt.__version__}, {a.precision.upper()} engine)"
    if not a.raw:
        name += (f" + architecture {'B' if arch.nd is not None else 'A'} (alpha {arch.alpha:.2f}"
                 + (", no gates)" if a.no_gates else f"; unfamiliar-input gate {arch.alpha_ood})"))
    results, _ = predict(model, arch, rt_arch, paths, labels, f"device: cuda  |  classifier: {name}  |  {len(paths)} image(s)",
                         "cuda (TensorRT)", a.box, a.batch, a.raw, a.top, a.no_gates, a.details)
    model.close()
    if a.selftest and not selftest(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
