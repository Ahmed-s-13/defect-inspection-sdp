"""Step 1: export the WHOLE enhanced_res (backbone + 8-view TTA + MLP heads + 5-NN) to ONNX, FP32 and FP16.

  python export_onnx.py                 the main model, gc7 (7 classes) -> onnx/enhanced_res_gc7_full_{fp32,fp16}.onnx

  input  "input"     float32 [N,3,224,224]   ImageNet-normalised 224 px crops (N = 1..any)
  output "probs"     float32 [N,7]           class probabilities, same as the repo's EnhancedRes(x)
  output "feats"     float32 [N,768]         the 8-view features (architecture B's AoN gate reads them)
  output "knn_dist"  float32 [N]             1 - mean cosine similarity to the 5 nearest training crops

The FP32 file runs everywhere: onnxruntime on any CPU (laptop, Raspberry Pi) and TensorRT (FP32 engine). The FP16 file
(backbone in half precision; head, kNN bank and outputs stay float32) is only for building TensorRT FP16 engines:
TensorRT 11 takes the precision from the ONNX file.

Then each file is checked against the repo's PyTorch model with onnxruntime (CPU) on 64 test crops (clean skin included):
probabilities, features, kNN distance and the answers of architecture B.
"""
import argparse

import numpy as np
import torch
from PIL import Image

from erc import FullModel, er, normalize
from infer_common import load_arch
from trt_paths import MODEL, ONNX_DIR, onnx_path


def export(model, path, opset=18):
    x = torch.randn(2, 3, 224, 224)
    kw = dict(input_names=["input"], output_names=["probs", "feats", "knn_dist"], opset_version=opset)
    try:  # dynamo exporter first
        torch.onnx.export(model, (x,), path, dynamo=True, dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1)}},
                          external_data=False, **kw)
        return "dynamo"
    except Exception as e:
        print(f"  dynamo exporter failed ({type(e).__name__}: {str(e)[:150]}), using the TorchScript exporter")
        torch.onnx.export(model, (x,), path, dynamo=False, do_constant_folding=True,
                          dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"}, "feats": {0: "batch"},
                                        "knn_dist": {0: "batch"}}, **kw)
        return "torchscript"


def same_answer(a, b):
    return a.action == b.action and sorted(a.classes) == sorted(b.classes)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL, choices=sorted(er.MODELS))
    ap.add_argument("--precisions", nargs="+", default=["fp32", "fp16"], choices=["fp32", "fp16"])
    ap.add_argument("--n-check", type=int, default=64, help="test crops to check each file on")
    a = ap.parse_args()
    import onnx
    import onnxruntime as ort

    ONNX_DIR.mkdir(exist_ok=True)
    rt, arch = load_arch(a.model)
    ref = er.EnhancedRes(er.MODELS[a.model], device=torch.device("cpu"))        # the repo's own model = the reference
    assert ref.classes == arch.classes, "the calibration file and the model have different class orders"

    crops = er.load_split("d1_gc7_fix")                                         # defects and no-defect crops
    test = crops[crops.split == "test"].reset_index(drop=True)
    test = test.iloc[:: max(1, len(test) // a.n_check)].head(a.n_check)
    print(f"checking on {len(test)} test crops ({(test.cls == 'no-defect').sum()} no-defect)")
    x = np.stack([np.asarray(Image.open(p).convert("RGB")).transpose(2, 0, 1) for p in test.crop_path])
    xb = normalize(torch.from_numpy(x)).contiguous()
    with torch.no_grad():
        f = ref.features(xb)
        rp, rf, rk = ref.proba_from_features(f).numpy(), f.numpy(), ref.knn_distance(f).numpy()
    rdec = arch.decide(rp, knn_dist=rk, feats=rf)

    for prec in a.precisions:
        net = FullModel(a.model, half=prec == "fp16")
        if prec == "fp32":
            with torch.no_grad():
                p, f_, k_ = (t.numpy() for t in net(xb))
            print(f"FullModel (PyTorch) vs repo EnhancedRes: max |prob diff| {np.abs(p - rp).max():.2e}, "
                  f"|feats| {np.abs(f_ - rf).max():.2e}, |knn| {np.abs(k_ - rk).max():.2e}")
        path = onnx_path(prec, a.model)
        print(f"exporting {path.name} ...", flush=True)
        how = export(net, str(path))
        onnx.checker.check_model(str(path))
        print(f"  exported with the {how} exporter, {path.stat().st_size / 1e6:.1f} MB")
        try:
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            for n in (1, len(xb)):  # the batch size really is dynamic
                p, f_, k_ = sess.run(["probs", "feats", "knn_dist"], {"input": xb[:n].numpy()})
            dec = arch.decide(p, knn_dist=k_, feats=f_)
            same = sum(same_answer(d, r) for d, r in zip(dec, rdec))
            print(f"  onnxruntime (CPU) vs repo EnhancedRes: max |prob diff| {np.abs(p - rp).max():.2e}, "
                  f"|feats| {np.abs(f_ - rf).max():.2e}, |knn| {np.abs(k_ - rk).max():.2e}; same class "
                  f"{(p.argmax(1) == rp.argmax(1)).mean() * 100:.0f}%; same architecture-B answer {same}/{len(dec)}")
        except Exception as e:  # CPU onnxruntime may lack some FP16 kernels; the GPU side is checked by verify.py
            print(f"  valid ONNX; onnxruntime CPU could not run it ({type(e).__name__}: {str(e)[:120]})")
    print("\ndone:", ONNX_DIR)


if __name__ == "__main__":
    main()
