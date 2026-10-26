"""Step 1: export the WHOLE enhanced_res (backbone + 8-view TTA + MLP heads + 5-NN) to ONNX, FP32 and FP16.

  python export_onnx.py

Writes onnx/enhanced_res_full_fp32.onnx and onnx/enhanced_res_full_fp16.onnx:
  input  "input"  float32 [N,3,224,224]   ImageNet-normalised 224 px crops (N = 1..any)
  output "probs"  float32 [N,6]           class probabilities, same as the repo's EnhancedRes(x)
FP16 = the backbone in half precision; head, kNN bank and outputs stay float32 (see erc.FullModel).
Then checks both against the repo's PyTorch EnhancedRes with onnxruntime (CPU) on 16 grouped-test crops.
"""
import numpy as np
import torch
from PIL import Image

from erc import ONNX_DIR, FullModel, er, normalize


def export(model, path, opset=18):
    x = torch.randn(2, 3, 224, 224)
    kw = dict(input_names=["input"], output_names=["probs"], opset_version=opset)
    try:  # dynamo exporter first
        torch.onnx.export(model, (x,), path, dynamo=True, dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1)}},
                          external_data=False, **kw)
        return "dynamo"
    except Exception as e:
        print(f"  dynamo exporter failed ({type(e).__name__}: {str(e)[:150]}), using the TorchScript exporter")
        torch.onnx.export(model, (x,), path, dynamo=False, do_constant_folding=True,
                          dynamic_axes={"input": {0: "batch"}, "probs": {0: "batch"}}, **kw)
        return "torchscript"


def main():
    import onnx
    import onnxruntime as ort

    ONNX_DIR.mkdir(exist_ok=True)
    import json
    (ONNX_DIR / "classes.json").write_text(json.dumps(FullModel().classes))   # class order of "probs", for predict_trt.py
    crops = er.load_split()
    test = crops[crops.split == "test"].reset_index(drop=True).iloc[:: 4255 // 16].head(16)
    x = np.stack([np.asarray(Image.open(p).convert("RGB")).transpose(2, 0, 1) for p in test.crop_path])
    xb = normalize(torch.from_numpy(x)).contiguous()
    ref = er.EnhancedRes(device=torch.device("cpu"))(xb).numpy()          # the repo's own model = the reference

    for prec in ["fp32", "fp16"]:
        net = FullModel(half=prec == "fp16")
        with torch.no_grad():
            mine = net(xb).numpy()
        if prec == "fp32":
            print(f"FullModel (PyTorch) vs repo EnhancedRes on 16 crops: max |prob diff| {np.abs(mine - ref).max():.2e}, "
                  f"same class {(mine.argmax(1) == ref.argmax(1)).mean() * 100:.0f}%")
        path = ONNX_DIR / f"enhanced_res_full_{prec}.onnx"
        print(f"exporting {path.name} ...", flush=True)
        how = export(net, str(path))
        onnx.checker.check_model(str(path))
        print(f"  exported with the {how} exporter, {path.stat().st_size / 1e6:.1f} MB")
        try:
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            for n in (1, 16):  # the batch size really is dynamic
                p, = sess.run(["probs"], {"input": xb[:n].numpy()})
            print(f"  onnxruntime vs repo EnhancedRes on 16 crops: max |prob diff| {np.abs(p - ref).max():.2e}, "
                  f"same class {(p.argmax(1) == ref.argmax(1)).mean() * 100:.0f}%")
        except Exception as e:  # CPU onnxruntime may lack some FP16 kernels; the GPU side is checked by predict_trt.py
            print(f"  valid ONNX; onnxruntime CPU could not run it ({type(e).__name__}: {str(e)[:120]})")
    print("\ndone:", ONNX_DIR)


if __name__ == "__main__":
    main()
