"""Step 1: export the enhanced_res backbone (with 8-view TTA inside the graph) to ONNX, FP32 and FP16.

  python export_onnx.py

Writes onnx/enhanced_res_fp32.onnx and onnx/enhanced_res_fp16.onnx:
  input  "input"  float32 [N,3,224,224]   ImageNet-normalised crops (N = 1..any)
  output "feats"  float32 [N,768]         TTA-averaged layer3+layer4 features (the MLP head's input)
  output "logits" float32 [N,6]           backbone fc + TTA ("r2 fc + TTA", no head)
Then checks both against PyTorch with onnxruntime (CPU) on 16 grouped-test crops.
"""
import numpy as np
import torch
from PIL import Image

from erc import ONNX_DIR, TTABackbone, er, load_backbone_cpu, normalize


def export(model, path, opset=18):
    x = torch.randn(2, 3, 224, 224)
    kw = dict(input_names=["input"], output_names=["feats", "logits"], opset_version=opset)
    try:  # dynamo exporter first
        torch.onnx.export(model, (x,), path, dynamo=True, dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1)}},
                          external_data=False, **kw)
        return "dynamo"
    except Exception as e:
        print(f"  dynamo exporter failed ({type(e).__name__}: {str(e)[:150]}), using the TorchScript exporter")
        torch.onnx.export(model, (x,), path, dynamo=False, do_constant_folding=True,
                          dynamic_axes={"input": {0: "batch"}, "feats": {0: "batch"}, "logits": {0: "batch"}}, **kw)
        return "torchscript"


def main():
    import onnx
    import onnxruntime as ort

    ONNX_DIR.mkdir(exist_ok=True)
    crops = er.load_split()
    test = crops[crops.split == "test"].reset_index(drop=True).iloc[:: max(1, 4255 // 16)].head(16)
    x = np.stack([np.asarray(Image.open(p).convert("RGB")).transpose(2, 0, 1) for p in test.crop_path])
    xb = normalize(torch.from_numpy(x)).contiguous()

    for prec in ["fp32", "fp16"]:
        backbone, classes = load_backbone_cpu()
        ref_net = TTABackbone(load_backbone_cpu()[0])                        # FP32 reference on CPU
        with torch.no_grad():
            rf, rl = ref_net(xb)
        net = TTABackbone(backbone, half=prec == "fp16")
        path = ONNX_DIR / f"enhanced_res_{prec}.onnx"
        print(f"exporting {path.name} ...", flush=True)
        how = export(net, str(path))
        onnx.checker.check_model(str(path))
        print(f"  exported with the {how} exporter, {path.stat().st_size / 1e6:.1f} MB")
        try:
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            for n in (1, 16):  # batch size really is dynamic
                f, l = sess.run(["feats", "logits"], {"input": xb[:n].numpy()})
            print(f"  onnxruntime vs PyTorch FP32 on 16 crops: max |feat diff| {np.abs(f - rf.numpy()).max():.2e}, "
                  f"max |logit diff| {np.abs(l - rl.numpy()).max():.2e}, "
                  f"same class (fc) {(l.argmax(1) == rl.numpy().argmax(1)).mean() * 100:.1f}%")
        except Exception as e:  # CPU onnxruntime may lack FP16 kernels; TensorRT/GPU side checks it in benchmark.py
            print(f"  valid ONNX; onnxruntime CPU could not run it ({type(e).__name__}: {str(e)[:120]}). "
                  "It is checked on the GPU by benchmark.py.")
    print("\ndone:", ONNX_DIR)


if __name__ == "__main__":
    main()
