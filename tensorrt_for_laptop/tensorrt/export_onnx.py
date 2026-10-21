"""Step 1: export the trained ResNet-18 to ONNX and pack everything the NVIDIA laptop needs into bundle/.

Run where the repo and its data are (the Mac): reads 1_main_model_resnet18 without changing it.

  python export_onnx.py                          default repo path: ../../Project Code/1_main_model_resnet18
  python export_onnx.py --main /path/to/defect-inspection-sdp/1_main_model_resnet18

bundle/ (copy this whole folder to the laptop, ~200 MB):
  resnet18_fp32.onnx   input "input" float32 [N,3,224,224] -> "logits" [N,6], "embedding" [N,512], N = 1..any
  resnet18_fp16.onnx   same inputs/outputs (float32), weights and maths in FP16 (for the FP16 TensorRT engine;
                       TensorRT 11 removed the --fp16 / BuilderFlag.FP16 switch, precision now comes from the ONNX)
  model_best.pt        the PyTorch weights (for the PyTorch side of the benchmark)
  mlp_head.pt          the MLP head (optional, if it exists)
  test/*.jpg + labels.csv   the 4,293 D1-test crops (never used for training or for any choice)

Then checks both ONNX files against PyTorch with onnxruntime (CPU) on a batch of test crops.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import BUNDLE, HERE, ResNetWithEmbedding, build_resnet, load_images, normalize


def export(model, path, opset):
    x = torch.randn(2, 3, 224, 224)
    kw = dict(input_names=["input"], output_names=["logits", "embedding"], opset_version=opset)
    try:  # torch.onnx's default (dynamo) exporter; needs the onnxscript package
        torch.onnx.export(model, (x,), path, dynamo=True, dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1)}},
                          external_data=False, **kw)
    except Exception as e:
        print(f"  dynamo exporter failed ({type(e).__name__}: {str(e)[:120]}), using the TorchScript exporter")
        torch.onnx.export(model, (x,), path, dynamo=False, do_constant_folding=True,
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}, "embedding": {0: "batch"}}, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default=str(HERE.parent.parent / "Project Code" / "1_main_model_resnet18"),
                    help="the 1_main_model_resnet18 folder (with data/work/)")
    ap.add_argument("--out", default=str(BUNDLE))
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--no-images", action="store_true", help="skip copying the test crops (if bundle/test exists)")
    a = ap.parse_args()
    main_dir, out = Path(a.main), Path(a.out)
    work = main_dir / "data" / "work"
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(work / "model_best.pt", map_location="cpu")
    classes = ck["classes"]
    shutil.copy(work / "model_best.pt", out / "model_best.pt")
    if (work / "mlp_head.pt").exists():
        shutil.copy(work / "mlp_head.pt", out / "mlp_head.pt")
    else:
        print(f"note: {work / 'mlp_head.pt'} not found (run train_mlp.py); the benchmark will skip the MLP head")

    if not a.no_images:
        split = pd.read_csv(work / "d1_split.csv")
        test = split[split.split == "test"].reset_index(drop=True)
        (out / "test").mkdir(exist_ok=True)
        files = []
        for i, r in test.iterrows():
            f = f"{i:05d}_{r.cls}.jpg"
            shutil.copy(main_dir / r.crop_path, out / "test" / f)
            files.append(f)
        pd.DataFrame({"file": files, "cls": test.cls, "source_crop": test.crop_path}).to_csv(out / "test" / "labels.csv", index=False)
        print(f"copied {len(test)} D1-test crops to {out / 'test'}")

    for prec in ["fp32", "fp16"]:
        model = ResNetWithEmbedding(build_resnet(ck["state"], len(classes)), half=prec == "fp16")
        path = out / f"resnet18_{prec}.onnx"
        print(f"exporting {path.name} ...")
        export(model, str(path), a.opset)

    # check against PyTorch (CPU, FP32) with onnxruntime
    import onnx
    import onnxruntime as ort
    x, y, cls = load_images(out / "test", out / "test" / "labels.csv")
    assert cls == classes
    xb = normalize(torch.from_numpy(x[:64])).numpy()
    with torch.no_grad():
        ref_logits, ref_emb = (t.numpy() for t in ResNetWithEmbedding(build_resnet(ck["state"], len(classes)))(torch.from_numpy(xb)))
    for prec in ["fp32", "fp16"]:
        path = out / f"resnet18_{prec}.onnx"
        onnx.checker.check_model(str(path))
        try:
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            for n in [1, 64]:  # the batch size really is dynamic
                lo, em = sess.run(["logits", "embedding"], {"input": xb[:n]})
            agree = (lo.argmax(1) == ref_logits.argmax(1)).mean()
            print(f"  {path.name}: {path.stat().st_size / 1e6:.1f} MB, onnxruntime vs PyTorch on 64 crops: "
                  f"max |logit diff| {np.abs(lo - ref_logits).max():.2e}, max |emb diff| {np.abs(em - ref_emb).max():.2e}, "
                  f"same class {agree * 100:.1f}%")
        except Exception as e:  # some onnxruntime CPU builds lack FP16 kernels; the GPU / TensorRT side has them
            print(f"  {path.name}: valid ONNX ({path.stat().st_size / 1e6:.1f} MB); onnxruntime CPU could not run it "
                  f"({type(e).__name__}: {str(e)[:100]}). It is checked on the GPU by benchmark.py.")
    print(f"\nbundle ready: {out}  ({sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) / 1e6:.0f} MB). "
          "Copy it with this folder to the NVIDIA laptop.")


if __name__ == "__main__":
    main()
