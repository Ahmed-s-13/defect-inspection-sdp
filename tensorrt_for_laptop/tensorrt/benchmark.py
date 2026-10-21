"""Step 3: the same trained ResNet-18 on PyTorch vs TensorRT, on the same GPU and the same 4,293 D1-test crops.

  python benchmark.py                          all four: PyTorch FP32, PyTorch FP16, TensorRT FP32, TensorRT FP16
  python benchmark.py --backends torch_fp32 trt_fp32 trt_fp16
  python benchmark.py --iters 1000             more timing runs (steadier numbers)

For every backend it reports:
  accuracy / macro-F1      the ResNet's own fc layer, on all 4,293 test crops
  + MLP head               accuracy when its 512-d embeddings go through the MLP head (the model predict.py uses)
  same as PyTorch FP32     % of crops where the backend predicts the same class as PyTorch FP32
  latency (batch 1)        one crop already on the GPU -> logits: mean, median (p50) and p95, in ms
  end-to-end (batch 1)     one crop from RAM: copy to GPU, normalise, model, result back to the CPU: mean and p95 ms
  throughput (batch 32)    crops per second, model only
  whole test set           wall time to classify all 4,293 crops in batches of 32 (copy + normalise + model + result)
  speed-up                 batch-1 latency of PyTorch FP32 / this backend

Results are printed and saved to results/results_<gpu>.md and .json. TensorRT engines are built first if missing
(trt_engine.py). Runs on the NVIDIA GPU; on a Mac it runs the PyTorch rows on the Apple GPU (no TensorRT there).
"""
import argparse
import json
import platform
import re
import time
from pathlib import Path

import numpy as np
import torch

from common import BUNDLE, HERE, MLPClassifier, ResNetWithEmbedding, build_resnet, load_images, normalize, pick_device, sync, timed

EXPECTED = {"accuracy": 0.8796, "macro_f1": 0.8125, "mlp_accuracy": 0.9010}  # from 1_main_model_resnet18/results


def f1_macro(y, p, k):
    f = []
    for c in range(k):
        tp, fp, fn = np.sum((p == c) & (y == c)), np.sum((p == c) & (y != c)), np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=str(BUNDLE))
    ap.add_argument("--backends", nargs="+", default=["torch_fp32", "torch_fp16", "trt_fp32", "trt_fp16"],
                    choices=["torch_fp32", "torch_fp16", "trt_fp32", "trt_fp16"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--device", default="auto", help="auto = CUDA, else Apple MPS, else CPU")
    a = ap.parse_args()
    run_start = time.perf_counter()  # whole benchmark, start to finish
    bundle = Path(a.bundle)
    device = pick_device(a.device)
    if device.type != "cuda":
        dropped = [b for b in a.backends if b.startswith("trt")]
        a.backends = [b for b in a.backends if not b.startswith("trt")]
        if dropped:
            print(f"WARNING: no NVIDIA GPU visible (device {device}); skipping {dropped}. See README step 1-3.")
        if not a.backends:
            raise SystemExit("nothing left to run")
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else (
        "Apple GPU (MPS)" if device.type == "mps" else platform.processor() or "CPU")
    torch.backends.cudnn.benchmark = True
    print(f"device: {device} ({gpu}), torch {torch.__version__}"
          + (f", CUDA {torch.version.cuda}" if device.type == "cuda" else ""))

    t0 = time.time()
    x_all, y, classes = load_images(bundle / "test", bundle / "test" / "labels.csv")
    print(f"loaded {len(y)} D1-test crops in {time.time() - t0:.0f}s ({x_all.nbytes / 1e6:.0f} MB in RAM)")
    x_cpu = torch.from_numpy(x_all)
    if device.type == "cuda":
        x_cpu = x_cpu.pin_memory()  # faster copies to the GPU, like a real camera pipeline would use
    ck = torch.load(bundle / "model_best.pt", map_location="cpu")
    assert ck["classes"] == classes
    head = MLPClassifier(bundle / "mlp_head.pt", device) if (bundle / "mlp_head.pt").exists() else None

    stream = torch.cuda.Stream() if device.type == "cuda" else None
    ctx = torch.cuda.stream(stream) if stream is not None else torch.inference_mode()
    rows, ref_pred = [], None
    with ctx, torch.inference_mode():
        for b in a.backends:
            name = {"torch_fp32": "PyTorch FP32", "torch_fp16": "PyTorch FP16",
                    "trt_fp32": "TensorRT FP32", "trt_fp16": "TensorRT FP16"}[b]
            print(f"\n== {name} ==", flush=True)
            backend_start = time.perf_counter()  # this backend: load/build + test set + all timing runs
            prec = b.split("_")[1]
            if b.startswith("torch"):
                model = ResNetWithEmbedding(build_resnet(ck["state"], len(classes)), half=prec == "fp16").to(device).eval()
            else:
                from trt_engine import TRTResNet, build, engine_path
                ep = engine_path(bundle, prec)
                if not ep.exists():
                    print(f"  building the {prec.upper()} engine first (once) ...", flush=True)
                    build(bundle / f"resnet18_{prec}.onnx", ep, a.batch)
                model = TRTResNet(ep, stream)
                if model.max_batch < a.batch:
                    raise SystemExit(f"engine max batch {model.max_batch} < --batch {a.batch}; rebuild with --max-batch")


           # model size: PyTorch = weights + buffers in memory, TensorRT = the engine file the GPU runs
            if b.startswith("torch"):
                size_mb = sum(t.numel() * t.element_size() for t in [*model.parameters(), *model.buffers()]) / 1e6
            else:
                size_mb = ep.stat().st_size / 1e6
 

            # accuracy + whole-test-set time (batches of 32, everything included)
            sync(device)
            t0 = time.perf_counter()
            P, E = [], []
            for i in range(0, len(y), a.batch):
                xb = normalize(x_cpu[i:i + a.batch].to(device, non_blocking=True)).contiguous()
                lo, em = model(xb)
                P.append(lo.argmax(1).cpu())
                E.append(em.clone())
            sync(device)
            total = time.perf_counter() - t0
            pred = torch.cat(P).numpy()
            emb = torch.cat(E)
            r = dict(backend=name, accuracy=float((pred == y).mean()), macro_f1=f1_macro(y, pred, len(classes)),
                     test_set_seconds=total, test_set_crops_per_s=len(y) / total, model_size_mb=size_mb)
            if head is not None:
                hp = torch.cat([head(emb[i:i + 1024]).argmax(1) for i in range(0, len(emb), 1024)]).cpu().numpy()
                r["mlp_accuracy"] = float((hp == y).mean())
                r["mlp_macro_f1"] = f1_macro(y, hp, len(classes))
            if ref_pred is None and b == "torch_fp32":
                ref_pred = pred
            r["same_as_torch_fp32"] = float((pred == ref_pred).mean()) if ref_pred is not None else None
            del E, emb

            # latency, batch 1, model only (input already on the GPU)
            x1 = normalize(x_cpu[:1].to(device)).contiguous()
            lat = timed(lambda: model(x1), device, a.warmup, a.iters)
            r.update(latency_mean_ms=lat.mean(), latency_p50_ms=np.median(lat), latency_p95_ms=np.percentile(lat, 95))

            # end-to-end, batch 1: crop in RAM -> GPU -> normalise -> model -> class on the CPU
            k = iter(range(10 ** 9))
            def e2e():
                j = next(k) % len(y)
                lo, _ = model(normalize(x_cpu[j:j + 1].to(device, non_blocking=True)).contiguous())
                return int(lo.argmax(1).item())
            e = timed(e2e, device, a.warmup, a.iters)
            r.update(e2e_mean_ms=e.mean(), e2e_p95_ms=np.percentile(e, 95))

            # throughput, batch 32, model only
            xb = normalize(x_cpu[:a.batch].to(device)).contiguous()
            tb = timed(lambda: model(xb), device, max(5, a.warmup // 5), max(20, a.iters // 5))
            r["throughput_crops_per_s"] = a.batch / (tb.mean() / 1000)
            r["total_run_seconds"] = time.perf_counter() - backend_start
            rows.append(r)
            print(f"  accuracy {r['accuracy']:.4f}  macro-F1 {r['macro_f1']:.4f}"
                  + (f"  | + MLP head {r['mlp_accuracy']:.4f}" if head is not None else "")
                  + f"  | latency {r['latency_mean_ms']:.2f} ms (p95 {r['latency_p95_ms']:.2f})"
                  + f"  | end-to-end {r['e2e_mean_ms']:.2f} ms  | {r['throughput_crops_per_s']:.0f} crops/s"
                  + f"  | test set {total:.2f}s", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ---- report
    base = next((r for r in rows if r["backend"] == "PyTorch FP32"), rows[0])
    for r in rows:
        r["speedup_vs_torch_fp32"] = base["latency_mean_ms"] / r["latency_mean_ms"]
        r["compression_vs_torch_fp32"] = base["model_size_mb"] / r["model_size_mb"]
    has_mlp = head is not None
    hdr = ["backend", "accuracy", "macro-F1"] + (["+ MLP head acc"] if has_mlp else []) + [
        "same class as PyTorch FP32", "latency b1 mean (ms)", "p50", "p95", "end-to-end b1 (ms)",
        "throughput b32 (crops/s)", "whole test set (s)", "speed-up (latency)", "model size (MB)", "compressed by",
        "total run time (s)"]
    md = [f"# ResNet-18: PyTorch vs TensorRT on {gpu}\n",
          f"{len(y)} D1-test crops, same trained model (model_best.pt). torch {torch.__version__}"
          + (f", CUDA {torch.version.cuda}" if device.type == "cuda" else "")
          + (f", TensorRT {__import__('tensorrt').__version__}" if any(b.startswith('trt') for b in a.backends) else "")
          + f". Latency = mean of {a.iters} runs after {a.warmup} warm-up runs.\n",
          "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        cells = [r["backend"], f"{r['accuracy']:.4f}", f"{r['macro_f1']:.4f}"]
        cells += [f"{r['mlp_accuracy']:.4f}"] if has_mlp else []
        cells += [f"{r['same_as_torch_fp32'] * 100:.2f}%" if r["same_as_torch_fp32"] is not None else "",
                  f"{r['latency_mean_ms']:.2f}", f"{r['latency_p50_ms']:.2f}", f"{r['latency_p95_ms']:.2f}",
                  f"{r['e2e_mean_ms']:.2f}", f"{r['throughput_crops_per_s']:.0f}", f"{r['test_set_seconds']:.2f}",
                  f"{r['speedup_vs_torch_fp32']:.2f}x", f"{r['model_size_mb']:.1f}",
                  f"{r['compression_vs_torch_fp32']:.2f}x ({(1 - 1 / r['compression_vs_torch_fp32']) * 100:.0f}% smaller)",
                  f"{r['total_run_seconds']:.1f}"]

        md.append("| " + " | ".join(cells) + " |")
    md.append(f"\nWhole benchmark run (loading the crops + every backend): {time.perf_counter() - run_start:.1f} s.")
    md.append(f"\nExpected accuracy (Mac, PyTorch FP32): fc {EXPECTED['accuracy']:.4f}, macro-F1 {EXPECTED['macro_f1']:.4f}, "
              f"+ MLP head {EXPECTED['mlp_accuracy']:.4f}.")
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", gpu).strip("_")
    (out / f"results_{tag}.md").write_text("\n".join(md))
    (out / f"results_{tag}.json").write_text(json.dumps(rows, indent=2, default=float))
    print("\n" + "\n".join(md))
    print(f"\nsaved {out / f'results_{tag}.md'}")


if __name__ == "__main__":
    main()
