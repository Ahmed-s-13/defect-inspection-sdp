"""Step 3: enhanced_res on PyTorch vs TensorRT, same GPU, same 4,255 grouped-test crops.

  python benchmark.py                          PyTorch FP32/FP16 + TensorRT FP32/FP16
  python benchmark.py --backends torch_fp32 trt_fp32
  python benchmark.py --iters 1000             more timing runs (steadier numbers)

Every backend runs the WHOLE enhanced_res (backbone + 8-view TTA + 5 MLPs + 5-NN, erc.FullModel): PyTorch on CUDA,
or the TensorRT engine built from it. FP16 = backbone in half precision, head and kNN in FP32. One crop = 8 views.

Per backend: accuracy, macro-F1 (95% interval resampling whole source groups), balanced accuracy, per-class recall,
% same prediction as PyTorch FP32, exact McNemar vs PyTorch FP32, latency batch 1 (model; end-to-end from RAM),
throughput (4 crops = 32 views per call), whole-test-set time, model size. Saved to results/results_enhanced_res_full_*.
(The earlier results_enhanced_res_* files are from the backbone-only engines with the head on PyTorch.)
"""
import argparse
import json
import math
import re
import time

import numpy as np
import pandas as pd
import torch

from erc import RESULTS, FullModel, er, normalize, sync, timed
from trt_runner import TRTModel, engine_path

EXPECTED = {"accuracy": 0.8045, "macro_f1": 0.7376, "balanced_accuracy": 0.7367}   # evaluate.py / ablation.csv
TOL = 0.003
NAMES = {"torch_fp32": "PyTorch FP32", "torch_fp16": "PyTorch FP16", "trt_fp32": "TensorRT FP32", "trt_fp16": "TensorRT FP16"}


def confusion(y, p, k):
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    return cm


def macro_f1_cm(cm):
    tp = np.diag(cm).astype(float)
    den = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0.0)))


def group_ci(y, p, groups, k, n=2000, seed=0):
    """95% interval of macro-F1, resampling whole source groups (crops of one scene move together)."""
    ug, gi = np.unique(groups, return_inverse=True)
    cm_g = np.zeros((len(ug), k, k), dtype=np.int64)
    np.add.at(cm_g, (gi, y, p), 1)
    rng = np.random.default_rng(seed)
    v = [macro_f1_cm(cm_g[rng.integers(0, len(ug), len(ug))].sum(0)) for _ in range(n)]
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def mcnemar_exact(ok_a, ok_b):
    b, c = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return b, c, min(1.0, 2 * tail)


def metrics(y, p, k):
    cm = confusion(y, p, k)
    rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
    return dict(accuracy=float((y == p).mean()), macro_f1=macro_f1_cm(cm), balanced_accuracy=float(rec.mean()),
                recall=[float(r) for r in rec])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backends", nargs="+", default=list(NAMES), choices=list(NAMES))
    ap.add_argument("--batch", type=int, default=4, help="crops per call (x8 views); TensorRT engines are built for <= 4")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--no-save", action="store_true", help="print only, don't write results/ (quick checks)")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("No NVIDIA GPU visible to PyTorch.")
    import tensorrt as trt
    run_start = time.perf_counter()
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    torch.backends.cudnn.benchmark = True
    print(f"device: cuda ({gpu}), torch {torch.__version__}, CUDA {torch.version.cuda}, TensorRT {trt.__version__}")

    # ---- data: grouped test crops
    crops = er.load_split()
    test = crops[crops.split == "test"].reset_index(drop=True)
    classes = torch.load(er.EWORK / er.HEAD_FILE, map_location="cpu")["classes"]
    k = len(classes)
    y = test.cls.map({c: i for i, c in enumerate(classes)}).values
    groups = test.src_group.values
    from PIL import Image
    t0 = time.time()
    x_all = np.empty((len(test), 3, 224, 224), dtype=np.uint8)
    for i, pth in enumerate(test.crop_path):
        x_all[i] = np.asarray(Image.open(pth).convert("RGB")).transpose(2, 0, 1)
    x_cpu = torch.from_numpy(x_all).pin_memory()
    print(f"loaded {len(test)} grouped-test crops ({test.src_group.nunique()} groups) in {time.time() - t0:.0f}s")

    stream = torch.cuda.Stream()
    rows, preds, oks = [], {}, {}
    with torch.cuda.stream(stream), torch.inference_mode():
        for b in a.backends:
            prec = b.split("_")[1]
            print(f"\n== {NAMES[b]} ==", flush=True)
            t_backend = time.perf_counter()
            if b.startswith("torch"):
                net = FullModel(half=prec == "fp16").to(device).eval()
                run = net
                size_mb = sum(t.numel() * t.element_size() for t in [*net.parameters(), *net.buffers()]) / 1e6
            else:
                ep = engine_path(prec)
                if not ep.exists():
                    raise SystemExit(f"{ep} missing: run `python trt_runner.py` first")
                run = TRTModel(ep, stream)
                size_mb = ep.stat().st_size / 1e6
                if run.max_batch < a.batch:
                    raise SystemExit(f"engine max batch {run.max_batch} < --batch {a.batch}")

            def classify(xn):
                return run(xn).argmax(1)

            # accuracy + whole-test-set time (everything included: copy, normalise, the whole model)
            sync(device)
            t0 = time.perf_counter()
            P = []
            for i in range(0, len(y), a.batch):
                xn = normalize(x_cpu[i:i + a.batch].to(device, non_blocking=True)).contiguous()
                P.append(classify(xn).cpu())
            sync(device)
            total = time.perf_counter() - t0
            pred = torch.cat(P).numpy()
            preds[b], oks[b] = pred, pred == y
            r = dict(backend=NAMES[b], **metrics(y, pred, k), test_set_seconds=total,
                     test_set_crops_per_s=len(y) / total, model_size_mb=size_mb)
            r["macro_f1_ci"] = group_ci(y, pred, groups, k)

            # latency, batch 1 (one crop = 8 views): the whole model / end-to-end from RAM
            x1 = normalize(x_cpu[:1].to(device)).contiguous()
            lat = timed(lambda: run(x1), device, a.warmup, a.iters)
            r.update(latency_mean_ms=lat.mean(), latency_p50_ms=float(np.median(lat)), latency_p95_ms=float(np.percentile(lat, 95)))
            kk = iter(range(10 ** 9))

            def e2e():
                j = next(kk) % len(y)
                return int(classify(normalize(x_cpu[j:j + 1].to(device, non_blocking=True)).contiguous())[0].item())
            e = timed(e2e, device, a.warmup, a.iters)
            r.update(e2e_mean_ms=e.mean(), e2e_p95_ms=float(np.percentile(e, 95)))
            xb = normalize(x_cpu[:a.batch].to(device)).contiguous()
            tb = timed(lambda: classify(xb), device, max(5, a.warmup // 5), max(20, a.iters // 5))
            r["throughput_crops_per_s"] = a.batch / (tb.mean() / 1000)
            r["total_run_seconds"] = time.perf_counter() - t_backend
            rows.append(r)
            lo, hi = r["macro_f1_ci"]
            print(f"  acc {r['accuracy']:.4f}  macro-F1 {r['macro_f1']:.4f} ({lo:.3f}-{hi:.3f})  bal {r['balanced_accuracy']:.4f}"
                  f"  | b1 model {r['latency_mean_ms']:.2f} ms, end-to-end {r['e2e_mean_ms']:.2f} ms"
                  f" | {r['throughput_crops_per_s']:.0f} crops/s | test set {total:.1f}s", flush=True)
            del run
            torch.cuda.empty_cache()

    # ---- compare with PyTorch FP32 (reference), sanity check against the repo's evaluate.py numbers
    ref = "torch_fp32" if "torch_fp32" in preds else a.backends[0]
    base = next(r for r in rows if r["backend"] == NAMES[ref])
    for b, r in zip(a.backends, rows):
        r["same_as_ref"] = float((preds[b] == preds[ref]).mean())
        r["mcnemar_b_c_p"] = mcnemar_exact(oks[ref], oks[b])
        r["speedup_vs_ref"] = base["latency_mean_ms"] / r["latency_mean_ms"]
    gate = ""
    if ref == "torch_fp32":
        okk = all(abs(base[m] - EXPECTED[m]) <= TOL for m in EXPECTED)
        gate = (f"PyTorch FP32 vs the repo's evaluate.py (0.8045 / 0.7376 / 0.7367): "
                f"{'MATCH' if okk else 'DIFFERENT'} ({base['accuracy']:.4f} / {base['macro_f1']:.4f} / {base['balanced_accuracy']:.4f}).")

    hdr = ["backend", "accuracy", "macro-F1 (95% group CI)", "balanced acc", "same class as FP32",
           "McNemar vs FP32 (b, c, p)", "latency b1 (ms)", "p95", "end-to-end b1 (ms)",
           "throughput b4 (crops/s)", "whole test set (s)", "speed-up", "size (MB)"]
    md = [f"# enhanced_res (whole model): PyTorch vs TensorRT on {gpu}\n",
          f"{len(y)} grouped D1-test crops ({test.src_group.nunique()} source groups), same trained model. "
          f"torch {torch.__version__}, CUDA {torch.version.cuda}, TensorRT {trt.__version__}. "
          f"Every row runs the whole model (backbone + 8-view TTA + MLP heads + 5-NN); the TensorRT rows need no PyTorch "
          f"for the model. Latency = mean of {a.iters} runs after {a.warmup} warm-up runs.\n",
          "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        bb, cc, pp = r["mcnemar_b_c_p"]
        lo, hi = r["macro_f1_ci"]
        md.append("| " + " | ".join([
            r["backend"], f"{r['accuracy']:.4f}", f"{r['macro_f1']:.4f} ({lo:.3f}-{hi:.3f})", f"{r['balanced_accuracy']:.4f}",
            f"{r['same_as_ref'] * 100:.2f}%", f"{bb}, {cc}, p={pp:.3f}",
            f"{r['latency_mean_ms']:.2f}", f"{r['latency_p95_ms']:.2f}", f"{r['e2e_mean_ms']:.2f}",
            f"{r['throughput_crops_per_s']:.1f}", f"{r['test_set_seconds']:.1f}", f"{r['speedup_vs_ref']:.2f}x",
            f"{r['model_size_mb']:.1f}"]) + " |")
    md.append("\nPer-class recall: " + "; ".join(
        f"{r['backend']}: " + ", ".join(f"{c} {v:.3f}" for c, v in zip(classes, r["recall"])) for r in rows))
    if gate:
        md.append("\n" + gate)
    md.append(f"\nWhole benchmark run: {time.perf_counter() - run_start:.0f} s. One run, one seed; RTX 2050 is not the Jetson Orin NX "
              "(relative speed-ups carry over, absolute ms do not).")
    print("\n" + "\n".join(md))
    if a.no_save:
        return
    RESULTS.mkdir(exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", gpu).strip("_")
    (RESULTS / f"results_enhanced_res_full_{tag}.md").write_text("\n".join(md), encoding="utf-8")
    (RESULTS / f"results_enhanced_res_full_{tag}.json").write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
    print(f"\nsaved {RESULTS / f'results_enhanced_res_full_{tag}.md'}")


if __name__ == "__main__":
    main()
