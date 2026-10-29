"""Step 3: enhanced_res gc7 on PyTorch vs onnxruntime (CPU) vs TensorRT, same machine, same 4,868 test crops, with
architecture B on top of every one.

  python benchmark.py                          PyTorch FP32/FP16 (GPU), onnxruntime FP32 (CPU), TensorRT FP32/FP16 (GPU)
  python benchmark.py --backends torch_fp32 ort_cpu trt_fp16
  python benchmark.py --iters 1000             more timing runs (steadier numbers)

Every backend runs the WHOLE model (backbone + 8-view TTA + 5 MLPs + 5-NN) and returns probs, features and the kNN
distance; architecture B (enhanced_res/arch_a/runtime.py) then turns them into the visor's answer. Needs the repo's data
(main_model_data_work.zip + d1_gc7_no_defect_crops.zip), the gc7 weights, both ONNX files and both engines.

Per backend: accuracy / macro-F1 (95% interval resampling whole source groups) / balanced accuracy on d1_gc7_fix-test,
% same class and same architecture-B answer as PyTorch FP32, exact McNemar vs PyTorch FP32, latency batch 1 (model;
end-to-end from RAM including the decision), throughput (4 crops = 32 views per call), whole-test-set time, model size.
Then architecture B's numbers on the calibration test half (2,752 crops) against calibration_gc7.json's expected ones.
Saved to results/.
"""
import argparse
import json
import math
import os
import platform
import re
import time

import numpy as np
import pandas as pd
import torch

from erc import RESULTS, FullModel, er, normalize, sync, timed
from infer_common import load_arch
from trt_paths import MODEL, engine_path, onnx_path

# what the dashboard's Model page and the repo's evaluate.py report for gc7 on d1_gc7_fix-test (4,868 crops)
EXPECTED = {"accuracy": 0.8036, "macro_f1": 0.7514}
TOL = 0.003
NAMES = {"torch_fp32": "PyTorch FP32 (GPU)", "torch_fp16": "PyTorch FP16 (GPU)", "ort_cpu": "onnxruntime FP32 (CPU)",
         "trt_fp32": "TensorRT FP32 (GPU)", "trt_fp16": "TensorRT FP16 (GPU)"}
ARCH_KEYS = [("cp_coverage", "CP coverage"), ("cp_mean_set_size", "mean set size"), ("cp_single_class", "single class"),
             ("human_attention", "inspector needed"), ("single_class_wrong", "single wrong class"),
             ("silent_miss", "silent miss"), ("no_defect_false_alarm", "no-defect false alarm"),
             ("gates_false_reject", "re-aim")]
CHECKED = ("cp_coverage", "cp_family_coverage", "cp_mean_set_size", "cp_single_class", "human_attention",
           "single_class_wrong", "gates_false_reject_family")          # what the repo's evaluate.py checks


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


def answers(dec):
    return [(d.action, tuple(sorted(d.classes))) for d in dec]


def make_backend(b, stream):
    """-> (run, device, size_mb); run(x normalised float32 tensor on `device`) -> (probs, feats, knn) torch tensors."""
    if b.startswith("torch"):
        net = FullModel(MODEL, half=b.endswith("fp16")).cuda().eval()
        size = sum(t.numel() * t.element_size() for t in [*net.parameters(), *net.buffers()]) / 1e6
        return net, torch.device("cuda"), size
    if b == "ort_cpu":
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path("fp32")), providers=["CPUExecutionProvider"])

        def run(x):
            return tuple(torch.from_numpy(o) for o in sess.run(["probs", "feats", "knn_dist"], {"input": x.numpy()}))
        return run, torch.device("cpu"), onnx_path("fp32").stat().st_size / 1e6
    from trt_runner import TRTModel
    ep = engine_path(b.split("_")[1])
    if not ep.exists():
        raise SystemExit(f"{ep} missing: run `python trt_runner.py`")
    return TRTModel(ep, stream), torch.device("cuda"), ep.stat().st_size / 1e6


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
    gpu = torch.cuda.get_device_name(0)
    cpu = f"{platform.processor()} ({os.cpu_count()} threads)"
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {gpu}, torch {torch.__version__}, CUDA {torch.version.cuda}, TensorRT {trt.__version__}; CPU: {cpu}")

    # ---- data: the d1_gc7_fix test crops (defects + no-defect), and architecture B
    rt, arch = load_arch(MODEL)
    crops = er.load_split("d1_gc7_fix")
    test = crops[crops.split == "test"].reset_index(drop=True)
    classes, k = arch.classes, len(arch.classes)
    y = test.cls.map({c: i for i, c in enumerate(classes)}).values
    groups = test.src_group.values
    sp = pd.read_csv(er.HERE / "arch_a/cal_split.csv")                          # which crops are architecture B's test half
    sp["crop_path"] = [str(er.MAIN / p) for p in sp.crop_path]
    fam = sp.drop_duplicates("src_group").set_index("src_group").split
    half = test[["crop_path"]].merge(sp[["crop_path", "split"]], on="crop_path", how="left").split
    th = (half.fillna(pd.Series(test.src_group.map(fam).to_numpy())).to_numpy() == "test")
    exp = arch.cal["expected_test_half"]
    from PIL import Image
    t0 = time.time()
    x_all = np.empty((len(test), 3, 224, 224), dtype=np.uint8)
    for i, pth in enumerate(test.crop_path):
        x_all[i] = np.asarray(Image.open(pth).convert("RGB")).transpose(2, 0, 1)
    x_cpu = torch.from_numpy(x_all).pin_memory()
    print(f"loaded {len(test)} test crops ({len(set(groups))} groups, {int((y == arch.nd).sum())} no-defect; "
          f"{int(th.sum())} in architecture B's test half) in {time.time() - t0:.0f}s")

    stream = torch.cuda.Stream()
    res, store = [], {}
    with torch.cuda.stream(stream), torch.inference_mode():
        for b in a.backends:
            print(f"\n== {NAMES[b]} ==", flush=True)
            t_backend = time.perf_counter()
            run, dev, size_mb = make_backend(b, stream)
            on_gpu = dev.type == "cuda"
            iters = a.iters if on_gpu else min(a.iters, 100)                    # a CPU run is ~80 ms: 100 is plenty

            def fetch(sl):
                x = x_cpu[sl]
                return normalize(x.to(dev, non_blocking=True) if on_gpu else x).contiguous()

            # accuracy + whole-test-set time (everything included: copy, normalise, the whole model, outputs to the CPU)
            sync(dev)
            t0 = time.perf_counter()
            P, F, K = [], [], []
            for i in range(0, len(y), a.batch):
                p, f, kd = run(fetch(slice(i, i + a.batch)))
                P.append(p.cpu()), F.append(f.cpu()), K.append(kd.cpu())
            sync(dev)
            total = time.perf_counter() - t0
            probs, feats, knn = (torch.cat(t).numpy() for t in (P, F, K))
            dec = arch.decide(probs, knn_dist=knn, feats=feats)
            pred = probs.argmax(1)
            r = dict(backend=NAMES[b], **metrics(y, pred, k), test_set_seconds=total, test_set_crops_per_s=len(y) / total,
                     model_size_mb=size_mb)
            r["macro_f1_ci"] = group_ci(y, pred, groups, k)
            r["arch"] = rt.summarise([d for d, t in zip(dec, th) if t], arch, probs[th], y[th], groups[th])
            store[b] = dict(pred=pred, ans=answers(dec), probs=probs, ok=pred == y)

            # latency, batch 1 (one crop = 8 views): the model / end-to-end from RAM including the decision
            x1 = fetch(slice(0, 1))
            lat = timed(lambda: run(x1), dev, a.warmup, iters)
            r.update(latency_mean_ms=lat.mean(), latency_p50_ms=float(np.median(lat)), latency_p95_ms=float(np.percentile(lat, 95)))
            kk = iter(range(10 ** 9))

            def e2e():
                j = next(kk) % len(y)
                p, f, kd = run(fetch(slice(j, j + 1)))
                return arch.decide(p.cpu().numpy(), knn_dist=kd.cpu().numpy(), feats=f.cpu().numpy())[0].action
            e = timed(e2e, dev, a.warmup, iters)
            r.update(e2e_mean_ms=e.mean(), e2e_p50_ms=float(np.median(e)), e2e_p95_ms=float(np.percentile(e, 95)))
            xb = fetch(slice(0, a.batch))
            tb = timed(lambda: run(xb), dev, max(5, a.warmup // 5), max(20, iters // 5))
            r["throughput_crops_per_s"] = a.batch / (tb.mean() / 1000)
            r["total_run_seconds"] = time.perf_counter() - t_backend
            res.append(r)
            print(f"  acc {r['accuracy']:.4f}  macro-F1 {r['macro_f1']:.4f} ({r['macro_f1_ci'][0]:.3f}-{r['macro_f1_ci'][1]:.3f})"
                  f"  | b1 model {r['latency_p50_ms']:.2f} ms (median), end-to-end {r['e2e_p50_ms']:.2f} ms"
                  f" | {r['throughput_crops_per_s']:.0f} crops/s | test set {total:.1f}s", flush=True)
            del run
            torch.cuda.empty_cache()
            time.sleep(15)                                                      # let the GPU / CPU cool before the next one

    # ---- compare with PyTorch FP32 (reference); sanity check against the numbers the team publishes
    ref = "torch_fp32" if "torch_fp32" in store else a.backends[0]
    base = next(r for r in res if r["backend"] == NAMES[ref])
    for b, r in zip(a.backends, res):
        s = store[b]
        r["same_class"] = float((s["pred"] == store[ref]["pred"]).mean())
        r["same_answer"] = float(np.mean([x == y_ for x, y_ in zip(s["ans"], store[ref]["ans"])]))
        r["mcnemar_b_c_p"] = mcnemar_exact(store[ref]["ok"], s["ok"])
        r["speedup_vs_ref"] = base["latency_p50_ms"] / r["latency_p50_ms"]
        tol = lambda q: 0.005 * (3 if q == "cp_mean_set_size" else 1)           # noqa: E731  (evaluate.py's tolerances)
        r["arch_off"] = {q: round(r["arch"][q] - exp[q], 4) for q in CHECKED if q in exp and abs(r["arch"][q] - exp[q]) > tol(q)}
        r["arch_match"] = not r["arch_off"]
    gate = ""
    if ref == "torch_fp32":
        okk = all(abs(base[m] - EXPECTED[m]) <= TOL for m in EXPECTED)
        gate = (f"PyTorch FP32 vs the dashboard's Model page / repo evaluate.py ({EXPECTED['accuracy']} / {EXPECTED['macro_f1']}): "
                f"{'MATCH' if okk else 'DIFFERENT'} ({base['accuracy']:.4f} / {base['macro_f1']:.4f}).")

    hdr = ["backend", "accuracy", "macro-F1 (95% group CI)", "balanced acc", "same class as PyTorch FP32",
           "same arch-B answer", "McNemar vs FP32 (b, c, p)", "latency b1, median (ms)", "p95", "end-to-end b1 + decision, median (ms)",
           "throughput b4 (crops/s)", "whole test set (s)", "speed-up", "size (MB)"]
    md = [f"# enhanced_res gc7 + architecture B: PyTorch vs onnxruntime vs TensorRT\n",
          f"{len(y)} d1_gc7_fix test crops ({len(set(groups))} source groups, {int((y == arch.nd).sum())} no-defect), same trained "
          f"model. GPU {gpu}, CPU {cpu}; torch {torch.__version__}, CUDA {torch.version.cuda}, TensorRT {trt.__version__}. Every "
          f"row runs the whole model (backbone + 8-view TTA + MLP heads + 5-NN); the onnxruntime and TensorRT rows need no "
          f"PyTorch. Latency = median of {a.iters} runs after {a.warmup} warm-up runs (onnxruntime: 100), 15 s pause between "
          f"backends.\n",
          "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in res:
        bb, cc, pp = r["mcnemar_b_c_p"]
        lo, hi = r["macro_f1_ci"]
        md.append("| " + " | ".join([
            r["backend"], f"{r['accuracy']:.4f}", f"{r['macro_f1']:.4f} ({lo:.3f}-{hi:.3f})", f"{r['balanced_accuracy']:.4f}",
            f"{r['same_class'] * 100:.2f}%", f"{r['same_answer'] * 100:.2f}%", f"{bb}, {cc}, p={pp:.3f}",
            f"{r['latency_p50_ms']:.2f}", f"{r['latency_p95_ms']:.2f}", f"{r['e2e_p50_ms']:.2f}",
            f"{r['throughput_crops_per_s']:.1f}", f"{r['test_set_seconds']:.1f}", f"{r['speedup_vs_ref']:.2f}x",
            f"{r['model_size_mb']:.1f}"]) + " |")
    md += [f"\n## Architecture B on its test half ({int(th.sum())} crops), every runtime\n",
           "| | " + " | ".join(n for _, n in ARCH_KEYS) + " | matches calibration_gc7.json |",
           "|---|" + "---|" * (len(ARCH_KEYS) + 1),
           "| expected (calibration_gc7.json) | " + " | ".join(f"{exp[q] * 100:.1f}%" if q != "cp_mean_set_size" else f"{exp[q]:.2f}"
                                                              for q, _ in ARCH_KEYS) + " | |"]
    for r in res:
        md.append(f"| {r['backend']} | " + " | ".join(f"{r['arch'][q] * 100:.1f}%" if q != "cp_mean_set_size" else f"{r['arch'][q]:.2f}"
                                                      for q, _ in ARCH_KEYS)
                  + f" | {'yes' if r['arch_match'] else 'NO: ' + ', '.join(f'{q} {d:+.4f}' for q, d in r['arch_off'].items())} |")
    md.append("\nThe check uses the repo's evaluate.py tolerances (0.005; 0.015 for the set size) on seven metrics, among them the "
              "share of photo families stopped by the re-aim gate: that one averages over 87 families, so a single borderline crop "
              "in a small family moves it by up to ~0.006. Different runtimes round differently and flip a handful of crops "
              "that sit exactly on a threshold (see 'same arch-B answer').")
    md.append("\nPer-class recall (top-1): " + "; ".join(
        f"{r['backend']}: " + ", ".join(f"{c} {v:.3f}" for c, v in zip(classes, r["recall"])) for r in res[:1]))
    if gate:
        md.append("\n" + gate)
    md.append(f"\nWhole benchmark run: {time.perf_counter() - run_start:.0f} s. One run, one seed; the RTX 2050 is not the Jetson Orin "
              "NX and this laptop CPU is not a Raspberry Pi (relative speed-ups carry over, absolute ms do not).")
    print("\n" + "\n".join(md))
    if a.no_save:
        return
    RESULTS.mkdir(exist_ok=True)
    tag = re.sub(r"[^A-Za-z0-9]+", "_", gpu).strip("_")
    (RESULTS / f"results_enhanced_res_{MODEL}_{tag}.md").write_text("\n".join(md), encoding="utf-8")
    (RESULTS / f"results_enhanced_res_{MODEL}_{tag}.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")
    print(f"\nsaved {RESULTS / f'results_enhanced_res_{MODEL}_{tag}.md'}")


if __name__ == "__main__":
    main()
