"""Step 2: build TensorRT engines for the WHOLE enhanced_res (adapted from tensorrt_for_laptop/tensorrt/trt_engine.py,
which hard-codes the legacy model's output names and sizes).

  python trt_runner.py                 builds engines/enhanced_res_full_fp32.engine and _fp16.engine from onnx/*.onnx
  python trt_runner.py --rebuild       rebuild even if they exist

An engine only runs on the GPU model and TensorRT version it was built with, so build it on the machine that uses it
(the Jetson builds its own). Two optimisation profiles: 1 crop (latency, the wearable's case) and 1..MAX crops tuned
for MAX (throughput). One crop = 8 TTA views inside, so MAX=4 crops means 32 views.
Engine in: input float32 [N,3,224,224]; out: probs float32 [N,6].

Building engines needs no PyTorch (only tensorrt + cuda-python, as in requirements.txt). TRTModel below keeps its
buffers in torch CUDA tensors and is only used by benchmark.py; predict_trt.py has its own PyTorch-free runner.
"""
import argparse
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ONNX_DIR, ENGINE_DIR = HERE / "onnx", HERE / "engines"                   # = erc.ONNX_DIR / erc.ENGINE_DIR
IN_SHAPE = (3, 224, 224)
N_CLASSES = 6


def engine_path(prec):
    return ENGINE_DIR / f"enhanced_res_full_{prec}.engine"


def gpu_name():
    """Name of CUDA device 0, or None if there is no usable NVIDIA GPU (cuda-python, no PyTorch)."""
    from cuda.bindings import runtime as rt
    err, n = rt.cudaGetDeviceCount()
    if err != rt.cudaError_t.cudaSuccess or n == 0:
        return None
    err, prop = rt.cudaGetDeviceProperties(0)
    return prop.name.decode(errors="replace").strip("\x00") if err == rt.cudaError_t.cudaSuccess else "NVIDIA GPU"


def build(onnx_path, out_path, max_batch=4, workspace_gb=1.0):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TensorRT 11: always strongly typed. TensorRT 10 (JetPack 6 on the Jetson) needs the STRONGLY_TYPED flag.
    flags = 0 if int(trt.__version__.split(".")[0]) >= 11 else 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise SystemExit("ONNX parse failed:\n" + "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    for opt in [1, max_batch]:  # profile 0: batch 1 only; profile 1: batch 1..max, tuned for max
        p = builder.create_optimization_profile()
        lo, hi = (1, 1) if opt == 1 else (1, max_batch)
        p.set_shape("input", (lo, *IN_SHAPE), (opt, *IN_SHAPE), (hi, *IN_SHAPE))
        config.add_optimization_profile(p)
    t0 = time.time()
    data = builder.build_serialized_network(network, config)
    if data is None:
        raise SystemExit(f"TensorRT could not build {onnx_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(data))
    out_path.with_suffix(".txt").write_text(f"tensorrt {trt.__version__}\ngpu {gpu_name()}\nmax_batch {max_batch}\n")
    print(f"  built {out_path.name} in {time.time() - t0:.0f}s ({out_path.stat().st_size / 1e6:.1f} MB)", flush=True)


class TRTModel:
    """Runs a built engine on torch CUDA tensors: x float32 [N,3,224,224] on the GPU -> probs [N,6]."""

    def __init__(self, path, stream):
        import tensorrt as trt
        import torch
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise SystemExit(f"could not load {path}: built with another TensorRT version or GPU? "
                             "Run `python trt_runner.py --rebuild`.")
        self.stream = stream
        self.max_batch = self.engine.get_tensor_profile_shape("input", 1)[2][0]
        self.ctx, self.out = {}, {}
        for prof, n in [(0, 1), (1, self.max_batch)]:
            ctx = self.engine.create_execution_context()
            ctx.set_optimization_profile_async(prof, stream.cuda_stream)
            self.ctx[prof] = ctx
            self.out[prof] = torch.empty((n, N_CLASSES), device="cuda")
        stream.synchronize()

    def __call__(self, x):
        n = x.shape[0]
        prof = 0 if n == 1 else 1
        ctx, out = self.ctx[prof], self.out[prof]
        ctx.set_input_shape("input", tuple(x.shape))
        ctx.set_tensor_address("input", x.data_ptr())
        ctx.set_tensor_address("probs", out.data_ptr())  # first n rows are used (contiguous)
        if not ctx.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return out[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precisions", nargs="+", default=["fp32", "fp16"], choices=["fp32", "fp16"])
    ap.add_argument("--max-batch", type=int, default=4, help="max crops per call (x8 views inside)")
    ap.add_argument("--workspace-gb", type=float, default=1.0)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    gpu = gpu_name()
    if gpu is None:
        raise SystemExit("No NVIDIA GPU visible (check the driver with nvidia-smi).")
    import tensorrt as trt
    print(f"TensorRT {trt.__version__} on {gpu}")
    for prec in a.precisions:
        out = engine_path(prec)
        if out.exists() and not a.rebuild:
            print(f"  {out.name} exists (use --rebuild)")
            continue
        print(f"building {prec.upper()} engine ...", flush=True)
        build(ONNX_DIR / f"enhanced_res_full_{prec}.onnx", out, a.max_batch, a.workspace_gb)


if __name__ == "__main__":
    main()
