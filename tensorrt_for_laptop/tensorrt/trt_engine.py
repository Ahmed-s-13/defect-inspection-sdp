"""Step 2 (on the NVIDIA machine): build TensorRT engines from the ONNX files, and run them.

  python trt_engine.py              builds bundle/engines/resnet18_fp32.engine and resnet18_fp16.engine
  python trt_engine.py --rebuild    rebuild even if the engine files exist

An engine is compiled for one GPU model and one TensorRT version, so it is always built on the machine that
runs it (on the Jetson it has to be built again there). Building takes about 1-5 minutes per engine.

TensorRT 11 notes: networks are strongly typed (no FP16 builder flag any more), so the FP16 engine is built from
resnet18_fp16.onnx, whose weights and maths are already FP16. Both engines take and return float32.

Each engine has two optimisation profiles: batch 1 (latency, one crop at a time like the wearable) and
batch 1-32 tuned for 32 (throughput). 32 x 224 x 224 crops fit easily in the RTX 2050's 4 GB.
"""
import argparse
import time
from pathlib import Path

import torch  # import torch before tensorrt (CUDA DLL order on Windows)

from common import BUNDLE

IN_SHAPE = (3, 224, 224)


def engine_path(bundle, prec):
    return Path(bundle) / "engines" / f"resnet18_{prec}.engine"


def build(onnx_path, out_path, max_batch=32, workspace_gb=1.0):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    # TensorRT 11: always strongly typed, no flags needed. TensorRT 10 (e.g. JetPack 6 on the Jetson) needs the
    # STRONGLY_TYPED flag, otherwise it may run the FP16 ONNX in FP32.
    flags = 0 if int(trt.__version__.split(".")[0]) >= 11 else 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
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
    (out_path.with_suffix(".txt")).write_text(f"tensorrt {trt.__version__}\ngpu {torch.cuda.get_device_name(0)}\n"
                                              f"max_batch {max_batch}\n")
    print(f"  built {out_path.name} in {time.time() - t0:.0f}s ({out_path.stat().st_size / 1e6:.1f} MB)")


class TRTResNet:
    """Runs a built engine on torch CUDA tensors: x float32 [N,3,224,224] on the GPU -> (logits, embedding).
    Uses profile 0 for N == 1 and profile 1 otherwise; output buffers are allocated once."""

    def __init__(self, path, stream):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise SystemExit(f"could not load {path}: built with another TensorRT version or GPU? "
                             "Run `python trt_engine.py --rebuild`.")
        self.stream = stream
        self.max_batch = self.engine.get_tensor_profile_shape("input", 1)[2][0]
        self.ctx, self.out = {}, {}
        for prof, n in [(0, 1), (1, self.max_batch)]:
            ctx = self.engine.create_execution_context()
            ctx.set_optimization_profile_async(prof, stream.cuda_stream)
            self.ctx[prof] = ctx
            self.out[prof] = {"logits": torch.empty((n, 6), device="cuda"),
                              "embedding": torch.empty((n, 512), device="cuda")}
        stream.synchronize()

    def __call__(self, x):
        n = x.shape[0]
        prof = 0 if n == 1 else 1
        ctx, out = self.ctx[prof], self.out[prof]
        ctx.set_input_shape("input", tuple(x.shape))
        ctx.set_tensor_address("input", x.data_ptr())
        for name, buf in out.items():
            ctx.set_tensor_address(name, buf.data_ptr())  # first n rows are used (contiguous)
        if not ctx.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        return out["logits"][:n], out["embedding"][:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default=str(BUNDLE))
    ap.add_argument("--precisions", nargs="+", default=["fp32", "fp16"], choices=["fp32", "fp16"])
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--workspace-gb", type=float, default=1.0, help="builder scratch memory on the GPU")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("No NVIDIA GPU visible to PyTorch (torch.cuda.is_available() is False). See README step 1-3.")
    import tensorrt as trt
    print(f"TensorRT {trt.__version__} on {torch.cuda.get_device_name(0)}")
    for prec in a.precisions:
        out = engine_path(a.bundle, prec)
        if out.exists() and not a.rebuild:
            print(f"  {out.name} exists (use --rebuild to rebuild)")
            continue
        print(f"building {prec.upper()} engine from resnet18_{prec}.onnx ...", flush=True)
        build(Path(a.bundle) / f"resnet18_{prec}.onnx", out, a.max_batch, a.workspace_gb)


if __name__ == "__main__":
    main()
