# ResNet-18: PyTorch vs TensorRT on the Windows laptop (RTX 2050)

**Goal:** before the Jetson Orin NX arrives, see how much faster TensorRT runs our trained ResNet-18 than
PyTorch does on the same NVIDIA GPU, and check that the accuracy stays the same.

This folder is **not in git**. It already contains everything the laptop needs:

```
tensorrt/
├── bundle/
│   ├── resnet18_fp32.onnx      the trained model, converted to ONNX (done on the Mac with export_onnx.py)
│   ├── resnet18_fp16.onnx      the same model in half precision (for the FP16 TensorRT engine)
│   ├── model_best.pt           the same model for PyTorch
│   ├── mlp_head.pt             the MLP head (so we also see the accuracy of the model predict.py uses)
│   └── test/                   the 4,293 D1-test crops + labels.csv
├── common.py                   shared code
├── export_onnx.py              PyTorch -> ONNX (already done, only needed again if the model changes)
├── trt_engine.py               ONNX -> TensorRT engine (run on the laptop)
└── benchmark.py                the test: PyTorch vs TensorRT, accuracy + latency + time
```

**Conversion path:** `model_best.pt` (PyTorch) → `resnet18_*.onnx` (on the Mac, done) → `resnet18_*.engine`
(TensorRT, built **on the laptop**). An engine only works on the GPU model it was built on, so it can't be made on the Mac.

All commands below go in **Windows PowerShell** (Start menu → type `PowerShell` → open it). No Visual Studio is needed.
The VS Code terminal also works, as long as it is a PowerShell terminal.

---

## Step 0: before you start

- **Plug the laptop into its charger.** On battery, the GPU is slowed down and the timings are meaningless.
- **Close Chrome, Teams and other big apps.** Task Manager showed RAM at 96% full. The test needs about 2 GB free.
- **Windows power mode:** Settings → System → Power & battery → Power mode → **Best performance**.
- **Lenovo Vantage (optional):** if it's installed, set Power / Thermal mode to **Extreme Performance**.

## Step 1: update the NVIDIA driver (required)

The laptop's driver is **537.70 from October 2023** (Task Manager shows it as 31.0.15.3770). Current
PyTorch and TensorRT builds use CUDA 13, which needs **driver 580 or newer**.

1. Go to <https://www.nvidia.com/en-us/drivers/>. Choose:
   - Product type: **GeForce**
   - Series: **GeForce RTX 20 Series (Notebooks)**
   - Product: **GeForce RTX 2050**
   - OS: **Windows 11**

   Download the latest **Game Ready** or **Studio** driver. The **NVIDIA App** does the same thing automatically.
2. Run the installer (Express), then **restart the laptop**.
3. Check it in PowerShell:

```powershell
nvidia-smi
```

The top line must show `Driver Version: 58x.xx` (or higher) and `CUDA Version: 13.x`. The table should list the
**NVIDIA GeForce RTX 2050**. If `nvidia-smi` is "not recognized", the driver didn't install; repeat step 1.

## Step 2: install Python 3.13 (64-bit)

```powershell
winget install -e --id Python.Python.3.13
```

Close PowerShell, open a new one, then check:

```powershell
py -3.13 --version
```

It should print `Python 3.13.x`. If `winget` isn't available, install "Python 3.13 (64-bit)" from
<https://www.python.org/downloads/windows/> and tick **"Add python.exe to PATH"**.

## Step 3: copy this folder to the laptop

On the Mac this folder is zipped as `Experiments/tensorrt_for_laptop.zip` (about 210 MB). Copy it to the laptop with
Google Drive or a USB stick. On the laptop, right-click it → **Extract All...** → type **`C:\sdp`** → Extract.
That creates `C:\sdp\tensorrt`.

Use a short path, and avoid a OneDrive or Desktop folder that syncs: long or synced paths break pip installs on Windows.

```powershell
cd C:\sdp\tensorrt
dir
```

You should see `bundle`, `benchmark.py`, `trt_engine.py`, and the other files listed above.

## Step 4: create a Python environment and install everything (one time, ~4 GB download)

```powershell
cd C:\sdp\tensorrt
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell says *"running scripts is disabled on this system"*, run the line below. It allows scripts in this
PowerShell window only. Then run the `Activate.ps1` line again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Your prompt now starts with `(.venv)`. Install the packages:

```powershell
python -m pip install --upgrade pip
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
pip install tensorrt-cu13==11.4.0.106 numpy==2.5.3 pandas==3.0.6 pillow==12.3.0
```

- **`torch` / `torchvision`:** PyTorch with NVIDIA GPU (CUDA 13.0) support, about 2.5 GB.
- **`tensorrt-cu13`:** NVIDIA TensorRT 11.4 for CUDA 13, about 1.5 GB. pip brings its own CUDA libraries;
  you **don't** need to install the CUDA Toolkit or cuDNN separately.

Check that Python sees the GPU and TensorRT:

```powershell
python -c "import torch, tensorrt; print(torch.cuda.is_available(), torch.cuda.get_device_name(0), 'TensorRT', tensorrt.__version__)"
```

It must print `True NVIDIA GeForce RTX 2050 TensorRT 11.4.0.106`. If it prints `False`, see Troubleshooting.

## Step 5: convert ONNX → TensorRT (one time, ~2-10 min)

```powershell
python trt_engine.py
```

This builds `bundle\engines\resnet18_fp32.engine` and `resnet18_fp16.engine`. TensorRT tries many GPU kernels
and keeps the fastest, which is why it takes a few minutes. You only do this once. Rebuild with
`python trt_engine.py --rebuild` after a driver or TensorRT update.

## Step 6: run the test (PyTorch and TensorRT, same model, same 4,293 test crops)

Everything in one go, about 3-5 min:

```powershell
python benchmark.py
```

Or one side at a time:

```powershell
python benchmark.py --backends torch_fp32 torch_fp16
```

```powershell
python benchmark.py --backends trt_fp32 trt_fp16
```

For steadier timings, add `--iters 1000`.

It prints one line per backend while running, then a summary table. The table is also saved to
`results\results_NVIDIA_GeForce_RTX_2050.md` (and `.json`). Send us that file.

### What the columns mean

| column | meaning |
|---|---|
| accuracy, macro-F1 | the ResNet's own output on all 4,293 test crops. On the Mac, PyTorch FP32 gives **0.8796 / 0.8125** |
| + MLP head acc | the same embeddings through the MLP head (the model in use). On the Mac: **0.9010** |
| same class as PyTorch FP32 | % of crops where this backend predicts the same defect as PyTorch FP32. ~100% = TensorRT kept the accuracy |
| latency b1 mean / p50 / p95 | time for **one crop** (already on the GPU) to go through the model, in milliseconds. p95 = 95% of runs were faster than this |
| end-to-end b1 | one crop from RAM: copy to GPU, normalise, run, read the answer. Closest to what the wearable does per crop |
| throughput b32 | crops per second when sent 32 at a time |
| whole test set (s) | seconds to classify all 4,293 crops (batches of 32, everything included) |
| speed-up (latency) | PyTorch FP32 latency ÷ this backend's latency. **2.00x = twice as fast** |

**What to look for:** TensorRT FP32 should match PyTorch's accuracy almost exactly and be faster. TensorRT FP16 should
be faster still, with "same class" around 99.9%. The rows also show whether FP16 changes accuracy at all; on the Mac,
FP16 agreed with FP32 on 99.95% of crops.

## The GPU shows 0% in Task Manager: is it on?

You don't need to enable anything. CUDA programs (PyTorch, TensorRT) **always run on the NVIDIA GPU**. The
"switch to the NVIDIA GPU" settings (NVIDIA Control Panel, Windows Graphics settings) only matter for games and
other graphics apps.

The 0% you saw was correct: nothing was using the GPU at that moment. And by default, Task Manager shows the **3D**,
Copy and Video graphs. **CUDA work doesn't appear on the 3D graph**, so it can stay at 0% even while the benchmark runs.

To see it:

1. **Task Manager:** Performance → **GPU 1 (NVIDIA GeForce RTX 2050)**. Click the small **˅** next to **"3D"**
   above the top-left graph, and pick **"Cuda"** (on some Windows versions it's called **"Compute_0"**). While
   `benchmark.py` runs, that graph goes up and **Dedicated GPU memory** rises to about 1-2 GB.
2. **Or `nvidia-smi`:** open a **second** PowerShell window while the benchmark runs:

```powershell
nvidia-smi -l 1
```

It refreshes every second. You'll see `python.exe` in the process list and `GPU-Util` well above 0%.
Press Ctrl+C to stop.

The benchmark also prints `device: cuda (NVIDIA GeForce RTX 2050)` at the start. If it says anything else, it
isn't using the GPU.

## Troubleshooting

| problem | fix |
|---|---|
| `torch.cuda.is_available()` is `False` | The driver is older than 580: redo step 1. Check `nvidia-smi` shows `CUDA Version: 13.x`. Also check you installed torch from the `cu130` link (`pip show torch` must show version `2.14.0+cu130`) |
| The driver can't be updated to 580+ | Use the CUDA 12 builds: `pip install --force-reinstall torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126`, then `pip uninstall -y tensorrt-cu13` and `pip install tensorrt-cu12==11.4.0.106` |
| `DLL load failed` / `WinError 126` when importing torch | Install the Microsoft Visual C++ Redistributable (x64): <https://aka.ms/vs/17/release/vc_redist.x64.exe>, then restart PowerShell |
| `out of memory` while building or running | The RTX 2050 has 4 GB. Close other apps, then run `python trt_engine.py --rebuild --max-batch 16 --workspace-gb 0.5` and `python benchmark.py --batch 16` |
| `could not load ... engine` | It was built with another TensorRT version or driver: `python trt_engine.py --rebuild` |
| very different timings between runs | Plug in the charger, set Best performance, close other apps, use `--iters 1000` |

## What this tells us about the Jetson

- **What carries over:** the **relative** result: does TensorRT give the same accuracy, and is it roughly 1.5-3x
  faster than PyTorch on the same GPU? The RTX 2050 and the Jetson Orin NX GPU are both NVIDIA Ampere GPUs.
- **What doesn't:** the absolute milliseconds. The Jetson's GPU is smaller and shares memory with its CPU.
- **On the Jetson:**
  - TensorRT comes with **JetPack**, not with `pip install tensorrt-cu13`.
  - Copy the same `bundle` (the ONNX files work as they are) and run `python3 trt_engine.py` there to build Jetson engines.
  - `trt_engine.py` also handles TensorRT 10, the version JetPack 6 ships.
- **About TensorRT 11:** it removed the old "--fp16" switch. That's why there is a separate `resnet18_fp16.onnx`
  (FP16 weights, float32 input and output) instead of a flag.
