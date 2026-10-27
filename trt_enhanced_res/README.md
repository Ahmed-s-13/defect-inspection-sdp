# enhanced_res on TensorRT

The main model (**enhanced_res**: ResNet-18 + 8-view TTA + MLP heads + 5-NN, see
[`../1_main_model_resnet18/enhanced_res/`](../1_main_model_resnet18/enhanced_res/README.md)) exported **as a whole**
to ONNX and run with **TensorRT**. `predict_trt.py` is the TensorRT version of `1_main_model_resnet18/predict.py`:
same crop, same model, same output, faster, and **no PyTorch needed** to run it (the step towards the Jetson).

**It gives the same answers as PyTorch.** On all 4,255 grouped D1-test crops, both engines (FP32 and FP16) predict
the same class as PyTorch for every crop: accuracy **0.8045**, macro-F1 **0.7376**, balanced accuracy 0.7367,
identical per-class recall. On the 30 `samples/` the top-2 classes match on every image (probabilities within 0.5%).

| on the RTX 2050 laptop (300 runs) | PyTorch FP32 | PyTorch FP16 | TensorRT FP32 | **TensorRT FP16** |
|---|---|---|---|---|
| one crop (8 views), whole model | 48.5 ms | 34.1 ms | 39.5 ms | **14.5 ms** |
| speed-up vs PyTorch FP32 | 1.00x | 1.42x | 1.23x | **3.34x** |
| all 4,255 test crops | 180 s | 126 s | 165 s | **55 s** |

Full table: [`results/results_enhanced_res_full_NVIDIA_GeForce_RTX_2050.md`](results/results_enhanced_res_full_NVIDIA_GeForce_RTX_2050.md).
Timings on this laptop GPU vary between runs (TensorRT FP32 measured 12-40 ms in different runs); FP16 is the
fastest in every run. The RTX 2050 is not the Jetson Orin NX: relative speed-ups carry over, absolute ms do not.

## 1. Download the ONNX files (team Google Drive)

The two ONNX files are 136 MB and 114 MB, over GitHub's 100 MB limit, so they are **not in git**. Download
**`enhanced_res_onnx.zip`** (227 MB) from the team's shared Google Drive (same place as `main_model_data_work.zip`).
Drive warns it can't scan files this big for viruses: click **Download anyway**.

Put the zip in the **repo root** (`defect-inspection-sdp/`) and unzip it there. It is laid out relative to the repo
root, so the files land in `trt_enhanced_res/onnx/`:

```
trt_enhanced_res/onnx/enhanced_res_full_fp32.onnx   136 MB   input "input" [N,3,224,224] -> output "probs" [N,6]
trt_enhanced_res/onnx/enhanced_res_full_fp16.onnx   114 MB   same, backbone in FP16 (head and kNN in FP32)
```

Optional check that the download is complete (PowerShell, in the repo root):

```powershell
Get-FileHash enhanced_res_onnx.zip -Algorithm SHA256
```

| file | SHA-256 |
|---|---|
| `enhanced_res_onnx.zip` | `6df49fc610cb850652ccc010cb6974f6d53fa5c2a7dca8eb8d66f2bfb600321d` |
| `enhanced_res_full_fp32.onnx` | `d52321b67a75537e1b027be1de66f861aff7d8fb37b2bc8fce97f3e47c39d7a9` |
| `enhanced_res_full_fp16.onnx` | `d254eb9fb4a1076b0eba376359c85db115ca65916c5e8607a30fb08db620f9a1` |

(Or skip the download and make them yourself with `python export_onnx.py`, see section 5.)

## 2. Set up Python (once)

Needs an NVIDIA GPU with **driver 580 or newer** (`nvidia-smi` shows `CUDA Version: 13.x`) and Python 3.13.
In PowerShell, from `trt_enhanced_res/`:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell says "running scripts is disabled", run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
and activate again. The `tensorrt_for_laptop/tensorrt/README.md` has more help for driver and install problems.
Check: `python -c "import tensorrt; print(tensorrt.__version__)"` prints `11.4.0.106`.

## 3. Build the TensorRT engines on your machine (once, ~2 min)

A TensorRT engine only runs on the GPU model and TensorRT version that built it, so the engines are **not shared**:
every machine builds its own from the ONNX files.

```powershell
python trt_runner.py
```

This writes `engines/enhanced_res_full_fp32.engine` and `_fp16.engine` (~150 MB and ~118 MB, git-ignored) and a
`.txt` note next to each with the GPU and TensorRT version they were built for. After a driver or TensorRT update:
`python trt_runner.py --rebuild`. Building needs no PyTorch, only the packages in `requirements.txt`.

## 4. Run it

From `trt_enhanced_res/`:

```powershell
python predict_trt.py ..\1_main_model_resnet18\samples\                      # 30 test crops, accuracy at the end (0.800)
python predict_trt.py photo.jpg --box 120 80 260 190                        # defect box in pixels: cut exactly like training
python predict_trt.py photo.jpg                                             # no box: centre square of the photo
python predict_trt.py photo.jpg --precision fp16                            # the faster FP16 engine
python predict_trt.py photo.jpg --raw --top 3                               # raw ranking without architecture A
python predict_trt.py --grouped-test                                        # all 4,255 test crops; must end "OK: TensorRT matches ..."
```

Output is the same as `predict.py`: [architecture A](../1_main_model_resnet18/enhanced_res/arch_a/README.md)'s answer,
e.g. `crack_01.jpg:  AMBIGUOUS: crack 52.1%   dent 44.0%   [true: crack]`. It uses `runtime.py` from the main folder,
which needs only numpy and PIL. Its quality gate, temperature and conformal set run here. The unfamiliar-input (OOD) gate
needs the model's 768-d features, which the engine doesn't output (`probs` only), so it is skipped. Adding a
`features` output to `export_onnx.py` would enable it. `--grouped-test` also checks architecture A's CP coverage and
set size against PyTorch.
`--grouped-test` also needs `1_main_model_resnet18/data/work/` (main_model_data_work.zip) and the file
`data/work/enhanced_res/split.csv` (written the first time the repo's `enhanced_res/evaluate.py` runs).

Like `predict.py`, the model has no "no damage" class: it always names one of the 6 defects.

## 5. Re-making the ONNX files and the benchmark (only if the model changes)

Needs PyTorch with CUDA and the extra packages listed at the bottom of `requirements.txt`, plus the enhanced_res
weights (`enhanced_res_model.zip`, GitHub Release `v2.0-enhanced-res`) and `main_model_data_work.zip`.

```powershell
python export_onnx.py      # whole model -> onnx/enhanced_res_full_{fp32,fp16}.onnx + classes.json, checked vs PyTorch
python trt_runner.py --rebuild
python benchmark.py        # PyTorch FP32/FP16 vs TensorRT FP32/FP16 on the 4,255 test crops (~12 min)
```

Only a change to the **model** (new weights, model code, preprocessing) needs a new ONNX. Changes to how results are
printed (like architecture A's decision layer) only need the same change in `predict_trt.py`. Re-running
`1_main_model_resnet18/enhanced_res/arch_a/calibrate.py` (new labels or split) needs no new ONNX either.

## Files

| file | what |
|---|---|
| `predict_trt.py` | classify images with the TensorRT engine (no PyTorch) |
| `trt_runner.py` | build the engines from the ONNX files; `TRTModel` runner used by `benchmark.py` |
| `export_onnx.py` | export the whole enhanced_res to ONNX and check it against PyTorch |
| `erc.py` | the exportable model (`FullModel`, `TTABackbone`) and shared paths |
| `benchmark.py` | accuracy + speed of PyTorch vs TensorRT |
| `onnx/classes.json` | class order of the `probs` output |
| `engines/*.txt` | GPU and TensorRT version the local engines were built with |
| `results/results_enhanced_res_full_*` | benchmark of the whole model (current) |
| `results/results_enhanced_res_NVIDIA_*`, `bench_run.log` | earlier benchmark: backbone on TensorRT, head on PyTorch (same accuracy) |
| `bench_full_run.log` | console log of the current benchmark run |

## On the Jetson

Copy this folder and the two ONNX files, then run `python3 trt_runner.py` there to build Jetson engines. TensorRT
comes with JetPack (TensorRT 10; `trt_runner.py` handles both 10 and 11), and `cuda-python` should match the
JetPack CUDA version. `predict_trt.py` then runs without PyTorch.
