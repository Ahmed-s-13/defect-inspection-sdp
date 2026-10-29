"""Make the two zips to share (both are git-ignored; they go on the team Google Drive):

  python make_zips.py            -> enhanced_res_gc7_onnx.zip   the two ONNX files, laid out relative to the repo root
                                                                  (teammates unzip it in the repo root: README.md step 1)
                                   enhanced_res_pi_bundle.zip   everything a Raspberry Pi needs, with its own README
                                                                  (copy to the Pi, unzip anywhere: no repo, no PyTorch)

Each zip is read back and every file compared with its source. SHA-256 of the zips and the ONNX files are printed:
paste them into README.md. Re-make both zips whenever export_onnx.py has produced new ONNX files.
"""
import argparse
import hashlib
import zipfile
from pathlib import Path

from trt_paths import HERE, MODEL, onnx_path

REPO = HERE.parent
ARCH_A = REPO / "1_main_model_resnet18" / "enhanced_res" / "arch_a"
SAMPLES = REPO / "1_main_model_resnet18" / "samples"

PI_REQUIREMENTS = """# Raspberry Pi 5 (64-bit Raspberry Pi OS) or any other CPU. No PyTorch, no TensorRT, no GPU.
# Tested: onnxruntime 1.31.0 on x86 Windows (Python 3.13); the Pi (ARM64) wheels come from PyPI the same way.
onnxruntime>=1.20
numpy>=1.26
pillow>=10
"""

PI_README = f"""# enhanced_res on the Raspberry Pi 5

The main model (enhanced_res gc7: ResNet-18 + MLP heads + kNN, seven classes) with the calibrated decision layer
(architecture B), running on the Pi's CPU through onnxruntime. One ONNX file, no PyTorch, no GPU.

## 1. Set up (once)

Needs the 64-bit Raspberry Pi OS (check: `uname -m` prints `aarch64`) and Python 3.11 or newer.

```bash
unzip enhanced_res_pi_bundle.zip -d enhanced_res_pi
cd enhanced_res_pi
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_pi.txt
```

## 2. Check that it works (30 samples, takes a few seconds)

```bash
python predict_onnx.py --selftest
```

It must end with `SELFTEST: 30/30 samples give the same answer ... as the PyTorch reference` and `OK`. That means the Pi
gives the same answers as the PyTorch model on the team's machines.

## 3. Use it

```bash
python predict_onnx.py photo.jpg --box 120 80 260 190    # the defect box in pixels of the photo (best)
python predict_onnx.py photo.jpg                          # no box: the centre square of the photo
python predict_onnx.py some_folder/                       # every image in a folder
python predict_onnx.py photo.jpg --details                # also the unfamiliar-input p-value and the full ranking
python predict_onnx.py --bench 20                         # speed: median ms for one crop (8 views)
python predict_onnx.py photo.jpg --threads 4              # onnxruntime threads (default: all 4 cores)
```

What the answer means:

| shown | meaning |
|---|---|
| (nothing shown: no damage) | only "no-defect" is likely enough to keep |
| POSSIBLE DENT: check | a defect and "no-defect" are both likely: a person should look |
| CRACK 88.0% | one class is likely enough to keep; the % is calibrated |
| AMBIGUOUS: crack 52% dent 31% | two or more classes kept: a person should look |
| CAN'T ASSESS: re-aim | not aircraft skin, or unlike any training photo: point the camera at the surface |
| UNCLEAR: re-inspect | no class is likely enough |

The set holds the true class about 90% of the time on normal crops. The model was trained on square crops around a
defect (the box plus 25%); `--box` cuts exactly that.

## Notes

- Use the FP32 file (`onnx/{onnx_path('fp32').name}`). The FP16 file is only for TensorRT on an NVIDIA GPU: it is not
  faster on a CPU.
- The 16 GB of RAM is far more than needed (about 1 GB while running). What matters is the ARM CPU: speed is set by
  `--bench`. One crop is 8 rotated/flipped views through a ResNet-18 plus the head.
- To change the model (new weights, new calibration), re-export on the PC (`python export_onnx.py`), re-make the bundle
  (`python make_zips.py`) and copy it over again.
"""


def sha(path_or_bytes):
    h = hashlib.sha256()
    if isinstance(path_or_bytes, (bytes, bytearray)):
        h.update(path_or_bytes)
    else:
        with open(path_or_bytes, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
    return h.hexdigest()


def write_zip(out, entries):
    """entries: [(arcname, source Path or bytes)]. Writes, reads back and compares every file."""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for name, src in entries:
            z.writestr(name, src) if isinstance(src, bytes) else z.write(src, name)
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None, f"{out.name} is corrupt"
        for name, src in entries:
            assert sha(z.read(name)) == sha(src), f"{name} differs inside {out.name}"
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB)\n  sha256 {sha(out)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO), help="folder for the zips (default: the repo root; *.zip is git-ignored)")
    a = ap.parse_args()
    out = Path(a.out)
    for prec in ("fp32", "fp16"):
        if not onnx_path(prec).exists():
            raise SystemExit(f"{onnx_path(prec)} missing: run `python export_onnx.py`")
    for f in (ARCH_A / "runtime.py", ARCH_A / f"calibration_{MODEL}.json", SAMPLES / "labels.csv", HERE / "samples_reference.json"):
        if not f.exists():
            raise SystemExit(f"{f} missing")

    rel = lambda p: p.relative_to(REPO).as_posix()           # noqa: E731
    write_zip(out / f"enhanced_res_{MODEL}_onnx.zip", [(rel(onnx_path(p)), onnx_path(p)) for p in ("fp32", "fp16")])
    pi = [(f"onnx/{onnx_path('fp32').name}", onnx_path("fp32")),
          ("predict_onnx.py", HERE / "predict_onnx.py"), ("infer_common.py", HERE / "infer_common.py"),
          ("trt_paths.py", HERE / "trt_paths.py"), ("samples_reference.json", HERE / "samples_reference.json"),
          ("arch_a/runtime.py", ARCH_A / "runtime.py"), (f"arch_a/calibration_{MODEL}.json", ARCH_A / f"calibration_{MODEL}.json"),
          ("requirements_pi.txt", PI_REQUIREMENTS.encode()), ("README_PI.md", PI_README.encode())]
    pi += [(f"samples/{f.name}", f) for f in sorted(SAMPLES.iterdir()) if f.is_file()]
    write_zip(out / "enhanced_res_pi_bundle.zip", pi)
    print("\nONNX file checksums (for README.md):")
    for prec in ("fp32", "fp16"):
        print(f"  {sha(onnx_path(prec))}  {onnx_path(prec).name}")


if __name__ == "__main__":
    main()
