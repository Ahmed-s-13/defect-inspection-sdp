"""File locations shared by every script in this folder (and by the Raspberry Pi bundle, which has the same layout).

  onnx/enhanced_res_<model>_full_<fp32|fp16>.onnx      the whole model as one ONNX file (not in git: Google Drive)
  engines/enhanced_res_<model>_full_<fp32|fp16>.engine the TensorRT engine built from it on THIS machine's GPU
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ONNX_DIR, ENGINE_DIR, RESULTS = HERE / "onnx", HERE / "engines", HERE / "results"
MODEL = "gc7"            # the main model: 7 classes (six defects + no-defect) with architecture B on top


def onnx_path(prec, model=MODEL):
    return ONNX_DIR / f"enhanced_res_{model}_full_{prec}.onnx"


def engine_path(prec, model=MODEL):
    return ENGINE_DIR / f"enhanced_res_{model}_full_{prec}.engine"
