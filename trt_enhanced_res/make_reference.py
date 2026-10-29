"""Record the PyTorch model's answers on the 30 samples (1_main_model_resnet18/samples/) in samples_reference.json.
predict_trt.py --selftest and predict_onnx.py --selftest compare against it, so a new machine (Raspberry Pi, Jetson, a
teammate's laptop) can check in seconds that its runtime gives the same answers as PyTorch.

Uses the repo's own predict.py cropping and the repo's EnhancedRes + architecture B, so it needs the repo's Python
environment (the repo-root .venv):
  ..\\.venv\\Scripts\\python.exe make_reference.py          (from trt_enhanced_res/)
Re-run it when the weights or the calibration change.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent / "1_main_model_resnet18"
sys.path.insert(0, str(MAIN))

import torch                                                        # noqa: E402

from enhanced_res.arch_a.runtime import CALIBRATIONS, ArchA        # noqa: E402
from enhanced_res.model import MODELS, EnhancedRes                  # noqa: E402
from klshift.common import IMG_EXTS, eval_transform                 # noqa: E402
from predict import to_crop                                         # noqa: E402

model, arch = EnhancedRes(MODELS["gc7"]), ArchA(CALIBRATIONS["gc7"])
assert model.classes == arch.classes
files = sorted(f for f in (MAIN / "samples").iterdir() if f.suffix.lower() in IMG_EXTS)
tf = eval_transform()
x = torch.stack([tf(to_crop(f)) for f in files])
with torch.no_grad():
    f = model.features(x)
    dec = arch.decide(model.proba_from_features(f).cpu().numpy(), knn_dist=model.knn_distance(f).cpu().numpy(),
                      feats=f.cpu().numpy())
ref = {p.name: {"action": d.action, "classes": d.classes} for p, d in zip(files, dec)}
(HERE / "samples_reference.json").write_text(json.dumps(ref, indent=1))
acts = {}
for r in ref.values():
    acts[r["action"]] = acts.get(r["action"], 0) + 1
print(f"{len(ref)} samples -> samples_reference.json; actions: {acts}")
