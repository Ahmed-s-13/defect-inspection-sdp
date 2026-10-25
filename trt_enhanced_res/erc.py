"""Shared paths and the exportable enhanced_res backbone (8-view TTA done inside the graph).

Lives outside git on purpose (PROJECT_HANDOFF rule: new experiments stay out of the repo). Reads the repo and the
earlier TensorRT bundle without changing them.
"""
import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "defect-inspection-sdp"
MAIN = REPO / "1_main_model_resnet18"
TRT_OLD = REPO / "tensorrt_for_laptop" / "tensorrt"          # normalize / timed / sync live here
ONNX_DIR = HERE / "onnx"
ENGINE_DIR = HERE / "engines"
RESULTS = HERE / "results"

for p in (MAIN / "enhanced_res", MAIN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# the repo's own code: EnhancedRes, feats, dihedral_t, load_split (model.py also puts MAIN on sys.path)
import model as er                                            # noqa: E402  (enhanced_res/model.py)
import importlib.util as _ilu                                 # noqa: E402

# the old bundle's common.py (normalize / timed / sync), loaded by path so it can't clash with other "common" modules
_spec = _ilu.spec_from_file_location("old_trt_common", TRT_OLD / "common.py")
old = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(old)
normalize, timed, sync = old.normalize, old.timed, old.sync


class TTABackbone(nn.Module):
    """x [N,3,224,224] float32 (ImageNet-normalised) -> (feats [N,768], logits [N,6]).

    feats  = pooled layer3 (256) + layer4 (512), averaged over the 8 flips/90-degree rotations: exactly
             enhanced_res.model.tta_feats, the input of the MLP head.
    logits = the backbone's own fc on the TTA-averaged layer4 part (fc is linear, so this equals averaging the
             8 views' logits): the simpler "r2 fc + TTA" classifier, no MLP head or kNN.
    half=True runs the network in FP16 but takes and returns float32 (TensorRT 11 engines are strongly typed:
    precision comes from the ONNX, there is no FP16 switch).
    """

    def __init__(self, backbone, half=False):
        super().__init__()
        self.m = backbone.half() if half else backbone
        self.half = half

    def forward(self, x):
        n = x.shape[0]
        if self.half:
            x = x.half()
        views = torch.cat([er.dihedral_t(x, k) for k in range(8)], 0)      # [8N,3,224,224]
        f = er.feats(self.m, views).float().view(8, n, -1).mean(0)         # [N,768]
        logits = self.m.fc(f[:, 256:].to(self.m.fc.weight.dtype)).float()
        return f, logits


def load_backbone_cpu():
    """The trained enhanced_res backbone (fp32, eval, CPU) and the class names."""
    m, classes = er.load_backbone(er.EWORK / er.BACKBONE_FILE, "cpu")
    return m, classes
