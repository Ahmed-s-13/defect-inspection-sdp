"""Shared code for the scripts that need PyTorch (export_onnx.py, benchmark.py, make_reference.py): the repo's model
code, and FullModel, the whole enhanced_res in one exportable graph.

predict_trt.py and predict_onnx.py do not import this (they need no PyTorch).
Reads 1_main_model_resnet18 without changing it.
"""
import sys
import time

import numpy as np
import torch
from torch import nn

from trt_paths import HERE, MODEL, ONNX_DIR, RESULTS  # noqa: F401  (re-exported for the scripts)

REPO = HERE.parent                                           # this folder lives in the repo root
MAIN = REPO / "1_main_model_resnet18"

for p in (MAIN / "enhanced_res", MAIN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# the repo's own code: EnhancedRes, MODELS, feats, dihedral_t, load_split (model.py also puts MAIN on sys.path)
import model as er                                            # noqa: E402  (enhanced_res/model.py)

# ImageNet normalisation and timing helpers
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def normalize(x_uint8):
    """uint8 NCHW tensor (any device) -> float32, ImageNet-normalised, same as the training eval_transform."""
    mean = torch.as_tensor(MEAN, device=x_uint8.device).view(1, 3, 1, 1)
    std = torch.as_tensor(STD, device=x_uint8.device).view(1, 3, 1, 1)
    return (x_uint8.float() / 255.0 - mean) / std


def timed(fn, device, warmup, iters):
    """Latency of fn() in ms per call: list of `iters` timings after `warmup` untimed calls."""
    for _ in range(warmup):
        fn()
    sync(device)
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync(device)
        out.append((time.perf_counter() - t0) * 1000)
    return np.array(out)


class TTABackbone(nn.Module):
    """x [N,3,224,224] float32 (ImageNet-normalised) -> features [N,768]: pooled layer3 (256) + layer4 (512),
    averaged over the 8 flips/90-degree rotations (exactly enhanced_res.model.tta_feats), the input of the head.
    half=True runs the network in FP16 but takes and returns float32 (TensorRT 11 engines are strongly typed:
    precision comes from the ONNX, there is no FP16 switch)."""

    def __init__(self, backbone, half=False):
        super().__init__()
        self.m = backbone.half() if half else backbone
        self.half = half

    def forward(self, x):
        n = x.shape[0]
        if self.half:
            x = x.half()
        views = torch.cat([er.dihedral_t(x, k) for k in range(8)], 0)      # [8N,3,224,224]
        return er.feats(self.m, views).float().view(8, n, -1).mean(0)      # [N,768]


class FullModel(nn.Module):
    """x [N,3,224,224] float32 (ImageNet-normalised) -> (probs [N,C], feats [N,768], knn_dist [N]): the whole
    enhanced_res exactly as EnhancedRes does it (backbone + 8-view TTA, 5 MLPs averaged, blended with the 5-NN vote,
    class-prior correction), plus the two numbers architecture B's unfamiliar-input gate needs:
      feats     EnhancedRes.features(x)
      knn_dist  EnhancedRes.knn_distance(feats) = 1 - mean cosine similarity to the 5 nearest training crops
                (the same top-5 similarities the kNN vote already computes)

    The kNN uses a one-hot label table + gather instead of scatter_add, so it exports to ONNX/TensorRT.
    half=True runs only the backbone in FP16; the head, the kNN bank and the outputs stay FP32 (the top-5 neighbour
    search is sensitive to rounding)."""

    def __init__(self, model=MODEL, half=False):
        super().__init__()
        work = er.MODELS[model]
        backbone, _ = er.load_backbone(work / er.BACKBONE_FILE, "cpu")
        self.tta = TTABackbone(backbone, half=half)
        ck = torch.load(work / er.HEAD_FILE, map_location="cpu")
        assert ck["tta"], "the exported graph always does the 8-view TTA"
        assert float(ck["alpha"]) > 0, "the exported graph needs the kNN bank (alpha > 0)"
        self.classes = ck["classes"]
        self.heads = nn.ModuleList()
        for st in ck["heads"]:
            h = er.Head(ck["d_in"], len(self.classes))
            h.load_state_dict(st)
            self.heads.append(h.eval())
        self.c0, self.c1 = ck["cols"]
        self.alpha, self.tau, self.k = float(ck["alpha"]), float(ck["tau"]), 5
        self.register_buffer("prior", ck["prior"].float())
        bank = ck["knn_bank"].float()
        self.register_buffer("bank_t", (bank / bank.norm(dim=1, keepdim=True).clamp_min(1e-12)).T.contiguous())
        self.register_buffer("onehot", torch.eye(len(self.classes))[ck["knn_labels"].long()])
        self.eval()

    def forward(self, x):
        f = self.tta(x)[:, self.c0:self.c1]
        p = torch.stack([h(f).softmax(1) for h in self.heads]).mean(0)
        q = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
        sim, idx = (q @ self.bank_t).topk(self.k, dim=1)
        w = 1.0 / (1.0 - sim).clamp_min(1e-6)
        knn = (self.onehot[idx] * w.unsqueeze(2)).sum(1)
        p = (1 - self.alpha) * p + self.alpha * knn / knn.sum(1, keepdim=True)
        if self.tau > 0:
            p = p / self.prior ** self.tau
            p = p / p.sum(1, keepdim=True)
        return p, f, 1 - sim.mean(1)
