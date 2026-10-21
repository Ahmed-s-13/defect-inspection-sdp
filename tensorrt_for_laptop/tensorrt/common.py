"""Shared code for the ONNX / TensorRT experiment. Self-contained on purpose: the NVIDIA laptop only needs
this folder and the bundle/ made by export_onnx.py, not the git repo or its 2.8 GB data zip.

Mirrors 1_main_model_resnet18 (klshift/common.py and klshift/mlp.py) without changing it:
  - the model is a stock torchvision ResNet-18 with fc = Linear(512, 6)
  - crops are 224x224 JPEGs, normalised with the ImageNet mean/std
  - the MLP head is 5 MLPs averaged, blended with a 5-NN vote over the D1-train embeddings
"""
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn

HERE = Path(__file__).resolve().parent
BUNDLE = HERE / "bundle"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EMB_DIM = 512


def pick_device(name="auto"):
    """'auto' = the NVIDIA GPU (CUDA) when there is one, else the Apple GPU (MPS), else CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def build_resnet(state, n_classes):
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model.load_state_dict(state)
    return model.eval()


class ResNetWithEmbedding(nn.Module):
    """ResNet-18 returning (logits, 512-d embedding). With half=True the network runs in FP16 but takes and
    returns float32, which is how an FP16 TensorRT 11 engine is built (strongly typed ONNX)."""

    def __init__(self, model, half=False):
        super().__init__()
        self.m = model.half() if half else model
        self.half = half

    def forward(self, x):
        m = self.m
        if self.half:
            x = x.half()
        x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
        x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
        emb = torch.flatten(m.avgpool(x), 1)
        return m.fc(emb).float(), emb.float()


def load_images(folder, labels_csv):
    """All crops of the bundle as one uint8 NCHW array (650 MB for the 4,293 test crops, normalised later on the
    GPU to keep RAM low), plus labels and class names."""
    import pandas as pd
    from PIL import Image
    lab = pd.read_csv(labels_csv)
    classes = sorted(lab.cls.unique())
    x = np.empty((len(lab), 3, 224, 224), dtype=np.uint8)
    for i, f in enumerate(lab.file):
        x[i] = np.asarray(Image.open(Path(folder) / f).convert("RGB")).transpose(2, 0, 1)
    return x, np.array([classes.index(c) for c in lab.cls]), classes


def normalize(x_uint8):
    """uint8 NCHW tensor (any device) -> float32, ImageNet-normalised, same as the training eval_transform."""
    mean = torch.as_tensor(MEAN, device=x_uint8.device).view(1, 3, 1, 1)
    std = torch.as_tensor(STD, device=x_uint8.device).view(1, 3, 1, 1)
    return (x_uint8.float() / 255.0 - mean) / std


# ---------------------------------------------------------------- MLP head (same as klshift/mlp.py)

class MLPHead(nn.Module):
    def __init__(self, n_classes, hidden=(1024, 512, 256), dropout=0.5, in_dropout=0.2):
        super().__init__()
        self.register_buffer("mu", torch.zeros(EMB_DIM))
        self.register_buffer("sd", torch.ones(EMB_DIM))
        layers, d = [nn.Dropout(in_dropout)], EMB_DIM
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, emb):
        return self.net((emb - self.mu) / self.sd)


def l2n(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


class MLPClassifier:
    """mlp_head.pt from train_mlp.py: 5 MLPs averaged, blended with a 5-NN vote. Embedding -> class probabilities."""

    def __init__(self, path, device):
        ck = torch.load(path, map_location="cpu")
        self.heads = []
        for st in ck["states"]:
            h = MLPHead(len(ck["classes"]), **ck["arch"])
            h.load_state_dict(st)
            self.heads.append(h.to(device).eval())
        self.classes, self.k, self.w = ck["classes"], ck["knn_k"], ck["knn_weight"]
        if self.w > 0:
            self.bank = l2n(ck["knn_bank"].float().to(device))
            self.bank_y = ck["knn_labels"].long().to(device)

    @torch.no_grad()
    def __call__(self, emb):
        p = torch.stack([h(emb).softmax(1) for h in self.heads]).mean(0)
        if self.w > 0:
            sim, idx = (l2n(emb) @ self.bank.T).topk(self.k, dim=1)
            w = 1.0 / (1.0 - sim).clamp_min(1e-6)
            knn = torch.zeros_like(p).scatter_add_(1, self.bank_y[idx], w)
            p = (1 - self.w) * p + self.w * knn / knn.sum(1, keepdim=True)
        return p


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
