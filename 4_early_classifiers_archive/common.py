"""Shared code for both models and the evaluation.

Everything that must be identical between Model 1 (plain CNN) and Model 2
(SupCon) lives here: data loading, augmentation, the network wrapper, the
metrics, the image corruptions and the masking used by the shortcut tests.
"""
from __future__ import annotations

import io
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models
from torchvision import transforms as T

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BACKGROUND = "no_damage"


# --------------------------------------------------------------------------- basics
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_metadata(data_dir: str | Path) -> tuple[pd.DataFrame, list[str]]:
    data_dir = Path(data_dir)
    if not (data_dir / "metadata.csv").exists():
        found = sorted(str(p.parent) for p in data_dir.glob("*/metadata.csv"))
        hint = f" Did you mean one of: {', '.join(found)}?" if found else " Run prepare_data.py first."
        raise SystemExit(f"{data_dir} is not a prepared dataset (no metadata.csv).{hint}")
    df = pd.read_csv(data_dir / "metadata.csv")
    classes = json.loads((data_dir / "classes.json").read_text())
    df["y"] = df["label"].map({c: i for i, c in enumerate(classes)}).astype(int)
    return df, classes


# --------------------------------------------------------------------------- data
class CropDataset(Dataset):
    """One item = one defect crop. `pre` is an optional PIL-level operation
    (a corruption or a mask) applied before the tensor transform."""

    def __init__(self, df, data_dir, transform, two_views=False, pre=None):
        self.paths = [str(Path(data_dir) / p) for p in df["path"].tolist()]
        self.y = df["y"].to_numpy()
        self.boxes = df[["bx0", "by0", "bx1", "by1"]].to_numpy(dtype=float)
        self.transform, self.two_views, self.pre = transform, two_views, pre

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.pre is not None:
            img = self.pre(img, self.boxes[i])
        if self.two_views:
            x = torch.stack([self.transform(img), self.transform(img)])
        else:
            x = self.transform(img)
        return x, int(self.y[i])


def train_transform(size: int, strong: bool = False):
    """Label-preserving augmentation. Hue is barely touched on purpose:
    colour is real evidence for corrosion, so we must not destroy it."""
    aug = [
        T.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.8, 1.25)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply([T.ColorJitter(0.3, 0.3, 0.2, 0.03)], p=0.8),
    ]
    if strong:  # the two-view recipe used for contrastive training
        aug += [T.RandomGrayscale(p=0.2),
                T.RandomApply([T.GaussianBlur(9, sigma=(0.1, 2.0))], p=0.3)]
    return T.Compose(aug + [T.ToTensor(), T.Normalize(MEAN, STD)])


def eval_transform(size: int):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


def balanced_sampler(y: np.ndarray) -> WeightedRandomSampler:
    """Each class is drawn equally often (minority crops repeat within an epoch)."""
    counts = np.bincount(y)
    w = 1.0 / counts[y]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(y), replacement=True)


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).clip(min=1)
    return torch.as_tensor(len(y) / (num_classes * counts), dtype=torch.float)


def make_loader(ds, batch_size, shuffle=False, sampler=None, workers=2, drop_last=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle and sampler is None,
                      sampler=sampler, num_workers=workers, drop_last=drop_last,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


# --------------------------------------------------------------------------- models
def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class Classifier(nn.Module):
    """body -> global average pool -> dropout -> linear.

    `arch="simple"` is the plain CNN (5 conv blocks, trained from scratch).
    `arch="resnet18"/"resnet50"` swap in a torchvision ResNet body.
    Both expose the same three steps so Grad-CAM and PCA work on either."""

    def __init__(self, arch: str, num_classes: int, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.arch = arch
        if arch == "simple":
            chans = [3, 32, 64, 128, 256, 256]
            self.body = nn.Sequential(*[_block(a, b) for a, b in zip(chans[:-1], chans[1:])])
            self.feat_dim = chans[-1]
        elif arch in ("resnet18", "resnet34", "resnet50"):
            net = getattr(models, arch)(weights="DEFAULT" if pretrained else None)
            self.feat_dim = net.fc.in_features
            self.body = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool,
                                      net.layer1, net.layer2, net.layer3, net.layer4)
        else:
            raise ValueError(f"unknown arch {arch}")
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(self.feat_dim, num_classes)

    def embed(self, x):
        return self.pool(self.body(x)).flatten(1)

    def forward(self, x):
        return self.fc(self.drop(self.embed(x)))


def save_checkpoint(path, model: Classifier, classes, img_size, method, extra=None):
    torch.save({"arch": model.arch, "classes": classes, "img_size": img_size, "method": method,
                "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_checkpoint(path, device) -> tuple[Classifier, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = Classifier(ck["arch"], len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


# --------------------------------------------------------------------------- metrics
@torch.no_grad()
def predict(model, loader, device, return_embeddings=False):
    model.eval()
    probs, ys, embs = [], [], []
    for x, y in loader:
        e = model.embed(x.to(device))
        probs.append(torch.softmax(model.fc(e), 1).cpu().numpy())
        ys.append(np.asarray(y))
        if return_embeddings:
            embs.append(e.cpu().numpy())
    probs, ys = np.concatenate(probs), np.concatenate(ys)
    return (probs, ys, np.concatenate(embs)) if return_embeddings else (probs, ys)


def expected_calibration_error(probs, y, bins=10) -> float:
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(ece)


def compute_metrics(y, probs, classes) -> dict:
    pred = probs.argmax(1)
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    present = [i for i in labels if s[i] > 0]
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(np.mean([r[i] for i in present])),
        "macro_f1": float(np.mean([f[i] for i in present])),
        "ece": expected_calibration_error(probs, y),
        "per_class": {classes[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                   "f1": float(f[i]), "support": int(s[i])} for i in labels},
        "confusion": confusion_matrix(y, pred, labels=labels).tolist(),
    }


def macro_f1(y, probs) -> float:
    return float(f1_score(y, probs.argmax(1), average="macro", zero_division=0))


def bal_acc(y, probs) -> float:
    return float(balanced_accuracy_score(y, probs.argmax(1)))


# --------------------------------------------------------------------------- corruptions
# Conditions a helmet camera will meet in a hangar that the training photos may not
# cover. Each takes a PIL image and a severity 1..3. Same idea as ImageNet-C
# (Hendrycks & Dietterich, ICLR 2019), restricted to the plausible ones.
def _np(img):
    return np.asarray(img, dtype=np.float32) / 255.0


def _pil(a):
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))


def c_dark(img, s):
    return ImageEnhance.Brightness(img).enhance([0.7, 0.5, 0.3][s - 1])


def c_bright(img, s):
    return ImageEnhance.Brightness(img).enhance([1.3, 1.6, 2.0][s - 1])


def c_low_contrast(img, s):
    return ImageEnhance.Contrast(img).enhance([0.7, 0.5, 0.3][s - 1])


def c_defocus(img, s):
    return img.filter(ImageFilter.GaussianBlur([1, 2, 4][s - 1]))


def c_motion_blur(img, s):
    return _pil(uniform_filter1d(_np(img), size=[5, 9, 15][s - 1], axis=1, mode="nearest"))


def c_noise(img, s):
    rng = np.random.default_rng(0)
    a = _np(img)
    return _pil(a + rng.normal(0, [0.04, 0.08, 0.15][s - 1], a.shape))


def c_jpeg(img, s):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=[40, 20, 10][s - 1])
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def c_warm_light(img, s):
    g = [0.08, 0.16, 0.28][s - 1]
    return _pil(_np(img) * np.array([1 + g, 1.0, 1 - g], dtype=np.float32))


CORRUPTIONS = {"dark": c_dark, "bright": c_bright, "low_contrast": c_low_contrast,
               "defocus": c_defocus, "motion_blur": c_motion_blur, "sensor_noise": c_noise,
               "jpeg": c_jpeg, "warm_light": c_warm_light}


class corruption:
    """Picklable (a lambda is not, and macOS/Windows DataLoader workers need pickling)."""

    def __init__(self, name, severity):
        self.name, self.severity = name, severity

    def __call__(self, img, box):
        return CORRUPTIONS[self.name](img, self.severity)


# --------------------------------------------------------------------------- shortcut masks
def _box_px(img, box):
    w, h = img.size
    x0, y0, x1, y1 = box
    return int(round(x0 * w)), int(round(y0 * h)), int(round(x1 * w)), int(round(y1 * h))


def hide_defect(img, box):
    """Paint over the annotated defect with the mean colour of its surroundings.
    A model that still names the class has not been looking at the defect."""
    if np.isnan(box).any():
        return img
    a = np.asarray(img).copy()
    x0, y0, x1, y1 = _box_px(img, box)
    outside = np.ones(a.shape[:2], bool)
    outside[y0:y1, x0:x1] = False
    fill = a[outside].mean(0) if outside.any() else a.reshape(-1, 3).mean(0)
    a[y0:y1, x0:x1] = fill
    return Image.fromarray(a)


def hide_context(img, box):
    """Keep only the annotated defect; replace everything around it with the
    crop's mean colour. A model that understands the defect should not care."""
    if np.isnan(box).any():
        return img
    a = np.asarray(img)
    x0, y0, x1, y1 = _box_px(img, box)
    out = np.empty_like(a)
    out[:] = a.reshape(-1, 3).mean(0)
    out[y0:y1, x0:x1] = a[y0:y1, x0:x1]
    return Image.fromarray(out)
