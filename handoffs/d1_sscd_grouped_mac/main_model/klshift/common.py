"""Shared code: paths, file-name handling, cropping, transforms, model, dataset."""
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Keep downloaded ImageNet weights in this workspace.
torch.hub.set_dir(str(Path(__file__).resolve().parents[2] / "models/torch"))
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

DATASETS = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10"]  # raw downloads
DEFECT = ["D1", "D2", "D3", "D4", "D5"]                             # defect datasets
NO_DAMAGE = ["D6", "D7", "D8", "D9"]                                # whole-aircraft, no-damage datasets
# Derived crop sets cut from a raw dataset in a different way (see prepare.py):
#   D6p-D9p  skin patches from inside the aircraft box;  D1nd  D1 areas away from every defect box
DERIVED = {"D6p": ("D6", "patch"), "D7p": ("D7", "patch"), "D8p": ("D8", "patch"), "D9p": ("D9", "patch"),
           "D1nd": ("D1", "background"),
           "D1nc": ("D1", "cropper")}  # made by run_cropper.py (the Cropper), not prepare.py
# Subsets of a raw dataset's crops by class (no new crops). D10 = fuselage mock-up video frames.
CLASS_SUBSETS = {"D10n": ("D10", ["normal"]), "D10a": ("D10", ["anomaly-big", "anomaly-small"])}
ALL_SETS = DATASETS + list(DERIVED) + list(CLASS_SUBSETS)
VIDEO = ["D10"]  # video frames: near-copies are found without chaining (see embed.py)
SEPARATE = ["D10", "D10n", "D10a", "D1nc"]  # analysed in their own shift.py run (--sets), not in the main tables


def source_of(ds):
    """Raw dataset a crop set was cut from."""
    if ds in CLASS_SUBSETS:
        return CLASS_SUBSETS[ds][0]
    return DERIVED[ds][0] if ds in DERIVED else ds
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CROP_SIZE = 224
MARGIN = 0.25  # added on each side of the box, as a fraction of box width/height
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet

# Class map used only in the analysis (per-class KL). None = no D1 counterpart.
CLASS_MAP = {
    "D2": {"crack": "Crack", "dent": "Dent", "corrosion": "Corrosion"},
    "D3": {"crack": "Crack", "dent": "Dent", "corrosion": "Corrosion"},
    "D4": {"cracking": "Crack", "surface corrosion": "Corrosion",
           "leading edge corrosion": "Corrosion", "paint peeling": "Paint-Peel-Off"},
    "D5": {"dent": "Dent", "fastener damage": "Missing-Head", "rupture": None},
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------- file names

_RF_SUFFIX = re.compile(r"\.rf\.[0-9a-fA-F]+$")       # Roboflow: name_jpg.rf.<hash>
_EXT_SUFFIX = re.compile(r"(_(jpe?g|png|bmp|webp))+$", re.I)
_TILE_SUFFIX = re.compile(r"_\d{1,2}_\d{1,2}$")         # tiles: DJI_0688_04_02


def original_stem(filename):
    """Undo Roboflow renaming: 'IMG_1_jpg.rf.ab12.jpg' -> 'IMG_1'."""
    stem = Path(filename).stem
    stem = _RF_SUFFIX.sub("", stem)
    return _EXT_SUFFIX.sub("", stem)


def photo_stem(filename):
    """Stem shared by all tiles of one photo: 'DJI_0688_04_02' -> 'DJI_0688'."""
    return _TILE_SUFFIX.sub("", original_stem(filename))


def distinctive_stem(stem):
    """Generic names like '0001' or 'img12' would match by accident across datasets."""
    return len(stem) >= 10 and not stem.isdigit()


def class_key(name):
    return name.strip().lower().replace("_", " ").replace("-", " ")


class UnionFind:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


# ---------------------------------------------------------------- cropping

def square_crop_box(x1, y1, x2, y2, img_w, img_h, margin=MARGIN):
    """Square box around (x1,y1,x2,y2) plus margin, clamped to the image by
    shifting inward (never padding). Returns integer (left, top, side)."""
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh) * (1 + 2 * margin)
    side = int(round(min(side, img_w, img_h)))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    left = int(round(cx - side / 2))
    top = int(round(cy - side / 2))
    left = min(max(left, 0), img_w - side)
    top = min(max(top, 0), img_h - side)
    return left, top, side


# ---------------------------------------------------------------- model / data

def eval_transform():
    # Crops are already CROP_SIZE x CROP_SIZE; identical for every dataset.
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def train_transform():
    return T.Compose([
        T.RandomResizedCrop(CROP_SIZE, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(10),
        T.ColorJitter(0.2, 0.2, 0.2, 0.02),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def build_model(num_classes, pretrained=True):
    """Stock torchvision ResNet-18; only fc is replaced."""
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def embed_batch(model, x):
    """512-d global-average-pool output, i.e. the input to fc."""
    m = model
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    x = m.layer4(m.layer3(m.layer2(m.layer1(x))))
    return torch.flatten(m.avgpool(x), 1)


class CropDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None, pil_fn=None):
        self.paths = list(paths)
        self.labels = None if labels is None else list(labels)
        self.transform = transform or eval_transform()
        self.pil_fn = pil_fn  # optional PIL -> PIL corruption applied before transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.pil_fn is not None:
            img = self.pil_fn(img)
        y = -1 if self.labels is None else self.labels[i]
        return self.transform(img), y


def load_crops(work, ds):
    return pd.read_csv(Path(work) / "crops" / f"{ds}.csv")
