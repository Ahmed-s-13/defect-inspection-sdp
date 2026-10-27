"""enhanced_res: the main model. Shared pieces: paths, the grouped split, features, the MLP head, the predictor.

The model is a ResNet-18 backbone (trained by train_backbone.py) followed by an MLP head (build_head.py):
  features  pooled layer3 (256) + layer4 (512) activations = 768 numbers per crop, averaged over the crop's
            8 flips/90-degree rotations (test-time augmentation)
  head      5 MLPs (softmax averaged), blended with a 5-nearest-neighbour vote over the train crops' features,
            then an optional class-prior correction p(c) / prior(c)^tau
Two trained versions, same architecture and file names (git-ignored, GitHub Releases):
  gc7  the main model since 28 Oct 2026: 7 classes (the six defects + no-defect), trained on d1_gc7.
       data/work/enhanced_res/gc7/ (enhanced_res_gc7_model.zip, Release v3.0-gc7)
  v2   the 6-class model on d1_grouped: data/work/enhanced_res/ (enhanced_res_model.zip, Release v2.0-enhanced-res).
       EnhancedRes() with no argument still loads it, so code built on the 6 classes (trt_enhanced_res, the healer's
       screw-head scorer, arch_a/experiments) is unchanged.

  from enhanced_res.model import EnhancedRes, MODELS
  m = EnhancedRes(MODELS["gc7"])          # the main model; EnhancedRes() = v2. Picks CUDA / MPS / CPU
  probs = m(x)                            # x: (N, 3, 224, 224) crops, eval_transform() normalised
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent                                     # 1_main_model_resnet18
if str(MAIN) not in sys.path:
    sys.path.insert(0, str(MAIN))

from klshift.common import build_model, pick_device   # noqa: E402
from klshift.mlp import knn_proba, l2n                 # noqa: E402

WORK = MAIN / "data/work"
EWORK = WORK / "enhanced_res"                          # weights, runs, caches (git-ignored)
GROUPS = MAIN / "d1_grouped/groups.csv"                # the leak-free split (in git)
NO_DEFECT = MAIN / "d1_gc7/no_defect.csv"              # d1_gc7's 7th class (in git; crops in data/work/crops/no_defect/)
RESULTS = HERE / "results"
BACKBONE_FILE, HEAD_FILE = "enhanced_res_backbone.pt", "enhanced_res_head.pt"
MODELS = {"gc7": EWORK / "gc7", "v2": EWORK}           # weights folder of each trained version
MAIN_MODEL = "gc7"


def load_split(data="d1_grouped"):
    """Every crop with its source group and its split (a crop follows its source image).
    d1_grouped: the D1 defect crops. d1_gc7: the same + the no-defect crops (d1_gc7/no_defect.csv), class
    "no-defect"; a "source" column says d1 / healed / clean_cropper."""
    if data == "d1_gc7":
        c = load_split("d1_grouped").assign(source="d1")
        nd = pd.read_csv(NO_DEFECT)
        nd["crop_path"] = [str(MAIN / p) for p in nd.crop_path]
        return pd.concat([c, nd[c.columns]], ignore_index=True)
    assert data == "d1_grouped", data
    f = EWORK / "split.csv"
    if not f.exists():
        c = pd.read_csv(WORK / "crops/D1.csv")
        g = pd.read_csv(GROUPS).rename(columns={"group": "src_group"})
        c["key"] = c.image.str.replace("content/Aircraft_dataset/images/", "", regex=False)
        c = c.merge(g[["key", "src_group", "new_split"]], on="key", how="left", validate="many_to_one")
        assert c.src_group.notna().all(), "crops/D1.csv and d1_grouped/groups.csv do not match"
        c = c.rename(columns={"new_split": "split"})[["crop_path", "cls", "image", "key", "src_group", "split"]]
        f.parent.mkdir(parents=True, exist_ok=True)
        c.to_csv(f, index=False)
    c = pd.read_csv(f)
    c["crop_path"] = [str(MAIN / p) for p in c.crop_path]
    return c


def load_backbone(path, device):
    ck = torch.load(path, map_location="cpu")
    m = build_model(len(ck["classes"]), pretrained=False)
    m.load_state_dict(ck["state"])
    return m.to(device).eval(), ck["classes"]


def feats(m, x):
    """Pooled layer3 (256) and layer4 (512) activations of a torchvision ResNet-18."""
    x = m.maxpool(m.relu(m.bn1(m.conv1(x))))
    l3 = m.layer3(m.layer2(m.layer1(x)))
    l4 = m.layer4(l3)
    return torch.cat([l3.mean((2, 3)), l4.mean((2, 3))], 1)


def dihedral_t(x, k):
    """k = 0..7: the 4 quarter turns, then the same with a horizontal flip."""
    x = torch.rot90(x, k % 4, (2, 3))
    return torch.flip(x, (3,)) if k >= 4 else x


def tta_feats(m, x):
    return torch.stack([feats(m, dihedral_t(x, k)) for k in range(8)]).mean(0)


class Head(nn.Module):
    """train_mlp.py's MLPHead with a configurable input size: standardise -> [Linear, BN, GELU, Dropout] x3."""

    def __init__(self, d_in, n_classes, hidden=(1024, 512, 256), dropout=0.5, in_dropout=0.2):
        super().__init__()
        self.register_buffer("mu", torch.zeros(d_in))
        self.register_buffer("sd", torch.ones(d_in))
        layers, d = [nn.Dropout(in_dropout)], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


class EnhancedRes:
    """The saved main model: backbone + MLP head. Call it on a batch of normalised 224x224 crops."""

    def __init__(self, work=EWORK, device=None):
        work = Path(work)
        for f in (work / BACKBONE_FILE, work / HEAD_FILE):
            if not f.exists():
                zipname = "enhanced_res_gc7_model.zip (Release v3.0-gc7)" if work.name == "gc7" else \
                    "enhanced_res_model.zip (Release v2.0-enhanced-res)"
                raise SystemExit(f"missing {f}: unzip {zipname} in the repo root, or train it (enhanced_res/README.md)")
        self.device = device or pick_device()
        ck = torch.load(work / HEAD_FILE, map_location="cpu")
        self.backbone, classes = load_backbone(work / BACKBONE_FILE, self.device)
        self.classes = ck["classes"]
        assert classes == self.classes, "class order of the backbone and the head differ"
        self.heads = []
        for st in ck["heads"]:
            h = Head(ck["d_in"], len(self.classes))
            h.load_state_dict(st)
            self.heads.append(h.to(self.device).eval())
        self.cols = slice(*ck["cols"])
        self.tta, self.alpha, self.tau = ck["tta"], ck["alpha"], ck["tau"]
        self.prior = ck["prior"].float().to(self.device)
        if self.alpha > 0:
            self.bank = l2n(ck["knn_bank"].float().to(self.device))
            self.bank_y = ck["knn_labels"].long().to(self.device)

    @torch.no_grad()
    def features(self, x):
        x = x.to(self.device)
        f = tta_feats(self.backbone, x) if self.tta else feats(self.backbone, x)
        return f[:, self.cols]

    @torch.no_grad()
    def proba_from_features(self, f):
        p = torch.stack([h(f).softmax(1) for h in self.heads]).mean(0)
        if self.alpha > 0:
            p = (1 - self.alpha) * p + self.alpha * knn_proba(self.bank, self.bank_y, l2n(f), 5, len(self.classes))
        if self.tau > 0:
            p = p / self.prior ** self.tau
            p = p / p.sum(1, keepdim=True)
        return p

    @torch.no_grad()
    def knn_distance(self, f, k=5):
        """1 - mean cosine similarity to the k nearest train features: architecture A's unfamiliar-input score."""
        return 1 - (l2n(f) @ self.bank.T).topk(k, dim=1).values.mean(1)

    @torch.no_grad()
    def __call__(self, x):
        return self.proba_from_features(self.features(x))


def group_bootstrap_ci(y, p, groups, n=2000, seed=0):
    """95% interval of macro-F1, resampling whole source groups (crops of one scene move together)."""
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    vals = [f1_score(y[ii], p[ii], average="macro")
            for ii in (np.concatenate([idx[g] for g in rng.choice(ug, len(ug))]) for _ in range(n))]
    return np.quantile(vals, [0.025, 0.975])
