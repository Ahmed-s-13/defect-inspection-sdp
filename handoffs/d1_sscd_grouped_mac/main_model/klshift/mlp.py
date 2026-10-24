"""The MLP head: a better last step for the trained ResNet-18 (see train_mlp.py).

The ResNet-18 turns a crop into a 512-d embedding; its own fc layer turns that into 6 class scores.
The MLP head replaces only that last step: an ensemble of small MLPs, averaged with a k-nearest-
neighbour vote against the D1-train embeddings. The ResNet-18 itself is unchanged.
"""
from pathlib import Path
import hashlib

import numpy as np
import torch
from torch import nn

EMB_DIM = 512
HEAD_FILE = "mlp_head.pt"  # saved next to model_best.pt in <work>


def sync(device):
    """Wait for queued GPU work (for honest timings)."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def load_d1_embeddings(path):
    """D1 rows only of embeddings.npz (D2-D10, D1nd, D1nc rows are skipped).

    Returns X (N, 512) float32, a dict of columns (split, cls, crop_path, image, group) and the sorted
    class list, which is the order the ResNet-18 checkpoint uses.
    """
    if not Path(path).exists():
        raise SystemExit(f"missing {path}: unzip main_model_data_work.zip in the repo root, or run embed.py")
    z = np.load(path, allow_pickle=False)
    d1 = z["dataset"] == "D1"
    meta = {k: z[k][d1] for k in ["split", "cls", "crop_path", "image", "group"]}
    return z["emb"][d1].astype(np.float32), meta, sorted(set(meta["cls"].tolist()))


class MLPHead(nn.Module):
    """512-d embedding -> standardise -> [Linear, BatchNorm, GELU, Dropout] x len(hidden) -> class logits.
    The standardisation (train mean / std) is stored in the module, so it takes raw embeddings."""

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


class Ensemble(nn.Module):
    """Average of the softmax outputs of several MLP heads (different seeds)."""

    def __init__(self, heads):
        super().__init__()
        self.heads = nn.ModuleList(heads)

    def forward(self, emb):
        return torch.stack([h(emb).softmax(1) for h in self.heads]).mean(0)


def l2n(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def knn_proba(bank, bank_y, q, k, n_classes, chunk=2048):
    """Cosine kNN with 1/distance weights, run on whatever device the tensors are on.
    bank (N, 512) and q (M, 512) must be L2-normalised; bank_y (N,) long."""
    out = []
    for i in range(0, len(q), chunk):
        sim, idx = (q[i:i + chunk] @ bank.T).topk(k, dim=1)
        w = 1.0 / (1.0 - sim).clamp_min(1e-6)
        p = torch.zeros(len(sim), n_classes, device=q.device)
        p.scatter_add_(1, bank_y[idx], w)
        out.append(p / p.sum(1, keepdim=True))
    return torch.cat(out)


class MLPClassifier:
    """The saved MLP head (<work>/mlp_head.pt): MLP ensemble, blended with kNN over the D1-train embeddings.
    Takes raw 512-d ResNet-18 embeddings, returns class probabilities. Everything runs on `device`."""

    def __init__(self, path, device):
        if not Path(path).exists():
            raise SystemExit(f"missing {path}: run `python train_mlp.py` (about 1 minute)")
        ck = torch.load(path, map_location="cpu")
        provenance = ck.get("provenance")
        if provenance is None:
            raise SystemExit("MLP checkpoint lacks grouped-training provenance. Retrain with main_model/train_mlp.py.")
        for key, name in [("split_sha256", "d1_split.csv"), ("checkpoint_sha256", "model_best.pt")]:
            current = Path(path).parent / name
            if not current.exists() or hashlib.sha256(current.read_bytes()).hexdigest() != provenance[key]:
                raise SystemExit("Stale MLP checkpoint: grouped split or backbone changed. Rebuild embeddings and MLP.")

        heads = []
        for state in ck["states"]:
            h = MLPHead(len(ck["classes"]), **ck["arch"])
            h.load_state_dict(state)
            heads.append(h)
        self.ens = Ensemble(heads).to(device).eval()
        self.classes, self.device = ck["classes"], device
        self.k, self.w = ck["knn_k"], ck["knn_weight"]
        if self.w > 0:
            self.bank = l2n(ck["knn_bank"].float().to(device))
            self.bank_y = ck["knn_labels"].long().to(device)

    @torch.no_grad()
    def __call__(self, emb):
        emb = torch.as_tensor(emb, dtype=torch.float32, device=self.device)
        p = self.ens(emb)
        if self.w > 0:
            p = (1 - self.w) * p + self.w * knn_proba(self.bank, self.bank_y, l2n(emb), self.k, len(self.classes))
        return p
