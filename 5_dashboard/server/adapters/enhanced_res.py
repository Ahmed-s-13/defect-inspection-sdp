"""The main model: enhanced_res (ResNet-18 features over 8 views -> MLP ensemble + kNN head), answered through
architecture B (enhanced_res/arch_a with calibration_gc7.json: the unfamiliar-input gate (kNN distance + AoN),
temperature and a conformal set; the quality gate is off). The entry names the weights folder and its calibration."""
import numpy as np
import torch

from server import paths  # noqa: F401  (puts the repo on sys.path)
from enhanced_res.arch_a.runtime import CALIBRATION, ArchA
from enhanced_res.model import BACKBONE_FILE, HEAD_FILE, EnhancedRes


class EnhancedResAdapter:
    @classmethod
    def check(cls, entry):
        missing = [f for f in (BACKBONE_FILE, HEAD_FILE) if not (entry.weights / f).exists()]
        if missing:
            zipname = "enhanced_res_gc7_model.zip (GitHub Release v3.0-gc7)" if entry.weights.name == "gc7" else \
                "enhanced_res_model.zip (GitHub Release v2.0-enhanced-res)"
            return f"missing {', '.join(missing)} in {entry.weights}: unzip {zipname} in the repo root"
        cal = entry.calibration or CALIBRATION
        if not cal.exists():
            return f"missing {cal}: pull the Project Code repo (the calibration file is in git)"
        return None

    def __init__(self, entry, device):
        self.m = EnhancedRes(entry.weights, device)
        self.calibration = entry.calibration or CALIBRATION
        self.arch = ArchA(self.calibration)
        assert self.arch.classes == list(self.m.classes), f"{self.calibration.name} was made for another class order"
        self.classes = list(self.m.classes)
        self.uses_tta = bool(self.m.tta)
        self.device = device

    @torch.no_grad()
    def predict(self, x, crops):
        """x: normalised crops, crops: the same 224 px crops as PIL images (used only if the quality gate is on).
        Returns the model's probabilities (N, C) and the decision layer's answer for each crop."""
        f = self.m.features(x)
        p = self.m.proba_from_features(f).cpu().numpy()
        return p, self.arch.decide(p, crops=crops, knn_dist=self.m.knn_distance(f).cpu().numpy(),
                                   feats=f.cpu().numpy())

    @torch.no_grad()
    def raw_features(self, paths, batch=64):
        """8-view features of saved 224 px crops, all 768 columns, as in the team's feature cache (build_head.extract)."""
        from torch.utils.data import DataLoader
        from enhanced_res.model import tta_feats
        from klshift.common import CropDataset
        return np.concatenate([tta_feats(self.m.backbone, x.to(self.device)).cpu().numpy()
                               for x, _ in DataLoader(CropDataset(list(paths)), batch_size=batch, num_workers=0)])

    @torch.no_grad()
    def cached(self, F, chunk=4096):
        """From the model's own cached features: probabilities, the kNN distance and the model's features (for AoN)."""
        probs, dists, feats = [], [], []
        for i in range(0, len(F), chunk):
            f = torch.from_numpy(np.ascontiguousarray(F[i:i + chunk])).float().to(self.device)[:, self.m.cols]
            probs.append(self.m.proba_from_features(f).cpu().numpy())
            dists.append(self.m.knn_distance(f).cpu().numpy())
            feats.append(f.cpu().numpy())
        return np.concatenate(probs), np.concatenate(dists), np.concatenate(feats)
