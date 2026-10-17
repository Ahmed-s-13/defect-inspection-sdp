"""The Cropper: a standard way to cut "no damage" patches out of defect photos.

Every photo goes through the same four steps, driven by one config (cropper.yaml):

  1. propose   random squares, each sized like one of the photo's own defect crops (same zoom as the
               defect class, so zoom level can't give the class away)
  2. exclude   drop any square touching a labelled defect box enlarged by `margin` on each side,
               or touching a square already proposed
  3. surface   CLIP zero-shot: keep squares that look like aircraft skin, not sky, grass or ground.
               The threshold is calibrated on the dataset's own defect crops (always aircraft skin):
               it is set so that `calibrate_keep` of them would pass
  4. damage    CLIP zero-shot score for "damaged metal" vs "clean metal"; recorded for review and
               only used as a filter if `damage_threshold` is set (it is a weak signal)

At most `keep_per_image` squares per photo are kept, highest surface score first.
"""
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import yaml

from klshift.common import square_crop_box


@dataclass
class CropperConfig:
    candidates_per_image: int = 8
    keep_per_image: int = 2
    margin: float = 0.25
    min_side: int = 32
    max_tries: int = 200
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    surface_prompts: list = field(default_factory=lambda: [
        "a close-up photo of an aircraft's metal skin",
        "a close-up photo of a painted metal panel",
        "a close-up photo of an airplane fuselage surface with rivets",
        "a close-up photo of an aircraft wing surface",
    ])
    other_prompts: list = field(default_factory=lambda: [
        "a photo of grass", "a photo of the sky", "a photo of trees and leaves", "a photo of the ground",
        "a photo of a concrete floor", "a photo of asphalt", "a photo of a person", "a photo of a building",
    ])
    surface_threshold: float = 0.5   # used only when calibrate_keep is None
    calibrate_keep: float = 0.90     # share of the dataset's defect crops the surface check must keep
    damage_prompts: list = field(default_factory=lambda: [
        "a close-up photo of a crack in metal", "a close-up photo of a dent in a metal panel",
        "a close-up photo of corrosion on metal", "a close-up photo of a scratch on metal",
        "a close-up photo of peeling paint", "a close-up photo of a missing rivet",
    ])
    clean_prompts: list = field(default_factory=lambda: [
        "a close-up photo of clean undamaged metal", "a close-up photo of a smooth intact painted panel",
    ])
    damage_threshold: float = None  # None = record only
    seed: int = 0

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            return cls(**(yaml.safe_load(f) or {}))

    def to_dict(self):
        return asdict(self)


class Cropper:
    def __init__(self, cfg: CropperConfig, device="cpu"):
        self.cfg, self.device = cfg, device
        self.model = None

    # ---------------------------------------------------------------- steps 1 + 2

    def propose(self, img_w, img_h, boxes, rng):
        """boxes: (x1, y1, x2, y2) defect boxes in pixels. Returns [(left, top, side)] that touch no
        keep-out zone and no other proposal. Sizes come from this photo's own defect crops."""
        c = self.cfg
        if not boxes:
            return []
        keep_out = [(x1 - c.margin * (x2 - x1), y1 - c.margin * (y2 - y1),
                     x2 + c.margin * (x2 - x1), y2 + c.margin * (y2 - y1)) for x1, y1, x2, y2 in boxes]
        sides = [square_crop_box(*b, img_w, img_h)[2] for b in boxes]
        out, taken = [], []
        for _ in range(c.max_tries):
            if len(out) == c.candidates_per_image:
                break
            side = int(min(rng.choice(sides), img_w, img_h))
            if side < c.min_side:
                continue
            left, top = int(rng.integers(0, img_w - side + 1)), int(rng.integers(0, img_h - side + 1))
            r = (left, top, left + side, top + side)
            if any(r[0] < b[2] and b[0] < r[2] and r[1] < b[3] and b[1] < r[3] for b in keep_out + taken):
                continue
            out.append((left, top, side))
            taken.append(r)
        return out

    # ---------------------------------------------------------------- steps 3 + 4

    def _load(self):
        import open_clip
        c = self.cfg
        model, _, _ = open_clip.create_model_and_transforms(c.clip_model, pretrained=c.clip_pretrained)
        self.model = model.eval().to(self.device)
        tok = open_clip.get_tokenizer(c.clip_model)
        with torch.no_grad():
            def enc(prompts):
                t = self.model.encode_text(tok(prompts).to(self.device)).float()
                return t / t.norm(dim=-1, keepdim=True)
            self.t_surface = enc(c.surface_prompts + c.other_prompts)
            self.t_damage = enc(c.damage_prompts + c.clean_prompts)
        self.mean = torch.tensor(open_clip.OPENAI_DATASET_MEAN, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(open_clip.OPENAI_DATASET_STD, device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def score(self, patches):
        """patches: uint8 array (n, 224, 224, 3). Returns (surface_prob, damage_prob), each (n,).
        Each is the softmax mass on its prompt group (CLIP logit scale 100)."""
        if self.model is None:
            self._load()
        x = torch.from_numpy(np.ascontiguousarray(patches)).to(self.device).permute(0, 3, 1, 2).float() / 255
        f = self.model.encode_image((x - self.mean) / self.std).float()
        f = f / f.norm(dim=-1, keepdim=True)
        ns, nd = len(self.cfg.surface_prompts), len(self.cfg.damage_prompts)
        ps = (100 * f @ self.t_surface.T).softmax(-1)[:, :ns].sum(-1)
        pd_ = (100 * f @ self.t_damage.T).softmax(-1)[:, :nd].sum(-1)
        return ps.cpu().numpy(), pd_.cpu().numpy()

    def select(self, surface, damage):
        """Indices of candidates to keep, best surface score first; also a reason per candidate."""
        c = self.cfg
        reason = np.where(surface < c.surface_threshold, "not_surface", "ok").astype(object)
        if c.damage_threshold is not None:
            reason[(reason == "ok") & (damage >= c.damage_threshold)] = "maybe_damage"
        ok = np.flatnonzero(reason == "ok")
        keep = ok[np.argsort(-surface[ok])][:c.keep_per_image]
        reason[np.setdiff1d(ok, keep)] = "over_limit"
        reason[keep] = "accepted"
        return keep, reason
