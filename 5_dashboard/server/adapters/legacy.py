"""The earlier model (ResNet-18 + MLP head on D1's leaky split), for comparison only. No report, no calibration."""
from server import paths  # noqa: F401
from predict import Legacy


class LegacyAdapter:
    @classmethod
    def check(cls, entry):
        if (entry.weights / "model_best.pt").exists():
            return None
        return (f"missing model_best.pt in {entry.weights}: unzip trained_models.zip "
                "(GitHub Release v1.0-models) in the repo root")

    def __init__(self, entry, device):
        self.m = Legacy(entry.weights, device)
        self.classes = list(self.m.classes)
        self.uses_tta = False
        self.device = device

    def predict(self, x, crops):
        return self.m(x.to(self.device)).cpu().numpy(), None

    def cached(self, F):
        raise NotImplementedError("the legacy model has no feature cache")
