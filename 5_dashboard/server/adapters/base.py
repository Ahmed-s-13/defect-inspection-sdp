"""What every model adapter offers the server. A new main model needs only a class with this shape."""
from typing import Protocol

import numpy as np
import torch


class ModelAdapter(Protocol):
    classes: list[str]
    uses_tta: bool
    device: torch.device

    @classmethod
    def check(cls, entry) -> str | None:
        """None when the weights are present, else a one-line reason with the fix."""

    def predict(self, x: torch.Tensor, crops: list) -> tuple[np.ndarray, list | None]:
        """x: (N,3,224,224) normalised crops, crops: the same crops as PIL images -> (probabilities (N,C),
        one architecture-A Decision per crop, or None when the model has no decision layer)."""

    def cached(self, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """From the model's own cached features: probabilities and the unfamiliar-input score (used by the report)."""
