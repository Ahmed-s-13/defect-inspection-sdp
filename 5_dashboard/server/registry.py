"""The model registry: models.yaml names every servable model and which one is main."""
import threading
from dataclasses import dataclass
from pathlib import Path

import yaml

from server.paths import DASH
from klshift.common import pick_device


class ModelUnavailable(Exception):
    def __init__(self, model_id, reason):
        super().__init__(reason)
        self.model_id, self.reason = model_id, reason


@dataclass(frozen=True)
class ModelEntry:
    id: str
    label: str
    adapter: str
    description: str
    weights: Path
    trained_on: str
    leak_free: bool
    features_cache: Path | None
    summary: Path | None
    calibration: Path | None   # architecture A's fitted numbers (None: the repo's default, calibration.json)
    split: str                 # what the report is computed on: d1_grouped or d1_gc7

    @classmethod
    def from_yaml(cls, mid, d, base):
        rel = lambda k: (base / d[k]).resolve() if d.get(k) else None  # noqa: E731
        return cls(mid, d.get("label", mid), d["adapter"], d.get("description", ""), rel("weights"),
                   d.get("trained_on", ""), bool(d.get("leak_free", False)), rel("features_cache"), rel("summary"),
                   rel("calibration"), d.get("split", "d1_grouped"))


class Registry:
    def __init__(self, path=DASH / "models.yaml", device=None):
        from server.adapters import ADAPTERS
        self._adapters = ADAPTERS
        path = Path(path)
        raw = yaml.safe_load(path.read_text())
        self.entries = {k: ModelEntry.from_yaml(k, v, path.parent) for k, v in raw["models"].items()}
        for e in self.entries.values():
            if e.adapter not in ADAPTERS:
                raise ValueError(f"model {e.id}: unknown adapter {e.adapter!r} (known: {', '.join(ADAPTERS)})")
        self.main = raw["main"]
        if self.main not in self.entries:
            raise ValueError(f"main: {self.main!r} is not one of the models in {path}")
        self.active_id = self.main
        self.device = device or pick_device()
        self._loaded = {}
        self._failed = {}  # model id -> reason the weights failed to load
        self._lock = threading.Lock()

    def entry(self, mid=None):
        return self.entries[mid or self.active_id]

    def availability(self, mid):
        if mid in self._failed:
            return False, self._failed[mid]
        e = self.entries[mid]
        reason = self._adapters[e.adapter].check(e)
        return reason is None, reason

    def is_loaded(self, mid):
        return mid in self._loaded

    def get(self, mid=None):
        mid = mid or self.active_id
        with self._lock:
            if mid not in self._loaded:
                ok, reason = self.availability(mid)
                if not ok:
                    raise ModelUnavailable(mid, reason)
                e = self.entries[mid]
                try:
                    self._loaded[mid] = self._adapters[e.adapter](e, self.device)
                except BaseException as err:  # truncated .pt, git-lfs pointer, torch mismatch, SystemExit from the repo loader
                    self._failed[mid] = f"failed to load {e.label} from {e.weights}: {err}"
                    raise ModelUnavailable(mid, self._failed[mid]) from err
            return self._loaded[mid]

    def active(self):
        return self.get(self.active_id)

    def set_active(self, mid):
        if mid not in self.entries:
            raise KeyError(mid)
        self.active_id = mid

    def describe(self):
        models = []
        for mid, e in self.entries.items():
            ok, reason = self.availability(mid)
            models.append({"id": mid, "label": e.label, "description": e.description, "trained_on": e.trained_on,
                           "leak_free": e.leak_free, "available": ok, "reason": reason, "is_main": mid == self.main})
        loaded = self.is_loaded(self.active_id)
        return {"main": self.main, "active": self.active_id, "device": str(self.device), "loaded": loaded,
                "classes": self._loaded[self.active_id].classes if loaded else None, "models": models}


registry = Registry()
