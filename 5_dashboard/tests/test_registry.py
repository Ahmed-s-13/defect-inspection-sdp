import pytest

from server.registry import ModelUnavailable, Registry


def test_main_resolves_to_enhanced_res():
    r = Registry()
    assert r.main == "enhanced_res"
    assert r.active_id == "enhanced_res"
    assert r.availability("enhanced_res") == (True, None)


def test_missing_weights_reported(tmp_path):
    y = tmp_path / "models.yaml"
    y.write_text("main: m\nmodels:\n  m:\n    label: m\n    adapter: enhanced_res\n    weights: ./nowhere\n")
    r = Registry(y)
    ok, reason = r.availability("m")
    assert not ok
    assert "unzip enhanced_res_model.zip" in reason
    with pytest.raises(ModelUnavailable):
        r.active()


def test_unknown_adapter_rejected(tmp_path):
    y = tmp_path / "models.yaml"
    y.write_text("main: m\nmodels:\n  m:\n    label: m\n    adapter: nope\n    weights: .\n")
    with pytest.raises(ValueError, match="unknown adapter"):
        Registry(y)


def test_main_must_exist(tmp_path):
    y = tmp_path / "models.yaml"
    y.write_text("main: x\nmodels:\n  m:\n    label: m\n    adapter: legacy\n    weights: .\n")
    with pytest.raises(ValueError, match="main"):
        Registry(y)


def test_describe_lists_models_and_loads_lazily():
    r = Registry()
    d = r.describe()
    assert d["main"] == "enhanced_res" and d["active"] == "enhanced_res"
    assert {m["id"] for m in d["models"]} == {"enhanced_res", "legacy"}
    assert d["loaded"] is False and d["classes"] is None
    a = r.active()
    assert a.classes == ["corrosion", "crack", "dent", "missing-head", "paint-peel-off", "scratch", "no-defect"]
    assert r.describe()["loaded"] is True


def test_load_failure_becomes_unavailable_with_reason(tmp_path, monkeypatch):
    from server.adapters import ADAPTERS

    class Broken:
        @classmethod
        def check(cls, entry):
            return None

        def __init__(self, entry, device):
            raise RuntimeError("PytorchStreamReader failed reading zip archive")

    monkeypatch.setitem(ADAPTERS, "broken", Broken)
    y = tmp_path / "models.yaml"
    y.write_text("main: m\nmodels:\n  m:\n    label: m\n    adapter: broken\n    weights: .\n")
    r = Registry(y)
    with pytest.raises(ModelUnavailable, match="failed to load"):
        r.active()
    ok, reason = r.availability("m")
    assert not ok and "PytorchStreamReader" in reason
