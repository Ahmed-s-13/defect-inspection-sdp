import io
import json

import numpy as np
from PIL import Image

from server.paths import SAMPLES


def _jpg(color="grey", size=(120, 90)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def test_models(client):
    d = client.get("/api/models").json()
    assert d["main"] == "enhanced_res" and d["active"] == "enhanced_res"


def test_models_endpoint_reports_unavailable(client, tmp_path, monkeypatch):
    from server import app as appmod
    from server.registry import Registry
    y = tmp_path / "models.yaml"
    y.write_text("main: m\nmodels:\n  m:\n    label: m\n    adapter: enhanced_res\n    weights: ./nowhere\n")
    monkeypatch.setattr(appmod, "registry", Registry(y))
    d = client.get("/api/models").json()
    assert d["models"][0]["available"] is False and "unzip" in d["models"][0]["reason"]
    r = client.post("/api/predict", files={"file": ("x.jpg", _jpg(), "image/jpeg")})
    assert r.status_code == 503 and r.json()["detail"]["state"] == "system_fault"


def test_predict_matches_predict_py(client):
    from enhanced_res.arch_a.runtime import temper
    from server.states import STATE
    from klshift.common import eval_transform
    from predict import to_crop
    from server import app as appmod
    adapter = appmod.registry.active()
    for name in ["crack_01.jpg", "dent_02.jpg", "scratch_03.jpg"]:
        p = SAMPLES / name
        c = to_crop(p)
        ref, (dec,) = adapter.predict(eval_transform()(c).unsqueeze(0), [c])
        r = client.post("/api/predict", files={"file": (name, p.read_bytes(), "image/jpeg")})
        assert r.status_code == 200
        body = r.json()
        assert np.allclose(body["probs"], temper(ref, adapter.arch.T)[0], atol=1e-5)   # the calibrated %
        assert body["state"] == STATE[dec.action] and body["answer"] == dec.classes       # same answer as predict.py
        assert len(body["views"]) == 8 and body["square"]["side"] > 0


def test_predict_square_is_used_and_clamped(client):
    p = SAMPLES / "crack_01.jpg"
    r = client.post("/api/predict", files={"file": ("c.jpg", p.read_bytes(), "image/jpeg")},
                    data={"square": "-50,-50,100"})
    assert r.json()["square"] == {"left": 0, "top": 0, "side": 100}


def test_predict_rejects_non_image(client):
    r = client.post("/api/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415 and "notes.txt" in r.json()["detail"]


def test_predict_rejects_malformed_square(client):
    r = client.post("/api/predict", files={"file": ("x.jpg", _jpg(), "image/jpeg")}, data={"square": "1,2"})
    assert r.status_code == 422


def test_batch_stream_survives_bad_file(client):
    files = [("files", ("a.jpg", _jpg("red"), "image/jpeg")),
             ("files", ("b.heic", b"\x00\x01garbage", "image/heic")),
             ("files", ("c.jpg", _jpg("blue"), "image/jpeg"))]
    r = client.post("/api/batch", files=files)
    lines = [json.loads(l) for l in r.text.strip().splitlines()]
    assert [l["index"] for l in sorted(lines, key=lambda l: l["index"])] == [0, 1, 2]
    by = {l["name"]: l for l in lines}
    assert by["a.jpg"]["ok"] and by["c.jpg"]["ok"] and not by["b.heic"]["ok"]
    assert by["a.jpg"]["thumb"].startswith("data:image/jpeg")


def test_samples_and_safe_paths(client):
    s = client.get("/api/samples").json()
    assert len(s) == 35 and {"file", "cls"} <= set(s[0])        # the repo's 30 defect samples + 5 no-defect
    assert sum(x["cls"] == "no-defect" for x in s) == 5
    assert client.get(f"/api/samples/{s[0]['file']}").status_code == 200
    assert client.get("/api/samples/..%2Fpredict.py").status_code == 404
    assert client.get("/api/crops/-1").status_code == 404
    assert client.get("/api/crops/999999999").status_code == 404


def test_report_endpoint(client):
    rep = client.get("/api/report").json()
    assert rep["arch_a"]["alpha"] == 0.1 and len(rep["classes"]) == 7
    first_error = rep["arch_a"]["silent_errors"][0]["id"]
    assert client.get(f"/api/crops/{first_error}").status_code == 200


def test_batch_stream_survives_decompression_bomb(client):
    from tests.test_preprocess import _png_header_only
    files = [("files", ("a.jpg", _jpg("red"), "image/jpeg")),
             ("files", ("huge.png", _png_header_only(40000, 40000), "image/png")),
             ("files", ("c.jpg", _jpg("blue"), "image/jpeg"))]
    lines = [json.loads(l) for l in client.post("/api/batch", files=files).text.strip().splitlines()]
    by = {l["name"]: l for l in lines}
    assert by["a.jpg"]["ok"] and by["c.jpg"]["ok"] and not by["huge.png"]["ok"]
