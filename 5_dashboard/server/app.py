"""Defect Test Bench API. Serves the registry's active model and the built web app on one port."""
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from server import registry as regmod
from server.paths import EXTRA_SAMPLES, INSPECT_SAMPLES, SAMPLES, WEB_DIST
from server.preprocess import ImageError, crop, default_square, jpeg_data_url, open_image, parse_square, to_batch, \
    view_thumbs
from server.registry import ModelUnavailable
from server.report import ReportUnavailable, crop_path_of, get_report
from server.states import describe

registry = regmod.registry
_infer = threading.Lock()
BATCH_CHUNK = 16


def _safe(fn):
    try:
        fn()
    except Exception as e:  # reported through /api/models; never crash the server
        print(f"load: {e}")


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=lambda: (_safe(registry.active), _safe(lambda: get_report(registry))), daemon=True).start()
    yield


app = FastAPI(title="Defect Test Bench", lifespan=lifespan)


def _unavailable(e: ModelUnavailable):
    return HTTPException(503, {"state": "system_fault", "model": e.model_id, "reason": e.reason})


def _run(images_and_squares):
    """Crop, predict and describe a list of (PIL image, Square). Returns (results, crops)."""
    return _run_crops([crop(img, sq) for img, sq in images_and_squares])


def _run_crops(crops):
    try:
        adapter = registry.active()
    except ModelUnavailable as e:
        raise _unavailable(e)
    x = to_batch(crops)
    with _infer:
        t0 = time.perf_counter()
        probs, decisions = adapter.predict(x, crops)
        ms = (time.perf_counter() - t0) * 1000 / len(crops)
    arch = getattr(adapter, "arch", None)
    out = [describe(adapter.classes, probs[i].tolist(), decisions[i] if decisions else None, arch)
           | {"ms": round(ms, 1)} for i in range(len(crops))]
    return out, crops


@app.get("/api/models")
def models():
    return registry.describe()


class ActiveIn(BaseModel):
    id: str


@app.post("/api/models/active")
def set_active(body: ActiveIn):
    if body.id not in registry.entries:
        raise HTTPException(404, f"unknown model {body.id}")
    registry.set_active(body.id)
    threading.Thread(target=lambda: _safe(registry.active), daemon=True).start()
    return registry.describe()


@app.post("/api/predict")
def predict(file: UploadFile = File(...), square: str | None = Form(None)):
    try:
        img = open_image(file.file.read())
    except ImageError as e:
        raise HTTPException(415, f"{file.filename}: {e}")
    try:
        sq = parse_square(square, *img.size) if square else default_square(*img.size)
    except ValueError as e:
        raise HTTPException(422, str(e))
    (res,), (c,) = _run([(img, sq)])
    return res | {"name": file.filename, "image": {"width": img.width, "height": img.height},
                  "square": sq._asdict(), "crop": jpeg_data_url(c), "views": view_thumbs(c)}


@app.post("/api/batch")
def batch(files: list[UploadFile] = File(...)):
    try:
        registry.active()  # fail before streaming
    except ModelUnavailable as e:
        raise _unavailable(e)
    payload = [(f.filename, f.file.read()) for f in files]  # read now: uploads close when this returns
    for f in files:
        f.file.close()

    def gen():
        for start in range(0, len(payload), BATCH_CHUNK):
            good = []
            for i, (name, data) in enumerate(payload[start:start + BATCH_CHUNK], start):
                try:
                    img = open_image(data)
                    good.append((i, name, crop(img, default_square(*img.size))))  # keep only the 224 px crop
                    del img
                except Exception as e:  # one bad file never ends the stream
                    msg = str(e) if isinstance(e, ImageError) else f"could not read: {e}"
                    yield json.dumps({"index": i, "name": name, "ok": False, "error": msg}) + "\n"
            if good:
                res, crops = _run_crops([c for _, _, c in good])
                for (i, name, _), r, c in zip(good, res, crops):
                    yield json.dumps(r | {"index": i, "name": name, "ok": True,
                                          "thumb": jpeg_data_url(c.resize((160, 160)), 85)}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/api/report")
def report():
    try:
        rep = get_report(registry)
    except ModelUnavailable as e:
        raise _unavailable(e)
    except ReportUnavailable as e:
        raise HTTPException(404, str(e))
    if rep is None:
        raise HTTPException(404, f"{registry.entry().label} has no Model page report; switch to the main model")
    return rep


def _samples(dirs=(SAMPLES, EXTRA_SAMPLES)):
    """Batch's random samples: the repo's 30 defect crops, then the dashboard's 5 no-defect ones (or, with
    dirs=(INSPECT_SAMPLES,), Inspect's one clear crop per class); each with the folder it is in."""
    frames = [pd.read_csv(d / "labels.csv")[["file", "cls"]].assign(dir=str(d))
              for d in dirs if (d / "labels.csv").exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["file", "cls", "dir"])


def _known(s):
    """Only the samples whose class the active model knows (the legacy model has no no-defect class)."""
    try:
        s = s[s.cls.isin(registry.active().classes)]
    except ModelUnavailable:
        pass
    return s[["file", "cls"]].to_dict("records")


@app.get("/api/samples")
def samples():
    return _known(_samples())


@app.get("/api/inspect-samples")
def inspect_samples():
    """One test crop per class that the main model answers clearly, for trying Inspect. Picked for clarity, not
    at random: Batch's samples are the random ones."""
    return _known(_samples((INSPECT_SAMPLES,)))


@app.get("/api/samples/{name}")
def sample(name: str):
    s = _samples((SAMPLES, EXTRA_SAMPLES, INSPECT_SAMPLES))
    hit = s[s.file == name]
    if hit.empty:
        raise HTTPException(404, "no such sample")
    return FileResponse(Path(hit.dir.iloc[0]) / name)


@app.get("/api/crops/{i}")
def crop_image(i: int):
    p = crop_path_of(i, registry.entry().split)
    if p is None or not p.exists():
        raise HTTPException(404, "no such crop")
    return FileResponse(p)


if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
else:
    @app.get("/", response_class=HTMLResponse)
    def no_build():
        return "<p>The web app is not built yet: run <code>./run.sh</code> (or <code>npm run build</code> in web/).</p>"
