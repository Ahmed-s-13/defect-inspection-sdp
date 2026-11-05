"""Save everything static dash shows that normally comes from the dashboard's server, so the site needs no server.
It asks the real server (5_dashboard, run in-process and only read from) the same questions the pages ask, and writes
the answers as files under web/public/static-data/:

  models.json            the main model's entry (static dash shows only that model)
  report.json            the Model and Uncertainty pages' report
  samples.json           Batch's 35 samples          inspect_samples.json   Inspect's 7 sample photos (static
                         dash's own picks, samples/inspect/ from scripts/pick_inspect.py)
  samples/<file>         the sample photos           crops/<id>.jpg         the crops the report shows
  predictions.json       for every sample: its answer, square, crop and 8 views (as /api/predict gives them, with the
                         images saved under predict/), and its Batch answer and thumbnail (as /api/batch gives them)

Run it again whenever the model, its calibration or the samples change:

    "../.venv/bin/python" export_static_data.py      (from 6_static_dash/, with any Python that runs 5_dashboard)
"""
import base64
import csv
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DASH = HERE.parent / "5_dashboard"
OUT = HERE / "web/public/static-data"
sys.path.insert(0, str(DASH))

from fastapi.testclient import TestClient  # noqa: E402

from server import app as appmod  # noqa: E402
from server.app import _samples  # noqa: E402
from server.paths import EXTRA_SAMPLES, SAMPLES  # noqa: E402

INSPECT = HERE / "samples/inspect"  # static dash's clearer Inspect photos (scripts/pick_inspect.py)


def save_data_url(url, path):
    """A data: URL from the server -> a file; returns its path relative to the site root."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(url.split(",", 1)[1]))
    return str(path.relative_to(OUT.parent))


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    with TestClient(appmod.app) as c:
        get = lambda p: (r := c.get(p), r.raise_for_status(), r.json())[2]  # noqa: E731
        models = get("/api/models")
        main_id = models["main"]
        models["models"] = [m for m in models["models"] if m["id"] == main_id]  # no switching in static dash
        models["active"], models["loaded"] = main_id, True
        report = get("/api/report")
        models["classes"] = report["classes"]
        batch = get("/api/samples")
        inspect = [{"file": r["file"], "cls": r["cls"]} for r in csv.DictReader(open(INSPECT / "labels.csv"))]

        # the photos themselves, from the folders the server reads them from
        all_samples = _samples((SAMPLES, EXTRA_SAMPLES))
        wanted = {s["file"] for s in batch + inspect}
        for row in all_samples.itertuples():
            if row.file in wanted:
                (OUT / "samples").mkdir(exist_ok=True)
                shutil.copy(Path(row.dir) / row.file, OUT / "samples" / row.file)
        for s in inspect:
            shutil.copy(INSPECT / s["file"], OUT / "samples" / s["file"])

        # Inspect: every sample through /api/predict, at the default square (static dash's square is fixed)
        preds = {}
        for name in sorted(wanted):
            r = c.post("/api/predict", files={"file": (name, (OUT / "samples" / name).read_bytes(), "image/jpeg")})
            r.raise_for_status()
            body = r.json()
            stem = Path(name).stem
            body["crop"] = save_data_url(body["crop"], OUT / "predict" / stem / "crop.jpg")
            body["views"] = [save_data_url(v, OUT / "predict" / stem / f"view{k}.jpg") for k, v in enumerate(body["views"])]
            preds[name] = {"predict": body}

        # Batch: the 35 samples through /api/batch, as one run, exactly as the page streams them
        files = [("files", (s["file"], (OUT / "samples" / s["file"]).read_bytes(), "image/jpeg")) for s in batch]
        r = c.post("/api/batch", files=files)
        r.raise_for_status()
        for line in r.text.splitlines():
            if not line.strip():
                continue
            ln = json.loads(line)
            if ln.get("thumb"):
                ln["thumb"] = save_data_url(ln["thumb"], OUT / "predict" / Path(ln["name"]).stem / "thumb.jpg")
            preds.setdefault(ln["name"], {})["batch"] = ln
        # samples only Inspect offers still get a batch line, in case they are opened from Batch later
        for name, p in preds.items():
            if "batch" not in p:
                pr = {k: v for k, v in p["predict"].items() if k not in ("image", "square", "crop", "views", "name")}
                p["batch"] = pr | {"index": 0, "name": name, "ok": True, "thumb": p["predict"]["crop"]}

        # the crops the report links to (its gallery of silent mistakes)
        for e in report["arch_a"].get("silent_errors", []):
            r = c.get(f"/api/crops/{e['id']}")
            r.raise_for_status()
            (OUT / "crops").mkdir(exist_ok=True)
            (OUT / "crops" / f"{e['id']}.jpg").write_bytes(r.content)

    for name, data in (("models", models), ("report", report), ("samples", batch), ("inspect_samples", inspect),
                       ("predictions", preds)):
        (OUT / f"{name}.json").write_text(json.dumps(data))
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"{len(batch)} batch + {len(inspect)} inspect samples, {len(preds)} predictions, "
          f"{len(report['arch_a'].get('silent_errors', []))} report crops -> {OUT} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
