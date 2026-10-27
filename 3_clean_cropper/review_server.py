"""Review page for clean_cropper: label candidates by hand, every click is saved to review/labels_<set>.json.

  python review_server.py          then open http://127.0.0.1:8792

Sets (one tab each, a set appears once its review/<set>.csv exists):
  audit   a random sample of candidates (sample_audit.py), labelled clean / damage / background / other: measures how
          well score.py's scores find background and hidden damage, and sets pick.py's cuts
  final   the crops pick.py picked: keep / reject
Each card: the candidate crop | its whole photo with the square (red) and the labelled defect boxes (yellow).
Python standard library only (+ pandas to read the csvs).
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REVIEW = HERE / "review"
THUMBS = HERE / "work/thumbs"            # made by sample_audit.py / pick.py (git-ignored)
PORT = 8792
LOCK = threading.Lock()
LABELS = {"audit": ("clean", "damage", "background", "other"), "final": ("keep", "reject")}


def labels_file(s):
    return REVIEW / f"labels_{s}.json"


def load(s):
    f = labels_file(s)
    return json.loads(f.read_text()) if f.exists() else {}


def items(s):
    ids = pd.read_csv(REVIEW / f"{s}.csv").id
    sc = pd.read_csv(HERE / "scores.csv").set_index("id").loc[ids].reset_index()
    cols = ["id", "split", "group", "p_skin", "p_damage", "yolo_conf", "yolo_cls", "clip_top"]
    return json.loads(sc[cols].to_json(orient="records"))


class Handler(BaseHTTPRequestHandler):
    def send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self.send((REVIEW / "index.html").read_bytes(), "text/html; charset=utf-8")
        if self.path == "/api/sets":
            sets = [s for s in LABELS if (REVIEW / f"{s}.csv").exists()]
            return self.send(json.dumps(dict(sets=sets, labels=LABELS)).encode(), "application/json")
        if self.path.startswith("/api/items/"):
            s = self.path.rsplit("/", 1)[1]
            if s in LABELS and (REVIEW / f"{s}.csv").exists():
                with LOCK:
                    body = dict(items=items(s), labels=load(s))
                return self.send(json.dumps(body).encode(), "application/json")
        if self.path.startswith("/thumbs/"):
            f = THUMBS / Path(self.path).name
            if f.suffix == ".jpg" and f.exists():
                return self.send(f.read_bytes(), "image/jpeg")
        self.send(b"not found", "text/plain", 404)

    def do_POST(self):
        s = self.path.rsplit("/", 1)[1] if self.path.startswith("/api/label/") else None
        if s not in LABELS:
            return self.send(b"not found", "text/plain", 404)
        upd = json.loads(self.rfile.read(int(self.headers["Content-Length"])))   # {id: label | null}
        with LOCK:
            d = load(s)
            for k, v in upd.items():
                if v is None:
                    d.pop(k, None)
                elif v in LABELS[s]:
                    d[k] = v
            tmp = labels_file(s).with_suffix(".tmp")
            tmp.write_text(json.dumps(d, indent=0, sort_keys=True))
            tmp.replace(labels_file(s))
        self.send(json.dumps(dict(saved=len(d))).encode(), "application/json")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"review page: http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
