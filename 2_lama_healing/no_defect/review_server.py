"""Review page for the healed crops: click the bad ones, every click is saved to review/decisions.json.

  python review_server.py          then open http://127.0.0.1:8791

Shows each crop as original (LaMa area outlined) | healed. Tabs: "to review" (flagged by make_crops.py), "auto kept",
"dropped" (rules; shown for checking, can be overridden). Decisions: keep / reject; unset = the auto decision.
Python standard library only.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REVIEW = HERE / "review"
THUMBS = HERE / "work/thumbs"            # made by make_crops.py (git-ignored)
DECISIONS = REVIEW / "decisions.json"
PORT = 8791
LOCK = threading.Lock()


def load_decisions():
    return json.loads(DECISIONS.read_text()) if DECISIONS.exists() else {}


def crops():
    df = pd.read_csv(HERE / "crops.csv")
    cols = ["id", "split", "cls", "route", "auto", "why", "detail_ratio", "colour_diff", "seam_ratio"]
    return json.loads(df[cols].to_json(orient="records"))


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
        if self.path == "/api/crops":
            with LOCK:
                body = dict(crops=crops(), decisions=load_decisions())
            return self.send(json.dumps(body).encode(), "application/json")
        if self.path.startswith("/thumbs/"):
            f = THUMBS / Path(self.path).name
            if f.suffix == ".jpg" and f.exists():
                return self.send(f.read_bytes(), "image/jpeg")
        self.send(b"not found", "text/plain", 404)

    def do_POST(self):
        if self.path != "/api/decide":
            return self.send(b"not found", "text/plain", 404)
        upd = json.loads(self.rfile.read(int(self.headers["Content-Length"])))   # {id: "keep" | "reject" | null}
        with LOCK:
            d = load_decisions()
            for k, v in upd.items():
                if v is None:
                    d.pop(k, None)
                elif v in ("keep", "reject"):
                    d[k] = v
            tmp = DECISIONS.with_suffix(".tmp")
            tmp.write_text(json.dumps(d, indent=0, sort_keys=True))
            tmp.replace(DECISIONS)
        self.send(json.dumps(dict(saved=len(d))).encode(), "application/json")

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    print(f"review page: http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
