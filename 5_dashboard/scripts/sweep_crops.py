"""Slide the model's 224 px square over one photo and record architecture A's answer at every position.
Feeds the "Where the square sits" scene on the Model page (web/public/sweep/).

    python scripts/sweep_crops.py
"""
import json
import sys
from pathlib import Path

DASH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DASH))

from PIL import Image  # noqa: E402

from server.preprocess import Square, crop, to_batch  # noqa: E402
from server.registry import registry  # noqa: E402
from server.states import describe  # noqa: E402

D1 = DASH.parent / "1_main_model_resnet18/data/raw/D1/content/Aircraft_dataset"   # D1_raw.zip
PHOTO = D1 / "images/test/aircraft_000139.jpg"           # in d1_grouped's train split
LABEL = D1 / "labels/test/aircraft_000139.txt"
OUT = DASH / "web/public/sweep"
SIDE, STRIDE, CHUNK = 224, 16, 64
STATES = ["detected", "ambiguous", "unclear", "retake", "cannot_assess", "clear"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    img = Image.open(PHOTO).convert("RGB")
    w, h = img.size
    xs, ys = range(0, w - SIDE + 1, STRIDE), range(0, h - SIDE + 1, STRIDE)
    squares = [Square(x, y, SIDE) for y in ys for x in xs]
    m = registry.active()
    cells = []
    for i in range(0, len(squares), CHUNK):
        crops = [crop(img, sq) for sq in squares[i:i + CHUNK]]
        probs, decisions = m.predict(to_batch(crops), crops)
        for p, d in zip(probs, decisions):
            r = describe(m.classes, p.tolist(), d, m.arch)
            cells.append([STATES.index(r["state"]), [round(v * 1000) for v in r["probs"]],
                          sum(1 << m.classes.index(c) for c in r["answer"]),
                          None if r["p_quality"] is None else round(r["p_quality"], 3),
                          None if r["p_ood"] is None else round(r["p_ood"], 3)])
        print(f"{min(i + CHUNK, len(squares))}/{len(squares)}", end="\r", flush=True)
    cx, cy, bw, bh = map(float, LABEL.read_text().split()[1:5])
    meta = {"photo": PHOTO.name, "w": w, "h": h, "side": SIDE, "stride": STRIDE, "cols": len(xs), "rows": len(ys),
            "classes": list(m.classes), "states": STATES, "set_threshold": 1 - m.arch.q,
            "label_box": [round((cx - bw / 2) * w), round((cy - bh / 2) * h), round(bw * w), round(bh * h)],
            "fields": ["state", "probs_permille", "answer_mask", "p_quality", "p_ood"], "cells": cells}
    (OUT / "sweep.json").write_text(json.dumps(meta, separators=(",", ":")))
    img.save(OUT / "photo.jpg", quality=86)
    print(f"\n{len(cells)} squares -> {OUT}")


if __name__ == "__main__":
    main()
