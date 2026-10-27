"""One D1 photo through the clean cropper (3_clean_cropper), and how d1_gc7 is put together, for the
Data page. Read-only use of the repo: its candidate, score and pick tables, the photo and its labels.
Writes web/public/cleancrop/ (the photo and its 3 candidate squares at 224 px) and web/src/data/cleancrop.json.

    python scripts/clean_cropper_samples.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

DASH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DASH))

import pandas as pd  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from server.paths import REPO_MAIN  # noqa: E402
from enhanced_res.model import load_split  # noqa: E402

CC = REPO_MAIN.parent / "3_clean_cropper"
sys.path.insert(0, str(REPO_MAIN.parent / "2_lama_healing"))
from hcore import RAW, d1_label_file, read_yolo  # noqa: E402

PHOTO = "train/aircraft_012512.jpg"   # grass, an unlabelled wire and clean panel among its three candidates
KEEP_OUT, MARGIN = 0.5, 0.25          # 3_clean_cropper/candidates.py
OUT_PUB, OUT_JSON = DASH / "web/public/cleancrop", DASH / "web/src/data/cleancrop.json"
r3 = lambda v: round(float(v), 3)  # noqa: E731


def main():
    OUT_PUB.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(CC / "scores.csv").merge(pd.read_csv(CC / "p_clean.csv"), on="id")
    kept = pd.read_csv(CC / "no_defect_clean.csv")
    kept["id"] = kept.crop_path.str.split("/").str[-1].str[:-4]
    picks = json.loads((CC / "pick.json").read_text())
    audit = Counter(json.loads((CC / "review/labels_audit.json").read_text()).values())

    img = ImageOps.exif_transpose(Image.open(RAW / "images" / PHOTO)).convert("RGB")
    w, h = img.size
    img.save(OUT_PUB / "photo.jpg", quality=90)
    boxes = [[r3(x1), r3(y1), r3(x2), r3(y2)] for _, x1, y1, x2, y2 in read_yolo(d1_label_file(PHOTO), w, h)]
    keep_out = [[r3(x1 - KEEP_OUT * (x2 - x1)), r3(y1 - KEEP_OUT * (y2 - y1)), r3(x2 + KEEP_OUT * (x2 - x1)),
                 r3(y2 + KEEP_OUT * (y2 - y1))] for x1, y1, x2, y2 in boxes]
    cands = []
    for k, c in enumerate(s[s.key == PHOTO].sort_values("id").itertuples()):
        Image.open(CC / c.crop).save(OUT_PUB / f"c{k}.jpg", quality=92)
        cands.append({"left": int(c.left), "top": int(c.top), "side": int(c.side), "p_skin": r3(c.p_skin),
                      "p_damage": r3(c.p_damage), "p_clean": r3(c.p_clean), "clip": c.clip_top, "busy": r3(c.busy),
                      "picked": bool((kept.id == c.id).any())})
    assert sum(c["picked"] for c in cands) == 1
    row = kept[kept.key == PHOTO].iloc[0]

    # d1_gc7: the D1 defect crops of d1_grouped + the no-defect crops, each in its photo's split
    d = load_split("d1_gc7")
    by_class = d.groupby(["cls", "split"]).size().unstack(fill_value=0)
    nd = d[d.cls == "no-defect"].groupby(["source", "split"]).size().unstack(fill_value=0)
    n_audit = sum(audit.values())
    out = {
        "photo": {"key": PHOTO, "w": w, "h": h, "group": int(row.src_group), "split": row.split,
                  "sides": sorted({int(round(min(max(b[2] - b[0], b[3] - b[1]) * (1 + 2 * MARGIN), w, h))) for b in boxes})},
        "boxes": boxes, "keep_out": keep_out, "candidates": cands,
        "funnel": {"candidates": int(len(s)), "photos": int(s.key.nunique()), "groups": int(s.group.nunique()),
                   "audit": n_audit, "audit_share": {k: r3(v / n_audit) for k, v in audit.items()},
                   "auc": picks["cv"]["auc"], "clean_in_top": picks["cv"]["clean_share_in_top"],
                   "picked": picks["picked"], "picked_groups": picks["groups"], "picked_split": picks["by_split"],
                   "bins": len(picks["by_bin"]), "per_bin": picks["by_bin"]["0"], "kept": int(len(kept))},
        "gc7": {"classes": {c: {k: int(by_class.loc[c, k]) for k in ("train", "val", "test")} for c in by_class.index},
                "no_defect": {src: {k: int(nd.loc[src, k]) for k in ("train", "val", "test")} for src in nd.index},
                "extra": int(len(pd.read_csv(REPO_MAIN / "d1_gc7/extra_eval.csv")))},
        # the LaMa check from gc7's test report: defect crops with a clean patch LaMa-filled vs the same crops untouched
        "decoys": json.loads((REPO_MAIN / "enhanced_res/results/gc7/summary.json").read_text())["decoys"],
    }
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("photo", "candidates", "funnel")}, indent=1))
    print(by_class, nd, sep="\n")


if __name__ == "__main__":
    main()
