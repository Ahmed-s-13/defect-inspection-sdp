"""Heal one D1 photo per d1_grouped group with the 2_lama_healing healer (candidate "no defect" images).

  python heal_groups.py                 all 861 groups (resumes: groups already in manifest.csv are skipped)
  python heal_groups.py --limit 5       first 5 groups only (test)

Per group: the first photo (by key) that has at least one labelled box. Its healed version is written to
healed/<split>/<group>_<stem>.png, where <split> is the group's d1_grouped split, so a healed test photo never
lands in train. Outputs, all in this folder:
  manifest.csv   one row per group: photo, split, boxes, how much of the photo was healed, screw heads put back
  boxes.csv      one row per defect box: what the healer did to it (same columns as 2_lama_healing's log.csv)
  run.log        live progress
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
HEALER = HERE.parent                                             # 2_lama_healing
sys.path.insert(0, str(HEALER))
import heal as hf                                                 # noqa: E402
from hcore import RAW, d1_label_file, read_yolo                   # noqa: E402

GROUPS = HERE.parents[1] / "1_main_model_resnet18/d1_grouped/groups.csv"
WORK = HERE / "work"                                             # git-ignored: healed photos, crops, thumbnails
OUT = WORK / "healed"
MANIFEST, BOXES = HERE / "manifest.csv", HERE / "boxes.csv"
M_COLS = ["group", "split", "key", "file", "w", "h", "n_boxes", "classes", "healed_frac", "mh_boxes",
          "heads_put_back", "mh_flagged", "mh_too_small", "secs"]
B_COLS = ["group", "split", "key", "box_i", "cls", "box", "route", "donor", "p_fastener", "hole_method", "hole_conf",
          "hole", "qc_metrics", "qc"]


def pick(groups):
    """{group: (split, key)} for the first photo of each group with a non-empty label file; and the groups without."""
    out, none = {}, []
    for gid, g in groups.sort_values("key").groupby("group", sort=True):
        for key in g.key:
            f = d1_label_file(key)
            if f.exists() and f.read_text().strip():
                out[gid] = (g.new_split.iloc[0], key)
                break
        else:
            none.append(gid)
    return out, none


def appender(path, cols):
    new = not path.exists()
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, cols, extrasaction="ignore")
    if new:
        w.writeheader()
    return f, w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="only the first N groups (test)")
    a = ap.parse_args()

    groups = pd.read_csv(GROUPS)
    assert groups.groupby("group").new_split.nunique().max() == 1, "a group spans two splits"
    chosen, unlabelled = pick(groups)
    done = set(pd.read_csv(MANIFEST).group) if MANIFEST.exists() else set()
    todo = [g for g in sorted(chosen) if g not in done][:a.limit]

    WORK.mkdir(exist_ok=True)
    live = open(WORK / "heal_run.log", "a", encoding="utf-8", buffering=1)

    def say(s):
        print(s, flush=True)
        live.write(s + "\n")

    say(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')}: {len(chosen)} groups with a labelled photo "
        f"({len(unlabelled)} without: {unlabelled}), {len(done)} already done, {len(todo)} to do")
    healer = hf.Healer()
    say(f"device {healer.dev.type}, donor bank {len(healer.bank)} heads")
    mf, mw = appender(MANIFEST, M_COLS)
    bf, bw = appender(BOXES, B_COLS)
    t_all = time.time()
    for n, gid in enumerate(todo, 1):
        split, key = chosen[gid]
        t0 = time.time()
        img = np.asarray(ImageOps.exif_transpose(Image.open(RAW / "images" / key)).convert("RGB")).copy()
        h, w = img.shape[:2]
        boxes = read_yolo(d1_label_file(key), w, h)
        clean, out, rows = healer.heal(img, boxes)
        dst = OUT / split / f"{gid:04d}_{Path(key).stem}.png"
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), out[..., ::-1])
        route = lambda k: sum(r["route"] == k for r in rows)
        secs = time.time() - t0
        for r in rows:
            r.update(group=gid, split=split, key=key)
            bw.writerow(r)
        mw.writerow(dict(group=gid, split=split, key=key, file=str(dst.relative_to(HERE)), w=w, h=h,
                         n_boxes=len(boxes), classes=" ".join(sorted({b[0] for b in boxes})),
                         healed_frac=round((hf.defect_mask(img.shape, boxes) > 0).mean(), 3),
                         mh_boxes=sum(r["cls"] == hf.MH for r in rows),
                         heads_put_back=sum(r["route"].startswith("head") for r in rows),
                         mh_flagged=route("lama only (no donor passed)"),
                         mh_too_small=route("lama only (too small for a head)"), secs=round(secs, 1)))
        mf.flush()
        bf.flush()
        eta = (time.time() - t_all) / n * (len(todo) - n)
        say(f"[{n}/{len(todo)}] group {gid} {split} {key}: {len(boxes)} box(es) | {secs:.1f}s | ETA {eta / 60:.0f} min")
    mf.close()
    bf.close()
    say(f"done: {len(todo)} photos in {(time.time() - t_all) / 60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
