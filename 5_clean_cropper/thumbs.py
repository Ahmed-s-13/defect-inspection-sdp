"""Review thumbnails: candidate crop | whole photo with the square (red) and the labelled defect boxes (yellow)."""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2_lama_healing"))
from hcore import RAW, d1_label_file, read_yolo                   # noqa: E402

SIZE = 224


def make(rows):
    """rows: DataFrame with id, key, left, top, side, crop -> review/thumbs/<id>.jpg (skips existing ones)."""
    out = HERE / "work/thumbs"
    out.mkdir(parents=True, exist_ok=True)
    for key, cs in rows.groupby("key"):
        todo = [c for c in cs.itertuples() if not (out / f"{c.id}.jpg").exists()]
        if not todo:
            continue
        img = np.asarray(ImageOps.exif_transpose(Image.open(RAW / "images" / key)).convert("RGB"))
        h, w = img.shape[:2]
        k = SIZE / h
        small = cv2.resize(img, (max(1, round(w * k)), SIZE), interpolation=cv2.INTER_AREA)
        for _, x1, y1, x2, y2 in read_yolo(d1_label_file(key), w, h):
            cv2.rectangle(small, (int(x1 * k), int(y1 * k)), (int(x2 * k), int(y2 * k)), (255, 230, 0), 1)
        for c in todo:
            ctx = small.copy()
            cv2.rectangle(ctx, (int(c.left * k), int(c.top * k)), (int((c.left + c.side) * k), int((c.top + c.side) * k)),
                          (255, 40, 40), 2)
            crop = np.asarray(Image.open(HERE / c.crop).convert("RGB"))
            Image.fromarray(np.concatenate([crop, np.full((SIZE, 6, 3), 255, np.uint8), ctx], 1)).save(out / f"{c.id}.jpg", quality=85)


if __name__ == "__main__":
    make(pd.read_csv(HERE / "candidates.csv").head(5))
