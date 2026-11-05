"""Static dash's Inspect photos: one d1_gc7_fix test crop per class (never trained on), chosen by eye from the
contact sheets of scripts/inspect_candidates.py for how clearly they show the defect. Like the dashboard's own
picks (5_dashboard/scripts/inspect_samples.py), a pick is used only if the served model answers it with its true
class alone and the unfamiliar-input check passes comfortably (p > 0.1), so the demo shows each class working;
otherwise the next choice is tried. Picked for clarity, so they say nothing about accuracy (Batch's are random).
Writes samples/inspect/<class>.jpg + labels.csv. Reads 5_dashboard's server in-process; changes nothing there.

    ../../Experiments/.venv/bin/python scripts/pick_inspect.py      (from 6_static_dash/)
"""
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "5_dashboard"))
from server.preprocess import crop, default_square, to_batch  # noqa: E402
from server.registry import registry  # noqa: E402
from server.states import describe  # noqa: E402

OUT = HERE.parent / "samples/inspect"
CHOICES = {  # class -> contact-sheet tiles (scripts/candidates/<class>.jpg), best first
    "crack": [35, 3, 30, 11, 19],             # a branching crack across grey paint
    "corrosion": [11, 7, 4, 14, 22],          # heavy rust on a panel
    "dent": [17, 10, 16, 43, 35],             # a dent in an engine inlet lip
    "missing-head": [3, 1, 5, 16, 2],         # an empty countersunk hole
    "paint-peel-off": [21, 33, 0, 24, 20],    # paint peeled off an access panel
    "scratch": [0, 9, 3, 2, 14],              # a panel covered in scratches
    "no-defect": [38, 30, 3, 16, 23],         # an intact screw head
}


def main():
    a = registry.active()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls, tiles in CHOICES.items():
        sheet = pd.read_csv(HERE / f"candidates/{cls}.csv", index_col="tile")
        for t in tiles:
            path = Path(sheet.crop_path[t])
            img = Image.open(path).convert("RGB")
            c = crop(img, default_square(*img.size))                 # exactly what Inspect does with the file
            probs, (dec,) = a.predict(to_batch([c]), [c])
            r = describe(a.classes, probs[0].tolist(), dec, a.arch)
            ok = r["answer"] == [cls] and r["p_ood"] is not None and r["p_ood"] > 0.1
            print(f"{cls:15s} tile {t:2d}  {r['state']:10s} {r['answer']} {r['confidence']:.3f}  ood p {r['p_ood']}  {'USE' if ok else 'skip'}")
            if ok:
                shutil.copy(path, OUT / f"{cls}.jpg")
                rows.append(dict(file=f"{cls}.jpg", cls=cls, source_crop=str(path), group=int(sheet.src_group[t])))
                break
        else:
            raise SystemExit(f"no choice for {cls} passed")
    pd.DataFrame(rows).to_csv(OUT / "labels.csv", index=False)


if __name__ == "__main__":
    main()
