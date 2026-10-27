"""The one-per-class sample photos on the Inspect page: d1_gc7 test crops (never trained on, one per photo group)
that the served model answers clearly, so a demo shows each class working. Chosen by eye from the test crops whose
architecture A answer is the true class alone with both checks passed comfortably (p > 0.3); this script copies
them and checks that is still true. They are picked for clarity, so they say nothing about accuracy: the Batch
page's 35 samples are the random ones (the repo's make_samples.py + scripts/no_defect_samples.py).
Writes samples/inspect/<class>.jpg + samples/inspect/labels.csv.

    python scripts/inspect_samples.py
"""
import shutil
import sys
from pathlib import Path

DASH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DASH))

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from server.paths import REPO_MAIN  # noqa: E402
from server.preprocess import crop, default_square, to_batch  # noqa: E402
from server.registry import registry  # noqa: E402
from server.states import describe  # noqa: E402

OUT = DASH / "samples/inspect"
CROPS = "data/work/crops/D1/content__Aircraft_dataset__images__"
PICKS = {  # class -> crop (relative to 1_main_model_resnet18), all in d1_gc7's test split
    "corrosion": CROPS + "train__aircraft_012716__0.jpg",            # rust around a screw head (group 524)
    "crack": CROPS + "train__aircraft_004016__1.jpg",                # cracks across a painted panel (group 137)
    "dent": CROPS + "test__aircraft_005101__0.jpg",                  # a dimpled panel (group 18)
    "missing-head": CROPS + "train__aircraft_007551__0.jpg",         # an empty countersunk hole (group 221)
    "paint-peel-off": CROPS + "train__aircraft_014600__4.jpg",       # paint flaking off grey skin (group 374)
    "scratch": CROPS + "train__aircraft_011208__4.jpg",              # one straight scratch (group 803)
    "no-defect": "data/work/crops/no_defect/clean/test/train__aircraft_004805__c2.jpg",  # plain panel (group 331)
}


def main():
    a = registry.active()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for cls, rel in PICKS.items():
        img = Image.open(REPO_MAIN / rel).convert("RGB")
        c = crop(img, default_square(*img.size))           # exactly what Inspect does with the file
        probs, (dec,) = a.predict(to_batch([c]), [c])
        r = describe(a.classes, probs[0].tolist(), dec, a.arch)
        assert r["answer"] == [cls] and r["p_quality"] > 0.1 and r["p_ood"] > 0.1, (cls, r["state"], r["answer"])
        name = f"{cls}.jpg"
        shutil.copy(REPO_MAIN / rel, OUT / name)
        rows.append(dict(file=name, cls=cls, source_crop=rel))
        print(f"{cls:15s} {r['state']:9s} {r['confidence']:.3f}  quality p {r['p_quality']:.2f}  unfamiliar p {r['p_ood']:.2f}")
    pd.DataFrame(rows).to_csv(OUT / "labels.csv", index=False)


if __name__ == "__main__":
    main()
