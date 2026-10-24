"""Copy a few held-out test crops into samples/ so the main model can be tried right away.

Only crops from the test split of d1_grouped (the leak-free split enhanced_res uses) are taken: the main
model never trained on them or on any copy of their photos, and they were not used to pick epochs or settings.
One crop per source group, chosen at random with a fixed seed. The legacy model (predict.py --legacy) was
trained on D1's original split, which contains copies of some of these photos, so its accuracy here is optimistic.

  python make_samples.py              5 per class -> samples/*.jpg + samples/labels.csv
  python predict.py samples/ --compare
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from enhanced_res.model import MAIN, load_split


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = load_split()
    d = d[d.split == "test"].sample(frac=1, random_state=a.seed).drop_duplicates("src_group")
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    rows = []
    for c in sorted(d.cls.unique()):
        for i, r in enumerate(d[d.cls == c].head(a.per_class).itertuples(), 1):
            name = f"{c}_{i:02d}.jpg"
            shutil.copy(r.crop_path, out / name)
            rows.append(dict(file=name, cls=c, source_crop=str(Path(r.crop_path).relative_to(MAIN))))
    pd.DataFrame(rows).to_csv(out / "labels.csv", index=False)
    print(f"copied {len(rows)} grouped-test crops to {out} ({a.per_class} per class), labels in {out / 'labels.csv'}")


if __name__ == "__main__":
    main()
