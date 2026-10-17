"""Copy a few held-out D1-test crops into samples/ so the model can be tried right away.

Only D1-test crops are used: neither the ResNet-18 nor the MLP head trained on them, and they were not
used to pick epochs or settings. One crop per image group, chosen at random with a fixed seed.

  python make_samples.py              5 per class -> samples/*.jpg + samples/labels.csv
  python predict.py samples/ --compare
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="data/work", help="folder with d1_split.csv")
    ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    d = pd.read_csv(Path(a.work) / "d1_split.csv")
    d = d[d.split == "test"].sample(frac=1, random_state=a.seed).drop_duplicates("group_id")
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    rows = []
    for c in sorted(d.cls.unique()):
        for i, r in enumerate(d[d.cls == c].head(a.per_class).itertuples(), 1):
            name = f"{c}_{i:02d}.jpg"
            shutil.copy(r.crop_path, out / name)
            rows.append(dict(file=name, cls=c, source_crop=r.crop_path))
    pd.DataFrame(rows).to_csv(out / "labels.csv", index=False)
    print(f"copied {len(rows)} D1-test crops to {out} ({a.per_class} per class), labels in {out / 'labels.csv'}")


if __name__ == "__main__":
    main()
