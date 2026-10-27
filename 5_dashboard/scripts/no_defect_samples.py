"""Five no-defect test crops of d1_gc7 next to the repo's 30 defect samples, so Inspect and Batch can try the 7th
class. Picked like the repo's make_samples.py: test split only, one crop per source group, at random with a fixed
seed. Writes samples/no-defect_0N.jpg + samples/labels.csv (the server lists them after the repo's samples).

    python scripts/no_defect_samples.py
"""
import shutil
import sys
from pathlib import Path

DASH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DASH))

import pandas as pd  # noqa: E402

from server.paths import REPO_MAIN  # noqa: E402
from enhanced_res.model import load_split  # noqa: E402

OUT, PER_CLASS, SEED = DASH / "samples", 5, 0


def main():
    d = load_split("d1_gc7")
    d = d[(d.split == "test") & (d.cls == "no-defect")].sample(frac=1, random_state=SEED).drop_duplicates("src_group")
    OUT.mkdir(exist_ok=True)
    rows = []
    for i, r in enumerate(d.head(PER_CLASS).itertuples(), 1):
        name = f"no-defect_{i:02d}.jpg"
        shutil.copy(r.crop_path, OUT / name)
        rows.append(dict(file=name, cls="no-defect", source=r.source, source_crop=str(Path(r.crop_path).relative_to(REPO_MAIN))))
    pd.DataFrame(rows).to_csv(OUT / "labels.csv", index=False)
    print(pd.DataFrame(rows)[["file", "source"]].to_string(index=False))


if __name__ == "__main__":
    main()
