"""Write the corrected crops of the 47 photos in corrections.csv into crops/ (in git, about 1.5 MB), so a computer
with only the team's data/work crops can use d1_grouped_fix / d1_gc7_fix without the raw D1 photos.

enhanced_res/model.py::load_split reads data/work_fix (the full re-cut) when it exists; otherwise it takes
data/work/crops/D1.csv and swaps in crops/crops.csv for these 47 photos. Every other photo's crops are unchanged by
the corrections, so the two give the same split.

From 1_main_model_resnet18/, after the re-cut (apply_corrections.py on data/raw/D1, then
prepare.py --work data/work_fix --datasets D1):
  python d1_grouped/label_corrections/make_crop_pack.py
"""
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[1]
OUT = HERE / "crops"


def corrected_images():
    """'content/Aircraft_dataset/images/<split>/<stem>' of each photo whose labels were corrected (no extension)."""
    c = pd.read_csv(HERE / "corrections.csv", keep_default_na=False)
    return {"content/Aircraft_dataset/images/" + f[len("labels/"):-len(".txt")] for f in c.label_file.unique()}


def stem(image):
    return image.rsplit(".", 1)[0]


def main():
    fix = pd.read_csv(MAIN / "data/work_fix/crops/D1.csv")
    keep = corrected_images()
    rows = fix[fix.image.map(stem).isin(keep)].copy()
    assert rows.image.map(stem).nunique() == len(keep), "a corrected photo has no crops in data/work_fix"
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir()
    for p in rows.crop_path:
        shutil.copy(MAIN / p, OUT / Path(p).name)
    rows["crop_path"] = [str((OUT / Path(p).name).relative_to(MAIN)) for p in rows.crop_path]
    rows.to_csv(OUT / "crops.csv", index=False)
    print(f"{len(rows)} crops of {len(keep)} photos -> {OUT.relative_to(MAIN)}")


if __name__ == "__main__":
    main()
