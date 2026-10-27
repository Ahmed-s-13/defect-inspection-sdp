"""Step 6 of clean_cropper: apply the final review -> no_defect_clean.csv (the crops to train on).

  python finalize.py

A picked crop is used unless it was rejected on the review page (undecided = kept). The kept crops are copied to
1_main_model_resnet18/data/work/crops/no_defect/clean/<split>/, where d1_gc7 reads them, and listed in
no_defect_clean.csv with paths relative to 1_main_model_resnet18 (like data/work/crops/D1.csv), cls = "no-defect",
the photo's d1_grouped group and split, and the square in the photo. Same columns as
../2_lama_healing/no_defect/no_defect_healed.csv where they overlap.
"""
import json
import shutil
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent / "1_main_model_resnet18"
DEST = Path("data/work/crops/no_defect/clean")           # relative to MAIN

if __name__ == "__main__":
    f = pd.read_csv(HERE / "review/final.csv").merge(pd.read_csv(HERE / "scores.csv"), on="id")
    lf = HERE / "review/labels_final.json"
    dec = json.loads(lf.read_text()) if lf.exists() else {}
    f["final"] = f.id.map(dec).fillna("keep")
    out = f[f.final == "keep"]
    dst = [DEST / s / Path(c).name for s, c in zip(out.split, out.crop)]
    for c, d in zip(out.crop, dst):
        (MAIN / d).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE / c, MAIN / d)
    out = pd.DataFrame(dict(crop_path=[str(d) for d in dst], cls="no-defect", image=out.key, key=out.key,
                            src_group=out.group, split=out.split, left=out.left, top=out.top, side=out.side))
    out.to_csv(HERE / "no_defect_clean.csv", index=False)
    print(f"{len(out)} of {len(f)} picked crops kept ({(f.final == 'reject').sum()} rejected on review)")
    print(out.split.value_counts().to_string(), f"\ngroups {out.src_group.nunique()}")
