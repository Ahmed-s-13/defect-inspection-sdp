"""Shared paths and helpers for the healing task. Reuses the shift task's code."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KL = HERE.parent / "1_main_model_resnet18"     # main model: crops, D1 split, D1 ResNet-18
MSRD_DIR = HERE.parent / "3_teacher_student_msrd"  # teacher-student code (msrd_anomaly.py)
WORK = HERE / "work"
RES = HERE / "results" / "healing"
for p in (KL, MSRD_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from klshift.common import (CROP_SIZE, MEAN, STD, pick_device,  # noqa: E402,F401
                            seed_everything, square_crop_box)


def kl_path(p):
    """Paths in the shift task's CSVs are relative to 1_main_model_resnet18/."""
    p = Path(p)
    return p if p.is_absolute() else KL / p


def d1_defects():
    """D1 defect crops with split and the box in 224-px crop coordinates (bx1, by1, bx2, by2)."""
    import pandas as pd
    s = pd.read_csv(KL / "data/work/d1_split.csv")[["crop_path", "group_id", "split"]]
    return defects("D1").merge(s, on="crop_path")


def defects(ds):
    """Defect crops of one dataset with the box in 224-px crop coordinates. For D2-D5 the
    shift task's in_d1 flag (original image hash-matches a D1 image) is attached."""
    import numpy as np
    import pandas as pd
    c = pd.read_csv(KL / f"data/work/crops/{ds}.csv")
    if ds != "D1":
        e = np.load(KL / "results/embeddings.npz", allow_pickle=True)
        flags = pd.DataFrame({"crop_path": e["crop_path"], "in_d1": e["in_d1"].astype(bool)})
        c = c.merge(flags.drop_duplicates("crop_path"), on="crop_path", how="left")
    f = CROP_SIZE / c.crop_side
    for k, src, off in [("bx1", "x1", "crop_left"), ("by1", "y1", "crop_top"),
                        ("bx2", "x2", "crop_left"), ("by2", "y2", "crop_top")]:
        c[k] = ((c[src] - c[off]) * f).clip(0, CROP_SIZE)
    c["path"] = [str(kl_path(p)) for p in c.crop_path]
    return c


def d1_clean(filtered=False):
    """Real clean patches from step 1. The unfiltered set is the main one (user decision after
    step 2); filtered=True drops the top-5% teacher-student scores (sensitivity check only)."""
    import pandas as pd
    c = pd.read_csv(WORK / "clean_D1.csv")
    f = WORK / "clean_D1_scores.csv"
    if filtered and f.exists():
        c = c.merge(pd.read_csv(f)[["crop_path", "ts_score", "dropped"]], on="crop_path")
        c = c[~c.dropped]
    c["path"] = [str(HERE / p) for p in c.crop_path]
    return c


def group_half(c, seed):
    """Random half of the clean set's image groups. The artifact probes take real clean
    patches from one half and modified ones from the other, so no image is in both classes."""
    import numpy as np
    gid = c.group_id.unique()
    return set(np.random.default_rng(seed).choice(gid, len(gid) // 2, replace=False))
