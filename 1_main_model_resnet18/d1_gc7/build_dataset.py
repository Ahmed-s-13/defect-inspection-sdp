"""d1_gc7: d1_grouped's six defect classes + a 7th class "no-defect" (the dataset the main model, gc7, is trained on).

  python d1_gc7/build_dataset.py          (from 1_main_model_resnet18/) -> d1_gc7/no_defect.csv, extra_eval.csv, dataset.md

The six defect classes are d1_grouped's D1 crops, unchanged (enhanced_res.model.load_split). The no-defect crops come
from two modules, both reviewed by hand:
  healed   ../2_lama_healing/no_defect/no_defect_healed.csv   one photo per group with its labelled defects removed by
                                                              the healer, cut at the defect crop squares (1,858)
  clean    ../5_clean_cropper/no_defect_clean.csv             squares from the unboxed parts of the photos, sorted for
                                                              background and unboxed damage (3,000)
Each no-defect crop is cut from a d1_grouped photo, so it takes that photo's group and split. This script checks every
one against d1_grouped/groups.csv and fails on any mismatch, so no source photo reaches two splits.

Ratio: groups can't move between splits, and the no-defect crops come out 59/21/20 (one healed photo per group and
the clean cropper's per-group cap favour val and test, which have many small groups). So val and test no-defect are
trimmed to the defect classes' 70/15/15: each keeps 15/70 of the train count, taken round-robin over its groups (every
group keeps crops) and in proportion to the two sources. The trimmed crops are never trained on; they are listed in
extra_eval.csv as a second no-defect check.

enhanced_res.model.load_split("d1_gc7") = the D1 crops + no_defect.csv. The crop images are in
data/work/crops/no_defect/ (d1_gc7_no_defect_crops.zip on the team Drive, or re-made by the two finalize.py scripts).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent
sys.path.insert(0, str(MAIN / "enhanced_res"))

import pandas as pd  # noqa: E402

from model import GROUPS, load_split  # noqa: E402

REPO = MAIN.parent
SOURCES = {"healed": REPO / "2_lama_healing/no_defect/no_defect_healed.csv",
           "clean_cropper": REPO / "5_clean_cropper/no_defect_clean.csv"}
COLS = ["crop_path", "cls", "image", "key", "src_group", "split", "source"]


def main():
    g = pd.read_csv(GROUPS).set_index("key")
    d = load_split("d1_grouped").assign(source="d1")
    nd = pd.concat([pd.read_csv(f).assign(source=s) for s, f in SOURCES.items()], ignore_index=True)[COLS]

    # every crop's group and split must be its photo's d1_grouped group and split
    for name, x in (("defect", d), ("no-defect", nd)):
        gg = g.loc[x.key]
        bad = (gg.group.values != x.src_group.values) | (gg.new_split.values != x.split.values)
        assert not bad.any(), f"{name}: {bad.sum()} crops with the wrong group or split, e.g. {x[bad].head(3)}"
    missing = [p for p in nd.crop_path if not (MAIN / p).exists()]
    assert not missing, f"{len(missing)} no-defect crops missing, e.g. {missing[:2]}: unzip d1_gc7_no_defect_crops.zip"
    assert not nd.crop_path.duplicated().any()
    assert (nd.groupby("src_group").split.nunique() == 1).all()

    n_target = round(len(nd[nd.split == "train"]) * 15 / 70)
    keep, extra = [nd[nd.split == "train"]], []
    for sp in ("val", "test"):
        x = nd[nd.split == sp].sample(frac=1, random_state=0)
        for src, xs in x.groupby("source"):
            n = round(n_target * len(xs) / len(x))
            xs = xs.assign(r=xs.groupby("src_group").cumcount()).sort_values(["r"], kind="stable")
            keep.append(xs.iloc[:n].drop(columns="r"))
            extra.append(xs.iloc[n:].drop(columns="r"))
    extra = pd.concat(extra, ignore_index=True)
    nd = pd.concat(keep, ignore_index=True)
    extra.to_csv(HERE / "extra_eval.csv", index=False)
    nd.to_csv(HERE / "no_defect.csv", index=False)
    full = pd.concat([d[COLS], nd], ignore_index=True)

    t = pd.crosstab(full.cls, full.split)[["train", "val", "test"]]
    t["val %"] = (100 * t.val / t.sum(axis=1)).round(1)
    t["test %"] = (100 * t.test / t[["train", "val", "test"]].sum(axis=1)).round(1)
    s = pd.crosstab(nd.source, nd.split)[["train", "val", "test"]]
    grp = full.groupby("split").src_group.nunique()[["train", "val", "test"]]
    ndg = nd.groupby("split").src_group.nunique()[["train", "val", "test"]]
    share = (full[full.cls == "no-defect"].split.value_counts() / full.split.value_counts())[["train", "val", "test"]]
    md = ["# d1_gc7", "", "Crops per class and split (each crop in its photo's d1_grouped split):", "",
          t.to_markdown(), "", "No-defect crops by source:", "", s.to_markdown(), "",
          f"Source groups per split: {grp.to_dict()}; with a no-defect crop: {ndg.to_dict()}.", "",
          f"No-defect share of each split: {share.round(3).to_dict()}.", "",
          f"Trimmed from val / test to match 70/15/15 (extra_eval.csv, never trained on): "
          f"{extra.split.value_counts().to_dict()}, by source {extra.source.value_counts().to_dict()}.", "",
          "Checked: every crop's group and split equal its photo's in d1_grouped/groups.csv; no group in two splits; "
          "all files exist."]
    (HERE / "dataset.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
