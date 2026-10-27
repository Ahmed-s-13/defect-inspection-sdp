# d1_gc7: the dataset of the main model

**d1_gc7 = [d1_grouped](../d1_grouped/README.md)'s six defect classes + a 7th class, `no-defect`.** It is what the
main model, [enhanced_res gc7](../enhanced_res/README.md), is trained and tested on. The split is still d1_grouped's
leak-free split (`../d1_grouped/groups.csv`): every crop, defect or not, sits in its source photo's group and split,
so no photo or copy of it is in both train and test.

| class | train | val | test |
|---|---|---|---|
| corrosion | 1,765 | 379 | 378 |
| crack | 6,111 | 1,307 | 1,311 |
| dent | 8,190 | 1,754 | 1,757 |
| missing-head | 1,971 | 299 | 344 |
| paint-peel-off | 1,202 | 263 | 265 |
| scratch | 940 | 201 | 200 |
| **no-defect** | **2,868** | **615** | **615** |
| of which healed / clean | 1,056 / 1,812 | 246 / 369 | 250 / 365 |

The six defect classes are the D1 crops, unchanged (`data/work/crops/D1.csv`). The no-defect crops come from two
modules, both reviewed by hand, and are all cut from d1_grouped photos:

| source | module | what | crops |
|---|---|---|---|
| healed | [`2_lama_healing/no_defect`](../../2_lama_healing/no_defect/README.md) | one photo per group with its labelled defects removed by the healer, cut at the defect crop squares (the same spot as a real defect, at the same scale) | 1,858 |
| clean | [`5_clean_cropper`](../../5_clean_cropper/README.md) | squares from the unboxed parts of the photos, sorted for background and unboxed damage | 3,000 |

## Files

| file | what |
|---|---|
| `build_dataset.py` | combines the two lists, checks every crop's group and split against `d1_grouped/groups.csv` (fails on any mismatch), trims val/test, writes the files below |
| `no_defect.csv` | the 7th class: `crop_path` (relative to `1_main_model_resnet18/`), `cls`, `image`, `key`, `src_group`, `split`, `source` |
| `extra_eval.csv` | 760 reviewed no-defect crops trimmed from val/test (never trained on): a second check |
| `dataset.md` | the counts above, as written by `build_dataset.py` |

In code: `enhanced_res.model.load_split("d1_gc7")` gives every crop (the D1 crops + `no_defect.csv`), and
`load_split()` still gives d1_grouped's six classes.

**The crop images** are in `data/work/crops/no_defect/{healed,clean}/<split>/` (git-ignored):
`d1_gc7_no_defect_crops.zip` on the team Drive, unzipped in the repo root. Or re-make them with the two modules'
`finalize.py`.

## Why the val/test no-defect counts were trimmed

Groups can't move between splits. Taken whole, the reviewed no-defect crops come out 59/21/20, because one healed photo
per group and the clean cropper's per-group cap favour val and test, which have many small groups. So val and test keep
15/70 of the train count each, the defect classes' 70/15/15. The kept crops are taken round-robin over groups (every
group keeps crops) and in proportion to the two sources. The 760 trimmed crops are in `extra_eval.csv`.

```bash
python d1_gc7/build_dataset.py        # from 1_main_model_resnet18/, after both modules' finalize.py
```
