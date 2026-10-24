# d1_grouped: a leak-free train/val/test split of D1

**This is the split the main model (`../enhanced_res/`) is trained and tested on.** The split itself is
[`groups.csv`](groups.csv): one row per D1 image with its source group and its split. Full numbers are in
[`report.md`](report.md).

## Why

D1 is mostly copies. One photo appears flipped, rotated, cropped, recoloured or greyscaled many times, and D1's
shipped split put copies of the same photo in both train and test. Scores on that split measured memory rather
than skill. The same ResNet-18 + MLP model scores 90.1% on the old split and 78.1% here: the 12-point
difference is inflation the leak was hiding.

## The framework: fingerprint → confirm → group → check

1. **Fingerprint every image so flips and rotations don't matter.** Two compact summaries per image, each
   computed on all 8 flips/90° rotations: a perceptual hash (64 bits) and an ImageNet ResNet-50 embedding
   (2,048 numbers, an intermediate activation).
2. **Find candidates, then confirm them.** Each image's nearest neighbours by fingerprint are candidate copies.
   Each candidate pair is confirmed with SIFT keypoint matching plus a RANSAC geometric fit. That catches small
   rotations, crops and colour shifts the fingerprints miss.
   - **Same-source rule:** ≥ 8 matched points, or ≥ 6 with cosine ≥ 0.75, or cosine ≥ 0.85, or phash ≤ 2.
   - **How it was set:** by eye on the sheets in [`sheets/calibration/`](sheets/calibration/). Looser rules
     chained unrelated images into giant groups: phash ≤ 8 alone linked about 9,000 images through the black
     corners of rotated copies.
3. **Group, then split.** Confirmed pairs are joined into groups (A matches B and B matches C, so all three are
   one group): 14,760 images become 861 groups. Whole groups go to train/val/test, and every defect class is
   split exactly 70/15/15 (StratifiedGroupKFold, then a local search over whole groups).
4. **Check, fix, repeat.** Every val/test image is compared with every image in the other splits. Any match
   found is merged into its group and the split is redone, until the check finds nothing.

| | images | groups | boxes |
|---|---:|---:|---:|
| train | 10,332 | 502 | 21,339 |
| val | 2,214 | 184 | 4,572 |
| test | 2,214 | 175 | 4,574 |

**Limits.** Re-shoots of the same damage from a clearly different angle can still sit in different splits; the
closest remaining cross-split pairs are of that kind ([`sheets/leak_check_top_*.jpg`](sheets/)). Many groups are
large: a single damage site can have hundreds of copies ([`sheets/largest_groups.jpg`](sheets/largest_groups.jpg)).
So the effective sample size is closer to 861 than to 14,760.

This is separate from the teammates' SSCD grouping in `../experiments/d1_sscd_grouped/`, which reaches a
similar conclusion with a different copy detector and its own split.

## Getting the data

- **To train or test the ResNet:** nothing extra. `groups.csv` (in git) plus the D1 crops in `data/work/`
  (`main_model_data_work.zip`) is all `../enhanced_res/` needs.
- **The image dataset in YOLO format** (images copied into train/val/test, labels as boxes in alphabetical
  class order, `data.yaml`): download `d1_grouped` from the team's shared Google Drive into
  `1_main_model_resnet18/data/d1_grouped/`, or rebuild it as below.

## Rebuilding it

Needs D1's raw photos (`D1_raw.zip`) and `imagehash` + `opencv-python` (in `requirements.txt`). Run from this
folder:

```bash
python group.py features   # fingerprints + SIFT keypoints for every image (~25 min, caches in ../data/work/d1_grouped_cache)
python group.py pairs      # candidate pairs + SIFT verification (~3 min)
python split.py            # group -> split -> leak-check rounds; writes groups.csv, report.md, leak_check.csv, sheets/
```

| file | what it is |
|---|---|
| `groups.csv` | the split: D1 key, D1's original split, source group, new split, box count, dominant class |
| `report.md` | counts per split and class, leak-check results |
| `leak_check.csv` | each val/test image's closest image in another split (cosine, phash, SIFT inliers) |
| `sheets/` | final leak-check sheets and the largest groups; `calibration/` holds the sheets used to set the rule |
