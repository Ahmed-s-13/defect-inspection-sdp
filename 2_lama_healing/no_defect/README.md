# Healed crops for the "no-defect" class

The healer removes every labelled defect from one D1 photo per d1_grouped group. The healed photo is cut at exactly the
squares of the real defect crops, the bad crops are filtered out and the rest are reviewed by hand. The result,
`no_defect_healed.csv` (**1,858 crops**), is half of the 7th class of [`d1_gc7`](../../1_main_model_resnet18/d1_gc7/README.md),
which trains the main model (enhanced_res gc7). The other half comes from [`3_clean_cropper`](../../3_clean_cropper/README.md).

No trained model chooses which crops are kept. enhanced_res is not used, because these crops were made to train it. The
filter uses the healer's own log, image statistics and a person's review.

## Steps (from this folder)

| step | command | output | time |
|---|---|---|---|
| 1. heal one photo per group | `python heal_groups.py` | `work/healed/<split>/`, `manifest.csv`, `boxes.csv` | 49 min (CPU) |
| 2. cut crops, measure, decide | `python make_crops.py` | `work/crops/<split>/`, `work/thumbs/`, `crops.csv` | 20 s |
| 3. review by hand | `python review_server.py`, open http://127.0.0.1:8791 | `review/decisions.json` | you |
| 4. final list | `python finalize.py --pending drop` | `no_defect_healed.csv`, crops in `1_main_model_resnet18/data/work/crops/no_defect/healed/` | 2 s |
| 5. LaMa-shortcut probe | `python probe.py` | `probe/results.json`, `probe/heatmaps_*.jpg` | 13 min (MPS) |
| 6. decoys (optional) | `python decoys.py`, then `python check_decoys.py` | `work/decoys/`, `decoys.csv` | ~2 h for all D1 (MPS) |

The tables, the review decisions and the reports are in git. The images in `work/` are not. To rebuild them you need
`D1_raw.zip`, `main_model_data_work.zip`, `lama_weights.zip` and the v2 weights (`enhanced_res_model.zip`, used by the
screw-head scorer). **To train or test the model you don't need any of this.** The final crops are
`d1_gc7_no_defect_crops.zip` on the team Drive.

Every crop keeps its photo's **d1_grouped** split (train / val / test), not D1's own folder, so no test photo reaches
training.

## 1. Healing (849 photos)

The first photo (by key) in each group that has a non-empty label file. 12 groups have none: 24 182 349 458 492 499
560 570 583 706 766 800. `manifest.csv` has one row per photo, including `healed_frac`, the share of the photo LaMa
repainted. `boxes.csv` records what the healer did to each of the 3,138 boxes.

## 2. Crops and automatic decisions (2,631 crops)

A healed crop sits on exactly the square of a real defect crop: box + 25% margin, bicubic to 224 px, JPEG 95. So
each healed crop pairs with a defect crop (`source_crop`). LaMa's fill covers most of each crop, because it repaints
the box grown by 12% per side.

**Dropped by rules (443):**
- A missing-head box with no head put back: 251 were too small for a head and 52 had no donor that passed. A hole
  filled with plain skin doesn't look like an intact fastener.
- The photo was over half repainted (96). These come out as blank texture (`sheets/over_half_healed.jpg`).
- Under 3% of the crop is untouched photo (56). It is all LaMa, with nothing real left.

**Flagged for review (520), on image statistics only.** `sheets/flags_*.jpg` show the worst crop, the 10th
percentile and the 25th percentile of each flag.
- `colour patch`: the Lab distance between the fill and the skin around it is over 18. These are mostly leftover
  stains, dark blobs and rust halos.
- `seam`: edge strength along the fill's border, compared with elsewhere, is over 1.8. This is mostly a defect
  running past its box, such as a scratch whose end is outside the label.
- `blank fill`: fine detail in the fill divided by fine detail in the untouched part is under 0.35. This catches
  smeared fills, but also fills over flat sky next to foliage, so it is the weakest signal.

**Auto kept (1,668):** nothing looks off.

## 3–4. Review and final list

Each card shows the original crop (yellow = the area LaMa repainted) next to the healed crop. A click switches it
between keep and reject and is saved to `review/decisions.json`. Reject a crop when damage is still visible, the fill
is an obvious smear or patch, or it no longer looks like a real surface.

The review rejected 329 crops and kept 190 of the flagged ones. One flagged crop was left undecided and dropped. The
final list has **1,858 crops**: train 1,056, val 413 and test 389, from 746 groups. `finalize.py` copies them to
`1_main_model_resnet18/data/work/crops/no_defect/healed/<split>/`. It writes `no_defect_healed.csv` with the columns
of the D1 crop list (`crop_path, cls, image, key, src_group, split`, with `cls = no-defect`), plus `healed_from` (the
removed defect class) and `source_crop`.

## 5. LaMa-shortcut probe (`probe.py`)

The probe asks whether a small model trained from scratch can tell LaMa fill from untouched photo in the kept crops.
It cuts 32 px patches from the 224 px crops:
- **fill:** fully inside the LaMa area and at least 4 px from its edge, so seams don't count;
- **real:** fully outside the LaMa area and within 48 px of it, so the skin and light are the same.

Each crop gives equal numbers of both. The probe is trained on train-split crops and scored with ROC AUC on test
(0.5 means it can't tell them apart):
- colour-only control: 0.57;
- texture statistics with gradient boosting: 0.81;
- small CNN: 0.73, which drops to 0.60 when the patches are augmented.

**The cue is smoothness.** Laplacian variance in the fill is under half that of the untouched photo (47 vs 115).
That is a moderate shortcut, "unnaturally smooth = no defect", which a defect classifier can pick up because most
defects add texture.

**Checked on the trained model.** gc7 recognises healed crops *less* often than clean-cropper crops (88% vs 93%).
Defect crops with a clean patch LaMa-filled are called no-defect only a little more often than the same crops
untouched: 4.7% vs 3.0% of 301 crops (`decoys.py`). The model leans on the damage, not on LaMa's smoothness.

## 6. Decoys (`decoys.py`, `check_decoys.py`)

A decoy is a real defect crop with a rectangle of *clean* skin LaMa-filled, from a single LaMa pass on the whole
photo, the same way the healed photos were made. The defect is untouched and the label stays the same.

Decoys serve two purposes:
- **a test:** a model that learned "LaMa fill = no defect" calls them no-defect;
- **a fix:** adding them to training would put LaMa fill in every class.

The run stopped at 750 of 14,615 photos; `python decoys.py` resumes it. A decoy covers at most about 30% of a crop
(median 23%), because the defect sits in the middle. Healed crops are about 52% fill.
