# D1 label corrections (27 October 2026)

**Some of D1's original YOLO boxes have been changed.** This folder records every change, so the edited labels can be
re-applied on another machine or undone.

| | count |
|---|---:|
| boxes moved onto the defect | 46 |
| boxes moved and given a different class | 5 |
| boxes deleted (on undamaged skin; the defect is already boxed in the photo) | 3 |
| **label lines changed** | **54** (in 47 label files; d1_grouped split of the photo's family: 21 train, 4 val, 29 test) |

## Why

Many D1 photos are augmented copies of one scene (rotated, sheared, cropped, resized). In some of them the box did not
follow the image: it sits next to the damage, on undamaged skin or an intact screw, while the other copies of the same
scene box the damage itself. The model was then trained and tested on crops of clean skin labelled "crack" or "dent".

## How they were found

1. **Comparing copies.** `calibration/box_audit.py` (local) matched each D1 photo to up to 8 copies of the same
   d1_grouped family (ORB features + RANSAC affine). It mapped their boxes onto the photo and flagged boxes that most
   matched copies don't overlap. 2,944 of 29,464 boxes were flagged. Re-ranking by distance from the spot the copies
   agree on left 149 candidate "misplaced" boxes; the rest were mostly real damage boxed with a different extent, very
   small boxes, or a different defect boxed in the copies.
2. **Visual review.** Each of the 149 candidates was reviewed visually (full photo plus zooms of the old box and the
   copies' consensus spot). A box was changed only when the old box clearly sits on something that is not damage
   **and** the copies clearly agree on real damage nearby. 118 candidates were left alone: their box was on real damage
   (the copies had simply boxed a different defect), or the case was not clear-cut.

The reviewer was Claude (Claude Code), at the project member's request.
3. **Two unclear photos, corrected with a best guess.** In `aircraft_004212` every box is one of the six missing-head
   holes rotated by 90° (box x, y = 1 − hole y, hole x), so all six were rotated back onto the holes. In
   `aircraft_003422` two scratch boxes sit on undamaged skin while the scratch the copies mark (the long crease) is
   already boxed in this photo, so they were deleted.
4. **The rest of family 28.** Its error was repeated across most of its augmented copies, so copy-voting alone missed
   it. The box of the corrected copies (and of far-view copies checked to be right) was mapped onto every other photo
   of the family with the same feature matching, allowing for mirrored copies, and every proposed box was checked
   visually: 22 more boxes moved onto the torn hole. Family 28 now has 25 corrected boxes.

## What changed, exactly

`corrections.csv`, one row per changed line:

| column | meaning |
|---|---|
| `label_file`, `line` | the YOLO file under `data/raw/D1/content/Aircraft_dataset/` and its 0-based line number |
| `action` | `move` (new coordinates, same class), `move+relabel` (new coordinates and class), `delete` (line blanked) |
| `old_cls` / `new_cls` | class before / after |
| `old` / `new` | the full YOLO line before / after (`class cx cy w h`, normalised) |
| `copies_agreeing` | matched copies that agree on the new spot / copies matched (empty for the two best-guess photos) |
| `family`, `split` | d1_grouped photo family and split of the photo |

- **New positions** are the median of the matched copies' boxes mapped onto the photo.
- **New classes** (5 boxes) follow the class those copies give the defect: 4 dent → crack (the spot is a crack line), 1 scratch → dent (a dented panel).
- **The deletions:** one in `aircraft_013656` (two wrong boxes pointed at the same dent, so one was moved and the other removed) and two in `aircraft_003422`. A deleted line is left blank rather than removed, so line numbers stay stable; `prepare.py` skips blank lines.

## Applying or undoing

The raw D1 photos and labels are not in git; they come from `D1_raw.zip`. On a fresh copy of D1, from `1_main_model_resnet18/`:

```bash
python d1_grouped/label_corrections/apply_corrections.py --check    # what would change
python d1_grouped/label_corrections/apply_corrections.py            # apply
python d1_grouped/label_corrections/apply_corrections.py --revert   # undo
```

A line is only changed when it still equals the expected text, so running the script twice is safe.

**Not changed:** the COCO-style `Aircraft_{train,val,test}.json` files shipped with D1. Nothing in this repo reads them.

## What this affects

The D1 crops, `d1_grouped/groups.csv` (which uses box counts) and everything trained or calibrated on them were made
with the **old** boxes. To use the corrections, run `prepare.py` again for D1, then re-run what depends on the crops
(enhanced_res evaluation; architecture A's `calibrate.py`). The 54 lines are about 0.2% of D1's 29,464 boxes, so the
headline numbers should barely move.
