# Healing framework: LaMa + screw-head transplant for missing heads

**Result in one line:** on 25 unseen test photos, every defect gets healed by LaMa, and missing heads get a real
screw head put back: of the 33 missing heads big enough to fix, **12 look good, 6 passable, 10 poor** (mostly a
wrong donor such as a paint chip), and 5 were left as plain skin and flagged. One run, judged by eye.

## How it works (`heal.py`)
Per photo, per labelled defect box (class from the D1 labels; `--image/--box/--cls` for your own photos):
1. **One LaMa pass** removes every defect (box grown 12%, whole photo, native resolution).
2. **Not missing-head** -> done (LaMa result).
3. **Missing-head**:
   - box under 16 px (distant shots, 2-8 px boxes) -> LaMa only, tagged "too small for a head";
   - measure the hole and its countersink ring (`fastener.measure_hole`);
   - find intact heads **in the same photo** (`fastener.find_donors`): round shapes clear of every labelled box,
     not a hole (centre darker than 0.35 x the skin = hole), similar size/shape, nearby, and accepted by the
     **"is it a screw head" scorer** (below); try the best 3;
   - else try the best 3 heads of the **donor bank** (best colour/shape match);
   - each paste: donor scaled to the hole (same-photo donors keep their own size if larger), Poisson blending
     (`hcore.transplant`), then a **quality check** (not still a hole, brightness, seams, scale, head visible);
   - nothing passes -> **LaMa only, flagged**.
4. Outputs per run in `out/<run>/`: healed photos, `sheet_*.png` (original | LaMa only | healed), `log.csv` (one
   line per box: route, donor, scores, check numbers), `summary.md`, live `run.log`. The runs behind this report are
   in [`runs/`](runs/) (sheets as JPG, plus `zoom_heads.jpg`: every pasted head at native resolution).

## The parts that had to be built
- **Donor bank** (`build_donor_bank.py`, `donor_bank/`): 206 head candidates mined from 140 **train** photos;
  the **16 clean intact heads** were kept by eye (Phillips, slotted, domed rivets, flush; several paint colours).
- **"Is it a screw head" scorer**: the hard part. Tried and dropped:
  - enhanced_res's p(missing-head): clean Phillips heads scored 0.03-0.04;
  - template matching with the bank: plain skin and stripes scored as high as heads.
  - **Used**: logistic regression on enhanced_res backbone features, trained on the curated candidates
    (16 heads vs 157 non-heads, 33 ambiguous left out). 5-fold CV: 15/16 heads found, 10/157 false heads.
    Threshold 0.5, checked by eye on val candidates ([`tuning/calibrate_val.jpg`](tuning/calibrate_val.jpg):
    each candidate with its score).
- **Hole vs head test**: measured centre/skin brightness: holes 0.06-0.24, intact heads 0.45-0.75 -> cut at 0.35.
- Tuning was done on **val** photos only; the test photos were run once for this report.

## Results
| run | photos | missing heads fixable (>= 16 px) | head from same photo | from bank | LaMa only (flagged) | tiny (< 16 px) |
|---|---|---|---|---|---|---|
| val (tuning) | 12 | 8 | 4 | 4 | 0 | 74 |
| **test** | 25 (+8 other-class photos) | 33 | 22 | 6 | 5 | 149 |
| the 5 hand-fixed photos | 5 | 9 | 8 | 1 | 0 | 0 |

**Test, by eye** ([`runs/test_25/zoom_heads.jpg`](runs/test_25/zoom_heads.jpg)):
| verdict | heads | notes |
|---|---|---|
| good / excellent | **12** | clean Phillips heads (photos 6, 12, 16, 25), flush heads, domed rivets |
| passable | 6 | right idea, slightly off colour, size or smudged |
| poor | 10 | 7 blobs or tiny dots (weak donors), **3 white paint chips pasted as "heads" (photo 24)** |
| LaMa only, flagged | 5 | no donor passed |

**The 5 photos fixed by hand before** ([`runs/manual5/`](runs/manual5/), hand-made version in
[`missing_head_by_hand/`](missing_head_by_hand/report.md)): the framework fixes all 9 missing heads
(I had done 5): 4 good, 3 passable, 2 poor. The hand-made versions were better (1 excellent, 3 good, 1 passable):
a person still picks donors better.

Other classes: 98 boxes on test, all LaMa only, as designed. Speed: ~55 s for 33 photos (~1-2 s per photo,
longer for photos with dozens of boxes).

## What fails, and what would fix it
1. **Wrong donors** (paint chips, blobs): the head scorer is trained on only 16 heads. More curated heads and
   hard negatives (paint chips, stains, letters) are the main lever; a stricter threshold trades heads for safety.
2. **Size / colour slightly off** on some: matching the pasted head's brightness to its neighbours would help.
3. **Semi-automatic mode** for best quality: show the top 3 donor proposals and let a person pick (the hand-made
   results show how much that helps).
4. The `enhanced_res` classifier can't confirm a fix (it has no "intact fastener" class), so judging stays visual.

## Evidence in this folder
| folder | what |
|---|---|
| [`runs/test_25/`](runs/test_25/) | the final test run: sheets, every pasted head (`zoom_heads.jpg`), `log.csv`, `summary.md` |
| [`runs/val_12/`](runs/val_12/) | the last tuning run on val photos |
| [`runs/manual5/`](runs/manual5/) | the healer on the 5 photos first fixed by hand |
| [`missing_head_by_hand/`](missing_head_by_hand/report.md) | the hand-made proof of concept (donors picked by eye) |
| [`lama_vs_cdgan/`](lama_vs_cdgan/report.md) | why LaMa: LaMa vs our CD-GAN vs OpenCV on 10 defects |
| [`tuning/`](tuning/) | screw-head scorer candidates on val with scores; hole/donor check on the 5 hand-fixed photos |

The code is in the folder above (`heal.py`, `fastener.py`, `hcore.py`, `build_donor_bank.py`, `heal_gui.py`).
The tuning sheets were made with small helper scripts that are not kept; their logic is in `fastener.py`.
