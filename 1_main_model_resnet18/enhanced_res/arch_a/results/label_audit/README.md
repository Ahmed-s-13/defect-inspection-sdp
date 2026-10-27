# Label audit: high-confidence crack/dent disagreements

Crops in d1_grouped's test split (cal and test halves) where the given label is crack and enhanced_res says dent, or the reverse, with model confidence >= 0.90: **153 crops, 114 photos, 17 photo families** (101 given crack, 52 given dent).

## How to review

1. Look at `reference.jpg` first: what crack and dent usually look like when the model and label agree.
2. Open each `family_<id>.jpg`. Top: the source photos, flagged boxes in red with their review id. Bottom: each flagged crop with the given label and the model's answer.
3. In `review.csv`, fill `reviewer_label` with crack / dent / both / neither / unsure, and anything useful in `reviewer_notes`. Crops in one family are near-copies (augmented versions of one scene), so one decision usually covers the whole family.
4. Ideally a GAES inspector makes the call: the question is what an inspector would write down.

`given_in_cp_set_a05` / `a10`: whether architecture 0's CP set still contained the given label.

## Families (most flagged first)

| family | split | flagged crops | photos | given crack | given dent | median model confidence | sheet |
|---:|---|---:|---:|---:|---:|---:|---|
| 420 | test | 48 | 11 | 48 | 0 | 0.968 | `family_420.jpg` |
| 491 | test | 19 | 19 | 0 | 19 | 0.965 | `family_491.jpg` |
| 28 | cal | 13 | 13 | 13 | 0 | 0.963 | `family_28.jpg` |
| 75 | cal | 13 | 13 | 0 | 13 | 0.948 | `family_75.jpg` |
| 34 | test | 11 | 10 | 11 | 0 | 0.951 | `family_34.jpg` |
| 25 | test | 10 | 10 | 10 | 0 | 0.960 | `family_25.jpg` |
| 137 | cal | 9 | 9 | 0 | 9 | 0.926 | `family_137.jpg` |
| 126 | test | 6 | 5 | 0 | 6 | 0.924 | `family_126.jpg` |
| 428 | cal | 5 | 5 | 5 | 0 | 0.968 | `family_428.jpg` |
| 325 | test | 5 | 5 | 0 | 5 | 0.932 | `family_325.jpg` |
| 544 | cal | 4 | 4 | 4 | 0 | 0.942 | `family_544.jpg` |
| 332 | test | 4 | 4 | 4 | 0 | 0.936 | `family_332.jpg` |
| 225 | test | 2 | 2 | 2 | 0 | 0.929 | `family_225.jpg` |
| 522 | test | 1 | 1 | 1 | 0 | 0.924 | `family_522.jpg` |
| 150 | cal | 1 | 1 | 1 | 0 | 0.944 | `family_150.jpg` |
| 609 | cal | 1 | 1 | 1 | 0 | 0.964 | `family_609.jpg` |
| 787 | test | 1 | 1 | 1 | 0 | 0.964 | `family_787.jpg` |
