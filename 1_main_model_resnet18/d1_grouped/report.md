# D1 regrouped split (leak-free)

14760 images (all of D1 train+val+test) fall into **861 source groups**. A pair counts as the same source (one photo augmented, or another shot of the same damage) when SIFT+RANSAC finds >= 8 affine inliers, or >= 6 with ResNet-50 cosine >= 0.75, or cosine >= 0.85, or phash <= 2 (cosine and phash over the 8 flips/rotations; 116879 of 260113 candidate pairs). Groups are split 70/15/15: StratifiedGroupKFold on the dominant class, then a local search over whole groups toward 70/15/15 per class (seed 8); no group spans two splits.

Group sizes: 125 singletons, median 6, mean 17.14, max 611; size histogram {10: 320, 3: 196, 1: 125, 6: 62, 2: 49, 4: 31, 5: 26, 9: 21, 7: 19, 8: 12} (10 = 10+).

## Splits

| new_split   |   images |   groups |   boxes |
|:------------|---------:|---------:|--------:|
| train       |    10332 |      502 |   21339 |
| val         |     2214 |      184 |    4572 |
| test        |     2214 |      175 |    4574 |


## Boxes per class per split

|                |   train |   val |   test |
|:---------------|--------:|------:|-------:|
| corrosion      |    1765 |   379 |    378 |
| crack          |    6131 |  1314 |   1314 |
| dent           |    8201 |  1757 |   1758 |
| missing-head   |    3029 |   649 |    649 |
| paint-peel-off |    1244 |   266 |    267 |
| scratch        |     969 |   207 |    208 |


Share of each class's boxes (%):

|                |   train |   val |   test |
|:---------------|--------:|------:|-------:|
| corrosion      |      70 |    15 |     15 |
| crack          |      70 |    15 |     15 |
| dent           |      70 |    15 |     15 |
| missing-head   |      70 |    15 |     15 |
| paint-peel-off |      70 |    15 |     15 |
| scratch        |      70 |    15 |     15 |


## Where the images came from (rows: D1's split, columns: new split)

| split   |   train |   val |   test |
|:--------|--------:|------:|-------:|
| train   |    8224 |  1779 |   1805 |
| val     |    1056 |   227 |    193 |
| test    |    1052 |   208 |    216 |


## Leak check

Every val image against every train+test image, and every test image against every train+val image: full cosine and phash search over the 8 flips/rotations, and SIFT+RANSAC on its 50 nearest other-split images by cosine plus its phash nearest. Matches this check found were added to the groups and the split redone until it found none (new matches per round: [117, 1, 0]).

- flagged as a copy by the grouping rule in the final split: **0 / 4428**
- best scores: inliers >= 8: 0, phash <= 2: 0, cosine >= 0.85: 0
- best other-split cosine: >=0.9: 0, >=0.85: 0, >=0.8: 287, >=0.75: 1240
- best other-split inliers: >=12: 0, >=8: 0, >=6: 54, >=5: 415, >=4: 1894

Sheets (val/test image | its closest image in another split): sheets/leak_check_top_{inliers,cos,phash}.jpg (the 24 closest pairs by each score) and sheets/leak_check_random.jpg (24 random val/test images).

## Dataset

data/d1_grouped/dataset/ holds copied images (not links into D1) and labels as boxes (polygon rows converted to their bounding box) in the model's class order 0 corrosion, 1 crack, 2 dent, 3 missing-head, 4 paint-peel-off, 5 scratch. Its data.yaml has no `path`, so it works in place or zipped for upload.
