# d1_gc7

Crops per class and split (each crop in its photo's d1_grouped split):

| cls            |   train |   val |   test |   val % |   test % |
|:---------------|--------:|------:|-------:|--------:|---------:|
| corrosion      |    1765 |   379 |    378 |    15   |     15   |
| crack          |    6111 |  1307 |   1311 |    15   |     15   |
| dent           |    8190 |  1754 |   1757 |    15   |     15   |
| missing-head   |    1971 |   299 |    344 |    11.4 |     13.2 |
| no-defect      |    2868 |   615 |    615 |    15   |     15   |
| paint-peel-off |    1202 |   263 |    265 |    15.2 |     15.3 |
| scratch        |     940 |   201 |    200 |    15   |     14.9 |

No-defect crops by source:

| source        |   train |   val |   test |
|:--------------|--------:|------:|-------:|
| clean_cropper |    1812 |   369 |    365 |
| healed        |    1056 |   246 |    250 |

Source groups per split: {'train': 496, 'val': 178, 'test': 175}; with a no-defect crop: {'train': 462, 'val': 166, 'test': 162}.

No-defect share of each split: {'train': 0.124, 'val': 0.128, 'test': 0.126}.

Trimmed from val / test to match 70/15/15 (extra_eval.csv, never trained on): {'val': 419, 'test': 341}, by source {'clean_cropper': 454, 'healed': 306}.

Checked: every crop's group and split equal its photo's in d1_grouped/groups.csv; no group in two splits; all files exist.
