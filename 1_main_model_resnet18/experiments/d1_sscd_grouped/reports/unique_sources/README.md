# D1 estimated source families and excess copies

Across **14,760 images**, the matching pipeline estimates **1,691 source families** and **13,069 excess copies (88.54%)** after retaining one representative per family.

These are estimates of shared visual origin, not a proven count of synthetic augmentations. A family may contain different crops, re-encodings, near-duplicates or overlapping views. Its chosen representative is the lowest image ID, not a claim about which file was the original.

The strict first pass found 1,701 groups. Adding 16 geometric/pixel refinement links gives the estimate above. The **720 broad split blocks are not used as the unique-image estimate**: those deliberately combine uncertain or merely similar scenes for split safety.

| Class | Images containing class | Estimated source families | Excess copies | Excess % |
|---|---:|---:|---:|---:|
| crack | 6,270 | 610 | 5,660 | 90.27% |
| dent | 6,909 | 731 | 6,178 | 89.42% |
| corrosion | 1,267 | 192 | 1,075 | 84.85% |
| scratch | 497 | 255 | 242 | 48.69% |
| missing-head | 908 | 313 | 595 | 65.53% |
| paint-peel-off | 822 | 299 | 523 | 63.63% |

Singleton families: **419**. Images belonging to multi-image families: **14,341**. The latter is different from the excess-copy count because it includes each family's representative.

## Interpretation

Not ground truth or a confidence interval. Missed matches increase estimated families; erroneous merges decrease them. Per-class counts are distinct images/families containing that label; multi-class images and families occur in several rows, so rows do not sum to the overall total. Raw labels are used before the 4-pixel crop filter.

No manual ground-truth validation set was available to calibrate precision/recall or estimate an uncertainty interval. This number should be reported as a matching-based estimate, not an exact property of D1.

## Files and reproduction

- `per_class.csv`: class-level table.
- `summary.json`: definitions, counts and family-size distribution.
- `source_estimate_manifest.csv`: every image, family ID, class membership, family size and representative path.

From this experiment directory:

```sh
.venv/bin/python scripts/estimate_unique_sources.py
```

## D1 SSCD augmentation/copy estimates

Dataset: **`d1_sscd_grouped`**. Counts refer to raw images, before bounding-box cropping. Estimated augmented/copy images means **images minus one representative per matched source family**. This includes crops, re-encodings and near-copies; synthetic augmentation provenance and the true original file are not known. The 720 broader split blocks are used only to prevent split leakage, not to count unique sources.

**Overall: 14,760 images; 1,691 estimated source families; 13,069 excess copies (88.54%).**

### Per class

| Class | Images | Estimated source families | Estimated augmented/copy images | Estimated copy % |
|---|---:|---:|---:|---:|
| crack | 6,270 | 610 | 5,660 | 90.27% |
| dent | 6,909 | 731 | 6,178 | 89.42% |
| corrosion | 1,267 | 192 | 1,075 | 84.85% |
| scratch | 497 | 255 | 242 | 48.69% |
| missing-head | 908 | 313 | 595 | 65.53% |
| paint-peel-off | 822 | 299 | 523 | 63.63% |

### Active train/validation/test split

| Split | Images | Estimated source families | Estimated augmented/copy images | Estimated copy % |
|---|---:|---:|---:|---:|
| train | 10,330 | 1,238 | 9,092 | 88.02% |
| val | 2,216 | 241 | 1,975 | 89.12% |
| test | 2,214 | 212 | 2,002 | 90.42% |

These percentages measure redundancy **within each split**, not leakage between splits. All estimated families remain within one split; undetected relatives may still exist. Percentages use each row’s image count as denominator. Multi-class images/families appear in multiple class rows, so class rows do not sum to the total.

**Confidence: not calibrated.** No defensible numerical confidence score or confidence interval is available. SSCD similarity is not a probability. Missed matches overestimate unique families; false merges underestimate them. Treat these as matching-based estimates, not ground truth.

The reproducible report also provides `per_split_class.csv` for each class within each split.


- `per_split.csv`: active split copy estimates.
- `per_split_class.csv`: class-by-split copy estimates.
