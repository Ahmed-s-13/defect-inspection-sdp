# gc7 (Ahmed) vs gc7_fix (corrected D1 boxes), on the same crops

Same recipe, split, no-defect crops and seed (0); gc7_fix only differs in the 54 corrected D1 boxes (`d1_grouped/label_corrections/`). Each model is scored on both versions of val and test. **B's model is chosen on val**; test is read once, for the report. One training run each, so differences of about a point may be run-to-run noise.

## val

| | gc7 on original crops | gc7_fix on original crops | gc7 on corrected crops | gc7_fix on corrected crops |
|---|---:|---:|---:|---:|
| 7-class accuracy | 77.2% | 77.3% | 77.3% | 77.3% |
| macro-F1 | 0.703 | 0.697 | 0.704 | 0.698 |
| defects called no-defect | 3.2% | 3.2% | 3.2% | 3.1% |
| no-defect recall (healed / clean) | 90.4% (88.2% / 91.9%) | 89.3% (86.6% / 91.1%) | 90.4% (88.2% / 91.9%) | 89.3% (86.6% / 91.1%) |

Recall per class, val, corrected crops:

| class | gc7 | gc7_fix |
|---|---:|---:|
| corrosion | 65.4% | 62.3% |
| crack | 69.4% | 70.8% |
| dent | 85.3% | 87.4% |
| missing-head | 83.3% | 81.9% |
| paint-peel-off | 65.8% | 57.4% |
| scratch | 47.5% | 42.5% |
| no-defect | 90.4% | 89.3% |

## test

| | gc7 on original crops | gc7_fix on original crops | gc7 on corrected crops | gc7_fix on corrected crops |
|---|---:|---:|---:|---:|
| 7-class accuracy | 79.8% | 80.1% | 80.4% | 80.7% |
| macro-F1 | 0.747 (0.705–0.781) | 0.744 (0.704–0.778) | 0.752 (0.709–0.787) | 0.749 (0.709–0.783) |
| defects called no-defect | 3.9% | 2.6% | 3.5% | 2.2% |
| no-defect recall (healed / clean) | 91.1% (88.0% / 93.2%) | 90.1% (85.6% / 93.2%) | 91.1% (88.0% / 93.2%) | 90.1% (85.6% / 93.2%) |

Recall per class, test, corrected crops:

| class | gc7 | gc7_fix |
|---|---:|---:|
| corrosion | 62.2% | 59.5% |
| crack | 76.1% | 77.2% |
| dent | 85.3% | 87.8% |
| missing-head | 95.1% | 95.3% |
| paint-peel-off | 71.7% | 65.7% |
| scratch | 53.0% | 46.0% |
| no-defect | 91.1% | 90.1% |

