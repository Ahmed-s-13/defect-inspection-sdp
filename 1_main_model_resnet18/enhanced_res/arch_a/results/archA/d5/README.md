# Distribution shift: the D5 outside test

1008 D5 crops whose photo is not a copy of a D1 photo (the team's list of 1040; 32 dropped because their crop numbering moved), from 276 near-duplicate groups: {'missing-head': 828, 'dent': 180}. D5 only has two of the six classes. Everything is calibrated on D1 (calibration.json: T on val, thresholds and gates on val + cal half / train) and applied unchanged. No retraining, nothing refitted on D5.

## Model and calibration under shift (prob-T probabilities)

| set | accuracy | mean stated confidence | ECE |
|---|---:|---:|---:|
| D1 test half | 79.1% | 80.8% | 0.035 |
| D5 outside test | 84.0% | 81.0% | 0.071 |

What the model predicts on D5 (rows: true class):

| true | corrosion | crack | dent | missing-head | paint-peel-off | scratch |
|---|---:|---:|---:|---:|---:|---:|
| dent | 3 | 5 | 119 | 2 | 1 | 50 |
| missing-head | 51 | 10 | 14 | 728 | 15 | 10 |

## CP coverage: does the D1-calibrated guarantee survive?

| alpha | method | set | coverage | group coverage | dent | missing-head | mean set size | single class |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0.05 | architecture 0 | D1 test half | 92.8% | 93.5% | 96.8% | 98.5% | 2.10 | 61.3% |
| 0.05 | architecture 0 | D5 outside test | 96.5% | 97.7% | 90.0% | 97.9% | 2.16 | 62.2% |
| 0.05 | architecture A (CP only) | D1 test half | 95.7% | 96.3% | 98.3% | 98.5% | 2.62 | 45.8% |
| 0.05 | architecture A (CP only) | D5 outside test | 97.9% | 98.6% | 92.8% | 99.0% | 2.64 | 51.0% |
| 0.10 | architecture 0 | D1 test half | 88.5% | 87.1% | 94.9% | 97.0% | 1.34 | 74.5% |
| 0.10 | architecture 0 | D5 outside test | 92.2% | 93.3% | 78.9% | 95.0% | 1.37 | 76.1% |
| 0.10 | architecture A (CP only) | D1 test half | 91.5% | 91.7% | 96.3% | 97.0% | 1.65 | 65.2% |
| 0.10 | architecture A (CP only) | D5 outside test | 95.0% | 96.3% | 87.2% | 96.7% | 1.68 | 66.4% |
| 0.20 | architecture 0 | D1 test half | 77.2% | 73.1% | 87.2% | 92.4% | 0.95 | 94.5% |
| 0.20 | architecture 0 | D5 outside test | 81.8% | 83.0% | 61.1% | 86.4% | 0.94 | 93.6% |
| 0.20 | architecture A (CP only) | D1 test half | 85.1% | 83.6% | 92.6% | 96.2% | 1.17 | 84.1% |
| 0.20 | architecture A (CP only) | D5 outside test | 89.4% | 90.3% | 74.4% | 92.6% | 1.16 | 85.5% |

## Full pipeline (gates + CP) as visor actions

| alpha | set | single class right | single class wrong (no warning) | ambiguous | ambiguous holds truth | re-take | re-aim | human attention |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | D1 test half | 40.2% | 3.1% | 50.1% | 97.9% | 4.3% | 2.4% | 56.7% |
| 0.05 | D5 outside test | 50.1% | 0.5% | 42.8% | 97.2% | 5.2% | 1.5% | 49.4% |
| 0.10 | D1 test half | 54.8% | 6.7% | 31.9% | 96.1% | 4.3% | 2.4% | 38.5% |
| 0.10 | D5 outside test | 62.5% | 2.2% | 28.7% | 93.4% | 5.2% | 1.5% | 35.3% |
| 0.20 | D1 test half | 67.3% | 12.1% | 13.9% | 87.2% | 4.3% | 2.4% | 20.6% |
| 0.20 | D5 outside test | 75.0% | 6.0% | 12.4% | 79.2% | 5.2% | 1.5% | 19.0% |

Quality-gate reasons on D5 (flagged crops): {'sharpness': 51, 'clipped_lo': 10, 'brightness': 1}
