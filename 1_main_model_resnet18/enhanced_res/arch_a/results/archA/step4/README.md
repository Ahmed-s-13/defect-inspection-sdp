# Architecture A, step 4: from prediction set to visor action

Quality gate alpha 0.05, unfamiliar-input gate alpha 0.08, then prob-T + family-weighted global LAC. Thresholds from calibration.json (val + cal half); everything below on the test half. No retraining.

## The action table

| pipeline outcome | visor shows | inspector |
|---|---|---|
| quality gate fails (blur, dark, overexposed, glare) | **Bad image: re-take** | re-takes the shot |
| unfamiliar-input gate fails (not like any training crop) | **Can't assess: re-aim** | re-aims the camera |
| CP set has 1 class | **Crack 88%** (class + prob-T calibrated score) | no step needed |
| CP set has 2+ classes | **Ambiguous: crack or dent** | reviews; screenshot saved for later |
| CP set is empty | **Unclear: re-inspect** | inspects by eye |

Human attention = everything except a single shown class. The costly case is a single class that is wrong: it is shown with no warning.

## Normal defect crops (all 2,417 test-half crops)

| CP alpha | single class shown | ...right | ...wrong (no warning) | ambiguous | ambiguous holds truth | unclear | re-take | re-aim | **human attention** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.05 | 43.3% | 40.2% | 3.1% | 50.1% | 97.9% | 0.0% | 4.3% | 2.4% | **56.7%** |
| 0.10 | 61.5% | 54.8% | 6.7% | 31.9% | 96.1% | 0.0% | 4.3% | 2.4% | **38.5%** |
| 0.20 | 79.4% | 67.3% | 12.1% | 13.9% | 87.2% | 0.0% | 4.3% | 2.4% | **20.6%** |

## Every test set at CP alpha 0.10

| set | single class: right | single class: wrong | ambiguous | unclear | re-take | re-aim | human attention |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean (all 2,417 test-half crops) | 54.8% | 6.7% | 31.9% | 0.0% | 4.3% | 2.4% | 38.5% |
| clean | 47.2% | 7.6% | 40.0% | 0.0% | 3.1% | 2.2% | 45.2% |
| blur s1 | 41.6% | 6.3% | 41.8% | 0.0% | 4.0% | 6.3% | 52.1% |
| blur s2 | 36.5% | 8.0% | 43.0% | 0.0% | 4.2% | 8.3% | 55.5% |
| motion blur s1 | 42.1% | 6.5% | 41.8% | 0.0% | 2.7% | 6.9% | 51.4% |
| motion blur s2 | 32.5% | 10.5% | 38.5% | 0.0% | 3.1% | 15.4% | 57.0% |
| dark s1 | 33.5% | 5.1% | 30.2% | 0.0% | 15.7% | 15.6% | 61.5% |
| dark s2 | 0.0% | 0.0% | 0.0% | 0.0% | 19.5% | 80.5% | 100.0% |
| overexposed s1 | 20.6% | 1.6% | 18.4% | 0.0% | 37.3% | 22.1% | 77.8% |
| overexposed s2 | 3.8% | 0.5% | 4.3% | 0.0% | 46.7% | 44.7% | 95.7% |
| glare s1 | 30.9% | 11.8% | 35.1% | 0.0% | 10.8% | 11.4% | 57.3% |
| glare s2 | 21.3% | 12.7% | 25.0% | 0.0% | 29.8% | 11.2% | 66.0% |
| noise s1 | 0.0% | 0.4% | 0.9% | 0.0% | 72.2% | 26.6% | 99.6% |
| noise s2 | 0.0% | 0.0% | 0.0% | 0.0% | 13.6% | 86.4% | 100.0% |
| jpeg s1 | 23.1% | 1.6% | 29.7% | 0.0% | 1.1% | 44.5% | 75.2% |
| jpeg s2 | 4.9% | 0.0% | 6.0% | 0.0% | 0.0% | 89.2% | 95.1% |
| off-target | - | 7.8% | 7.5% | 0.0% | 2.8% | 82.0% | 92.2% |

## Human attention on every set, by CP alpha

| set | alpha 0.05 | alpha 0.10 | alpha 0.20 |
|---|---:|---:|---:|
| clean (all 2,417 test-half crops) | 56.7% | 38.5% | 20.6% |
| clean | 64.4% | 45.2% | 25.0% |
| blur s1 | 67.6% | 52.1% | 28.8% |
| blur s2 | 74.0% | 55.5% | 35.3% |
| motion blur s1 | 68.5% | 51.4% | 28.9% |
| motion blur s2 | 74.3% | 57.0% | 35.3% |
| dark s1 | 76.5% | 61.5% | 43.2% |
| dark s2 | 100.0% | 100.0% | 100.0% |
| overexposed s1 | 83.5% | 77.8% | 68.4% |
| overexposed s2 | 98.2% | 95.7% | 93.1% |
| glare s1 | 75.6% | 57.3% | 35.1% |
| glare s2 | 78.3% | 66.0% | 51.5% |
| noise s1 | 100.0% | 99.6% | 99.1% |
| noise s2 | 100.0% | 100.0% | 100.0% |
| jpeg s1 | 84.3% | 75.2% | 57.3% |
| jpeg s2 | 98.2% | 95.1% | 91.9% |
| off-target | 96.5% | 92.2% | 88.5% |

![workload](workload.png)
