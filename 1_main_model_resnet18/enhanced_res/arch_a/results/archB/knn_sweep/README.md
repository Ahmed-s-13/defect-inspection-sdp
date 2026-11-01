# Architecture B OOD neighbour-count experiment

OOD k only; classifier k=5 unchanged; deployment unchanged

Lowest mean family FRR on 100 seeded 44/44 family halvings of calibration; no test selection. This diagnostic tunes on calibration, so deployment would require independent recalibration.

Controlled fit-only AoN, fixed ResNet features and bank. Each k has separately calibrated p-values. Test comparisons are exploratory; existing test data were examined in previous work. No downstream CP claims.

Calibration-selected candidate: k=1. This is not proof of an ideal k. Gate alpha=0.05.

| k | Calibration family FRR | D1 test FRR | D5 FRR | Held-out non-aircraft FAR | Gate AUPRC |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.224% | 1.344% | 0.298% | 6.400% | 99.536% |
| 3 | 1.310% | 1.453% | 0.298% | 6.492% | 99.521% |
| 5 | 1.316% | 1.417% | 0.298% | 6.538% | 99.515% |
| 7 | 1.308% | 1.490% | 0.298% | 6.538% | 99.516% |
| 10 | 1.312% | 1.381% | 0.298% | 6.584% | 99.521% |
| 15 | 1.334% | 1.308% | 0.298% | 6.630% | 99.522% |
| 20 | 1.312% | 1.308% | 0.298% | 6.538% | 99.519% |
| 30 | 1.290% | 1.381% | 0.298% | 6.538% | 99.513% |
| 50 | 1.266% | 1.417% | 0.298% | 6.492% | 99.487% |

## Deployed AoN reference (non-aircraft categories were training-exposed)

| k | D1 FRR | D5 FRR | Training-exposed FAR |
|---:|---:|---:|---:|
| 1 | 1.381% | 0.298% | 1.427% |
| 3 | 1.526% | 0.298% | 1.473% |
| 5 | 1.490% | 0.298% | 1.473% |
| 7 | 1.562% | 0.298% | 1.427% |
| 10 | 1.453% | 0.298% | 1.427% |
| 15 | 1.381% | 0.298% | 1.427% |
| 20 | 1.381% | 0.298% | 1.381% |
| 30 | 1.490% | 0.298% | 1.381% |
| 50 | 1.599% | 0.298% | 1.381% |

The first table uses the controlled, fit-only AoN; the second uses the deployed AoN. The 5%–95% calibration split ranges in JSON describe split variation, not confidence intervals. Minimizing FRR alone does not optimize detection; compare FAR as well.

## Exact deployed rejected crops

The CSV records each rejected D1 test crop, original image identifier, class, family and both detector p-values. Paths are relative to the repository root. A detector fires at p <= 0.025. Reproduce with `python 1_main_model_resnet18/enhanced_res/arch_a/experiments/archB_knn_sweep.py`.

[Rejected crop manifest](rejected_crops.csv)
