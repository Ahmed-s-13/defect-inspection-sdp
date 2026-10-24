# ResNet-18 on d1_sscd_grouped

ImageNet-initialized ResNet-18 with linear classifier; no MLP

Best checkpoint selected by validation macro-F1. See d1_history.csv for every epoch.

## val: all_crops

Accuracy **0.6773**; macro-F1 **0.5882**; balanced accuracy **0.6284**.

| Class | Precision | Recall | F1 | Support (weighted for family metrics) |
|---|---:|---:|---:|---:|
| corrosion | 0.3535 | 0.3862 | 0.3692 | 378.00 |
| crack | 0.6965 | 0.7496 | 0.7221 | 1310.00 |
| dent | 0.9147 | 0.6781 | 0.7788 | 1755.00 |
| missing-head | 0.6865 | 0.8546 | 0.7614 | 392.00 |
| paint-peel-off | 0.5016 | 0.5946 | 0.5442 | 259.00 |
| scratch | 0.2713 | 0.5075 | 0.3536 | 201.00 |

## val: equal_source_family

Accuracy **0.6531**; macro-F1 **0.6121**; balanced accuracy **0.6266**.

| Class | Precision | Recall | F1 | Support (weighted for family metrics) |
|---|---:|---:|---:|---:|
| corrosion | 0.4209 | 0.3580 | 0.3869 | 20.37 |
| crack | 0.5443 | 0.6688 | 0.6002 | 51.35 |
| dent | 0.8799 | 0.7110 | 0.7865 | 85.83 |
| missing-head | 0.7604 | 0.8974 | 0.8232 | 19.95 |
| paint-peel-off | 0.5027 | 0.5602 | 0.5299 | 15.25 |
| scratch | 0.5284 | 0.5639 | 0.5456 | 36.26 |

## test: all_crops

Accuracy **0.7218**; macro-F1 **0.6739**; balanced accuracy **0.7449**.

| Class | Precision | Recall | F1 | Support (weighted for family metrics) |
|---|---:|---:|---:|---:|
| corrosion | 0.5199 | 0.7937 | 0.6283 | 378.00 |
| crack | 0.7469 | 0.6802 | 0.7119 | 1310.00 |
| dent | 0.9351 | 0.6980 | 0.7993 | 1755.00 |
| missing-head | 0.8393 | 0.9592 | 0.8952 | 392.00 |
| paint-peel-off | 0.3894 | 0.6769 | 0.4944 | 260.00 |
| scratch | 0.4209 | 0.6617 | 0.5145 | 201.00 |

## test: equal_source_family

Accuracy **0.7405**; macro-F1 **0.6929**; balanced accuracy **0.7314**.

| Class | Precision | Recall | F1 | Support (weighted for family metrics) |
|---|---:|---:|---:|---:|
| corrosion | 0.3160 | 0.6645 | 0.4283 | 9.33 |
| crack | 0.5267 | 0.6475 | 0.5809 | 27.89 |
| dent | 0.9037 | 0.7553 | 0.8229 | 83.00 |
| missing-head | 0.9041 | 0.9443 | 0.9238 | 26.57 |
| paint-peel-off | 0.8016 | 0.6880 | 0.7405 | 42.61 |
| scratch | 0.6358 | 0.6889 | 0.6613 | 21.61 |

Family-weighted metrics give each estimated source family equal influence; they do not establish that families are truly independent.

Families are estimated, not ground truth. No calibrated grouping confidence or training-seed uncertainty. Test was not used for model selection.

## Saved artifacts

The completed selected weights are archived at [models/resnet18_sscd_grouped/model_best.pt](../../models/resnet18_sscd_grouped/model_best.pt), alongside their matching local split manifest. The default operational checkpoint remains in `../../data/d1_sscd_grouped/work/model_best.pt`.

- [Artifact provenance and configuration](model_artifacts.json)
- [Per-epoch training history](d1_history.csv)
- [Confusion matrix](d1_confusion.png) and [training curves](d1_training_curves.png)
- [Validation predictions](val_predictions.csv) and [test predictions](test_predictions.csv)
- [Machine-readable metrics](metrics.json)
- `source_snapshot/`: saved training/evaluation code and requirements

Weights are stored locally in this repository and remain git-ignored. This run trained ResNet-18 only; no new MLP checkpoint exists.
