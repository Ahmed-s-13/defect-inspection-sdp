# Defect classification: first results (10 Oct 2026)

_Script names below refer to the old layout: `train_baseline.py` is now `model_1_plain_cnn/train.py` and `train_supcon.py` is now `model_2_supcon_resnet18/train.py`. The ResNet-18 cross-entropy control model is not kept._

Both models trained and evaluated on a MacBook (Apple GPU) with the code in `defect-classification`. Test sets are whole photos the models never saw (re-split by original photo).

Settings: plain CNN at defaults (up to 60 epochs). SupCon + cRT with `--arch resnet18 --batch-size 32 --epochs 30` (ResNet-50 was too heavy for the laptop).

| | Aircraft: plain CNN | Aircraft: SupCon ResNet-18 | Turbine: plain CNN | Turbine: SupCon ResNet-18 |
|---|---|---|---|---|
| Test crops / photos | 282 / 177 | 282 / 177 | 1,365 / 382 | 1,365 / 382 |
| Accuracy | 0.798 | 0.915 | 0.783 | 0.900 |
| Balanced accuracy | 0.800 | 0.900 | 0.858 | 0.882 |
| Macro-F1 (95% interval) | 0.793 (0.731 to 0.838) | 0.905 (0.840 to 0.948) | 0.780 (0.615 to 0.804) | 0.886 (0.853 to 0.910) |
| Worst class recall | 0.791 (dent) | 0.859 (crack) | 0.577 (paint peeling) | 0.765 (cracking) |
| Train minus test macro-F1 | +0.065 | +0.095 | -0.005 | +0.101 |
| Macro-F1 kept under corruption | 91% | 96% | 79% | 88% |
| Weakest condition (severe) | low contrast 0.389 | JPEG 0.589 | sensor noise 0.111 | sensor noise 0.169 |
| Balanced accuracy with defect hidden (chance) | 0.487 (0.333) | 0.620 (0.333) | 0.647 (0.167) | 0.430 (0.167) |
| Context reliance | 33% | 51% | 70% | 37% |
| Grad-CAM focus on defect | 1.47x | 1.69x | 1.46x | 1.76x |
| Calibration error | 0.039 | 0.038 | 0.027 | 0.046 |

## Reading

- SupCon + cRT is ahead on accuracy, macro-F1 and robustness on both datasets. On turbine the balanced-accuracy gap (0.882 vs 0.858) is within the intervals.
- SupCon starts from ImageNet weights and the plain CNN from scratch; the control run below separates the two effects on aircraft.
- Shortcut reliance is not consistently lower for either model: SupCon is worse on aircraft, the plain CNN is worse on turbine. Both stay well above chance with the defect painted over.
- Cracks are SupCon's weakest class on both datasets (recall 0.859 and 0.765).
- On turbine, both models collapse under heavy sensor noise, and the plain CNN underfits (train score equals test score).
- No "no damage" class yet, so these numbers measure telling defects apart, not finding them. The hide-the-defect test also leaks box shape, so treat it as a comparison between models.

## Control run on aircraft: pretrained ResNet-18 with plain cross-entropy

Same backbone and pretraining as the SupCon model, same loss and recipe as the plain CNN (`train_baseline.py --arch resnet18 --lr 1e-4`, early stop at epoch 26).

| Aircraft | Plain CNN | ResNet-18 cross-entropy | SupCon ResNet-18 |
|---|---|---|---|
| Macro-F1 (95% interval) | 0.793 (0.731 to 0.838) | 0.872 (0.812 to 0.915) | 0.905 (0.840 to 0.948) |
| Balanced accuracy | 0.800 | 0.870 | 0.900 |
| Worst class recall | 0.791 (dent) | 0.814 (corrosion) | 0.859 (crack) |
| Train minus test macro-F1 | +0.065 | +0.122 | +0.095 |
| Macro-F1 under corruption | 0.723 (91% kept) | 0.817 (94% kept) | 0.867 (96% kept) |
| Context reliance | 33% | 55% | 51% |
| Grad-CAM focus on defect | 1.47x | 1.66x | 1.69x |
| Calibration error | 0.039 | 0.065 | 0.038 |

- About 8 of the 11 macro-F1 points between the plain CNN and SupCon come from ImageNet pretraining. The contrastive step adds about 3 more, which is within the intervals on 282 test crops.
- SupCon is ahead of the control on every row, but by small margins; the clearest are robustness under corruption and calibration.
- The higher context reliance on aircraft arrives with the pretrained ResNet, not with the contrastive loss.

## Not yet run

ResNet-18 cross-entropy control on turbine, `data_audit.py`, held-out photo source, cross-dataset test, `no_damage` class.
