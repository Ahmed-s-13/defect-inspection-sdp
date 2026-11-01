# DINOv2 OOD gates in Architecture B

Architecture B, quality off, fixed OOD alpha 0.05; family-weighted p-values; Bonferroni for two-score gates. No test tuning. FAR means gate passage, not visor defect display. AoN trained on fit categories only; role=test excluded. Deployed reference saw more off-target categories and is not a held-out comparison. D6 rejection need not be a false rejection of usable skin.

Frozen pretrained vit_small_patch14_dinov2.lvd142m, 224 pixels, 384 features, no classification head. k=5. AoN logistic regression C=0.5, balanced; all D1 training features used.

Calibration: 2116 crops / 88 families. Test: 2752 crops / 87 families. D5: 1008 crops. Held-out off-target: 2172 crops.

| Gate | D1 FRR | D1 family FRR | D5 FRR | Held-out FAR | FAR centre / box | D6 skin rejection | D6 whole rejection |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet AoN (fit-only) | 1.53% | 4.84% | 0.10% | 3.64% | 2.03% / 5.25% | 65.20% | 98.00% |
| DINOv2 AoN | 2.69% | 4.72% | 2.38% | 0.41% | 0.00% / 0.83% | 95.40% | 100.00% |
| DINOv2 kNN | 4.98% | 5.48% | 0.40% | 3.78% | 2.49% / 5.06% | 85.40% | 99.20% |
| ResNet kNN | 2.54% | 4.69% | 0.50% | 67.86% | 70.26% / 65.47% | 19.40% | 12.60% |
| ResNet AoN + kNN (fit-only) | 1.42% | 3.85% | 0.30% | 6.54% | 3.22% / 9.85% | 44.80% | 93.40% |
| DINOv2 AoN + kNN | 3.49% | 4.33% | 1.39% | 0.55% | 0.00% / 1.10% | 93.80% | 100.00% |
| Deployed B (reference only) | 1.49% | 4.61% | 0.30% | 1.47% | 0.46% / 2.49% | 71.90% | 97.60% |

## Threshold-independent ranking

OOD is positive. AUPRC is sklearn average_precision_score, as in the Outlier Exposure paper code (https://github.com/hendrycks/outlier-exposure/blob/master/utils/display_results.py). Gate ranking uses 1 minus its conformal p-value; raw ranking uses the standalone detector score before conformal calibration. Pooled D1 versus held-out off-target crops; prevalence differs from the paper, so these are not benchmark-comparable results.

| Gate | Gate AUPRC | Raw score AUPRC | Gate AUROC | Gate AUPRC at 1:5 | Raw AUPRC at 1:5 |
|---|---:|---:|---:|---:|---:|
| ResNet AoN (fit-only) | 99.63% | 99.64% | 99.67% | 98.90% | 98.92% |
| DINOv2 AoN | 99.38% | 99.92% | 99.72% | 97.65% | 99.75% |
| DINOv2 kNN | 98.43% | 99.26% | 99.06% | 94.56% | 97.76% |
| ResNet kNN | 87.68% | 87.74% | 92.14% | 66.27% | 66.42% |
| ResNet AoN + kNN (fit-only) | 99.52% | — | 99.59% | 98.46% | — |
| DINOv2 AoN + kNN | 98.60% | — | 99.41% | 94.74% | — |
| Deployed B (reference only) | 99.74% | — | 99.86% | 98.99% | — |

The 1:5 columns reweight OOD crops so total OOD weight is one fifth of ID weight, matching the paper's OOD prevalence without discarding observations. This does not match its datasets, per-dataset averaging, training method or repeated-run protocol. These columns are prevalence-adjusted estimates, not a reproduction of the paper.

FAR here is pooled over crops; JSON also records per-category rates and their equal-category average. Older Architecture A tables averaged categories, so their FARs are not directly comparable. D1 intervals in JSON resample evaluation families, conditional on the fitted calibration. This experiment changes only the OOD gate: it does not recalibrate LAC or measure downstream CP/visor outcomes. No deployed model or calibration was changed.

Run: `1_main_model_resnet18/experiments/d1_sscd_grouped/.venv/bin/python 1_main_model_resnet18/enhanced_res/arch_a/experiments/archB_dinov2_ood.py`
