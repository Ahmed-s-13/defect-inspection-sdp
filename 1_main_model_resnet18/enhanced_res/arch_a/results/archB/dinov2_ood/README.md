# DINOv2 OOD gates in Architecture B

Architecture B, quality off, fixed OOD alpha 0.08; family-weighted p-values; Bonferroni for two-score gates. No test tuning. FAR means gate passage, not visor defect display. AoN trained on fit categories only; role=test excluded. Deployed reference saw more off-target categories and is not a held-out comparison. D6 rejection need not be a false rejection of usable skin.

Frozen pretrained vit_small_patch14_dinov2.lvd142m, 224 pixels, 384 features, no classification head. k=5. AoN logistic regression C=0.5, balanced; all D1 training features used.

Calibration: 2116 crops / 88 families. Test: 2752 crops / 87 families. D5: 1008 crops. Held-out off-target: 2172 crops.

| Gate | D1 FRR | D1 family FRR | D5 FRR | Held-out FAR | FAR centre / box | D6 skin rejection | D6 whole rejection |
|---|---:|---:|---:|---:|---:|---:|---:|
| ResNet AoN (fit-only) | 3.38% | 7.95% | 0.40% | 2.12% | 1.01% / 3.22% | 74.60% | 99.80% |
| DINOv2 AoN | 4.03% | 7.23% | 5.26% | 0.37% | 0.00% / 0.74% | 96.90% | 100.00% |
| DINOv2 kNN | 7.96% | 8.63% | 1.29% | 2.07% | 1.29% / 2.85% | 91.00% | 99.20% |
| ResNet kNN | 4.14% | 6.53% | 0.89% | 49.95% | 51.75% / 48.16% | 33.40% | 26.60% |
| ResNet AoN + kNN (fit-only) | 2.98% | 7.31% | 0.30% | 3.41% | 1.93% / 4.88% | 65.10% | 98.20% |
| DINOv2 AoN + kNN | 5.23% | 7.29% | 2.38% | 0.41% | 0.00% / 0.83% | 96.80% | 100.00% |
| Deployed B (reference only) | 2.87% | 6.86% | 0.30% | 0.05% | 0.00% / 0.09% | 80.10% | 99.20% |

FAR here is pooled over crops; JSON also records per-category rates and their equal-category average. Older Architecture A tables averaged categories, so their FARs are not directly comparable. D1 intervals in JSON resample evaluation families, conditional on the fitted calibration. This experiment changes only the OOD gate: it does not recalibrate LAC or measure downstream CP/visor outcomes. No deployed model or calibration was changed.

Run: `1_main_model_resnet18/experiments/d1_sscd_grouped/.venv/bin/python 1_main_model_resnet18/enhanced_res/arch_a/experiments/archB_dinov2_ood.py`
