# Family-balanced validation for temperature scaling

Same hybrid probabilities, OOD alpha 0.05, CP alpha 0.10 and family-weighted LAC. Only the temperature fitting objective changes. Both temperatures fitted on all validation crops, using identical optimizer and bounds. Refit CP cutoff separately on the same accepted calibration crops. Test data used only for evaluation. No deployment files changed.

Validation: 4818 crops / 178 families. Accepted calibration: 2065 crops. Test: 2752 crops (2711 accepted).

| Metric | Crop-weighted temperature | Family-weighted temperature |
|---|---:|---:|
| Temperature | 1.056212 | 1.089562 |
| All test: crop_nll | 0.708536 | 0.708974 |
| All test: family_nll | 0.752950 | 0.752456 |
| All test: crop_ece_15_bins | 0.039850 | 0.040037 |
| All test: family_ece_15_bins | 0.032209 | 0.026740 |
| Accepted: coverage | 90.9627% | 90.9627% |
| Accepted: family_coverage | 90.6738% | 90.7174% |
| Accepted: macro_coverage | 88.2543% | 88.2543% |
| Accepted: singleton_rate | 69.4209% | 69.2733% |
| Accepted: singleton_accuracy | 90.4888% | 90.5751% |
| Accepted: mean set size | 1.521579 | 1.518259 |
| silent_miss | 0.3310% | 0.3310% |
| no_defect_false_alarm | 27.7612% | 28.0597% |
| human_attention | 31.6134% | 31.7587% |

Paired comparison:
```json
{
  "accepted_sets_changed": 17,
  "accepted_newly_covered": 1,
  "accepted_newly_missed": 1,
  "visor_actions_changed": 6,
  "family_minus_crop_coverage_difference_ci95": [
    -0.0011140510616147684,
    0.0011265595847934785
  ]
}
```

ECE uses 15 equal-width confidence bins. Family-weighted metrics assign each observed family equal total weight. The paired coverage interval resamples test families conditional on this fitted pipeline; it does not include training/calibration uncertainty. Test comparisons are exploratory, not proof of superiority. OOD FRR/FAR cannot change because neither OOD detector uses temperature.

Run from repository root: `python 1_main_model_resnet18/enhanced_res/arch_a/experiments/archB_family_temperature.py`.
