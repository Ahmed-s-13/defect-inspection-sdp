# Family-weighted scores vs family-mean scores

Same gc7 model, deployed temperature, OOD gate, alpha 0.10, and top-class fallback. Baseline: weighted quantile of individual scores, each family total weight 1. Alternative: arithmetic mean of accepted true-label scores within each family, then the ceil((G+1)*0.90)-th sorted family mean. Apply each resulting threshold to individual test crops. Calibration only fits thresholds; test half is report-only. Mean-score calibration is an experimental heuristic for crop-level prediction, not a demonstrated crop-level coverage guarantee.

Calibration: 2116 crops (2014 accepted), 88 families. Evaluation: 2752 crops (2673 accepted), 87 families.

| Metric | Family-weighted | Family mean |
|---|---:|---:|
| Probability cutoff | 6.6950% | 34.7005% |
| Silent defect miss | 0.3310% | 2.3997% |
| Clean crop false alarm | 26.8657% | 12.8358% |
| Human referral | 29.9782% | 5.9230% |
| Accepted: coverage | 0.903853 | 0.815937 |
| Accepted: macro_coverage | 0.875463 | 0.779798 |
| Accepted: family_coverage | 0.895998 | 0.810174 |
| Accepted: mean_set_size | 1.444444 | 1.031425 |
| Accepted: singleton_rate | 0.720913 | 0.968575 |
| Accepted: mean_confidence | 0.813498 | 0.813498 |

Paired comparison:

```json
{
  "accepted_sets_changed": 682,
  "accepted_newly_covered": 0,
  "accepted_newly_missed": 235,
  "visor_actions_changed": 663,
  "mean_minus_weighted_accepted_coverage_ci95": [
    -0.1252322866217713,
    -0.062020111783031
  ]
}
```

Intervals resample whole evaluation families, conditional on this calibration. This is one paired evaluation, not evidence for model selection or universal superiority. Deployed calibration is unchanged.

## Literature and interpretation

Lee, Barber and Willett, Distribution-free inference with hierarchical data, equation (6) and Theorem 1: https://arxiv.org/html/2306.06342v4#S2.SS2

The individual-score weighting and G+1 correction in runtime.family_threshold match the HCP quantile construction (verified independently here). HCP has a marginal guarantee under hierarchical exchangeability: exchangeable groups and exchangeable observations within groups, with a new observation from a new group. The mathematical construction supports the weighting; this does not establish those assumptions for D1 augmentation families or validate the complete OOD-selected pipeline. Our OOD reference scores are fitted on the same calibration half used for selection, so selection is not simply an independently fixed filter.

Averaging scores first changes the calibrated quantity to a family mean. Applying that quantile to individual crop scores does not inherit a guarantee for those individual scores. Means conceal within-family high-score tails. The empirical comparison keeps the existing top-class fallback in both variants.
