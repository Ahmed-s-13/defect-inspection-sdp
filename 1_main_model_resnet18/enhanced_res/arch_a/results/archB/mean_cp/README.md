# Family-weighted scores vs family-mean scores

Same gc7 model, deployed temperature, OOD gate, alpha 0.10, and top-class fallback. Baseline: weighted quantile of individual scores, each family total weight 1. Alternative: arithmetic mean of accepted true-label scores within each family, then the ceil((G+1)*0.90)-th sorted family mean. Apply each resulting threshold to individual test crops. Calibration only fits thresholds; test half is report-only. Mean-score calibration is an experimental heuristic for crop-level prediction, not a demonstrated crop-level coverage guarantee.

Calibration: 2116 crops (2065 accepted), 88 families. Evaluation: 2752 crops (2711 accepted), 87 families.

| Metric | Family-weighted | Family mean |
|---|---:|---:|
| Probability cutoff | 6.2640% | 31.7582% |
| Silent defect miss | 0.3310% | 2.4824% |
| Clean crop false alarm | 28.0597% | 12.2388% |
| Human referral | 31.7587% | 6.1410% |
| Accepted: coverage | 0.909627 | 0.817042 |
| Accepted: macro_coverage | 0.882543 | 0.782918 |
| Accepted: family_coverage | 0.907174 | 0.805901 |
| Accepted: mean_set_size | 1.518259 | 1.047215 |
| Accepted: singleton_rate | 0.692733 | 0.952785 |
| Accepted: mean_confidence | 0.799090 | 0.799090 |

Paired comparison:

```json
{
  "accepted_sets_changed": 756,
  "accepted_newly_covered": 0,
  "accepted_newly_missed": 251,
  "visor_actions_changed": 711,
  "mean_minus_weighted_accepted_coverage_ci95": [
    -0.13152274146913176,
    -0.06531205759734071
  ]
}
```

Intervals resample whole evaluation families, conditional on this calibration. This is one paired evaluation, not evidence for model selection or universal superiority. Deployed calibration is unchanged.

## Literature and interpretation

Lee, Barber and Willett, Distribution-free inference with hierarchical data, equation (6) and Theorem 1: https://arxiv.org/html/2306.06342v4#S2.SS2

The individual-score weighting and G+1 correction in runtime.family_threshold match the HCP quantile construction (verified independently here). HCP has a marginal guarantee under hierarchical exchangeability: exchangeable groups and exchangeable observations within groups, with a new observation from a new group. The mathematical construction supports the weighting; this does not establish those assumptions for D1 augmentation families or validate the complete OOD-selected pipeline. Our OOD reference scores are fitted on the same calibration half used for selection, so selection is not simply an independently fixed filter.

Averaging scores first changes the calibrated quantity to a family mean. Applying that quantile to individual crop scores does not inherit a guarantee for those individual scores. Means conceal within-family high-score tails. The empirical comparison keeps the existing top-class fallback in both variants.
