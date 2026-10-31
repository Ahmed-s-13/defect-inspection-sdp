# Family-weighted vs equal-crop LAC

Same gc7 weights, prior correction, deployed temperature, OOD gate, alpha 0.10, and top-class fallback. Baseline uses family-weighted individual scores. Alternative gives every accepted calibration crop weight 1 and uses the ceil((N+1)*0.90)-th sorted true-label score. Thresholds are fitted on the same OOD-passed calibration half only. Test half is report-only. The equal-crop procedure is standard LAC arithmetic, but crop exchangeability is not established for this grouped dataset.

Calibration: 2116 crops (2014 accepted), 88 families. Evaluation: 2752 crops (2673 accepted), 87 families.

| Metric | Family-weighted | Equal crop |
|---|---:|---:|
| Probability cutoff | 6.6950% | 12.1707% |
| Silent defect miss | 0.3310% | 0.4965% |
| Clean crop false alarm | 26.8657% | 20.8955% |
| Human referral | 29.9782% | 22.0203% |
| Accepted: coverage | 0.903853 | 0.876169 |
| Accepted: macro_coverage | 0.875463 | 0.843749 |
| Accepted: family_coverage | 0.895998 | 0.866959 |
| Accepted: mean_set_size | 1.444444 | 1.250655 |
| Accepted: singleton_rate | 0.720913 | 0.802843 |
| Accepted: mean_confidence | 0.813498 | 0.813498 |

Paired comparison:

```json
{
  "accepted_sets_changed": 407,
  "accepted_newly_covered": 0,
  "accepted_newly_missed": 74,
  "visor_actions_changed": 239,
  "equal_crop_minus_family_accepted_coverage_ci95": [
    -0.041842696921821555,
    -0.01722120976059644
  ]
}
```

Intervals resample whole evaluation families, conditional on this calibration. This is one paired evaluation, not evidence for model selection or universal superiority. Deployed calibration is unchanged.
