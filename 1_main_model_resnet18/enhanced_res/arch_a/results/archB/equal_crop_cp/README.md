# Family-weighted vs equal-crop LAC

Same gc7 weights, prior correction, deployed temperature, OOD gate, alpha 0.10, and top-class fallback. Baseline uses family-weighted individual scores. Alternative gives every accepted calibration crop weight 1 and uses the ceil((N+1)*0.90)-th sorted true-label score. Thresholds are fitted on the same OOD-passed calibration half only. Test half is report-only. The equal-crop procedure is standard LAC arithmetic, but crop exchangeability is not established for this grouped dataset.

Calibration: 2116 crops (2065 accepted), 88 families. Evaluation: 2752 crops (2711 accepted), 87 families.

| Metric | Family-weighted | Equal crop |
|---|---:|---:|
| Probability cutoff | 6.2640% | 10.7696% |
| Silent defect miss | 0.3310% | 0.4965% |
| Clean crop false alarm | 28.0597% | 24.4776% |
| Human referral | 31.7587% | 24.0552% |
| Accepted: coverage | 0.909627 | 0.882700 |
| Accepted: macro_coverage | 0.882543 | 0.850332 |
| Accepted: family_coverage | 0.907174 | 0.874988 |
| Accepted: mean_set_size | 1.518259 | 1.309849 |
| Accepted: singleton_rate | 0.692733 | 0.770933 |
| Accepted: mean_confidence | 0.799090 | 0.799090 |

Paired comparison:

```json
{
  "accepted_sets_changed": 443,
  "accepted_newly_covered": 0,
  "accepted_newly_missed": 73,
  "visor_actions_changed": 243,
  "equal_crop_minus_family_accepted_coverage_ci95": [
    -0.04247592645472428,
    -0.015607273388790584
  ]
}
```

Intervals resample whole evaluation families, conditional on this calibration. This is one paired evaluation, not evidence for model selection or universal superiority. Deployed calibration is unchanged.
