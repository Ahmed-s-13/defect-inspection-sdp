# Raw hybrid vs temperature-scaled LAC

Same gc7 weights, prior correction, OOD gate, alpha 0.10, family weights, and top-class fallback. Refit q separately on OOD-passed calibration half only; T is deployed val-fitted T or 1. Test half is report-only.

Calibration: 2116 crops (2014 accepted), 88 families. Evaluation: 2752 crops (2673 accepted), 87 families.

| Metric | Temperature-scaled | Raw hybrid |
|---|---:|---:|
| Probability cutoff | 6.6950% | 6.0159% |
| Silent defect miss | 0.3310% | 0.3724% |
| Clean crop false alarm | 26.8657% | 26.8657% |
| Human referral | 29.9782% | 29.9055% |
| Accepted: coverage | 0.903853 | 0.903853 |
| Accepted: macro_coverage | 0.875463 | 0.874871 |
| Accepted: family_coverage | 0.895998 | 0.895181 |
| Accepted: mean_set_size | 1.444444 | 1.448934 |
| Accepted: singleton_rate | 0.720913 | 0.721661 |
| Accepted: mean_confidence | 0.813498 | 0.832431 |

Paired comparison:

```json
{
  "accepted_sets_changed": 16,
  "accepted_newly_covered": 1,
  "accepted_newly_missed": 1,
  "visor_actions_changed": 4,
  "raw_minus_scaled_accepted_coverage_ci95": [
    -0.0011583908795372777,
    0.0010711035350906486
  ]
}
```

Intervals resample whole evaluation families, conditional on this calibration. This is one paired evaluation, not evidence for model selection or universal superiority. Deployed calibration is unchanged.
