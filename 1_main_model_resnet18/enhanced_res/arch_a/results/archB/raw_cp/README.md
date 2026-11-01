# Raw hybrid vs temperature-scaled LAC

Same gc7 weights, prior correction, OOD gate, alpha 0.10, family weights, and top-class fallback. Refit q separately on OOD-passed calibration half only; T is deployed val-fitted T or 1. Test half is report-only.

Calibration: 2116 crops (2065 accepted), 88 families. Evaluation: 2752 crops (2711 accepted), 87 families.

| Metric | Temperature-scaled | Raw hybrid |
|---|---:|---:|
| Probability cutoff | 6.2640% | 5.4407% |
| Silent defect miss | 0.3310% | 0.3310% |
| Clean crop false alarm | 28.0597% | 27.4627% |
| Human referral | 31.7587% | 31.0320% |
| Accepted: coverage | 0.909627 | 0.908890 |
| Accepted: macro_coverage | 0.882543 | 0.882290 |
| Accepted: family_coverage | 0.907174 | 0.906204 |
| Accepted: mean_set_size | 1.518259 | 1.513833 |
| Accepted: singleton_rate | 0.692733 | 0.700111 |
| Accepted: mean_confidence | 0.799090 | 0.829513 |

Paired comparison:

```json
{
  "accepted_sets_changed": 42,
  "accepted_newly_covered": 1,
  "accepted_newly_missed": 3,
  "visor_actions_changed": 23,
  "raw_minus_scaled_accepted_coverage_ci95": [
    -0.002960406353591903,
    0.0008302545288846653
  ]
}
```

Intervals resample whole evaluation families, conditional on this calibration. This is one paired evaluation, not evidence for model selection or universal superiority. Deployed calibration is unchanged.
