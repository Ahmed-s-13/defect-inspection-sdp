# OOD cutoff changed from 0.08 to 0.05

> Historical OOD-only change, before family-weighted temperature fitting. Its pipeline figures use the former crop-weighted T = 1.056212; current figures are in [the Architecture B report](../README.md). OOD-only FAR/FRR remain valid.

User-directed operating-point change. CP alpha stays 0.10. The trained ResNet, MLP, kNN bank, AoN and temperature are unchanged. LAC cutoffs were refitted on 2,065 gate-passed calibration crops (previously 2,014); no test crops fitted these cutoffs.

| Metric | Previous OOD 0.08 | OOD 0.05 at this step |
|---|---:|---:|
| D1 crop false rejection | 2.87% | 1.49% |
| D1 family-average false rejection | 6.86% | 4.61% |
| Overall CP coverage | 90.37% | 91.02% |
| Family-average CP coverage | 89.55% | 90.59% |
| Defect silent misses | 0.33% | 0.33% |
| Clean crops with any message | 26.87% | 27.76% |
| Human referral | 29.98% | 31.61% |
| Mean candidate set size | 1.4818 | 1.5400 |

Current external results:

| Protocol | D1 FRR | D5 FRR | Off-target FAR |
|---|---:|---:|---:|
| Deployed B (reference only) | 1.49% | 0.30% | 1.47% |
| ResNet AoN + kNN (fit-only) | 1.42% | 0.30% | 6.54% |

The deployed reference has training exposure to benchmark off-target categories. Its 1.47% FAR must not be presented as held-out generalization. The controlled fit-only comparison excludes evaluation categories from training and gives 6.54% FAR (previously 3.41%) and 1.42% D1 FRR (previously 2.98%). FAR is OOD-gate passage, not visor defect display. The detailed [current OOD comparison](../dinov2_ood/README.md) and [pipeline report](../README.md) use the new threshold.

The old calibration is preserved in calibration_ood_0.08.json for reproducibility. Older local_x and runtime-device studies remain historical and are labeled with their original settings; no claim is made that those historical measurements were rerun.

## Why k stays 5

Both the enhanced_res class-probability branch and the OOD distance branch use five nearest training crops. build_head.py fixes k=5 and tunes the hybrid blend weight; it does not establish five as optimal for gc7. Historical architecture-A OOD experiments tested k=1 and k=20, with mixed trade-offs and no universal winner. A new grouped validation sweep could justify a different k, independently for classification and OOD. Seven classes do not imply six neighbours.
