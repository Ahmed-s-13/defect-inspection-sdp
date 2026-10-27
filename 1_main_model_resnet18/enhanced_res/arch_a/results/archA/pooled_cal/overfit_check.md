# Overfitting check (enhanced_res v2, 27 October 2026)

> Run while pooling was being tried, with OOD α 0.06 and the quality gate at 0.10. Since then pooling has been rolled back, the OOD α is 0.08 and the quality α is 0.05. See [`FINDINGS.md`](FINDINGS.md).

## 1. OOD α chosen without looking at the test families

200 random halvings of the 266 val + cal-half families: calibrate on one half, measure the gates' false reject on the other. (Val crops can't be judged against a quality reference built from themselves without bias: they look slightly more typical, so this check slightly understates quality false rejects on the val families.)

| OOD α | families stopped, val + cal halves (mean) | test half (judged once) |
|---:|---:|---:|
| 0.04 | 15.0% | 13.5% |
| 0.05 | 15.8% | 13.7% |
| 0.06 | 16.7% | 14.0% |
| 0.07 | 17.5% | 15.3% |
| 0.08 | 18.4% | 15.7% |

Largest α within the 15% budget using val + cal only: **0.04** (deployed: 0.06).

## 2. Coverage per class on the test half (target 90%; not guaranteed per class)

| class | crops | cal half alone (q 0.958) | pooled (q 0.952) |
|---|---:|---:|---:|
| corrosion | 257 | 84.4% | 82.5% |
| crack | 636 | 87.3% | 86.5% |
| dent | 1129 | 96.3% | 95.7% |
| missing-head | 132 | 97.0% | 97.0% |
| paint-peel-off | 142 | 84.5% | 84.5% |
| scratch | 121 | 86.0% | 82.6% |

## 3. How much moved at the threshold

Test-half crops whose true class was inside the set with q 0.958 but drops out with q 0.952: **0.9%** of crops. That is the whole coverage change from pooling.

