# OOD α without the test half (enhanced_res v2)

> Run with the quality gate at α 0.10. Since 28 October the quality gate is at 0.05 ([`../quality_alpha/`](../quality_alpha/README.md)). That stops fewer normal families, and the union bound alone now caps the two gates at 0.05 + 0.08 = 13%.

The test half is not read. Rule: α_quality + α_OOD = the 15% false-reject budget (union bound, no tuning). Check 1: 500 random halvings of the 88 cal-half families (calibrate on 44, measure on the other 44). Check 2: the off-target sets and D5, gates calibrated on the whole cal half.

| OOD α | α_q + α_OOD | normal families stopped, cal-half halvings: mean (5th–95th pct) | off-target photos pass | wallpapers pass | static pass | D5 stopped |
|---:|---:|---:|---:|---:|---:|---:|
| 0.03 | 0.13 | 8.5% (1.9–16.3%) | 7.5% | 27.7% | 0.0% | 14.2% |
| 0.04 | 0.14 | 8.5% (1.9–16.3%) | 6.4% | 21.3% | 0.0% | 14.2% |
| 0.05 | 0.15 | 9.9% (2.8–17.8%) | 4.8% | 18.2% | 0.0% | 14.2% |
| 0.06 | 0.16 | 11.1% (3.6–19.5%) | 3.7% | 16.5% | 0.0% | 14.3% |
| 0.07 | 0.17 | 11.8% (4.5–20.6%) | 2.8% | 14.2% | 0.0% | 14.4% |
| 0.08 | 0.18 | 12.9% (5.5–21.4%) | 0.9% | 10.8% | 0.0% | 14.4% |
