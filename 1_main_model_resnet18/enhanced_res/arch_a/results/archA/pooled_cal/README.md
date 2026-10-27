# Pooling val into architecture A's calibration (enhanced_res v2)

Families: cal half 88, val 178, train 496; test half 87. Nothing retrained. `today`: everything calibrated on the cal half. `pooled`: CP and the OOD gate on val + cal half, the quality gate on cal half + train (reference: val). `val only`: CP and OOD on val alone, to see the bias from val having picked the model and fitted T.

## 1. The fixed test half (87 families)

| | today | pooled | val only | pooled, OOD α 0.04 | pooled, OOD α 0.05 | pooled, OOD α 0.06 | pooled, OOD α 0.07 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CP threshold q | 0.958 | 0.952 | 0.952 | 0.952 | 0.952 | 0.952 | 0.952 |
| CP coverage per crop / per family (target 90%) | 91.5% / 91.7% | 90.6% / 90.2% | 90.8% / 91.0% | 90.6% / 90.2% | 90.6% / 90.2% | 90.6% / 90.2% | 90.6% / 90.2% |
| mean set size / single-class answers | 1.65 / 65.2% | 1.54 / 68.2% | 1.55 / 67.8% | 1.54 / 68.2% | 1.54 / 68.2% | 1.54 / 68.2% | 1.54 / 68.2% |
| normal families stopped by the gates | 13.7% | 15.7% | 15.3% | 13.5% | 13.7% | 14.0% | 15.3% |
| re-take / re-aim (crops) | 17.4% / 2.4% | 17.5% / 3.6% | 17.0% / 3.5% | 18.2% / 1.4% | 18.1% / 1.6% | 18.0% / 2.1% | 17.7% / 2.7% |
| inspector needed | 49.0% | 47.3% | 47.3% | 46.7% | 46.7% | 46.9% | 47.1% |
| single wrong class shown without warning | 6.5% | 6.7% | 6.7% | 6.9% | 6.9% | 6.8% | 6.8% |
| D5 stopped (normal defects, other photos) | 14.4% | 14.9% | 14.5% | 14.8% | 14.8% | 14.8% | 14.9% |
| off-target photos passing | 0.9% | 0.1% | 0.1% | 3.6% | 2.4% | 0.9% | 0.3% |
| wallpapers / static passing | 10.8% / 0.0% | 8.2% / 0.0% | 8.5% / 0.0% | 16.2% / 0.0% | 13.8% / 0.0% | 11.2% / 0.0% | 10.0% / 0.0% |

## 2. 200 random re-splits of the 175 test families (88 cal / 87 test)

How much the numbers move from one draw of calibration families to the next. `pooled` adds val (and train, for the quality gate) to every draw.

| | today: mean (5th–95th pct) | pooled: mean (5th–95th pct) |
|---|---:|---:|
| CP coverage per crop | 91.8% (87.1%–95.8%) | 91.3% (87.8%–94.5%) |
| CP coverage per family | 90.8% (85.5%–95.1%) | 90.2% (87.2%–93.3%) |
| mean set size | 1.67 (1.44–1.98) | 1.56 (1.47–1.67) |
| normal families stopped | 15.0% (9.1%–20.9%) | 16.4% (11.9%–20.5%) |

### OOD α sweep with pooling: normal families stopped by the gates, mean (5th–95th pct) over the draws

| OOD α | normal families stopped | CP coverage per family |
|---:|---:|---:|
| 0.04 | 13.4% (9.6%–16.8%) | 90.2% |
| 0.05 | 13.9% (10.1%–17.3%) | 90.2% |
| 0.06 | 14.7% (10.6%–18.8%) | 90.2% |
| 0.07 | 15.7% (11.1%–19.7%) | 90.2% |
| 0.08 | 16.4% (11.9%–20.5%) | 90.2% |
| draws with per-crop coverage below 90% | 25.5% | 25.5% |
