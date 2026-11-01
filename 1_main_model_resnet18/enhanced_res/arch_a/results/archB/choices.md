# Architecture B: choices made inside the cal half (test half not used)

> Historical experiment: measured before the OOD change, using OOD α 0.08 where applicable (the threshold sweep lists its own values). Current B uses OOD α 0.05 and CP α 0.10. These recorded numbers are not current-deployment results; see the regenerated [Architecture B report](README.md).

Model gc7, crops d1_gc7_fix, quality gate off. 200 random halvings of the 88 cal-half families (fit on 44, measure on the other 44); mean (5th–95th percentile).

## 1. OOD α

| OOD α | defect crops stopped (per family) | no-defect crops stopped (per family) | off-target photos (AoN trained on these) passing | wallpaper patches passing | static passing |
|---:|---:|---:|---:|---:|---:|
| 0.05 | 1.4% (0.0–3.7%) | 1.2% (0.0–4.0%) | 1.6% | 29.8% | 21.2% |
| 0.08 | 5.0% (0.9–10.9%) | 3.6% (0.0–7.5%) | 0.1% | 19.5% | 12.5% |
| 0.10 | 6.8% (1.6–12.5%) | 4.9% (1.3–9.2%) | 0.0% | 13.8% | 6.9% |
| 0.13 | 9.0% (3.0–14.7%) | 6.7% (1.7–14.2%) | 0.0% | 7.5% | 3.8% |
| 0.15 | 10.5% (5.0–17.4%) | 8.4% (2.4–16.1%) | 0.0% | 5.8% | 3.4% |

## 2. Miss guard (OOD α 0.08, CP α 0.1)

Silent miss = a defect crop shown as nothing. A lone no-defect with the guard failing is shown as "POSSIBLE …: check" instead. With 44 calibration families, α_miss below about 1/45 can't be certified.

| miss guard | silent miss | no-defect shown nothing |
|---|---:|---:|
| none (plain LAC) | 0.5% (0.1–1.0%) | 71.6% (62.5–79.8%) |
| α_miss 0.02 | 0.0% (0.0–0.0%) | 0.0% (0.0–0.0%) |
| α_miss 0.05 | 0.5% (0.1–1.0%) | 71.6% (62.5–79.8%) |
| α_miss 0.10 | 0.5% (0.1–1.0%) | 71.6% (62.5–79.8%) |

## 3. Damaged images on the silent path (deployed B, no quality gate)

428 cal-half defect crops (≤ 6 per family) and 14 synthetic corruptions of them: share shown as nothing (a silent miss).

| set | shown nothing |
|---|---:|
| clean | 0.9% |
| blur s1 | 3.3% |
| blur s2 | 3.5% |
| motion blur s1 | 2.3% |
| motion blur s2 | 3.3% |
| dark s1 | 4.2% |
| dark s2 | 0.9% |
| overexposed s1 | 0.7% |
| overexposed s2 | 0.0% |
| glare s1 | 2.1% |
| glare s2 | 1.6% |
| noise s1 | 0.0% |
| noise s2 | 0.0% |
| jpeg s1 | 5.4% |
| jpeg s2 | 4.9% |
