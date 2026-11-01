# Calibrators for architecture B (local experiment): fitted on val, judged on the cal half

> Historical experiment: measured before the OOD change, using OOD α 0.08 where applicable (the threshold sweep lists its own values). Current B uses OOD α 0.05 and CP α 0.10. These recorded numbers are not current-deployment results; see the regenerated [Architecture B report](../README.md).

gc7 on d1_gc7_fix; val 4818 crops, cal half 2116 crops / 88 families (every family weight 1). B's T = 1.056; vector temperatures corrosion 0.97, crack 1.69, dent 1.48, missing-head 0.84, paint-peel-off 0.80, scratch 0.82, no-defect 0.75. Lower is better for every metric; mid-range gap = mean shown confidence − accuracy on crops at 40–80% (positive = over-confident). [Δ vs temperature, 95% family-bootstrap interval].

| calibrator | NLL | Brier | ECE | class-wise ECE | mid-range gap |
|---|---:|---:|---:|---:|---:|
| none | 0.7568 [-0.0027, +0.0134] | 0.3386 [+0.0001, +0.0047] | 0.0596 [-0.0145, +0.0284] | 0.0332 [-0.0055, +0.0002] | 0.0634 [-0.0366, +0.0294] |
| temperature (B) | 0.7519 | 0.3364 | 0.0460 | 0.0358 | 0.0572 |
| vector | 0.7216 [-0.0881, +0.0249] | 0.3317 [-0.0172, +0.0073] | 0.0459 [-0.0268, +0.0248] | 0.0324 [-0.0118, +0.0033] | 0.0516 [-0.0987, +0.0588] |
| vector+bias | 0.7292 [-0.0963, +0.0494] | 0.3345 [-0.0250, +0.0236] | 0.0477 [-0.0281, +0.0272] | 0.0331 [-0.0108, +0.0059] | 0.0952 [-0.0453, +0.1374] |
| isotonic | 0.7072 [-0.1089, +0.0053] | 0.3255 [-0.0297, +0.0067] | 0.0371 [-0.0285, +0.0239] | 0.0268 [-0.0142, +0.0010] | 0.0525 [-0.0741, +0.0705] |

## CP at α 0.1 with each calibrator (200 halvings of the cal half; mean)

| calibrator | family coverage | mean set size | Δ set size vs temperature (5th–95th over halvings) | single class |
|---|---:|---:|---:|---:|
| none | 91.5% | 1.939 | -0.012 to +0.043 | 56.5% |
| temperature (B) | 91.5% | 1.928 | +0.000 to +0.000 | 56.2% |
| vector | 91.6% | 1.852 | -0.846 to +0.432 | 41.7% |
| vector+bias | 91.5% | 1.887 | -0.754 to +0.515 | 42.4% |
| isotonic | 91.5% | 1.691 | -0.937 to +0.159 | 57.0% |

## Reliability of the shown percentage (top class), cal half, per-family weighted

| shown confidence | crops | temperature (B) acc − conf | vector acc − conf | isotonic acc − conf |
|---|---:|---:|---:|---:|
| 0%–40% | 72 | -5.3 pts (72) | +8.5 pts (43) | +8.8 pts (74) |
| 40%–50% | 102 | +1.4 pts (102) | -3.3 pts (108) | -2.3 pts (122) |
| 50%–60% | 120 | -10.0 pts (120) | -8.1 pts (122) | -7.1 pts (151) |
| 60%–70% | 148 | -1.6 pts (148) | -14.2 pts (141) | -0.9 pts (167) |
| 70%–80% | 190 | -9.1 pts (190) | -0.8 pts (186) | -10.6 pts (152) |
| 80%–90% | 891 | -7.6 pts (891) | -9.2 pts (683) | -2.4 pts (297) |
| 90%–100% | 593 | +1.8 pts (593) | +2.6 pts (833) | -0.0 pts (1153) |
