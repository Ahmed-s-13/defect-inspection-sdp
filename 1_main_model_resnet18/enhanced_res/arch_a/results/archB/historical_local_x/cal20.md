# Train on 80%, calibrate on the held-out 20% of train (the paper's split) vs ours

> Historical experiment: measured before the OOD change, using OOD α 0.08 where applicable (the threshold sweep lists its own values). Current B uses OOD α 0.05 and CP α 0.10. These recorded numbers are not current-deployment results; see the regenerated [Architecture B report](../README.md).

d1_gc7_fix. The held-out 20% of train families = 8625 crops / 166 families. Judged on 44 cal-half families never used to calibrate, 200 halvings, mean (5th–95th). OOD α 0.08, CP α 0.1. Per-family weighted except set size, silent miss and false alarm (per crop).

| setup | T | cal families | accuracy | macro-F1 | coverage | set size | silent miss | no-defect false alarm | false reject |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gc7_fix (all train), cal = 44 cal-half families | 1.152 | 44 | 76.1% (72.4–80.3) | 71.8% (66.8–76.6) | 91.4% (85.4–96.6) | 1.968 | 0.4% (0.1–0.8) | 36.7% (27.0–47.0) | 2.1% (0.3–5.4) |
| x_cal20 (80% train), cal = held-out 20% of train | 1.102 | 166 | 75.9% (71.2–80.4) | 72.8% (67.2–77.5) | 93.4% (91.5–95.2) | 2.053 | 1.0% (0.4–1.5) | 21.0% (16.2–26.1) | 3.0% (1.4–4.5) |
| x_cal20 (80% train), cal = 20% of train + 44 cal-half families | 1.102 | 210 | 75.9% (71.2–80.4) | 72.8% (67.2–77.5) | 92.9% (90.5–95.1) | 1.975 | 0.9% (0.4–1.5) | 19.5% (14.4–24.3) | 4.0% (1.4–7.4) |
