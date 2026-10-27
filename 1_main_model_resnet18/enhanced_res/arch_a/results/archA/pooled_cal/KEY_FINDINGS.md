# Architecture A, finalised (v2): key findings, 27 Oct 2026

> **Decision (27 October 2026, later the same day): pooling was rolled back.** The user chose to calibrate on the cal half alone again and never to use the test half for any choice again. Architecture A is back to its earlier numbers: coverage 91.5% / 91.7%, set size 1.65, gates stop 13.7% of families, inspector needed 49.0%, silent wrong answers 6.5%. On 28 October the quality gate then moved to α 0.05 ([`../quality_alpha/`](../quality_alpha/README.md)). Current numbers: gates stop 6.7% of families, inspector needed 38.5%, silent wrong answers 6.7% ([`../offtarget/rates.md`](../offtarget/rates.md)).
>
> OOD α stays at **0.08**. It was first tuned on the test half. Re-checked without the test half (`experiments/archA_alpha_check.py`, [`../alpha_check/`](../alpha_check/README.md)), 500 halvings of the cal half stop 12.9% of normal families on average, within the 15% budget. Two caveats:
> - the union bound alone guarantees only 18%;
> - α 0.05 would guarantee 15%, at the cost of 4.8% instead of 0.9% of off-target photos passing.
>
> The sections below record what pooling did, which is why it was rolled back.


1. Calibration now uses 3x the photo families with no new data.
   - CP set and OOD gate: val + cal half (88 -> 266 families).
   - Quality gate: cal half + train (584 families).

2. The guarantee is steadier between calibration draws.
   - Per-family coverage: 87.2-93.3% (was 85.5-95.1%).
   - Mean set size: 1.47-1.67 (was 1.44-1.98).

3. Val bias is small. It costs about 0.7 points of coverage, which stays on target: 90.6% per crop, 90.2% per family.

4. OOD alpha 0.08 was over budget: it stopped 16.4% of normal families on average. The old 13.7% was a lucky test half.
   - New OOD alpha: 0.06, stopping 14.7% on average (14.0% on the test half).
   - Off-target catch is unchanged: 0.9% of off-target photos and 0% of static pass; the hard-hat photo is still re-aim.

5. Visor: sets are smaller (1.54 vs 1.65) and more answers are a single class (68% vs 65%).
   - Inspector needed: 46.9% (was 49.0%).
   - Silent wrong answers: 6.8% (was 6.5%), because coverage now sits closer to 90%.

6. The quality gate and the OOD gate run in parallel. Re-aim wins if both fire. The docs now say so.

7. gc7 (architecture B, next) still uses the old gates. It needs the no-defect crops from the team Drive before recalibrating.
   - Watch: the quality gate calls 22% of clean, plain panels "blurry".

8. Still the biggest gap: no helmet-camera calibration data. Every guarantee holds for D1-style photos only.

9. Overfitting audit. The model, T and the regression are not overfit. Two things are:
   - OOD alpha 0.06 (and 0.08 before it) was chosen while looking at the test families. Chosen from val + cal only, it would be 0.04: 3.6% of off-target photos pass instead of 0.9%.
   - The test half has been reused for many choices, so its numbers are mildly optimistic.

10. Worse with pooling: rare-class coverage fell (scratch 86.0% -> 82.6%, corrosion 84.4% -> 82.5%), silent wrong answers rose 6.5% -> 6.8%, and D5 dent coverage fell 87.2% -> 85.0%.
