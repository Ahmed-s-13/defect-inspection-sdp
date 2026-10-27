# Architecture A, finalised on enhanced_res v2: all findings (27 October 2026)

> **Decision (27 October 2026, later the same day): pooling was rolled back.** The user chose to calibrate on the cal half alone again and never to use the test half for any choice again. Architecture A is back to its earlier numbers: coverage 91.5% / 91.7%, set size 1.65, gates stop 13.7% of families, inspector needed 49.0%, silent wrong answers 6.5%. On 28 October the quality gate then moved to α 0.05 ([`../quality_alpha/`](../quality_alpha/README.md)). Current numbers: gates stop 6.7% of families, inspector needed 38.5%, silent wrong answers 6.7% ([`../offtarget/rates.md`](../offtarget/rates.md)).
>
> OOD α stays at **0.08**. It was first tuned on the test half. Re-checked without the test half (`experiments/archA_alpha_check.py`, [`../alpha_check/`](../alpha_check/README.md)), 500 halvings of the cal half stop 12.9% of normal families on average, within the 15% budget. Two caveats:
> - the union bound alone guarantees only 18%;
> - α 0.05 would guarantee 15%, at the cost of 4.8% instead of 0.9% of off-target photos passing.
>
> The sections below record what pooling did, which is why it was rolled back.


This round asked whether architecture A could calibrate on more photo families without new data. It also asked what else had to change to finalise architecture A before work moves to gc7 (architecture B). The numbers come from [`README.md`](README.md) in this folder, made by `experiments/archA_pooled_cal.py`, and from the rebuilt reports in `results/archA/`. Nothing was retrained.

## 1. Calibration data: pooling val with the cal half

- **Before:** every conformal threshold (CP set, quality gate, OOD gate) was fitted on the cal half alone, which is 88 photo families. Each photo family counts once, so 88 was the real sample size.
- **Data already available:** val has 178 families and train has 496. Val was only used for temperature, the quality reference, and picking the final model out of 15 backbone/head combinations.
- **Pooled setup:**
  - CP threshold and OOD gate: val + cal half, **266 families (3×)**.
  - Quality gate calibration scores: cal half + train, **584 families (6.6×)**.
  - The quality gate uses pixel statistics only, so train crops are as valid as any.
  - The OOD gate can't use train crops: they are its kNN bank and the AoN regression's positives.
- **Val bias is small.**
  - Calibrating on val alone gives 90.8% / 91.0% coverage on the test half.
  - Pooled gives 90.6% / 90.2%.
  - Cal half alone gave 91.5% / 91.7%.
  - The drop is about 0.7 points. It comes from val having picked the model and fitted T, plus the smaller finite-sample correction with more families (1/89 vs 1/267). Coverage stays on target.
- **The spread between calibration draws narrows.** Over 200 random re-splits of the 175 test families (5th–95th percentile):

  | | cal half alone | pooled |
  |---|---|---|
  | per-family coverage | 85.5–95.1% | **87.2–93.3%** |
  | per-crop coverage | 87.1–95.8% | **87.8–94.5%** |
  | mean set size | 1.44–1.98 | **1.47–1.67** |
  | normal families stopped (pooled at OOD α 0.06) | 9.1–20.9% | **10.6–18.8%** |
- **Sets get smaller.** On the test half, mean set size went from 1.65 to 1.54 and single-class answers from 65% to 68%. The inspector is needed for 46.9% of crops instead of 49.0%.
- **The cost is a slightly higher silent error rate.** A single wrong class is shown without warning 6.8% of the time instead of 6.5%. That's the expected trade when coverage moves from 91.5% down to 90.6%, closer to the 90% target. The cal half alone was slightly over-covering.
- **Pooling doesn't stop draws from falling below 90%.** 25.5% of draws land below 90% per crop with pooling and without. That's inherent to a 90% target: the mean sits about a point above it, so a quarter of draws fall below. Only more families narrow it further.

## 2. The OOD gate's α: 0.08 → 0.06

- **0.08 was over budget.** With pooling, OOD α 0.08 stopped 15.7% of normal families on the test half and 16.4% on average across draws, above the ≤ 15% budget.
- **The old figure was lucky.** Even before pooling, α 0.08 averaged 15.0% across draws. The 13.7% on the fixed test half was a favourable draw: the α had been tuned to that one half.
- **α 0.06 is the largest that stays inside the budget on average** (14.7% across draws; 14.0% on the test half):

  | OOD α | families stopped (mean across draws) | off-target photos passing | wallpapers passing |
  |---:|---:|---:|---:|
  | 0.04 | 13.4% | 3.6% | 16.2% |
  | 0.05 | 13.9% | 2.4% | 13.8% |
  | **0.06** | **14.7%** | **0.9%** | **11.2%** |
  | 0.07 | 15.7% | 0.3% | 10.0% |
  | 0.08 | 16.4% | 0.1% | 8.2% |
- **Nothing lost on off-target input.** At α 0.06 with pooling:
  - off-target photos passing: 0.9%, the same as before;
  - wallpapers: 11.3% (was 10.8%);
  - random static: 0%;
  - whole-aircraft crops: 1.2%;
  - FGVC undamaged skin patches: 21.9% (was 21.6%);
  - the user's hard-hat photo: still "re-aim (not aircraft skin)".
- **D5 false reject:** 14.8% (was 14.4%). Re-take 13.3%, re-aim 1.5%.
- **D5 coverage:** 94.4% (was 95.0%). Dent is the weak class at 85.0% (was 87.2%). The two D5 classes stay unbalanced.

## 3. Deployed numbers (enhanced_res v2, test half: 2,417 crops, 87 families, α 0.10)

| | before (cal half, OOD 0.08) | **now (pooled, OOD 0.06)** |
|---|---:|---:|
| CP threshold q | 0.958 | 0.952 |
| CP coverage per crop / per family | 91.5% / 91.7% | **90.6% / 90.2%** |
| mean set size / single-class answers | 1.65 / 65% | **1.54 / 68%** |
| normal families stopped by the gates | 13.7% | **14.0%** |
| re-take / re-aim (crops) | 17.4% / 2.4% | 18.0% / 2.1% |
| inspector needed | 49.0% | **46.9%** |
| single wrong class shown without warning | 6.5% | 6.8% |
| D5 stopped / D5 coverage | 14.4% / 95.0% | 14.8% / 94.4% |
| off-target photos / wallpapers / static passing | 0.9% / 10.8% / 0% | 0.9% / 11.3% / 0% |
| temperature T, ECE | 1.342, 0.035 | unchanged |

Other checks:
- `evaluate.py --model v2` reports OK against the new expected numbers.
- `predict.py samples/`: 43% shown as one class, all right; 17% re-aim.

## 4. Other changes made in this round

- **The two gates are parallel**, and the docs now say so.
  - The quality gate reads pixels; the OOD gate reads the features from the model's single forward pass.
  - Both p-values reach one decision, where re-aim wins if both fire.
  - "Quality gate → OOD gate" only ever described which message wins, not the order of computation.
- **The reports now use the deployed CP threshold** from calibration.json. Before, each report fitted its own threshold on the cal half. This affects the visor report, D5, the gallery and the quality examples.
- **`calibrate.py`:**
  - pools the calibration data as described above;
  - caches the train quality statistics;
  - fits the AoN regression per model, on that model's own kNN bank and features.
- **Teammate's work merged.** Ahmed's commits (gc7 as the main model; renumbered folders) were pulled and our work re-applied, with conflicts resolved in favour of both sides.
- **gc7 (architecture B) is still on the old gates.** Its `calibration_gc7.json` uses kNN-only at α 0.10, sharpness on the low side only, and calibration on the cal half alone. It needs:
  - the gc7 weights: GitHub Release `v3.0-enhanced_res_gc7`;
  - the no-defect crops: `d1_gc7_no_defect_crops.zip` on the team Drive. They aren't on this Mac yet.

## 5. Open items

- **gc7 / architecture B:**
  - Recalibrate with U1, pooling and OOD α re-chosen for gc7's own budget.
  - Check the teammate's finding that the quality gate calls 22% of clean, plain panels "blurry". A featureless panel has a low Laplacian variance, just like a blurred photo. Our two-sided sharpness check doesn't address that side.
- **Helmet-camera calibration data** is still the main gap. Every guarantee here holds for D1-style photos only.
- **TensorRT** still runs without the OOD gate, because the ONNX export has no `features` output.
- **Label corrections:** 54 D1 boxes were corrected on 27 October. The crops aren't re-cut yet, so these numbers still use the old boxes.

## 6. Overfitting audit (`experiments/archA_overfit_check.py`, [`overfit_check.md`](overfit_check.md))

**What holds up (no sign of overfitting):**
- **The model is untouched.** No weights changed. T is one number fitted on 4,203 val crops.
- **The AoN regression generalises.** It has 769 parameters fitted on about 25k crops. AUROC on categories it never saw is 0.986–0.999.
- **Pooling val adds little bias.**
  - Calibrating on val alone gives 90.8% coverage on the test half; the cal half alone gave 91.5%.
  - Val was used to pick the best epoch and the best of 15 backbone/head combinations. That makes it slightly optimistic, but by under a point.
- **Train crops are valid for the quality gate.** Calibrated on train alone, it rejects 11.7% of test-half families; on the cal half alone, 11.2%.

**What is bad, got worse, or didn't improve:**
- **OOD α 0.06 was chosen using the test families.** The 200 random re-splits included them. So was α 0.08 before it: it was tuned on the fixed test half.
  - Chosen from val + cal only, the largest α within the 15% budget is **0.04**.
  - At 0.04 on the test half:
    - families stopped: 13.5%;
    - off-target photos passing: 3.6% (vs 0.9% at 0.06);
    - wallpapers passing: 16.2% (vs 11.2%).
- **Rare classes lost coverage with pooling.** On the test half (cal half alone → pooled):
  - scratch 86.0% → 82.6%;
  - corrosion 84.4% → 82.5%;
  - crack 87.3% → 86.5%;
  - paint-peel-off unchanged at 84.5%;
  - dent and missing-head stay at 96–97%.
  - Overall coverage stays at 90%, but four of six classes were already below it and three moved further down. 0.9% of crops dropped out of their set when q moved from 0.958 to 0.952.
- **Small regressions:**
  - silent wrong answers 6.5% → 6.8%;
  - D5 false reject 14.4% → 14.8%;
  - D5 coverage 95.0% → 94.4% (dent 87.2% → 85.0%);
  - wallpapers passing 10.8% → 11.3%;
  - FGVC undamaged patches 21.6% → 21.9%.
- **Not improved:**
  - About a quarter of calibration draws still land below 90% coverage (25.5% with and without pooling).
  - The quality gate rejects about 11–12% of normal families against its nominal 10%.
  - The deployed off-target pass rate (0.9%) is partly in-sample. The held-out figure is 2.3% / 4.0%.
- **The test half is no longer untouched.** It was used to compare LAC/APS, family weighting, temperature kinds, gate α values and the off-target budget. Every test-half number is therefore mildly optimistic. Only new photos, ideally from the helmet, give a clean estimate.
- **The splits differ in image sharpness.** Median Laplacian variance: val 35, train 41, cal half 62, test half 84. Conformal guarantees assume calibration and test photos are alike, and they aren't fully. That gap is also why the val + cal estimate of the gates' false reject (16.7% at α 0.06) is higher than the test half's (14.0%).
