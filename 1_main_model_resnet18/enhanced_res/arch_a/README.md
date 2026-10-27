# Architecture A: the calibrated decision layer of enhanced_res

> **This is how the project turns enhanced_res's output into an answer.** It replaces the fixed "top 2 classes with
> their softmax %" display. **It is not a new model:** nothing is retrained. Architecture A adds two checks before
> the model's answer and a calibrated, conformal way of choosing how many classes to show. Its numbers are fitted
> per model version and live in git:
> - [`calibration_gc7.json`](calibration_gc7.json): **the main model, gc7** (7 classes incl. no-defect), fitted on d1_gc7;
> - [`calibration.json`](calibration.json): the 6-class v2, fitted on d1_grouped. `trt_enhanced_res` reads this one.
>
> The study and the results below were done on v2. What changes with gc7 is in **"gc7: a 7th class"**.

`predict.py` and `trt_enhanced_res/predict_trt.py` use it by default (`--raw` gives the old top-2 output).

## What it does, per crop

Steps 1a and 1b run in parallel, on the same crop: the quality gate needs only the pixels, the OOD gate the features
from the model's single forward pass. Both p-values reach one decision: if both gates fire, **re-aim** wins (pointing the
camera at aircraft skin is the useful instruction); which crops are stopped doesn't depend on that order.

| order | step | how | if it fires, the visor shows |
|---|---|---|---|
| 0 | model | enhanced_res as before: 8-view features → 5 MLP heads (softmax averaged) + 5-NN vote, blended 50/50 | |
| 1a | **quality gate** | 4 pixel statistics of the grey crop: sharpness (variance of the Laplacian), brightness, share of blown-out (≥ 250) and black (≤ 5) pixels. The most unusual one, compared with normal val crops, becomes a conformal p-value against the calibration crops. Sharpness is checked on both sides: too blurry or far too sharp (noise) | **BAD IMAGE: re-take (blurry / too dark / glare ...)** if p ≤ 0.05 |
| 1b | **unfamiliar-input (OOD) gate** | two scores on the crop's 768-d features, each a conformal p-value: (a) **kNN distance**, 1 − mean cosine similarity to the 5 nearest training crops; (b) **AoN** (short for "aircraft or not"), a logistic regression on the same features, D1 crops vs people, faces, helmets, hands, gloves, tools, vehicles and hangar / airport / industrial scenes. Combined by Bonferroni (p = 2 × the smaller) | **CAN'T ASSESS: re-aim (not aircraft skin / unlike any training crop)** if p ≤ 0.08 |
| 2 | **temperature** | p^(1/T), renormalised, T = 1.34 fitted on val. Same top class, honest percentages | |
| 3 | **conformal set (LAC)** | keep every class with tempered p ≥ 1 − q̂ (α = 0.10, one global threshold, every photo family weighted equally) | 1 class: **CRACK 88.0%** · 2+: **AMBIGUOUS: crack 52% dent 31%** · 0: **UNCLEAR** |

**The quality gate is kept in architecture A (α 0.05), but it is legacy from here on.** Architecture B (the gc7
model) is not built around it. [`results/archA/no_quality/`](results/archA/no_quality/README.md) has what dropping it
would change.

**The guarantees, and what they are not:**
- **The set:** on normal crops like d1_grouped's, it contains the true class about 90% of the time (α = 0.10).
- **The gates:** together they stop 6.7% of normal photo families (budget ≤ 15%; by the union bound at most 0.05 + 0.08 = 13%).
- **Not guaranteed:** coverage per class, on new data sources, or on bad images that get through the gates.
- **"No damage":** with v2, architecture A can't say a crop is undamaged. gc7 adds that as a 7th class, and
  architecture A treats it like any other class (next section).

## gc7: a 7th class (no-defect)

**Adding a class doesn't break architecture A.** `runtime.py` reads the class list from the calibration file,
and split conformal prediction works for any number of classes. What changes are the fitted numbers, so a new model
needs `calibrate.py` (`python enhanced_res/arch_a/calibrate.py` for gc7, `--model v2` for v2). For gc7:
- **Temperature** is fitted on d1_gc7 val, including its no-defect crops.
- **The cal/test halves** are cal_split.csv's: each no-defect crop goes to its photo family's half.
- **The unfamiliar-input gate** uses gc7's kNN bank, which holds the no-defect train crops.
- **The quality gate's reference** is d1_gc7 val.

Test half: 2,752 crops in 87 families, 335 of them no-defect. α = 0.10. These gc7 numbers were fitted with the
earlier gates (kNN distance only at α 0.10, sharpness checked on the low side only). **`calibration_gc7.json` is
the one place the earlier gate remains**: re-running `calibrate.py` (default `--model gc7`) moves gc7 to the final
gate. It needs the d1_gc7 no-defect crops (`d1_gc7_no_defect_crops.zip`, team Drive) and the gc7 weights.

| | v2 | **gc7** |
|---|---|---|
| T | 1.34 | 1.06 |
| CP coverage per crop / per family (target 90%) | 91.5% / 91.7% | **92.0% / 92.2%** |
| mean set size / single-class answers | 1.65 / 65% | **1.70 / 65%** |
| single wrong class shown without warning | 6.7% | **5.3%** |
| inspector needed (re-take, re-aim, ambiguous) | 38.5% | **46.3%** |
| no-defect coverage | n/a | 98% |
| defects shown as a single "no-defect" (a silent miss) | n/a | **0.3%** |

Gates on the test half, by crop source:

| | re-take (quality) | re-aim (unfamiliar) |
|---|---|---|
| defects | 14.1% | 3.9% |
| healed no-defect | 6.8% | 6.8% |
| clean no-defect | **21.8%** | 4.0% |

**What the study found** (`../results/gc7/conformal_study.json`):
- **Keeping v2's numbers for gc7 is not unsafe, but it is wasteful.** The old T over-softens gc7: coverage is
  94%, sets triple to 3.3 classes, and only 19% of answers name one class.
- **v2's unfamiliar-input bank would block clean surface.** It holds no clean crops, so it sends 30–44% of them to
  "re-aim". gc7's bank flags 4–7%.
- **Over 100 random cal/test splits,** coverage at α = 0.10 averages 91.3% (5th–95th percentile 87–94%).
  Scratch and corrosion stay below 90% (79% and 84%), as they did with v2.

**Known limit: plain panels look "blurry".** The quality gate flags 22% of clean-cropper crops, most of them for low
sharpness: a featureless panel has a low Laplacian variance, the same signal as a blurred photo. Including no-defect
crops in the reference didn't fix this. Options: a sharpness test that compares edges with texture, or a separate
reference for crops the model calls no-defect.

## Results (d1_grouped, test half: 2,417 crops, 87 photo families; α = 0.10)

Calibration uses the other half of d1_grouped's test split (1,838 crops, 88 families); temperature and the quality reference use val. Nothing is retrained. **The test half is used for reporting only**: no choice is made on it. (Pooling val into the calibration was tried and rolled back: [`results/archA/pooled_cal/`](results/archA/pooled_cal/FINDINGS.md).)

| what | result |
|---|---|
| stated confidence vs reality (ECE) | 0.105 → **0.035** with temperature; accuracy unchanged (79.1%) |
| CP coverage, per crop / per family | **91.5% / 91.7%** (target 90%); 200 random re-splits: 91.8% per crop (5th–95th pct 87.1–95.8%) |
| CP set size / single-class answers | 1.65 / 65% (fixed top-2 always shows 2) |
| per-class coverage (100 re-splits) | dent 96%, missing-head 97%, crack 90%, paint-peel-off 86%, scratch 84%, corrosion 83% |
| real off-target photos (people, helmets, hands, gloves, hangars, airports, runways, storage rooms; Open Images + Places365) | categories the AoN never trained on, at the deployed α 0.08: **5.8% pass on average** (helmets 2.8%, gloves 2.9%, hands 4.2%, hangars 6.5–11.5%, runways 15.5%); the old kNN-only gate let 54–70% through. The shipped AoN, which trained on all of them, lets 1.0% through ([`results/archA/offtarget/rates.md`](results/archA/offtarget/rates.md)) |
| random static, solid colours, gradients (320 images) | **100% stopped** |
| a photo of a person in a hard hat | **re-aim (not aircraft skin)**; the old gate showed "CRACK 86%" |
| quality gate at α 0.05, synthetic damage stopped (mild / strong; cal half, test half not used) | dark 26 / 100%, overexposed 59 / 92%, glare 23 / 34%, noise 95 / 100%, blur 13 / 16%, motion blur 11 / 16% |
| normal crops stopped by a gate | **6.6%** per crop, **6.7%** averaged per family (re-take 4.3%, re-aim 2.4%); D5: 6.6% |
| **inspector needed** (re-take, re-aim, ambiguous) | **38.5%** of normal crops |
| **single wrong class shown with no warning** | **6.7%** (the old top-1 display: 20.9%) |
| D5 outside test (calibrated on D1 only) | coverage 95.0%, but dent 87.2%; ECE 0.071 |

**Known limits:**
- The synthetic damage isn't real helmet footage, and there was no partial-shadow test.
- Blur mostly gets through. On strongly blurred crops that pass, coverage is about 79%.
- Off-target photos are often stopped with "re-take" rather than "re-aim": high-resolution photos are far sharper than D1 crops, so the both-sided sharpness check fires first. Either way, no defect is shown.
- 24.2% of undamaged aircraft-skin patches (FGVC) still pass and get a defect label. Only a "no damage" class closes that: architecture B (the gc7 model) is next, then C (anomaly detection).
- The AoN regression was fitted on Open Images and Places365 photos; hangar, helmet and hand photos from the real helmet camera are still untested.
- Rare classes (corrosion, scratch, paint-peel-off) are below 90%, and per-class thresholds can't fix that with about 12–21 calibration families per class.
- 54 D1 boxes were found misplaced and corrected on 27 October 2026 ([`d1_grouped/label_corrections/`](../../d1_grouped/label_corrections/README.md)). The numbers here still come from the old boxes until the crops are re-cut and `calibrate.py` is re-run.

## Why these choices (each one tested against its alternatives)

| choice | alternative tested | why this one |
|---|---|---|
| probability temperature (p^(1/T)) | logit temperature | same ECE (0.035), lower NLL, keeps the top class by construction; the kNN half has no logits |
| family-weighted thresholds | one weight per crop | crops of one photo are near-copies. Family coverage 90.6% vs 87.7% |
| LAC score | APS, RAPS | same coverage with smaller sets (1.65 vs 2.17) and far more single answers (65% vs 8%). RAPS tuned to λ = 0, which is APS |
| one global threshold | per-class (Mondrian), pooled | per-class makes every class 92–96% but cuts single answers to 20%, and breaks at α 0.05 (3 classes in every set). Kept as an option for when more data exists |
| kNN distance + AoN on the model's own features | kNN alone (k 1/5/20, layer4 only, per predicted class); AoN alone | at the 15% budget: kNN alone lets 60–74% of real off-target photos through; the combination reaches 2–4% at that budget (α 0.11; about 5% at the deployed α 0.08) with **no extra model**, and the kNN half still stops random static that the regression alone misses |
| keep a quality gate | OOD gate alone (α 0.08, or 0.13 with the freed budget) | without it, far more bad images get through (all damage: 49% → 57–66%; strong overexposure 8% → 40–49%, noise 5% → 60–72%, strong darkening 0.3% → 16–25%), and silent wrong answers on them rise (5.2% → 6.2–7.1%). The OOD gate barely reacts to exposure, noise or glare. Gain: fewer good photos stopped (7.4% → 4.5% at OOD 0.08) |
| quality α 0.05 (28 Oct) | 0.10 (`results/archA/quality_alpha/`; cal half only) | 0.05 halves false rejects (normal families 12.3% → 7.4%, D5 14.4% → 6.6%). The cost: most glare and mild dimming now get through (strong glare 3.9% → 66.4% passing, mild darkening 38% → 74%) |
| one combined quality score | 4 separate tests (Bonferroni) | with 88 calibration families the separate tests can never fire. Two extra glare statistics made it worse |
| CP α 0.10 | 0.05, 0.20 | trade-off between inspector workload and silent errors: 56.7% / 3.1% at α 0.05, 38.5% / 6.7% at 0.10, 20.6% / 12.1% at 0.20 |
| calibration on the cal half alone (88 families) | val + cal half pooled (266); val alone (178) | pooling narrowed the spread between calibration draws (per-family coverage 87–93% vs 85–95%) but lowered rare-class coverage (scratch 86 → 83%, corrosion 84 → 83%) and changed little else; rolled back |
| OOD α 0.08, sharpness both sides (27 Oct) | kNN-only at α 0.025 / 0.10; α 0.03–0.08 | first chosen on the test half; re-checked without it: 500 halvings of the cal half stop 12.9% of normal families on average (budget 15%; [`results/archA/alpha_check/`](results/archA/alpha_check/README.md)). With the quality gate at 0.05, the union bound now guarantees at most 13% on its own. Both-sided sharpness also stops noise and lowers D5 false rejects (23.5% → 14.4%) |

## Using it

From `1_main_model_resnet18/`:

```bash
python predict.py samples/                       # gc7 + architecture A answers (+ accuracy and actions: samples/ has labels)
python predict.py samples/ --model v2            # the 6-class v2 with calibration.json
python predict.py photo.jpg --box 375 326 486 411 --details    # also the gate p-values and the full ranking
python predict.py photo.jpg --alpha 0.05         # larger sets, fewer misses (0.05 / 0.10 / 0.20 are calibrated)
python predict.py photo.jpg --raw                # the old top-2 output, no architecture A
python enhanced_res/evaluate.py --split test     # gc7 check + architecture A check on the test half (--model v2 for v2)
```

In code:

```python
from enhanced_res.model import EnhancedRes, MODELS
from enhanced_res.arch_a.runtime import CALIBRATIONS, ArchA
model, arch = EnhancedRes(MODELS["gc7"]), ArchA(CALIBRATIONS["gc7"])   # v2: EnhancedRes(), ArchA()
                                                           # ArchA(..., alpha=0.05) for another calibrated alpha
f = model.features(x)                                      # x: (N, 3, 224, 224) normalised crops
dec = arch.decide(model.proba_from_features(f).cpu().numpy(), crops=pil_crops,
                  knn_dist=model.knn_distance(f).cpu().numpy())
print(dec[0].text(), dec[0].action, dec[0].classes, dec[0].scores)
```

**TensorRT** (`trt_enhanced_res/predict_trt.py`) still runs **v2** with `calibration.json`; moving it to gc7 needs a
new ONNX export of the gc7 weights (7 outputs) and `ArchA(CALIBRATIONS["gc7"])`. It runs the quality gate, temperature and the conformal set; `runtime.py` needs only numpy and PIL. The engine outputs `probs` only, so the OOD gate (both parts need the 768-d features) is skipped there. Enabling it needs a `features` output added to the ONNX export (owned by the `trt_enhanced_res/` team); the AoN part is then one dot product on those features. `--grouped-test` also checks the CP coverage and set size against PyTorch.

**Re-calibrate** whenever the model weights, the labels or the split change. It retrains nothing:

```bash
python enhanced_res/arch_a/calibrate.py          # ~10 min the first time; writes calibration.json
```

## Files

| file | what it is |
|---|---|
| `runtime.py` | the decision layer: gates, temperature, CP set, `ArchA.decide()` (numpy + PIL only) |
| `calibrate.py` | fits T, the CP thresholds and the gates' calibration scores; writes `calibration_gc7.json` (default) or `calibration.json` (`--model v2`) with the expected test-half numbers |
| `calibration_gc7.json` | everything `runtime.py` needs for gc7, fitted on d1_gc7 (`ArchA(CALIBRATIONS["gc7"])`) |
| `calibration.json` | everything `runtime.py` needs for v2, fitted on d1_grouped (`ArchA()`, `trt_enhanced_res`) |
| `cal_split.csv` | which d1_grouped test crops are the calibration half (88 families) vs the test half (87); seed 0, balanced by class. gc7's no-defect crops follow their family |
| `experiments/` | the study behind every choice (run from `1_main_model_resnet18/`, results cached in `data/work/enhanced_res/calibration/`) |
| `results/` | the reports of those runs |

Inside `experiments/` and `results/`:

| step | script | report |
|---|---|---|
| architecture 0: plain CP baseline, top-1/2/3 vs CP | `arch0_cp.py` | [`results/arch0/`](results/arch0/README.md) |
| temperature + family weighting | `archA_cp.py` | [`results/archA/step1/`](results/archA/step1/README.md) |
| quality and OOD gates on synthetic damage and off-target images | `archA_gates.py` | [`results/archA/step2/`](results/archA/step2/README.md) |
| LAC vs APS vs RAPS; global vs per-class vs pooled | `archA_sets.py` | [`results/archA/step3/`](results/archA/step3/README.md) |
| full pipeline as visor actions, workload per α | `archA_visor.py` | [`results/archA/step4/`](results/archA/step4/README.md) |
| D5 outside test | `archA_d5.py` | [`results/archA/d5/`](results/archA/d5/README.md) |
| caught vs missed examples (blur, glare, exposure) | `archA_examples.py` | `results/archA/examples/quality_examples.html` |
| OOD gate, temperature and CP coverage with example crops; false accept / reject rates of every stage (deployed gates) | `archA_gallery.py` | `results/archA/examples/archA_gallery.html` (local only, not in git: it embeds third-party wallpaper patches; run `archA_gallery.py` to make it) |
| realistic off-target benchmark (people, helmets, hands, hangars; Open Images + Places365; local only) | `offtarget_data.py` | data in `data/work/enhanced_res/offtarget/` (not in git) |
| pooled calibration (val + cal half): tried, audited for overfitting, rolled back | `archA_pooled_cal.py`, `archA_overfit_check.py` | [`results/archA/pooled_cal/`](results/archA/pooled_cal/FINDINGS.md) |
| dropping the quality gate (OOD only at 0.08 / 0.13) vs keeping it (test half not used) | `archA_no_quality.py` | [`results/archA/no_quality/`](results/archA/no_quality/README.md) |
| quality α 0.10 vs 0.05, false reject vs false accept (test half not used) | `archA_quality_alpha.py` | [`results/archA/quality_alpha/`](results/archA/quality_alpha/README.md) |
| OOD α checked without the test half | `archA_alpha_check.py` | [`results/archA/alpha_check/`](results/archA/alpha_check/README.md) |
| false accept / false reject per category, held out and deployed | `archA_rates.py` | [`results/archA/offtarget/rates.md`](results/archA/offtarget/rates.md) |
| the deployed gate on every off-target set, undamaged aircraft, D5 and the user photo | `archA_offtarget_check.py` | [`results/archA/offtarget/deployed.md`](results/archA/offtarget/deployed.md) |

Methods: split conformal prediction and LAC (Vovk et al. 2005; Sadinle et al. 2019), APS/RAPS (Romano et al. 2020; Angelopoulos et al. 2021), temperature scaling (Guo et al. 2017), kNN OOD (Sun et al. 2022), conformal outlier p-values (Bates et al. 2023), variance of the Laplacian (Pech-Pacheco et al. 2000).
