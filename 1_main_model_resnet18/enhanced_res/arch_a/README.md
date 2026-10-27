# Architecture A: the calibrated decision layer of enhanced_res

> **This is how the project turns enhanced_res's output into an answer.** It replaces the fixed "top 2 classes with
> their softmax %" display. **It is not a new model:** the enhanced_res weights (Release `v2.0-enhanced-res`) are
> unchanged and nothing is retrained. Architecture A adds two checks before the model's answer and a calibrated,
> conformal way of choosing how many classes to show. All of its numbers are fitted on d1_grouped and live in
> [`calibration.json`](calibration.json) (in git, about 0.4 MB).

`predict.py` and `trt_enhanced_res/predict_trt.py` use it by default (`--raw` gives the old top-2 output).

## What it does, per crop

| order | step | how | if it fires, the visor shows |
|---|---|---|---|
| 1 | **quality gate** | 4 pixel statistics of the grey crop: sharpness (variance of the Laplacian), brightness, share of blown-out (≥ 250) and black (≤ 5) pixels. The most unusual one, compared with normal val crops, becomes a conformal p-value against the calibration crops | **BAD IMAGE: re-take (blurry / too dark / glare ...)** if p ≤ 0.10 |
| 2 | model | enhanced_res as before: 8-view features → 5 MLP heads (softmax averaged) + 5-NN vote, blended 50/50 | |
| 3 | **unfamiliar-input (OOD) gate** | 1 − mean cosine similarity of the crop's 768-d features to its 5 nearest training crops → conformal p-value | **CAN'T ASSESS: re-aim** if p ≤ 0.10 |
| 4 | **temperature** | p^(1/T), renormalised, T = 1.34 fitted on val. Same top class, honest percentages | |
| 5 | **conformal set (LAC)** | keep every class with tempered p ≥ 1 − q̂ (α = 0.10, one global threshold, every photo family weighted equally) | 1 class: **CRACK 88.0%** · 2+: **AMBIGUOUS: crack 52% dent 31%** · 0: **UNCLEAR** |

**The guarantees, and what they are not:**
- **The set:** on normal crops like d1_grouped's, it contains the true class about 90% of the time (α = 0.10).
- **Each gate:** it flags about α of normal photo families (10% each).
- **Not guaranteed:** coverage per class, on new data sources, or on bad images that get through the gates.
- **No "no damage" class:** architecture A can't say a crop is undamaged. That needs a Good class (architectures B/C).

## Results (d1_grouped, test half: 2,417 crops, 87 photo families; α = 0.10)

Calibration uses the other half of d1_grouped's test split (1,838 crops, 88 families). Train and val are unchanged.

| what | result |
|---|---|
| stated confidence vs reality (ECE) | 0.105 → **0.035** with temperature; accuracy unchanged (79.1%) |
| CP coverage, per crop / per family | **91.5% / 91.7%** (target 90%); 100 random re-splits: 91.8% (83–97%) |
| CP set size / single-class answers | 1.65 / 65% (fixed top-2 always shows 2) |
| per-class coverage (100 re-splits) | dent 96%, missing-head 97%, crack 90%, paint-peel-off 86%, scratch 84%, corrosion 83% |
| OOD gate, off-target patches | AUROC **0.94** (energy 0.71, top probability 0.70); stops 77% at α 0.10, **82.5% with both gates** (was 45% / 60% at α 0.025) |
| random static, solid colours, gradients (320 images) | **100% stopped** |
| quality gate, synthetic damage stopped (mild / strong) | dark 65 / 100%, overexposed 74 / 97%, glare 68 / 98%, blur 21 / 25%, motion blur 12 / 22% |
| normal crops stopped by a gate | **20.0%** per crop, 17.8% averaged per family (quality 15.1%, OOD 4.9%) |
| **inspector needed** (re-take, re-aim, ambiguous) | **47.1%** of normal crops |
| **single wrong class shown with no warning** | **6.2%** (the old top-1 display: 20.9%) |
| D5 outside test (calibrated on D1 only) | coverage 95.0%, but dent 87.2%; ECE 0.071 |

**Known limits:**
- The synthetic damage isn't real helmet footage, and there was no partial-shadow test.
- Blur mostly gets through. On strongly blurred crops that pass, coverage is about 79%.
- Rare classes (corrosion, scratch, paint-peel-off) are below 90%, and per-class thresholds can't fix that with about 12–21 calibration families per class.
- Some D1 crack/dent labels may be wrong: [`results/label_audit/`](results/label_audit/README.md) is waiting for GAES review.

## Why these choices (each one tested against its alternatives)

| choice | alternative tested | why this one |
|---|---|---|
| probability temperature (p^(1/T)) | logit temperature | same ECE (0.035), lower NLL, keeps the top class by construction; the kNN half has no logits |
| family-weighted thresholds | one weight per crop | crops of one photo are near-copies. Family coverage 90.6% vs 87.7% |
| LAC score | APS, RAPS | same coverage with smaller sets (1.65 vs 2.17) and far more single answers (65% vs 8%). RAPS tuned to λ = 0, which is APS |
| one global threshold | per-class (Mondrian), pooled | per-class makes every class 92–96% but cuts single answers to 20%, and breaks at α 0.05 (3 classes in every set). Kept as an option for when more data exists |
| kNN feature distance | energy, max probability | 0.94 vs 0.71 / 0.70 AUROC on off-target. The model's probabilities stay confident on things it has never seen |
| one combined quality score | 4 separate tests (Bonferroni) | with 88 calibration families the separate tests can never fire. Two extra glare statistics made it worse |
| CP α 0.10 | 0.05, 0.20 | trade-off between inspector workload and silent errors: 63.5% / 2.9% at α 0.05, 47.1% / 6.2% at 0.10, 31.2% / 10.6% at 0.20 |
| OOD α 0.10 (since 27 Oct; was 0.025) | 0.025, 0.05, sharpness both sides | lowest false accept on off-target (40.5% → 17.5%) and heavy JPEG (12.7% → 6.7%); costs 4.2 points more false rejects per crop (`results/archA/static/`) |

## Using it

From `1_main_model_resnet18/`:

```bash
python predict.py samples/                       # architecture A answers (+ accuracy and actions: samples/ has labels)
python predict.py photo.jpg --box 375 326 486 411 --details    # also the gate p-values and the full ranking
python predict.py photo.jpg --alpha 0.05         # larger sets, fewer misses (0.05 / 0.10 / 0.20 are calibrated)
python predict.py photo.jpg --raw                # the old top-2 output, no architecture A
python enhanced_res/evaluate.py --split test     # model check + architecture A check on the test half
```

In code:

```python
from enhanced_res.model import EnhancedRes
from enhanced_res.arch_a.runtime import ArchA
model, arch = EnhancedRes(), ArchA()                       # ArchA(alpha=0.05) for another calibrated alpha
f = model.features(x)                                      # x: (N, 3, 224, 224) normalised crops
dec = arch.decide(model.proba_from_features(f).cpu().numpy(), crops=pil_crops,
                  knn_dist=model.knn_distance(f).cpu().numpy())
print(dec[0].text(), dec[0].action, dec[0].classes, dec[0].scores)
```

**TensorRT** (`trt_enhanced_res/predict_trt.py`): it runs the quality gate, temperature and the conformal set; `runtime.py` needs only numpy and PIL. The engine outputs `probs` only, so the OOD gate is skipped there. Enabling it needs a `features` output added to the ONNX export (owned by the `trt_enhanced_res/` team). `--grouped-test` also checks the CP coverage and set size against PyTorch.

**Re-calibrate** whenever the model weights, the labels or the split change. It retrains nothing:

```bash
python enhanced_res/arch_a/calibrate.py          # ~10 min the first time; writes calibration.json
```

## Files

| file | what it is |
|---|---|
| `runtime.py` | the decision layer: gates, temperature, CP set, `ArchA.decide()` (numpy + PIL only) |
| `calibrate.py` | fits T, the CP thresholds and the gates' calibration scores; writes `calibration.json` with the expected test-half numbers |
| `calibration.json` | everything `runtime.py` needs, fitted on d1_grouped |
| `cal_split.csv` | which d1_grouped test crops are the calibration half (88 families) vs the test half (87); seed 0, balanced by class |
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
| OOD gate, temperature and CP coverage with example crops; false accept / reject rates of every stage | `archA_gallery.py` | `results/archA/examples/archA_gallery.html` |
| crack/dent label review sheet for GAES | `label_audit.py` | [`results/label_audit/`](results/label_audit/README.md) |

Methods: split conformal prediction and LAC (Vovk et al. 2005; Sadinle et al. 2019), APS/RAPS (Romano et al. 2020; Angelopoulos et al. 2021), temperature scaling (Guo et al. 2017), kNN OOD (Sun et al. 2022), conformal outlier p-values (Bates et al. 2023), variance of the Laplacian (Pech-Pacheco et al. 2000).
