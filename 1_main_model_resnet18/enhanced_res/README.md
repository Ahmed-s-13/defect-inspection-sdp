# enhanced_res: the main model

> **The model the project uses is enhanced_res gc7** (since 28 October 2026): a ResNet-18 + MLP head with **7
> classes**, the six defects plus **no-defect**, trained and tested on [`d1_gc7`](../d1_gc7/README.md). d1_gc7 is
> the leak-free [`d1_grouped`](../d1_grouped/README.md) split plus reviewed crops of intact surface. Same
> architecture and recipe as before; the earlier 6-class version (**v2**) is kept and still loads with
> `EnhancedRes()`. The earlier model (`../train.py` + `../train_mlp.py`) is kept as the legacy baseline.
>
> **Its output goes through [architecture A](arch_a/README.md)** (`arch_a/`): a quality gate, an unfamiliar-input
> gate, temperature scaling and a conformal prediction set, refitted for gc7 (`arch_a/calibration_gc7.json`; still on the earlier kNN-only gate until it is recalibrated
> with the final gate).
> `predict.py` shows its answer ("NO-DEFECT 92.9%", "CRACK 88.0%", "AMBIGUOUS: crack 52% dent 31%", "BAD IMAGE:
> re-take", ...).

## gc7: the 7-class model (d1_gc7 test: 4,255 defect crops + 615 no-defect crops, 175 source groups)

| | v2 (6 classes) | **gc7** |
|---|---|---|
| 7-class accuracy / macro-F1 | n/a | **0.798 / 0.747** |
| no-defect crops called no-defect | 0% (it calls them dent 54%, scratch 24%) | **91.1%** (healed 88.0%, clean 93.2%) |
| defects called no-defect (top answer) | n/a | 3.9% (scratch 8.5%, crack 5.8%) |
| defects: accuracy / macro-F1 when it names a defect class | 0.804 / 0.738 | **0.805 / 0.742** |

- **No cost on defects.** Ranking only the six defect classes, gc7 is as good as v2. Recall per class (top answer):
  corrosion 0.62, crack 0.74, dent 0.85, missing-head 0.94, paint-peel-off 0.72, scratch 0.53, no-defect 0.91.
- **Fewer missed defects with a cut.** Say no-defect only when p(no-defect) is above a cut chosen on val
  (`results/gc7/summary.json`, "dial"). At a cut that misses 1% of val defects, it misses **0.9%** of test defects
  and still catches 80% of clean crops. At 2%, it misses 2.4% and catches 88%.
- **It did not learn "LaMa fill = no defect".** Healed crops are recognised *less* often than clean-cropper crops
  (88% vs 93%). Defect crops with a clean patch LaMa-filled are called no-defect only slightly more often than the
  same crops untouched: 4.7% vs 3.0% of 301 crops (`../../2_lama_healing/no_defect/decoys.py`).
- **Training:** best epoch 5 of 10 (val 7-class macro-F1 0.685). It was already at 0.680 at epoch 3, then flat
  within ±0.003 while the train loss kept falling. v2's backbone behaved the same way.
- **Architecture A** works with 7 classes once recalibrated: coverage 92.0% at α = 0.10, mean set size 1.70, 65%
  single answers, and 0.3% of defects shown as a single "no-defect". Details in [`arch_a/README.md`](arch_a/README.md).

Every choice (epoch, kNN weight 0.25, prior correction 0.5) was made on val. Test was only read for the report
(`results/gc7/`). One training run (seed 0).

## v2: the 6-class model (grouped D1 test: 4,255 crops from 175 source groups)

| model | accuracy | macro-F1 (95% group interval) | balanced accuracy |
|---|---|---|---|
| ResNet-18 + MLP head, same recipe as before, retrained on d1_grouped | 0.781 | 0.715 (0.670-0.754) | 0.711 |
| **enhanced_res** | **0.804** | **0.738** (0.692-0.776) | **0.737** |

- **Is the gain real?** The macro-F1 gain is +0.022 with a 95% interval of +0.002 to +0.045 (resampling whole
  source groups). On the same test crops, enhanced_res is right where the baseline is wrong 251 times; the
  reverse happens 152 times (McNemar p = 9e-7).
- **Per-class recall:** crack 0.74 → 0.80, paint-peel-off 0.63 → 0.68, scratch 0.49 → 0.54. Corrosion stays at
  about 0.59, dent at 0.88, missing-head at 0.94.
- **Train vs test:** 98.6% train accuracy against 80.4% test. The model still memorises its ~500 training
  scenes; more independent data is the main lever now.
- **Outside test, D5 photos not in D1** (1,040 crops: 853 missing-head, 187 dent): accuracy 0.760 → 0.840.
  That gain comes from missing-head (0.75 → 0.88), while dent recall fell (0.80 → 0.65): D5's small hail-style
  dents get called "scratch". Averaged over the two classes there is no clear gain.

D1's original split (where the legacy model scored 0.901) put copies of the same photos in train and test, so its
numbers are not comparable with these. Full numbers are in [`results/report.md`](results/report.md); every
backbone × head step is in [`results/ablation.csv`](results/ablation.csv).

## What changed from the plain ResNet-18 + MLP head, and why

| change | why | effect (val → test macro-F1) |
|---|---|---|
| **Group- and class-balanced sampling** | a scene shot 500 times no longer counts 500× per epoch, and rare classes (scratch, corrosion, paint-peel-off) are drawn more often | alone: little change |
| **Gentler training**: lower lr for pretrained layers, warm-up + cosine decay, label smoothing 0.1, EMA of the weights | the plain model memorised within 4 epochs; this slows it down and makes the picked epoch less noisy | with the sampler: val 0.651 → 0.678, test 0.715 → 0.724 |
| **Layer3 features** next to layer4 in the MLP head | mid-level texture, not just the final summary | small |
| **Test-time augmentation**: features averaged over the crop's 8 flips/rotations | D1 is full of flipped/rotated copies; the answer should not depend on orientation | biggest single gain: test 0.726 → 0.743 |
| **Augmented training views** for the MLP head (3 per train crop) | the backbone memorised its own training crops, so their features are cleaner than test features | val up slightly (kept), test down slightly |
| Class-prior correction | push predictions toward rare classes | val chose "off": not used |

Every choice (backbone, MLP epochs, kNN weight, prior correction) was made on **val**. Test and D5 were only
read for the report.

## Getting the model

The weights are not in git (like all `*.pt` files). Unzip each file in the repo root:

| file | where | gives |
|---|---|---|
| `enhanced_res_gc7_model.zip` | GitHub **Releases**, `v3.0-gc7` | **gc7**: `data/work/enhanced_res/gc7/enhanced_res_backbone.pt` (45 MB) + `enhanced_res_head.pt` (100 MB) |
| `enhanced_res_model.zip` | GitHub **Releases**, `v2.0-enhanced-res` | v2: `data/work/enhanced_res/enhanced_res_backbone.pt` + `enhanced_res_head.pt` (also used by the healer's screw-head scorer and `trt_enhanced_res`) |
| `d1_gc7_no_defect_crops.zip` | team Google Drive | the no-defect crops of d1_gc7: `data/work/crops/no_defect/` (needed to test or retrain gc7, not to run it) |
| `main_model_data_work.zip` | team Google Drive | the D1 crops and the rest of `data/work/` |

The splits are in git: `../d1_grouped/groups.csv` and `../d1_gc7/no_defect.csv`.

## Quick check (no training)

From `1_main_model_resnet18/`:

```bash
python enhanced_res/evaluate.py             # gc7: d1_gc7 test + val (+ D5) + architecture A; ends with "OK: results match ..."
python enhanced_res/evaluate.py --model v2  # the 6-class v2 on grouped test + val (+ D5), as before
python predict.py samples/ --compare        # 30 grouped-test crops, with the legacy model's answer next to it
python predict.py photo.jpg --box 375 326 486 411
```

`predict.py` uses gc7 + architecture A by default (`--model v2` for the 6-class model, `--raw` for the plain
ranking, `--legacy` for the earlier model). If the gc7 weights are missing it says so and uses v2. In code
(architecture A in code: [`arch_a/README.md`](arch_a/README.md)):

```python
from enhanced_res.model import EnhancedRes, MODELS
model = EnhancedRes(MODELS["gc7"])   # the main model (7 classes); EnhancedRes() alone loads v2 (6 classes)
probs = model(x)                     # x: (N, 3, 224, 224) crops normalised with klshift.common.eval_transform()
model.classes                        # [..., 'scratch', 'no-defect']: no-defect is the last column
```

## Training it again

All commands from `1_main_model_resnet18/`. Runs, caches and weights go to `data/work/enhanced_res/`.

**gc7** (needs the d1_gc7 no-defect crops):

```bash
python enhanced_res/train_backbone.py --data d1_gc7 --out gc7   # ~25 min: 7-class backbone (best epoch 5 of 10)
python enhanced_res/build_head_gc7.py                            # ~15 min: the head, data/work/enhanced_res/gc7/, results/gc7/
python enhanced_res/arch_a/calibrate.py                          # ~10 min: architecture A for gc7, arch_a/calibration_gc7.json
```

`--data d1_gc7` uses v2's recipe unchanged with 7 classes. no-defect is the last class and gets about 14% of the
sampler's draws. The data sources are described in `../d1_gc7/README.md`.

**v2** and the ablation behind it:

```bash
python enhanced_res/train_backbone.py --sampler base       --recipe base     --out r0_plain           # baseline
python enhanced_res/train_backbone.py --sampler groupclass --recipe base     --out r1_sampler
python enhanced_res/train_backbone.py --sampler groupclass --recipe enhanced --out r2_sampler_recipe    # enhanced_res
python enhanced_res/external_d5.py     # D5 outside test (needs D5 raw photos and ../d1_grouped caches)
python enhanced_res/build_head.py      # MLP heads on all 3 backbones, picks the final model on val, writes results/
```

Each backbone takes 30-60 min on an M-series Mac; `build_head.py` takes about an hour (features for 3 backbones,
15 heads). `build_head.py --report-only` reuses `results/ablation.csv` and refits only the baseline and the final model.

| file | what it is |
|---|---|
| `model.py` | paths, the splits (`load_split("d1_gc7")` / `load_split()`), feature extraction, the MLP head, and `EnhancedRes` (load + predict; `MODELS["gc7"]` / `MODELS["v2"]`) |
| `train_backbone.py` | step 1: trains a ResNet-18 backbone with or without the sampler / recipe; `--data d1_gc7` for gc7 |
| `build_head.py` | step 2 for v2: features + MLP heads, ablation, final-model choice on val, report |
| `build_head_gc7.py` | step 2 for gc7: the same head (h4) on the 7-class backbone, test report in `results/gc7/` |
| `external_d5.py` | the D5 outside test set: D5 photos not copied from D1 (87% of D5 is) |
| `evaluate.py` | re-tests a saved model (`--model gc7`, default, or `v2`) without training, and checks architecture A on the test half |
| **`arch_a/`** | **architecture A**: `runtime.py` (the decision layer), `calibrate.py`, `calibration_gc7.json` (gc7) and `calibration.json` (v2), the study (`experiments/`, `results/`) |
| `results/` | v2: report, ablation table, summary, confusion matrix, D5 test set (`d5/`); **`results/gc7/`**: gc7's test report |
