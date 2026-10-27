# enhanced_res: the main model

> **This is the defect classifier the project uses.** A ResNet-18 + MLP head, trained and tested on the leak-free
> [`d1_grouped`](../d1_grouped/README.md) split. The earlier model (`../train.py` + `../train_mlp.py`) is kept as
> the legacy baseline.
>
> **Its output goes through [architecture A](arch_a/README.md)** (`arch_a/`): a quality gate, an unfamiliar-input
> gate, temperature scaling and a conformal prediction set, fitted on d1_grouped without retraining. `predict.py`
> shows its answer ("CRACK 88.0%", "AMBIGUOUS: crack 52% dent 31%", "BAD IMAGE: re-take", ...) instead of the
> old top-2 list.

## Results (grouped D1 test: 4,255 crops from 175 source groups)

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

The weights are not in git (like all `*.pt` files). Download `enhanced_res_model.zip` from this repo's
**Releases** page (`v2.0-enhanced-res`) and unzip it in the repo root; it gives `1_main_model_resnet18/data/work/enhanced_res/enhanced_res_backbone.pt` (45 MB) and
`enhanced_res_head.pt` (91 MB). You also need `main_model_data_work.zip` for the D1 crops. The split itself is
`../d1_grouped/groups.csv`, which is in git.

## Quick check (no training)

From `1_main_model_resnet18/`:

```bash
python enhanced_res/evaluate.py        # grouped test + val (+ D5 if its crops exist) + architecture A; ends with "OK: results match ..."
python predict.py samples/ --compare   # 30 grouped-test crops, with the legacy model's answer next to it
python predict.py photo.jpg --box 375 326 486 411
```

`predict.py` uses enhanced_res + architecture A by default (`--raw` for the plain ranking, `--legacy` for the
earlier model). In code (architecture A in code: [`arch_a/README.md`](arch_a/README.md)):

```python
from enhanced_res.model import EnhancedRes
model = EnhancedRes()          # CUDA / MPS / CPU
probs = model(x)               # x: (N, 3, 224, 224) crops normalised with klshift.common.eval_transform()
```

## Training it again

All commands from `1_main_model_resnet18/`. Runs, caches and weights go to `data/work/enhanced_res/`.

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
| `model.py` | paths, the grouped split, feature extraction, the MLP head, and `EnhancedRes` (load + predict) |
| `train_backbone.py` | step 1: trains a ResNet-18 backbone with or without the sampler / recipe |
| `build_head.py` | step 2: features + MLP heads, ablation, final-model choice on val, report |
| `external_d5.py` | the D5 outside test set: D5 photos not copied from D1 (87% of D5 is) |
| `evaluate.py` | re-tests the saved model without training, and checks architecture A on the test half |
| **`arch_a/`** | **architecture A**: `runtime.py` (the decision layer), `calibrate.py`, `calibration.json`, the study (`experiments/`, `results/`) |
| `results/` | report, ablation table, summary, confusion matrix, D5 test set (`d5/`) |
