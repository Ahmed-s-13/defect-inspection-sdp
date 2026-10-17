# 4. Early classifiers (archive)

**Archived. These were our first models (10 Oct 2026), before the main model in
`1_main_model_resnet18`.** They are kept for reference and for the write-up; new work
should start from the main model.

Two supervised classifiers for defect crops from Roboflow aircraft (D2, v22) and
wind-turbine (D4) datasets, plus an evaluation built around the four risks:
overfitting, shortcut learning, distribution shift and uneven sampling.
Results: [`RESULTS.md`](RESULTS.md) and [`model_summary.xlsx`](model_summary.xlsx).

| folder / file | what it is |
|---|---|
| **`model_1_plain_cnn/`** | **Model 1**: plain 5-block CNN from scratch, cross-entropy. `train.py` + `results/aircraft`, `results/turbine` |
| **`model_2_supcon_resnet18/`** | **Model 2**: supervised contrastive ResNet-18 + class-balanced classifier (SupCon + cRT). `train.py` + `results/aircraft`, `results/turbine` |
| `common.py` | shared by both models: data loading, network wrapper, metrics, corruptions, masks |
| `prepare_data.py` | downloads a Roboflow export and turns it into a classification set with a leak-free split |
| `data_audit.py` | checks imbalance and how guessable the class is without looking at the defect |
| `evaluate.py` | one report per trained model covering the four risks and PCA |
| `compare.py` | side-by-side table of several evaluated models |
| `tools/make_synthetic.py` | fake dataset for a 5-minute pipeline check |

Not kept: the "ResNet-18 cross-entropy" control run (superseded by the main model),
the prepared datasets (`data/`, poor quality, rebuildable with `prepare_data.py`),
and the trained weights (`results/*/model.pt`, not in git).

## Three things about the datasets that shaped the design

**1. Both datasets are object-detection datasets, not classification datasets.**
They contain bounding boxes, and one image can hold several defect types. So
`prepare_data.py` cuts one square crop per annotated box (the box plus a 25%
margin) and the task becomes "which defect is in this patch?". This matches the
wearable, which looks at a patch of surface at a time. The alternative is
image-level multi-label classification; it throws the box away, and the box is
what makes the shortcut tests below possible.

**2. The second dataset is wind-turbine blades, not aircraft.** It is a useful
proxy (painted composite surfaces, cracks, corrosion, drone imagery) and a hard
test of distribution shift, but results on it are not aircraft results.

**3. The splits shipped by Roboflow should not be used.**
- Aircraft: versions 24 and 25 are augmented copies (2,924 images from about
  1,200 originals). Version 22 is the raw set with no split. Use 22, which is
  the preset.
- Turbine: filenames such as `DJI_0688_04_02` and `DJI_0688_04_03` are tiles cut
  from the same drone photo, and the shipped 80/10/10 split is per tile.
  `prepare_data.py` prints what share of the shipped test tiles have a sibling
  tile in train.

`prepare_data.py` therefore re-splits everything 70/15/15 so that all crops
from one original photo stay on one side, stratified by class.

Neither dataset has a "no damage" class. A model trained on them must call
every patch some defect. `--background-per-image 1` adds defect-free crops as a
`no_damage` class, which is closer to what the helmet needs. It is off by
default because unannotated damage in the background becomes label noise, and
on drone images the easy negatives are sky. Worth running both ways.

## The two models

**Model 1, plain CNN** (`model_1_plain_cnn/train.py`). Five blocks of
conv-BN-ReLU-conv-BN-ReLU-maxpool (32 to 256 channels), global average pooling,
dropout, one linear layer; about 2.4 M weights, trained from scratch with
cross-entropy. It carries only the standard safeguards: augmentation, dropout,
weight decay, early stopping on validation macro-F1, class-weighted loss.

**Model 2, SupCon + cRT** (`model_2_supcon_resnet18/train.py`). Same labels, used in two stages:

1. *Supervised contrastive learning* (Khosla et al., NeurIPS 2020). Each crop is
   augmented twice; an ImageNet-pretrained ResNet-18 (ResNet-50 is an option, too heavy for a laptop) is trained so that all
   views of one class sit together in embedding space and other classes sit
   apart. The paper reports better robustness to natural corruptions and less
   sensitivity to hyperparameters than cross-entropy, which are the two things
   the meeting notes ask for.
2. *Classifier re-training* (Kang et al., ICLR 2020). The encoder is frozen and
   a linear layer is trained on it with every class sampled equally often.
   Their finding is that imbalance hurts the classifier much more than the
   features, so only the classifier needs rebalancing.

One departure from the papers: stage 1 uses class-balanced batches by default so
a rare class always has same-class partners in the batch. Kang et al. found
natural-frequency sampling best for feature learning under cross-entropy;
`--stage1-sampling instance` gives that variant. Try both if a class is rare.

**Control run.** Model 2 differs from Model 1 in three ways at once: the
architecture, ImageNet pretraining, and the loss. A pretrained ResNet-18 with plain
cross-entropy was run once as a control (see `RESULTS.md`); about 8 of the 11
macro-F1 points came from pretraining. That model became the main model in
`1_main_model_resnet18`, so the control is not kept here.

## How each risk is handled and measured

| risk | in training | in `evaluate.py` |
|---|---|---|
| Overfitting | augmentation, dropout, weight decay, early stopping; Model 2 keeps the stage-1 encoder with the best validation k-NN score | train vs. val vs. test macro-F1, curves, intervals that resample whole photos |
| Shortcut learning | square crops with no padding (no frame cue); Model 2 compares each crop to every other crop of its class | hide the defect (score should collapse), hide the context (score should hold), share of Grad-CAM heat inside the annotated box |
| Distribution shift | colour/blur/crop augmentation | 8 camera and lighting conditions at 3 severities; score per photo source; `--holdout-source` keeps a whole source unseen; `--shift-data` tests on the other dataset |
| Uneven sampling | class-weighted loss (Model 1), balanced sampling (Model 2) | macro-F1, balanced accuracy, per-class recall, majority-class reference |

**PCA** is used as a diagnostic on the learned features: how many components
carry 95% of the variance, how the score changes when a linear classifier sees
only the first k components, and a 2-D plot coloured by class and by photo
source. A feature space that clusters by source rather than by class is the
"wing = corrosion" problem made visible. `data_audit.py` also reports a
PCA-of-raw-pixels classifier as a no-network reference.

## Running it (Colab with a GPU runtime, or any machine with PyTorch)

Run from this folder. Install from the repo root with `pip install -r requirements.txt`.

```bash
# 0. optional 5-minute check that everything runs
python tools/make_synthetic.py --out data/_raw/synth
python prepare_data.py --name synth --yolo-dir data/_raw/synth
python model_1_plain_cnn/train.py --data data/synth --out runs/synth_cnn --img-size 96 --epochs 5

# 1. data (free Roboflow account -> Settings -> API Keys). Never commit the key.
python prepare_data.py --preset aircraft --api-key "$ROBOFLOW_API_KEY"
python prepare_data.py --preset turbine  --api-key "$ROBOFLOW_API_KEY"
python data_audit.py --data data/aircraft

# 2. train (settings used for the archived results)
python model_1_plain_cnn/train.py       --data data/aircraft --out model_1_plain_cnn/results/aircraft
python model_2_supcon_resnet18/train.py --data data/aircraft --out model_2_supcon_resnet18/results/aircraft --batch-size 32 --epochs 30

# 3. evaluate and compare
for r in model_1_plain_cnn/results/aircraft model_2_supcon_resnet18/results/aircraft; do
  python evaluate.py --run $r --data data/aircraft \
    --shift-data data/turbine --class-map "cracking=crack,surface corrosion=corrosion,leading edge corrosion=corrosion"
done
python compare.py model_1_plain_cnn/results/aircraft model_2_supcon_resnet18/results/aircraft
```

Repeat steps 2 and 3 with `data/turbine` (drop `--shift-data`, or point it back
at `data/aircraft` with the map reversed). If you downloaded a zip from the
Roboflow page by hand, unzip it and pass `--yolo-dir` instead of `--api-key`.

In Colab, put `!` in front of each line. A smaller `--batch-size` makes Model 2 lighter if memory is short.

## Options worth knowing

| option | effect | trade-off |
|---|---|---|
| `--background-per-image 1` | adds a `no_damage` class | closer to the product; noisier labels |
| `--holdout-source NAME` | one photo source is never trained on and is reported separately | strongest shift test available without new data; fewer training crops |
| `--context 0.25` | margin around each box | more context helps recognition and invites shortcuts |
| `--group-regex '^(IMG_\d{8})'` | splits by capture day instead of by photo | guards against burst shots of one defect; coarser split |
| `--balance sampler / none` (Model 1) | how imbalance is handled | `none` shows how much the imbalance costs |

## Limits

Limits to keep in mind when reading results:
- "Photo source" is guessed from filename prefixes. It is a diagnostic, not ground truth.
- Splitting by photo does not catch two photos of the same defect taken seconds apart.
- The hide-the-defect test paints a rectangle, and the rectangle's shape is itself a hint.
  Use it to compare models rather than as an absolute number.
- The cross-dataset test maps turbine classes onto aircraft classes by name.
  Blade erosion and aluminium corrosion are different physical things; treat a
  low score there as expected, and a high one with suspicion.

## References

- Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020. https://arxiv.org/abs/2004.11362
- Kang et al., "Decoupling Representation and Classifier for Long-Tailed Recognition", ICLR 2020. https://arxiv.org/abs/1910.09217
- Geirhos et al., "Shortcut Learning in Deep Neural Networks", Nature Machine Intelligence 2020. https://arxiv.org/abs/2004.07780
- Hendrycks and Dietterich, "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations", ICLR 2019. https://arxiv.org/abs/1903.12261
- Selvaraju et al., "Grad-CAM", ICCV 2017. https://arxiv.org/abs/1610.02391
- Datasets (CC BY 4.0): https://universe.roboflow.com/lemi-debele/aircraft-surface-damage and https://universe.roboflow.com/123-hmdgs/-6-classes-wind-turbine-dataset
