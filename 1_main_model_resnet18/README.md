# 1. Main model: ResNet-18 on D1 (+ dataset shift analysis + the Cropper)

> **Current grouped-D1 work:** use [experiments/d1_sscd_grouped/README.md](experiments/d1_sscd_grouped/README.md). It includes the SSCD source-family estimate, the saved conservative grouped split, and adapted ResNet-18/MLP scripts. The scores below describe the historical experiment, not the new split. The first grouped training attempt was interrupted before a completed epoch; no new trained checkpoint exists yet.


> **This is the model the project currently uses.** Everything else in the repo
> either builds on it (`2_healing_experiment`) or came before it (`4_early_classifiers_archive`).

A stock torchvision ResNet-18 (ImageNet weights, only `fc` replaced) is trained on
D1 (Kaggle Aircraft_dataset, 6 defect classes). Its last layer is then replaced by a more
accurate **MLP head** trained on its 512-d embeddings (step 7, see below). The ResNet is also
used as a fixed feature extractor: D2–D10 are fed through it and the KL divergence between their 512-d
embedding distributions is measured, to see how far each public dataset is from D1.

The full task spec is in [`docs/TASK_SPEC.md`](docs/TASK_SPEC.md), the interpretation
in [`findings.md`](findings.md) and the generated report in
[`results/summary.md`](results/summary.md).

## Headline results (D1-test, 4,293 crops)

| metric | ResNet-18 (own `fc` layer) | **ResNet-18 + MLP head (in use)** |
|---|---|---|
| accuracy | 0.880 | **0.901** |
| macro-F1 | 0.813 | **0.837** |
| balanced accuracy | 0.819 | **0.832** |
| train macro-F1 (overfitting check) | 0.939 | |

Per-class recall, the confusion matrix and the training curves: ResNet-18 in
[`results/d1_report.md`](results/d1_report.md), MLP head in [`results/mlp/report.md`](results/mlp/report.md).

## Quick check: use the trained model without training

After unzipping `trained_models.zip` and `main_model_data_work.zip` (see the top-level README):

```bash
python evaluate.py      # ~1-2 min: re-tests model_best.pt on D1-test and D1-val
python train_mlp.py     # ~40 s: trains the MLP head (data/work/mlp_head.pt) from results/embeddings.npz
python evaluate_mlp.py  # a few seconds: re-tests the MLP head on D1-test and D1-val
```

Each prints accuracy, macro-F1, per-class recall and the confusion matrix, and checks them
against `results/d1_report.json` / `results/mlp/report.json`. `OK: results match ...` means the
model, the data and your Python setup are all working.

```bash
python predict.py samples/ --compare                 # 30 held-out test crops, MLP head vs ResNet fc
python predict.py photo.jpg --box 375 326 486 411   # defect box in pixels, cut exactly like training
python predict.py photo.jpg                          # no box: uses the centre square of the image
python predict.py data/work/crops/D1/                # every image in a folder
python predict.py photo.jpg --resnet-only            # the ResNet's own fc layer instead of the MLP head
```

Output looks like `aircraft_000933.jpg: CRACK  59.8%   (corrosion 40.2%, dent 0.0%)`.
`predict.py` uses the MLP head when `data/work/mlp_head.pt` exists, and otherwise falls back to the
ResNet's `fc` layer with a note. `samples/` holds 5 D1-test crops per class plus `labels.csv`. None
of them were used for training or for any choice, and the accuracy on them is printed at the end.
Two things to keep in mind:
- The model was trained on square crops around a defect. With `--box` the input matches
  training; without it, a wide shot of a whole panel will give unreliable answers.
- There is **no "no damage" class yet**. The model always names one of the 6 defects,
  even for a clean surface. Read the answer as "if this is a defect, which one".

## What is in this folder

| path | what it is |
|---|---|
| `klshift/common.py` | shared code: paths, dataset list, Roboflow file names, square crop, transforms, **the ResNet-18 builder**, dataset class |
| `download.py` | step 0: download D1–D10 (Kaggle / Roboflow / direct) |
| `prepare.py` | step 1: cut one square crop per box (box + 25%, 224 px) for every dataset |
| `overlap.py` | step 2: perceptual-hash duplicate check between datasets |
| `train.py` | step 3+4: leak-free D1 split and **training of the main ResNet-18** |
| `embed.py` | step 5: 512-d embeddings for every dataset |
| `shift.py` | step 6: KL divergence (Gaussian + k-NN), figures, `summary.md` |
| **`evaluate.py`** | **re-test the trained model on D1-test, no training needed** |
| **`predict.py`** | **classify the defect in any image with the trained model (ResNet-18 + MLP head)** |
| **`klshift/mlp.py`** | **the MLP head**: MLP ensemble + kNN on the 512-d embeddings |
| **`train_mlp.py`** | step 7: trains the MLP head on the D1 embeddings |
| **`evaluate_mlp.py`** | re-test the MLP head on D1-test, no training needed |
| `make_samples.py`, `samples/` | 30 held-out D1-test crops for a quick `predict.py` try |
| **`klshift/cropper.py`** | **the Cropper**: cuts "no damage" patches out of defect photos |
| **`run_cropper.py`** + **`cropper.yaml`** | runs the Cropper; all its settings are in the YAML |
| `make_synthetic.py` | tiny fake D1–D5 for a 2-minute pipeline check |
| `findings.md` | hand-written interpretation, inserted into `results/summary.md` on every run |
| `results/` | reports, CSV tables and figures from the last real run |
| `logs/` | console output of the last real runs |
| `data/` | **not in git** (~21 GB): raw downloads, crops, the trained model. See below |

## The MLP head (step 7)

The ResNet-18's own last layer (`fc`) is a single straight-line (linear) split of the 512-d
embedding into 6 classes. The MLP head replaces only that layer. The ResNet itself is not
retrained, so the cost on the wearable is the same ResNet pass plus a small extra step.

- **5 small neural networks (MLPs)**, 512 → 1024 → 512 → 256 → 6 with BatchNorm, GELU and dropout. They
  are trained with noise added to the embeddings and with label smoothing, so they learn curved class
  boundaries without memorising the train crops. Each starts from a different random seed, and their
  probabilities are averaged.
- **5-nearest-neighbour vote.** The embedding is compared with the 20,051 D1-train embeddings
  (cosine distance), and the 5 closest vote for their classes.
- The two are **averaged 50/50**. The weight was picked on D1-val.

Only D1 embeddings of the D1-trained ResNet are used (`results/embeddings.npz`, `dataset == "D1"`).
Train fits, val picks each MLP's best epoch and the blend weight, and **test is used only for the final
numbers**.

| head on the same embeddings (D1-test) | accuracy | macro-F1 |
|---|---|---|
| ResNet `fc` | 0.880 | 0.813 |
| kNN only | 0.898 | 0.834 |
| 5 MLPs only | 0.895 | 0.833 |
| **5 MLPs + kNN (in use)** | **0.901** | **0.837** |

On the same test crops the MLP head fixes 144 of the ResNet's mistakes and makes 52 new ones.
An exact McNemar test gives p = 3.5e-11, so the gain is not noise. Recall goes up for 5 of the 6 classes
(dent 0.899 → 0.941, corrosion 0.709 → 0.738). Scratch recall drops slightly (0.711 → 0.697), but its
precision rises (0.659 → 0.749). Training on 4 extra augmented views per crop (`--aug-views 4`) gave
the same result, so it is off by default.

All MLP-head scripts run on the Apple GPU (MPS) when there is one: the MLPs, the batches, the noise and
the kNN search all stay on the GPU. `train_mlp.py` takes about 40 s on an M-series MacBook, and `--cpu`
forces the CPU. `data/work/mlp_head.pt` (63 MB, includes the kNN bank) is git-ignored like
`model_best.pt`. Rebuild it with `python train_mlp.py`.

The MLP head works alongside the Cropper: like the ResNet `fc`, it only says *which* defect. The Cropper's
`D1nc` patches are already in `embeddings.npz`, so a 7th "no damage" class could be added to the head.

## The Cropper (no-damage patches)

None of the defect datasets has a "no damage" class. The Cropper makes one by cutting
patches out of the defect photos themselves, away from every annotated defect:

1. **Propose:** random squares sized like the photo's own defect crops.
2. **Exclude:** drop squares touching a defect box (+25%).
3. **Surface check:** keep squares that CLIP zero-shot scores as aircraft skin. The threshold is calibrated so that `calibrate_keep` (90%) of the dataset's own defect crops pass.
4. **Damage score:** a CLIP damage score, recorded only.

All settings are in `cropper.yaml`. CLIP comes from `open_clip_torch`; the first run
downloads the ViT-B-32 LAION-2B weights.

```bash
python run_cropper.py --dataset D1 --name D1nc --config cropper.yaml
python embed.py --datasets D1nc --append
python shift.py --sets D1 D1nd D1nc D5 --out results/cropper/kl --findings none
```

Outputs:
- `data/work/crops/D1nc.csv`: the accepted patches.
- `results/cropper/D1nc/`: step counts, score plots and sample sheets (`report.md`), plus `check_list.csv` and `check_sheet_*.png` for a hand check.
- `results/cropper/kl/`: the KL and separability comparison.

The fixed-threshold run is kept in `results/cropper/D1nc_fixed_threshold/` for comparison.

## Data and the trained model

The data is about 21 GB and is **not stored in git**. Everything in `data/` can be
rebuilt with `download.py` + `prepare.py`. Layout:

```
data/
├── raw/D1 … D10/        downloads (7 GB)
├── src/                 large source archives for D6, D9, D10 (11 GB)
└── work/                everything derived (3 GB)
    ├── crops/           224-px crops per dataset + one CSV each
    ├── d1_split.csv     the leak-free D1 train/val/test split (used by the healing experiment)
    ├── model_best.pt    ← the trained main ResNet-18 (45 MB)
    └── mlp_head.pt      ← the MLP head from train_mlp.py (63 MB)
```

`model_best.pt`, `data/work/` and `results/embeddings.npz` are shared as zip files
outside git. See "Getting the data and trained models" in the top-level README.

## Setup

From the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Every script takes `--help`, uses a fixed seed, and defaults to `data/raw`, `data/work` and `results/`. Run them from this folder.

## Pipeline check on synthetic data (about 2 minutes)

```bash
python make_synthetic.py --out synthetic/raw
python prepare.py  --raw synthetic/raw --work synthetic/work
python overlap.py  --work synthetic/work --results synthetic/results
python train.py    --work synthetic/work --results synthetic/results --epochs 4
python embed.py    --work synthetic/work --results synthetic/results
python shift.py    --work synthetic/work --results synthetic/results --findings none
```

## Real run

Credentials are read from environment variables only, never from files in the repo:
`ROBOFLOW_API_KEY`, plus `KAGGLE_USERNAME` and `KAGGLE_KEY` (or `~/.kaggle/kaggle.json`).

```bash
export ROBOFLOW_API_KEY=...   # your own key, do not commit it
```

| step | command | output |
|---|---|---|
| 0. download | `python download.py --list` then `python download.py --get D1 D2:22 D3:1 D4:2 D5:3 D6 D7:15 D8:11 D9 D10` | `data/raw/D*/` |
| 1. crops | `python prepare.py` (also cuts D6p-D9p skin patches and D1nd no-damage patches) | `data/work/crops/`, `dataset_stats.csv` |
| 2. overlap | `python overlap.py --thr 6` | `results/overlap_matrix.csv`, `overlap_heatmap.png`, `duplicate_pairs_sheet.png` |
| 3+4. split + train | `python train.py` (use `--train-frac 0.5` if too slow) | `data/work/model_best.pt`, `results/d1_report.md`, curves, confusion matrix |
| 5. embeddings | `python embed.py --datasets D1 D2 D4 D5 D6 D7 D8 D9 D6p D7p D8p D9p D1nd D10 D10n D10a` (D3 dropped) | `results/embeddings.npz`, `data/work/embeddings_corrupt.npz` |
| 6. KL + figures | `python shift.py --k 32` | `results/kl_matrix.csv`, figures (incl. `overlap_circles_*.png`), `summary.md` (inserts `findings.md`) |
| 6b. D10 vs D1/D5 | `python shift.py --sets D1 D5 D10n D10a --out results/d10 --findings none` | `results/d10/` (D10 is left out of the main tables) |
| 7. MLP head | `python train_mlp.py` | `data/work/mlp_head.pt`, `results/mlp/` (report, confusion matrices, curves) |

## Main choices

- **Crops.** One square crop per box: the box plus 25% on each side. The crop is clamped inside the image by shifting it, never padded, then resized to 224 and saved as JPEG q95, identically for all datasets.
- **Duplicates.** phash Hamming distance ≤ 6, or the same original file stem. Roboflow's `_jpg.rf.<hash>` suffix is stripped first, and the stem must be distinctive (at least 10 characters and not only digits).
- **D1 split.** 70/15/15 with `StratifiedGroupKFold`. A group joins crops of one image, tiles of one photo (`_rr_cc` suffix) and D1-internal near-duplicates.
- **KL.** PCA is fitted on D1-held-out. The Gaussian estimate uses Ledoit–Wolf, alongside a k-NN estimate (k = 5). Every comparison uses a common n (cap 2000) and 5 seeds. Reference halves are split by image group.
- **Sampling.** KL samples keep one image per within-dataset near-copy cluster (`--all-copies` turns this off) and at most one random crop per image (`--per-crop` turns this off). Sibling crops of one image are strongly correlated and otherwise inflate the k-NN noise floor about 100-fold.
- **Dataset versions.** D2 = Roboflow v22 (raw), D4 = v2, D5 = v3. D3 exists only as a 3× augmented export and is 98% D2, so it appears in the overlap table only.
- **No-damage sets.** D6 = FGVC-Aircraft 2013b (bottom 20 px banner removed; the official server is slow, so a copy of the same archive unpacked under `data/src/D6` is used when present), D7 = Roboflow aircraft-detection-model-citoe v15 (3× augmented; copies grouped by file name), D8 = Roboflow aircraft-kdg2w v11, D9 = Open Images V7 validation+test images with "Fixed-wing aircraft" boxes (group-of and depiction boxes dropped). Each has one box per plane. `D6p`–`D9p` are 2 skin patches per plane box, and `D1nd` is up to 2 patches per D1 image away from every defect box. Inside D6–D9, copies are grouped by file name only, because phash matches different planes photographed against plain sky.
- **D10.** Kaggle orvile/anomaly-detection-dataset-fuselage: fuselage mock-up video frames with a normal/anomaly label per frame, converted to two 720×720 squares per frame. `D10n`/`D10a` are its normal and anomaly frames. Video near-copies are grouped without chaining (`video_leaders` in `embed.py`).
