# 1. Main model: ResNet-18 on D1 (+ dataset shift analysis + the Cropper)

> **This is the model the project currently uses.** Everything else in the repo
> either builds on it (`2_healing_experiment`) or came before it (`4_early_classifiers_archive`).

A stock torchvision ResNet-18 (ImageNet weights, only `fc` replaced) is trained on
D1 (Kaggle Aircraft_dataset, 6 defect classes). It is then used as a fixed feature
extractor: D2–D10 are fed through it and the KL divergence between their 512-d
embedding distributions is measured, to see how far each public dataset is from D1.

The full task spec is in [`docs/TASK_SPEC.md`](docs/TASK_SPEC.md), the interpretation
in [`findings.md`](findings.md) and the generated report in
[`results/summary.md`](results/summary.md).

## Headline results (D1-test, 4,293 crops)

| metric | value |
|---|---|
| accuracy | 0.880 |
| macro-F1 | 0.813 |
| balanced accuracy | 0.819 |
| train macro-F1 (overfitting check) | 0.939 |

Per-class recall, the confusion matrix and the training curves are in
[`results/d1_report.md`](results/d1_report.md).

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
| **`klshift/cropper.py`** | **the Cropper**: cuts "no damage" patches out of defect photos |
| **`run_cropper.py`** + **`cropper.yaml`** | runs the Cropper; all its settings are in the YAML |
| `make_synthetic.py` | tiny fake D1–D5 for a 2-minute pipeline check |
| `findings.md` | hand-written interpretation, inserted into `results/summary.md` on every run |
| `results/` | reports, CSV tables and figures from the last real run |
| `logs/` | console output of the last real runs |
| `data/` | **not in git** (~21 GB): raw downloads, crops, the trained model. See below |

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
    └── model_best.pt    ← the trained main ResNet-18 (45 MB)
```

`model_best.pt` and `results/embeddings.npz` (365 MB) are shared outside git (see the
top-level README). Put them back at those paths to skip retraining.

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

## Main choices

- **Crops.** One square crop per box: the box plus 25% on each side. The crop is clamped inside the image by shifting it, never padded, then resized to 224 and saved as JPEG q95, identically for all datasets.
- **Duplicates.** phash Hamming distance ≤ 6, or the same original file stem. Roboflow's `_jpg.rf.<hash>` suffix is stripped first, and the stem must be distinctive (at least 10 characters and not only digits).
- **D1 split.** 70/15/15 with `StratifiedGroupKFold`. A group joins crops of one image, tiles of one photo (`_rr_cc` suffix) and D1-internal near-duplicates.
- **KL.** PCA is fitted on D1-held-out. The Gaussian estimate uses Ledoit–Wolf, alongside a k-NN estimate (k = 5). Every comparison uses a common n (cap 2000) and 5 seeds. Reference halves are split by image group.
- **Sampling.** KL samples keep one image per within-dataset near-copy cluster (`--all-copies` turns this off) and at most one random crop per image (`--per-crop` turns this off). Sibling crops of one image are strongly correlated and otherwise inflate the k-NN noise floor about 100-fold.
- **Dataset versions.** D2 = Roboflow v22 (raw), D4 = v2, D5 = v3. D3 exists only as a 3× augmented export and is 98% D2, so it appears in the overlap table only.
- **No-damage sets.** D6 = FGVC-Aircraft 2013b (bottom 20 px banner removed; the official server is slow, so a copy of the same archive unpacked under `data/src/D6` is used when present), D7 = Roboflow aircraft-detection-model-citoe v15 (3× augmented; copies grouped by file name), D8 = Roboflow aircraft-kdg2w v11, D9 = Open Images V7 validation+test images with "Fixed-wing aircraft" boxes (group-of and depiction boxes dropped). Each has one box per plane. `D6p`–`D9p` are 2 skin patches per plane box, and `D1nd` is up to 2 patches per D1 image away from every defect box. Inside D6–D9, copies are grouped by file name only, because phash matches different planes photographed against plain sky.
- **D10.** Kaggle orvile/anomaly-detection-dataset-fuselage: fuselage mock-up video frames with a normal/anomaly label per frame, converted to two 720×720 squares per frame. `D10n`/`D10a` are its normal and anomaly frames. Video near-copies are grouped without chaining (`video_leaders` in `embed.py`).
