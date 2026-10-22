# Main model on the grouped D1 split

This is a local adaptation of `1_main_model_resnet18` from the original GitHub checkout. It is ready to train using `../data/d1_sscd_grouped/`; no full grouped-data training run has been performed yet.

## Which network does what?

| Component | Network / method | Purpose |
|---|---|---|
| Duplicate detector | Pretrained SSCD ResNet-50 | Find images derived from the same photo; only used to prepare the split |
| Main defect classifier | ImageNet-pretrained torchvision ResNet-18 | Convert a 224×224 defect crop into 512 features and six defect scores |
| Baseline head | Linear layer, 512 → 6 | Trained together with ResNet-18 by `train.py` |
| Optional improved head | Five MLPs, each 512 → 1024 → 512 → 256 → 6 | Learn a more flexible boundary using frozen ResNet-18 features |
| Optional kNN blend | Cosine 5-nearest-neighbour vote over training features | Blended with the MLP probabilities; weight chosen on validation |
| YOLO | Annotation format only | Supplies box coordinates; no YOLO detector is trained or used by these scripts |

The classes are corrosion, crack, dent, missing-head, paint-peel-off, and scratch. This is classification of an already selected defect area, not object detection or segmentation. There is no clean/no-damage class.

## Is MLP used by default?

- `train.py` trains ResNet-18 and its **linear head only**.
- `embed.py` extracts fresh features from that trained ResNet-18.
- `train_mlp.py` trains the **five-MLP ensemble** and chooses its kNN blend using validation. This is a separate step.
- `predict.py` uses `mlp_head.pt` automatically when it exists and matches the backbone/split. Otherwise it reports a fallback to the linear head. Use `--resnet-only` to force the baseline.

The original run used a 50/50 MLP/kNN blend. On this split, the weight is selected again; it must not be assumed to remain 0.5.

## Commands from the workspace root

```sh
.venv/bin/python main_model/train.py --workers 0
.venv/bin/python main_model/embed.py --workers 0
.venv/bin/python main_model/train_mlp.py
.venv/bin/python main_model/evaluate.py --workers 0
.venv/bin/python main_model/evaluate_mlp.py
.venv/bin/python main_model/predict.py "/absolute/path/to/photo.jpg" --box 120 80 260 190
```

`--box` means x1 y1 x2 y2 pixel coordinates. Without a box, prediction uses the centre square, so use close-up defect crops rather than wide aircraft photos.

Run `--help` on any script for options. Defaults resolve to this workspace, regardless of the current working directory. The virtual environment path in the commands assumes you are in the workspace root.

## Inputs and outputs

- Frozen crop manifest: `../data/d1_sscd_grouped/work/d1_split.csv`.
- Training images: paths in that manifest, under `work/crops/D1/`.
- Fresh backbone checkpoint: `../data/d1_sscd_grouped/work/model_best.pt`.
- Backbone training/evaluation report: `../data/d1_sscd_grouped/results/d1_report.md`.
- Embeddings: `../data/d1_sscd_grouped/results/embeddings.npz`.
- MLP checkpoint: `../data/d1_sscd_grouped/work/mlp_head.pt`.
- MLP results: `../data/d1_sscd_grouped/results/mlp/`.

The old supplied D1 checkpoint is deliberately not installed as the new experiment's model. It may already have learned from new test scenes.

## Changes from the repository

1. `train.py` loads and validates the frozen grouped manifest instead of regrouping and re-splitting data.
2. Default work/results paths point to the local grouped experiment.
3. ResNet checkpoint records the split SHA256. Evaluation rejects checkpoints from another split.
4. `embed.py` is a focused D1-only extractor; it records the split and backbone checksums.
5. MLP training rejects stale embeddings. MLP loading rejects a stale head after the split or backbone changes.
6. ImageNet weights and matplotlib caches stay inside this workspace.

The backbone architecture and original training recipe remain the baseline: AdamW, class-weighted cross-entropy, and the original image augmentations. The previously described scene-balanced sampler, EMA, warm-up, mid-layer features, and eight-view test-time augmentation are **not implemented in this baseline**. The MLP retains its original optional `--aug-views`, feature noise, dropout, and label smoothing.

Test results are for final reporting. Choose hyperparameters using validation; repeated changes based on test scores would undermine the split.

## Evaluation interpretation

Grouping does not delete augmentation copies. Several crops from one scene may still appear within the same test split. The inherited reports compute crop-level metrics, so scenes with many copies carry more weight. For a final scientific report, also aggregate by `group_id`/`split_block` and bootstrap whole blocks. A new crop-level accuracy should not be treated as accuracy on that many independent scenes.

Crop preparation matches the original code: each box becomes a square with 25% margin on each side, shifted inside the image, resized to 224 pixels. Boxes with either side under 4 pixels are omitted. That leaves 28,637 crops from 30,485 annotations; 145 source images contribute no usable classifier crops but remain in the raw split.
