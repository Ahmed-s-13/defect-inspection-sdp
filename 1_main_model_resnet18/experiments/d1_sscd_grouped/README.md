# D1 copy detection and grouped training

Start here. This folder contains the D1 photos, an SSCD duplicate audit, a saved grouped dataset, and a local copy of the main defect-classification model prepared to use that split.

This is the primary grouped-D1 experiment inside the project repository. Use **this folder's `main_model/` scripts** for grouped training. The older scripts two directories above remain the historical baseline.

## What SSCD is, in plain language

SSCD means **Self-Supervised Copy Detection** (the paper title is *A Self-Supervised Descriptor for Image Copy Detection*). It gives each photo a numerical fingerprint. Photos derived from the same original should have similar fingerprints even when someone changes the colour, crops them, or flips them.

It belongs to the same family of ideas as a Siamese network: during training, transformed images go through an encoder with shared weights, and contrastive learning encourages copies to be close and unrelated images to be farther apart. In our use, there is **one pretrained encoder**, applied once to each photo. We compare its outputs afterwards; we did not train a new two-image classifier on D1.

The SSCD model here uses **ResNet-50**, producing 512 numbers per photo. Its trained weights live in `models/sscd_disc_mixup.torchscript.pt`. Our Python scripts load that file. SSCD is separate from the aircraft-defect classifier.

Official source and model documentation: https://github.com/facebookresearch/sscd-copy-detection

## What was done

1. Extracted all 14,760 supplied D1 images and their bounding-box annotations into `data/D1/`.
2. Ran the pretrained SSCD model on every image and saved the fingerprints in `reports/sscd_embeddings.npy`.
3. Retrieved potential copies, tested flip/rotation behaviour, and viewed side-by-side contact sheets. Clear copies existed across D1's shipped splits.
4. Added perceptual hashes for all eight flip/quarter-turn orientations. Checked candidates using matching visual landmarks (SIFT + RANSAC), or very close pixel agreement plus matching hashes. The initial grouping examined 144,703 candidate pairs, skipping redundant checks once images were already connected.
5. Formed algorithmic source groups from accepted links. These are hypotheses about shared origin, not a guarantee that every group is perfect.
6. Built broader **split blocks**: images with high enough SSCD similarity are kept together even when we cannot confidently confirm their relationship. This deliberately prioritizes avoiding train/test overlap over maximizing the number of separate scenes. These blocks must not be described as the number of original photos.
7. Allocated whole blocks approximately 70/15/15, balancing class crop counts. No defect-model predictions or accuracy scores were used to select the split.
8. Checked lower-score candidates by aligning their images and measuring pixel agreement, and added explicit visual-review co-location constraints. Saved the split, raw-image/label links, and 224×224 defect crops. Checked all accepted links and all-pairs SSCD similarities across splits, then visually audited residual matches.

Earlier threshold-0.75 and threshold-0.50 drafts still had copies in visual review. Their metadata is preserved under `data/d1_sscd_grouped/drafts/`; they are **not the active split**. Later grouping also uses lower-score geometric/pixel matches and explicit review constraints. Current counts and limitations are in `data/d1_sscd_grouped/README.md` and `split_summary.json`.

This is a practical, audited split, not a proof that every augmented sibling has been identified. The saved evidence makes it possible to improve the grouping without guessing.

## Where to look

| Path | What you will find |
|---|---|
| `data/d1_sscd_grouped/README.md` | Active split counts, class balance, checks, and limitations |
| `data/d1_sscd_grouped/images/train/`, `val/`, `test/` | Photos arranged by the new split; these are links to the original local photos |
| `data/d1_sscd_grouped/labels/` | Corresponding YOLO-format bounding-box labels |
| `data/d1_sscd_grouped/split_manifest.csv` | One row per photo: original split, new split, source group, split block, checksum |
| `data/d1_sscd_grouped/work/d1_split.csv` | The crop-level split actually read by ResNet-18 training |
| `data/d1_sscd_grouped/work/crops/D1/` | Usable 224×224 defect crops |
| `main_model/README.md` | Model architecture, defaults, commands, and local changes |
| `reports/FINDINGS.md` | Historical first SSCD pilot; not the current split report |
| `reports/review_01.jpg` … `review_06.jpg` | Original side-by-side duplicate candidates |
| `reports/final_split_review/` | Largest-group and residual cross-split review sheets for the active split |
| `data/d1_sscd_grouped/pair_evidence.csv` | Pair-check evidence; skipped same-component pairs need no new link |
| `scripts/` | Python files that run the audit and prepare/check the grouped data |
| `models/` | Pretrained SSCD and ImageNet ResNet-18 weights |

You can open the JPEG sheets in Finder or click them in Codex. A CSV opens as a table in a spreadsheet app or as text in an editor. Do not edit the active split CSV to experiment: its checksum is used to keep model checkpoints and embeddings consistent.

## Run the main model

Open a terminal in this folder, or run:

```sh
cd "/Users/wbk/Documents/GitHub/defect-inspection-sdp/1_main_model_resnet18/experiments/d1_sscd_grouped"
```

The `.venv` environment and required packages are already installed. First verify the data:

```sh
.venv/bin/python scripts/check_ready.py
```

Then train a fresh **ResNet-18** on the saved grouped split:

```sh
.venv/bin/python main_model/train.py --workers 0
```

The training script reads the existing split; it does not create a new one. It starts from ImageNet weights, not the old D1 checkpoint. Default: up to 30 epochs, early stopping after 5 epochs without validation macro-F1 improvement. It selects CUDA, Apple MPS, or CPU in that order, depending on availability.

To add the MLP ensemble and nearest-neighbour head after ResNet training:

```sh
.venv/bin/python main_model/embed.py --workers 0
.venv/bin/python main_model/train_mlp.py
```

Evaluate or predict after training:

```sh
.venv/bin/python main_model/evaluate.py --workers 0
.venv/bin/python main_model/evaluate_mlp.py
.venv/bin/python main_model/predict.py "/absolute/path/to/defect_crop.jpg"
```

Prediction automatically uses the MLP ensemble + kNN **if its matching checkpoint exists**. Otherwise it uses the ResNet-18 linear classifier and prints a note. `--resnet-only` explicitly selects the linear classifier.

A first training attempt was interrupted before any completed epoch or trained checkpoint. The data-loading adjustment is prepared; training has not yet been restarted here. The readiness check performs one disposable training batch and saves no trained model. Old reported accuracy is not a result for this new split.

## Re-run SSCD work

```sh
.venv/bin/python scripts/sscd_audit.py
.venv/bin/python scripts/sscd_transform_probe.py
.venv/bin/python scripts/verify_review.py
```

The first command reuses cached fingerprints. Do not reuse that cache if images, ordering, model, or preprocessing change. The transformation probe uses 32 selected source images and is not a real-world recall estimate.

`build_groups.py` and `prepare_grouped.py` document how grouping and split generation were done. The split writer refuses to overwrite an existing frozen manifest. Create a separately versioned experiment for future regrouping; do not regenerate it while comparing models. Dataset preparation used `main_model/prepare.py --raw data --work data/d1_sscd_grouped/work --datasets D1 --workers 1`; do not rerun this over the active prepared inventory because it replaces crop metadata.

## Portability

Keep `data/D1` with `data/d1_sscd_grouped`: the raw split folders contain relative symbolic links. Crop manifests and `data.yaml` currently use absolute workspace paths. If moving to another computer, copy the source images and crops, rewrite those paths, and record the new split checksum before training. `requirements.txt` records the installed environment.

## Unique-source estimate

The matching-based estimate is **1,691 source families and 13,069 excess copies (88.54%)**. See `reports/unique_sources/README.md` for the overall and per-class table, definitions, and limits. These estimates exclude the broad precautionary split-block merges.
