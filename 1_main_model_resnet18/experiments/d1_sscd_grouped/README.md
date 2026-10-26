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

## Alternative grouping: `d1_grouped` (local only, git-ignored)

A second, independently built leak-free split of the same 14,760 D1 images lives next to this folder in [`../d1_grouped/`](../d1_grouped/README.md). It is **git-ignored** (3.2 GB with images and cache); only the comparison below is tracked. It was built with perceptual hashes and ResNet-50 features over all 8 flips/rotations, SIFT+RANSAC confirmation, union-find grouping (861 groups), and a 70/15/15 group-level split with a repeat-until-clean leak check. It is similar in spirit to the SSCD pipeline here, but uses different fingerprints and thresholds, and a different split.

Comparison (script `scripts/compare_d1_grouped.py`, full report [`reports/compare_d1_grouped/README.md`](reports/compare_d1_grouped/README.md)):

- **Splits differ a lot**: only 53.9% of images land in the same split in both (val overlap 5.7%, test overlap 9.3% Jaccard). Both are independent random group-level allocations, so scores on one test set are not comparable to the other.
- **SSCD strict groups nest inside d1_grouped groups**: 99.6% of the 1,701 SSCD strict groups lie wholly inside one d1_grouped group (every SSCD pair is also a d1_grouped pair). d1_grouped is coarser: its 861 groups merge several SSCD families each.
- **SSCD split blocks vs d1_grouped groups**: 93.4% of d1_grouped groups sit inside one SSCD block, but the SSCD blocks are the more conservative unit (35% of SSCD-block pairs are co-grouped in d1_grouped).
- **Cross-leakage**: 6 SSCD strict groups (41 images) straddle splits of d1_grouped; 28 d1_grouped groups (2,883 images) straddle splits of the SSCD split. Neither split is leak-free by the other's standard, though the SSCD blocks are the stricter unit.

### Union split and Siamese audit

- **Union split** (`manifests/union_split.csv`, `union_split.portable.csv`; `scripts/build_union_split.py`): images linked by *either* an SSCD split block or a d1_grouped group, then whole components split 70/15/15 (613 components, largest 1,924 images; 10,860 / 1,908 / 1,992 images; 19,970 / 4,376 / 4,291 crops; corrosion is 66.9/18.2/14.9). No SSCD or d1_grouped group straddles it. It is a candidate, not yet the active split: the frozen SSCD split and its checksum are unchanged.
- **Siamese audit** (`scripts/siamese_audit.py`, [report](reports/siamese_audit/README.md)): a shared-weight ResNet-18 trained contrastively on SSCD strict groups (15% held out; held-out nearest-neighbour hit rate 99.8%). Share of val+test images with a train match scoring >= 0.90: original D1 split 96.6%, SSCD 1.3%, d1_grouped 5.2%, **union 0.4%**. The top flagged union pairs are mostly same-hardware, different-scene crops, a few possible same-object re-shoots. Because the Siamese learned from the same groupings, it is a consistency check, not independent ground truth.

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

### Train on any split (detached, GPU, stays awake)

```sh
.venv/bin/python scripts/train_on_split.py union        # or: original | sscd | d1_grouped
```

It refuses to start without the Apple GPU (MPS), runs `main_model/train.py` in its own session under `caffeinate -ims` (survives screen-off and closing the terminal/app), and writes logs and `status.json` to `reports/runs/<split>/` and weights to `data/splits/<split>/work/model_best.pt`. Keep the Mac on AC power: lid-close sleep, power loss and shutdown cannot be prevented. Extra arguments go to `train.py` (e.g. `--epochs 30`).

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

The earlier attempt was interrupted. The current detached ResNet-18 run is documented in [reports/resnet18_run/](reports/resnet18_run/README.md); consult status.json there for its current state. The readiness check performs one disposable training batch and saves no trained model. Old reported accuracy is not a result for this new split.

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

## Completed trained model

ResNet-18 training is complete: best epoch 10 of 15. [Results and artifacts](reports/resnet18_run/README.md); [archived model](models/resnet18_sscd_grouped/README.md). Test accuracy 72.18%, macro-F1 0.6739; equal-source-family accuracy 74.05%, macro-F1 0.6929. No MLP was trained.
