# Edge-AI Wearable for Visual Inspection of Aircraft Structures: model code

## Main model — 24 October 2026

**The defect classifier in use is [`enhanced_res`](1_main_model_resnet18/enhanced_res/README.md)**: a ResNet-18 +
MLP head trained and tested on [`d1_grouped`](1_main_model_resnet18/d1_grouped/README.md), a leak-free re-split
of D1 (no photo or copy of it in both train and test). Grouped D1-test accuracy **80.4%**, macro-F1 **0.738**,
against 78.1% / 0.715 for the previous ResNet-18 + MLP recipe retrained on the same split. Run it with
`python predict.py` in `1_main_model_resnet18/`; the weights are `enhanced_res_model.zip` on this repo's **Releases** page (`v2.0-enhanced-res`).

## Current D1 grouping work — 22 October 2026

<!-- SSCD_RESNET_RUN_START -->
**ResNet-18 training and evaluation completed.** Best epoch 10 of 15. Test accuracy **72.18%**, macro-F1 **0.6739**; equal-source-family accuracy **74.05%**, macro-F1 **0.6929**. [Saved weights and provenance](1_main_model_resnet18/experiments/d1_sscd_grouped/models/resnet18_sscd_grouped/README.md). [Overall and per-class results](1_main_model_resnet18/experiments/d1_sscd_grouped/reports/resnet18_run/README.md), including both ordinary crop metrics and equal-source-family metrics. [Run status](1_main_model_resnet18/experiments/d1_sscd_grouped/reports/resnet18_run/status.json).
<!-- SSCD_RESNET_RUN_END -->

The primary SSCD/grouped-data experiment is now in **[1_main_model_resnet18/experiments/d1_sscd_grouped/](1_main_model_resnet18/experiments/d1_sscd_grouped/README.md)**. It contains the runnable code, local data, frozen split, model integration, audits and progress status.

- [Estimated unique sources and copies, including per class](1_main_model_resnet18/experiments/d1_sscd_grouped/reports/unique_sources/README.md): **1,691 estimated source families; 13,069 excess copies (88.54%)**. This is a matching-based estimate, not verified synthetic-augmentation provenance.
- [Grouped model instructions](1_main_model_resnet18/experiments/d1_sscd_grouped/main_model/README.md): ResNet-18, with an optional MLP/kNN head. SSCD uses a separate ResNet-50. YOLO is the annotation format only.
- [Frozen portable manifests](1_main_model_resnet18/experiments/d1_sscd_grouped/manifests/README.md): 10,330 / 2,216 / 2,214 raw images and 20,046 / 4,295 / 4,296 usable crops in train/val/test.
- Current training status and results are recorded in the run section above.

**Alternative grouping.** A second leak-free split, `d1_grouped` (861 groups; phash + ResNet-50 + SIFT), is kept locally at `1_main_model_resnet18/experiments/d1_grouped/` and is git-ignored. [Comparison with the SSCD split](1_main_model_resnet18/experiments/d1_sscd_grouped/reports/compare_d1_grouped/README.md): SSCD strict families nest inside d1_grouped groups (99.6%), but the two splits agree on only 53.9% of images.

A candidate **union split** (SSCD blocks + d1_grouped groups) and a Siamese leakage audit are described [here](1_main_model_resnet18/experiments/d1_sscd_grouped/README.md); the union split has the fewest near-duplicate train matches of the three grouped splits.

<!-- SSCD_COPY_ESTIMATES_START -->
## D1 SSCD augmentation/copy estimates

Dataset: **`d1_sscd_grouped`**. Counts refer to raw images, before bounding-box cropping. Estimated augmented/copy images means **images minus one representative per matched source family**. This includes crops, re-encodings and near-copies; synthetic augmentation provenance and the true original file are not known. The 720 broader split blocks are used only to prevent split leakage, not to count unique sources.

**Overall: 14,760 images; 1,691 estimated source families; 13,069 excess copies (88.54%).**

### Per class

| Class | Images | Estimated source families | Estimated augmented/copy images | Estimated copy % |
|---|---:|---:|---:|---:|
| crack | 6,270 | 610 | 5,660 | 90.27% |
| dent | 6,909 | 731 | 6,178 | 89.42% |
| corrosion | 1,267 | 192 | 1,075 | 84.85% |
| scratch | 497 | 255 | 242 | 48.69% |
| missing-head | 908 | 313 | 595 | 65.53% |
| paint-peel-off | 822 | 299 | 523 | 63.63% |

### Active train/validation/test split

| Split | Images | Estimated source families | Estimated augmented/copy images | Estimated copy % |
|---|---:|---:|---:|---:|
| train | 10,330 | 1,238 | 9,092 | 88.02% |
| val | 2,216 | 241 | 1,975 | 89.12% |
| test | 2,214 | 212 | 2,002 | 90.42% |

These percentages measure redundancy **within each split**, not leakage between splits. All estimated families remain within one split; undetected relatives may still exist. Percentages use each row’s image count as denominator. Multi-class images/families appear in multiple class rows, so class rows do not sum to the total.

**Confidence: not calibrated.** No defensible numerical confidence score or confidence interval is available. SSCD similarity is not a probability. Missed matches overestimate unique families; false merges underestimate them. Treat these as matching-based estimates, not ground truth.

The reproducible report also provides `per_split_class.csv` for each class within each split.

See [full SSCD report](1_main_model_resnet18/experiments/d1_sscd_grouped/reports/unique_sources/README.md) and [class-by-split CSV](1_main_model_resnet18/experiments/d1_sscd_grouped/reports/unique_sources/per_split_class.csv).
<!-- SSCD_COPY_ESTIMATES_END -->

The older accuracy results below are historical and may be inflated by undetected augmentation overlap. They are not results on the new grouped split. Bulk data, weights, caches and environments remain git-ignored; code, documentation, manifests and compact reports are reviewable in git.


Senior Design Project, American University of Sharjah, with Gulf Aviation Engineering
Services (GAES). A wearable (target: Jetson Orin NX class) runs an onboard model that
tells the engineer whether the aircraft surface patch in view is fine or damaged, and
which defect it is. Everything here uses public datasets.

## Repository map

| folder | status | what it is |
|---|---|---|
| [`1_main_model_resnet18/`](1_main_model_resnet18/) | **MAIN MODEL, in use** | **enhanced_res** ([`enhanced_res/`](1_main_model_resnet18/enhanced_res/)): ResNet-18 + MLP head for 6 defect classes, trained on the leak-free [`d1_grouped/`](1_main_model_resnet18/d1_grouped/) split (grouped D1-test accuracy 0.804). Also the legacy ResNet-18 + MLP head, the dataset-shift (KL divergence) analysis across 10 public datasets, and **the Cropper** for "no damage" patches. Most of our data and results live here. |
| [`2_lama_healing/`](2_lama_healing/) | **healer, in use** | Removes defects from photos with **LaMa** (pretrained inpainting) and puts a real screw head back where one is missing (copied from the same photo or a donor bank). GUI + batch command line. Replaces the earlier healing experiment (U-Net inpainter, still in git history). |
| [`3_teacher_student_msrd/`](3_teacher_student_msrd/) | support code | MSRD teacher–student anomaly model, used by the earlier healing experiment (now removed). |
| [`4_early_classifiers_archive/`](4_early_classifiers_archive/) | archived | Our first models: plain CNN and SupCon ResNet-18, on smaller Roboflow datasets. |

Each folder has its own README with the run order. Run scripts from inside their folder.

```
1_main_model_resnet18  ──(enhanced_res backbone, D1 photos + labels)──▶  2_lama_healing
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code picks the device as CUDA, then Apple MPS, then CPU.

## Getting the data and trained models

The datasets are about 23 GB, so they are **not** in this repository. GitHub rejects
files over 100 MB and slows down badly past a few GB. Small outputs (reports, CSV
tables, figures) **are** in git under each folder's `results/`.

Instead, three zip files are shared. Each one is laid out relative to the repo root,
so you unzip it **inside the cloned repo folder** and every file lands where the code expects it.

| file | size | where to download | what it gives you |
|---|---|---|---|
| `enhanced_res_model.zip` | 136 MB | this repo's **Releases** page (`v2.0-enhanced-res`) | **the main model**: `data/work/enhanced_res/` (backbone + MLP head) |
| `trained_models.zip` | 69 MB | this repo's **Releases** page (`v1.0-models`) | the legacy ResNet-18 (`model_best.pt`), the MSRD teacher–student and the checkpoints of the earlier (removed) healing experiment |
| `lama_weights.zip` | 206 MB | team shared Google Drive | the healer's LaMa model: `2_lama_healing/weights/big-lama.pt` |
| `main_model_data_work.zip` | 2.8 GB | team shared Google Drive (ask a team member for access) | all crops, the D1 split, hashes, embeddings: enough to evaluate, embed and run KL |
| `D1_raw.zip` | 756 MB | team shared Google Drive | the original D1 photos + labels: needed for the Cropper, the healer's `--d1` mode and rebuilding d1_grouped |
| `d1_grouped` folder | ~850 MB | team shared Google Drive | the d1_grouped split as a ready YOLO dataset; put its `dataset/` in `1_main_model_resnet18/data/d1_grouped/`. Not needed for the ResNet: the split itself (`d1_grouped/groups.csv`) is in git |

Google Drive tips: download each zip on its own rather than the whole folder (Drive would
wrap them in another zip), and for the 2.8 GB file click **Download anyway** when Drive says
it can't scan it for viruses.

```bash
git clone https://github.com/Ahmed-s-13/defect-inspection-sdp.git
cd defect-inspection-sdp
# put the downloaded zips in this folder, then:
unzip enhanced_res_model.zip
unzip trained_models.zip
unzip main_model_data_work.zip
unzip D1_raw.zip
shasum -a 256 -c SHA256SUMS.txt   # optional: check the downloads (file is next to the zips)
```

Then check that everything works:

```bash
cd 1_main_model_resnet18
python enhanced_res/evaluate.py      # main model; should end with "OK: results match the original enhanced_res run."
python predict.py samples/ --compare
```

The legacy model is checked the same way; its small MLP head is built
locally (about 40 s):

```bash
python evaluate.py              # should end with "OK: results match the original training run"
python train_mlp.py             # ~40 s: builds the MLP head (data/work/mlp_head.pt), not in the zips yet
python evaluate_mlp.py          # should end with "OK: results match the original MLP training run."
```

The other raw datasets (D2–D10, ~17 GB) are only needed to re-run `prepare.py` from
scratch. Download them yourself with `1_main_model_resnet18/download.py` and your own
free Kaggle/Roboflow keys (see that folder's README).

## API keys

No keys are stored in this repository. Scripts that download data read them from
environment variables (`ROBOFLOW_API_KEY`, `KAGGLE_USERNAME`, `KAGGLE_KEY`) or from
`~/.kaggle/kaggle.json`, which is outside the repo. `.env` and `kaggle.json` files are
git-ignored. Never paste a key into a script, notebook or README.

## Dataset licensing

D1 (Kaggle Aircraft_dataset) has an unknown license and is assembled from other
sources. Some result figures show crops from it. Check this before making the repo public.

## Portable Mac handoff

[Mac handoff instructions](handoffs/d1_sscd_grouped_mac/README.md): a self-contained fresh-training package with real images, frozen split, model code and Mac setup. A local ZIP is available at `handoffs/d1_sscd_grouped_mac.zip`. It excludes the ongoing run’s checkpoint.
