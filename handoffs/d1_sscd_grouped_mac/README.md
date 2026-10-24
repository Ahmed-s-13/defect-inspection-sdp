# d1_sscd_grouped — Mac handoff

This is a self-contained **fresh-training package**, prepared 22 October 2026. It has real images/labels and crops, the frozen SSCD split, adapted repository ResNet-18 code, ImageNet starting weights, and optional MLP code. It contains no symlinks, personal virtual environment, SSCD cache, or partly trained checkpoint. The sender's ongoing run is separate.

## Setup on your Mac

Install Python 3.14 first if needed. The included pinned environment was tested on an Apple Silicon Mac with Python 3.14; Intel Macs have not been validated and compatible PyTorch wheels may differ. Setup needs internet access to install packages. Training uses local files afterwards.

Extract the whole folder into its final location. In Terminal, type `cd `, drag this folder into Terminal, and press Return. Then run:

```sh
zsh setup_mac.command
```

This creates your own `.venv`, installs dependencies, sets paths for your Mac, and runs a disposable model/data readiness check. If moving the folder later, rebuild the virtual environment and rerun setup before training. Do not move it during or after training without explicitly migrating checkpoint provenance.

## Train the repository ResNet-18

```sh
caffeinate -is .venv/bin/python -u main_model/train.py --workers 4
```

Keep this Terminal open. The display may turn off; keep the lid open and Mac powered. For a detached offline-capable run after setup:

```sh
mkdir -p reports
nohup caffeinate -is .venv/bin/python -u main_model/train.py --workers 4 > reports/train.log 2>&1 < /dev/null &
```

Apple Silicon uses MPS when available; otherwise CPU (much slower). Up to 30 epochs, batch 64, AdamW learning rate 0.0001, weight decay 0.05, class-weighted loss and online augmentation. Early stop after 5 epochs without validation macro-F1 improvement. This is the original ResNet-18 baseline recipe; no family-balanced sampler, EMA, or learning-rate warm-up. The optional MLP is a separate training step.

Outputs: `data/d1_sscd_grouped/work/model_best.pt` and `data/d1_sscd_grouped/results/d1_report.md`, history CSV and confusion matrix. After training finishes, produce full per-class precision/recall/F1 plus family-weighted val/test results:

```sh
mkdir -p reports/resnet18_run
.venv/bin/python scripts/report_resnet.py
```

Read `reports/resnet18_run/README.md`. Do not tune hyperparameters using test scores.

Optional MLP/kNN head, after backbone training:

```sh
.venv/bin/python main_model/embed.py --workers 4
.venv/bin/python main_model/train_mlp.py
.venv/bin/python main_model/evaluate_mlp.py
```

## Dataset and split

14,760 raw images: **10,330 train / 2,216 val / 2,214 test**. Prepared crops: **20,046 / 4,295 / 4,296**. Photos are in `data/d1_sscd_grouped/images/{train,val,test}`; corresponding YOLO-format annotations are in `labels/`. The classifier learns from bounding-box crops; it is not a YOLO detector.

`manifests/split_manifest.csv` records exact image assignments. `reports/unique_sources/source_estimate_manifest.csv` records estimated families. The estimated 1,691 families and 13,069 excess copies (88.54%) are not certified augmentation provenance. Broader 720 split blocks keep uncertain matches together. Detected families do not cross splits, but undetected copies may remain. Validation/test still contain within-split copies; report family-weighted metrics alongside ordinary crop accuracy. No calibrated grouping confidence score is available.

The manifests are portable in the distributed folder; setup generates local absolute paths and a new checksum. Never regenerate or shuffle the split. No original `data/D1` directory is needed. `image` metadata can retain original dataset filenames; active file access uses `crop_path` or the new raw-image `path`.

This package supports running the model on the frozen SSCD split. Rebuilding SSCD grouping is outside this handoff; the full matching pipeline remains in the sender's experiment.
