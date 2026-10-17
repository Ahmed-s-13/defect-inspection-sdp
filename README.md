# Edge-AI Wearable for Visual Inspection of Aircraft Structures: model code

Senior Design Project, American University of Sharjah, with Gulf Aviation Engineering
Services (GAES). A wearable (target: Jetson Orin NX class) runs an onboard model that
tells the engineer whether the aircraft surface patch in view is fine or damaged, and
which defect it is. Everything here uses public datasets.

## Repository map

| folder | status | what it is |
|---|---|---|
| [`1_main_model_resnet18/`](1_main_model_resnet18/) | **MAIN MODEL, in use** | ResNet-18 defect classifier trained on D1 (6 classes), the dataset-shift (KL divergence) analysis across 10 public datasets, and **the Cropper** for "no damage" patches. Most of our data and results live here. |
| [`2_healing_experiment/`](2_healing_experiment/) | experiment, in progress | "Heal" defect crops by inpainting to create non-defective training patches. Builds on folders 1 and 3. |
| [`3_teacher_student_msrd/`](3_teacher_student_msrd/) | support code | MSRD teacher–student anomaly model, used by the healing experiment. |
| [`4_early_classifiers_archive/`](4_early_classifiers_archive/) | archived | Our first models: plain CNN and SupCon ResNet-18, on smaller Roboflow datasets. |

Each folder has its own README with the run order. Run scripts from inside their folder.

```
1_main_model_resnet18  ──(D1 crops, split, trained model, KL code)──▶  2_healing_experiment
3_teacher_student_msrd ──(MSRD model)─────────────────────────────────▶  2_healing_experiment
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code picks the device as CUDA, then Apple MPS, then CPU.

## Data and trained models (not in git)

The datasets are about 23 GB, so they are **not** in this repository. GitHub rejects
files over 100 MB and slows down badly past a few GB.

| what | size | where it goes | how to get it |
|---|---|---|---|
| raw datasets D1–D10 | ~18 GB | `1_main_model_resnet18/data/raw`, `data/src` | `python download.py --get …` (see folder 1 README; needs your own Kaggle/Roboflow keys) |
| crops, D1 split, hashes | ~3 GB | `1_main_model_resnet18/data/work` | `python prepare.py` (or the team's shared drive copy) |
| **main model weights** `model_best.pt` | 45 MB | `1_main_model_resnet18/data/work/` | GitHub Release of this repo |
| D1 embeddings `embeddings.npz` | 365 MB | `1_main_model_resnet18/results/` | `python embed.py`, or the shared drive |
| healing work files | ~1.4 GB | `2_healing_experiment/work` | re-run steps 1–3 of the healing experiment |

Small outputs (reports, CSV tables, figures) **are** in git under each folder's `results/`.

## API keys

No keys are stored in this repository. Scripts that download data read them from
environment variables (`ROBOFLOW_API_KEY`, `KAGGLE_USERNAME`, `KAGGLE_KEY`) or from
`~/.kaggle/kaggle.json`, which is outside the repo. `.env` and `kaggle.json` files are
git-ignored. Never paste a key into a script, notebook or README.

## Dataset licensing

D1 (Kaggle Aircraft_dataset) has an unknown license and is assembled from other
sources. Some result figures show crops from it. Check this before making the repo public.
