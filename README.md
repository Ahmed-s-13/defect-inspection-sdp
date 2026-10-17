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

## Getting the data and trained models

The datasets are about 23 GB, so they are **not** in this repository. GitHub rejects
files over 100 MB and slows down badly past a few GB. Small outputs (reports, CSV
tables, figures) **are** in git under each folder's `results/`.

Instead, three zip files are shared. Each one is laid out relative to the repo root,
so you unzip it **inside the cloned repo folder** and every file lands where the code expects it.

| file | size | where to download | what it gives you |
|---|---|---|---|
| `trained_models.zip` | 69 MB | this repo's **Releases** page (`v1.0-models`) | the main ResNet-18 (`model_best.pt`), the MSRD teacher–student and the healer checkpoints |
| `main_model_data_work.zip` | 2.8 GB | team OneDrive folder (link shared privately) | all crops, the D1 split, hashes, embeddings: enough to evaluate, embed, run KL and healing steps 2–4 |
| `D1_raw.zip` | 756 MB | team OneDrive folder | the original D1 photos + labels: needed for the Cropper and healing step 1 |

```bash
git clone https://github.com/Ahmed-s-13/defect-inspection-sdp.git
cd defect-inspection-sdp
# put the downloaded zips in this folder, then:
unzip trained_models.zip
unzip main_model_data_work.zip
unzip D1_raw.zip
shasum -a 256 -c SHA256SUMS.txt   # optional: check the downloads (file is next to the zips)
```

The other raw datasets (D2–D10, ~17 GB) are only needed to re-run `prepare.py` from
scratch. Download them yourself with `1_main_model_resnet18/download.py` and your own
free Kaggle/Roboflow keys (see that folder's README). The healing experiment's
`work/` files (~1.4 GB) are not shared; re-create them by running its steps 1–3.

## API keys

No keys are stored in this repository. Scripts that download data read them from
environment variables (`ROBOFLOW_API_KEY`, `KAGGLE_USERNAME`, `KAGGLE_KEY`) or from
`~/.kaggle/kaggle.json`, which is outside the repo. `.env` and `kaggle.json` files are
git-ignored. Never paste a key into a script, notebook or README.

## Dataset licensing

D1 (Kaggle Aircraft_dataset) has an unknown license and is assembled from other
sources. Some result figures show crops from it. Check this before making the repo public.
