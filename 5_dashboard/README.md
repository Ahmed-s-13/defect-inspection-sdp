# 5. Dashboard: the model, the data and a test bench in the browser

A local web app for the project's main model (**enhanced_res gc7 + architecture B**: the calibrated decision layer
with a no-defect class and no image-quality check). It runs on your own computer:
a small Python server loads the model, and the page opens in your browser at http://127.0.0.1:8765.

| page | what it shows | needs |
|---|---|---|
| **Model** | the test results of the main model: accuracy, how it works (7 swipeable cards), per-class recall, the no-defect class, the sliding-window demo, its worst silent mistakes | gc7 weights + the D1 and no-defect crops |
| **Uncertainty** | calibration and conformal prediction, inferences first: how often the answer holds the true class (per class, with examples), what the percentages mean, how many classes to show, five crops followed through architecture B, alphas you can change, how the aircraft-or-not check works (interactive chart), what it refuses | gc7 weights + the D1 and no-defect crops (for the live report) |
| **Data** | how we went from D1 (leaky, 6 classes) to d1_gc7 (leak-free, 7 classes): grouping, the healer, the clean cropper, cropping; then every dataset we used (aircraft and not), with samples and architecture B's answer | nothing (built into the page) |
| **Inspect** | drop or paste one photo, drag the model's 224 px square, see architecture B's answer, the calibrated %, the 8 views and the unfamiliar-input check | gc7 weights |
| **Batch** | drop a folder (or run the 35 samples): accuracy with a 95% interval, confusion matrix, CSV export | gc7 weights |
| **Helmet** | SAQR Prototype 1 in 3D (`web/public/helmet/saqr_prototype_1.glb`), slowly turning: drag to turn it, click a component (or pick it from the list) to fly in and read what it is | nothing (built into the page) |

Keys: `1` to `6` switch pages · `P` presentation mode (`←` `→` step through a page) · `⌘V`/`Ctrl+V` paste an
image · arrow keys move the square once it has focus (Shift resizes).

## Setup on your computer (once)

Everything below is run from the **repo root** (`defect-inspection-sdp/`) unless it says otherwise.

**1. Python.** Python 3.13 and the repo's requirements (the dashboard's packages, FastAPI and uvicorn, are in the
same file). If you already have the repo's `.venv`, just re-run the `pip install` line to get the new packages.

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Node.js 20 or newer** (https://nodejs.org), to build the web page (the first time and after updates). `node -v` checks it.

**3. The model and the data.** Download these and unzip each one **in the repo root** (each zip already holds the
right folders, see the top-level README, "Getting the data and trained models"):

| file | where | needed for |
|---|---|---|
| `enhanced_res_gc7_model.zip` | GitHub **Releases**, `v3.0-gc7` | **everything** (the main model) |
| `main_model_data_work.zip` | team Google Drive | the Model page (the D1 test and validation crops) |
| `d1_gc7_no_defect_crops.zip` | team Google Drive | the Model page (the no-defect crops) |
| `trained_models.zip` | GitHub **Releases**, `v1.0-models` | optional: the legacy model in the model menu |

Without the two Drive zips, Inspect, Batch and Data still work; the Model page says which zip is missing. For the
legacy model also run `python train_mlp.py` once in `1_main_model_resnet18/` (about 40 s).

## Start it

macOS / Linux:

```bash
cd 5_dashboard
./run.sh
```

`run.sh` uses the repo's `.venv` (or the Python in `PYTHON=...`), builds the web page the first time and again
whenever `web/` changed, e.g. after a `git pull` (`npm install` + `npm run build`, about a minute), starts the server
and opens the browser. Stop it with `Ctrl+C`.

Windows (PowerShell, from the repo root, with the `.venv` active):

```powershell
cd 5_dashboard\web; npm install; npm run build; cd ..
python -m uvicorn server.app:app --host 127.0.0.1 --port 8765
```

then open http://127.0.0.1:8765. After a `git pull`, run the `npm` line again so the page matches the server.

**The first start on a new computer:** the Model page shows "Computing the report…" for about 2 minutes while the
model reads the 9,686 validation and test crops once (the corrected crops, d1_gc7_fix, that architecture B is
calibrated on; the 155 crops whose boxes were corrected on 27 Oct come with the repo, in
`1_main_model_resnet18/d1_grouped/label_corrections/crops/`, so no extra download is needed). The result is kept in
`5_dashboard/cache/`, so later starts take seconds. The model runs on CUDA,
Apple MPS or the CPU, whichever is there.

The top-right chip shows the model and its state (green: ready). If it is red, click it: it says which file is
missing and which zip gives it.

**The sample photos.** Inspect offers one test crop per class, picked because the model answers each one clearly,
so a demo shows every class working. They say nothing about accuracy. Batch's 35 samples are random test crops (one per
photo group), so its accuracy figure is honest. All of them come from the test split and were never trained on.

## Check that it works

- Batch → **Run the 35 samples**: 35 held-out test crops (5 per class, `no-defect_*` included). The top class is right
  for 31 of 35 (88.6%), including all 5 no-defect samples (3 are answered **No defect**, so the visor shows nothing;
  the other two are **Possible …: check** and go to a person, as architecture B intends when it is less sure).
- Model page: the headline reads **81.1%** (the right defect class on 4,253 unseen defect crops, corrected labels),
  macro-F1 **0.752** over all 7 classes, the answer holds the true class **90.4%** of the time, and 0.3% of defects
  are shown as no defect.
- Tests (optional):

```bash
cd 5_dashboard
python -m pytest tests -q        # 41 tests: registry, crops = predict.py, report = the team's numbers, API
cd web && npm test               # 43 tests: labels, metrics, geometry, format, the Data scene, uploads, reveals, the pipeline data
```

## Problems

| you see | do |
|---|---|
| `missing packages` when starting | `pip install -r requirements.txt` in the repo root, with the `.venv` active |
| `npm: command not found` | install Node.js 20+, then `REBUILD=1 ./run.sh` |
| `address already in use` | another copy is running; stop it, or `PORT=8766 ./run.sh` |
| red model chip: `missing enhanced_res_backbone.pt ...` | unzip `enhanced_res_gc7_model.zip` in the repo root |
| Model page: `needs the validation and test crops` | unzip `main_model_data_work.zip` and `d1_gc7_no_defect_crops.zip` in the repo root |
| page looks old after a `git pull` | `REBUILD=1 ./run.sh` (rebuilds the page) |

## How it is built

```
5_dashboard/
├── run.sh            start script
├── models.yaml       which models the dashboard serves, and which one is main (see below)
├── server/           FastAPI server; imports the model code from 1_main_model_resnet18 (read-only)
│   ├── app.py        the API (/api/predict, /api/batch, /api/report, /api/samples, ...) and the built page
│   ├── adapters/     one class per model type (enhanced_res, legacy)
│   ├── report.py     the Model page's numbers, from the model's own features; checked against the team's results
│   ├── states.py     architecture B's answer -> what the page shows (Defect detected, No defect, Ambiguous, ...)
│   └── preprocess.py the 224 px crop, exactly as predict.py cuts it
├── web/              the page: React + Vite + Motion (web/src), its pictures (web/public)
├── samples/          Batch: 5 no-defect test crops (the 30 defect ones come from 1_main_model_resnet18/samples/);
│                     samples/inspect/: Inspect's one clear test crop per class
├── scripts/          make the Data and Model pages' pictures and numbers (web/public, web/src/data)
├── tests/            pytest
├── DESIGN.md         colours, type and the rules the page follows
└── cache/            (not in git) the computed report and features
```

- Every number on the Model page is recomputed from the model, not typed in. The tests check that it matches the team's
  results (gc7 on the corrected crops, `enhanced_res/results/gc7_fix/compare.md`, and architecture B's
  `arch_a/calibration_gc7.json`).
- Nothing is chosen on the test split. The thresholds, temperature and cut-offs all come from validation or calibration
  data, as in the team's code.
- The server only reads the other folders; it writes only to `5_dashboard/cache/`.

**Changing the main model.** Add an entry to `models.yaml` (weights folder, calibration file, split, summary) and set
`main:`. A model with a new interface needs an adapter in `server/adapters/` (see `base.py`).

**Regenerating the page data** (only when the model or the datasets change; the results are in git):

| script | makes | needs, besides the gc7 weights |
|---|---|---|
| `python scripts/model_anatomy.py` | Model → How it works | the D1 crops |
| `python scripts/pipeline_scene.py` | Uncertainty → Follow a crop through architecture B, Try other alphas | the D1 and no-defect crops; the off-target features (local) for the aircraft-or-not chart |
| `python scripts/uncertainty_data.py` (after `pipeline_scene.py`) | Uncertainty → coverage by class, the aircraft-or-not chart; Data → every dataset we used | the D1, no-defect, D5 and FGVC crops; the off-target benchmark and its cached features (local); Open Images' validation metadata for the photo credits |
| `python scripts/sweep_crops.py` | Model → the sliding square | `D1_raw.zip` |
| `python scripts/healer_samples.py` | Data → the healer slider | `D1_raw.zip`, the healer's `no_defect/work/healed/` |
| `python scripts/clean_cropper_samples.py` | Data → the clean cropper and d1_gc7 | `D1_raw.zip`, `3_clean_cropper/work/` |
| `python scripts/no_defect_samples.py` | `samples/no-defect_*.jpg` (Batch, random) | the no-defect crops |
| `python scripts/inspect_samples.py` | `samples/inspect/` (Inspect's chips, picked for clarity) | the D1 and no-defect crops |

Run them from `5_dashboard/`, then `REBUILD=1 ./run.sh`.
