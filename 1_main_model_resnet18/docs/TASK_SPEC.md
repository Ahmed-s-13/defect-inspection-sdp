# Project context and task spec: dataset shift (KL divergence) with the D1 ResNet-18

## The project

**Edge-AI Wearable for Detailed Visual Inspection of Aircraft Structures.** This is a Senior Design Project (ELE 490) at the American University of Sharjah, run with Gulf Aviation Engineering Services (GAES). The team has four members and no leader, so write "we", not "I", and don't single anyone out in project documents.

- **Concept.** A standalone helmet (or another wearable) runs an onboard model. It tells the engineer whether the patch of aircraft surface they are looking at is fine or needs maintenance. Each finding comes with a confidence (e.g. "CRACK DETECTED / Confidence: 96%"). The helmet has no wireless link to anything; alerts are voice, vibration or LED. The target hardware is a Jetson Orin NX class computer.
- **Requirement.** The model must be very accurate with a very low error rate, because it is meant for real maintenance work.
- **Plan.** First, build a full model with no size or compute limits and get it to agreed evaluation criteria. Then build a device-constrained model, then integrate it into the helmet. The full model should be done by the end of Fall 2026 (around end of November).
- **Data.** GAES will not provide a dataset, so everything uses public datasets.
- **Professor's guidance.** Keep the network direct and simple; GAES cares that it works, not that it is fancy. The four risks to watch are:
  - overfitting
  - shortcut learning (e.g. the model learns "wing = corrosion")
  - distribution shift from image conditions
  - sampling problems from uneven class counts

  PCA was suggested for dimensionality reduction.

**Starting point: nothing exists yet.** There is no code, no downloaded data and no trained model. Build everything from scratch in this folder.

## Hardware

The code runs on a MacBook with an Apple GPU (PyTorch MPS backend). Pick the device in this order: `cuda`, then `mps`, then `cpu`. Keep models laptop-sized; ResNet-18 is the backbone.

## The five datasets (D1 is the reference)

| ID | name | source | task / size | classes | license |
|---|---|---|---|---|---|
| **D1** | Kaggle Aircraft_dataset | kaggle.com/datasets/sakettiw11/aircraft-dataset | detection (YOLO), 14,760 images, 30,485 boxes, ~840 MB, ships train/val/test 11,808 / 1,476 / 1,476 | Crack, Dent, Corrosion, Scratch, Missing-Head, Paint-Peel-Off | "Unknown" (research/education) |
| D2 | Aircraft-Surface-Damage | universe.roboflow.com/lemi-debele/aircraft-surface-damage | detection, 1,200 raw images | crack, dent, corrosion | CC BY 4.0 |
| D3 | Aircraft (lemi-debele) | universe.roboflow.com/lemi-debele/aircraft-m3c6e | detection, 909 images, 1 version | crack, dent, corrosion | CC BY 4.0 |
| D4 | 6-class wind turbine | universe.roboflow.com/123-hmdgs/-6-classes-wind-turbine-dataset | detection, drone tiles of wind-turbine blades (not aircraft) | 6 classes incl. cracking, surface corrosion, leading edge corrosion, paint peeling | CC BY 4.0 |
| D5 | Aircraft Defect Detection (UTS) | universe.roboflow.com/university-of-technology-sydney-21uto/aircraft-defect-detection | detection, 9,352 images, 2 versions | Dent, Fastener Damage, Rupture | Public Domain |

**Downloading.**

- **D1** comes from Kaggle. Use the Kaggle CLI (`kaggle datasets download -d sakettiw11/aircraft-dataset`, which needs the user's Kaggle API token) or a zip the user downloads by hand.
- **D2–D5** come from Roboflow. Use the `roboflow` Python package with the user's API key and export in YOLO format, or use zips the user downloads by hand.
- **Credentials.** The user supplies credentials at run time. Never write them into files or commit them.

### Known traps in these datasets

1. **All five are detection datasets, not classification datasets.** They have bounding boxes, and one image can contain several defect types.
2. **Watch for augmented Roboflow versions.** These are augmented copies of the same originals and inflate the image count. For D2, version 22 is the raw 1,200-image set (versions 24/25 are augmented, about 2,900 images). D5's 9,352 images may also include augmented copies, so check both versions and use the raw one.
3. **D4's shipped split leaks.** Filenames like `DJI_0688_04_02` and `DJI_0688_04_03` are tiles cut from the same drone photo, and the shipped split puts siblings in both train and test.
4. **The datasets overlap. This is the most important fact for this task.**
   - D1's description says it was assembled from 24 public aircraft-defect datasets from Kaggle, Roboflow Universe, papers and repos, with class remapping and duplicate removal. So it very likely contains images from D2, D3 and D5.
   - D2 and D3 come from the same author and share filename patterns (e.g. `IMG_20230513_*.jpg`). D3 is probably a subset or earlier version of D2.
   - D5 also contains `IMG_20230511_*` / `IMG_20230513_*` files, so it likely shares photos with D2/D3.

   If the overlap isn't handled, the KL results will mostly measure "how many images do these datasets share". The D1-trained network would also have trained on some D2/D3/D5 images, making them look in-distribution for the wrong reason.
5. **Upstream resizing differs by source.** Roboflow exports are often resized, sometimes stretched to a fixed size. Record each dataset's original resolution and aspect ratio, because this is itself a source of shift.
6. **There is no "no damage" class in any of them.**

---

# Task: measure distribution shift between the five datasets with KL divergence

## Goal

Train a ResNet-18 classifier on D1. Use it as a fixed feature extractor: feed it D2, D3, D4 and D5 and collect their embeddings. Then compute the KL divergence between the embedding distributions of each dataset, and plot the embeddings so we can see how alike the datasets are.

The focus is on the datasets. The classifier only needs to be good enough that its features are meaningful.

## Constraints

- **Network.** Use the standard torchvision `resnet18` with normal `torch.nn.Conv2d` layers and ImageNet-pretrained weights. Do not change the ResNet configuration: no width or depth changes, no custom blocks. The only change from stock is replacing `fc` with a linear layer for D1's 6 classes.
- **Embedding.** The 512-d global-average-pool output, i.e. the input to `fc`. Use eval mode and `torch.no_grad()`.
- **Same preprocessing everywhere.** Cropping, resizing and normalization must be identical for all five datasets. Otherwise the KL measures our preprocessing instead of the data.
- **No training on D2–D5.** Never train or fine-tune on them; they are only fed through the network.
- **Plain PyTorch.** Use PyTorch, torchvision, numpy, scikit-learn and matplotlib. Add `imagehash` for duplicate detection and `umap-learn` for one plot. Don't add frameworks beyond these unless there is a clear reason.

## Steps

### 1. Download and convert to a common format

- **Crops.** Turn every dataset into a classification set of crops: one square crop per bounding box, using the box plus a 25% margin on each side.
  - Clamp the crop to the image. Shift it inward rather than padding, so there is no padding border the model could use as a cue.
  - Resize to one fixed size, 224 by default.
  - This matches the helmet, which looks at a patch of surface at a time.
- **Metadata.** Keep a CSV per dataset with: crop path, class, original image filename, box coordinates, original image size, and the source group (see the split rule below).
- **Class names.** Keep each dataset's own class names. Mapping happens only in the analysis.

### 2. Measure overlap between datasets (before any training)

- **Hashes.** Compute a perceptual hash (`imagehash.phash`) of every *original* image, before cropping, in all five datasets.
- **Duplicates.** Count two images as duplicates if the Hamming distance is ≤ 6, or if their filenames match. Save a sheet of matched pairs side by side so the user can check the threshold.
- **Table.** Output a 5×5 table: the share of images in row dataset X that also appear in column dataset Y. Show this to the user before going further; it is a finding in its own right.

### 3. Split D1 without leaks

- **Ignore D1's shipped split.** Re-split D1 70/15/15 (train / val / test), stratified by class.
- **Keep groups together.** All crops from one original image must go to the same side. So must near-duplicate images (same hash cluster) and tiles of one photo (shared filename stem).
- **Report.** Print how many groups and crops land in each split.

### 4. Train the ResNet-18 on D1

- **Setup.** Cross-entropy with class weights for the uneven class counts. Use simple augmentation (flips, small rotations, colour jitter, random resized crop), weight decay, and early stopping on validation macro-F1. Start with AdamW at lr 1e-4.
- **Speed.** D1 gives roughly 30k crops. Report the time per epoch early, and offer to subsample if it is too slow.
- **Test report.** Report on D1-test:
  - accuracy, macro-F1, balanced accuracy
  - per-class recall
  - confusion matrix
  - train vs test macro-F1, to check overfitting

  This shows whether the features can be trusted before the shift analysis.

### 5. Extract embeddings

- **D1.** Extract D1-train and D1-held-out (val + test) separately. Use D1-held-out as D1's sample, because training images were fit by the network and look artificially tight.
- **D2–D5.** Extract all images, with no split needed since they are never trained on.
- **Saving.** Save to `.npz` with: dataset, split, class, source filename, and a flag `in_d1` (true if the original image's hash matches a D1 image, plus whether that match is in D1-train).

### 6. KL divergence

- **Two versions of every result.** Compute every number for **(a)** all images and **(b)** only images with `in_d1 == False`. Version (b) is the honest measure of shift. If a dataset has almost nothing left after dedup, report that instead of a KL value.
- **Equal sample sizes.** Subsample each dataset to the same n with a fixed seed, and repeat over 5 seeds.
- **PCA.** Fit PCA on D1-held-out embeddings only and project everything into it. The default is k = 32; also report k ∈ {8, 16, 32, 64} and the share of D1 variance kept. A 512-d covariance cannot be estimated reliably from a few hundred to a few thousand points, so this step is required.
- **Primary estimator: Gaussian closed form.** Use Ledoit–Wolf shrinkage covariance (`sklearn.covariance.LedoitWolf`), then:
  KL(P‖Q) = ½ [ tr(Σq⁻¹Σp) + (μq−μp)ᵀ Σq⁻¹ (μq−μp) − k + ln(det Σq / det Σp) ]
  Use `slogdet` and `solve`, not explicit inverses.
- **Check estimator: k-NN** (Wang, Kulkarni & Verdú 2009), with k_nn = 5 in the same space:
  KL(P‖Q) ≈ (d/n) Σᵢ ln(νᵢ/ρᵢ) + ln(m/(n−1))
  Here ρᵢ is the distance from point i to its k-th neighbour within P (excluding itself), and νᵢ is the distance to its k-th neighbour in Q. It makes no Gaussian assumption.
- **Both directions.** KL is asymmetric, so report KL(P‖Q), KL(Q‖P) and the symmetric mean. Compute every pair among D1–D5, not only versus D1.
- **Reference values.** Without these the raw numbers cannot be interpreted.
  - **Noise floor:** two random halves of D1-held-out.
  - **In-distribution:** D1-train vs D1-held-out.
  - **Known-shift sanity check:** D1-held-out with synthetic corruptions (Gaussian noise, blur, low contrast, JPEG) at 3 severities. KL must rise with severity; if it doesn't, the setup is wrong.
- **Optional per-class KL.** Compute it for classes shared with D1, using this map:
  - D2/D3: crack→Crack, dent→Dent, corrosion→Corrosion
  - D4: cracking→Crack, surface corrosion→Corrosion, leading edge corrosion→Corrosion, paint peeling→Paint-Peel-Off (all different materials from aircraft)
  - D5: Dent→Dent, Fastener Damage→Missing-Head (loose match), Rupture→unmapped

## Output (in `results/`)

- **`overlap_matrix.csv`**, plus a heatmap of it and the duplicate-pairs sheet.
- **D1 classifier report**, with its training curves.
- **`embeddings.npz`**.
- **`kl_matrix.csv`**: both estimators, the all-images and deduplicated versions, mean and spread over seeds, plus the reference values.
- **Figures (PNG, readable when printed):**
  1. **Scatter in D1's PCA space** (first two components), points coloured by dataset, with a 2-σ ellipse per dataset. This is the main figure. Make one version with all images and one deduplicated; draw duplicates as hollow markers.
  2. **UMAP of the pooled embeddings,** coloured by dataset. The caption must say that distances between clusters are not meaningful.
  3. **Heatmap of pairwise symmetric KL** (deduplicated), annotated, with the noise floor and in-distribution reference in the caption.
  4. **KL vs corruption severity** for the sanity check.
- **`summary.md`**:
  - the dataset sizes
  - the overlap table
  - the D1 model's test results
  - the KL table with reference values
  - the figures
  - a few plain-language sentences on which datasets are closest to D1 and which are far
  - the caveats below

## Code shape

- **Structure.** Use a few small scripts, run in order, each with command-line arguments and a fixed random seed:
  - data preparation
  - overlap check
  - training
  - embedding extraction
  - shift analysis and figures

  Keep shared code (loading, transforms, model builder) in one module.
- **Pipeline check.** Include a tiny synthetic dataset generator and run the whole pipeline on it first, before the real downloads.
- **README.** Write a short README with the run order and commands.

## Caveats to state in the summary

- **Viewpoint.** KL is measured through a network trained on D1, so it measures shift *as this model sees it*.
- **Duplicates.** The deduplicated numbers are the ones to quote. The all-images numbers mainly show how much the datasets share.
- **Gaussian assumption.** Gaussian KL assumes each dataset is one blob, which is why the k-NN estimator is reported too.
- **Relative numbers only.** KL values depend on k and n. Compare them at fixed k and n against the reference values, never as absolute numbers.
- **Hash limits.** Hash matching misses heavily cropped or edited copies, so "deduplicated" means "no near-identical copies found".
- **Licensing.** D1's license is unknown and its sources carry their own licenses. That is fine for this analysis, but worth noting before D1 is used in anything published.

## Working style

- Plan before coding and confirm anything ambiguous, e.g. a dataset version, the hash threshold, or a folder layout that doesn't match expectations.
- Keep the code simple and direct. Every component should be needed for the result.
- Present options with trade-offs rather than silently picking one.
- Report what was and was not actually run.
