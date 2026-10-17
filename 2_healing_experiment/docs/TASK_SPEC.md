# Task: "heal" defective images, then build a two-stage classifier

Read `../1_main_model_resnet18/docs/TASK_SPEC.md` first for project context. This task builds on the distribution-shift work already in this folder.

## The idea in one paragraph

All our datasets contain only defective images; there is no "no damage" class. Instead of merging many datasets to find clean images, we generate them:

1. Take a patch we know is damaged (we have the bounding box).
2. Remove the defect.
3. Output a realistic non-damaged version of the same patch.

The healed patches become the "non-defective" class for a **two-stage classifier**:

- **Stage 1:** defective vs non-defective.
- **Stage 2:** if the patch is defective, which defect type is it.

Healing is a **training-data generator that runs offline**. The helmet never heals images at inference; it only runs stages 1 and 2.

Why paired healing is worth trying: each healed patch has the same surface, lighting, camera and context as its defective original. The only difference between the classes is the defect itself. That directly targets the shortcut-learning risk (the model learning "wing = corrosion" instead of the defect).

## What you will be given

The user will point you to two pieces of existing code.

1. **The ResNet-18 trained on D1** in this folder, from the distribution-shift task.
   - It is a stock torchvision `resnet18` (6 defect classes), not a Wide ResNet, even if the user calls it "wideresnet18".
   - Do not retrain or modify it.
   - Its uses here are:
     - (a) feature extractor for evaluation
     - (b) the Stage 2 defect classifier
2. **An unsupervised teacher–student anomaly detection model** from earlier work. It is probably reverse distillation, e.g. MSRD (Multi-Scale Reverse Distillation, Liu & Chung 2025), but **read the code before assuming anything**. Report back:
   - the teacher backbone and its pretrained weights
   - what the student and the bottleneck are
   - what training data format it expects (usually normal-only images, MVTec-style folders)
   - what it outputs (image-level score, pixel-level anomaly map)
   - how long training takes

   Keep its architecture as it is. Only add the data loading and wrappers needed for our patches.

Before writing new code, give the user a short summary of both codebases and a plan.

## Constraints

- **Plain PyTorch.** Use torchvision, numpy, scikit-learn and matplotlib, like the rest of the project. Ask before adding a new framework.
- **Stage 1 network.** The Stage 1 classifier is a stock torchvision `resnet18` (ImageNet-pretrained, only `fc` replaced, normal `nn.Conv2d`). This is the same rule as before: don't change the ResNet configuration.
- **Reuse the D1 split.** Use the D1 train/val/test split from the shift task (grouped by original image and hash cluster). Every model here (teacher–student, healer, Stage 1) trains on **D1-train only**. Pick on D1-val. Report on D1-test and on D2–D5. Use the shift task's `in_d1` dedup flags when reporting on D2–D5.
- **Same preprocessing.** Crop size, resizing and normalization must match the shift task.
- **Fixed seeds.** Use them everywhere. Report what was and wasn't actually run.

## Steps

### 1. Build a real clean-patch set from D1

- **Source.** Sample patches from regions of D1 images that don't touch any annotated box (keep a margin around boxes).
- **Size and preprocessing.** Match the size distribution of the defect crops, so patch scale is not a class cue. Process them the same way as the defect crops.
- **Label noise.** Background regions can contain unannotated defects. After step 2, score every clean patch with the teacher–student model. Drop the most anomalous few percent and save a grid of the dropped ones for the user to check.
- **Why this set matters.** It is used to train the teacher–student model and the healer. It is also the **real** non-defective test set. Stage 1 is only trustworthy if it works on real clean patches, not just healed ones.

### 2. Train the teacher–student model on clean patches

- **Training.** Train on D1-train clean patches with the existing code.
- **Evaluation on D1-val / D1-test:**
  - image-level AUROC (clean patches vs defect crops)
  - pixel-level overlap of the anomaly map with the defect box (rough, since boxes are not masks)
- **Baseline status.** This model is already a Stage 1 candidate that never sees a defect. Keep its numbers as the **baseline** that the healing approach must beat.

### 3. Build the healer

**Main approach: box-guided inpainting, using the teacher–student idea in the loss.**

- **Network.** A small U-Net in plain PyTorch.
  - Input: the patch with a masked region, plus the mask.
  - Output: the filled-in patch.
- **Training data.** Train on D1-train **clean** patches with random masks: random boxes whose sizes and aspect ratios follow the defect-box distribution, plus some irregular masks so rectangles are not the only shape.
- **Loss, all on the masked region and its border:**
  - L1
  - a feature loss computed with the **frozen teacher** network from the teacher–student model: healed patches should produce the same teacher features as real clean surface (this is the "something similar to teacher–student" part)
  - a small total-variation term

  Start without an adversarial loss. Add one only if results are blurry, and say so.
- **Healing at inference (offline):**
  1. Mask the annotated defect box, enlarged by a small margin.
  2. Inpaint it.
  3. Paste back with a feathered blend, so pixels outside the mask are unchanged.

**Optional second approach: whole-image teacher–student healing.** This is closer to how reverse distillation works and the user may want to see it.

- **How it works.** Train a decoder on clean patches that maps the student's reconstructed features back to pixels. At inference, a defective patch goes through teacher → bottleneck → student → decoder. Because the student was only trained on normal data, it tends to reconstruct normal features, so the decoded image should look defect-free.
- **Trade-offs versus the main approach.**
  - It needs no box.
  - It changes the whole patch, not just the defect, and it is usually blurrier.
  - Nothing guarantees the defect is gone.
- **When to build it.** Only build it after the main approach works, and compare both with the same metrics.

### 4. Check the healing actually worked

Healed images are plausible fakes, not ground truth, so check them several ways.

- **Visual grid.** Show columns for: original, mask, healed, anomaly map before, anomaly map after. Include ~10 random examples per class, plus the 10 worst by anomaly score.
- **Teacher–student score.** The anomaly score should drop from defect level to clean level. Report:
  - AUROC original vs healed (high is good)
  - AUROC healed vs real clean (≈ 0.5 is good; it means indistinguishable)
- **Embeddings from the D1 ResNet-18.** Reuse the shift-analysis code.
  - KL(healed ‖ real clean) should be much smaller than KL(defective ‖ real clean).
  - Make the same PCA scatter with three groups: defective, healed, real clean.
  - Also report how much the D1 classifier's confidence in the original defect class drops on healed patches.
- **Artifact probe. This check is critical.**
  - Train a small classifier to tell healed patches from real clean patches.
  - Do the same for "clean patches with a random region inpainted" vs real clean.
  - If it succeeds easily (AUROC near 1), the inpainting leaves a detectable fingerprint. Stage 1 would then learn "inpainting artifact = non-defective" and call every real clean patch defective at deployment.
  - Report this number prominently.

### 5. Train Stage 1 (defective vs non-defective)

**Configurations.** Train the stock ResNet-18 binary classifier under these settings and compare them:

| config | non-defective class | defective class |
|---|---|---|
| A | healed patches only | original defect crops |
| B | real clean patches only | original defect crops |
| C | healed + real clean | original defect crops |
| C-artifact-matched | healed + real clean | original defect crops, plus defect crops with a random *non-defect* region inpainted |

In C-artifact-matched, inpainting artifacts appear in both classes, so they stop being a cue. Healed pairs must stay on the same side of the split as their originals.

**Baseline.** Compare against the teacher–student anomaly score from step 2.

**Evaluate only on real images, never on healed ones:**

- D1-test real clean patches vs D1-test defect crops, plus defect crops from D2–D5. Real clean patches from D2–D5 are optional, if built the same way.
- **Metrics:** AUROC, and specificity at 99% and 95% defect recall. A missed defect is worse than a false alarm, so recall is fixed high.
- **Paired flip test:** for each test defect, the classifier should say "defective" on the original and "non-defective" on its healed version. Report the share that flip correctly.

### 6. Two-stage pipeline end to end

- **Pipeline.** Stage 1 (the best config from step 5) feeds Stage 2 (the existing D1 ResNet-18).
- **Report on D1-test and D2–D5:**
  - a confusion matrix with an extra "non-defective" row and column
  - how many defects Stage 1 missed, per defect class
  - Stage 2 accuracy on what Stage 1 passed through

## Output (in `results/healing/`)

- **Write-ups:**
  - summary of the two existing codebases
  - clean-patch set statistics, plus the grid of dropped patches
  - teacher–student baseline results
- **Healer:**
  - checkpoint, training curves, visual grids
- **Healing checks:**
  - teacher–student AUROCs, KL values, PCA scatter, artifact-probe AUROCs
- **Stage 1:**
  - a comparison table across configs A, B, C, C-artifact-matched and the teacher–student baseline
- **Two-stage report.**
- **`summary.md`:**
  - what worked and what didn't, in plain language
  - whether healing beat the teacher–student baseline and real clean patches on real test data
  - the caveats below

## Caveats to state in the summary

- **What healed images are.** Healed images are the healer's guess at what is under the defect. They are not real clean surfaces, and the inpainted region can only be as good as the clean training patches.
- **What counts as a result.** A Stage 1 that scores well on healed images but poorly on real clean patches has learned the artifact, not the defect. Only results on real images count.
- **Box precision.** Boxes are loose and the mask is rectangular, so some defect pixels can survive at the edges, and the mask shape itself can be a cue. That is why irregular masks and the artifact probe are included.
- **Label noise in the clean set.** The clean set comes from the same photos as the defects. It can contain unannotated damage, which the teacher–student filter reduces but does not remove.
- **Licensing.** D1's license is unknown (see `../1_main_model_resnet18/docs/TASK_SPEC.md`). Generated images inherit that.

## Working style

- Plan first; confirm anything unclear about the teacher–student code before building on it.
- Build in the order above and show the user results after steps 2, 4 and 5 before moving on.
- Keep the code simple. Every component should be needed for the result.
- Present options with trade-offs rather than silently choosing.
