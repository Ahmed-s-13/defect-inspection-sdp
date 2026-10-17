# The two existing codebases

## 1. Teacher–student anomaly model (`3_teacher_student_msrd/msrd_anomaly.py`)

It is **MSRD** (Multi-Scale Reverse Distillation, Liu & Chung 2025), built on Reverse Distillation (Deng & Li, CVPR 2022). It is one file, about 630 lines.

| item | what the code does |
|---|---|
| Teacher | Frozen torchvision `wide_resnet50_2`, ImageNet `IMAGENET1K_V1` weights. It outputs layer1/2/3 features at H/4, H/8 and H/16 (256/512/1024 channels). The code can also use `resnet18`, `resnet34` or `resnet50` (`--backbone`). BatchNorm is kept in eval mode. |
| Bottleneck (OCBE) | Multi-scale fusion: f1 and f2 are strided-conv'd down to H/16 and concatenated with f3. Three torchvision `Bottleneck` blocks (layer4-style, stride 2) then compress this to a 2048-ch H/32 embedding. |
| MSRD bridge | The embedding is projected by a 1×1 conv-BN-ReLU per decoder level and resized. It is fed into every decoder stage, not only the first. |
| Student | A 3-stage decoder. Each stage is ConvTranspose ×2, a fuse with the bridge, then 2 residual bottlenecks. It outputs s3, s2, s1, which must match f3, f2, f1. |
| Reconstruction head | Rebuilds the image from s1 (H/4 → H, sigmoid). |
| Loss | L_KD (sum over scales of mean 1 − cosine) + 0.9 · (1 − SSIM). |
| Trainable size | 75.8 M parameters (WRN-50 teacher). |
| Data format | MVTec AD folders: `train/good/*.png` (normal only), `test/<type>/*.png`, `ground_truth/<type>/*_mask.png`. It reads **PNG only**. Images are resized to 256 and given in [0,1]; the model applies ImageNet normalization inside. No augmentation. |
| Output | A per-pixel anomaly map: 1 − cos per scale, FPN-style top-down sum, upsampled, Gaussian σ = 4. The image score is the map's max. Also a reconstructed image. Metrics: image AUROC, pixel AUROC, AUPRO@0.3, FPR at 95%/100% recall, ms/image. |
| Optimizer | Adam, lr 0.005, betas (0.5, 0.999), batch 16, 200 epochs, seed 42. There is no validation or model selection; the last epoch is saved. |
| Checkpoint | `runs/metal_nut/msrd_metal_nut.pt` stores the non-teacher weights only. The teacher is rebuilt from torchvision. |
| Training time | On MVTec metal_nut (220 images, MPS): about 26 s/epoch, 200 epochs ≈ 1.45 h, so about 1.9 s per batch of 16 at 256 px. Test results: image AUROC 1.000, pixel AUROC 0.979, AUPRO 0.953, 3.2 ms/image. |

The existing checkpoint is trained on metal nuts, so it is not usable for aircraft patches. It has to be retrained on our clean patches.

## 2. Distribution-shift code with the D1 ResNet-18 (`KL-Divergence/`)

| item | what the code does |
|---|---|
| Shared module | `klshift/common.py`: `square_crop_box` (box + 25% margin per side, square, shifted inward, never padded), `CROP_SIZE` 224, ImageNet `MEAN/STD`, `build_model` (stock `resnet18`, fc → 6), `embed_batch` (512-d GAP), `CropDataset`, `train_transform`. |
| Crops | `prepare.py`: one 224×224 JPEG q95 crop per YOLO box, resized with BICUBIC. Boxes under 4 px are skipped (1,848 in D1), but those defects are still in the images. |
| D1 data | 14,760 images (mostly 640×640), 28,637 crops. Crop side before resize: median 160 px, IQR 76–298 px. On average, the box covers 31% of the crop area. |
| D1 split | `data/work/d1_split.csv`: 70/15/15, `StratifiedGroupKFold`, groups = image + photo stem + D1 near-duplicates. That is 10,293 / 2,194 / 2,128 images and 20,051 / 4,293 / 4,293 crops. 145 D1 images have no boxes and are not in the split. |
| D1 model | `data/work/model_best.pt` (`state`, `classes` = corrosion, crack, dent, missing-head, paint-peel-off, scratch). D1-test: accuracy 0.880, macro-F1 0.813. Weakest recall: paint-peel-off 0.69, corrosion 0.71, scratch 0.71. Training took about 2 min/epoch on MPS. |
| Embeddings | `results/embeddings.npz`: 54,146 crops × 512 with `dataset, split, cls, image, crop_path, group, in_d1, in_d1_train, rep`. |
| KL code | `shift.py`: `gauss_kl` (Ledoit–Wolf, closed form), `knn_kl` (k = 5), PCA fitted on D1-held-out, scatter with 2-σ ellipses. We will reuse these functions. |
