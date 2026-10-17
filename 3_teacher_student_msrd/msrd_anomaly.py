"""
msrd_anomaly.py
===============
MSRD anomaly segmentation on MVTec AD.

Implements:
    C.-L. Liu and C.-C. Chung, "Anomaly detection and segmentation in industrial
    images using multi-scale reverse distillation," Applied Soft Computing,
    vol. 168, 2025, Art. no. 112502. DOI: 10.1016/j.asoc.2024.112502
Built on the Reverse Distillation (RD) baseline:
    H. Deng and X. Li, "Anomaly detection via reverse distillation from
    one-class embedding," CVPR 2022.

What the model does
-------------------
One-class learning: it is trained on DEFECT-FREE images only. At test time it
produces a pixel-wise anomaly map (segmentation) and an image-level score.

    image --> [Teacher: frozen ImageNet WideResNet-50]  --> f1, f2, f3 (3 scales)
                        |
                        v
              [OCBE: multi-scale fusion + one-class bottleneck]  --> compact embedding
                        |
               +--------+-----------------------------+
               |  multi-scale aggregator (Sec. 4.2.1)  |  OCBE output fed to EVERY decoder layer
               v                                      v
              [Student decoder, 3 layers (Sec. 4.2.2)] --> s3, s2, s1 (must match f3, f2, f1)
                        |
                        v
              [Reconstruction module (Sec. 4.2.2)] --> rebuilt image, SSIM loss

    Loss (Eq. 3)       = L_KD + c * L_Recon,  L_Recon = 1 - SSIM(I, I_hat)  (paper: c = 0.9)
    Anomaly map (4.3.1)= FPN-style top-down aggregation of the 3 per-scale maps
                         + Gaussian smoothing; image score = max of the map (4.3.2)

On normal surfaces the student reproduces the teacher's features; on a defect it
cannot (it never saw one), so teacher/student disagreement lights up the defect.

Paper settings used as defaults: 256x256 input, ImageNet normalisation, batch 16,
Adam, 200 epochs, c = 0.9, WideResNet-50 teacher. The paper does not give the
learning rate, so we use the RD baseline's (0.005, betas 0.5/0.999). Exact layer
widths inside the aggregator are not specified; ours are noted in the code.

Dataset: MVTec AD (used in the paper)
------------------------------------
1. Download from https://www.mvtec.com/company/research/datasets/mvtec-ad
   (fill in the short form; you can download one category, e.g. metal_nut, or all 15).
2. Extract so the folder looks like:
       mvtec_anomaly_detection/
           metal_nut/
               train/good/*.png
               test/good/*.png, test/<defect_type>/*.png
               ground_truth/<defect_type>/*_mask.png
3. Licence: CC BY-NC-SA 4.0 (fine for academic, non-commercial work).

Install
-------
    pip install torch torchvision numpy pillow scipy scikit-learn matplotlib tqdm
    (With an NVIDIA GPU, install torch from https://pytorch.org first so it uses CUDA.)

Run (from the VS Code terminal, in the folder containing this file)
------------------------------------------------------------------
    python msrd_anomaly.py --data_root ./mvtec_anomaly_detection --category metal_nut
    python msrd_anomaly.py --category tile --epochs 50                  # quicker first try
    python msrd_anomaly.py --category tile --backbone resnet18          # lighter teacher (paper Sec. 6.2)
    python msrd_anomaly.py --category metal_nut --eval_only runs/metal_nut/msrd_metal_nut.pt

Outputs go to runs/<category>/: checkpoint, metrics.json, score histogram,
and per-image visualisations (image | ground truth | anomaly map | overlay | reconstruction).
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label as connected_components
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import models
from torchvision.models.resnet import Bottleneck
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")  # write figures to disk, no window needed
import matplotlib.pyplot as plt  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]


# =============================================================================
# Dataset
# =============================================================================
class MVTecDataset(Dataset):
    """Returns (image in [0,1], binary mask, image label 0/1, defect type)."""

    def __init__(self, root, category, split, img_size):
        self.cat_dir = Path(root) / category
        self.img_size = img_size
        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        self.samples = []
        split_dir = self.cat_dir / split
        for defect_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for img_path in sorted(defect_dir.glob("*.png")):
                if defect_dir.name == "good":
                    self.samples.append((img_path, None, 0, "good"))
                else:
                    mask_path = self.cat_dir / "ground_truth" / defect_dir.name / f"{img_path.stem}_mask.png"
                    self.samples.append((img_path, mask_path, 1, defect_dir.name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, label, defect_type = self.samples[idx]
        img = self.img_tf(Image.open(img_path).convert("RGB"))
        if mask_path is None or not mask_path.exists():
            mask = torch.zeros(1, self.img_size, self.img_size)
        else:
            mask = (self.mask_tf(Image.open(mask_path).convert("L")) > 0.5).float()
        return img, mask, label, defect_type


def check_dataset(root, category):
    cat_dir = Path(root) / category
    needed = [cat_dir / "train" / "good", cat_dir / "test"]
    if all(p.is_dir() for p in needed):
        return
    print("\n[!] MVTec AD category not found at:", cat_dir.resolve())
    print("    1. Download from https://www.mvtec.com/company/research/datasets/mvtec-ad")
    print(f"    2. Extract so that {cat_dir}/train/good/ and {cat_dir}/test/ exist")
    print("    3. Re-run with --data_root pointing at the extracted folder")
    print("    Available categories:", ", ".join(MVTEC_CATEGORIES))
    sys.exit(1)


# =============================================================================
# Model
# =============================================================================
def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)


def conv_bn_relu(in_ch, out_ch, k=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, stride, k // 2, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# name -> (torchvision constructor, weights, channels of layer1/2/3)
BACKBONES = {
    "wide_resnet50_2": (models.wide_resnet50_2, "Wide_ResNet50_2_Weights", (256, 512, 1024)),
    "resnet50": (models.resnet50, "ResNet50_Weights", (256, 512, 1024)),
    "resnet34": (models.resnet34, "ResNet34_Weights", (64, 128, 256)),
    "resnet18": (models.resnet18, "ResNet18_Weights", (64, 128, 256)),
}


class Teacher(nn.Module):
    """Frozen ImageNet backbone (paper default WideResNet-50, Sec. 6.2 compares ResNet-18/34/50).
    Returns features at 1/4, 1/8, 1/16 resolution."""

    def __init__(self, backbone="wide_resnet50_2", pretrained=True):
        super().__init__()
        ctor, weights_name, self.channels = BACKBONES[backbone]
        weights = getattr(models, weights_name).IMAGENET1K_V1 if pretrained else None
        net = ctor(weights=weights)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)   # c1 ch, H/4
        f2 = self.layer2(f1)  # c2 ch, H/8
        f3 = self.layer3(f2)  # c3 ch, H/16
        return [f1, f2, f3]


class OCBE(nn.Module):
    """One-Class Bottleneck Embedding (from RD, Sec. 3 of the paper).
    Multi-scale feature fusion (MFF): bring f1, f2 down to f3's resolution and concatenate.
    One-class embedding (OCE): ResNet 4th-stage bottleneck blocks compress to a low-res embedding.
    The compression is what stops the student simply copying anomalous features."""

    def __init__(self, channels, wide=True):
        super().__init__()
        c1, c2, c3 = channels
        emb = 2 * c3
        self.down1 = nn.Sequential(conv_bn_relu(c1, c2, 3, 2), conv_bn_relu(c2, c3, 3, 2))
        self.down2 = conv_bn_relu(c2, c3, 3, 2)
        downsample = nn.Sequential(nn.Conv2d(3 * c3, emb, 1, 2, bias=False), nn.BatchNorm2d(emb))
        bw = 128 if wide else 64
        self.oce = nn.Sequential(
            Bottleneck(3 * c3, emb // 4, stride=2, downsample=downsample, base_width=bw),
            Bottleneck(emb, emb // 4, base_width=bw),
            Bottleneck(emb, emb // 4, base_width=bw),
        )
        self.out_channels = emb

    def forward(self, feats):
        f1, f2, f3 = feats
        fused = torch.cat([self.down1(f1), self.down2(f2), f3], dim=1)  # 3*c3 ch, H/16
        return self.oce(fused)  # 2*c3 ch, H/32


class MultiLevelBridge(nn.Module):
    """MSRD multi-scale aggregator (Sec. 4.2.1, Fig. 4): the OCBE output is fed
    directly into every decoder layer, not only the first one.
    The paper does not give the layer details, so here the embedding is projected
    by a 1x1 conv for each decoder level and resized to that level's resolution.
    It carries the compressed OCBE embedding only (never raw teacher features),
    otherwise the student could copy a defect instead of failing to rebuild it."""

    def __init__(self, emb_ch, bridge_chs):
        super().__init__()
        self.proj = nn.ModuleList([conv_bn_relu(emb_ch, c, k=1) for c in bridge_chs])

    def forward(self, emb, sizes):
        return [F.interpolate(p(emb), size=s, mode="bilinear", align_corners=False)
                for p, s in zip(self.proj, sizes)]


class ResBottleneck(nn.Module):
    """Light residual block used to refine decoder features."""

    def __init__(self, ch):
        super().__init__()
        mid = ch // 4
        self.body = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            conv3x3(mid, mid), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=False), nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return F.relu(x + self.body(x))


class DecoderStage(nn.Module):
    """One student decoder layer (Sec. 4.2.2): a 2D transposed convolution doubles
    the spatial size and halves the channels, then the aggregator features are fused
    in and residual blocks refine the result."""

    def __init__(self, in_ch, out_ch, bridge_ch, n_blocks=2):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, 2, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.fuse = conv_bn_relu(out_ch + bridge_ch, out_ch, k=1)
        self.refine = nn.Sequential(*[ResBottleneck(out_ch) for _ in range(n_blocks)])

    def forward(self, x, bridge):
        x = self.up(x)
        x = self.fuse(torch.cat([x, bridge], dim=1))
        return self.refine(x)


class ReconstructionHead(nn.Module):
    """Reconstruction module at the tail of the decoder (Sec. 4.2.2): rebuilds the
    input image from the finest student features (H/4 -> H), trained with SSIM loss."""

    def __init__(self, in_ch=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 128, 2, 2, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            conv_bn_relu(128, 128),
            nn.ConvTranspose2d(128, 64, 2, 2, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            conv_bn_relu(64, 64),
            nn.Conv2d(64, 3, 1),
            nn.Sigmoid(),  # image in [0, 1]
        )

    def forward(self, x):
        return self.net(x)


class MSRD(nn.Module):
    def __init__(self, backbone="wide_resnet50_2", pretrained=True):
        super().__init__()
        self.teacher = Teacher(backbone, pretrained)
        c1, c2, c3 = self.teacher.channels
        self.ocbe = OCBE(self.teacher.channels, wide=backbone.startswith("wide"))
        emb = self.ocbe.out_channels
        self.bridge = MultiLevelBridge(emb, (c3 // 4, c2 // 4, c1 // 4))
        self.dec3 = DecoderStage(emb, c3, c3 // 4)  # -> matches f3
        self.dec2 = DecoderStage(c3, c2, c2 // 4)   # -> matches f2
        self.dec1 = DecoderStage(c2, c1, c1 // 4)   # -> matches f1
        self.recon_head = ReconstructionHead(c1)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()  # teacher BatchNorm statistics must never change
        return self

    def trainable_modules(self):
        return [self.ocbe, self.bridge, self.dec3, self.dec2, self.dec1, self.recon_head]

    def forward(self, img):
        x = (img - self.mean) / self.std
        with torch.no_grad():
            t_feats = self.teacher(x)
        emb = self.ocbe(t_feats)
        sizes = [f.shape[-2:] for f in reversed(t_feats)]  # f3, f2, f1 sizes
        b3, b2, b1 = self.bridge(emb, sizes)
        s3 = self.dec3(emb, b3)
        s2 = self.dec2(s3, b2)
        s1 = self.dec1(s2, b1)
        recon = self.recon_head(s1)
        return t_feats, [s1, s2, s3], recon


# =============================================================================
# Losses and anomaly maps
# =============================================================================
def kd_loss(t_feats, s_feats):
    """L_KD (Eq. 2): sum over scales of the mean of M^k = 1 - cosine similarity (Eq. 1)."""
    return sum((1 - F.cosine_similarity(t, s, dim=1)).mean() for t, s in zip(t_feats, s_feats))


def _gaussian_window(size, sigma, channels, device):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] @ g[None, :]).expand(channels, 1, size, size).contiguous()
    return w


def ssim(x, y, window_size=11, sigma=1.5, return_map=False):
    """Structural similarity (Eq. 6) for images in [0, 1], Gaussian sliding window."""
    c = x.shape[1]
    w = _gaussian_window(window_size, sigma, c, x.device)
    pad = window_size // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=c)
    mu_y = F.conv2d(y, w, padding=pad, groups=c)
    sxx = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x ** 2
    syy = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y ** 2
    sxy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return ssim_map if return_map else ssim_map.mean()


def pyramid_anomaly_map(t_feats, s_feats, out_size, scale_weights=(1.0, 1.0, 1.0)):
    """FPN-style aggregation (Sec. 4.3.1, Fig. 6): start at the coarsest map,
    upsample it to the next scale and add, repeat down to the finest, then
    upsample to image size. (Original RD instead upsamples every map straight to
    image size and sums them.) No learnable weights, exactly as in Fig. 6."""
    maps = [(1 - F.cosine_similarity(t, s, dim=1)).unsqueeze(1) * w
            for t, s, w in zip(t_feats, s_feats, scale_weights)]  # finest -> coarsest
    fused = maps[-1]
    for m in reversed(maps[:-1]):
        fused = m + F.interpolate(fused, size=m.shape[-2:], mode="bilinear", align_corners=False)
    return F.interpolate(fused, size=out_size, mode="bilinear", align_corners=False)  # B,1,H,W


# =============================================================================
# Metrics
# =============================================================================
def compute_aupro(masks, amaps, max_fpr=0.3, n_thresholds=200):
    """Area under the per-region-overlap curve up to max_fpr, normalised to [0, 1]."""
    masks = masks.astype(bool)
    region_ids = np.zeros(masks.shape, dtype=np.int64)
    next_id = 1
    for i in range(masks.shape[0]):
        lab, n = connected_components(masks[i])
        if n:
            region_ids[i][lab > 0] = lab[lab > 0] + next_id - 1
            next_id += n
    n_regions = next_id - 1
    if n_regions == 0:
        return float("nan")
    region_sizes = np.bincount(region_ids.ravel(), minlength=n_regions + 1)[1:]
    normal_px = ~masks
    n_normal = normal_px.sum()

    thresholds = np.linspace(amaps.max(), amaps.min(), n_thresholds)
    fprs, pros = [], []
    for th in thresholds:
        pred = amaps >= th
        overlap = np.bincount(region_ids[pred], minlength=n_regions + 1)[1:]
        pros.append(np.mean(overlap / region_sizes))
        fprs.append((pred & normal_px).sum() / n_normal)
    fprs, pros = np.array(fprs), np.array(pros)
    keep = fprs <= max_fpr
    f, p = fprs[keep], pros[keep]
    if len(f) < len(fprs):  # interpolate the point exactly at max_fpr
        j = np.argmax(~keep)
        p_end = np.interp(max_fpr, [fprs[j - 1], fprs[j]], [pros[j - 1], pros[j]]) if j > 0 else pros[j]
        f, p = np.append(f, max_fpr), np.append(p, p_end)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2 renamed trapz
    return float(trapezoid(p, f) / max_fpr)


def fpr_at_recall(labels, scores, recall):
    """False-alarm rate on normal images when the threshold catches `recall` of defective images."""
    anomalous = np.sort(scores[labels == 1])
    k = int(np.floor((1 - recall) * len(anomalous)))
    thr = anomalous[min(k, len(anomalous) - 1)]
    return float(np.mean(scores[labels == 0] >= thr)), float(thr)


# =============================================================================
# Evaluation and visualisation
# =============================================================================
@torch.no_grad()
def evaluate(model, loader, device, args, vis_dir=None):
    model.eval()
    all_maps, all_masks, labels, types = [], [], [], []
    vis_store = []  # (img, mask, recon, label, type) for a subset
    n_normal_vis = n_anom_vis = 0
    half = max(args.vis_count // 2, 1)

    infer_time, n_imgs = 0.0, 0
    for img, mask, label, dtype in tqdm(loader, desc="Evaluating", leave=False):
        img = img.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        t_feats, s_feats, recon = model(img)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_time += time.perf_counter() - t0
        n_imgs += img.shape[0]
        amap = pyramid_anomaly_map(t_feats, s_feats, img.shape[-2:])
        if args.recon_map_weight > 0:
            amap = amap + args.recon_map_weight * (1 - ssim(recon, img, return_map=True)).mean(1, keepdim=True)
        amap = amap.squeeze(1).cpu().numpy()
        for i in range(amap.shape[0]):
            all_maps.append(gaussian_filter(amap[i], sigma=args.sigma))
            all_masks.append(mask[i, 0].numpy())
            labels.append(int(label[i]))
            types.append(dtype[i])
            if vis_dir is not None:
                is_anom = int(label[i]) == 1
                if (is_anom and n_anom_vis < half) or (not is_anom and n_normal_vis < half):
                    vis_store.append((img[i].cpu(), mask[i, 0], recon[i].cpu(), int(label[i]), dtype[i], len(all_maps) - 1))
                    n_anom_vis += is_anom
                    n_normal_vis += not is_anom

    amaps = np.stack(all_maps)
    masks = np.stack(all_masks)
    labels = np.array(labels)
    image_scores = amaps.reshape(len(amaps), -1).max(axis=1)

    metrics = {
        "image_AUROC": float(roc_auc_score(labels, image_scores)),
        "pixel_AUROC": float(roc_auc_score(masks.ravel().astype(np.uint8), amaps.ravel())),
        "pixel_AUPRO@0.3": compute_aupro(masks, amaps),
    }
    for r in (0.95, 1.0):
        fpr, thr = fpr_at_recall(labels, image_scores, r)
        metrics[f"image_FPR_at_{int(r * 100)}pct_recall"] = fpr
        metrics[f"image_threshold_at_{int(r * 100)}pct_recall"] = thr
    metrics["inference_ms_per_image"] = 1000 * infer_time / max(n_imgs, 1)

    if vis_dir is not None:
        save_visuals(vis_dir, vis_store, amaps, labels, image_scores, types)
    return metrics


def save_visuals(vis_dir, vis_store, amaps, labels, image_scores, types):
    vis_dir.mkdir(parents=True, exist_ok=True)
    vmin, vmax = np.percentile(amaps, 1), np.percentile(amaps, 99.9)

    for n, (img, mask, recon, label, dtype, idx) in enumerate(vis_store):
        img_np = img.permute(1, 2, 0).numpy()
        rec_np = recon.permute(1, 2, 0).numpy()
        fig, ax = plt.subplots(1, 5, figsize=(16, 3.6))
        ax[0].imshow(img_np); ax[0].set_title(f"Input ({dtype})")
        ax[1].imshow(mask.numpy(), cmap="gray"); ax[1].set_title("Ground truth")
        ax[2].imshow(amaps[idx], cmap="jet", vmin=vmin, vmax=vmax); ax[2].set_title(f"Anomaly map (score {image_scores[idx]:.3f})")
        ax[3].imshow(img_np); ax[3].imshow(amaps[idx], cmap="jet", alpha=0.45, vmin=vmin, vmax=vmax); ax[3].set_title("Overlay")
        ax[4].imshow(np.clip(rec_np, 0, 1)); ax[4].set_title("Reconstruction")
        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.savefig(vis_dir / f"{n:02d}_{'anom' if label else 'good'}_{dtype}.png", dpi=110)
        plt.close(fig)

    # Image-score distribution: how well normal and defective images separate.
    fig, a = plt.subplots(figsize=(7, 4))
    bins = np.linspace(image_scores.min(), image_scores.max(), 40)
    a.hist(image_scores[labels == 0], bins=bins, alpha=0.6, label="good")
    a.hist(image_scores[labels == 1], bins=bins, alpha=0.6, label="defective")
    a.set_xlabel("image anomaly score (max of map)"); a.set_ylabel("count"); a.legend()
    a.set_title("Image-level score distribution")
    plt.tight_layout()
    plt.savefig(vis_dir.parent / "score_histogram.png", dpi=110)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="MSRD-style anomaly segmentation on MVTec AD")
    p.add_argument("--data_root", default="./mvtec_anomaly_detection")
    p.add_argument("--category", default="metal_nut", choices=MVTEC_CATEGORIES)
    p.add_argument("--backbone", default="wide_resnet50_2", choices=list(BACKBONES),
                   help="teacher network (paper default wide_resnet50_2)")
    p.add_argument("--img_size", type=int, default=256, help="paper uses 256; must be divisible by 32")
    p.add_argument("--epochs", type=int, default=200, help="paper uses 200")
    p.add_argument("--batch_size", type=int, default=16, help="paper uses 16; lower to 8 if GPU memory runs out")
    p.add_argument("--lr", type=float, default=0.005, help="not stated in the paper; RD baseline value")
    p.add_argument("--lambda_recon", type=float, default=0.9,
                   help="c in Eq. 3, weight of the SSIM reconstruction loss (paper: 0.9)")
    p.add_argument("--recon_map_weight", type=float, default=0.0,
                   help="optionally add (1-SSIM) reconstruction error to the anomaly map at test time")
    p.add_argument("--sigma", type=float, default=4.0, help="Gaussian smoothing of the anomaly map (RD value)")
    p.add_argument("--eval_every", type=int, default=0, help="evaluate every N epochs (0 = only at the end)")
    p.add_argument("--vis_count", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--eval_only", default=None, help="path to a saved checkpoint to evaluate")
    p.add_argument("--no_pretrained", action="store_true", help="debug only: random teacher weights")
    return p.parse_args()


def main():
    args = parse_args()
    assert args.img_size % 32 == 0, "--img_size must be divisible by 32"
    set_seed(args.seed)
    check_dataset(args.data_root, args.category)
    device = get_device()
    out_dir = Path(args.out_dir) / args.category
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device} | category: {args.category} | output: {out_dir.resolve()}")

    train_set = MVTecDataset(args.data_root, args.category, "train", args.img_size)
    test_set = MVTecDataset(args.data_root, args.category, "test", args.img_size)
    n_anom = sum(s[2] for s in test_set.samples)
    print(f"Train (good only): {len(train_set)} | Test: {len(test_set)} ({n_anom} defective)")
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)

    if args.eval_only:  # rebuild with the backbone the checkpoint was trained with
        state = torch.load(args.eval_only, map_location="cpu")
        args.backbone = state.get("args", {}).get("backbone", args.backbone)
    model = MSRD(args.backbone, pretrained=not args.no_pretrained).to(device)
    n_train = sum(p.numel() for m in model.trainable_modules() for p in m.parameters())
    print(f"Teacher: {args.backbone} | trainable parameters: {n_train / 1e6:.1f} M (teacher frozen)")
    ckpt_path = out_dir / f"msrd_{args.category}.pt"

    if args.eval_only:
        model.load_state_dict(state["model"], strict=False)
    else:
        params = [p for m in model.trainable_modules() for p in m.parameters()]
        optimizer = torch.optim.Adam(params, lr=args.lr, betas=(0.5, 0.999))
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0, sum_kd, sum_rec, n = time.time(), 0.0, 0.0, 0
            for img, _, _, _ in train_loader:
                img = img.to(device, non_blocking=True)
                t_feats, s_feats, recon = model(img)
                l_kd = kd_loss(t_feats, s_feats)
                l_rec = 1 - ssim(recon, img)
                loss = l_kd + args.lambda_recon * l_rec
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                sum_kd += l_kd.item(); sum_rec += l_rec.item(); n += 1
            print(f"Epoch {epoch:3d}/{args.epochs} | L_KD {sum_kd / n:.4f} | L_Recon {sum_rec / n:.4f} "
                  f"| {time.time() - t0:.1f}s")
            if args.eval_every and epoch % args.eval_every == 0 and epoch != args.epochs:
                m = evaluate(model, test_loader, device, args)
                print("   ", " | ".join(f"{k} {v:.4f}" for k, v in m.items() if "threshold" not in k))

        trainable_state = {k: v for k, v in model.state_dict().items() if not k.startswith("teacher.")}
        torch.save({"model": trainable_state, "args": vars(args)}, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    metrics = evaluate(model, test_loader, device, args, vis_dir=out_dir / "visuals")
    print("\n===== Test results:", args.category, "=====")
    for k, v in metrics.items():
        print(f"  {k:32s} {v:.4f}")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"category": args.category, **metrics, "args": vars(args)}, f, indent=2)
    print(f"\nVisuals and metrics saved in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
