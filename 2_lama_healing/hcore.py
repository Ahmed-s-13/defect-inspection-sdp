"""Shared basics of the healer: paths, defect-box readers, LaMa, and the screw-head transplant."""
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MAIN = REPO / "1_main_model_resnet18"
RAW = MAIN / "data/raw/D1/content/Aircraft_dataset"                  # D1 photos + YOLO labels (D1_raw.zip)
LAMA_WEIGHTS = HERE / "weights/big-lama.pt"                           # lama_weights.zip (team Google Drive)
YOLO_CLASSES = ["crack", "dent", "corrosion", "scratch", "missing-head", "paint-peel-off"]  # D1 data.yaml order
CLASSES = sorted(YOLO_CLASSES)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
GROW = 0.12            # LaMa mask = defect box grown by 12% per side
HEAD_SCALE = 1.05      # a pasted head sits slightly over its countersink rim
MASK_PAD = 1.25        # Poisson-blend mask: head ellipse + a ring of surrounding skin


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------- defect boxes
def read_yolo(label_file, w, h):
    """YOLO label file (class cx cy bw bh, normalised, D1 class order) -> [(cls, x1, y1, x2, y2)] in pixels."""
    out = []
    for line in Path(label_file).read_text().split("\n"):
        p = line.split()
        if len(p) == 5:
            c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
            out.append((YOLO_CLASSES[c], (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def d1_label_file(key):
    """D1 photo key ("train/aircraft_000123.jpg") -> its YOLO label file."""
    split, name = key.split("/")
    return RAW / "labels" / split / (Path(name).stem + ".txt")


def read_boxes(key, w, h):
    """Boxes of a D1 photo by key."""
    f = d1_label_file(key)
    return read_yolo(f, w, h) if f.exists() else []


def find_label_file(img_path, labels_dir=None):
    """YOLO label file for an image: in labels_dir, next to the image, or in a sibling labels/ folder (D1 layout:
    .../images/<split>/x.jpg -> .../labels/<split>/x.txt)."""
    p = Path(img_path)
    cands = [Path(labels_dir) / (p.stem + ".txt")] if labels_dir else []
    cands.append(p.with_suffix(".txt"))
    parts = list(p.parts)
    if "images" in parts:
        i = len(parts) - 1 - parts[::-1].index("images")
        cands.append(Path(*parts[:i], "labels", *parts[i + 1:-1], p.stem + ".txt"))
    return next((c for c in cands if c.exists()), None)


def read_boxes_csv(csv_file):
    """CSV with columns file,cls,x1,y1,x2,y2 (pixels, class names) -> {file name: [(cls, x1, y1, x2, y2)]}."""
    out = {}
    with open(csv_file, newline="", encoding="utf-8-sig") as f:      # utf-8-sig: Excel's byte-order mark
        rows = [{k.strip().lower(): (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f)]
    need = {"file", "cls", "x1", "y1", "x2", "y2"}
    if rows and not need <= set(rows[0]):
        raise SystemExit(f"{csv_file}: needs the columns {sorted(need)}, found {sorted(rows[0])}")
    for r in rows:
        cls = r["cls"]
        if cls not in CLASSES:
            raise SystemExit(f"{csv_file}: unknown class '{cls}' (use one of {CLASSES})")
        out.setdefault(Path(r["file"]).name, []).append((cls, *(float(r[k]) for k in ("x1", "y1", "x2", "y2"))))
    return out


def defect_mask(shape, boxes, grow=GROW):
    """255 inside every box grown by `grow` of its size per side."""
    H, W = shape[:2]
    mask = np.zeros((H, W), np.uint8)
    for _, x1, y1, x2, y2 in boxes:
        gx, gy = grow * (x2 - x1), grow * (y2 - y1)
        mask[int(max(y1 - gy, 0)):int(np.ceil(min(y2 + gy, H))), int(max(x1 - gx, 0)):int(np.ceil(min(x2 + gx, W)))] = 255
    return mask


# ---------------------------------------------------------------- LaMa
def load_lama(dev):
    if not LAMA_WEIGHTS.exists():
        raise SystemExit(f"missing {LAMA_WEIGHTS}: download lama_weights.zip from the team Google Drive and unzip it "
                         "in the repo root (see 2_lama_healing/README.md)")
    return torch.jit.load(str(LAMA_WEIGHTS), map_location=dev).eval()


def lama(model, img, mask, dev):
    """img HxWx3 uint8, mask HxW uint8 (255 = fill) -> HxWx3 uint8. Pads to a multiple of 8 like simple-lama."""
    h, w = mask.shape
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None].float().div(255).to(dev)
    m = torch.from_numpy((mask > 0).astype(np.float32))[None, None].to(dev)
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    x, m = F.pad(x, (0, pw, 0, ph), mode="reflect"), F.pad(m, (0, pw, 0, ph), mode="reflect")
    with torch.inference_mode():
        y = model(x, m)[0, :, :h, :w]
    return (y.permute(1, 2, 0).clamp(0, 1) * 255).round().byte().cpu().numpy()


# ---------------------------------------------------------------- screw-head transplant
def transplant(dst, donor_img, hole, donor):
    """Paste the donor head (donor = (cx, cy, rx, ry) in donor_img) over the hole (cx, cy, rx, ry) in dst: scaled
    separately in x and y (perspective), then Poisson-blended (cv2.seamlessClone NORMAL_CLONE)."""
    hx, hy, hrx, hry = hole
    dx, dy, drx, dry = donor
    fx, fy = HEAD_SCALE * hrx / drx, HEAD_SCALE * hry / dry
    half = int(np.ceil(1.6 * max(drx, dry)))
    pad = np.pad(donor_img, ((half, half), (half, half), (0, 0)), mode="reflect")
    patch = pad[dy:dy + 2 * half, dx:dx + 2 * half]                          # donor centred
    patch = cv2.resize(patch, (max(3, round(2 * half * fx)) | 1, max(3, round(2 * half * fy)) | 1),
                       interpolation=cv2.INTER_CUBIC)
    ph, pw = patch.shape[:2]
    mask = np.zeros((ph, pw), np.uint8)
    cv2.ellipse(mask, (pw // 2, ph // 2), (int(MASK_PAD * HEAD_SCALE * hrx), int(MASK_PAD * HEAD_SCALE * hry)), 0, 0, 360, 255, -1)
    out = cv2.seamlessClone(np.ascontiguousarray(patch[..., ::-1]), np.ascontiguousarray(dst[..., ::-1]), mask,
                            (int(hx), int(hy)), cv2.NORMAL_CLONE)
    return out[..., ::-1]
