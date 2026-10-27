"""Turn uploaded bytes into exactly the crop predict.py would make, plus the 8 views the model averages over."""
import base64
import io
import math
from typing import NamedTuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps, UnidentifiedImageError

from server import paths  # noqa: F401
from enhanced_res.model import dihedral_t
from klshift.common import CROP_SIZE, eval_transform

MIN_SIDE = 16
MAX_SIDE = 6000                     # larger photos are downscaled before cropping (a 200 MP phone photo would use ~600 MB)
Image.MAX_IMAGE_PIXELS = 300_000_000  # Pillow refuses (DecompressionBombError) above twice this
_TF = eval_transform()


class ImageError(Exception):
    pass


class Square(NamedTuple):
    left: int
    top: int
    side: int


def open_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        if max(img.size) > MAX_SIDE:
            img.draft("RGB", (MAX_SIDE, MAX_SIDE))  # JPEG: decode at a reduced scale, cheaply
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Image.DecompressionBombError as e:
        raise ImageError("image too large to open safely") from e
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ImageError("not a readable image (JPEG, PNG, WebP, BMP or TIFF)") from e
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.BICUBIC)
    return img


def default_square(w: int, h: int) -> Square:
    """The centre square, exactly as predict.to_crop does without --box."""
    side = min(w, h)
    return Square((w - side) // 2, (h - side) // 2, side)


def clamp(left: float, top: float, side: float, w: int, h: int) -> Square:
    side = int(round(max(MIN_SIDE, min(side, w, h))))
    left = int(round(min(max(left, 0), w - side)))
    top = int(round(min(max(top, 0), h - side)))
    return Square(left, top, side)


def parse_square(text: str, w: int, h: int) -> Square:
    parts = (text or "").split(",")
    if len(parts) != 3:
        raise ValueError("square must be 'left,top,side' in image pixels")
    try:
        vals = [float(p) for p in parts]
    except ValueError as e:
        raise ValueError("square must be three numbers") from e
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("square must be finite numbers")
    return clamp(*vals, w, h)


def crop(img: Image.Image, sq: Square) -> Image.Image:
    return img.crop((sq.left, sq.top, sq.left + sq.side, sq.top + sq.side)).resize((CROP_SIZE, CROP_SIZE), Image.BICUBIC)


def to_batch(crops) -> torch.Tensor:
    return torch.stack([_TF(c) for c in crops])


def jpeg_data_url(img: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def view_thumbs(crop_img: Image.Image, size: int = 112) -> list[str]:
    """The 8 views (4 quarter turns, then the same flipped), made with the model's own dihedral_t."""
    t = TF.to_tensor(crop_img).unsqueeze(0)
    out = []
    for k in range(8):
        v = TF.to_pil_image(dihedral_t(t, k)[0]).resize((size, size), Image.BICUBIC)
        out.append(jpeg_data_url(v, 85))
    return out
