"""The SAQR logo for the Model page's title, plus the outlines its draw-in animation reveals it along.
Source: scripts/saqr_logo_source.png, the logo image (with its transparency) from "SAQR - Brand Identity.pdf". There it
faces right; the page shows it facing left, so it is mirrored here.
- web/public/brand/saqr_logo.png: the mirrored logo, cropped with a margin, laid on white and made transparent again
  by "colour to alpha", so its shading lives in the alpha (the page fills that shape with its own ink colour);
- web/src/data/logoMask.json: the logo's traced outlines (the edges of its shapes and holes) and the pen width the page
  draws them with as a mask, before filling it in, so the logo appears line by line like the helmet sketch.

    python scripts/logo_mask.py
"""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

DASH = Path(__file__).resolve().parents[1]
SRC, OUT_PNG, OUT_JSON = DASH / "scripts/saqr_logo_source.png", DASH / "web/public/brand/saqr_logo.png", DASH / "web/src/data/logoMask.json"
PAD = 0.08   # margin around the logo, as a share of its width


def main():
    src = ImageOps.mirror(Image.open(SRC).convert("RGBA"))
    a = np.asarray(src)[..., 3]
    ys, xs = np.nonzero(a > 10)
    p = round((xs.max() - xs.min()) * PAD)
    src = src.crop((max(xs.min() - p, 0), max(ys.min() - p, 0), min(xs.max() + p + 1, src.width), min(ys.max() + p + 1, src.height)))
    white = Image.new("RGBA", src.size, (255, 255, 255, 255))
    white.alpha_composite(src)
    rgb = np.asarray(white.convert("RGB")).astype(float) / 255
    # colour to alpha against white: the least alpha that explains each pixel as a colour over white
    alpha = (1 - rgb).max(-1)
    col = np.where(alpha[..., None] > 0, 1 - (1 - rgb) / np.maximum(alpha[..., None], 1e-6), 1)
    out = np.dstack([np.clip(col, 0, 1), alpha]) * 255
    out[alpha < 0.02] = 0
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out.round().astype(np.uint8), "RGBA").save(OUT_PNG, optimize=True)

    # the outlines: every edge of the logo's shapes (outer edges and holes such as the eye), left to right
    ink = (alpha > 0.35).astype(np.uint8) * 255
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    paths = []
    for c in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
        if cv2.contourArea(c) < 40:
            continue
        pts = cv2.approxPolyDP(c, 1.2, True)[:, 0]
        paths.append("M" + " L".join(f"{x},{y}" for x, y in pts) + " Z")
    h, w = ink.shape
    logo_w = xs.max() - xs.min()
    OUT_JSON.write_text(json.dumps({"w": w, "h": h, "pen": round(logo_w * 0.061, 1), "alphaMax": round(float(alpha.max()), 3),
                                    "paths": paths}, indent=1))
    print(f"{w}x{h}, {len(paths)} outlines, pen {logo_w * 0.061:.1f}, darkest alpha {alpha.max():.3f}")


if __name__ == "__main__":
    main()
