"""Screw-head logic for the healing framework: measure a missing-head hole, find intact donor heads, check the result.

All geometry is in photo pixels. An ellipse is (cx, cy, rx, ry): centre and axis-aligned half-widths (perspective
squashes round heads into ellipses; mh_transplant.transplant scales x and y separately to match).

  measure_hole(img, box)                  -> Hole: core (the dark hole) + ring (countersink = size the new head covers)
  find_donors(img, boxes, hole, k, ...)   -> best intact heads in the same photo, excluding every labelled box
  is_hole_like(gray, ellipse)             -> True if the centre is near-black next to the skin (a hole, not a head)
  quality_check(after, clean, hole, ...)  -> (ok, reasons incl. "metrics") for a pasted head
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

N_RAYS = 48


@dataclass
class Hole:
    ring: tuple            # (cx, cy, rx, ry): area the new head must cover
    core: tuple            # (cx, cy, rx, ry) of the dark hole, or None
    dark_thr: float        # grey level below which a pixel counts as "hole dark" in this photo
    conf: float            # 0..1, how sure the measurement is
    method: str            # "edge fit", "core x1.8", "box"
    info: dict = field(default_factory=dict)


def gray_of(img):
    return cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.float32)


def grad_of(gray):
    return cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))


def axis_aligned(e):
    """cv2.fitEllipse result ((cx, cy), (w, h), angle) -> (cx, cy, rx, ry) of its bounding box."""
    (cx, cy), (w, h), ang = e
    a, b, t = w / 2, h / 2, np.deg2rad(ang)
    return (float(cx), float(cy), float(np.hypot(a * np.cos(t), b * np.sin(t))), float(np.hypot(a * np.sin(t), b * np.cos(t))))


def ellipse_mask(shape, e, scale=1.0):
    m = np.zeros(shape[:2], np.uint8)
    cx, cy, rx, ry = e
    cv2.ellipse(m, (int(round(cx)), int(round(cy))), (max(1, int(rx * scale)), max(1, int(ry * scale))), 0, 0, 360, 255, -1)
    return m > 0


def radial_fit(grad, cx, cy, r_lo, r_hi):
    """Strongest edge along N_RAYS rays from (cx, cy) between r_lo(theta) and r_hi(theta); ellipse through them.
    r_lo / r_hi are callables of the angle. Returns ((cx, cy, rx, ry), conf) or (None, 0)."""
    h, w = grad.shape
    pts, peaks = [], []
    for t in np.linspace(0, 2 * np.pi, N_RAYS, endpoint=False):
        lo, hi = r_lo(t), r_hi(t)
        if hi <= lo + 2:
            continue
        r = np.linspace(lo, hi, int(2 * (hi - lo)) + 2)
        x, y = cx + r * np.cos(t), cy + r * np.sin(t)
        ok = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
        if ok.sum() < 3:
            continue
        g = grad[y[ok].round().astype(int), x[ok].round().astype(int)]
        i = int(g.argmax())
        pts.append((x[ok][i], y[ok][i]))
        peaks.append(g[i])
    if len(pts) < 10:
        return None, 0.0
    pts, peaks = np.float32(pts), np.float32(peaks)
    keep = peaks >= np.quantile(peaks, 0.3)                        # drop the weakest 30% (gaps, glare)
    if keep.sum() < 6:
        return None, 0.0
    e = axis_aligned(cv2.fitEllipse(pts[keep]))
    ex, ey, erx, ery = e
    d = np.hypot((pts[:, 0] - ex) / max(erx, 1), (pts[:, 1] - ey) / max(ery, 1))  # 1.0 = on the ellipse
    on = np.mean(np.abs(d - 1) < 0.15)                              # share of rays that agree with the fit
    contrast = float(np.median(peaks) / (np.median(grad) + 1e-3))
    return e, float(on * min(1.0, contrast / 6))


HOLE_RATIO = 0.35   # centre/skin brightness: holes measured 0.06-0.24, intact heads (incl. Phillips) 0.45-0.75


def centre_ratio(gray, e):
    """Mean brightness of the centre (40% of the ellipse) over the skin ring around it (1.3x-1.9x)."""
    inner = ellipse_mask(gray.shape, e, 0.4)
    ring = ellipse_mask(gray.shape, e, 1.9) & ~ellipse_mask(gray.shape, e, 1.3)
    return float(gray[inner].mean() / max(gray[ring].mean(), 1.0)) if inner.any() and ring.any() else 1.0


def is_hole_like(gray, e, thr=None):
    """A hole's centre is near-black compared with the skin around it; a head's centre (even with a dark
    Phillips cross) is not. thr is unused (kept for the call signature)."""
    return centre_ratio(gray, e) < HOLE_RATIO


def measure_hole(img, box):
    """Find the dark hole inside the labelled box, then the countersink ring around it."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    gray = gray_of(img)
    bx = gray[int(max(y1, 0)):int(min(y2, H)), int(max(x1, 0)):int(min(x2, W))]
    med = float(np.median(bx))
    thr = min(float(np.percentile(bx, 12)), 0.6 * med)            # darkest pixels, and clearly darker than the skin
    # --- the core: the dark blob nearest the box centre
    m = np.zeros((H, W), np.uint8)
    m[int(max(y1, 0)):int(min(y2, H)), int(max(x1, 0)):int(min(x2, W))] = (bx < thr)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    core, best = None, -1.0
    bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < max(12, 0.004 * bw * bh):
            continue
        dist = np.hypot((cent[i][0] - bcx) / bw, (cent[i][1] - bcy) / bh)
        if dist > 0.35:                                             # must sit in the central part of the box
            continue
        s = a * (1 - dist)
        if s > best:
            cnts, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cnts and len(cnts[0]) >= 5:
                best, core = s, axis_aligned(cv2.fitEllipse(cnts[0]))
    side_lo, side_hi = 0.15 * min(bw, bh), 0.36 * max(bw, bh)
    if core is None:                                                # no clear hole: ellipse from the box
        ring = (bcx, bcy, 0.25 * bw, 0.25 * bh)
        return Hole(ring, None, thr, 0.2, "box")
    cx, cy, crx, cry = core
    # --- the countersink ring: strongest edge outside the core, inside the box
    rc = lambda t: (crx * cry) / max(np.hypot(cry * np.cos(t), crx * np.sin(t)), 1e-3)
    ring, conf = radial_fit(grad_of(gray), cx, cy, lambda t: 1.3 * rc(t),
                            lambda t: min(3.2 * rc(t), 0.55 * min(bw, bh)))
    ratio = np.sqrt(ring[2] * ring[3] / max(crx * cry, 1)) if ring else 0
    if ring is None or not 1.25 <= ratio <= 3.2 or conf < 0.25 or np.hypot(ring[0] - cx, ring[1] - cy) > 0.5 * max(crx, cry):
        r = np.clip(1.8 * np.array([crx, cry]), side_lo, side_hi)
        return Hole((cx, cy, float(r[0]), float(r[1])), core, thr, 0.5, "core x1.8", {"edge_conf": conf})
    rx, ry = np.clip([ring[2], ring[3]], side_lo, side_hi)
    return Hole((ring[0], ring[1], float(rx), float(ry)), core, thr, conf, "edge fit")


def inside_any(x, y, rects):
    return any(a <= x <= c and b <= y <= d for a, b, c, d in rects)


def classifier_crop(img, e):
    """Square crop around an ellipse cut like the classifier's training crops (box + 25% margin, 224 px)."""
    H, W = img.shape[:2]
    cx, cy, rx, ry = e
    s = int(round(min(2 * max(rx, ry) * 1.5, W, H)))
    l = int(np.clip(round(cx - s / 2), 0, W - s))
    t = int(np.clip(round(cy - s / 2), 0, H - s))
    return cv2.resize(img[t:t + s, l:l + s], (224, 224), interpolation=cv2.INTER_CUBIC)


def overlaps_any(e, rects, pad=1.5):
    """Does the ellipse's box, enlarged pad x, touch any rect? (a donor must be clear of every defect)"""
    cx, cy, rx, ry = e
    a, b, c, d = cx - pad * rx, cy - pad * ry, cx + pad * rx, cy + pad * ry
    return any(a < x2 and c > x1 and b < y2 and d > y1 for x1, y1, x2, y2 in rects)


def find_donors(img, boxes, hole, k=3, grow=0.15, scorer=None, min_p=0.5):
    """Intact screw heads in the photo, best first. boxes: every labelled (x1, y1, x2, y2) in the photo (all are
    excluded, including the other missing heads). scorer(img, ellipses) -> "is it a screw head" probability per
    candidate (heal_framework.make_head_scorer); candidates below min_p are dropped. Neither enhanced_res's own
    p(missing-head) (clean Phillips heads got 0.03) nor template matching with the bank (plain skin scored as
    high as heads) worked as this test.
    Each donor: dict(e=(cx, cy, rx, ry), score, parts...)."""
    H, W = img.shape[:2]
    gray = gray_of(img)
    grad = grad_of(gray)
    _, _, hrx, hry = hole.ring
    rects = [(x1 - grow * (x2 - x1), y1 - grow * (y2 - y1), x2 + grow * (x2 - x1), y2 + grow * (y2 - y1))
             for x1, y1, x2, y2 in boxes]
    rmin, rmax = 0.55 * min(hrx, hry), 1.7 * max(hrx, hry)
    g8 = np.clip(gray, 0, 255).astype(np.uint8)
    circ = cv2.HoughCircles(g8, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(6, 0.8 * min(hrx, hry)),
                            param1=70, param2=max(9, 0.3 * min(hrx, hry)),
                            minRadius=max(4, int(rmin)), maxRadius=max(6, int(rmax)))
    if circ is None:
        return []
    diag = np.hypot(W, H)
    out = []
    for x, y, r in circ[0][:200]:
        if inside_any(x, y, rects) or x - r < 2 or y - r < 2 or x + r > W - 3 or y + r > H - 3:
            continue
        e, conf = radial_fit(grad, x, y, lambda t: 0.6 * r, lambda t: 1.45 * r)
        if e is None or conf < 0.2:
            continue
        ex, ey, erx, ery = e
        if (overlaps_any(e, rects) or ex - erx < 2 or ey - ery < 2 or ex + erx > W - 3 or ey + ery > H - 3
                or is_hole_like(gray, e)):
            continue
        size = np.sqrt(erx * ery / max(hrx * hry, 1))
        if not 0.6 <= size <= 1.6:
            continue
        aspect = 1 - min(1.0, abs(np.log((erx / max(ery, 1)) / (hrx / max(hry, 1)))))
        near = 1 - min(1.0, np.hypot(ex - hole.ring[0], ey - hole.ring[1]) / (0.5 * diag))
        size_sim = 1 - min(1.0, abs(np.log(size)) / np.log(2))
        out.append(dict(e=(float(ex), float(ey), float(erx), float(ery)), conf=conf,
                        size=float(size), size_sim=float(size_sim), aspect=float(aspect), near=float(near)))
    out.sort(key=lambda d: -(d["conf"] + d["size_sim"] + d["aspect"]))   # geometry first: keep the 40 most plausible
    out = out[:40]
    if scorer is not None and out:
        p = scorer(img, [d["e"] for d in out])
        for d, pi in zip(out, p):
            d["p_fastener"] = float(pi)
        out = [d for d in out if d["p_fastener"] >= min_p]
    for d in out:
        d["score"] = float(2.0 * d.get("p_fastener", 0.5) + 0.6 * d["conf"] + 1.0 * d["size_sim"] + 1.0 * d["aspect"]
                           + 0.6 * d["near"])
    out.sort(key=lambda d: -d["score"])
    picked = []
    for d in out:                                                   # one candidate per physical head
        if all(np.hypot(d["e"][0] - p["e"][0], d["e"][1] - p["e"][1]) > 0.8 * min(d["e"][2], d["e"][3]) for p in picked):
            picked.append(d)
        if len(picked) == k:
            break
    return picked


def detail(gray, e):
    """Mean edge strength inside an ellipse: how much visible structure (rim, recess) a head has."""
    m = ellipse_mask(gray.shape, e, 1.0)
    return float(grad_of(gray)[m].mean()) if m.any() else 0.0


def quality_check(after, clean, hole, donor_mean, fx, fy, mask_pad=1.25, donor_detail=None):
    """Is the pasted head believable? Returns (ok, reasons)."""
    reasons = {}
    g_after, g_clean = gray_of(after), gray_of(clean)
    head = ellipse_mask(after.shape, hole.ring, 1.0)
    d_after, d_clean = detail(g_after, hole.ring), detail(g_clean, hole.ring)
    if d_after < 1.6 * d_clean + 2.0:                               # head hardly visible on the plain LaMa skin
        reasons["head_not_visible"] = round(d_after / (d_clean + 1e-3), 2)
    elif donor_detail and d_after < 0.4 * donor_detail:            # blending washed most of the head away
        reasons["head_washed_out"] = round(d_after / donor_detail, 2)
    if is_hole_like(g_after, hole.ring):
        reasons["still_looks_like_a_hole"] = True
    m = float(g_after[head].mean())
    if not 0.6 <= m / max(donor_mean, 1) <= 1.6:
        reasons["brightness_ratio"] = round(m / max(donor_mean, 1), 2)
    edge = ellipse_mask(after.shape, hole.ring, mask_pad * 1.05) & ~ellipse_mask(after.shape, hole.ring, mask_pad * 0.95)
    seam = float(grad_of(g_after)[edge].mean() / (grad_of(g_clean)[edge].mean() + 1e-3))
    if seam > 3.0:
        reasons["seam_ratio"] = round(seam, 2)
    if not (0.5 <= fx <= 2.0 and 0.5 <= fy <= 2.0):
        reasons["scale"] = (round(fx, 2), round(fy, 2))
    metrics = dict(detail_vs_lama=round(float(d_after / (d_clean + 1e-3)), 2), detail=round(float(d_after), 1),
                   detail_vs_donor=round(float(d_after / donor_detail), 2) if donor_detail else None,
                   seam=round(float(seam), 2), brightness=round(float(m / max(donor_mean, 1)), 2))
    return not reasons, dict(reasons, metrics=metrics)
