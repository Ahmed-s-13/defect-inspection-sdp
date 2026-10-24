"""Outside test set: D5 (Roboflow aircraft damage) crops whose photo is not in D1.

D5 shares ~42% of its photos with D1, so a D5 photo counts as "seen" (and is dropped) when any of these hold:
  - the project's overlap check paired it with a D1 photo (phash <= 6 or same original file name)
  - perceptual hash within 8 bits of a D1 photo, over the 8 flips/90-degree rotations of the D5 photo
  - it meets d1_grouped's same-source rule with one of its 30 nearest D1 photos (ResNet-50 cosine over the 8
    orientations), checked with SIFT keypoint matching + RANSAC exactly as in d1_grouped: >= 8 inliers, or
    >= 6 with cosine >= 0.75, or cosine >= 0.85, or phash <= 2
A cosine cut-off alone was not enough: photos just under cosine 0.85 were still mirrored/recoloured D1 copies.
Wrongly dropping an unseen photo only shrinks the test set, while missing a copy would inflate it. D5's own near-duplicates (overlap check) are reduced
to one photo each, so no scene counts several times. Classes: Dent -> dent, Fastener Damage -> missing-head
(fastener damage includes missing heads but is broader), Rupture -> dropped (no D1 class).

Needs D5's raw photos (data/raw/D5) and crops (data/work/crops/D5.csv), plus d1_grouped's caches
(data/work/d1_grouped_cache, from ../d1_grouped/group.py features). From 1_main_model_resnet18/:
  python enhanced_res/external_d5.py      # ~15 min

Output: results/d5/ext_d5.csv (one row per kept crop), results/d5/d5_images.csv (the decision for every D5 photo)
and results/d5/d5_vs_d1_*.jpg for checking the cut-off by eye. Caches in data/work/enhanced_res/d5_cache.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent
sys.path.insert(0, str(MAIN))
GROUPED = MAIN / "data/work/d1_grouped_cache"          # d1_grouped's features (group.py features)
CACHE = MAIN / "data/work/enhanced_res/d5_cache"
OUT = HERE / "results/d5"

import imagehash
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image, ImageDraw, ImageOps

from klshift.common import UnionFind

RAW5 = MAIN / "data/raw/D5"
RAW1 = MAIN / "data/raw/D1"
CLASS = {"Dent": "dent", "Fastener Damage": "missing-head"}


def load(path):
    img = Image.open(path)
    img.draft("RGB", (512, 512))
    return img.convert("RGB")


def dihedral(img):
    m = ImageOps.mirror(img)
    return [img, img.rotate(90, expand=True), img.rotate(180), img.rotate(270, expand=True),
            m, ImageOps.flip(img), m.rotate(90, expand=True), m.rotate(270, expand=True)]


def popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


class Views(torch.utils.data.Dataset):
    """8 orientations of one D5 photo: ResNet-50 input tensors and phashes (same preprocessing as d1_grouped)."""
    def __init__(self, paths):
        self.paths = paths
        self.tf = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        v = dihedral(load(self.paths[i]))
        h = torch.tensor([int(str(imagehash.phash(x)), 16) - (1 << 63) for x in v], dtype=torch.int64)
        return torch.stack([self.tf(x) for x in v]), h


def sift_one(path, nkp=500, side=512):
    """d1_grouped's SIFT settings: 500 keypoints at 512 px, for the image and its mirror."""
    import cv2
    cv2.setNumThreads(1)
    img = load(path)
    f = side / max(img.size)
    if f < 1:
        img = img.resize((round(img.width * f), round(img.height * f)))
    g = np.asarray(img.convert("L"))
    sift = cv2.SIFT_create(nfeatures=nkp)
    out = []
    for im in (g, g[:, ::-1].copy()):
        kp, d = sift.detectAndCompute(im, None)
        kp = kp[:nkp]
        pts = np.array([k.pt for k in kp], np.float32).reshape(-1, 2)
        desc = (np.clip(d[:nkp], 0, 255).astype(np.uint8) if d is not None else np.zeros((0, 128), np.uint8))
        out.append((pts, desc))
    return out


_W = {}


def _winit():
    import cv2
    cv2.setNumThreads(1)
    _W["d5_desc"] = np.load(CACHE / "cache_d5_sift_desc.npy", mmap_mode="r")
    _W["d5_pts"] = np.load(CACHE / "cache_d5_sift_pts.npy", mmap_mode="r")
    _W["d5_cnt"] = np.load(CACHE / "cache_d5_sift_cnt.npy")
    _W["desc"] = np.load(GROUPED / "sift_desc.npy", mmap_mode="r")
    _W["pts"] = np.load(GROUPED / "sift_pts.npy", mmap_mode="r")
    _W["cnt"] = np.load(GROUPED / "sift_cnt.npy")
    _W["bf"] = cv2.BFMatcher(cv2.NORM_L2)


def _inliers(pi, di, j):
    """d1_grouped's guarded RANSAC affine inlier count between a D5 view (pts, desc) and D1 photo j."""
    import cv2
    cj = _W["cnt"][j, 0]
    if len(di) < 8 or cj < 8:
        return 0
    ms = _W["bf"].knnMatch(di.astype(np.float32), _W["desc"][j, 0, :cj].astype(np.float32), k=2)
    good = {}
    for m in ms:
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            if m[0].trainIdx not in good or m[0].distance < good[m[0].trainIdx].distance:
                good[m[0].trainIdx] = m[0]
    good = list(good.values())
    if len(good) < 6:
        return 0
    src = pi[[m.queryIdx for m in good]]
    dst = _W["pts"][j, 0][[m.trainIdx for m in good]]
    A, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=2000)
    if A is None:
        return 0
    sv = np.linalg.svd(A[:, :2], compute_uv=False)
    if sv[1] < 0.3 or sv[0] > 3.3 or sv[0] / sv[1] > 2.5:
        return 0
    inl = inl.ravel().astype(bool)
    for p in (src[inl], dst[inl]):
        if len(p) < 3 or np.ptp(p[:, 0]) < 60 or np.ptp(p[:, 1]) < 60:
            return 0
    return int(inl.sum())


def _verify(job):
    i, js = job
    views = [(_W["d5_pts"][i, m, :_W["d5_cnt"][i, m]], _W["d5_desc"][i, m, :_W["d5_cnt"][i, m]]) for m in (0, 1)]
    return [max(_inliers(p, d, j) for p, d in views) for j in js]


def tile(a, b, text, s=200):
    t = Image.new("RGB", (2 * s + 8, s + 20), "white")
    t.paste(load(a).resize((s, s)), (0, 20))
    t.paste(load(b).resize((s, s)), (s + 8, 20))
    ImageDraw.Draw(t).text((2, 4), text, fill="black")
    return t


def grid(tiles, path, cols=4):
    w, h = tiles[0].size
    g = Image.new("RGB", (cols * (w + 10), -(-len(tiles) // cols) * (h + 10)), "white")
    for k, t in enumerate(tiles):
        g.paste(t, ((k % cols) * (w + 10), (k // cols) * (h + 10)))
    g.save(path, quality=88)


def main():
    import argparse
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    crops = pd.read_csv(MAIN / "data/work/crops/D5.csv")
    imgs = sorted(crops.image.unique())
    paths = [RAW5 / i for i in imgs]
    assert all(p.exists() for p in paths), "missing D5 raw images"
    f = CACHE / "cache_d5_views.npz"
    if not f.exists():
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        net = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        net.fc = torch.nn.Identity()
        net = net.eval().to(dev)
        E, H = [], []
        with torch.no_grad():
            for k, (x, h) in enumerate(torch.utils.data.DataLoader(Views(paths), batch_size=16, num_workers=6)):
                e = torch.nn.functional.normalize(net(x.flatten(0, 1).to(dev)), dim=1).cpu().numpy()
                E.append(e.reshape(len(x), 8, -1).astype(np.float16))
                H.append(h.numpy())                                   # phash - 2**63, as int64
                if dev == "mps":
                    torch.mps.empty_cache()
                if k % 50 == 0:
                    print(f"  D5 views {min((k + 1) * 16, len(paths))}/{len(paths)}", flush=True)
        np.savez(f, emb=np.concatenate(E), ph=np.concatenate(H))
    z = np.load(f)
    e5 = torch.from_numpy(z["emb"].astype(np.float32))
    h5 = (z["ph"].astype(np.int64).view(np.uint64) + np.uint64(1 << 63))      # undo the signed shift

    idx1 = pd.read_csv(GROUPED / "index.csv")
    e1 = torch.from_numpy(np.load(GROUPED / "emb8.npy", mmap_mode="r")[:, 0].astype(np.float32))
    h1 = np.load(GROUPED / "phash8.npy")[:, 0]
    best_cos, best_cos_j, best_ph, best_ph_j = [], [], [], []
    for a in range(0, len(imgs), 256):
        s = torch.stack([e5[a:a + 256, o] @ e1.T for o in range(8)]).max(0).values
        v, j = s.max(1)
        best_cos += v.tolist()
        best_cos_j += j.tolist()
        d = popcount(h5[a:a + 256, :, None] ^ h1[None, None, :]).min(1)
        best_ph += d.min(1).astype(int).tolist()
        best_ph_j += d.argmin(1).tolist()
    im = pd.DataFrame({"image": imgs, "cos": best_cos, "cos_match": idx1.key.values[best_cos_j],
                       "phash": best_ph, "phash_match": idx1.key.values[best_ph_j]})
    dp = pd.read_csv(MAIN / "data/work/dup_pairs.csv")
    dp = dp[dp.dup]
    in_d1 = set(dp[(dp.ds_a == "D1") & (dp.ds_b == "D5")].image_b) | set(dp[(dp.ds_b == "D1") & (dp.ds_a == "D5")].image_a)
    im["overlap_check"] = im.image.isin(in_d1)
    # SIFT verification against each D5 photo's 30 nearest D1 photos by cosine
    from concurrent.futures import ProcessPoolExecutor
    if not (CACHE / "cache_d5_sift_cnt.npy").exists():
        desc = np.lib.format.open_memmap(CACHE / "cache_d5_sift_desc.npy", "w+", np.uint8, (len(paths), 2, 500, 128))
        pts = np.zeros((len(paths), 2, 500, 2), np.float32)
        cnt = np.zeros((len(paths), 2), np.int32)
        with ProcessPoolExecutor() as ex:
            for i, r in enumerate(ex.map(sift_one, paths, chunksize=32)):
                for m, (p, d) in enumerate(r):
                    pts[i, m, :len(p)], desc[i, m, :len(d)], cnt[i, m] = p, d, len(p)
        desc.flush()
        del desc
        np.save(CACHE / "cache_d5_sift_pts.npy", pts)
        np.save(CACHE / "cache_d5_sift_cnt.npy", cnt)
    K = 30
    nn_j, nn_c = [], []
    for a in range(0, len(imgs), 256):
        s5 = torch.stack([e5[a:a + 256, o] @ e1.T for o in range(8)]).max(0).values
        v, j = s5.topk(K, dim=1)
        nn_j.append(j.numpy())
        nn_c.append(v.numpy())
    nn_j, nn_c = np.concatenate(nn_j), np.concatenate(nn_c)
    with ProcessPoolExecutor(initializer=_winit) as ex:
        inl = np.array(list(ex.map(_verify, zip(range(len(imgs)), nn_j.tolist()), chunksize=16)))
    match = (inl >= 8) | ((inl >= 6) & (nn_c >= 0.75)) | (nn_c >= 0.85)
    im["best_inliers"] = inl.max(1)
    im["sift_match"] = match.any(1)
    im["seen_in_d1"] = im.overlap_check | (im.phash <= 2) | im.sift_match

    # one photo per D5-internal near-duplicate group
    uf = UnionFind()
    for r in dp[(dp.ds_a == "D5") & (dp.ds_b == "D5")].itertuples():
        uf.union(r.image_a, r.image_b)
    im["d5_group"] = [uf.find(i) for i in im.image]
    keep = im[~im.seen_in_d1].groupby("d5_group").head(1).image
    im["kept"] = im.image.isin(set(keep))
    im.to_csv(OUT / "d5_images.csv", index=False)

    c = crops[crops.image.isin(set(keep)) & crops.cls.isin(CLASS)].copy()
    c["cls"] = c.cls.map(CLASS)
    c[["crop_path", "cls", "image"]].to_csv(OUT / "ext_d5.csv", index=False)

    # sheets: dropped photos near the cut-off, and kept photos with their closest D1 photo
    dropped = im[im.seen_in_d1 & ~im.overlap_check & (im.cos < 0.85)].sort_values("best_inliers").head(24)
    kept = im[im.kept].sort_values("cos", ascending=False).head(24)
    kept2 = im[im.kept].sort_values("best_inliers", ascending=False).head(24)
    for name, d in [("dropped_closest_to_cutoff", dropped), ("kept_closest_to_d1", kept),
                    ("kept_most_sift_inliers", kept2)]:
        if len(d):
            grid([tile(RAW5 / r.image, RAW1 / "content/Aircraft_dataset/images" / r.cos_match,
                       f"cos {r.cos:.3f} phash {r.phash} inl {r.best_inliers}") for r in d.itertuples()], OUT / f"d5_vs_d1_{name}.jpg")
    print(f"D5 photos: {len(im)}; seen in D1: {int(im.seen_in_d1.sum())} (overlap check {int(im.overlap_check.sum())}, "
          f"phash<=2 {int((im.phash <= 2).sum())}, same-source rule {int(im.sift_match.sum())}); unseen: "
          f"{int((~im.seen_in_d1).sum())}; after D5-internal dedup: {len(keep)}")
    print(f"kept crops: {len(c)}", c.cls.value_counts().to_dict())


if __name__ == "__main__":
    main()
