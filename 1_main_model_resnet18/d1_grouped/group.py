"""Find "same source photo" pairs across all of D1 (train + val + test), for a leak-free regrouped split.

D1's file names are anonymised and its splits hold augmented copies of one another (flips, 90-degree rotations,
crops, greyscale, small arbitrary rotations with black corners, shear, colour shifts), so copies are found by
content in three stages, each cached under cache/:

  features  per image: phash and ImageNet ResNet-50 embedding of all 8 flips/90-degree rotations, and SIFT
            keypoints of the image and its mirror (SIFT is rotation/scale invariant but not mirror invariant)
  pairs     candidates = each image's top-k embedding neighbours (cosine, max over the 8 ops) plus its
            top-k phash neighbours (Hamming, min over the 8 ops); every candidate pair is then verified by
            SIFT matching (ratio test) + RANSAC affine fit, which catches small rotations, shear and crops that
            move phash and the embedding. Output: cache/pairs.csv with cos, phash, inliers per pair.
  sheet     pair sheets across score bands (sheets/*.jpg) for setting the match thresholds by eye

Nothing in the D1 folder is written to. Needs D1's raw photos in ../data/raw/D1 (D1_raw.zip); caches go to
../data/work/d1_grouped_cache (about 2.3 GB, git-ignored). Run from this folder:
  python group.py features      # ~25 min on an M-series Mac
  python group.py pairs         # ~3 min
  python group.py sheet         # optional pair sheets for checking the thresholds
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

HERE = Path(__file__).resolve().parent
MAIN = HERE.parent                                                  # 1_main_model_resnet18
CACHE = MAIN / "data/work/d1_grouped_cache"
D1 = MAIN / "data/raw/D1/content/Aircraft_dataset"
OPS = ["id", "rot90", "rot180", "rot270", "flipH", "flipV", "transpose", "transverse"]
NKP = 500           # SIFT keypoints kept per image
SIFT_SIDE = 512     # longest side for SIFT


def index():
    f = CACHE / "index.csv"
    if not f.exists():
        rows = [{"key": f"{s}/{p.name}", "split": s} for s in ["train", "val", "test"]
                for p in sorted((D1 / "images" / s).glob("*.jpg"))]
        pd.DataFrame(rows).to_csv(f, index=False)
    return pd.read_csv(f)


def load(key, side=512):
    img = Image.open(D1 / "images" / key)
    img.draft("RGB", (side, side))
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


# ---------------------------------------------------------------- features

def phash8(key):
    return [int(str(imagehash.phash(i)), 16) for i in dihedral(load(key))]


def sift_one(key):
    cv2.setNumThreads(1)
    img = load(key, SIFT_SIDE)
    s = SIFT_SIDE / max(img.size)
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)))
    g = np.asarray(img.convert("L"))
    sift = cv2.SIFT_create(nfeatures=NKP)
    out = []
    for im in (g, g[:, ::-1].copy()):
        kp, d = sift.detectAndCompute(im, None)
        kp, d = kp[:NKP], (d[:NKP] if d is not None else np.zeros((0, 128), np.float32))
        pts = np.zeros((NKP, 2), np.float32)
        desc = np.zeros((NKP, 128), np.uint8)
        pts[:len(kp)] = np.array([k.pt for k in kp], np.float32).reshape(-1, 2)
        desc[:len(kp)] = np.clip(d, 0, 255).astype(np.uint8)
        out.append((pts, desc, len(kp)))
    return out


class EmbedDS:
    """All 8 flips/rotations of one image, ImageNet-normalised at 224x224."""
    def __init__(self, keys):
        import torchvision
        self.keys = keys
        self.tf = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)), torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        import torch
        return torch.stack([self.tf(x) for x in dihedral(load(self.keys[i]))])


def embed_all(keys):
    import torch
    import torchvision

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    net = torchvision.models.resnet50(weights="IMAGENET1K_V2")
    net.fc = torch.nn.Identity()
    net = net.eval().to(device)
    out = np.lib.format.open_memmap(CACHE / "emb8.npy", "w+", np.float16, (len(keys), 8, 2048))
    dl = torch.utils.data.DataLoader(EmbedDS(keys), batch_size=16, num_workers=6)   # 16 images x 8 ops = 128 per batch
    i, t0 = 0, time.time()
    with torch.no_grad():
        for b in dl:
            f = net(b.flatten(0, 1).to(device))
            f = torch.nn.functional.normalize(f, dim=1).cpu().numpy().reshape(len(b), 8, -1)
            out[i:i + len(b)] = f.astype(np.float16)
            i += len(b)
            if device == "mps":
                torch.mps.empty_cache()
            if (i // 16) % 50 == 0:
                print(f"  embedded {i}/{len(keys)}  {time.time() - t0:.0f}s", flush=True)
    out.flush()


def features():
    keys = index().key.tolist()
    print(f"{len(keys)} images")
    if not (CACHE / "phash8.npy").exists():
        with ProcessPoolExecutor() as ex:
            h = np.array(list(ex.map(phash8, keys, chunksize=64)), np.uint64)
        np.save(CACHE / "phash8.npy", h)
        print("phash done", flush=True)
    if not (CACHE / "sift_desc.npy").exists():
        n = len(keys)
        desc = np.lib.format.open_memmap(CACHE / "sift_desc.tmp.npy", "w+", np.uint8, (n, 2, NKP, 128))
        pts = np.zeros((n, 2, NKP, 2), np.float32)
        cnt = np.zeros((n, 2), np.int32)
        with ProcessPoolExecutor() as ex:
            for i, r in enumerate(ex.map(sift_one, keys, chunksize=32)):
                for m, (p, d, c) in enumerate(r):
                    pts[i, m], desc[i, m], cnt[i, m] = p, d, c
                if i % 2000 == 0:
                    print(f"  sift {i}/{n}", flush=True)
        desc.flush()
        del desc
        np.save(CACHE / "sift_pts.npy", pts)
        np.save(CACHE / "sift_cnt.npy", cnt)
        os.rename(CACHE / "sift_desc.tmp.npy", CACHE / "sift_desc.npy")
        print("sift done", flush=True)
    if not (CACHE / "emb8.done").exists():
        embed_all(keys)
        (CACHE / "emb8.done").touch()
        print("embeddings done", flush=True)


# ---------------------------------------------------------------- pairs

def candidates(k_emb, k_ph, min_cos, max_ph):
    """Top-k neighbours per image by embedding cosine and by phash, both taken over the 8 flips/rotations."""
    import torch
    e = torch.from_numpy(np.load(CACHE / "emb8.npy", mmap_mode="r")[:]).float()
    n = len(e)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    db = e[:, 0].to(dev)
    h = np.load(CACHE / "phash8.npy")
    h0 = h[:, 0]
    cos = {}
    ph = {}
    for a in range(0, n, 512):
        b = min(a + 512, n)
        s = torch.full((b - a, n), -1.0, device=dev)
        for o in range(8):
            s = torch.maximum(s, e[a:b, o].to(dev) @ db.T)
        s[torch.arange(b - a), torch.arange(a, b)] = -1
        v, j = s.topk(k_emb, dim=1)
        v, j = v.cpu().numpy(), j.cpu().numpy()
        d = popcount(h[a:b, :, None] ^ h0[None, None, :]).min(1).astype(np.int16)    # (b-a) x n
        d[np.arange(b - a), np.arange(a, b)] = 99
        jp = np.argpartition(d, k_ph, axis=1)[:, :k_ph]
        for r in range(b - a):
            i = a + r
            for jj, vv in zip(j[r], v[r]):
                if vv >= min_cos:
                    p = (min(i, int(jj)), max(i, int(jj)))
                    cos[p] = max(cos.get(p, -1), float(vv))
            for jj in jp[r]:
                if d[r, jj] <= max_ph:
                    p = (min(i, int(jj)), max(i, int(jj)))
                    ph[p] = min(ph.get(p, 99), int(d[r, jj]))
        print(f"  candidates {b}/{n}: {len(set(cos) | set(ph))} pairs", flush=True)
    pairs = sorted(set(cos) | set(ph))
    # fill in the missing score of each pair so every row has both
    ii = np.array([p[0] for p in pairs])
    jj = np.array([p[1] for p in pairs])
    ec = np.empty(len(pairs), np.float32)
    for a in range(0, len(pairs), 20000):
        x, y = e[ii[a:a + 20000]], e[jj[a:a + 20000]]
        c1 = torch.einsum("nod,nd->no", x, y[:, 0]).max(1).values
        c2 = torch.einsum("nod,nd->no", y, x[:, 0]).max(1).values
        ec[a:a + 20000] = torch.maximum(c1, c2).numpy()
    pd_ = np.minimum(popcount(h[ii] ^ h0[jj][:, None]).min(1), popcount(h[jj] ^ h0[ii][:, None]).min(1))
    return pd.DataFrame({"i": ii, "j": jj, "cos": ec, "phash": pd_.astype(int)})


_W = {}


def _winit():
    cv2.setNumThreads(1)
    _W["desc"] = np.load(CACHE / "sift_desc.npy", mmap_mode="r")
    _W["pts"] = np.load(CACHE / "sift_pts.npy", mmap_mode="r")
    _W["cnt"] = np.load(CACHE / "sift_cnt.npy")
    _W["bf"] = cv2.BFMatcher(cv2.NORM_L2)


def _inliers(i, mi, j):
    """RANSAC affine inliers between image i (mirrored if mi) and image j.

    Guards against the false matches seen on random pairs: one-to-one matches only, a plausible affine
    (scale 0.3-3.3, anisotropy <= 2.5, as crops/shear/small rotations give) and inliers that cover a real
    area of both images rather than one rivet."""
    ci, cj = _W["cnt"][i, mi], _W["cnt"][j, 0]
    if ci < 8 or cj < 8:
        return 0, 0
    di = _W["desc"][i, mi, :ci].astype(np.float32)
    dj = _W["desc"][j, 0, :cj].astype(np.float32)
    ms = _W["bf"].knnMatch(di, dj, k=2)
    good = {}
    for m in ms:
        if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
            if m[0].trainIdx not in good or m[0].distance < good[m[0].trainIdx].distance:
                good[m[0].trainIdx] = m[0]
    good = list(good.values())
    if len(good) < 6:
        return len(good), 0
    src = _W["pts"][i, mi][[m.queryIdx for m in good]]
    dst = _W["pts"][j, 0][[m.trainIdx for m in good]]
    A, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0, maxIters=2000)
    if A is None:
        return len(good), 0
    sv = np.linalg.svd(A[:, :2], compute_uv=False)
    if sv[1] < 0.3 or sv[0] > 3.3 or sv[0] / sv[1] > 2.5:
        return len(good), 0
    inl = inl.ravel().astype(bool)
    for p in (src[inl], dst[inl]):
        if len(p) < 3 or np.ptp(p[:, 0]) < 60 or np.ptp(p[:, 1]) < 60:     # px at SIFT_SIDE 512
            return len(good), 0
    return len(good), int(inl.sum())


def verify(pair):
    i, j = pair
    r0, r1 = _inliers(i, 0, j), _inliers(i, 1, j)
    best = max(r0, r1, key=lambda r: r[1])
    return best[0], best[1], int(r1[1] > r0[1])


def pairs(a):
    f = CACHE / "candidates.csv"
    if not f.exists():
        candidates(a.k_emb, a.k_ph, a.min_cos, a.max_ph).to_csv(f, index=False)
    df = pd.read_csv(f)
    print(f"{len(df)} candidate pairs; verifying with SIFT + RANSAC", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(initializer=_winit) as ex:
        res = list(ex.map(verify, zip(df.i, df.j), chunksize=256))
    df["good"], df["inliers"], df["mirror"] = zip(*res)
    print(f"verified in {time.time() - t0:.0f}s")
    keys = index().key
    df["a"], df["b"] = keys[df.i].values, keys[df.j].values
    df.to_csv(CACHE / "pairs.csv", index=False)
    print(df.describe().to_string())


# ---------------------------------------------------------------- sheets

def pair_tile(ka, kb, text, red, s=200):
    t = Image.new("RGB", (2 * s + 8, s + 20), "white")
    t.paste(load(ka).resize((s, s)), (0, 20))
    t.paste(load(kb).resize((s, s)), (s + 8, 20))
    ImageDraw.Draw(t).text((2, 4), text, fill="red" if red else "black")
    return t


def grid(tiles, path, cols=4):
    w, h = tiles[0].size
    g = Image.new("RGB", (cols * (w + 10), -(-len(tiles) // cols) * (h + 10)), "white")
    for k, t in enumerate(tiles):
        g.paste(t, ((k % cols) * (w + 10), (k // cols) * (h + 10)))
    g.save(path, quality=88)


def sheet(a):
    df = pd.read_csv(CACHE / "pairs.csv")
    out = HERE / "sheets"
    out.mkdir(exist_ok=True)
    bands = [(0, 5), (5, 8), (8, 12), (12, 16), (16, 20), (20, 30), (30, 50), (50, 10 ** 6)]
    for lo, hi in bands:
        sub = df[(df.inliers >= lo) & (df.inliers < hi)]
        print(f"inliers {lo}-{hi}: {len(sub)} pairs")
        if a.cos_max is not None:
            sub = sub[sub.cos < a.cos_max]          # look at the hard ones (low embedding similarity)
        if not len(sub):
            continue
        sub = sub.sample(min(24, len(sub)), random_state=a.seed)
        tiles = [pair_tile(r.a, r.b, f"inl {r.inliers} good {r.good} cos {r.cos:.2f} ph {r.phash}"
                                     f"{' M' if r.mirror else ''}", False) for r in sub.itertuples()]
        grid(tiles, out / f"inliers_{lo:03d}-{min(hi, 999):03d}{a.tag}.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["features", "pairs", "sheet"])
    ap.add_argument("--k-emb", type=int, default=20)
    ap.add_argument("--k-ph", type=int, default=10)
    ap.add_argument("--min-cos", type=float, default=0.6)
    ap.add_argument("--max-ph", type=int, default=22)
    ap.add_argument("--cos-max", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    {"features": features, "pairs": lambda: pairs(a), "sheet": lambda: sheet(a)}[a.stage]()


if __name__ == "__main__":
    main()
