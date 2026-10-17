"""Step 5: extract 512-d embeddings (input to fc) for every crop of D1-D5.

Also extracts D1-held-out (val+test) under synthetic corruptions at 3 severities
for the known-shift sanity check.

Outputs:
  <results>/embeddings.npz     emb, dataset, split, cls, image, crop_path, group, in_d1, in_d1_train, rep
                               (group = near-copy cluster; rep = crop of the one image kept per cluster)
  <work>/embeddings_corrupt.npz emb, corruption, severity, row (index into embeddings.npz)
"""
import argparse
import io
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader

from klshift.common import (ALL_SETS, CLASS_SUBSETS, NO_DAMAGE, VIDEO, CropDataset, UnionFind, build_model, embed_batch, load_crops,
                            pick_device, seed_everything, source_of)

SEVERITIES = {  # (parameter per severity 1, 2, 3)
    "gaussian_noise": [0.04, 0.08, 0.16],   # std as a fraction of 255
    "blur": [1.0, 2.0, 3.0],                # Gaussian blur radius, px
    "low_contrast": [0.5, 0.3, 0.15],       # contrast factor (1 = unchanged)
    "jpeg": [20, 10, 5],                    # JPEG quality
}


class Corrupt:
    def __init__(self, kind, level):
        self.kind, self.level = kind, level

    def __call__(self, img):
        if self.kind == "gaussian_noise":
            rng = np.random.default_rng(zlib.crc32(img.tobytes()))
            x = np.asarray(img, dtype=np.float32) + rng.normal(0, self.level * 255, (*img.size[::-1], 3))
            return Image.fromarray(np.clip(x, 0, 255).astype(np.uint8))
        if self.kind == "blur":
            return img.filter(ImageFilter.GaussianBlur(self.level))
        if self.kind == "low_contrast":
            return ImageEnhance.Contrast(img).enhance(self.level)
        if self.kind == "jpeg":
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=self.level)
            return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        raise ValueError(self.kind)


def video_leaders(images, hashes, thr):
    """Greedy, non-chaining near-copy grouping for video frames: walk frames in order and keep a frame
    only if it is more than thr bits from every frame kept so far; others join their nearest kept frame.
    (Chaining would merge a whole slow pan into one group.)"""
    order = sorted(images)
    h = np.array([int(hashes[i], 16) for i in order], dtype=np.uint64)
    kept, group = [], {}
    for i, img in enumerate(order):
        if kept:
            d = np.bitwise_count(h[kept] ^ h[i])
            j = int(np.argmin(d))
            if d[j] <= thr:
                group[img] = order[kept[j]]
                continue
        kept.append(i)
        group[img] = img
    return group


@torch.no_grad()
def extract(model, paths, device, batch, workers, pil_fn=None):
    dl = DataLoader(CropDataset(paths, pil_fn=pil_fn), batch_size=batch, num_workers=workers)
    model.eval()
    return np.concatenate([embed_batch(model, x.to(device)).cpu().numpy() for x, _ in dl]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="data/work")
    ap.add_argument("--results", default="results")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--corrupt-n", type=int, default=0, help="cap on D1-held-out crops to corrupt (0 = all)")
    ap.add_argument("--datasets", nargs="+", default=ALL_SETS,
                    help="crop sets to embed, incl. derived ones like D6p or D1nd (D1 always included)")
    ap.add_argument("--thr", type=int, default=6, help="phash threshold for video-frame near-copies")
    ap.add_argument("--append", action="store_true",
                    help="embed only --datasets and add them to the existing embeddings.npz (replacing those sets); "
                         "skips the corruption sets")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    work, res = Path(a.work), Path(a.results)
    seed_everything(a.seed)
    device = pick_device()

    ck = torch.load(work / "model_best.pt", map_location="cpu")
    model = build_model(len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state"])
    model.to(device)

    split = pd.read_csv(work / "d1_split.csv")
    pairs = pd.read_csv(work / "dup_pairs.csv")
    pairs = pairs[pairs.dup]
    # make pairs symmetric, keep those whose other side is D1
    sym = pd.concat([pairs[["ds_a", "image_a", "ds_b", "image_b"]],
                     pairs[["ds_b", "image_b", "ds_a", "image_a"]].set_axis(
                         ["ds_a", "image_a", "ds_b", "image_b"], axis=1)])
    to_d1 = sym[sym.ds_b == "D1"]
    d1_train_imgs = set(split.image[split.split == "train"])
    match_d1 = set(zip(to_d1.ds_a, to_d1.image_a))
    match_d1_train = set((d, i) for d, i, j in zip(to_d1.ds_a, to_d1.image_a, to_d1.image_b) if j in d1_train_imgs)

    frames = []
    img_split = split.drop_duplicates("image").set_index("image")[["split", "group_id"]]
    hashes = pd.read_csv(work / "hashes.csv")
    for ds in [d for d in ALL_SETS if d == "D1" or d in a.datasets]:
        src = source_of(ds)
        f = work / "crops" / f"{src if ds in CLASS_SUBSETS else ds}.csv"
        if not f.exists():
            continue
        if ds in CLASS_SUBSETS:  # e.g. D10n / D10a: the source's crops of some classes
            c = load_crops(work, src)
            c = c[c.cls.isin(CLASS_SUBSETS[ds][1])].assign(dataset=ds).reset_index(drop=True)
        else:
            c = load_crops(work, ds)  # derived sets (D6p, D1nd, ...) inherit their source's overlap results
        if src == "D1":
            # D1 and D1nd follow the D1 split by original image
            c = c.drop(columns=[x for x in ["split", "group_id"] if x in c]).join(img_split, on="image", how="inner")
            c["group"] = "D1:g" + c.group_id.astype(str)
            c["in_d1"] = True
            c["in_d1_train"] = (c.split == "train").to_numpy() | np.array([("D1", i) in match_d1_train for i in c.image], bool)
        else:
            c["split"] = "all"
            c["in_d1"] = [(src, i) in match_d1 for i in c.image]
            c["in_d1_train"] = [(src, i) in match_d1_train for i in c.image]
        if src in VIDEO:
            hh = hashes[hashes.dataset == src].set_index("image").phash
            leader = video_leaders(c.image.unique(), hh, a.thr)
            c["group"] = [f"{ds}:{leader[i]}" for i in c.image]
            c["rep"] = c.image.isin(set(leader.values()))
            c["in_d1"] = c.groupby("group").in_d1.transform("any")
            c["in_d1_train"] = c.groupby("group").in_d1_train.transform("any")
        elif src != "D1":
            # near-copy clusters inside the dataset (phash or same original file name).
            # Tiles of one photo are different patches, so they are NOT merged here.
            # In the whole-aircraft sets (plane on plain sky) phash matches different planes, so there
            # only same-file-name copies (e.g. Roboflow augmentations) are merged.
            same = pairs[(pairs.ds_a == src) & (pairs.ds_b == src)]
            if src in NO_DAMAGE:
                same = same[same.name_match]
            uf = UnionFind()
            for r in same.itertuples():
                uf.union(("img", r.image_a), ("img", r.image_b))
            roots = {}
            c["group"] = [f"{ds}:c{roots.setdefault(uf.find(('img', im)), len(roots))}" for im in c.image]
            # a cluster is "in D1" if any of its copies is
            c["in_d1"] = c.groupby("group").in_d1.transform("any")
            c["in_d1_train"] = c.groupby("group").in_d1_train.transform("any")
        # one representative image per cluster (D1 clusters never straddle splits)
        if "rep" not in c:
            imgs = c[["group", "image"]].drop_duplicates().sample(frac=1, random_state=a.seed)
            reps = set(imgs.drop_duplicates("group").image)
            c["rep"] = c.image.isin(reps)
        frames.append(c[["dataset", "split", "cls", "image", "crop_path", "group", "in_d1", "in_d1_train", "rep"]])
    meta = pd.concat(frames, ignore_index=True)

    if a.append:
        meta = meta[meta.dataset.isin([d for d in a.datasets])].reset_index(drop=True)
    print(f"extracting {len(meta)} crops on {device} ...")
    emb = extract(model, meta.crop_path, device, a.batch, a.workers)
    if a.append:
        old = np.load(res / "embeddings.npz", allow_pickle=False)
        keep = ~np.isin(old["dataset"], meta.dataset.unique())
        merged = {k: np.concatenate([old[k][keep], (emb if k == "emb" else
                                     meta[k].to_numpy(dtype=bool if k.startswith("in_d1") or k == "rep" else str))])
                  for k in old.files}
        np.savez_compressed(res / "embeddings.npz", **merged)
        print(meta.groupby(["dataset", "split"]).agg(crops=("cls", "size"), rep_crops=("rep", "sum")).to_string())
        print(f"added {len(meta)} embeddings to {res / 'embeddings.npz'} ({int(keep.sum())} kept from before)")
        return
    out = {k: meta[k].to_numpy(dtype=bool if k.startswith("in_d1") or k == "rep" else str) for k in meta.columns}
    np.savez_compressed(res / "embeddings.npz", emb=emb, **out)
    s = meta.groupby(["dataset", "split"]).agg(crops=("cls", "size"), in_d1=("in_d1", "sum"),
                                               in_d1_train=("in_d1_train", "sum"), rep_crops=("rep", "sum"),
                                               clusters=("group", "nunique"))
    print(s.to_string())

    held = np.flatnonzero(((meta.dataset == "D1") & meta.split.isin(["val", "test"])).to_numpy())
    if a.corrupt_n and len(held) > a.corrupt_n:
        held = np.sort(np.random.default_rng(a.seed).choice(held, a.corrupt_n, replace=False))
    E, K, S, R = [], [], [], []
    for kind, levels in SEVERITIES.items():
        for sev, lv in enumerate(levels, 1):
            print(f"  corruption {kind} severity {sev} ({len(held)} crops)", flush=True)
            E.append(extract(model, meta.crop_path.iloc[held], device, a.batch, a.workers, Corrupt(kind, lv)))
            K += [kind] * len(held)
            S += [sev] * len(held)
            R.append(held)
    np.savez_compressed(work / "embeddings_corrupt.npz", emb=np.concatenate(E), corruption=np.array(K),
                        severity=np.array(S), row=np.concatenate(R))
    print("saved", res / "embeddings.npz", "and", work / "embeddings_corrupt.npz")


if __name__ == "__main__":
    main()
