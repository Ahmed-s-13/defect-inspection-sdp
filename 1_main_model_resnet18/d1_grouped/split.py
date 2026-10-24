"""Group D1 copies into source-photo clusters, make a leak-free 70/15/15 split, check it, and write it as a YOLO set.

Needs cache/pairs.csv from group.py. Steps:
  1. a candidate pair is the same source (the same photo augmented, or another shot of the same damage) when
     SIFT+RANSAC finds >= 8 affine inliers, or >= 6 with ResNet-50 cosine >= 0.75, or cosine >= 0.85, or
     phash <= 2 (cosine and phash over the 8 flips/rotations). Thresholds set by eye on sheets/: phash 3-8 alone
     gives false matches (the black corners of rotated copies dominate the hash) and chains everything into
     one group; inliers 6-12 and cosine 0.85-0.95 are almost all the same damage, often a burst shot SIFT barely
     matches, but 6-7 inliers at cosine < 0.75 also link unrelated airliner photos; cosine 0.80-0.82 alone
     starts chaining unrelated panels into 1000+ image groups
  2. union-find the matches into groups; StratifiedGroupKFold (20 folds, stratified on each group's dominant
     class: 3 folds test, 3 val, 14 train), then a local search that moves/swaps whole groups toward 70/15/15
     for every class's boxes and for images. No group crosses splits.
  3. leak check: every val/test image against every image in the other splits (full cosine and phash search over
     the 8 flips/rotations, SIFT on its --check-k nearest, more than group.py looked at). Matches it finds are
     added to the groups (and kept in cache/extra_pairs.csv) and steps 2-3 repeat until it finds
     none; sheets/leak_check_*.jpg shows the closest cross-split pairs of the final split for checking by eye
  4. write ../data/d1_grouped/dataset/ (images copied, not linked, so it stands apart from D1; labels as boxes in
     the model's alphabetical class order) with data.yaml for YOLO, plus groups.csv and report.md here.
     groups.csv is the split itself: enhanced_res reads it, so it is kept in git.

Nothing in the D1 folder is written to. Needs group.py's caches. Run from this folder:
  python split.py               # ~10-30 min: a few group -> split -> leak-check rounds
"""
import argparse
import shutil
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import group as G

HERE = G.HERE
OUT = G.MAIN / "data/d1_grouped/dataset"
D1_NAMES = ["crack", "dent", "corrosion", "scratch", "missing-head", "paint-peel-off"]
NAMES = sorted(D1_NAMES)                     # the model's class order: corrosion, crack, dent, missing-head, ...
TO_MODEL = {i: NAMES.index(n) for i, n in enumerate(D1_NAMES)}
SPLITS = ["train", "val", "test"]


def read_labels(key):
    """D1 label file -> list of (model class, xc, yc, w, h). Polygon rows become their bounding box, as
    Ultralytics does when it trains a detector on segment labels."""
    rows = []
    for line in (G.D1 / "labels" / key.replace(".jpg", ".txt")).read_text().split("\n"):
        v = line.split()
        if not v:
            continue
        c, xy = int(v[0]), np.array(v[1:], float)
        if len(xy) == 4:
            box = xy
        else:
            x, y = np.clip(xy[0::2], 0, 1), np.clip(xy[1::2], 0, 1)
            box = [(x.min() + x.max()) / 2, (y.min() + y.max()) / 2, x.max() - x.min(), y.max() - y.min()]
        rows.append((TO_MODEL[c], *box))
    return rows


def union_find(n, edges):
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)])
    _, gid = np.unique(roots, return_inverse=True)
    return gid


def split_groups(df, boxes, seeds, steps=40000, init=None):
    """StratifiedGroupKFold over groups (20 folds: 3 test, 3 val, 14 train) for a start, then a local search that
    moves single groups or swaps two between splits while it lowers the worst gap between any class's box share
    (or the image share) and 70/15/15. Keeps the best over the seeds. With init (each image's split from the
    previous round) it warm-starts instead: every group goes to the split holding most of its images, so later
    rounds only move what they must and the leak check is not faced with a new set of split boundaries."""
    target = np.array([0.70, 0.15, 0.15])
    gid = df.group.values
    ng = gid.max() + 1
    gcnt = np.zeros((ng, boxes.shape[1] + 1))                     # per group: boxes of each class, images
    np.add.at(gcnt, gid, np.c_[boxes, np.ones(len(df))])
    total = gcnt.sum(0)
    gcls = df.groupby("group").dom.agg(lambda s: Counter(s).most_common(1)[0][0]).values

    def cost(tot):                                                # tot: 3 splits x (classes + images)
        err = np.abs(tot / total - target[:, None])
        return err.max() + 0.1 * err.mean()

    best = None
    for seed in ([0] if init is not None else seeds):
        rng = np.random.default_rng(seed)
        sp = np.zeros(ng, int)
        if init is not None:
            votes = np.zeros((ng, 3))
            np.add.at(votes, (gid, pd.Series(init).map({s: k for k, s in enumerate(SPLITS)}).values), 1)
            sp = votes.argmax(1)
        else:
            folds = StratifiedGroupKFold(20, shuffle=True, random_state=seed).split(np.zeros(ng), gcls,
                                                                                    np.arange(ng))
            for f, (_, te) in enumerate(folds):
                sp[te] = 2 if f < 3 else 1 if f < 6 else 0
        tot = np.stack([gcnt[sp == k].sum(0) for k in range(3)])
        c = cost(tot)
        for _ in range(steps):
            g = rng.integers(ng)
            if rng.random() < 0.5:                                    # move g to another split
                to = (sp[g] + rng.integers(1, 3)) % 3
                new = tot.copy()
                new[sp[g]] -= gcnt[g]
                new[to] += gcnt[g]
                nc = cost(new)
                if nc < c:
                    tot, c, sp[g] = new, nc, to
            else:                                                     # swap g with a group in another split
                h = rng.integers(ng)
                if sp[g] == sp[h]:
                    continue
                new = tot.copy()
                new[sp[g]] += gcnt[h] - gcnt[g]
                new[sp[h]] += gcnt[g] - gcnt[h]
                nc = cost(new)
                if nc < c:
                    tot, c = new, nc
                    sp[g], sp[h] = sp[h], sp[g]
        if best is None or c < best[0]:
            best = (c, seed, sp.copy(), np.abs(tot / total - target[:, None]).max())
    print(f"split seed {best[1]}: worst share gap {best[3]:.3f}", flush=True)
    return np.array(SPLITS)[best[2][gid]], best[1]


def is_match(inliers, cos, phash, a):
    """The same-source rule (see the module docstring)."""
    return ((inliers >= a.min_inliers) | ((inliers >= a.weak_inliers) & (cos >= a.weak_cos)) |
            (cos >= a.min_cos) | (phash <= 2))


def cos_phash(ii, jj, e, h):
    """Cosine and phash distance of pairs (ii[n], jj[n]), each over the 8 flips/rotations of either image."""
    import torch
    ii, jj = np.asarray(ii, int), np.asarray(jj, int)
    cos = np.empty(len(ii), np.float32)
    for s in range(0, len(ii), 20000):
        x, y = e[ii[s:s + 20000]], e[jj[s:s + 20000]]
        cos[s:s + 20000] = torch.maximum(torch.einsum("nod,nd->no", x, y[:, 0]).max(1).values,
                                         torch.einsum("nod,nd->no", y, x[:, 0]).max(1).values).numpy()
    ph = np.minimum(G.popcount(h[ii] ^ h[jj, 0][:, None]).min(1), G.popcount(h[jj] ^ h[ii, 0][:, None]).min(1))
    return cos, ph.astype(int)


def load_feats():
    import torch
    e = torch.from_numpy(np.array(np.load(G.CACHE / "emb8.npy", mmap_mode="r"))).float()
    return e, np.load(G.CACHE / "phash8.npy")


def pair_scores(pairs, e, h):
    """cos / phash / SIFT inliers of (i, j) pairs, as in cache/pairs.csv."""
    df = pd.DataFrame(sorted(pairs), columns=["i", "j"])
    df["cos"], df["phash"] = cos_phash(df.i, df.j, e, h)
    with ProcessPoolExecutor(initializer=G._winit) as ex:
        df["inliers"] = [r[1] for r in ex.map(G.verify, zip(df.i, df.j), chunksize=64)]
    return df


def check_leaks(df, k, a, e, h):
    """Every val image against train+test, every test image against train+val: its k nearest other-split images
    by cosine (over the 8 flips/rotations, full search) plus its phash nearest, each scored by cosine, phash and
    SIFT inliers. Returns one row per val/test image (best scores and whether any pair meets the rule) and the
    pairs that meet the rule."""
    import torch
    sp = df.new_split.values
    jobs = []
    for s in ["val", "test"]:
        q, other = np.flatnonzero(sp == s), np.flatnonzero(sp != s)
        db = e[other, 0]
        for b in range(0, len(q), 512):
            qi = q[b:b + 512]
            sim = torch.stack([e[qi, o] @ db.T for o in range(8)]).max(0).values
            j = sim.topk(k, dim=1).indices.numpy()
            d = G.popcount(h[qi, :, None] ^ h[other, 0][None, None, :]).min(1)
            for r, i in enumerate(qi):
                jobs += [(int(i), int(jj)) for jj in set(other[j[r]].tolist()) | {int(other[d[r].argmin()])}]
    jb = pd.DataFrame(jobs, columns=["i", "j"])
    jb["cos"], jb["phash"] = cos_phash(jb.i, jb.j, e, h)
    with ProcessPoolExecutor(initializer=G._winit) as ex:
        jb["inliers"] = [r[1] for r in ex.map(G.verify, jobs, chunksize=256)]
    jb["match"] = is_match(jb.inliers, jb.cos, jb.phash, a)
    keys = df.key.values
    out = pd.DataFrame({"i": jb.i.unique()}).set_index("i")
    for col, asc in [("cos", False), ("inliers", False), ("phash", True)]:
        top = jb.sort_values(col, ascending=asc).groupby("i").head(1).set_index("i")
        out[col + "_best"], out[col + "_match"] = top[col], top.j
    out["flagged"] = jb.groupby("i").match.any()
    out["key"] = keys[out.index]
    out["split"] = sp[out.index]
    for c in ["cos_match", "phash_match", "inliers_match"]:
        out[c + "_split"] = sp[out[c].values]
        out[c] = keys[out[c].values]
    return out.reset_index(drop=True), set(zip(jb.i[jb.match].tolist(), jb.j[jb.match].tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-inliers", type=int, default=8)
    ap.add_argument("--weak-inliers", type=int, default=6)
    ap.add_argument("--weak-cos", type=float, default=0.75)
    ap.add_argument("--min-cos", type=float, default=0.85)
    ap.add_argument("--check-k", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--max-rounds", type=int, default=15)
    a = ap.parse_args()

    df = G.index()
    pairs = pd.read_csv(G.CACHE / "pairs.csv")
    match = is_match(pairs.inliers, pairs.cos, pairs.phash, a)
    print(f"{int(match.sum())} matched pairs of {len(pairs)} candidates")
    edges = set(zip(pairs.i[match].tolist(), pairs.j[match].tolist()))
    # matches earlier runs' leak checks found beyond the candidate pairs (only those that pass the current rule)
    extra_f = G.CACHE / "extra_pairs.csv"
    extra = pd.read_csv(extra_f) if extra_f.exists() else pd.DataFrame(columns=["i", "j", "cos", "phash", "inliers"])
    ok = is_match(extra.inliers, extra.cos, extra.phash, a)
    edges |= set(zip(extra.i[ok].astype(int).tolist(), extra.j[ok].astype(int).tolist()))
    print(f"{int(ok.sum())} extra matched pairs from earlier leak checks", flush=True)

    labels = [read_labels(k) for k in df.key]
    boxes = np.zeros((len(df), len(NAMES)), int)
    for n, rows in enumerate(labels):
        for r in rows:
            boxes[n, r[0]] += 1
    df["n_boxes"] = boxes.sum(1)
    df["dom"] = np.where(df.n_boxes > 0, boxes.argmax(1), len(NAMES))       # len(NAMES) = background
    e, h = load_feats()

    # group -> split -> leak check; pairs the check finds across splits are copies the top-k candidate search
    # missed (big groups fill an image's neighbour list with its own copies), so add them and go again
    n_found = []
    for rnd in range(a.max_rounds):
        df["group"] = union_find(len(df), edges)
        df["new_split"], s = split_groups(df, boxes, range(a.seeds), init=df.new_split.values if rnd else None)
        seed = s if rnd == 0 else seed
        assert df.groupby("group").new_split.nunique().max() == 1
        leaks, found = check_leaks(df, a.check_k, a, e, h)
        found = {(min(i, j), max(i, j)) for i, j in found} - edges
        print(f"round {rnd}: {df.group.nunique()} groups, {len(found)} cross-split matches found", flush=True)
        n_found.append(len(found))
        if not found:
            break
        edges |= found
        extra = pd.concat([extra, pair_scores(found, e, h)]).drop_duplicates(["i", "j"])
        extra.to_csv(extra_f, index=False)
    df.drop(columns="dom").assign(dominant=[(NAMES + ["background"])[d] for d in df.dom]) \
      .to_csv(HERE / "groups.csv", index=False)
    leaks.to_csv(HERE / "leak_check.csv", index=False)
    flag = leaks.flagged
    sheets = HERE / "sheets"
    sheets.mkdir(exist_ok=True)
    for col, asc, name in [("inliers_best", False, "inliers"), ("cos_best", False, "cos"),
                           ("phash_best", True, "phash")]:
        top = leaks.sort_values(col, ascending=asc).head(24)
        m = col.replace("_best", "_match")
        G.grid([G.pair_tile(r.key, getattr(r, m), f"{r.split} | {getattr(r, m + '_split')}  inl {r.inliers_best} "
                                                  f"cos {r.cos_best:.2f} ph {r.phash_best}", False)
                for r in top.itertuples()], sheets / f"leak_check_top_{name}.jpg")
    smp = leaks.sample(24, random_state=0)
    G.grid([G.pair_tile(r.key, r.cos_match, f"{r.split} | nearest {r.cos_match_split}  cos {r.cos_best:.2f} "
                                            f"inl {r.inliers_best}", False) for r in smp.itertuples()],
           sheets / "leak_check_random.jpg")

    # write the dataset
    if OUT.exists():
        shutil.rmtree(OUT)
    for s in SPLITS:
        (OUT / "images" / s).mkdir(parents=True)
        (OUT / "labels" / s).mkdir(parents=True)
    for key, s, rows in zip(df.key, df.new_split, labels):
        name = Path(key).name
        shutil.copy2(G.D1 / "images" / key, OUT / "images" / s / name)
        (OUT / "labels" / s / name.replace(".jpg", ".txt")).write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in rows))
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(NAMES))
    (OUT / "data.yaml").write_text("# D1 regrouped so no source photo spans two splits (see 1_main_model_resnet18/d1_grouped/report.md).\n"
                                   "# No 'path': Ultralytics uses this file's folder as the dataset root.\n"
                                   f"train: images/train\nval: images/val\ntest: images/test\nnames:\n{names}\n")

    # report
    gs = df.groupby("group").size()
    split_tab = df.groupby("new_split").agg(images=("key", "size"), groups=("group", "nunique"),
                                            boxes=("n_boxes", "sum")).reindex(SPLITS)
    cls = pd.DataFrame(boxes, columns=NAMES).groupby(df.new_split.values).sum().reindex(SPLITS).T
    cls_pct = (cls.div(cls.sum(1), axis=0) * 100).round(1)
    origin = pd.crosstab(df.split, df.new_split).reindex(index=SPLITS, columns=SPLITS)
    md = [f"# D1 regrouped split (leak-free)\n",
          f"{len(df)} images (all of D1 train+val+test) fall into **{len(gs)} source groups**. "
          f"A pair counts as the same source (one photo augmented, or another shot of the same damage) when "
          f"SIFT+RANSAC finds >= {a.min_inliers} affine inliers, or >= {a.weak_inliers} with ResNet-50 cosine >= "
          f"{a.weak_cos}, or cosine >= {a.min_cos}, or phash <= 2 (cosine and phash over the 8 flips/rotations; "
          f"{int(match.sum())} of {len(pairs)} "
          f"candidate pairs). Groups are split 70/15/15: StratifiedGroupKFold on the dominant class, then a local search over "
          f"whole groups toward 70/15/15 per class (seed {seed}); no group spans two splits.\n",
          f"Group sizes: {(gs == 1).sum()} singletons, median {int(gs.median())}, mean {gs.mean():.2f}, "
          f"max {gs.max()}; size histogram {dict(Counter(np.minimum(gs, 10)).most_common())} (10 = 10+).\n",
          "## Splits\n", split_tab.to_markdown(), "\n",
          "## Boxes per class per split\n", cls.to_markdown(), "\n", "Share of each class's boxes (%):\n",
          cls_pct.to_markdown(), "\n",
          "## Where the images came from (rows: D1's split, columns: new split)\n", origin.to_markdown(), "\n",
          "## Leak check\n",
          f"Every val image against every train+test image, and every test image against every train+val image: "
          f"full cosine and phash search over the 8 flips/rotations, and SIFT+RANSAC on its {a.check_k} nearest "
          f"other-split images by cosine plus its phash nearest. Matches this check found were added to the "
          f"groups and the split redone until it found none (new matches per round: {n_found}).\n",
          f"- flagged as a copy by the grouping rule in the final split: **{int(flag.sum())} / {len(leaks)}**\n"
          f"- best scores: inliers >= {a.min_inliers}: {int((leaks.inliers_best >= a.min_inliers).sum())}, "
          f"phash <= 2: {int((leaks.phash_best <= 2).sum())}, cosine >= {a.min_cos}: "
          f"{int((leaks.cos_best >= a.min_cos).sum())}\n"
          f"- best other-split cosine: "
          + ", ".join(f">={t}: {int((leaks.cos_best >= t).sum())}" for t in [0.9, 0.85, 0.8, 0.75]) + "\n"
          f"- best other-split inliers: "
          + ", ".join(f">={t}: {int((leaks.inliers_best >= t).sum())}" for t in [12, 8, 6, 5, 4]) + "\n",
          "Sheets (val/test image | its closest image in another split): sheets/leak_check_top_{inliers,cos,phash}.jpg "
          "(the 24 closest pairs by each score) and sheets/leak_check_random.jpg (24 random val/test images).\n",
          "## Dataset\n",
          "data/d1_grouped/dataset/ holds copied images (not links into D1) and labels as boxes (polygon rows converted to their "
          "bounding box) in the model's class order " + ", ".join(f"{i} {n}" for i, n in enumerate(NAMES)) +
          ". Its data.yaml has no `path`, so it works in place or zipped for upload.\n"]
    (HERE / "report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
