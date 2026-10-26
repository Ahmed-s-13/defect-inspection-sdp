"""Union split: link images that share an SSCD split block OR a d1_grouped group, then
split the connected components 70/15/15 with every class's crop counts balanced.

Writes manifests/union_split.csv (one row per image) and manifests/union_split.portable.csv
(one row per crop, same columns as d1_split.portable.csv).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import compare_d1_grouped as cmp  # noqa: E402

MAN = HERE.parent / "manifests"
SPLITS = np.array(["train", "val", "test"])
TARGET = np.array([0.70, 0.15, 0.15])
SEED = 42


def components(df):
    n = len(df)
    lab = np.concatenate([df.sscd_block.values, df.dg_group.values + df.sscd_block.max() + 1])
    A = coo_matrix((np.ones(2 * n), (np.tile(np.arange(n), 2), lab)), shape=(n, lab.max() + 1)).tocsr()
    return connected_components(A @ A.T, directed=False)[1]


def allocate(counts, rng, iters=200000):
    """counts: components x classes crop counts. Returns split index per component."""
    tot = counts.sum(0)
    want = TARGET[:, None] * tot[None, :]                    # 3 x classes
    scale = np.maximum(want, 1)
    cw = 1.0 / np.maximum(tot, 1) ** 0.0
    assign = np.zeros(len(counts), int)
    cur = np.zeros_like(want)
    for i in np.argsort(-counts.sum(1)):                      # greedy, largest first
        gain = [(((cur[s] + counts[i] - want[s]) / scale[s]) ** 2).sum() for s in range(3)]
        s = int(np.argmin(gain)); assign[i] = s; cur[s] += counts[i]
    obj = lambda c: float((((c - want) / scale) ** 2).sum())
    best = obj(cur)
    for _ in range(iters):                                    # local search: move or swap
        i = rng.integers(len(counts)); s = rng.integers(3)
        if s == assign[i]:
            continue
        new = cur.copy(); new[assign[i]] -= counts[i]; new[s] += counts[i]; mv = [(i, s)]
        if rng.random() < 0.5:
            js = np.flatnonzero(assign == s)
            if len(js):
                j = rng.choice(js); new[s] -= counts[j]; new[assign[i]] += counts[j]; mv.append((j, assign[i]))
        o = obj(new)
        if o < best:
            old = [(k, assign[k]) for k, _ in mv]
            for k, t in mv:
                assign[k] = t
            cur, best = new, o
    return assign


def main():
    df = cmp.load()
    df["union_group"] = components(df)
    crops = pd.read_csv(HERE.parent / "data" / "d1_sscd_grouped" / "work" / "crops" / "D1.csv")
    crops["key"] = crops["image"].str.extract(r"images/(.+)$")[0]
    crops = crops.merge(df[["key", "union_group"]], on="key", validate="m:1")
    ct = pd.crosstab(crops.union_group, crops.cls).reindex(range(df.union_group.max() + 1), fill_value=0)
    assign = allocate(ct.values.astype(float), np.random.default_rng(SEED))
    comp_split = pd.Series(SPLITS[assign], index=ct.index)
    df["union_split"] = df.union_group.map(comp_split)
    crops["split"] = crops.union_group.map(comp_split)
    assert df.groupby("union_group").union_split.nunique().max() == 1

    df[["key", "path", "original_split", "sscd_group", "sscd_block", "dg_group", "union_group",
        "sscd_split", "dg_split", "union_split"]].to_csv(MAN / "union_split.csv", index=False)
    port = pd.read_csv(MAN / "d1_split.portable.csv")
    port["key"] = port["image"].str.extract(r"images/(.+)$")[0]
    port = port.drop(columns=["split", "group_id"]).merge(
        crops[["crop_path", "union_group", "split"]].assign(
            crop_path=lambda d: "data/d1_sscd_grouped/work/crops/D1/" + d.crop_path.str.split("/").str[-1]),
        on="crop_path", validate="1:1").rename(columns={"union_group": "group_id"})
    port.drop(columns="key")[["crop_path", "cls", "image", "group_id", "split", "image_id", "source_group"]] \
        .to_csv(MAN / "union_split.portable.csv", index=False)

    shares = pd.crosstab(crops.cls, crops.split, normalize="index").reindex(columns=SPLITS)
    print("components", df.union_group.nunique(), "largest", df.union_group.value_counts().max())
    print(df.union_split.value_counts().to_dict(), crops.split.value_counts().to_dict())
    print((shares * 100).round(1))


if __name__ == "__main__":
    main()
