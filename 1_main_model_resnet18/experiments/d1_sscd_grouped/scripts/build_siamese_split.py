"""Siamese split: link images whose Siamese cosine is >= THRESH, split whole components 70/15/15.

Uses only the Siamese embeddings (data/siamese/embeddings.npy from siamese_audit.py audit).
Writes manifests/siamese_split.csv. Then evaluates it against the SSCD and d1_grouped groupings.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_union_split as bu  # noqa: E402
import compare_d1_grouped as cmp  # noqa: E402
import siamese_audit as sa  # noqa: E402

THRESH = 0.90


def main():
    df = cmp.load()
    z = np.load(sa.WORK / "embeddings.npy")
    n = len(z)
    S = z @ z.T
    np.fill_diagonal(S, -1)
    ii, jj = np.nonzero(S >= THRESH)
    _, lab = connected_components(csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(n, n)), directed=False)
    df["siamese_group"] = lab

    crops = pd.read_csv(bu.HERE.parent / "data" / "d1_sscd_grouped" / "work" / "crops" / "D1.csv")
    crops["key"] = crops["image"].str.extract(r"images/(.+)$")[0]
    crops = crops.merge(df[["key", "siamese_group"]], on="key", validate="m:1")
    ct = pd.crosstab(crops.siamese_group, crops.cls).reindex(range(lab.max() + 1), fill_value=0)
    assign = bu.allocate(ct.values.astype(float), np.random.default_rng(bu.SEED))
    comp = pd.Series(bu.SPLITS[assign], index=ct.index)
    df["siamese_split"] = df.siamese_group.map(comp)
    df[["key", "siamese_group", "siamese_split", "sscd_split", "dg_split", "original_split"]].to_csv(
        bu.MAN / "siamese_split.csv", index=False)

    print("threshold", THRESH, "groups", df.siamese_group.nunique(), "largest", df.siamese_group.value_counts().max())
    print(df.siamese_split.value_counts().to_dict(), crops.siamese_group.map(comp).value_counts().to_dict())
    print((pd.crosstab(crops.cls, crops.siamese_group.map(comp), normalize="index")[list(bu.SPLITS)] * 100).round(1))
    for g in ["sscd_group", "sscd_block", "dg_group"]:
        bad = df.groupby(g).siamese_split.nunique() > 1
        print(f"{g}: {int(bad.sum())} groups straddle the Siamese split "
              f"({int(df[df[g].isin(bad[bad].index)].shape[0])} images)")


if __name__ == "__main__":
    main()
