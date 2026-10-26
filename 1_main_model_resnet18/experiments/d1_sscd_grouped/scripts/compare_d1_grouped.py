"""Compare the SSCD grouped split (d1_sscd_grouped) with the alternative d1_grouped split.

Both cover the same 14,760 D1 images. Images are joined on (original D1 split, file name).
Writes tables, a Markdown report and figures to reports/compare_d1_grouped/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
OUT = HERE.parent / "reports" / "compare_d1_grouped"
OUT.mkdir(parents=True, exist_ok=True)
SPLITS = ["train", "val", "test"]


def md_table(d):
    d = d.reset_index()
    rows = ["| " + " | ".join(str(c) for c in d.columns) + " |", "|" + "---|" * len(d.columns)]
    rows += ["| " + " | ".join(f"{v:,}" if isinstance(v, (int, np.integer)) else str(v) for v in r) + " |" for r in d.values.tolist()]
    return "\n".join(rows)


def pairs(n):
    return n * (n - 1) // 2


def load():
    s = pd.read_csv(HERE.parent / "manifests" / "split_manifest.csv")
    s["key"] = s["path"].str.extract(r"images/(.+)$")[0]
    s = s.rename(columns={"split": "sscd_split", "group_id": "sscd_group", "split_block": "sscd_block"})
    g = pd.read_csv(EXP / "d1_grouped" / "groups.csv")
    g = g.rename(columns={"split": "g_original_split", "new_split": "dg_split", "group": "dg_group"})
    df = s.merge(g, on="key", validate="1:1")
    assert len(df) == len(s) == len(g) == 14760
    assert (df.original_split == df.g_original_split).all()
    return df


def contingency(a, b):
    return pd.crosstab(a, b)


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*keyword-only.*")


def pair_stats(a, b):
    ct = contingency(a, b).values
    n = ct.sum()
    both = sum(pairs(x) for x in ct.ravel())
    pa = sum(pairs(x) for x in ct.sum(1))
    pb = sum(pairs(x) for x in ct.sum(0))
    exp = pa * pb / pairs(n)
    ari = (both - exp) / (0.5 * (pa + pb) - exp)
    return dict(pairs_A=pa, pairs_B=pb, pairs_both=both, pair_jaccard=both / (pa + pb - both),
                frac_A_pairs_kept_in_B=both / pa, frac_B_pairs_kept_in_A=both / pb, ARI=ari)


def nesting(a, b, name_a, name_b):
    """For each A group: how many B groups it spans, and share of images in its largest B group."""
    ct = contingency(a, b)
    size = ct.sum(1)
    spans = (ct > 0).sum(1)
    top = ct.max(1) / size
    return pd.DataFrame({"size": size, "spans": spans, "top_share": top}).rename_axis(name_a)


def heat(ax, m, title, xl, yl, fmt="{:,}"):
    im = ax.imshow(m.values, cmap="Blues")
    ax.set_xticks(range(m.shape[1]), m.columns)
    ax.set_yticks(range(m.shape[0]), m.index)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_title(title, fontsize=10)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m.values[i, j]
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    color="white" if v > m.values.max() * 0.55 else "black", fontsize=9)


def main():
    df = load()
    md = ["# d1_sscd_grouped vs d1_grouped\n"]

    # ---- 1. split assignment ----
    sp = contingency(df["sscd_split"], df["dg_split"]).reindex(index=SPLITS, columns=SPLITS)
    sp.to_csv(OUT / "split_crosstab.csv")
    agree = np.trace(sp.values) / sp.values.sum()
    orig = SPLITS
    o_s = contingency(df["original_split"], df["sscd_split"]).reindex(index=orig, columns=SPLITS)
    o_g = contingency(df["original_split"], df["dg_split"]).reindex(index=orig, columns=SPLITS)
    # same-split intersections per split
    inter = {s: ((df.sscd_split == s) & (df.dg_split == s)).sum() for s in SPLITS}
    union = {s: ((df.sscd_split == s) | (df.dg_split == s)).sum() for s in SPLITS}
    md.append("## 1. Split assignment (images)\n")
    md.append(f"Same split in both: **{np.trace(sp.values):,} / {sp.values.sum():,} ({agree:.1%})**.\n")
    md.append("Rows: SSCD split, columns: d1_grouped split.\n\n" + md_table(sp) + "\n")
    md.append("Per-split overlap:\n\n| split | SSCD | d1_grouped | intersection | Jaccard |\n|---|---:|---:|---:|---:|")
    for s in SPLITS:
        md.append(f"| {s} | {sp.loc[s].sum():,} | {sp[s].sum():,} | {inter[s]:,} | {inter[s]/union[s]:.1%} |")
    md.append("")

    # ---- 2. groupings ----
    df["sscd_group"] = df["sscd_group"].astype(int)
    levels = {"SSCD strict group (algorithmic source family)": "sscd_group",
              "SSCD split block (what was actually kept together)": "sscd_block"}
    gs = {}
    md.append("## 2. Grouping structure\n")
    md.append("| grouping | images | groups | singletons | median size | mean | max |\n|---|---:|---:|---:|---:|---:|---:|")
    for name, col in [*levels.items(), ("d1_grouped group", "dg_group")]:
        sz = df.groupby(col).size()
        gs[col] = sz
        md.append(f"| {name} | {len(df):,} | {len(sz):,} | {(sz==1).sum():,} | {sz.median():.0f} | {sz.mean():.1f} | {sz.max():,} |")
    md.append("")

    md.append("## 3. Pair-level agreement\n")
    md.append("A pair of images is *co-grouped* if both are in the same group. 'Kept' = also co-grouped in the other grouping.\n")
    md.append("| A | B | pairs in A | pairs in B | in both | Jaccard | A pairs kept in B | B pairs kept in A | ARI |\n|---|---|---:|---:|---:|---:|---:|---:|---:|")
    stat_rows = []
    for name, col in levels.items():
        st = pair_stats(df[col], df["dg_group"])
        stat_rows.append({"A": col, "B": "dg_group", **st})
        md.append(f"| {col} | dg_group | {st['pairs_A']:,} | {st['pairs_B']:,} | {st['pairs_both']:,} | "
                  f"{st['pair_jaccard']:.1%} | {st['frac_A_pairs_kept_in_B']:.1%} | {st['frac_B_pairs_kept_in_A']:.1%} | {st['ARI']:.3f} |")
    pd.DataFrame(stat_rows).to_csv(OUT / "pair_agreement.csv", index=False)
    md.append("")

    # ---- 4. nesting ----
    md.append("## 4. Where one grouping's families sit inside the other\n")
    nest = {}
    for name, col in levels.items():
        for a, b, la, lb in [(col, "dg_group", col, "d1_grouped"), ("dg_group", col, "d1_grouped", col)]:
            n = nesting(df[a], df[b], la, lb)
            nest[(a, b)] = n
            n.to_csv(OUT / f"nesting_{a}_in_{b}.csv")
            w = n["size"]
            inside = (n["spans"] == 1)
            md.append(f"**{la} groups vs {lb}**: {len(n):,} groups; {inside.sum():,} ({inside.mean():.1%}) lie entirely inside one "
                      f"{lb} group, covering {w[inside].sum():,} images ({w[inside].sum()/w.sum():.1%}). "
                      f"Median groups spanned: {n['spans'].median():.0f}, max {n['spans'].max()}. "
                      f"Image-weighted mean share in the largest overlapping group: {(n['top_share']*w).sum()/w.sum():.1%}.\n")

    # ---- 5. leakage in terms of the other split ----
    md.append("## 5. Cross-checking leakage: one grouping judged by the other's splits\n")
    md.append("If grouping A's families were truly the same source, they should not straddle splits of B.\n")
    md.append("| A groups | split of | groups straddling >1 split | images in them | co-grouped pairs across splits |\n|---|---|---:|---:|---:|")
    for name, col in [*levels.items(), ("d1_grouped group", "dg_group")]:
        for sc, lab in [("sscd_split", "SSCD"), ("dg_split", "d1_grouped")]:
            ct = contingency(df[col], df[sc])
            span = (ct > 0).sum(1)
            bad = span > 1
            tot = sum(pairs(x) for x in ct.sum(1))
            within = sum(pairs(x) for x in ct.values.ravel())
            md.append(f"| {name} | {lab} | {bad.sum():,} | {ct.sum(1)[bad].sum():,} | {tot-within:,} |")
    md.append("")

    # ---- 6. per class ----
    cls = df.copy()
    md.append("## 6. Per-class view (dominant class of d1_grouped)\n")
    pc = cls.groupby("dominant").apply(lambda d: pd.Series({
        "images": len(d), "same_split_%": 100 * (d.sscd_split == d.dg_split).mean()}), include_groups=False)
    md.append(md_table(pc.round(1).astype({"images": int})) + "\n")

    # ---- figures ----
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    heat(ax[0], sp, "SSCD split (rows) vs d1_grouped split (cols)", "d1_grouped", "SSCD")
    heat(ax[1], o_s, "Original D1 split -> SSCD split", "SSCD", "original D1")
    heat(ax[2], o_g, "Original D1 split -> d1_grouped split", "d1_grouped", "original D1")
    fig.tight_layout(); fig.savefig(OUT / "fig1_split_crosstabs.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.logspace(0, np.log10(max(s.max() for s in gs.values())) + 0.1, 30)
    for col, lab in [("sscd_group", "SSCD strict group"), ("sscd_block", "SSCD split block"), ("dg_group", "d1_grouped group")]:
        ax[0].hist(gs[col], bins=bins, histtype="step", lw=2, label=f"{lab} (n={len(gs[col]):,})")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("images per group"); ax[0].set_ylabel("groups")
    ax[0].set_title("Group size distribution"); ax[0].legend(fontsize=8)
    for col, lab in [("sscd_group", "SSCD strict group"), ("sscd_block", "SSCD split block"), ("dg_group", "d1_grouped group")]:
        sz = np.sort(gs[col].values)[::-1]
        ax[1].plot(np.arange(1, len(sz) + 1), np.cumsum(sz) / sz.sum(), lw=2, label=lab)
    ax[1].set_xscale("log"); ax[1].set_xlabel("groups, largest first"); ax[1].set_ylabel("cumulative share of images")
    ax[1].set_title("How concentrated is the data in large groups"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig2_group_sizes.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for r, (col, lab) in enumerate([("sscd_group", "SSCD strict group"), ("sscd_block", "SSCD split block")]):
        n = nest[(col, "dg_group")]
        ax[r, 0].scatter(n["size"], n["spans"], s=10, alpha=.5)
        ax[r, 0].set_xscale("log"); ax[r, 0].set_yscale("log")
        ax[r, 0].set_xlabel(f"{lab} size (images)"); ax[r, 0].set_ylabel("d1_grouped groups it spans")
        ax[r, 0].set_title(f"{lab}: split across d1_grouped groups")
        m = nest[("dg_group", col)]
        ax[r, 1].scatter(m["size"], m["spans"], s=10, alpha=.5, color="tab:orange")
        ax[r, 1].set_xscale("log"); ax[r, 1].set_yscale("log")
        ax[r, 1].set_xlabel("d1_grouped group size (images)"); ax[r, 1].set_ylabel(f"{lab}s it spans")
        ax[r, 1].set_title(f"d1_grouped group: split across {lab}s")
    fig.tight_layout(); fig.savefig(OUT / "fig3_nesting_scatter.png", dpi=150); plt.close(fig)

    # 12 largest d1_grouped groups: how each decomposes into SSCD strict groups
    top = gs["dg_group"].nlargest(12).index[::-1]
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")
    for i, g in enumerate(top):
        parts = df[df.dg_group == g]["sscd_group"].value_counts()
        left = 0
        for k, (sid, c) in enumerate(parts.items()):
            if k < 8:
                ax.barh(i, c, left=left, color=cmap(k), edgecolor="white", lw=.5)
            else:
                ax.barh(i, parts.iloc[8:].sum(), left=left, color="lightgray", edgecolor="white", lw=.5)
                break
            left += c
        ax.text(parts.sum() + 5, i, f"{len(parts)} SSCD groups", va="center", fontsize=8)
    ax.set_yticks(range(len(top)), [f"d1_grouped {g}" for g in top])
    ax.set_xlabel("images (colours = SSCD strict groups, largest first; grey = the rest)")
    ax.set_title("The 12 largest d1_grouped groups, split into SSCD strict groups")
    fig.tight_layout(); fig.savefig(OUT / "fig4_largest_groups_overlap.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, col, lab in [(ax[0], "sscd_group", "SSCD strict group"), (ax[1], "dg_group", "d1_grouped group")]:
        n = nest[("sscd_group", "dg_group")] if col == "sscd_group" else nest[("dg_group", "sscd_group")]
        other = "d1_grouped" if col == "sscd_group" else "SSCD strict"
        counts = n["spans"].clip(upper=10).value_counts().sort_index()
        a.bar(counts.index.astype(str).str.replace("10", "10+"), counts.values)
        a.set_xlabel(f"number of {other} groups the {lab} touches"); a.set_ylabel("groups"); a.set_title(f"{lab}: how many {other} groups it spans")
    fig.tight_layout(); fig.savefig(OUT / "fig5_spans_hist.png", dpi=150); plt.close(fig)

    md.append("## Figures\n")
    for f, c in [("fig1_split_crosstabs.png", "Split cross-tabs"), ("fig2_group_sizes.png", "Group sizes"),
                 ("fig3_nesting_scatter.png", "Nesting scatter"), ("fig4_largest_groups_overlap.png", "Largest groups"),
                 ("fig5_spans_hist.png", "Groups spanned")]:
        md.append(f"![{c}]({f})\n")
    (OUT / "README.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
