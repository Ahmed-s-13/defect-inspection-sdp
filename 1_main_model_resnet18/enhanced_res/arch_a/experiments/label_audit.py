"""Label-audit sheets: crack/dent crops the model gets wrong with high confidence, for a human to re-check.

From 1_main_model_resnet18/:
  python enhanced_res/arch_a/experiments/label_audit.py            # confidence >= 0.90 (default)
  python enhanced_res/arch_a/experiments/label_audit.py --min-conf 0.8

Reads the cached test probabilities from arch0_cp.py (no model run). Selects every crop in d1_grouped's test split
(cal and test halves) whose given label is crack and prediction dent, or the reverse, with top probability >= --min-conf.
Writes enhanced_res/arch_a/results/label_audit/:
  review.csv              one row per flagged crop, with empty reviewer_label / reviewer_notes columns to fill in
  family_<id>.jpg         per photo family: the source photos (flagged boxes red, other boxes grey) and the flagged crops
  reference.jpg           confidently *correct* crack and dent crops from the same split, to compare against
  README.md               counts per family and how to review
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import arch0_cp as A
from model import MAIN

OUT = HERE.parent / "results" / "label_audit"
RAW = MAIN / "data/raw/D1/content/Aircraft_dataset/images"
PAIR = ("crack", "dent")


def load():
    z = np.load(A.CACHE / "probs_test.npz", allow_pickle=True)
    P, cls = z["p"], list(z["classes"])
    sp = pd.read_csv(HERE.parent / "cal_split.csv")
    sp["crop_path"] = [str(MAIN / p) for p in sp.crop_path]
    d = pd.DataFrame({"crop_path": list(z["crop_path"])}).merge(sp, on="crop_path", how="left", validate="1:1")
    meta = pd.read_csv(MAIN / "data/work/crops/D1.csv")
    meta["crop_path"] = [str(MAIN / p) for p in meta.crop_path]
    d = d.merge(meta[["crop_path", "x1", "y1", "x2", "y2"]], on="crop_path", how="left", validate="1:1")
    y = d.cls.map({c: i for i, c in enumerate(cls)}).to_numpy()
    cal = (d.split == "cal").to_numpy()
    q = {a: A.lac_threshold(P[cal], y[cal], a) for a in (0.05, 0.10)}
    d["pred"] = [cls[i] for i in P.argmax(1)]
    d["conf"] = P.max(1)
    d["p_crack"], d["p_dent"] = P[:, cls.index("crack")], P[:, cls.index("dent")]
    d["p_given"] = P[np.arange(len(d)), y]
    for a, qa in q.items():
        d[f"given_in_cp_set_a{int(a * 100):02d}"] = d.p_given >= 1 - qa
    d["box"] = d.crop_path.str.extract(r"__(\d+)\.jpg$")[0].astype(int)
    return d


def font(size):
    return ImageFont.load_default(size=size)


def family_sheet(f, allboxes, path):
    """Top: source photos with boxes. Bottom: flagged crops with their review ids."""
    W, T, cols = 300, 150, 8
    photos = sorted(f.key.unique())
    pc = 4
    prow = (len(photos) + pc - 1) // pc
    crow = (len(f) + cols - 1) // cols
    head = 40
    sheet = Image.new("RGB", (max(pc * W, cols * T), head + prow * (W + 20) + crow * (T + 44)), "white")
    dr = ImageDraw.Draw(sheet)
    fam = f.src_group.iloc[0]
    dr.text((6, 8), f"family {fam}: {len(f)} flagged crops in {len(photos)} photos  "
                    f"(given {f.cls.value_counts().to_dict()}, model says the other class)  red = flagged box, grey = other boxes",
            fill="black", font=font(15))
    for i, k in enumerate(photos):
        im = Image.open(RAW / k).convert("RGB")
        s = W / max(im.size)
        im = im.resize((int(im.width * s), int(im.height * s)))
        di = ImageDraw.Draw(im)
        flagged = set(f[f.key == k].crop_path)
        for _, r in allboxes[allboxes.key == k].iterrows():
            red = r.crop_path in flagged
            di.rectangle([r.x1 * s, r.y1 * s, r.x2 * s, r.y2 * s], outline=(220, 0, 0) if red else (150, 150, 150),
                         width=3 if red else 1)
            if red:
                rid = f.loc[f.crop_path == r.crop_path, "review_id"].iloc[0]
                di.text((r.x1 * s + 2, r.y1 * s + 1), str(rid), fill=(220, 0, 0), font=font(12))
        x, y = (i % pc) * W, head + (i // pc) * (W + 20)
        sheet.paste(im, (x, y + 18))
        dr.text((x + 4, y + 2), k, fill="black", font=font(12))
    y0 = head + prow * (W + 20)
    for i, (_, r) in enumerate(f.iterrows()):
        im = Image.open(r.crop_path).convert("RGB").resize((T, T))
        ImageDraw.Draw(im).rectangle([0, 0, T - 1, T - 1], outline=(220, 0, 0), width=3)
        x, y = (i % cols) * T, y0 + (i // cols) * (T + 44)
        sheet.paste(im, (x, y))
        dr.text((x + 3, y + T + 1), f"#{r.review_id}  given {r.cls}", fill="black", font=font(11))
        dr.text((x + 3, y + T + 14), f"model {r.pred} {r.conf:.2f}", fill=(220, 0, 0), font=font(11))
        dr.text((x + 3, y + T + 27), f"crack {r.p_crack:.3f} dent {r.p_dent:.3f}", fill="black", font=font(11))
    sheet.save(path, quality=88)


def reference_sheet(d, path, n=16):
    """Confidently correct crack and dent crops, so the reviewer sees what each label usually looks like."""
    T, cols = 150, 8
    sheet = Image.new("RGB", (cols * T, 2 * (30 + 2 * (T + 16))), "white")
    dr = ImageDraw.Draw(sheet)
    for j, c in enumerate(PAIR):
        ok = d[(d.cls == c) & (d.pred == c) & (d.conf >= 0.93)]
        ok = ok.groupby("src_group").head(2).sample(min(n, len(ok)), random_state=0)
        y0 = j * (30 + 2 * (T + 16))
        dr.text((6, y0 + 8), f"reference: given {c}, model agrees (confidence >= 0.93), {len(ok)} crops from "
                             f"{ok.src_group.nunique()} families", fill="black", font=font(14))
        for i, (_, r) in enumerate(ok.iterrows()):
            im = Image.open(r.crop_path).convert("RGB").resize((T, T))
            x, y = (i % cols) * T, y0 + 30 + (i // cols) * (T + 16)
            sheet.paste(im, (x, y))
            dr.text((x + 3, y + T + 1), f"fam {r.src_group}", fill="black", font=font(11))
    sheet.save(path, quality=88)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-conf", type=float, default=0.90)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("family_*.jpg"):
        old.unlink()

    d = load()
    flag = d[((d.cls == "crack") & (d.pred == "dent")) | ((d.cls == "dent") & (d.pred == "crack"))]
    flag = flag[flag.conf >= a.min_conf].copy()
    order = flag.src_group.value_counts().index
    flag["src_group"] = pd.Categorical(flag.src_group, order)
    flag = flag.sort_values(["src_group", "key", "box"]).reset_index(drop=True)
    flag["src_group"] = flag.src_group.astype(int)
    flag["review_id"] = np.arange(1, len(flag) + 1)

    cols = ["review_id", "src_group", "split", "key", "box", "cls", "pred", "conf", "p_crack", "p_dent",
            "given_in_cp_set_a05", "given_in_cp_set_a10", "crop_path"]
    rev = flag[cols].rename(columns={"src_group": "family", "key": "photo", "cls": "given_label",
                                     "pred": "model_label", "conf": "model_confidence"})
    rev["crop_path"] = [str(Path(p).relative_to(MAIN)) for p in rev.crop_path]
    rev["reviewer_label"] = ""     # fill in: crack / dent / both / neither / unsure
    rev["reviewer_notes"] = ""
    rev.round(4).to_csv(OUT / "review.csv", index=False)

    for fam, f in flag.groupby("src_group", sort=False):
        family_sheet(f, d[d.key.isin(f.key)], OUT / f"family_{fam}.jpg")
    reference_sheet(d, OUT / "reference.jpg")

    by = flag.groupby("src_group", sort=False).agg(
        crops=("review_id", "size"), photos=("key", "nunique"),
        given_crack=("cls", lambda s: (s == "crack").sum()), given_dent=("cls", lambda s: (s == "dent").sum()),
        median_conf=("conf", "median"), split=("split", "first"))
    md = ["# Label audit: high-confidence crack/dent disagreements", "",
          f"Crops in d1_grouped's test split (cal and test halves) where the given label is crack and enhanced_res "
          f"says dent, or the reverse, with model confidence >= {a.min_conf:.2f}: **{len(flag)} crops, "
          f"{flag.key.nunique()} photos, {flag.src_group.nunique()} photo families** "
          f"({(flag.cls == 'crack').sum()} given crack, {(flag.cls == 'dent').sum()} given dent).", "",
          "## How to review", "",
          "1. Look at `reference.jpg` first: what crack and dent usually look like when the model and label agree.",
          "2. Open each `family_<id>.jpg`. Top: the source photos, flagged boxes in red with their review id. "
          "Bottom: each flagged crop with the given label and the model's answer.",
          "3. In `review.csv`, fill `reviewer_label` with crack / dent / both / neither / unsure, and anything useful "
          "in `reviewer_notes`. Crops in one family are near-copies (augmented versions of one scene), so one "
          "decision usually covers the whole family.",
          "4. Ideally a GAES inspector makes the call: the question is what an inspector would write down.", "",
          "`given_in_cp_set_a05` / `a10`: whether architecture 0's CP set still contained the given label.", "",
          "## Families (most flagged first)", "",
          "| family | split | flagged crops | photos | given crack | given dent | median model confidence | sheet |",
          "|---:|---|---:|---:|---:|---:|---:|---|"]
    for fam, r in by.iterrows():
        md.append(f"| {fam} | {r.split} | {r.crops} | {r.photos} | {r.given_crack} | {r.given_dent} | "
                  f"{r.median_conf:.3f} | `family_{fam}.jpg` |")
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
