"""Audit a prepared dataset before training anything on it.

    python data_audit.py --data data/aircraft        -> data/aircraft/audit.md

Answers two questions with no neural network involved:
  * Sampling: how uneven are the classes, and does each photo source contain
    every class or is class tied to where the photo came from?
  * Shortcuts: how well can the class be guessed WITHOUT looking at the defect?
    Each "floor" below is a logistic regression on a deliberately dumb input.
    A high floor means the dataset offers that shortcut, and any model trained
    on it should be assumed to take it until the shortcut tests say otherwise.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import common as C


def floor(Xtr, ytr, Xte, yte, model=None):
    model = model or make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    return float(f1_score(yte, model.fit(Xtr, ytr).predict(Xte), average="macro", zero_division=0))


def features(df, data_dir):
    colour, context, pixels = [], [], []
    for path, box in zip(df["path"], df[["bx0", "by0", "bx1", "by1"]].to_numpy(dtype=float)):
        a = np.asarray(Image.open(Path(data_dir) / path).convert("RGB").resize((64, 64)), dtype=np.float32) / 255
        colour.append(np.r_[a.mean((0, 1)), a.std((0, 1))])
        outside = np.ones((64, 64), bool)
        if not np.isnan(box).any():
            x0, y0, x1, y1 = np.round(box * 64).astype(int)
            outside[y0:y1, x0:x1] = False
        ctx = a[outside] if outside.sum() > 10 else a.reshape(-1, 3)
        context.append(np.r_[ctx.mean(0), ctx.std(0)])
        pixels.append(np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((24, 24)), dtype=np.float32).ravel() / 255)
    geom = np.c_[df["bx1"] - df["bx0"], df["by1"] - df["by0"]]
    geom = np.nan_to_num(np.c_[geom, geom[:, 0] * geom[:, 1], geom[:, 0] / np.clip(geom[:, 1], 1e-3, None)])
    return np.array(colour), np.array(context), np.array(pixels), geom


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    df, classes = C.load_metadata(a.data)
    df = df[df.split != "ood"].reset_index(drop=True)
    tr, te = (df.split == "train").to_numpy(), (df.split == "test").to_numpy()
    y = df["y"].to_numpy()
    md = [f"# Data audit: {Path(a.data).name}", "", "## Sampling", ""]

    counts = pd.crosstab(df["label"], df["split"]).reindex(classes)[["train", "val", "test"]]
    counts["total"] = counts.sum(axis=1)
    counts["share"] = (counts["total"] / counts["total"].sum()).map("{:.1%}".format)
    md += [counts.to_markdown(), "",
           f"Largest class is {counts['total'].max() / counts['total'].min():.1f}x the smallest. "
           f"{df['group'].nunique()} independent photos; the busiest contributes {df.groupby('group').size().max()} crops.", ""]

    by_src = pd.crosstab(df["label"], df["source"]).reindex(classes)
    md += ["Class by photo source (source is guessed from the filename prefix):", "", by_src.to_markdown(), ""]
    if by_src.shape[1] > 1:
        chi2 = chi2_contingency(by_src.to_numpy())[0]
        v = float(np.sqrt(chi2 / (by_src.to_numpy().sum() * (min(by_src.shape) - 1))))
        md += [f"Association between class and source (Cramer's V): **{v:.2f}** "
               "(0 = every source has the same class mix; 1 = source tells you the class). "
               "The closer to 1, the more a model can score by recognising the source instead of the defect.", ""]

    colour, context, pixels, geom = features(df, a.data)
    src1h = OneHotEncoder(handle_unknown="ignore").fit_transform(df[["source"]]).toarray()
    rows = [
        ("random guessing", floor(y[tr, None], y[tr], y[te, None], y[te], DummyClassifier(strategy="uniform", random_state=0))),
        ("photo source only", floor(src1h[tr], y[tr], src1h[te], y[te])),
        ("colour of the surroundings only (defect excluded)", floor(context[tr], y[tr], context[te], y[te])),
        ("average colour of the whole crop", floor(colour[tr], y[tr], colour[te], y[te])),
        ("shape of the annotation box only", floor(geom[tr], y[tr], geom[te], y[te])),
        ("PCA of 24x24 raw pixels, 40 components", floor(pixels[tr], y[tr], pixels[te], y[te], make_pipeline(
            StandardScaler(), PCA(min(40, int(tr.sum()) - 1)), LogisticRegression(max_iter=3000, class_weight="balanced")))),
    ]
    md += ["## Shortcut floors", "", "Test macro-F1 of a linear model that sees only the named input:", "",
           "| input | test macro-F1 |", "|---|---|"] + [f"| {n} | {v:.3f} |" for n, v in rows] + ["",
           "Read a trained network's score against these, not against zero. If \"surroundings only\" or \"photo source only\" "
           "is already high, a high network score proves little until `evaluate.py` shows it survives hiding the context."]
    out = Path(a.data) / "audit.md"
    out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
