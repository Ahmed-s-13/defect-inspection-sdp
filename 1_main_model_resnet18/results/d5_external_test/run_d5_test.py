"""Outside test: model 1 (the main ResNet-18 trained on D1, data/work/model_best.pt) on D5, with and without the MLP head.

D5 = Roboflow aircraft damage (classes Dent, Fastener Damage, Rupture). Mapping to D1's classes:
  Dent -> dent, Fastener Damage -> missing-head (fastener damage is a bit broader), Rupture -> no D1 class
  (only reported as "what does the model call it").

D5 shares many photos with D1 (some of them were in D1-train), so three sets are reported:
  A. strict unseen   1,040 crops of 278 D5 photos not in D1, from Experiments/enhanced_res/ext_d5.csv
                     (overlap check + phash over flips/rotations + SIFT matching). The fair outside-test number.
  B. all D5 (mapped) every Dent / Fastener Damage crop, split by the project's own overlap check into
                     "photo also in D1-train" and "not in D1-train". Shows how much the duplicates inflate the score.
  C. Rupture         prediction counts only.

Runs the real model on the crop images (Apple GPU / MPS). Writes next to this file:
  report.md, report.json, predictions_d5.csv, confusion_strict_unseen.png

  cd "Project Code/1_main_model_resnet18" && python3 results/d5_external_test/run_d5_test.py
"""
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
MAIN = OUT.parents[1]
sys.path.insert(0, str(MAIN))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluate import load_model
from klshift.common import CropDataset, embed_batch, pick_device
from klshift.mlp import MLPClassifier

CLASS = {"Dent": "dent", "Fastener Damage": "missing-head"}
STRICT = MAIN.parents[1] / "Experiments" / "enhanced_res" / "ext_d5.csv"


def main():
    device = pick_device()
    model, classes = load_model(MAIN / "data/work", device)
    head = MLPClassifier(MAIN / "data/work/mlp_head.pt", device)
    print(f"model 1: {MAIN / 'data/work/model_best.pt'}  |  device: {device}")

    d5 = pd.read_csv(MAIN / "data/work/crops/D5.csv")
    z = np.load(MAIN / "results/embeddings.npz", allow_pickle=False)
    m = z["dataset"] == "D5"
    seen = dict(zip(z["crop_path"][m], z["in_d1_train"][m]))  # project overlap check, from embed.py
    d5["photo_in_d1_train"] = d5.crop_path.map(seen).fillna(False).astype(bool)
    strict = set(Path(p).name for p in pd.read_csv(STRICT).crop_path) if STRICT.exists() else set()
    d5["strict_unseen"] = d5.crop_path.map(lambda p: Path(p).name in strict)
    d5["true"] = d5.cls.map(CLASS)

    dl = DataLoader(CropDataset([str(MAIN / p) for p in d5.crop_path]), batch_size=128, num_workers=4)
    P_fc, P_mlp, C_fc, C_mlp = [], [], [], []
    with torch.no_grad():
        for x, _ in dl:
            emb = embed_batch(model, x.to(device))
            pf, pm = torch.softmax(model.fc(emb), 1), head(emb)
            P_fc.append(pf.argmax(1).cpu()); C_fc.append(pf.max(1).values.cpu())
            P_mlp.append(pm.argmax(1).cpu()); C_mlp.append(pm.max(1).values.cpu())
    d5["pred_fc"] = [classes[i] for i in torch.cat(P_fc).tolist()]
    d5["pred_mlp"] = [classes[i] for i in torch.cat(P_mlp).tolist()]
    d5["conf_fc"], d5["conf_mlp"] = torch.cat(C_fc).numpy().round(4), torch.cat(C_mlp).numpy().round(4)
    d5[["crop_path", "cls", "true", "photo_in_d1_train", "strict_unseen", "pred_fc", "conf_fc", "pred_mlp", "conf_mlp"]].to_csv(
        OUT / "predictions_d5.csv", index=False)

    def score(df):
        r = {"crops": len(df), "photos": int(df.image.nunique())}
        for k in ["fc", "mlp"]:
            ok = df[f"pred_{k}"] == df.true
            r[f"{k}_accuracy"] = float(ok.mean())
            rec = {c: float(ok[df.true == c].mean()) for c in ["dent", "missing-head"] if (df.true == c).any()}
            r[f"{k}_recall"] = rec
            r[f"{k}_balanced_accuracy"] = float(np.mean(list(rec.values())))
        return r

    mapped = d5[d5.true.notna()]
    sets = {"A. strict unseen (fair outside test)": mapped[mapped.strict_unseen],
            "B1. all D5 dent + fastener damage": mapped,
            "B2. photo also in D1-train (seen)": mapped[mapped.photo_in_d1_train],
            "B3. photo not in D1-train (project check)": mapped[~mapped.photo_in_d1_train]}
    rep = {k: score(v) for k, v in sets.items()}
    rup = d5[d5.cls == "Rupture"]
    rep["C. Rupture: predicted as (ResNet fc)"] = rup.pred_fc.value_counts().to_dict()
    rep["C. Rupture: predicted as (MLP head)"] = rup.pred_mlp.value_counts().to_dict()
    (OUT / "report.json").write_text(json.dumps(rep, indent=2))

    A = sets["A. strict unseen (fair outside test)"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
    for ax, k, title in [(axes[0], "fc", "Model 1 (ResNet fc)"), (axes[1], "mlp", "Model 1 + MLP head")]:
        cm = pd.crosstab(A.true, A[f"pred_{k}"]).reindex(index=["dent", "missing-head"], columns=classes, fill_value=0)
        cmn = cm.div(cm.sum(axis=1), axis=0).to_numpy()
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm.iat[i, j]}\n{cmn[i, j] * 100:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if cmn[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
        ax.set_yticks([0, 1], ["dent", "missing-head\n(fastener damage)"])
        ax.set(xlabel="predicted", title=f"{title}: D5 strict unseen ({len(A)} crops)")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_strict_unseen.png", dpi=150)
    plt.close(fig)

    d1 = json.loads((MAIN / "results/mlp/report.json").read_text())
    md = ["# Outside test on D5: model 1 (ResNet-18 trained on D1)\n",
          "Model: `data/work/model_best.pt` (the main model, trained on the original D1 split; not d1_grouped), alone "
          "and with its MLP head (`data/work/mlp_head.pt`). Nothing was retrained or tuned on D5. "
          "D5 classes mapped to D1: Dent → dent, Fastener Damage → missing-head. "
          "Rupture has no D1 class and is listed separately. "
          f"Device: {device}.\n",
          "| test set | crops | photos | ResNet fc accuracy | + MLP head accuracy | fc balanced acc | MLP balanced acc | "
          "dent recall (fc / MLP) | missing-head recall (fc / MLP) |", "|---|---|---|---|---|---|---|---|---|"]
    for k, r in rep.items():
        if not k.startswith(("A", "B")):
            continue
        dr = f"{r['fc_recall'].get('dent', float('nan')):.3f} / {r['mlp_recall'].get('dent', float('nan')):.3f}"
        mr = f"{r['fc_recall'].get('missing-head', float('nan')):.3f} / {r['mlp_recall'].get('missing-head', float('nan')):.3f}"
        bold = "**" if k.startswith("A") else ""
        md.append(f"| {bold}{k}{bold} | {r['crops']} | {r['photos']} | {bold}{r['fc_accuracy']:.4f}{bold} | "
                  f"{bold}{r['mlp_accuracy']:.4f}{bold} | {r['fc_balanced_accuracy']:.4f} | {r['mlp_balanced_accuracy']:.4f} | {dr} | {mr} |")
    md += [f"\nFor comparison, on D1-test (same model): ResNet fc accuracy {d1['test_resnet_fc']['accuracy']:.4f}, "
           f"+ MLP head {d1['test']['accuracy']:.4f}; recall dent {d1['test_resnet_fc']['per_class_recall']['dent']:.3f} / "
           f"{d1['test']['per_class_recall']['dent']:.3f}, missing-head {d1['test_resnet_fc']['per_class_recall']['missing-head']:.3f} / "
           f"{d1['test']['per_class_recall']['missing-head']:.3f}.\n",
           "## What the model calls the strict-unseen crops\n",
           "| true class | model | " + " | ".join(classes) + " |", "|---|---|" + "---|" * len(classes)]
    for t in ["dent", "missing-head"]:
        for k, name in [("fc", "ResNet fc"), ("mlp", "+ MLP head")]:
            vc = A[A.true == t][f"pred_{k}"].value_counts()
            n = (A.true == t).sum()
            md.append(f"| {t} | {name} | " + " | ".join(f"{vc.get(c, 0)} ({vc.get(c, 0) / n * 100:.0f}%)" for c in classes) + " |")
    md += ["\n## Rupture (no D1 class): what the model calls it\n",
           f"{len(rup)} crops. ResNet fc: " + ", ".join(f"{c} {n}" for c, n in rup.pred_fc.value_counts().items()) +
           ". MLP head: " + ", ".join(f"{c} {n}" for c, n in rup.pred_mlp.value_counts().items()) + ".\n",
           "![confusion](confusion_strict_unseen.png)\n"]
    (OUT / "report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
