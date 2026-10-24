"""Test the trained main model (enhanced_res) without training it.

Loads data/work/enhanced_res/enhanced_res_backbone.pt + enhanced_res_head.pt, runs them on the grouped D1 test and
val crops (and the D5 outside test when its crops are there), and compares the numbers with results/ablation.csv
from the original run. If they match, the download and setup are working. Writes nothing.

From 1_main_model_resnet18/:
  python enhanced_res/evaluate.py                 # test + val + D5, a few minutes (8 views per crop)
  python enhanced_res/evaluate.py --split test

Needs data/work/ (main_model_data_work.zip) and data/work/enhanced_res/ (enhanced_res_model.zip).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader

from model import MAIN, RESULTS, EnhancedRes, group_bootstrap_ci, load_split
from klshift.common import CropDataset

TOL = 0.003   # GPU arithmetic can flip a borderline crop or two between runs


def run(model, paths, batch, workers):
    out = []
    for x, _ in DataLoader(CropDataset(paths), batch_size=batch, num_workers=workers):
        out.append(model(x).argmax(1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", nargs="+", default=["test", "val", "d5"], choices=["train", "val", "test", "d5"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    model = EnhancedRes()
    classes = model.classes
    cidx = {c: i for i, c in enumerate(classes)}
    lab = list(range(len(classes)))
    ref = pd.read_csv(RESULTS / "ablation.csv").set_index(["backbone", "head"]).loc[("r2", "h4")]
    crops = load_split()
    print(f"model: enhanced_res ({len(classes)} classes), device: {model.device}")

    ok = True
    for s in a.split:
        if s == "d5":
            f = RESULTS / "d5/ext_d5.csv"
            d = pd.read_csv(f)
            d["crop_path"] = [str(MAIN / p) for p in d.crop_path]
            if not Path(d.crop_path.iloc[0]).exists():
                print("\nD5: crops not found (data/work/crops/D5), skipped")
                continue
        else:
            d = crops[crops.split == s]
        y = d.cls.map(cidx).values
        p = run(model, list(d.crop_path), a.batch, a.workers)
        name = "D5 outside test (photos not in D1)" if s == "d5" else f"grouped D1-{s}"
        m = {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro", labels=sorted(set(y)))}
        if s != "d5":
            m["balanced_accuracy"] = balanced_accuracy_score(y, p)
        refs = {"test": {"accuracy": ref.test_acc, "macro_f1": ref.test_macro_f1, "balanced_accuracy": ref.test_bal_acc},
                "val": {"accuracy": ref.val_acc, "macro_f1": ref.val_macro_f1},
                "d5": {"accuracy": ref.d5_acc}}.get(s, {})
        print(f"\n{name}: {len(d)} crops")
        print(f"  {'metric':20s} {'now':>8s} {'original':>9s}")
        for k, v in m.items():
            r = refs.get(k)
            same = r is not None and abs(v - r) <= TOL
            ok &= r is None or same
            print(f"  {k:20s} {v:8.4f} {'' if r is None else f'{r:9.4f}'}  {'' if r is None else ('match' if same else 'DIFFERENT')}")
        present = sorted(set(y))
        rec = recall_score(y, p, labels=present, average=None, zero_division=0)
        print("  per-class recall: " + ", ".join(f"{classes[c]} {r:.3f}" for c, r in zip(present, rec)))
        if s == "test":
            lo, hi = group_bootstrap_ci(y, p, d.src_group.values)
            print(f"  macro-F1 95% interval (resampling whole source groups): {lo:.3f}-{hi:.3f}")
            cm = pd.DataFrame(confusion_matrix(y, p, labels=lab), index=classes, columns=classes)
            print("  confusion matrix (rows = true, columns = predicted):")
            print("\n".join("    " + line for line in cm.to_string().splitlines()))
    print("\nOK: results match the original enhanced_res run." if ok else
          "\nWARNING: some numbers differ from results/ablation.csv. Check that the data and model are the shared versions.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
