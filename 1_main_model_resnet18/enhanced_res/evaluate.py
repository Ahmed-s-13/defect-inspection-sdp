"""Test the trained main model (enhanced_res) without training it.

Loads data/work/enhanced_res/enhanced_res_backbone.pt + enhanced_res_head.pt, runs them on the grouped D1 test and
val crops (and the D5 outside test when its crops are there), and compares the numbers with results/ablation.csv
from the original run. If they match, the download and setup are working. On the test split it also checks
architecture A (arch_a/: gates + temperature + conformal set) on the test half against the numbers calibrate.py
stored in arch_a/calibration.json. Writes nothing.

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

from model import HERE, MAIN, RESULTS, EnhancedRes, group_bootstrap_ci, load_split
from arch_a.runtime import ArchA, summarise
from klshift.common import CropDataset
from PIL import Image

TOL = 0.003   # GPU arithmetic can flip a borderline crop or two between runs


def run(model, paths, batch, workers):
    """Probabilities and kNN distances (architecture A's unfamiliar-input score) for every crop."""
    P, D = [], []
    for x, _ in DataLoader(CropDataset(paths), batch_size=batch, num_workers=workers):
        f = model.features(x)
        P.append(model.proba_from_features(f).cpu())
        D.append(model.knn_distance(f).cpu())
    return torch.cat(P).numpy(), torch.cat(D).numpy()


def check_arch_a(d, prob, dist, classes):
    """Architecture A on the test half of the grouped test split vs calibration.json's expected numbers."""
    arch = ArchA()
    sp = pd.read_csv(HERE / "arch_a/cal_split.csv")
    sp["crop_path"] = [str(MAIN / p) for p in sp.crop_path]
    half = d[["crop_path"]].merge(sp[["crop_path", "split"]], on="crop_path", how="left").split.to_numpy()
    t = half == "test"
    dt = d[t]
    y = dt.cls.map({c: i for i, c in enumerate(classes)}).to_numpy()
    dec = arch.decide(prob[t], crops=[Image.open(p).convert("RGB") for p in dt.crop_path], knn_dist=dist[t])
    m = summarise(dec, arch, prob[t], y, dt.src_group.to_numpy())
    exp = arch.cal["expected_test_half"]
    print(f"\narchitecture A on the test half ({t.sum()} crops, {dt.src_group.nunique()} families; "
          f"alpha {arch.alpha:.2f}, T {arch.T:.3f}):")
    print(f"  {'metric':20s} {'now':>8s} {'expected':>9s}")
    ok = True
    for k in ("cp_coverage", "cp_family_coverage", "cp_mean_set_size", "cp_single_class", "human_attention",
              "single_class_wrong"):
        same = abs(m[k] - exp[k]) <= 0.005 * (3 if k == "cp_mean_set_size" else 1)
        ok &= same
        print(f"  {k:20s} {m[k]:8.4f} {exp[k]:9.4f}  {'match' if same else 'DIFFERENT'}")
    print("  visor actions: " + ", ".join(f"{k} {v * 100:.1f}%" for k, v in m["actions"].items()))
    return ok


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
        prob, dist = run(model, list(d.crop_path), a.batch, a.workers)
        p = prob.argmax(1)
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
            ok &= check_arch_a(d, prob, dist, classes)
    print("\nOK: results match the original enhanced_res run (and architecture A's calibration)." if ok else
          "\nWARNING: some numbers differ from results/ablation.csv. Check that the data and model are the shared versions.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
