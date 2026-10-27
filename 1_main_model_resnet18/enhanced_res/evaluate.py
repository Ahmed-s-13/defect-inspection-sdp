"""Test the trained main model (enhanced_res) without training it.

Loads the weights, runs them on the test and val crops (and the D5 outside test when its crops are there), and
compares the numbers with the original run. If they match, the download and setup are working. On the test split it
also checks architecture A (arch_a/: gates + temperature + conformal set) on the test half against the numbers
calibrate.py stored. Writes nothing.
  --model gc7 (default)  the main model: 7 classes on d1_gc7, data/work/enhanced_res/gc7/, checked against
                         results/gc7/summary.json and arch_a/calibration_gc7.json
  --model v2             the 6-class model on d1_grouped, data/work/enhanced_res/, checked against
                         results/ablation.csv and arch_a/calibration.json

From 1_main_model_resnet18/:
  python enhanced_res/evaluate.py                 # gc7: test + val + D5, a few minutes (8 views per crop)
  python enhanced_res/evaluate.py --split test
  python enhanced_res/evaluate.py --model v2

Needs data/work/ (main_model_data_work.zip), the no-defect crops for gc7 (d1_gc7_no_defect_crops.zip) and the
weights (enhanced_res_gc7_model.zip for gc7, enhanced_res_model.zip for v2).
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

import json

from model import HERE, MAIN, MODELS, RESULTS, EnhancedRes, group_bootstrap_ci, load_split
from arch_a.runtime import CALIBRATIONS, ArchA, summarise
from klshift.common import CropDataset
from PIL import Image

TOL = 0.003   # GPU arithmetic can flip a borderline crop or two between runs


def run(model, paths, batch, workers):
    """Probabilities, kNN distances and features (architecture A's unfamiliar-input gate) for every crop."""
    P, D, F = [], [], []
    for x, _ in DataLoader(CropDataset(paths), batch_size=batch, num_workers=workers):
        f = model.features(x)
        P.append(model.proba_from_features(f).cpu())
        D.append(model.knn_distance(f).cpu())
        F.append(f.cpu())
    return torch.cat(P).numpy(), torch.cat(D).numpy(), torch.cat(F).numpy()


def check_arch_a(d, prob, dist, feats, classes, model):
    """Architecture A on the test half of the test split vs the calibration file's expected numbers."""
    arch = ArchA(CALIBRATIONS[model])
    sp = pd.read_csv(HERE / "arch_a/cal_split.csv")
    sp["crop_path"] = [str(MAIN / p) for p in sp.crop_path]
    fam = sp.drop_duplicates("src_group").set_index("src_group").split       # no-defect crops: their family's half
    half = d[["crop_path"]].merge(sp[["crop_path", "split"]], on="crop_path", how="left").split
    half = half.fillna(pd.Series(d.src_group.map(fam).to_numpy())).to_numpy()
    t = half == "test"
    dt = d[t]
    y = dt.cls.map({c: i for i, c in enumerate(classes)}).to_numpy()
    dec = arch.decide(prob[t], crops=[Image.open(p).convert("RGB") for p in dt.crop_path], knn_dist=dist[t],
                      feats=feats[t])
    m = summarise(dec, arch, prob[t], y, dt.src_group.to_numpy())
    exp = arch.cal["expected_test_half"]
    print(f"\narchitecture A on the test half ({t.sum()} crops, {dt.src_group.nunique()} families; "
          f"alpha {arch.alpha:.2f}, T {arch.T:.3f}):")
    print(f"  {'metric':20s} {'now':>8s} {'expected':>9s}")
    ok = True
    for k in ("cp_coverage", "cp_family_coverage", "cp_mean_set_size", "cp_single_class", "human_attention",
              "single_class_wrong", "gates_false_reject_family"):
        same = abs(m[k] - exp[k]) <= 0.005 * (3 if k == "cp_mean_set_size" else 1)
        ok &= same
        print(f"  {k:20s} {m[k]:8.4f} {exp[k]:9.4f}  {'match' if same else 'DIFFERENT'}")
    print("  visor actions: " + ", ".join(f"{k} {v * 100:.1f}%" for k, v in m["actions"].items()))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["gc7", "v2"], default="gc7")
    ap.add_argument("--split", nargs="+", default=["test", "val", "d5"], choices=["train", "val", "test", "d5"])
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    model = EnhancedRes(MODELS[a.model])
    classes = model.classes
    cidx = {c: i for i, c in enumerate(classes)}
    lab = list(range(len(classes)))
    if a.model == "v2":
        ref = pd.read_csv(RESULTS / "ablation.csv").set_index(["backbone", "head"]).loc[("r2", "h4")]
        refs_all = {"test": {"accuracy": ref.test_acc, "macro_f1": ref.test_macro_f1, "balanced_accuracy": ref.test_bal_acc},
                    "val": {"accuracy": ref.val_acc, "macro_f1": ref.val_macro_f1}, "d5": {"accuracy": ref.d5_acc}}
    else:
        S = json.loads((RESULTS / "gc7/summary.json").read_text())
        refs_all = {"test": {"accuracy": S["test"]["accuracy"], "macro_f1": S["test"]["macro_f1"]},
                    "val": {"accuracy": S["val"]["accuracy"], "macro_f1": S["val"]["macro_f1"]}}
    crops = load_split("d1_gc7" if a.model == "gc7" else "d1_grouped")
    print(f"model: enhanced_res {a.model} ({len(classes)} classes), device: {model.device}")

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
        prob, dist, feats = run(model, list(d.crop_path), a.batch, a.workers)
        p = prob.argmax(1)
        name = "D5 outside test (photos not in D1)" if s == "d5" else f"{'d1_gc7' if a.model == 'gc7' else 'grouped D1'}-{s}"
        m = {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro", labels=sorted(set(y)))}
        if s != "d5":
            m["balanced_accuracy"] = balanced_accuracy_score(y, p)
        refs = refs_all.get(s, {})
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
            ok &= check_arch_a(d, prob, dist, feats, classes, a.model)
    src = "results/gc7/summary.json" if a.model == "gc7" else "results/ablation.csv"
    print(f"\nOK: results match the original enhanced_res {a.model} run (and architecture A's calibration)." if ok else
          f"\nWARNING: some numbers differ from {src}. Check that the data and model are the shared versions.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
