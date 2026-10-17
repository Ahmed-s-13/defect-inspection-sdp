"""Test the trained main model without training it.

Loads <work>/model_best.pt, runs it on the D1 test (and val) crops of the saved split,
and compares the numbers with results/d1_report.json from the original training run.
If they match, the download and setup are working.

  python evaluate.py              D1-test + D1-val, about 1-2 minutes
  python evaluate.py --split test

Needs data/work/ (main_model_data_work.zip) and model_best.pt (trained_models.zip).
Writes nothing.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader

from klshift.common import CropDataset, build_model, pick_device
from train import predict


def load_model(work, device):
    ck = torch.load(Path(work) / "model_best.pt", map_location="cpu")
    model = build_model(len(ck["classes"]), pretrained=False)
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), ck["classes"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default="data/work")
    ap.add_argument("--results", default="results", help="where the original d1_report.json is")
    ap.add_argument("--split", nargs="+", default=["test", "val"], choices=["train", "val", "test"])
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    work = Path(a.work)
    for f in [work / "model_best.pt", work / "d1_split.csv"]:
        if not f.exists():
            raise SystemExit(f"missing {f}: unzip trained_models.zip and main_model_data_work.zip in the repo root")
    device = pick_device()
    model, classes = load_model(work, device)
    split = pd.read_csv(work / "d1_split.csv")
    split["y"] = split.cls.map({c: i for i, c in enumerate(classes)})
    ref_file = Path(a.results) / "d1_report.json"
    ref = json.loads(ref_file.read_text()) if ref_file.exists() else {}
    lab = list(range(len(classes)))
    print(f"model: {work / 'model_best.pt'} ({len(classes)} classes), device: {device}")

    ok = True
    for s in a.split:
        d = split[split.split == s]
        dl = DataLoader(CropDataset(d.crop_path, d.y), batch_size=a.batch, num_workers=a.workers)
        y, p = predict(model, dl, device)
        m = {"accuracy": accuracy_score(y, p), "macro_f1": f1_score(y, p, average="macro"),
             "balanced_accuracy": balanced_accuracy_score(y, p)}
        print(f"\nD1-{s}: {len(d)} crops")
        print(f"  {'metric':20s} {'now':>8s} {'original':>9s}")
        for k, v in m.items():
            r = ref.get(f"{s}_{k}")
            same = r is not None and abs(v - r) < 1e-3
            ok &= r is None or same
            print(f"  {k:20s} {v:8.4f} {'' if r is None else f'{r:9.4f}'}  {'' if r is None else ('match' if same else 'DIFFERENT')}")
        rec = recall_score(y, p, labels=lab, average=None, zero_division=0)
        print("  per-class recall: " + ", ".join(f"{c} {r:.3f}" for c, r in zip(classes, rec)))
        if s == "test":
            cm = pd.DataFrame(confusion_matrix(y, p, labels=lab), index=classes, columns=classes)
            print("  confusion matrix (rows = true, columns = predicted):")
            print("\n".join("    " + line for line in cm.to_string().splitlines()))
    print("\nOK: results match the original training run." if ok else
          "\nWARNING: some numbers differ from results/d1_report.json. Check that the data and model are the shared versions.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
