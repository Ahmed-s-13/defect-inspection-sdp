"""Re-test the trained MLP head (step 7) on D1-test and D1-val, without training.

  python evaluate_mlp.py                 from the saved embeddings, a few seconds on the Apple GPU
  python evaluate_mlp.py --from-images   re-embeds the crops through the ResNet-18 first
                                         (checks the whole image -> prediction path, about 1 min per split)

Compares the numbers with results/mlp/report.json from train_mlp.py. Writes nothing.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader

from evaluate import load_model
from klshift.common import CropDataset, embed_batch, pick_device
from klshift.mlp import HEAD_FILE, MLPClassifier, load_d1_embeddings


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=str(Path(__file__).resolve().parents[1] / "data/d1_sscd_grouped/work"), help="folder with model_best.pt and mlp_head.pt")
    ap.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "data/d1_sscd_grouped/results"), help="folder with embeddings.npz")
    ap.add_argument("--report", default="results/mlp/report.json", help="numbers from train_mlp.py to compare with")
    ap.add_argument("--split", nargs="+", default=["test", "val"], choices=["train", "val", "test"])
    ap.add_argument("--from-images", action="store_true", help="re-embed the crops instead of using embeddings.npz")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cpu", action="store_true", help="force CPU (default: Apple GPU / MPS)")
    a = ap.parse_args()
    work = Path(a.work)
    device = torch.device("cpu") if a.cpu else pick_device()
    head = MLPClassifier(work / HEAD_FILE, device)
    X, meta, classes = load_d1_embeddings(Path(a.results) / "embeddings.npz")
    assert classes == head.classes, "class order of embeddings.npz and the MLP head differ"
    y = np.array([classes.index(c) for c in meta["cls"]])
    ref = json.loads(Path(a.report).read_text()) if Path(a.report).exists() else {}
    resnet = load_model(work, device)[0] if a.from_images else None
    lab = list(range(len(classes)))
    print(f"head: {work / HEAD_FILE}  |  device: {device}  |  kNN weight {head.w}")

    ok = True
    for s in a.split:
        m = meta["split"] == s
        if a.from_images:
            dl = DataLoader(CropDataset(meta["crop_path"][m]), batch_size=a.batch, num_workers=a.workers)
            with torch.no_grad():
                Xs = torch.cat([embed_batch(resnet, x.to(device)) for x, _ in dl])
        else:
            Xs = X[m]
        p = head(Xs).argmax(1).cpu().numpy()
        res = {"accuracy": accuracy_score(y[m], p), "macro_f1": f1_score(y[m], p, average="macro"),
               "balanced_accuracy": balanced_accuracy_score(y[m], p)}
        print(f"\nD1-{s}: {m.sum()} crops" + (" (re-embedded from the images)" if a.from_images else ""))
        print(f"  {'metric':20s} {'now':>8s} {'original':>9s}")
        for k, v in res.items():
            r = ref.get(s, {}).get(k)
            same = r is not None and abs(v - r) < (5e-3 if a.from_images else 1e-3)
            ok &= r is None or same
            print(f"  {k:20s} {v:8.4f} {'' if r is None else f'{r:9.4f}'}  {'' if r is None else ('match' if same else 'DIFFERENT')}")
        rec = recall_score(y[m], p, labels=lab, average=None, zero_division=0)
        print("  per-class recall: " + ", ".join(f"{c} {r:.3f}" for c, r in zip(classes, rec)))
        if s == "test":
            cm = confusion_matrix(y[m], p, labels=lab)
            w = max(map(len, classes))
            print("  confusion matrix (rows = true, columns = predicted):")
            print("    " + " " * w + "".join(f"{c[:6]:>8s}" for c in classes))
            for c, r in zip(classes, cm):
                print(f"    {c:{w}s}" + "".join(f"{v:8d}" for v in r))
    print("\nOK: results match the original MLP training run." if ok else
          "\nWARNING: some numbers differ from results/mlp/report.json.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
