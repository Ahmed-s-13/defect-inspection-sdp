"""Step 4 diagnostic: how much of the artifact-probe signal is just the second JPEG save?

Generated crops are decoded and re-saved (JPEG q95) once more than real crops. This probes
'real clean patch re-saved q95, nothing inpainted' vs 'real clean patch', with the same
disjoint image-group halves and the same fine-tuned ResNet-18 as the artifact probe.
"""
import argparse
import json

import hcommon as H

from PIL import Image
from sklearn.metrics import roc_auc_score

import bincls


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = H.pick_device()
    c = H.d1_clean()
    half = H.group_half(c, a.seed)  # same halves as checks.probe_sets
    out_dir = H.WORK / "reencoded" / "D1"
    out_dir.mkdir(parents=True, exist_ok=True)
    S = {}
    for s in ["train", "val", "test"]:
        real = c[(c.split == s) & ~c.group_id.isin(half)]
        re = c[(c.split == s) & c.group_id.isin(half)]
        paths = []
        for p in re.path:
            q = out_dir / p.split("/")[-1]
            if not q.exists():
                Image.open(p).convert("RGB").save(q, quality=95)
            paths.append(str(q))
        S[s] = (list(real.path) + paths, [0] * len(real) + [1] * len(paths))
    m, vauc, hist = bincls.train_binary(*S["train"], *S["val"], device, epochs=a.epochs, patience=a.epochs,
                                        workers=a.workers, seed=a.seed)
    r = {"reencode_only_vs_clean_finetuned_val_AUROC": vauc,
         "reencode_only_vs_clean_finetuned_test_AUROC":
             roc_auc_score(S["test"][1], bincls.predict(m, S["test"][0], device, workers=a.workers)),
         "epochs": len(hist)}
    print(r)
    (H.RES / "step4_probe_reencode.json").write_text(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
