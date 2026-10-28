"""Ahmed's gc7 vs gc7_fix (the same recipe on D1 with the 27 Oct box corrections), on the same crops.

From 1_main_model_resnet18/ (after build_head_gc7.py --name gc7_fix --data d1_gc7_fix):
  python enhanced_res/compare_gc7_fix.py

Both models are scored on both versions of val and test: the original crops (d1_gc7) and the corrected crops
(d1_gc7_fix; 54 boxes moved, relabelled or deleted, 21 train / 4 val / 29 test lines). B's model is chosen on val only;
test is reported once. One training run (seed 0) each, so a small gap may be run-to-run noise.
Writes results/gc7_fix/compare.md and compare.json.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from arch_a import calibrate as C
from model import EWORK, MODELS, RESULTS, EnhancedRes, group_bootstrap_ci, load_split

OUT = RESULTS / "gc7_fix"


def scores(p, d, classes, ci=False):
    cid = {c: i for i, c in enumerate(classes)}
    y = d.cls.map(cid).to_numpy()
    a = p.argmax(1)
    nd = cid["no-defect"]
    defect, clean = y != nd, y == nd
    r = {"crops": len(d), "accuracy": float((a == y).mean()), "macro_f1": float(f1_score(y, a, average="macro")),
         "defects_called_no_defect": float((a[defect] == nd).mean()),
         "no_defect_recall": float((a[clean] == nd).mean()),
         "no_defect_recall_healed": float((a[clean & (d.source == "healed").to_numpy()] == nd).mean()),
         "no_defect_recall_clean": float((a[clean & (d.source == "clean_cropper").to_numpy()] == nd).mean()),
         "recall": {c: float((a[y == i] == i).mean()) for i, c in enumerate(classes)},
         "called_no_defect_by_class": {c: float((a[y == i] == nd).mean()) for i, c in enumerate(classes) if i != nd}}
    if ci:
        lo, hi = group_bootstrap_ci(y, a, d.src_group.to_numpy())
        r["macro_f1_ci"] = [float(lo), float(hi)]
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {v: load_split(v) for v in ("d1_gc7", "d1_gc7_fix")}
    res = {}
    for name in ("gc7", "gc7_fix"):
        m = EnhancedRes(MODELS[name])
        paths = sorted(set().union(*[set(d.crop_path[d.split.isin(["val", "test"])]) for d in data.values()]))
        _, P, _ = C.outputs(m, paths, EWORK / f"compare_outputs_{name}.npz")
        prob = dict(zip(paths, P))
        for v, d in data.items():
            for sp in ("val", "test"):
                dd = d[d.split == sp].reset_index(drop=True)
                p = np.stack([prob[c] for c in dd.crop_path])
                res[(name, v, sp)] = scores(p, dd, m.classes, ci=sp == "test")
                print(name, v, sp, round(res[(name, v, sp)]["macro_f1"], 4), flush=True)
    (OUT / "compare.json").write_text(json.dumps({"|".join(k): v for k, v in res.items()}, indent=1))

    pct = lambda v: f"{v * 100:.1f}%"
    md = ["# gc7 (Ahmed) vs gc7_fix (corrected D1 boxes), on the same crops", "",
          "Same recipe, split, no-defect crops and seed (0); gc7_fix only differs in the 54 corrected D1 boxes "
          "(`d1_grouped/label_corrections/`). Each model is scored on both versions of val and test. "
          "**B's model is chosen on val**; test is read once, for the report. One training run each, so differences of "
          "about a point may be run-to-run noise.", ""]
    rows = [("7-class accuracy", lambda r: pct(r["accuracy"])),
            ("macro-F1", lambda r: f"{r['macro_f1']:.3f}" + (f" ({r['macro_f1_ci'][0]:.3f}–{r['macro_f1_ci'][1]:.3f})" if "macro_f1_ci" in r else "")),
            ("defects called no-defect", lambda r: pct(r["defects_called_no_defect"])),
            ("no-defect recall (healed / clean)", lambda r: f"{pct(r['no_defect_recall'])} ({pct(r['no_defect_recall_healed'])} / {pct(r['no_defect_recall_clean'])})")]
    for sp in ("val", "test"):
        md += [f"## {sp}", "", "| | gc7 on original crops | gc7_fix on original crops | gc7 on corrected crops | "
               "gc7_fix on corrected crops |", "|---|---:|---:|---:|---:|"]
        keys = [("gc7", "d1_gc7"), ("gc7_fix", "d1_gc7"), ("gc7", "d1_gc7_fix"), ("gc7_fix", "d1_gc7_fix")]
        for lbl, f in rows:
            md.append(f"| {lbl} | " + " | ".join(f(res[(n, v, sp)]) for n, v in keys) + " |")
        classes = list(res[("gc7", "d1_gc7", sp)]["recall"])
        md += ["", f"Recall per class, {sp}, corrected crops:", "", "| class | gc7 | gc7_fix |", "|---|---:|---:|"]
        md += [f"| {c} | {pct(res[('gc7', 'd1_gc7_fix', sp)]['recall'][c])} | {pct(res[('gc7_fix', 'd1_gc7_fix', sp)]['recall'][c])} |"
               for c in classes]
        md.append("")
    (OUT / "compare.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
