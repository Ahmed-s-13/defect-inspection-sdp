"""gc7, step 2: build the full enhanced_res head on the 7-class backbone, test it once, compare it with v2.

  python enhanced_res/build_head_gc7.py        (from 1_main_model_resnet18/; ~15 min on an M-series Mac)
  python enhanced_res/build_head_gc7.py --name gc7_fix --data d1_gc7_fix     the same on D1 with the 27 Oct box
                                               corrections (runs/gc7_fix -> data/work/enhanced_res/gc7_fix, results/gc7_fix)

Needs data/work/enhanced_res/runs/gc7/backbone.pt (train_backbone.py --data d1_gc7 --out gc7), the d1_gc7 crops, and
the v2 model (data/work/enhanced_res/, for the comparison). The head is build_head.py's h4, unchanged: pooled
layer3+layer4 features averaged over the 8 flips/rotations, 3 augmented views of every train crop, 5 MLPs + a 5-NN
vote; the kNN weight and the prior correction are picked on val. Writes:
  data/work/enhanced_res/gc7/enhanced_res_backbone.pt + enhanced_res_head.pt     the main model (Release v3.0-gc7)
  results/gc7/summary.json, report.md, confusion_gc7.png                           test results (evaluate.py checks them)

Test sets, only read for the report:
  test       d1_gc7 test: 4,255 defect crops + 615 no-defect crops
  extra      d1_gc7/extra_eval.csv: no-defect crops trimmed from val/test to keep 70/15/15 (never trained on)
  decoys     val/test defect crops with a clean patch LaMa-filled (../2_lama_healing/no_defect/decoys.py), when their
             crops are in 2_lama_healing/no_defect/work/: if gc7 calls them "no defect" more often than the same crops
             untouched, it learned the LaMa trace instead of "no damage"
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader

import build_head as bh
from model import (BACKBONE_FILE, EWORK, HEAD_FILE, MAIN, MODELS, RESULTS, EnhancedRes, group_bootstrap_ci,
                   load_split)
from klshift.common import CropDataset, pick_device
from train_mlp import plot_confusion

HEALER_ND = MAIN.parent / "2_lama_healing/no_defect"
ND = 6                       # index of no-defect
MISS = [0.01, 0.02, 0.05]    # val defect-miss targets for the p(no-defect) cut


def r(v):
    return round(float(v), 4)


def decoy_set():
    """Decoy crops (val/test, with a LaMa patch) and their untouched originals; empty if the crops aren't there."""
    f = HEALER_ND / "decoys.csv"
    if not f.exists():
        return pd.DataFrame(columns=["crop_path", "cls", "original"])
    d = pd.read_csv(f)
    d = d[d.has_decoy & d.split.isin(["val", "test"])]
    d = pd.DataFrame(dict(crop_path=[str(HEALER_ND / p) for p in d.crop], cls=d.cls,
                          original=[str(MAIN / p) for p in d.source_crop]))
    return d[[Path(p).exists() for p in d.crop_path]].reset_index(drop=True)


@torch.no_grad()
def v2_proba(paths):
    m = EnhancedRes(MODELS["v2"])
    return torch.cat([m(x).cpu() for x, _ in DataLoader(CropDataset(list(paths)), batch_size=64, num_workers=4)]).numpy()


@torch.no_grad()
def head_proba(pc, X, n_classes, device):
    """The fitted head (run_head's pieces) on TTA features."""
    T = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)  # noqa: E731
    X = T(X[:, slice(*pc["cols"])])
    heads = []
    for st in pc["heads"]:
        h = bh.Head(pc["d_in"], n_classes)
        h.load_state_dict(st)
        heads.append(h.to(device).eval())
    p = torch.stack([h(X).softmax(1) for h in heads]).mean(0)
    p = (1 - pc["alpha"]) * p + pc["alpha"] * bh.knn_proba(bh.l2n(T(pc["knn_bank"])), T(pc["knn_labels"]).long(),
                                                            bh.l2n(X), 5, n_classes)
    if pc["tau"]:
        p = p / T(pc["prior"]) ** pc["tau"]
        p = p / p.sum(1, keepdim=True)
    return p.cpu().numpy()


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="gc7", help="run name under runs/, and the model / results folder")
    ap.add_argument("--data", default="d1_gc7", choices=["d1_gc7", "d1_gc7_fix"])
    a = ap.parse_args()
    RUN, OUT = EWORK / f"runs/{a.name}/backbone.pt", RESULTS / a.name
    device = pick_device()
    OUT.mkdir(parents=True, exist_ok=True)
    crops = load_split(a.data)
    ck = torch.load(RUN, map_location="cpu", weights_only=False)
    classes = ck["classes"]
    assert classes[ND] == "no-defect" and set(classes) == set(crops.cls)
    cidx = {c: i for i, c in enumerate(classes)}
    y, sp = crops.cls.map(cidx).values, crops.split.values
    extra = pd.read_csv(MAIN / "d1_gc7/extra_eval.csv")
    extra["crop_path"] = [str(MAIN / p) for p in extra.crop_path]
    dec = decoy_set()

    bh.BACKBONES[a.name] = RUN
    n, ne = len(crops), len(extra)
    paths = list(crops.crop_path) + list(extra.crop_path) + list(dec.crop_path) + list(dec.original)
    plain, tta, aug = bh.extract(a.name, paths, device, list(crops.crop_path[sp == "train"]))
    F = dict(plain=plain[:n], tta=tta[:n], aug=aug, ext_plain=plain[n:n + ne], ext_tta=tta[n:n + ne])
    prob, pc = bh.run_head("h4", F, y, sp, extra, classes, device)

    dst = MODELS[a.name]
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy(RUN, dst / BACKBONE_FILE)
    torch.save(dict(classes=classes, backbone=BACKBONE_FILE, **pc,
                    note="gc7 (Release v3.0-gc7): enhanced_res's h4 head on the 7-class backbone. features = pooled "
                         "layer3+layer4, averaged over the 8 flips/rotations; prob = ((1-alpha)*mean softmax of heads + "
                         "alpha*5NN) / prior**tau"), dst / HEAD_FILE)

    te, va = sp == "test", sp == "val"
    yt, pt, pv = y[te], prob["test"], prob["val"]
    at = pt.argmax(1)
    d, nd = yt < ND, yt == ND
    src = crops.source.values[te]
    p_v2 = v2_proba(crops.crop_path[te])
    nd_ = len(dec)
    p_dec = head_proba(pc, tta[n + ne:n + ne + nd_], len(classes), device) if nd_ else np.zeros((0, len(classes)))
    p_org = head_proba(pc, tta[n + ne + nd_:], len(classes), device) if nd_ else np.zeros((0, len(classes)))
    yd = dec.cls.map(cidx).values

    S = dict(model=a.name, data=a.data, classes=classes, head=dict(knn_weight=pc["alpha"], tau=pc["tau"]),
             backbone_epoch=int(ck["epoch"]))
    S["val"] = dict(accuracy=r((pv.argmax(1) == y[va]).mean()), macro_f1=r(f1_score(y[va], pv.argmax(1), average="macro")))
    lo, hi = group_bootstrap_ci(yt, at, crops.src_group.values[te])
    S["test"] = dict(accuracy=r((at == yt).mean()), macro_f1=r(f1_score(yt, at, average="macro")), macro_f1_ci=[r(lo), r(hi)],
                     recall={c: r((at[yt == i] == i).mean()) for i, c in enumerate(classes)})
    pd6 = p_v2[d].argmax(1)
    S["test_defects"] = dict(
        n=int(d.sum()),
        v2=dict(accuracy=r((pd6 == yt[d]).mean()), macro_f1=r(f1_score(yt[d], pd6, average="macro"))),
        gc7_top_answer=dict(accuracy=r((at[d] == yt[d]).mean()),
                            macro_f1=r(f1_score(yt[d], at[d], average="macro", labels=list(range(ND))))),
        gc7_best_defect_class=dict(accuracy=r((pt[d, :ND].argmax(1) == yt[d]).mean()),
                                   macro_f1=r(f1_score(yt[d], pt[d, :ND].argmax(1), average="macro"))),
        called_no_defect=r((at[d] == ND).mean()),
        called_no_defect_by_class={c: r((at[d][yt[d] == i] == ND).mean()) for i, c in enumerate(classes[:ND])})
    S["test_no_defect"] = dict(
        n=int(nd.sum()), recall=r((at[nd] == ND).mean()),
        recall_by_source={s: r((at[nd & (src == s)] == ND).mean()) for s in ("healed", "clean_cropper")},
        v2_calls={c: r((p_v2[nd].argmax(1) == i).mean()) for i, c in enumerate(classes[:ND])})
    pe = prob["ext"]
    S["extra_no_defect"] = dict(n=ne, recall=r((pe.argmax(1) == ND).mean()),
                                recall_by_source={s: r((pe[extra.source.values == s].argmax(1) == ND).mean())
                                                  for s in ("healed", "clean_cropper")})
    yv = y[va]
    S["dial"] = []
    for m in MISS:
        cut = float(np.quantile(pv[yv < ND, ND], 1 - m))
        call = lambda p: np.where(p[:, ND] > cut, ND, p[:, :ND].argmax(1))  # noqa: E731
        ct = call(pt)
        S["dial"].append(dict(val_miss_target=m, cut=r(cut), test_defects_missed=r((ct[d] == ND).mean()),
                              test_defect_accuracy=r((ct[d] == yt[d]).mean()), test_no_defect_recall=r((ct[nd] == ND).mean()),
                              healed_recall=r((ct[nd & (src == "healed")] == ND).mean()),
                              clean_recall=r((ct[nd & (src == "clean_cropper")] == ND).mean()),
                              extra_recall=r((call(pe) == ND).mean())))
    if nd_:
        S["decoys"] = dict(n=nd_, called_no_defect=dict(decoy=r((p_dec.argmax(1) == ND).mean()),
                                                        original=r((p_org.argmax(1) == ND).mean())),
                           mean_p_no_defect=dict(decoy=r(p_dec[:, ND].mean()), original=r(p_org[:, ND].mean())),
                           right_class=dict(decoy=r((p_dec.argmax(1) == yd).mean()), original=r((p_org.argmax(1) == yd).mean())))
    (OUT / "summary.json").write_text(json.dumps(S, indent=1))
    plot_confusion(confusion_matrix(yt, at, labels=list(range(len(classes)))), classes, f"{a.name}: {a.data} test",
                   OUT / f"confusion_{a.name}.png")
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
