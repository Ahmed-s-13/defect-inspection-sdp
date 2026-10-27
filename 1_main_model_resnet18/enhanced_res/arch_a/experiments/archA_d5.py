"""Distribution-shift test: architecture 0 and architecture A on the D5 outside test (photos not copied from D1).

From 1_main_model_resnet18/:
  python enhanced_res/arch_a/experiments/archA_d5.py

Needs data/raw/D5 (Roboflow D5 version 3) and data/work/crops/D5.csv (prepare.py --datasets D5).

Test set: the team's D5 outside test (enhanced_res/results/d5/ext_d5.csv: D5 crops whose photo is not a copy of a D1
photo), restricted to crops whose current crop file and class agree with that list (crop numbering moved for a few
boxes). D5 has only two of our six classes: Dent -> dent, Fastener Damage -> missing-head. Near-duplicate groups inside
D5 (enhanced_res/results/d5/d5_images.csv, d5_group) are the "families" for family-level numbers.

Everything is calibrated on D1 only (calibration.json: T on val, CP thresholds and the OOD gate on val + cal half, the quality gate on cal half + train) and then
applied unchanged to D5. Nothing is refitted on D5: the point is whether the guarantees survive a new source.
  architecture 0   model as shipped, crop-weighted LAC
  architecture A   the deployed calibration.json: quality gate (0.10, sharpness both sides) -> OOD gate (kNN +
                   AoN, 0.08) -> prob-T + family-weighted global LAC
Writes enhanced_res/arch_a/results/archA/d5/.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

import arch0_cp as A0
import archA_cp as A1
import archA_gates as G
import archA_visor as V
from model import MAIN, EnhancedRes
from klshift.common import CropDataset
from klshift.mlp import knn_proba, l2n

CACHE = A0.CACHE
OUT = HERE.parent / "results" / "archA" / "d5"
NAME = {"Dent": "dent", "Fastener Damage": "missing-head"}
ALPHAS = (0.05, 0.10, 0.20)


def d5_crops():
    ext = pd.read_csv(MAIN / "enhanced_res/results/d5/ext_d5.csv")
    cur = pd.read_csv(MAIN / "data/work/crops/D5.csv")[["crop_path", "cls"]].rename(columns={"cls": "d5_cls"})
    d = ext.merge(cur, on="crop_path", how="inner")
    d = d[d.d5_cls.map(NAME) == d.cls].reset_index(drop=True)
    grp = pd.read_csv(MAIN / "enhanced_res/results/d5/d5_images.csv")[["image", "d5_group"]]
    d = d.merge(grp, on="image", how="left")
    d["d5_group"] = d.d5_group.fillna(d.image)
    d["crop_path"] = [str(MAIN / p) for p in d.crop_path]
    return d, len(ext)


@torch.no_grad()
def d5_parts(d):
    f = CACHE / "feats_d5.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        if list(z["crop_path"]) == list(d.crop_path):
            return z["feats"], z["logits"], z["knn"]
    m = EnhancedRes()
    F = [m.features(x) for x, _ in DataLoader(CropDataset(list(d.crop_path)), batch_size=64, num_workers=4)]
    feats = torch.cat(F)
    logits = torch.stack([h(feats) for h in m.heads])
    knn = knn_proba(m.bank, m.bank_y, l2n(feats), 5, len(m.classes))
    out = dict(feats=feats.cpu().numpy(), logits=logits.cpu().numpy(), knn=knn.cpu().numpy())
    np.savez(f, crop_path=np.array(d.crop_path), **out)
    return out["feats"], out["logits"], out["knn"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = G.prepare()
    te, is_cal, y, g, P, classes = D["te"], D["is_cal"], D["y"], D["g"], D["P"], D["classes"]
    cid = {c: i for i, c in enumerate(classes)}
    _, _, Lt, Kt = A1.model_parts(__import__("model").load_split(), "test")
    P0 = A1.blend(Lt, Kt)                                       # architecture 0: as shipped

    d, n_ext = d5_crops()
    yd, gd = d.cls.map(cid).to_numpy(), d.d5_group.to_numpy()
    fd, Ld, Kd = d5_parts(d)
    p0 = A1.blend(Ld, Kd)
    pA = A1.temper(p0, D["T_prob"])
    print(f"D5 outside test: {len(d)} crops (of the team's {n_ext}), {d.d5_group.nunique()} near-duplicate groups, "
          f"{d.cls.value_counts().to_dict()}")

    # gates on D5, calibrated on D1
    imgs = [Image.open(p).convert("RGB") for p in d.crop_path]
    pq, why, pood, _, _, _ = G.deployed(D, imgs, fd)
    # same gates on the D1 test half, for comparison
    tm = ~is_cal
    imgs1 = [Image.open(p).convert("RGB") for p in te.crop_path[tm]]
    pq1, _, pood1, _, _, _ = G.deployed(D, imgs1, D["ft"][tm])

    pct = lambda v: "-" if pd.isna(v) else f"{v * 100:.1f}%"
    rows, act_rows = [], []
    for a in ALPHAS:
        q0 = A0.lac_threshold(P0[is_cal], y[is_cal], a)
        qA = G.deployed_q(a)
        for arch, p_d5, p_d1, q in [("architecture 0", p0, P0[tm], q0), ("architecture A (CP only)", pA, P[tm], qA)]:
            for setname, pp, yy, gg in [("D1 test half", p_d1, y[tm], g[tm]), ("D5 outside test", p_d5, yd, gd)]:
                S = A0.lac_sets(pp, q)
                hit = S[np.arange(len(yy)), yy]
                r = {"alpha": a, "method": arch, "set": setname, "coverage": hit.mean(),
                     "group coverage": pd.Series(hit).groupby(gg).mean().mean(), "mean set size": S.sum(1).mean(),
                     "single class": (S.sum(1) == 1).mean()}
                for k in ("dent", "missing-head"):
                    m = yy == cid[k]
                    r[f"coverage {k}"] = hit[m].mean()
                rows.append(r)
        for setname, pp, yy, ppq, ppo in [("D1 test half", P[tm], y[tm], pq1, pood1), ("D5 outside test", pA, yd, pq, pood)]:
            act, S = V.actions(ppq, ppo, pp, qA)
            act_rows.append({"alpha": a, "set": setname, **V.summarise(act, S, yy)})
    cov = pd.DataFrame(rows)
    acts = pd.DataFrame(act_rows)

    acc = {"D1 test half": float((P[tm].argmax(1) == y[tm]).mean()), "D5 outside test": float((pA.argmax(1) == yd).mean())}
    conf = {"D1 test half": float(P[tm].max(1).mean()), "D5 outside test": float(pA.max(1).mean())}
    ece = {"D1 test half": A1.ece(P[tm], y[tm]), "D5 outside test": A1.ece(pA, yd)}
    pred5 = pd.crosstab(d.cls, pd.Series([classes[i] for i in pA.argmax(1)], name="predicted"))

    md = ["# Distribution shift: the D5 outside test", "",
          f"{len(d)} D5 crops whose photo is not a copy of a D1 photo (the team's list of {n_ext}; {n_ext - len(d)} "
          "dropped because their crop numbering moved), from "
          f"{d.d5_group.nunique()} near-duplicate groups: {d.cls.value_counts().to_dict()}. D5 only has two of the six "
          "classes. Everything is calibrated on D1 (calibration.json: T on val, thresholds and gates on val + cal half / train) and applied "
          "unchanged. No retraining, nothing refitted on D5.", "",
          "## Model and calibration under shift (prob-T probabilities)", "",
          "| set | accuracy | mean stated confidence | ECE |", "|---|---:|---:|---:|"]
    for s in acc:
        md.append(f"| {s} | {pct(acc[s])} | {pct(conf[s])} | {ece[s]:.3f} |")
    md += ["", "What the model predicts on D5 (rows: true class):", "",
           "| true | " + " | ".join(pred5.columns) + " |", "|---|" + "---:|" * len(pred5.columns)]
    for t, r in pred5.iterrows():
        md.append(f"| {t} | " + " | ".join(str(v) for v in r) + " |")
    md += ["", "## CP coverage: does the D1-calibrated guarantee survive?", "",
           "| alpha | method | set | coverage | group coverage | dent | missing-head | mean set size | single class |",
           "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in cov.iterrows():
        md.append(f"| {r.alpha:.2f} | {r.method} | {r.set} | {pct(r.coverage)} | {pct(r['group coverage'])} | "
                  f"{pct(r['coverage dent'])} | {pct(r['coverage missing-head'])} | {r['mean set size']:.2f} | "
                  f"{pct(r['single class'])} |")
    md += ["", "## Full pipeline (gates + CP) as visor actions", "",
           "| alpha | set | single class right | single class wrong (no warning) | ambiguous | ambiguous holds truth | re-take | re-aim | human attention |",
           "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in acts.iterrows():
        md.append(f"| {r.alpha:.2f} | {r.set} | {pct(r['single class, right'])} | {pct(r['single class, wrong'])} | "
                  f"{pct(r['ambiguous: review'])} | {pct(r['ambiguous holds truth'])} | {pct(r['bad image: re-take'])} | "
                  f"{pct(r[V.ACTIONS[4]])} | {pct(r['human attention'])} |")
    md += ["", f"Quality-gate reasons on D5 (flagged crops): "
                f"{pd.Series(why[pq <= G.arch().alpha_q]).value_counts().to_dict()}", ""]
    (OUT / "README.md").write_text("\n".join(md))
    cov.to_csv(OUT / "coverage.csv", index=False)
    acts.to_csv(OUT / "visor_actions.csv", index=False)
    print("\n".join(md))


if __name__ == "__main__":
    main()
