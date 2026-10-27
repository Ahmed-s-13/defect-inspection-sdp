"""The deployed architecture A (calibration.json) on every off-target set, undamaged aircraft, D5 and the user's photo.

From 1_main_model_resnet18/ (after offtarget_data.py):
  python enhanced_res/arch_a/experiments/archA_offtarget_check.py

Note: the deployed AoN regression was fitted on ALL benchmark off-target categories (fit + test), so its
numbers here are in-sample for those categories. Held out from fitting (helmets, hands, gloves, hangars, airports,
runways, storage rooms) it lets 2.3% (whole frame) / 4.0% (box patch) through.
Wallpapers, random static, undamaged aircraft, D5 and the user photo were never used for fitting.
Writes enhanced_res/arch_a/results/archA/offtarget/deployed.md.
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch
from PIL import Image

import archA_cp as A1
import archA_d5 as A5
import archA_gates as G
import archA_static as ST
from arch_a import runtime as R
from klshift.common import eval_transform
from model import MAIN, EnhancedRes

OUT = HERE.parent / "results" / "archA" / "offtarget"
OFF = MAIN / "data/work/enhanced_res/offtarget"
CACHE = A1.CACHE / "bench"


def batches(items, n=64):
    for i in range(0, len(items), n):
        chunk = items[i:i + n]
        yield [Image.open(x).convert("RGB") if isinstance(x, (str, Path)) else x.convert("RGB") for x in chunk]


class ER:
    name = "enhanced_res"

    def __init__(self, m):
        self.m, self.tf = m, eval_transform()

    @torch.no_grad()
    def __call__(self, ims):
        f = self.m.features(torch.stack([self.tf(i) for i in ims]))
        return torch.cat([f, self.m.proba_from_features(f)], 1).cpu().numpy()   # 768 feats + 6 probs


def features(ext, key, items):
    """Cached features of a list of crop paths (in-memory images are not cached)."""
    paths = all(isinstance(x, (str, Path)) for x in items)
    f = CACHE / f"{ext.name}_{key}.npz"
    if paths and f.exists():
        z = np.load(f, allow_pickle=True)
        if list(z["items"]) == [str(x) for x in items]:
            return z["x"]
    t0, out = time.time(), []
    for i, b in enumerate(batches(items)):
        out.append(ext(b))
        print(f"\r  {ext.name} {key}: {min((i + 1) * 64, len(items))}/{len(items)}", end="", flush=True)
    x = np.concatenate(out)
    print(f"  ({(time.time() - t0) / max(len(items), 1) * 1000:.1f} ms/crop)")
    if paths:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(f, items=np.array([str(i) for i in items]), x=x)
    return x


def main():
    a = R.ArchA()
    m = EnhancedRes()
    ext = ER(m)
    head = torch.load(MAIN / "data/work/enhanced_res/enhanced_res_head.pt", map_location="cpu")
    bank = head["knn_bank"].float().numpy()
    bank /= np.linalg.norm(bank, axis=1, keepdims=True)
    man = pd.read_csv(OFF / "manifest.csv")
    man["path"] = [str(MAIN / p) for p in man.crop_path]
    d5, _ = A5.d5_crops()
    D = G.prepare()
    sets = {"bench": list(man.path), "d5": list(d5.crop_path), "wallpapers": D["sets"]["off-target"],
            "static": [im for ims in ST.nonsense(np.random.default_rng(0)).values() for im in ims]}
    act = {}
    for s, items in sets.items():
        x = features(ext, s, items)
        f, p = x[:, :768], x[:, 768:]
        ims = [Image.open(i).convert("RGB") if isinstance(i, str) else i for i in items]
        dec = a.decide(p, crops=ims, knn_dist=R.knn_distance(f, bank), feats=f)
        act[s] = np.array([d.action for d in dec])
        if s == "bench":
            for i in np.flatnonzero((man.source == "user").to_numpy()):
                print(f"helmet photo ({man['mode'][i]}): {dec[i].text()}")
    stopped = lambda s: np.isin(act[s], (R.RETAKE, R.REAIM))
    rows = []
    for (ro, c), grp in man.groupby(["role", "category"]):
        for mode in ("centre", "box"):
            mm = (man.role == ro).to_numpy() & (man.category == c).to_numpy() & (man["mode"] == mode).to_numpy()
            if mm.any():
                rows.append({"role": ro, "category": c, "mode": mode, "crops": int(mm.sum()),
                             "passes (shown as a defect / ambiguous)": 1 - stopped("bench")[mm].mean(),
                             "re-take": (act["bench"][mm] == R.RETAKE).mean(), "re-aim": (act["bench"][mm] == R.REAIM).mean()})
    t = pd.DataFrame(rows)
    pct = lambda v: f"{v * 100:.1f}%"
    md = ["# Deployed architecture A on the off-target benchmark", "",
          f"calibration.json: quality alpha {a.alpha_q} (sharpness {a.sides['sharpness']}), OOD alpha {a.alpha_ood} "
          f"(kNN + AoN, Bonferroni), CP alpha {a.alpha}. The AoN regression saw the fit AND "
          "test categories (in-sample); wallpapers, static, undamaged aircraft, D5 and the user photo it never saw.", "",
          f"- D5 outside test (normal defects, false reject): {pct(stopped('d5').mean())} stopped "
          f"(re-take {pct((act['d5'] == R.RETAKE).mean())}, re-aim {pct((act['d5'] == R.REAIM).mean())})",
          f"- macOS wallpaper patches: {pct(1 - stopped('wallpapers').mean())} pass",
          f"- random static / colours / gradients: {pct(1 - stopped('static').mean())} pass", "",
          "| role | category | mode | crops | passes | re-take | re-aim |", "|---|---|---|---:|---:|---:|---:|"]
    for _, r in t.iterrows():
        md.append(f"| {r['role']} | {r['category']} | {r['mode']} | {r['crops']} | "
                  f"{pct(r['passes (shown as a defect / ambiguous)'])} | {pct(r['re-take'])} | {pct(r['re-aim'])} |")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "deployed.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
