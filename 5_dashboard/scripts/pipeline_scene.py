"""Four test crops and one off-target photo through architecture B, one per visor outcome, for the "Follow a crop through architecture B"
scene on the Model page, and the alpha panel after it.
Writes web/src/data/pipeline.json and web/public/pipeline/ (the five crops).

    python scripts/pipeline_scene.py            # write the scene's data
    python scripts/pipeline_scene.py --list     # print candidate crops for each outcome (to choose PICKS)

The crops come from the half of d1_gc7_fix's test split that architecture B was not calibrated on. For the alpha
panel every (OOD alpha, CP alpha) pair is refitted the way enhanced_res/arch_a/calibrate.py does it: the OOD
p-values of the cal-half crops, the CP threshold on the cal crops that pass the gate; then measured on the test half.
The served pair must reproduce calibration_gc7.json (its q and expected_test_half), which this script checks.
"""
import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

DASH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DASH))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from server.paths import REPO_MAIN  # noqa: E402
from server.registry import registry  # noqa: E402
from server.report import _features, _load_split  # noqa: E402
from enhanced_res.arch_a.runtime import family_threshold, pvalue, temper  # noqa: E402

OUT_PUB, OUT_JSON = DASH / "web/public/pipeline", DASH / "web/src/data/pipeline.json"
OOD_ALPHAS, CP_ALPHAS = [0.02, 0.05, 0.08, 0.10, 0.15], [0.05, 0.10, 0.20]
OFF_FEATS = REPO_MAIN / "data/work/enhanced_res/arch_a_offtarget_feats_gc7.npz"
OFF_MANIFEST = REPO_MAIN / "data/work/enhanced_res/offtarget/manifest.csv"
OUTCOMES = ["show", "ambiguous", "possible", "nothing", "reaim"]
# outcome -> crop file name, chosen by eye from --list: a clear example of each, from four different photo groups,
# the true class in the set (names, not paths: corrected D1 crops live in data/work_fix or in
# d1_grouped/label_corrections/crops, depending on the computer)
PICKS = {
    "show": "content__Aircraft_dataset__images__test__aircraft_000969__0.jpg",       # a crack through a hole
    "ambiguous": "content__Aircraft_dataset__images__test__aircraft_008665__0.jpg",  # paint peel-off or corrosion
    "possible": "content__Aircraft_dataset__images__train__aircraft_012430__1.jpg",  # a faint dent
    "nothing": "test__aircraft_003982__c0.jpg",                                      # plain, textured skin
}
# re-aim: not a D1 crop (every D1 crop is aircraft skin) but an off-target photo from the team's off-target benchmark
# (enhanced_res/arch_a/experiments/offtarget_data.py, local only). An Open Images photo of a glove, CC BY 2.0: it is
# copied into web/public/pipeline/ with its credit, which the page shows. Author from Open Images'
# validation-images-with-rotation.csv (data/src/offtarget/oi/).
OFF_PICK = {
    "file": "data/work/enhanced_res/offtarget/crops/box/openimages/0beaf0e11475b0af.jpg", "about": "a glove",
    "credit": {"author": "zeevveez", "url": "https://www.flickr.com/photos/zeevveez/8369251023",
               "license": "CC BY 2.0", "license_url": "https://creativecommons.org/licenses/by/2.0/", "source": "Open Images V7"},
}


def nonempty(Pt, q):
    """The CP sets at threshold q; a set that would be empty keeps its top class, as enhanced_res/arch_a/runtime.py does."""
    S = Pt >= 1 - q
    S[np.arange(len(Pt)), Pt.argmax(1)] |= ~S.any(1)
    return S


def actions(po, S, a_ood, nd):
    """Architecture B's visor action per crop, as enhanced_res/arch_a/runtime.py decides it (no miss guard)."""
    n, has_nd = S.sum(1), S[:, nd]
    act = np.where(n == 0, "unclear", np.where(has_nd, np.where(n == 1, "nothing", "possible"),
                                                np.where(n == 1, "show", "ambiguous")))
    return np.where(po <= a_ood, "reaim", act)


def metrics(act, S, y, g, nd, classes):
    hit = S[np.arange(len(y)), y]
    _, inv = np.unique(g, return_inverse=True)
    right = hit & (S.sum(1) == 1)
    defect, clean = y != nd, y == nd
    return {
        "coverage": float(hit.mean()),
        "family_coverage": float((np.bincount(inv, hit) / np.bincount(inv)).mean()),
        "set_size": float(S.sum(1).mean()),
        "mix": {k: float((act == k).mean()) for k in OUTCOMES + ["unclear"]},
        "silent_miss": float((act[defect] == "nothing").mean()),
        "clean_silent": float((act[clean] == "nothing").mean()),
        "reaim": float((act == "reaim").mean()),
        "wrong_no_warning": float(((act == "show") & ~right).mean()),
        "attention": float(1 - np.isin(act, ["show", "nothing"]).mean()),
        "class_coverage": {c: float(hit[y == k].mean()) for k, c in enumerate(classes)},
    }


def r3(v):
    return float(round(float(v), 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print candidate crops for each outcome")
    args = ap.parse_args()

    entry, adapter = registry.entry(), registry.active()
    arch, classes = adapter.arch, adapter.classes
    nd = classes.index("no-defect")
    split = _load_split(entry.split)
    rows = np.flatnonzero(((split.split == "val") | (split.split == "test")).values)   # the report's rows (its cache)
    P, D, F = adapter.cached(_features(entry, adapter, split, rows))
    sub = split.iloc[rows].reset_index(drop=True)
    half = dict(pd.read_csv(adapter.calibration.parent / "cal_split.csv")[["src_group", "split"]].values)
    sub["half"] = np.where(sub.split == "test", sub.src_group.map(half), "val")
    y = np.array([classes.index(c) for c in sub.cls])
    g = sub.src_group.to_numpy()

    # the two OOD parts, against the calibration scores in calibration_gc7.json
    aon = arch.aon_score(F)
    p_knn, p_aon = pvalue(arch.o_cal, arch.o_groups, D), pvalue(arch.lr["cal_scores"], arch.o_groups, aon)
    po = np.minimum(1, 2 * np.minimum(p_knn, p_aon))
    Pt = temper(P, arch.T)
    cal, tst = (sub.half == "cal").to_numpy(), (sub.half == "test").to_numpy()

    # every (OOD alpha, CP alpha): refit q on the cal crops that pass, measure on the test half
    grid, buckets = {}, {}
    for ao in OOD_ALPHAS:
        passed = cal & (po > ao)
        for ac in CP_ALPHAS:
            q = family_threshold(Pt[passed], y[passed], g[passed], ac)
            S = nonempty(Pt, q)
            act = actions(po, S, ao, nd)
            key = f"{ao:.2f}|{ac:.2f}"
            grid[key] = {"q": q, **metrics(act[tst], S[tst], y[tst], g[tst], nd, classes)}
            buckets[key] = act
    served = f"{arch.alpha_ood:.2f}|{arch.alpha:.2f}"
    exp = arch.cal["expected_test_half"]
    s = grid[served]
    assert abs(s["q"] - arch.q) < 2e-3, (s["q"], arch.q)
    assert abs(s["coverage"] - exp["cp_coverage"]) < 3e-3 and abs(s["reaim"] - exp["actions"]["re-aim"]) < 3e-3, s
    assert abs(s["silent_miss"] - exp["silent_miss"]) < 2e-3 and abs(s["set_size"] - exp["cp_mean_set_size"]) < 1e-2, s
    print(f"served cell reproduces calibration_gc7.json: q {s['q']:.4f}, coverage {s['coverage']:.4f}, "
          f"re-aim {s['reaim']:.4f}, silent miss {s['silent_miss']:.4f}")

    S0 = nonempty(Pt, s["q"])
    act0 = buckets[served]
    rel = lambda p: str(Path(p).resolve().relative_to(REPO_MAIN.resolve()))  # noqa: E731
    if args.list:
        moves = np.array(["/".join(buckets[k][i] for k in grid) for i in range(len(sub))])
        for o in OUTCOMES:
            idx = np.flatnonzero(tst & (act0 == o))
            right = S0[idx, y[idx]]
            # comfortable examples first: the right class in the set, far from the gate (or deep in it, for re-aim)
            key = -po[idx] if o != "reaim" else po[idx]
            idx = idx[np.lexsort((key, ~right))]
            print(f"\n== {o}: {len(np.flatnonzero(tst & (act0 == o)))} test crops")
            for i in idx[:14]:
                st = [classes[k] for k in np.argsort(-Pt[i]) if S0[i, k]]
                changes = len(set(moves[i].split("/")))
                print(f"  {rel(sub.crop_path[i])}  true {sub.cls[i]:15s} set {st}  p_ood {po[i]:.3f}  "
                      f"grp {g[i]}  outcomes over grid {changes}")
        return

    assert all(PICKS.values()), "choose PICKS first (python scripts/pipeline_scene.py --list)"
    OUT_PUB.mkdir(parents=True, exist_ok=True)
    rel_paths = sub.crop_path.map(lambda p: Path(p).name)
    crops = []
    for o, path in PICKS.items():
        hits = np.flatnonzero((rel_paths == path).to_numpy() & tst)
        assert len(hits) == 1, (o, path)
        i = hits[0]
        assert act0[i] == o, (o, path, act0[i])
        assert S0[i, y[i]], (o, path, "the true class is not in the set")
        shutil.copy(sub.crop_path[i], OUT_PUB / f"{o}.jpg")
        crops.append({
            "outcome": o, "image": f"/pipeline/{o}.jpg", "truth": sub.cls[i], "source": sub.source[i] if "source" in sub else "d1",
            "raw": [r3(v) for v in P[i]], "tempered": [r3(v) for v in Pt[i]],
            "knn_dist": r3(D[i]), "aon_score": r3(aon[i]), "p_knn": r3(p_knn[i]), "p_aon": r3(p_aon[i]), "p_ood": r3(po[i]),
            "set": [classes[k] for k in np.argsort(-Pt[i]) if S0[i, k]],
            "outcomes": {k: str(buckets[k][i]) for k in grid},
        })
    assert len({g[np.flatnonzero((rel_paths == p).to_numpy() & tst)[0]] for p in PICKS.values()}) == len(PICKS), \
        "the D1 crops should come from different photo groups"

    # the off-target photo: the same stages, from the off-target benchmark's cached 8-view outputs
    off = np.load(OFF_FEATS, allow_pickle=True)
    j = int(np.flatnonzero(off["crop_path"] == str(REPO_MAIN / OFF_PICK["file"]))[0])
    fo, Po, Do = off["feats"][j:j + 1], off["probs"][j:j + 1], off["knn_dist"][j:j + 1]
    ao_ = arch.aon_score(fo)
    pk, pa = pvalue(arch.o_cal, arch.o_groups, Do), pvalue(arch.lr["cal_scores"], arch.o_groups, ao_)
    po_ = np.minimum(1, 2 * np.minimum(pk, pa))
    Pto = temper(Po, arch.T)
    outs = {k: str(actions(po_, nonempty(Pto, v["q"]), float(k.split("|")[0]), nd)[0]) for k, v in grid.items()}
    assert outs[served] == "reaim", outs[served]
    shutil.copy(REPO_MAIN / OFF_PICK["file"], OUT_PUB / "reaim.jpg")
    crops.append({
        "outcome": "reaim", "image": "/pipeline/reaim.jpg", "truth": "off-target", "about": OFF_PICK["about"],
        "credit": OFF_PICK["credit"], "raw": [r3(v) for v in Po[0]], "tempered": [r3(v) for v in Pto[0]],
        "knn_dist": r3(Do[0]), "aon_score": r3(ao_[0]), "p_knn": r3(pk[0]), "p_aon": r3(pa[0]), "p_ood": r3(po_[0]),
        "set": [classes[k] for k in np.argsort(-Pto[0]) if nonempty(Pto, s["q"])[0, k]], "outcomes": outs,
    })

    # where a part alone stops a crop: its p-value reaches alpha_ood / 2 (Bonferroni over the two parts)
    def cut(cal_scores, a):
        xs = np.sort(np.asarray(cal_scores))
        ps = pvalue(cal_scores, arch.o_groups, xs)
        return float(xs[np.argmax(ps <= a)]) if (ps <= a).any() else float(xs[-1])

    # kNN: the calibration crops' distances
    o_cal = np.asarray(arch.o_cal)
    hi = float(np.quantile(o_cal, 0.995))
    counts, edges = np.histogram(np.clip(o_cal, None, hi), bins=30, range=(float(o_cal.min()), hi))
    knn = {"edges": [r3(e) for e in edges], "counts": counts.tolist(), "n": len(o_cal),
           "cut": {f"{a:.2f}": r3(cut(o_cal, a / 2)) for a in OOD_ALPHAS}}

    # AoN: aircraft-skin crops (the model's train bank) against the off-target crops it was fitted on, on the AoN
    # score's own axis (logit) and the features' main other direction
    rng = np.random.default_rng(0)
    train = split[split.split == "train"]
    tr = train.groupby("cls", group_keys=False).apply(lambda d: d.sample(min(len(d), 50), random_state=0))
    tr = pd.concat([tr, train[train.cls == "no-defect"].sample(100, random_state=1)]).drop_duplicates("crop_path")
    _, _, Ftr = adapter.cached(adapter.raw_features(tr.crop_path.values))
    man = pd.read_csv(OFF_MANIFEST)
    keep = man[man.source.isin(["openimages", "places365"])].crop_path.map(lambda p: str(REPO_MAIN / p))
    oi = np.flatnonzero(np.isin(off["crop_path"], keep.to_numpy()))
    oi = rng.choice(oi, min(300, len(oi)), replace=False)
    Foff = off["feats"][oi]
    lr = arch.lr
    w = lr["coef"] / np.linalg.norm(lr["coef"])
    zs = lambda X: (np.asarray(X, float) - lr["mu"]) / lr["sd"]   # noqa: E731
    logit = lambda X: zs(X) @ lr["coef"] + lr["intercept"]        # noqa: E731
    Z = np.vstack([zs(Ftr), zs(Foff)])
    R = Z - np.outer(Z @ w, w)                                      # remove the AoN direction, keep the rest
    pc = np.linalg.svd(R - R.mean(0), full_matrices=False)[2][0]
    yax = lambda X: (zs(X) - np.outer(zs(X) @ w, w) - R.mean(0)) @ pc   # noqa: E731
    lg = lambda s_: float(np.log(s_ / (1 - s_)))                    # noqa: E731
    pick_rows = [np.flatnonzero((rel_paths == p).to_numpy() & tst)[0] for p in PICKS.values()]
    aon_plot = {
        "skin": [[r3(a), r3(b)] for a, b in zip(logit(Ftr), yax(Ftr))],
        "off": [[r3(a), r3(b)] for a, b in zip(logit(Foff), yax(Foff))],
        "crops": [[r3(a), r3(b)] for a, b in zip(logit(np.vstack([F[pick_rows], fo])), yax(np.vstack([F[pick_rows], fo])))],
        "cut": {f"{a:.2f}": r3(lg(cut(lr["cal_scores"], a / 2))) for a in OOD_ALPHAS},
    }

    out = {
        "made": date.today().isoformat(), "classes": classes, "T": r3(arch.T),
        "served": {"ood": arch.alpha_ood, "cp": arch.alpha}, "ood_alphas": OOD_ALPHAS, "cp_alphas": CP_ALPHAS,
        "test_half": {"n": int(tst.sum()), "families": int(len(set(g[tst])))},
        "crops": crops, "grid": {k: {kk: (r3(vv) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in grid.items()},
        "knn": knn, "aon": aon_plot,
    }
    for v in out["grid"].values():
        v["mix"] = {k: r3(x) for k, x in v["mix"].items()}
        v["class_coverage"] = {k: r3(x) for k, x in v["class_coverage"].items()}
    OUT_JSON.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT_JSON.relative_to(DASH)} ({OUT_JSON.stat().st_size // 1024} kB) and {len(crops)} crops")
    for c in crops:
        moved = sorted({v for v in c["outcomes"].values()})
        print(f"  {c['outcome']:9s} {c['truth']:15s} set {c['set']}  p_ood {c['p_ood']:.3f}  over the grid: {moved}")


if __name__ == "__main__":
    main()
