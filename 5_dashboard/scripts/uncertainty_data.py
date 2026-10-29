"""Data for the Uncertainty page (coverage by class, the aircraft-or-not chart) and for the dataset
sections at the end of the Data page (every dataset's size, where it was used, and samples with architecture B's
answer). Writes web/src/data/uncertainty.json, web/src/data/datasets.json and the sample crops in
web/public/datasets/.

    python scripts/uncertainty_data.py

Needs the D1, no-defect, D5 and FGVC crops and the off-target benchmark (enhanced_res/arch_a/experiments/
offtarget_data.py) with its cached features; Open Images authors come from validation-images-with-rotation.csv in
data/src/offtarget/oi/.
"""
import html
import json
import shutil
import sys
from datetime import date
from pathlib import Path

DASH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DASH))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from server.paths import REPO_MAIN  # noqa: E402
from server.preprocess import to_batch  # noqa: E402
from server.registry import registry  # noqa: E402
from server.report import _features, _load_split  # noqa: E402
from server.states import describe  # noqa: E402
from enhanced_res.arch_a.runtime import pvalue, temper  # noqa: E402

sys.path.insert(0, str(REPO_MAIN / "enhanced_res/arch_a/experiments"))
OUT_D = DASH / "web/public/datasets"
J_U, J_D = DASH / "web/src/data/uncertainty.json", DASH / "web/src/data/datasets.json"
OFF = REPO_MAIN / "data/work/enhanced_res/offtarget"
OFF_FEATS = REPO_MAIN / "data/work/enhanced_res/arch_a_offtarget_feats_gc7.npz"
OI_META = REPO_MAIN / "data/src/offtarget/oi/validation-images-with-rotation.csv"
rng = np.random.default_rng(0)
r4 = lambda v: float(round(float(v), 4))  # noqa: E731


def answer_of(a, imgs):
    """Architecture B's answer for each PIL crop, as the Inspect page shows it."""
    probs, decs = a.predict(to_batch(imgs), imgs)
    out = []
    for p, d in zip(probs, decs):
        r = describe(a.classes, list(map(float, p)), d, a.arch)
        out.append({"state": r["state"], "answer": r["answer"], "top": r["top"], "confidence": r4(r["confidence"])})
    return out


def save(img, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").resize((224, 224)).save(folder / f"{name}.jpg", quality=88)
    return f"/{folder.name}/{name}.jpg"


def main():
    entry, a = registry.entry(), registry.active()
    arch, classes = a.arch, a.classes
    nd = classes.index("no-defect")
    for d in (OUT_D, DASH / "web/public/uncertainty"):   # uncertainty/: example crops, no longer used
        if d.exists():
            shutil.rmtree(d)

    # ------------------------------------------------------------ the test half, as served
    split = _load_split(entry.split)
    rows = np.flatnonzero(((split.split == "val") | (split.split == "test")).values)
    P, D, F = a.cached(_features(entry, a, split, rows))
    sub = split.iloc[rows].reset_index(drop=True)
    half = dict(pd.read_csv(a.calibration.parent / "cal_split.csv")[["src_group", "split"]].values)
    tst = ((sub.split == "test") & (sub.src_group.map(half) == "test")).to_numpy()
    y = np.array([classes.index(c) for c in sub.cls])
    g = sub.src_group.to_numpy()
    po = np.minimum(1, 2 * np.minimum(pvalue(arch.o_cal, arch.o_groups, D), pvalue(arch.lr["cal_scores"], arch.o_groups, arch.aon_score(F))))
    Pt, S = arch.sets(P)
    passed = po > arch.alpha_ood
    hit = S[np.arange(len(y)), y]
    alone = S.sum(1) == 1

    by_class = []
    for k, c in enumerate(classes):
        t = tst & (y == k)
        tp = t & passed
        shown = tst & passed & alone & S[:, k]
        row = {"cls": c, "n": int(t.sum()), "coverage": r4(hit[t].mean()), "coverage_passed": r4(hit[tp].mean()),
               "in_set_right": r4((y[tst & passed & S[:, k]] == k).mean()), "alone_right": r4((y[shown] == k).mean()),
               "alone_n": int(shown.sum()), "groups": int(len(set(g[t])))}
        by_class.append(row)
    overall = {"n": int(tst.sum()), "groups": int(len(set(g[tst]))), "coverage": r4(hit[tst].mean()),
               "coverage_passed": r4(hit[tst & passed].mean()), "single": r4(alone[tst & passed].mean()),
               "set_size": r4(S[tst].sum(1).mean()),
               "single_right": r4(hit[tst & passed & alone].mean()), "multi_right": r4(hit[tst & passed & ~alone].mean())}

    # ------------------------------------------------------------ the aircraft-or-not chart
    lr = arch.lr
    w = lr["coef"] / np.linalg.norm(lr["coef"])
    zs = lambda X: (np.asarray(X, float) - lr["mu"]) / lr["sd"]   # noqa: E731
    logit = lambda X: zs(X) @ lr["coef"] + lr["intercept"]        # noqa: E731
    train = split[split.split == "train"]
    tr = train.sample(frac=1, random_state=0).groupby("cls").head(140)
    _, _, Ftr = a.cached(a.raw_features(tr.crop_path.values))
    off = np.load(OFF_FEATS, allow_pickle=True)
    man = pd.read_csv(OFF / "manifest.csv")
    man["full"] = [str(REPO_MAIN / p) for p in man.crop_path]
    pos = {p: i for i, p in enumerate(off["crop_path"])}
    aon_set = man[man.source.isin(["openimages", "places365"]) & man.full.isin(pos)]
    aon_set = aon_set.sample(frac=1, random_state=0).groupby("category").head(60)
    Foff = off["feats"][[pos[p] for p in aon_set.full]]
    Z = np.vstack([zs(Ftr), zs(Foff)])
    R = Z - np.outer(Z @ w, w)
    mean = R.mean(0)
    pc = np.linalg.svd(R - mean, full_matrices=False)[2][0]
    yax = lambda X: (zs(X) - np.outer(zs(X) @ w, w) - mean) @ pc   # noqa: E731
    cats = sorted(aon_set.category.unique())
    xs = np.sort(np.asarray(lr["cal_scores"]))
    ps = pvalue(lr["cal_scores"], arch.o_groups, xs)
    stop = float(xs[np.argmax(ps <= arch.alpha_ood / 2)])        # the AoN part alone stops a crop past here
    pipe = json.loads((DASH / "web/src/data/pipeline.json").read_text())
    aon = {
        "skin": [[r4(u), r4(v), classes.index(c)] for u, v, c in zip(logit(Ftr), yax(Ftr), tr.cls)],
        "off": [[r4(u), r4(v), cats.index(c), 1 if r == "test" else 0] for u, v, c, r in zip(logit(Foff), yax(Foff), aon_set.category, aon_set.role)],
        "categories": cats, "classes": classes, "stop_logit": r4(np.log(stop / (1 - stop))),
        "pipeline": [{"outcome": c["outcome"], "image": c["image"], "xy": xy} for c, xy in zip(pipe["crops"], pipe["aon"]["crops"])],
        "n_pos": int(len(train)), "n_neg": int(man[man.source.isin(["openimages", "places365"]) & man.role.isin(["fit", "test"])].shape[0]),
    }
    gates = json.loads((REPO_MAIN / "enhanced_res/arch_a/results/archB/gates.json").read_text())

    J_U.write_text(json.dumps({"made": date.today().isoformat(), "overall": overall, "by_class": by_class,
                               "aon": aon, "gates": gates["sets"]}, separators=(",", ":")))

    # ------------------------------------------------------------ the datasets
    oi_meta = pd.read_csv(OI_META, usecols=["ImageID", "Author", "OriginalLandingURL", "License"]).set_index("ImageID")

    def pick(paths, n, credit=None):
        paths = list(paths)
        idx = rng.choice(len(paths), min(n, len(paths)), replace=False)
        return [paths[i] for i in idx]

    def samples(name, paths, labels, credits=None):
        imgs = [Image.open(p).convert("RGB") for p in paths]
        outs = answer_of(a, imgs)
        res = []
        for j, (im, o, lab) in enumerate(zip(imgs, outs, labels)):
            slug = "".join(ch if ch.isalnum() else "-" for ch in str(lab).lower()).strip("-")
            s = {"image": save(im, OUT_D, f"{name}-{j}-{slug}"), "label": lab, **o}   # named by content: no stale pairs
            if credits and credits[j]:
                s["credit"] = credits[j]
            res.append(s)
        return res

    d1 = pd.read_csv(REPO_MAIN / "data/work/crops/D1.csv")
    test_d1 = sub[tst & (sub.cls != "no-defect")]
    nd_csv = pd.read_csv(REPO_MAIN / "d1_gc7/no_defect.csv")
    extra = pd.read_csv(REPO_MAIN / "d1_gc7/extra_eval.csv")
    d5 = pd.read_csv(REPO_MAIN / "data/work/crops/D5.csv")
    d5 = d5[d5.crop_path.map(lambda p: (REPO_MAIN / p).exists())]
    fg = man[man.source == "fgvc-aircraft"]
    splits = split[split.cls != "no-defect"].groupby("split").size().to_dict()

    bench = np.load(REPO_MAIN / "data/work/enhanced_res/archB/gc7_bench.npz", allow_pickle=True)   # archB_report.py's cache
    bpos = {str(p): i for i, p in enumerate(bench["crop_path"])}

    def outcome_of(paths, n=300):
        """Share of each visor action over a dataset: from the cached outputs, else the model on up to n crops."""
        paths = [str(p) for p in paths]
        if all(p in bpos for p in paths):
            ii = [bpos[p] for p in paths]
            Pb, Db, Fb, k = bench["probs"][ii], bench["knn_dist"][ii], bench["feats"][ii], len(ii)
        else:
            paths = list(rng.choice(paths, min(n, len(paths)), replace=False))
            Pb, Db, Fb = a.cached(a.raw_features(np.array(paths)))
            k = len(paths)
        dec = arch.decide(Pb, knn_dist=Db, feats=Fb)
        return pd.Series([d.action for d in dec]).value_counts(normalize=True), k

    act, on = outcome_of(fg.full)
    fg_out = {k: r4(v) for k, v in act.items()}
    pd1 = test_d1.sample(frac=1, random_state=1).groupby("cls").head(1)
    heal = nd_csv[(nd_csv.source == "healed") & (nd_csv.split == "test")].sample(3, random_state=0)
    clean = nd_csv[(nd_csv.source == "clean_cropper") & (nd_csv.split == "test")].sample(3, random_state=0)
    d5s = d5.sample(frac=1, random_state=0).groupby("cls").head(2)
    fgs = fg.sample(frac=1, random_state=0).groupby("category").head(3)
    aircraft = [
        {"name": "D1 (Kaggle aircraft damage)", "id": "d1", "size": f"{14760:,} photos, {len(d1):,} defect crops",
         "detail": f"6 defect classes; grouped into a leak-free split: {splits.get('train', 0):,} train / {splits.get('val', 0):,} val / {splits.get('test', 0):,} test crops",
         "used": "Train, validation, calibration and test (the model and architecture B)",
         "samples": samples("d1", pd1.crop_path, pd1.cls.tolist())},
        {"name": "No-defect crops (healer + clean cropper)", "id": "nd",
         "size": f"{len(nd_csv):,} crops + {len(extra):,} extra", "detail": "Undamaged skin from D1's own photos: defects painted out (healed) or clean areas cut out",
         "used": "Train, validation, calibration and test (the 7th class); the extra 760 test only",
         "samples": samples("nd", [str(REPO_MAIN / p) for p in list(heal.crop_path) + list(clean.crop_path)], ["no-defect (healed)"] * 3 + ["no-defect (clean)"] * 3)},
        {"name": "D5 (Roboflow aircraft damage)", "id": "d5", "size": f"{len(d5):,} crops",
         "detail": "Dent, rupture and fastener damage photographed by others", "used": "Outside test only (never trained or calibrated on)",
         "samples": samples("d5", [str(REPO_MAIN / p) for p in d5s.crop_path], d5s.cls.tolist())},
        {"name": "FGVC-Aircraft (D6)", "id": "fgvc", "size": f"{len(fg):,} crops used ({(fg.category.str.contains('patch')).sum():,} skin patches, {(fg.category.str.contains('whole')).sum():,} whole aircraft)",
         "detail": "Undamaged aircraft photographed at a distance", "used": "Test only: what B does with clean aircraft from another camera",
         "outcome": fg_out, "outcome_n": int(on),
         "samples": samples("fgvc", fgs.full, fgs.category.tolist())},
    ]

    def oi_credit(p):
        iid = Path(p).stem
        if iid in oi_meta.index:
            r = oi_meta.loc[iid]
            return {"author": html.unescape(str(r.Author)), "url": str(r.OriginalLandingURL), "license": "CC BY 2.0"}
        return None

    nonair = []
    for src, title, lic, used in [
        ("openimages", "Open Images V7 (Google)", "CC BY 2.0 per photo", "Trains the aircraft-or-not check; some categories were also left out once, to test it on kinds of photo it had not seen"),
        ("places365", "Places365 (MIT CSAIL)", "research / non-commercial", "Trains the aircraft-or-not check; some scenes (hangars, runways, terminals) were left out once, to test it"),
        ("dtd", "Describable Textures (DTD)", "research", "Tried as extra training photos on 28 Oct and dropped: the check got worse at stopping kinds of photo it had not seen"),
        ("places365-plain", "Places365, plain surfaces", "research / non-commercial", "Tried with DTD and left out"),
    ]:
        m = man[man.source == src]
        if src == "openimages":
            m = m[m.full.map(lambda p: Path(p).stem in oi_meta.index)]
        cats = m.groupby("category").size()
        s = m.sample(frac=1, random_state=0).groupby("category").head(1).sample(min(6, len(cats)), random_state=0)
        if src == "openimages":   # the glove: a plain black glove on white (also the pipeline's re-aim crop)
            s = pd.concat([s[s.category != "Glove"], m[m.full.str.endswith("box/openimages/0beaf0e11475b0af.jpg")]]).tail(6).tail(6)
        cr = [oi_credit(p) for p in s.full] if src == "openimages" else None
        allm = man[man.source == src]
        act, on = outcome_of(allm.full)
        nonair.append({"name": title, "id": src, "licence": lic, "size": f"{allm.image.nunique():,} photos, {len(allm):,} crops",
                       "categories": int(allm.category.nunique()), "fit": int((allm.role == "fit").sum()), "test": int((allm.role == "test").sum()),
                       "used": used, "outcome": {k: r4(v) for k, v in act.items()}, "outcome_n": int(on), "samples": samples(src, s.full, s.category.tolist(), cr)})
    import archA_gates as G   # noqa: E402
    import archA_static as ST  # noqa: E402
    wall = G.offtarget_patches(6, np.random.default_rng(0))
    kinds = [kv for kv in ST.nonsense(np.random.default_rng(0)).items() if kv[0] != "grey static"][:6]
    stat = [ims[0] for _, ims in kinds]
    def gen_outcome(imgs):
        acts = []
        for i in range(0, len(imgs), 64):
            ch = imgs[i:i + 64]
            acts += [d.action for d in a.predict(to_batch(ch), ch)[1]]
        return {k: r4(v) for k, v in pd.Series(acts).value_counts(normalize=True).items()}, len(imgs)
    wall_all = G.offtarget_patches(G.N_OFF, np.random.default_rng(0))
    stat_all = [im for ims in ST.nonsense(np.random.default_rng(0)).values() for im in ims]
    w_out, w_n = gen_outcome(wall_all)
    s_out, s_n = gen_outcome(stat_all)
    nonair.append({"name": "Wallpaper patches (macOS)", "outcome": w_out, "outcome_n": w_n, "id": "wall", "licence": "Apple, local copies", "size": f"{G.N_OFF} patches",
                   "used": "Test only: patches of non-aircraft photos", "samples": [
                       {"image": save(im, OUT_D, f"wall-{j}"), "label": "wallpaper patch", **o} for j, (im, o) in enumerate(zip(wall, answer_of(a, wall)))]})
    nonair.append({"name": "Static, solid colours, gradients", "outcome": s_out, "outcome_n": s_n, "id": "static", "licence": "made in code", "size": "synthetic",
                   "used": "Test only: inputs with no content at all", "samples": [
                       {"image": save(im, OUT_D, f"static-{j}"), "label": k, **o}
                       for j, ((k, _), im, o) in enumerate(zip(kinds, stat, answer_of(a, stat)))]})

    J_D.write_text(json.dumps({"made": date.today().isoformat(), "aircraft": aircraft, "nonaircraft": nonair}, separators=(",", ":")))
    print(f"wrote {J_U.name} ({J_U.stat().st_size // 1024} kB), {J_D.name} ({J_D.stat().st_size // 1024} kB)")
    print("coverage", overall)
    for d in aircraft + nonair:
        print(f"  {d['name']:40s} {d['size']:45s} " + " ".join(s["state"] for s in d["samples"]))


if __name__ == "__main__":
    main()
