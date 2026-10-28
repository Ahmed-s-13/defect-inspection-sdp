"""The Model-view report, computed from a model's 8-view features on its split (d1_gc7_fix for the main model: d1_gc7
with the 27 Oct D1 box corrections), plus architecture B (the team's calibrated decision layer on gc7, quality gate
off) on the test half it was not calibrated on."""
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from server.paths import CACHE

TARGET_MACRO_F1 = 0.85  # NFR2, SDP Report Part II
REPORT_VERSION = "v6"   # bump when the report's shape or meaning changes, so cached reports are rebuilt
_mem = {}
_lock = threading.Lock()
_splits = {}


class ReportUnavailable(Exception):
    """The data the report needs is not on this computer; the message says which download gives it."""
ND = "no-defect"
V2_SUMMARY = "enhanced_res/results/summary.json"  # the 6-class v2's report, in the repo
FIX_SUMMARY = "enhanced_res/results/gc7_fix/summary.json"  # v2 scored on the corrected (d1_gc7_fix) defect crops


def _load_split(name="d1_grouped"):
    if name not in _splits:
        from enhanced_res.model import load_split
        try:
            _splits[name] = load_split(name)
        except (OSError, AssertionError) as e:
            raise ReportUnavailable(f"The Model page needs the crop lists in 1_main_model_resnet18/data/work/: unzip "
                                    f"main_model_data_work.zip (team Drive) in the repo root ({e})") from e
    return _splits[name]


def crop_path_of(i, split="d1_grouped"):
    s = _load_split(split)
    if not (0 <= i < len(s)):
        return None
    return Path(s.crop_path.iloc[i])


def reliability_bins(conf, correct, n_bins=10):
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            out.append({"lo": round(b / n_bins, 2), "hi": round((b + 1) / n_bins, 2), "n": int(m.sum()),
                        "confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean())})
    return out


def ece(conf, correct, n_bins=15):
    """Expected calibration error: the gap between stated confidence and accuracy, averaged over bins by size.
    15 bins, (lo, hi], as in enhanced_res/arch_a/experiments/archA_cp.py, so the numbers match the team's."""
    edges = np.linspace(0, 1, n_bins + 1)
    return float(sum(m.mean() * abs(conf[m].mean() - correct[m].mean())
                     for m in ((conf > lo) & (conf <= hi) for lo, hi in zip(edges[:-1], edges[1:])) if m.any()))


def silent_errors(decisions, probs_t, y, rows, groups, classes, most=12):
    """Single-class answers that were wrong (no warning shown), most confident first, at most 2 per photo group."""
    from enhanced_res.arch_a.runtime import SHOW
    wrong = [k for k, d in enumerate(decisions) if d.action == SHOW and classes[y[k]] not in d.classes]
    wrong.sort(key=lambda k: -probs_t[k].max())
    out, per = [], {}
    for k in wrong:
        if per.get(groups[k], 0) < 2:
            per[groups[k]] = per.get(groups[k], 0) + 1
            out.append({"id": int(rows[k]), "true": classes[y[k]], "pred": decisions[k].classes[0],
                        "confidence": float(decisions[k].scores[0])})
        if len(out) == most:
            break
    return out


# architecture A's (v2) gate tests, a fallback only (enhanced_res/arch_a/results/archA/): share of each kind of bad input stopped by both gates.
# Architecture B's own results (results/archB/gates.json) are used instead when they are there.
GATE_SETS = [("off-target", "Patches of non-aircraft photos"), ("dark s2", "Very dark"),
             ("glare s2", "Strong glare"), ("overexposed s2", "Overexposed"), ("jpeg s2", "Heavy JPEG artefacts"),
             ("noise s2", "Heavy sensor noise"), ("motion blur s2", "Motion blur"), ("blur s2", "Out of focus")]
STATIC_SETS = ("colour static", "grey static", "gaussian static", "salt & pepper", "blocky static",
               "smooth noise blobs", "solid colour", "gradient")


def gate_tests(root, alpha_ood):
    """Read the team's published gate results at the served alpha; None when the files are missing."""
    import pandas as pd
    b = root.parent / "archB/gates.json"
    if b.exists():
        g = json.loads(b.read_text())
        if abs(g.get("alpha_ood", -1) - alpha_ood) < 1e-9:
            return {"sets": sorted(g["sets"], key=lambda r: -r["stopped"]), "source": g["source"]}
    try:
        ops = pd.read_csv(root / "step2/operating_points.csv", index_col=0)
        rates = pd.read_csv(root / "static/rates.csv")
    except OSError:
        return None
    col = f"both a={alpha_ood:g}"
    if col not in ops.columns:   # the study has no column for this alpha
        return None
    out =[{"set": k, "label": label, "stopped": float(ops.loc[k, col])} for k, label in GATE_SETS if k in ops.index]
    opt = rates[rates.option.str.contains(f"OOD {alpha_ood:.2f}", regex=False)]
    static = opt[opt.set.isin(STATIC_SETS)]["false accept (passes both gates)"]
    if len(static):
        out.insert(1, {"set": "static", "label": "Random static, solid colours, gradients",
                       "stopped": float(1 - static.max())})
    return {"sets": sorted(out, key=lambda r: -r["stopped"]), "source": "enhanced_res/arch_a/results/archA/"}


def arch_a_report(adapter, split, probs, dists, y, feats=None):
    """The decision layer (architecture B for gc7) on the half of the test split it was not calibrated on
    (enhanced_res/arch_a/cal_split.csv), at every calibrated alpha. The quality gate runs on the saved 224 px crops
    only if it is on. With a no-defect class, "clear" is the share shown nothing (the set is only no-defect), and the
    gate is also given per crop source (D1 defects, healed, clean cropper)."""
    import pandas as pd
    from PIL import Image
    from enhanced_res.arch_a.runtime import NOTHING, REAIM, RETAKE, SHOW, ArchA, summarise, temper
    cal = pd.read_csv(adapter.calibration.parent / "cal_split.csv")
    half = dict(zip(cal.src_group, cal.split))  # whole photo families go to one half (crop file numbers can differ)
    rows = np.flatnonzero(((split.split == "test") & (split.src_group.map(half) == "test")).values)
    groups = split.src_group.values[rows]
    crops = [Image.open(split.crop_path.iloc[i]).convert("RGB") for i in rows] if adapter.arch.use_quality else None
    p, d, yt = probs[rows], dists[rows], y[rows]
    f = None if feats is None else feats[rows]
    pt = temper(p, adapter.arch.T)
    by_alpha, default = {}, None
    classes = adapter.classes
    nd = classes.index(ND) if ND in classes else None
    for key in sorted(adapter.arch.cal["cp"]["q"]):
        arch = ArchA(adapter.calibration, alpha=float(key))
        dec = arch.decide(p, crops=crops, knn_dist=d, feats=f)
        by_alpha[key] = summarise(dec, arch, p, yt, groups) | {"set_threshold": float(1 - arch.q)}
        if nd is not None:  # crops shown nothing (no defect), and defects answered that way (a silent miss)
            clear = np.array([x.action == NOTHING or (x.action == SHOW and x.classes == [ND]) for x in dec])
            by_alpha[key] |= {"clear": float(clear.mean()), "defect_called_clear": float(clear[yt != nd].mean()),
                              "clear_right": float(clear[yt == nd].mean())}
        if abs(arch.alpha - adapter.arch.alpha) < 1e-9:
            default = dec
    c = adapter.arch.cal
    correct = p.argmax(1) == yt
    by_source = None
    if nd is not None:
        src = split.source.values[rows]
        act = np.array([x.action for x in default])
        by_source = [{"source": k, "n": int((src == k).sum()), "retake": float((act[src == k] == RETAKE).mean()),
                      "reaim": float((act[src == k] == REAIM).mean())}
                     for k in ("d1", "healed", "clean_cropper") if (src == k).any()]
    return {
        "alpha": adapter.arch.alpha, "temperature": adapter.arch.T,
        "alpha_quality": adapter.arch.alpha_q, "alpha_ood": adapter.arch.alpha_ood,
        "cal": {"crops": c["data"]["cal_crops"], "families": c["data"]["cal_families"], "created": c.get("created")},
        "test_half": {"n": int(len(rows)), "families": int(len(set(groups)))},
        "ece": {"raw": ece(p.max(1), correct), "tempered": ece(pt.max(1), correct)},
        "reliability": {"raw": reliability_bins(p.max(1), correct), "tempered": reliability_bins(pt.max(1), correct)},
        "by_alpha": by_alpha,
        "expected": c.get("expected_test_half"),
        "silent_errors": silent_errors(default, pt, yt, rows, groups, adapter.classes),
        "gates_by_source": by_source,
        # architecture B's own tests (results/archB/gates.json); A's (6-class v2) only as a fallback
        "gate_tests": gate_tests(adapter.calibration.parent / "results/archA", adapter.arch.alpha_ood),
    }


def no_defect_report(split, probs, y, classes, summ):
    """What the 7th class adds, on the test split: clean crops recognised (per source), defects called no-defect,
    and the six-class result when the model's best defect class is taken. The val-chosen p(no-defect) cut, the
    LaMa check and what the 6-class v2 calls clean crops come from the team's gc7 summary."""
    nd = classes.index(ND)
    test = (split.split == "test").values
    pt, yt, src = probs[test], y[test], split.source.values[test]
    top = pt.argmax(1)
    is_nd, is_d = yt == nd, yt != nd
    best = np.delete(pt, nd, axis=1).argmax(1)  # best of the six defect classes (no-defect is the last column)
    d6 = [c for c in classes if c != ND]
    return {
        "n": int(is_nd.sum()), "n_defects": int(is_d.sum()),
        "recall": float((top[is_nd] == nd).mean()),
        "recall_by_source": {k: float((top[is_nd & (src == k)] == nd).mean()) for k in ("healed", "clean_cropper")},
        "n_by_source": {k: int((is_nd & (src == k)).sum()) for k in ("healed", "clean_cropper")},
        "defects_called_clear": float((top[is_d] == nd).mean()),
        "defects_called_clear_by_class": {c: float((top[yt == classes.index(c)] == nd).mean()) for c in d6},
        "defects_best": {"accuracy": float((best[is_d] == yt[is_d]).mean()),
                         "macro_f1": float(f1_score(yt[is_d], best[is_d], average="macro"))},
        "v2_defects": summ.get("test_defects", {}).get("v2"),
        "v2_calls": summ.get("test_no_defect", {}).get("v2_calls"),
        "extra": summ.get("extra_no_defect"), "dial": summ.get("dial"), "decoys": summ.get("decoys"),
        "train": {k: int(((split.split == k) & (split.cls == ND)).sum()) for k in ("train", "val", "test")},
    }


def _key(entry):
    from enhanced_res.arch_a.runtime import CALIBRATION
    files = sorted(entry.weights.glob("*.pt")) + [p for p in (entry.features_cache, entry.summary,
                                                             entry.calibration or CALIBRATION) if p]
    return REPORT_VERSION + "|" + entry.id + "|" + "|".join(f"{p.name}:{p.stat().st_mtime_ns}" for p in files if p.exists())


def _features(entry, adapter, split, rows):
    """The 8-view features of the given rows: from the team's cache (models.yaml features_cache) when it is there,
    else computed once with the model from the saved crops and kept in cache/ (only the rows the report reads)."""
    if entry.features_cache and entry.features_cache.exists() and entry.split == "d1_gc7":   # its rows are d1_gc7's
        return np.load(entry.features_cache)["tta" if adapter.uses_tta else "plain"][rows]
    stamp = "|".join(f"{p.name}:{p.stat().st_size}" for p in sorted(entry.weights.glob("*.pt")))
    f = CACHE / f"feats_{entry.id}_{entry.split}.npz"
    try:
        z = np.load(f)
        if str(z["stamp"]) == stamp and np.array_equal(z["rows"], rows):
            return z["F"]
    except (OSError, KeyError, ValueError):
        pass  # missing or made for other weights: compute
    print(f"report: computing 8-view features of {len(rows):,} validation and test crops (once, a few minutes)", flush=True)
    F = adapter.raw_features(split.crop_path.values[rows])
    CACHE.mkdir(exist_ok=True)
    np.savez(f, F=F, rows=rows, stamp=stamp)
    return F


def build_report(entry, adapter):
    split = _load_split(entry.split)
    classes = adapter.classes
    val, test = (split.split == "val").values, (split.split == "test").values
    rows = np.flatnonzero(val | test)  # the report reads only these; train rows stay zero
    missing = [p for p in split.crop_path.values[rows] if not Path(p).exists()]
    if missing:
        raise ReportUnavailable(f"The Model page needs the validation and test crops ({len(missing):,} missing, e.g. "
                                f"{missing[0]}): unzip main_model_data_work.zip and d1_gc7_no_defect_crops.zip (team "
                                "Drive) in the repo root")
    p_sub, d_sub, f_sub = adapter.cached(_features(entry, adapter, split, rows))
    probs, dists = np.zeros((len(split), len(classes))), np.zeros(len(split))
    feats = np.zeros((len(split), f_sub.shape[1]), np.float32)
    probs[rows], dists[rows], feats[rows] = p_sub, d_sub, f_sub
    y = np.array([classes.index(c) for c in split.cls])
    pred, conf = probs.argmax(1), probs.max(1)
    correct = pred == y

    pr, rc, f1, sup = precision_recall_fscore_support(y[test], pred[test], labels=range(len(classes)), zero_division=0)
    summ = json.loads(entry.summary.read_text()) if entry.summary and entry.summary.exists() else {}
    gc7 = ND in classes
    if gc7:  # gc7's summary; the comparison is the 6-class v2 on the same defect crops
        from server.paths import REPO_MAIN
        fin = summ.get("test", {})
        from enhanced_res.model import group_bootstrap_ci   # on the crops scored here (the published one is d1_gc7's)
        ci = [float(v) for v in group_bootstrap_ci(y[test], pred[test], split.src_group.values[test])]
        v2 = json.loads((REPO_MAIN / V2_SUMMARY).read_text())["test"]["final"]
        if entry.split.endswith("_fix"):   # the 6-class v2 on the same corrected defect crops (gc7_fix's report)
            summ = summ | {"test_defects": json.loads((REPO_MAIN / FIX_SUMMARY).read_text())["test_defects"]}
        base = {"accuracy": summ["test_defects"]["v2"]["accuracy"], "macro_f1": summ["test_defects"]["v2"]["macro_f1"],
                "recall": v2["recall"] | {ND: 0.0}, "label": "the 6-class model (v2)", "defects_only": True}
    else:
        fin, base, ci = summ.get("test", {}).get("final", {}), summ.get("test", {}).get("base"), summ.get("test", {}).get("final_ci")
        base = base and base | {"label": "the plain ResNet-18 + MLP head", "defects_only": False}

    d5 = summ.get("d5")
    return {
        "key": _key(entry), "model": entry.id, "label": entry.label, "description": entry.description,
        "trained_on": entry.trained_on, "computed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classes": classes, "target_macro_f1": TARGET_MACRO_F1,
        "test": {"n": int(test.sum()), "groups": int(split.src_group[test].nunique()),
                 "accuracy": float(correct[test].mean()),
                 "macro_f1": float(f1_score(y[test], pred[test], average="macro")),
                 "balanced_accuracy": float(balanced_accuracy_score(y[test], pred[test])),
                 "ci": ci,
                 "confusion": confusion_matrix(y[test], pred[test], labels=range(len(classes))).tolist(),
                 "per_class": [{"cls": c, "precision": float(pr[k]), "recall": float(rc[k]), "f1": float(f1[k]),
                                "support": int(sup[k])} for k, c in enumerate(classes)]},
        "val": {"n": int(val.sum()), "accuracy": float(correct[val].mean())},
        "baseline": None if not base else {k: base[k] for k in ("accuracy", "macro_f1", "recall", "label", "defects_only")},
        "no_defect": no_defect_report(split, probs, y, classes, summ) if gc7 else None,
        "backbone_epoch": summ.get("backbone_epoch"), "head": summ.get("head"), "val_macro_f1": summ.get("val", {}).get("macro_f1"),
        "published": {"accuracy": fin.get("accuracy"), "macro_f1": fin.get("macro_f1")},
        "arch_a": arch_a_report(adapter, split, probs, dists, y, feats),
        "d5": None if not d5 else {"accuracy": d5["final_acc"], "base_accuracy": d5["base_acc"], "n": d5["n"],
                                    "counts": d5["counts"]},
        "provenance": {"weights": [{"file": p.name, "bytes": p.stat().st_size,
                                    "modified": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).date().isoformat()}
                                   for p in sorted(entry.weights.glob("*.pt"))],
                       "features_cache": str(entry.features_cache) if entry.features_cache.exists() else "computed (cache/)",
                       "summary": str(entry.summary),
                       "split": entry.split},
    }


def get_report(registry, model_id=None):
    entry = registry.entry(model_id)
    if entry.features_cache is None:  # a model with no report (the legacy model)
        return None
    key = _key(entry)
    with _lock:  # the warm-up thread and the first request must not build it twice
        if entry.id in _mem and _mem[entry.id]["key"] == key:
            return _mem[entry.id]
        f = CACHE / f"report_{entry.id}.json"
        try:
            rep = json.loads(f.read_text())
            if rep.get("key") == key:
                _mem[entry.id] = rep
                return rep
        except (OSError, ValueError):
            pass  # missing, half-written or unreadable: rebuild
        rep = build_report(entry, registry.get(entry.id))
        CACHE.mkdir(exist_ok=True)
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rep))
        os.replace(tmp, f)
        _mem[entry.id] = rep
        return rep
