"""Compare raw hybrid probabilities with temperature-scaled LAC at alpha 0.10.

Run from 1_main_model_resnet18/: python enhanced_res/arch_a/experiments/archB_raw_cp.py
Uses the existing calibration-output cache, checks row alignment and the deployed
reference, and writes only an experiment report. Does not modify calibration.
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

MAIN = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(MAIN))
from enhanced_res.arch_a import calibrate as C, runtime as R
from enhanced_res.model import load_split

OUT = Path(__file__).resolve().parents[1] / "results" / "archB" / "raw_cp"


def main():
    deployed = R.ArchA(R.CALIBRATIONS["gc7"])
    split = load_split(deployed.cal["data"]["split"])
    val = split[split.split == "val"].reset_index(drop=True)
    test = split[split.split == "test"].reset_index(drop=True)
    half = C.halves(test)
    cal, test = test[half == "cal"], test[half == "test"]
    z = np.load(C.CACHE["gc7"], allow_pickle=True)
    assert list(z["crop_path"]) == list(val.crop_path) + list(cal.crop_path) + list(test.crop_path)
    pc = z["probs"][len(val):len(val) + len(cal)]
    fc = z["feats"][len(val):len(val) + len(cal)]
    dc = z["knn_dist"][len(val):len(val) + len(cal)]
    p, f, d = (z[k][len(val) + len(cal):] for k in ("probs", "feats", "knn_dist"))
    cid = {c: i for i, c in enumerate(deployed.classes)}
    yc, y = cal.cls.map(cid).to_numpy(), test.cls.map(cid).to_numpy()
    gc, g = cal.src_group.to_numpy(), test.src_group.to_numpy()
    passed_cal = deployed.ood(dc, fc)[0] > deployed.alpha_ood
    passed = deployed.ood(d, f)[0] > deployed.alpha_ood
    results, arrays = {}, {}
    for name, temperature in (("temperature_scaled", deployed.T), ("raw_hybrid", 1.0)):
        arch = copy.deepcopy(deployed)
        arch.T = temperature
        arch.q = R.family_threshold(R.temper(pc[passed_cal], temperature), yc[passed_cal], gc[passed_cal], arch.alpha)
        if name == "temperature_scaled":
            assert np.isclose(arch.q, deployed.q, atol=1e-10, rtol=0), (arch.q, deployed.q)
        pt, sets = arch.sets(p)
        hit, size = sets[np.arange(len(y)), y], sets.sum(1)
        dec = arch.decide(p, knn_dist=d, feats=f)
        m = R.summarise(dec, arch, p, y, g)
        if name == "temperature_scaled":
            assert abs(m["cp_coverage"] - deployed.cal["expected_test_half"]["cp_coverage"]) < 1e-8
        m.update(temperature=temperature, q_hat=arch.q, probability_cutoff=1-arch.q)
        for label, mask in (("all", np.ones(len(y), bool)), ("accepted", passed)):
            by_class = {c: float(hit[mask & (y == k)].mean()) for k, c in enumerate(arch.classes)}
            _, inv = np.unique(g[mask], return_inverse=True)
            m[label] = {"n": int(mask.sum()), "coverage": float(hit[mask].mean()),
                        "macro_coverage": float(np.mean(list(by_class.values()))),
                        "family_coverage": float((np.bincount(inv, weights=hit[mask]) / np.bincount(inv)).mean()),
                        "mean_set_size": float(size[mask].mean()), "singleton_rate": float((size[mask] == 1).mean()),
                        "mean_confidence": float(pt[mask].max(1).mean()), "per_class": by_class,
                        "by_set_size": [{"size": k, "n": int((mask & (size == k)).sum()),
                                         "correct": int(hit[mask & (size == k)].sum())} for k in range(8)]}
        results[name] = m
        arrays[name] = {"sets": sets, "hit": hit, "size": size, "actions": np.array([v.action for v in dec])}
    a, b = arrays["temperature_scaled"], arrays["raw_hybrid"]
    # Paired cluster bootstrap: both methods use the same resampled test families.
    _, inv = np.unique(g[passed], return_inverse=True)
    counts = np.bincount(inv)
    diffs = np.bincount(inv, weights=b["hit"][passed].astype(float)-a["hit"][passed])
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(5000):
        idx = rng.integers(0, len(counts), len(counts))
        boot.append(float(diffs[idx].sum()/counts[idx].sum()))
    comparison = {"accepted_sets_changed": int(np.any(a["sets"] != b["sets"], axis=1)[passed].sum()),
                  "accepted_newly_covered": int((passed & ~a["hit"] & b["hit"]).sum()),
                  "accepted_newly_missed": int((passed & a["hit"] & ~b["hit"]).sum()),
                  "visor_actions_changed": int((a["actions"] != b["actions"]).sum()),
                  "raw_minus_scaled_accepted_coverage_ci95": list(map(float, np.quantile(boot, [.025, .975])))}
    report = {"method": "Same gc7 weights, prior correction, OOD gate, alpha 0.10, family weights, and top-class fallback. Refit q separately on OOD-passed calibration half only; T is deployed val-fitted T or 1. Test half is report-only.",
              "calibration_crops": len(cal), "accepted_calibration_crops": int(passed_cal.sum()),
              "calibration_families": int(cal.src_group.nunique()), "test_families": int(test.src_group.nunique()),
              "variants": results, "comparison": comparison}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Raw hybrid vs temperature-scaled LAC", "", report["method"], "",
             f"Calibration: {len(cal)} crops ({passed_cal.sum()} accepted), {cal.src_group.nunique()} families. Evaluation: {len(test)} crops ({passed.sum()} accepted), {test.src_group.nunique()} families.", "",
             "| Metric | Temperature-scaled | Raw hybrid |", "|---|---:|---:|"]
    for title, key in (("Probability cutoff", "probability_cutoff"), ("Silent defect miss", "silent_miss"), ("Clean crop false alarm", "no_defect_false_alarm"), ("Human referral", "human_attention")):
        lines.append(f"| {title} | {results['temperature_scaled'][key]*100:.4f}% | {results['raw_hybrid'][key]*100:.4f}% |")
    for key in ("coverage", "macro_coverage", "family_coverage", "mean_set_size", "singleton_rate", "mean_confidence"):
        x, v = (results[n]["accepted"][key] for n in ("temperature_scaled", "raw_hybrid"))
        lines.append(f"| Accepted: {key} | {x:.6f} | {v:.6f} |")
    lines += ["", "Paired comparison:", "", "```json", json.dumps(comparison, indent=2), "```", "",
              "Intervals resample whole evaluation families, conditional on this calibration. This is one paired evaluation, not evidence for model selection or universal superiority. Deployed calibration is unchanged."]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
