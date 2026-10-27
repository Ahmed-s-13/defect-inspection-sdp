"""False accept and false reject of the final architecture A, per off-target category (enhanced_res v2).

From 1_main_model_resnet18/ (after calibrate.py --model v2 and archA_offtarget_check.py, whose caches it reuses):
  python enhanced_res/arch_a/experiments/archA_rates.py

False reject = normal crops stopped by a gate. False accept = an off-target crop that passes both gates and is shown
as a defect or an ambiguous set. Two views of the off-target photos:
  deployed   calibration.json as shipped. Its AoN regression was fitted on every benchmark category (fit + test), so
             these numbers are in-sample.
  held out   the same gates, but AoN refitted on the "fit" categories only (people, faces, tools, ladders, vehicles,
             factories, garages, loading docks) and recalibrated on the cal half. The "test" categories (helmets,
             hands, gloves, hangars, airports, runways, repair shops, storage rooms) were never seen.
The test half is not used to choose anything: its numbers come from calibration.json's report-only check.
Writes enhanced_res/arch_a/results/archA/offtarget/rates.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression

import archA_offtarget_check as OC
from arch_a import calibrate as C
from arch_a import runtime as R
from model import MAIN, MODELS, EnhancedRes, load_split

GROUPS = {"people": ["Person", "Human face", "Human hand", "Glove", "Helmet"],
          "objects and vehicles": ["Tool", "Ladder", "Car", "Truck"],
          "hangar / airport / industrial scenes": ["hangar/indoor", "hangar/outdoor", "airport_terminal", "runway",
                                                   "repair_shop", "storage_room", "auto_factory", "engine_room",
                                                   "garage/indoor", "industrial_area", "loading_dock"]}


def main():
    a = R.ArchA(R.CALIBRATIONS["v2"])
    m = EnhancedRes(MODELS["v2"])
    man = pd.read_csv(OC.OFF / "manifest.csv")
    paths = [str(MAIN / p) for p in man.crop_path]
    x = OC.features(OC.ER(m), "bench", paths)
    f, P = x[:, :768], x[:, 768:]
    head = torch.load(MODELS["v2"] / "enhanced_res_head.pt", map_location="cpu")
    bank = head["knn_bank"].float().numpy()
    dist = R.knn_distance(f, bank / np.linalg.norm(bank, axis=1, keepdims=True))
    st = [R.quality_stats(Image.open(p).convert("RGB")) for p in paths]
    st = {s: np.array([d[s] for d in st]) for s in R.STATS}
    pq = R.pvalue(a.q_cal, a.q_groups, R.quality_score(a.ref, st, a.sides)[0])
    p_knn = R.pvalue(a.o_cal, a.o_groups, dist)

    # deployed AoN
    p_aon = R.pvalue(a.lr["cal_scores"], a.o_groups, a.aon_score(f))
    passed_dep = ~((pq <= a.alpha_q) | (np.minimum(1, 2 * np.minimum(p_knn, p_aon)) <= a.alpha_ood))

    # held-out AoN: refit on the fit categories only, recalibrate on the cal half (no test-half crops read)
    crops = load_split("d1_grouped")
    va = crops[crops.split == "val"].reset_index(drop=True)
    te = crops[crops.split == "test"].reset_index(drop=True)
    te["half"] = C.halves(te)
    cal = te[te.half == "cal"].reset_index(drop=True)
    z = np.load(C.CACHE["v2"], allow_pickle=True)
    nv, nc = len(va), len(cal)
    assert list(z["crop_path"][nv:nv + nc]) == list(cal.crop_path)
    f_cal = z["feats"][nv:nv + nc]
    fit = (man.role == "fit").to_numpy()
    X = np.concatenate([bank, f[fit]])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000)
    clf.fit((X - mu) / sd, np.r_[np.zeros(len(bank)), np.ones(fit.sum())])
    score = lambda F: clf.predict_proba((F - mu) / sd)[:, 1]
    p_aon_ho = R.pvalue(score(f_cal), cal.src_group.to_numpy(), score(f))
    passed_ho = ~((pq <= a.alpha_q) | (np.minimum(1, 2 * np.minimum(p_knn, p_aon_ho)) <= a.alpha_ood))

    exp = a.cal["expected_test_half"]
    pct = lambda v: f"{v * 100:.1f}%"
    md = ["# Architecture A: false accept and false reject (enhanced_res v2)", "",
          f"Final gates: quality α {a.alpha_q}, OOD α {a.alpha_ood} (kNN + AoN, Bonferroni), in parallel. "
          "False accept = an off-target crop that passes both gates and is shown as a defect. Each photo gives a "
          "whole-frame centre square and a box-sized patch; rates average the two.", "",
          "## False reject (normal defect crops wrongly stopped)", "",
          "| set | stopped | re-take | re-aim |", "|---|---:|---:|---:|",
          f"| D1 test half, per photo family (report only) | **{pct(exp['gates_false_reject_family'])}** | | |",
          f"| D1 test half, per crop (report only) | {pct(exp['gates_false_reject'])} | "
          f"{pct(exp['actions']['re-take'])} | {pct(exp['actions']['re-aim'])} |"]
    dep = (HERE.parent / "results/archA/offtarget/deployed.md").read_text()
    md += [l.replace("- D5 outside test (normal defects, false reject): ", "| D5 (other photos of normal defects) | ")
             .replace(" stopped (re-take ", " | ").replace(", re-aim ", " | ").rstrip(")") + " |"
           for l in dep.splitlines() if l.startswith("- D5")]
    md += ["", "## False accept (off-target photos shown as a defect)", "",
           "**Held out** = AoN never trained on that category (refitted on the fit categories only). "
           "**Deployed** = the shipped AoN, which trained on every category listed (in-sample, optimistic).", "",
           "| group | category | photos | held out | deployed |", "|---|---|---:|---:|---:|"]
    rows = []
    for g, cats in GROUPS.items():
        for c in cats:
            mm = (man.category == c).to_numpy()
            role = man.role[mm].iloc[0]
            ho = passed_ho[mm].mean() if role == "test" else None
            rows.append((g, c, mm.sum() // 2, role, ho, passed_dep[mm].mean()))
            md.append(f"| {g} | {c} ({role}) | {mm.sum() // 2} | {'-' if ho is None else pct(ho)} | "
                      f"{pct(passed_dep[mm].mean())} |")
    t = pd.DataFrame(rows, columns=["group", "cat", "n", "role", "ho", "dep"])
    md += ["", "### Averages", "", "| | held out | deployed |", "|---|---:|---:|"]
    for g in GROUPS:
        s = t[t.group == g]
        h = s[s.role == "test"]
        md.append(f"| {g} | {pct(h.ho.mean()) if len(h) else '-'} ({len(h)} held-out categories) | {pct(s.dep.mean())} |")
    for lbl, cats in (("humans (person, face, hand)", ["Person", "Human face", "Human hand"]),
                      ("helmets", ["Helmet"]), ("hands and gloves", ["Human hand", "Glove"])):
        s = t[t.cat.isin(cats)]
        h = s[s.role == "test"]
        md.append(f"| {lbl} | {pct(h.ho.mean()) if len(h) else '- (not held out)'} | {pct(s.dep.mean())} |")
    h = t[t.role == "test"]
    md.append(f"| **all held-out categories** | **{pct(h.ho.mean())}** | {pct(h.dep.mean())} |")
    md += ["", "Person and Human face trained the AoN in both versions, so they have no held-out number; Human hand, "
           "Glove and Helmet are the held-out human categories. The user's hard-hat photo: "
           + ("passed" if passed_dep[(man.source == 'user').to_numpy()].any() else "stopped (re-aim)") + "."]
    # bad images (synthetic damage) and the other sets, from the quality alpha comparison (cal half only)
    qa = HERE.parent / "results/archA/quality_alpha/README.md"
    if qa.exists():
        text = qa.read_text()
        fa = [l for l in text[text.index("## False accept"):].splitlines()
              if l.startswith("| ") and not l.startswith("| set ")]
        md += ["", "## False accept: bad images and other sets", "",
               "From [`../quality_alpha/`](../quality_alpha/README.md): synthetic damage of cal-half crops, gates "
               "recalibrated on half the cal-half families and measured on the other half (200 times; the test half is "
               f"not used). Deployed: quality α {a.alpha_q}.", "",
               "| set | quality α 0.10 (before) | quality α 0.05 (deployed) |", "|---|---:|---:|"] + fa
    out = HERE.parent / "results/archA/offtarget/rates.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
