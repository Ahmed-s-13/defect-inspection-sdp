"""Architecture A, step 4: the full pipeline (quality gate -> OOD gate -> CP set) turned into visor actions, and how
often the inspector has to step in, at each CP alpha.

From 1_main_model_resnet18/:
  python enhanced_res/arch_a/experiments/archA_visor.py          # reuses archA_gates.py's cached corrupted-crop features

Pipeline, in this order (the deployed calibration.json; no retraining):
  1. quality gate, alpha 0.05 (sharpness both sides)                     fail -> "Bad image: re-take"
  2. unfamiliar-input gate: kNN distance + AoN, alpha 0.08   fail -> "Can't assess: re-aim"
  3. CP set (prob-T probabilities, family-weighted global LAC, alpha 0.05 / 0.10 / 0.20):
       1 class        -> show the class and its calibrated score                  (no human step)
       2+ classes     -> "Ambiguous: <classes>", flag for review / screenshot     (human step)
       0 classes      -> "Unclear: re-inspect"                                   (human step)
Human attention = every outcome except a single shown class.
For labelled crops it also counts what the inspector is told: a single class that is right, a single class that is
wrong (shown with no warning; the costly case), and whether an ambiguous set holds the true class.

Sets: all 2,417 clean test-half crops; the 553-crop sample with 14 synthetic corruptions; 400 off-target patches.
Writes enhanced_res/arch_a/results/archA/step4/.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import arch0_cp as A0
import archA_cp as A1
import archA_gates as G

OUT = HERE.parent / "results" / "archA" / "step4"
ALPHAS_CP = (0.05, 0.10, 0.20)
ACTIONS = ["show class", "ambiguous: review", "unclear: re-inspect", "bad image: re-take", "can't assess: re-aim"]


def actions(pq, pood, p, q_cp):
    """Visor action per crop, plus the CP set (empty for gated crops)."""
    reaim = pood <= G.arch().alpha_ood                 # re-aim wins when both gates fire (runtime.py)
    retake = ~reaim & (pq <= G.arch().alpha_q)
    S = A0.lac_sets(p, q_cp)
    S[retake | reaim] = False
    size = S.sum(1)
    act = np.where(retake, ACTIONS[3], np.where(reaim, ACTIONS[4],
                   np.where(size == 1, ACTIONS[0], np.where(size >= 2, ACTIONS[1], ACTIONS[2]))))
    return act, S


def summarise(act, S, y=None):
    r = {a: float((act == a).mean()) for a in ACTIONS}
    r["human attention"] = 1 - r[ACTIONS[0]]
    if y is not None:
        shown = act == ACTIONS[0]
        right = S[np.arange(len(y)), y]
        r["single class, right"] = float((shown & right).mean())
        r["single class, wrong"] = float((shown & ~right).mean())
        amb = act == ACTIONS[1]
        r["ambiguous holds truth"] = float(right[amb].mean()) if amb.any() else np.nan
    else:
        r["single class, right"] = np.nan
        r["single class, wrong"] = r[ACTIONS[0]]          # any defect label on an off-target patch is wrong
        r["ambiguous holds truth"] = np.nan
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = G.prepare()
    te, is_cal, y, g, P = D["te"], D["is_cal"], D["y"], D["g"], D["P"]

    # clean: every test-half crop (not just the 553 sample)
    tm = ~is_cal
    imgs = [Image.open(p).convert("RGB") for p in te.crop_path[tm]]
    pq_clean, _, pood_clean, _, _, _ = G.deployed(D, imgs, D["ft"][tm])

    rows = []
    for a in ALPHAS_CP:
        q_cp = G.deployed_q(a)
        act, S = actions(pq_clean, pood_clean, P[tm], q_cp)
        rows.append({"set": "clean (all 2,417 test-half crops)", "alpha_cp": a, **summarise(act, S, y[tm])})
        for name in D["sets"]:
            p, pq, _, po, _, _, _ = G.deployed_set(D, name)
            act, S = actions(pq, po, p, q_cp)
            rows.append({"set": name, "alpha_cp": a,
                         **summarise(act, S, None if name == "off-target" else y[D["ev"]])})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "visor_actions.csv", index=False)

    # ------------------------------------------------------------------ figure: clean workload vs alpha
    c = res[res.set.str.startswith("clean (all")].set_index("alpha_cp")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bottom = np.zeros(len(c))
    for a_ in ACTIONS:
        ax.bar([f"alpha {v:.2f}" for v in c.index], c[a_], bottom=bottom, label=a_)
        bottom += c[a_].to_numpy()
    ax.set_ylabel("share of clean test-half crops"); ax.set_title("What the visor shows on normal defect crops")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1, .5))
    fig.tight_layout(); fig.savefig(OUT / "workload.png", dpi=150); plt.close(fig)

    # ------------------------------------------------------------------ report
    pct = lambda v: "-" if pd.isna(v) else f"{v * 100:.1f}%"
    md = ["# Architecture A, step 4: from prediction set to visor action", "",
          f"Quality gate alpha {G.arch().alpha_q}, unfamiliar-input gate alpha {G.arch().alpha_ood}, then prob-T + family-weighted "
          "global LAC. Thresholds from calibration.json (val + cal half); everything below on the test half. No retraining.", "",
          "## The action table", "",
          "| pipeline outcome | visor shows | inspector |", "|---|---|---|",
          "| quality gate fails (blur, dark, overexposed, glare) | **Bad image: re-take** | re-takes the shot |",
          "| unfamiliar-input gate fails (not like any training crop) | **Can't assess: re-aim** | re-aims the camera |",
          "| CP set has 1 class | **Crack 88%** (class + prob-T calibrated score) | no step needed |",
          "| CP set has 2+ classes | **Ambiguous: crack or dent** | reviews; screenshot saved for later |",
          "| CP set is empty | **Unclear: re-inspect** | inspects by eye |", "",
          "Human attention = everything except a single shown class. The costly case is a single class that is wrong: "
          "it is shown with no warning.", "",
          "## Normal defect crops (all 2,417 test-half crops)", "",
          "| CP alpha | single class shown | ...right | ...wrong (no warning) | ambiguous | ambiguous holds truth | unclear | re-take | re-aim | **human attention** |",
          "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    S1, AMB, UNC, RT, RA = ACTIONS[0], ACTIONS[1], ACTIONS[2], ACTIONS[3], ACTIONS[4]
    for _, r in res[res.set.str.startswith("clean (all")].iterrows():
        md.append(f"| {r.alpha_cp:.2f} | {pct(r[S1])} | {pct(r['single class, right'])} | "
                  f"{pct(r['single class, wrong'])} | {pct(r[AMB])} | {pct(r['ambiguous holds truth'])} | "
                  f"{pct(r[UNC])} | {pct(r[RT])} | {pct(r[RA])} | **{pct(r['human attention'])}** |")
    md += ["", "## Every test set at CP alpha 0.10", "",
           "| set | single class: right | single class: wrong | ambiguous | unclear | re-take | re-aim | human attention |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in res[res.alpha_cp == 0.10].iterrows():
        md.append(f"| {r.set} | {pct(r['single class, right'])} | {pct(r['single class, wrong'])} | {pct(r[AMB])} | "
                  f"{pct(r[UNC])} | {pct(r[RT])} | {pct(r[RA])} | {pct(r['human attention'])} |")
    md += ["", "## Human attention on every set, by CP alpha", "",
           "| set | " + " | ".join(f"alpha {a:.2f}" for a in ALPHAS_CP) + " |", "|---|" + "---:|" * len(ALPHAS_CP)]
    piv = res.pivot(index="set", columns="alpha_cp", values="human attention").reindex(res.set.unique())
    for n, r in piv.iterrows():
        md.append(f"| {n} | " + " | ".join(pct(v) for v in r) + " |")
    md += ["", "![workload](workload.png)", ""]
    (OUT / "README.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
