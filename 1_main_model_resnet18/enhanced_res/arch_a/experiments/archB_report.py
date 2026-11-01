"""Architecture B report: gc7 + the decision layer (quality gate off), next to architecture A (v2), on every set.

From 1_main_model_resnet18/ (after calibrate.py --model gc7 --data d1_gc7_fix and archB_choose.py):
  python enhanced_res/arch_a/experiments/archB_report.py

Nothing is chosen here. The headline question for B is the visor's binary call: does it stay silent on undamaged
skin, and does it ever stay silent on a defect? A re-aim on undamaged skin counts as the visor speaking.
  test half      d1_gc7_fix test half (87 families), read once: coverage, silent misses, no-defect shown nothing
  extra_eval     760 no-defect crops never trained on (d1_gc7/extra_eval.csv)
  FGVC / D6      1,000 undamaged skin patches and 500 whole-aircraft crops (never used for any fitting), per crop and
                 per photo (the visor speaks for a photo if any of its crops is shown)
  off-target     people, helmets, hands, hangars ...: B as shipped (its AoN saw these categories) and with an AoN
                 refitted on the fit categories only (held out); plus wallpaper patches, static, the user's photo
  D5             other photos of dents and missing heads
Writes enhanced_res/arch_a/results/archB/README.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

import archB_common as BC
from archB_common import R

OUT = BC.HERE.parent / "results" / "archB"
SPEAK_FREE = (R.NOTHING,)                     # the only outcome that shows nothing


def acts(dec):
    return np.array([d.action for d in dec])


def breakdown(a):
    """Share of crops per visor outcome."""
    return {"nothing": np.mean(a == R.NOTHING), "re-aim": np.mean(a == R.REAIM), "re-take": np.mean(a == R.RETAKE),
            "shown a defect": np.mean(np.isin(a, (R.SHOW, R.AMBIGUOUS, R.UNCLEAR, R.POSSIBLE)))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pct = lambda v: f"{v * 100:.1f}%"
    arch = {k: R.ArchA(p) for k, (_, p) in BC.ARCH.items()}
    B = arch["B"]
    nd = B.nd
    md = ["# Architecture B: gc7 + the calibrated layer, quality gate off", "",
          f"Model enhanced_res gc7 (7 classes, with no-defect), crops {B.cal['data']['split']} (D1 with the 27 Oct box "
          f"corrections), family-weighted validation T {B.T:.6f}, OOD α {B.alpha_ood} (kNN + AoN), CP α {B.alpha} (LAC, fitted on gate-passed "
          f"cal-half crops), miss guard {'α ' + str(B.guard[0]) if B.guard else 'none'}. Architecture A (v2, 6 "
          "classes, quality gate on) is shown next to it where the set allows. A re-aim on undamaged skin counts as "
          "the visor speaking.", ""]

    # ---------------------------------------------------------------- test half (report only)
    va, cal, tst = BC.halves(B.cal["data"]["split"])
    z = np.load(BC.C.CACHE["gc7"], allow_pickle=True)
    nv, nc = len(va), len(cal)
    sl = slice(nv + nc, None)
    assert list(z["crop_path"][sl]) == list(tst.crop_path)
    F, P, D = z["feats"][sl], z["probs"][sl], z["knn_dist"][sl]
    cid = {c: i for i, c in enumerate(B.classes)}
    y, g = tst.cls.map(cid).to_numpy(), tst.src_group.to_numpy()
    a = acts(B.decide(P, knn_dist=D, feats=F))
    S = B.sets(P)[1]
    hit = S[np.arange(len(y)), y]
    defect, clean = y != nd, y == nd
    src = tst.source.to_numpy()
    sm, fa = (a == R.NOTHING) & defect, (a != R.NOTHING) & clean
    lo, hi = BC.family_ci(sm[defect], g[defect])
    flo, fhi = BC.family_ci(fa[clean], g[clean])
    clo, chi = BC.family_ci(hit, g)
    md += ["## Test half (87 families; report only)", "",
           "| | value (95% interval over families) |", "|---|---:|",
           f"| **silent miss**: defect crops shown as nothing | **{pct(sm[defect].mean())}** ({pct(lo)}–{pct(hi)}) |",
           f"| **visor speaks on no-defect crops** (false alarm) | **{pct(fa[clean].mean())}** ({pct(flo)}–{pct(fhi)}) |",
           f"| … healed / clean-cropper no-defect | {pct(fa[clean & (src == 'healed')].mean())} / "
           f"{pct(fa[clean & (src == 'clean_cropper')].mean())} |",
           f"| CP coverage (target 90%) | {pct(hit.mean())} ({pct(clo)}–{pct(chi)}) |",
           f"| mean set size / single class | {S.sum(1).mean():.2f} / {pct((S.sum(1) == 1).mean())} |",
           f"| single wrong defect class shown without warning | {pct(((a == R.SHOW) & ~hit).mean())} |",
           f"| \"POSSIBLE …: check\" / re-aim | {pct(np.mean(a == R.POSSIBLE))} / {pct(np.mean(a == R.REAIM))} |",
           "", "Per class: coverage, and what the visor does.", "",
           "| class | crops | coverage | shown nothing | POSSIBLE | re-aim |", "|---|---:|---:|---:|---:|---:|"]
    for k, c in enumerate(B.classes):
        m = y == k
        md.append(f"| {c} | {m.sum()} | {pct(hit[m].mean())} | {pct(np.mean(a[m] == R.NOTHING))} | "
                  f"{pct(np.mean(a[m] == R.POSSIBLE))} | {pct(np.mean(a[m] == R.REAIM))} |")
    cov_d, cov_n = hit[defect].mean(), hit[clean].mean()
    md += ["", "**Coverage depends on the class mix.** The test half is 12% no-defect; a walkaround is mostly "
           "undamaged skin. Reweighted (defects keep their own mix):", "", "| share of undamaged crops | coverage | "
           "visor speaks, per crop |", "|---:|---:|---:|"]
    for w in (0.12, 0.5, 0.9):
        speak = w * fa[clean].mean() + (1 - w) * np.mean(a[defect] != R.NOTHING)
        md.append(f"| {w:.0%} | {pct(w * cov_n + (1 - w) * cov_d)} | {pct(speak)} |")
    md += ["", "Global LAC does not bound P(defect → shown as nothing) directly, and with 88 calibration families no "
           "rate below about 1/88 can be certified; the silent-miss figure above is a measurement, not a guarantee.", ""]

    # ---------------------------------------------------------------- the other sets
    man, d5, wall, static = BC.extra_sets()
    extra = pd.read_csv(BC.MAIN / "d1_gc7/extra_eval.csv")
    extra["path"] = [str(BC.MAIN / p) for p in extra.crop_path]
    out = {}
    for k, (mname, _) in BC.ARCH.items():
        for key, items in (("extra", list(extra.path)), ("bench", list(man.path)), ("d5", list(d5.crop_path)),
                           ("wall", wall), ("static", static)):
            F_, P_, D_ = BC.outputs(mname, key, items)
            out[(k, key)] = (acts(BC.decide(arch[k], items, F_, P_, D_)), F_, P_, D_)
            print(f"  {k} {key} done", flush=True)

    # extra_eval
    ae = out[("B", "extra")][0]
    md += ["## extra_eval: 760 no-defect crops never trained on", "",
           "| | B | of which healed / clean |", "|---|---:|---:|",
           f"| shown nothing | {pct(np.mean(ae == R.NOTHING))} | {pct(np.mean(ae[extra.source == 'healed'] == R.NOTHING))} / "
           f"{pct(np.mean(ae[extra.source == 'clean_cropper'] == R.NOTHING))} |",
           f"| visor speaks (false alarm) | {pct(np.mean(ae != R.NOTHING))} | |", ""]

    # FGVC / D6
    md += ["## FGVC / D6: undamaged aircraft (headline test; never used for fitting)", "",
           "| set | | A: nothing / re-aim / re-take / shown a defect | B: nothing / re-aim / shown a defect |",
           "|---|---|---:|---:|"]
    for cat in ("aircraft skin patch (no damage)", "whole aircraft (no damage)"):
        m = (man.category == cat).to_numpy()
        for unit in ("per crop", "per photo"):
            row = []
            for k in ("A", "B"):
                ak = out[(k, "bench")][0][m]
                if unit == "per photo":
                    img = man.image[m].to_numpy()
                    speak = pd.Series(ak != R.NOTHING).groupby(img).any()
                    row.append(f"visor speaks on {pct(speak.mean())} of {len(speak)} photos")
                else:
                    b = breakdown(ak)
                    row.append(" / ".join(pct(b[x]) for x in (("nothing", "re-aim", "re-take", "shown a defect")
                                                              if k == "A" else ("nothing", "re-aim", "shown a defect"))))
            md.append(f"| {cat} | {unit} | {row[0]} | {row[1]} |")
    md.append("")

    # off-target: shipped and held out
    aB, FB, PB, DB = out[("B", "bench")]
    fit = (man.role == "fit").to_numpy()
    bank = torch.load(BC.MODELS["gc7"] / "enhanced_res_head.pt", map_location="cpu")["knn_bank"].float().numpy()
    X = np.concatenate([bank, FB[fit]])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000)
    clf.fit((X - mu) / sd, np.r_[np.zeros(len(bank)), np.ones(fit.sum())])
    zc = np.load(BC.C.CACHE["gc7"], allow_pickle=True)
    f_cal = zc["feats"][nv:nv + nc]
    sc = lambda f: clf.predict_proba((f - mu) / sd)[:, 1]
    p_ho = np.minimum(1, 2 * np.minimum(R.pvalue(B.o_cal, B.o_groups, DB),
                                        R.pvalue(sc(f_cal), cal.src_group.to_numpy(), sc(FB))))
    Sb = B.sets(PB)[1]
    held = np.where(p_ho <= B.alpha_ood, R.REAIM,
                    np.where(Sb[:, nd] & (Sb.sum(1) == 1), R.NOTHING, R.SHOW))      # what matters: re-aim / nothing / shown
    md += ["## Off-target photos (each photo: whole-frame square + box patch)", "",
           "Held out = AoN refitted without that category (fit categories only); shipped = the deployed AoN, which "
           "saw every category below. Shown = a defect or POSSIBLE on the visor (the false accept that matters); "
           "nothing = silent.", "",
           "| category | A shown | B shipped: re-aim / nothing / shown | B held out: re-aim / nothing / shown |",
           "|---|---:|---:|---:|"]
    for role in ("fit", "test"):
        for cat in sorted(man.category[man.role == role].unique()):
            m = (man.category == cat).to_numpy()
            aA = out[("A", "bench")][0][m]
            bs, bh = breakdown(aB[m]), breakdown(held[m])
            md.append(f"| {cat} ({role}) | {pct(breakdown(aA)['shown a defect'])} | "
                      f"{pct(bs['re-aim'])} / {pct(bs['nothing'])} / {pct(bs['shown a defect'])} | "
                      + ("-" if role == "fit" else f"{pct(bh['re-aim'])} / {pct(bh['nothing'])} / {pct(bh['shown a defect'])}")
                      + " |")
    t = (man.role == "test").to_numpy()
    md.append(f"| **all held-out categories** | {pct(breakdown(out[('A', 'bench')][0][t])['shown a defect'])} | "
              f"{pct(breakdown(aB[t])['shown a defect'])} shown | {pct(breakdown(held[t])['shown a defect'])} shown |")
    for key, lbl in (("wall", "wallpaper patches"), ("static", "static / colours / gradients")):
        md.append(f"| {lbl} | {pct(breakdown(out[('A', key)][0])['shown a defect'])} | "
                  f"{' / '.join(pct(breakdown(out[('B', key)][0])[x]) for x in ('re-aim', 'nothing', 'shown a defect'))} | |")
    # for the dashboard's "What it refuses" panel: share of each set NOT shown as a defect (stopped or silent)
    gates = {"alpha_ood": B.alpha_ood, "source": "enhanced_res/arch_a/results/archB/",
             "sets": [{"set": "off-target held out", "label": "People, hands, helmets, hangars, runways (never trained on)",
                       "stopped": float(1 - breakdown(held[t])["shown a defect"])},
                      {"set": "wallpapers", "label": "Patches of non-aircraft photos",
                       "stopped": float(1 - breakdown(out[("B", "wall")][0])["shown a defect"])},
                      {"set": "static", "label": "Random static, solid colours, gradients",
                       "stopped": float(1 - breakdown(out[("B", "static")][0])["shown a defect"])}]}
    (OUT / "gates.json").write_text(BC.json.dumps(gates, indent=1))
    u = (man.source == "user").to_numpy()
    md += ["", f"The user's hard-hat photo: A {', '.join(out[('A', 'bench')][0][u])}; B {', '.join(aB[u])}.", ""]

    # D5
    cidB = {c: i for i, c in enumerate(B.classes)}
    yd = d5.cls.map(cidB).to_numpy()
    for k in ("A", "B"):
        ak, _, Pk, _ = out[(k, "d5")]
        Sk = arch[k].sets(Pk)[1]
        ck = {c: i for i, c in enumerate(arch[k].classes)}
        hk = Sk[np.arange(len(d5)), d5.cls.map(ck).to_numpy()]
        out[(k, "d5_cov")] = (hk.mean(), np.mean(np.isin(ak, (R.REAIM, R.RETAKE))), np.mean(ak == R.NOTHING))
    md += ["## D5: other photos of dents and missing heads", "", "| | A | B |", "|---|---:|---:|",
           f"| CP coverage | {pct(out[('A', 'd5_cov')][0])} | {pct(out[('B', 'd5_cov')][0])} |",
           f"| stopped by a gate | {pct(out[('A', 'd5_cov')][1])} | {pct(out[('B', 'd5_cov')][1])} |",
           f"| silent miss (shown nothing) | - | {pct(out[('B', 'd5_cov')][2])} |", "",
           "Defects that never become a crop are silent misses no decision layer can see; how frames become crops on "
           "the helmet is still open."]
    (OUT / "README.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
