"""Picture book of architecture A: the unfamiliar-input (OOD) gate, temperature scaling, CP coverage, and the false
accept / false reject rates of every stage, with real crops for each case.

From 1_main_model_resnet18/:
  python enhanced_res/arch_a/experiments/archA_gallery.py     # reuses archA_gates.py's cached features

The deployed architecture A (calibration.json: quality gate 0.10, sharpness both sides, in parallel with the OOD gate = kNN distance +
AoN, 0.08 -> prob-T -> family-weighted LAC, alpha
0.10), thresholds from calibration.json (val + cal half). Normal crops: all 2,417 test-half crops (rates) and the 553-crop sample (the
corrupted copies). Off-target: 400 non-aircraft patches. D5 rows come from archA_d5.py's results.
Writes enhanced_res/arch_a/results/archA/examples/archA_gallery.html (self-contained, images embedded).
"""
import base64
import io
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
from model import load_split

OUT = HERE.parent / "results" / "archA" / "examples"
D5 = HERE.parent / "results" / "archA" / "d5"
N_SHOW, SEED, ALPHAS = 8, 0, (0.05, 0.10, 0.20)


# ------------------------------------------------------------------ html helpers
def b64img(im, size=130):
    buf = io.BytesIO()
    im.convert("RGB").resize((size, size)).save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def b64fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f'<img class="fig" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def pct(v):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.1f}%"


def pick(idx, fam, rng, n=N_SHOW):
    """Up to n random indices, one per family first."""
    idx = rng.permutation(np.asarray(idx))
    seen, first, rest = set(), [], []
    for i in idx:
        (rest if fam[i] in seen else first).append(i)
        seen.add(fam[i])
    return (first + rest)[:n]


def gallery(title, note, cards):
    body = "".join(cards) if cards else '<p class="none">none in this set</p>'
    return f'<h4>{title} <small>{note}</small></h4><div class="grid">{body}</div>'


def card(im, lines, tone=""):
    return (f'<div class="card {tone}"><img src="{b64img(im)}"><div class="cap">'
            + "<br>".join(lines) + "</div></div>")


def table(df, fmt=None):
    fmt = fmt or {}
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = "".join("<tr>" + "".join(f"<td>{fmt.get(c, str)(v)}</td>" for c, v in r.items()) + "</tr>"
                   for _, r in df.iterrows())
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'


def setstr(S_row, p_row, classes):
    idx = [k for k in np.argsort(-p_row) if S_row[k]]
    return " / ".join(f"{classes[k]} {p_row[k] * 100:.0f}%" for k in idx) or "(empty)"


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    D = G.prepare()
    te, is_cal, y, g, P, classes, T = D["te"], D["is_cal"], D["y"], D["g"], D["P"], D["classes"], D["T_prob"]
    tm = ~is_cal
    _, _, Lt, Kt = A1.model_parts(load_split(), "test")
    raw = A1.blend(Lt, Kt)                                   # the model as shipped (no temperature)
    assert np.allclose(A1.temper(raw, T), P), "test crop order differs from archA_gates.prepare()"
    yt, gt, Pt, Rt = y[tm], g[tm], P[tm], raw[tm]
    paths = te.crop_path[tm].tolist()
    imgs = [Image.open(p).convert("RGB") for p in paths]

    # gates on all clean test-half crops
    pq, why, pood, part, dist, notair = G.deployed(D, imgs, D["ft"][tm])
    q_cp = {a: G.deployed_q(a) for a in ALPHAS}
    S = Pt >= 1 - q_cp[0.10]
    ofail = pood <= G.arch().alpha_ood                  # re-aim wins when both gates fire (runtime.py)
    qfail = ~ofail & (pq <= G.arch().alpha_q)
    passed = ~(qfail | ofail)
    fam_mean = lambda m: float(pd.Series(m).groupby(gt).mean().mean())

    # corrupted / off-target sets
    sets = {}
    for name in D["sets"]:
        if name == "clean":
            continue
        p, q_, w_, po_, part_, dist_, na_ = G.deployed_set(D, name)
        sets[name] = dict(p=p, pq=q_, why=w_, dist=dist_, pood=po_, part=part_, notair=na_, imgs=D["sets"][name])
    ev = D["ev"]
    y_ev, g_ev = y[ev], g[ev]

    html = []
    # ================================================================== 0. definitions
    html.append(f"""<h1>Architecture A in pictures</h1>
<p class="lede">enhanced_res on d1_grouped, unchanged weights. Pipeline (calibration.json): quality gate (α {G.arch().alpha_q},
sharpness checked on both sides) → unfamiliar-input gate (kNN distance + AoN, α {G.arch().alpha_ood}) → temperature (T = {T:.3f}) → conformal set (LAC, α 0.10). Thresholds come from the calibration half
(1,838 crops, 88 photo families); everything below is measured on the other half (2,417 crops, 87 families), which
nothing was fitted on. Example crops are random (seeded), one per photo family where possible.</p>
<p class="lede"><b>False reject</b> = a good, normal crop that a stage stops or doubts (costs inspector time).
<b>False accept</b> = something a stage should have stopped but let through (a bad image, a non-aircraft image, or a
wrong single answer shown with no warning).</p>
<nav><a href="#ood">1. Unfamiliar-input gate</a> · <a href="#temp">2. Temperature</a> · <a href="#cp">3. CP coverage</a> ·
<a href="#rates">4. False accept / reject everywhere</a></nav>""")

    # ================================================================== 1. OOD
    off = sets["off-target"]
    A = G.arch()
    half = A.alpha_ood / A.n_parts                       # Bonferroni: each part fires at about alpha / 2
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2))
    for ax, key, cal_s, lab in ((axs[0], "dist", A.o_cal, "kNN distance: 1 − mean cosine similarity to the 5 nearest training crops"),
                                (axs[1], "notair", A.lr["cal_scores"], "AoN: probability of NOT aircraft skin")):
        vals = {"normal test-half crops": dist if key == "dist" else notair, "off-target patches": off[key],
                "heavy JPEG damage": sets["jpeg s2"][key]}
        bins = np.linspace(0, max(v.max() for v in vals.values()) * 1.02, 60)
        for (k, v), al in zip(vals.items(), (.6, .6, .5)):
            ax.hist(v, bins, alpha=al, density=True, label=k)
        cut = float(np.quantile(cal_s, 1 - half))
        ax.axvline(cut, color="k", ls="--", lw=1)
        ax.text(cut, ax.get_ylim()[1] * .9, "  ≈ cut-off", fontsize=8)
        ax.set_xlabel(lab, fontsize=8); ax.set_yticks([])
    axs[0].legend(fontsize=8)
    off_caught = off["pood"] <= G.arch().alpha_ood
    html.append(f"""<h2 id="ood">1. Unfamiliar-input (OOD) gate: "CAN'T ASSESS: re-aim"</h2>
<p class="lede">Two scores on each crop's 768 model features. <b>kNN distance</b>: how far the crop is from its 5 nearest
of the 20,179 training crops (far = unfamiliar). <b>AoN</b>: a logistic regression on the same features,
fitted on D1 crops vs photos of people, faces, helmets, hands, gloves, tools, vehicles and hangar / airport / industrial
scenes (Open Images, Places365). Each score becomes a conformal p-value on normal calibration crops; the gate fires when
2 × the smaller one is ≤ {A.alpha_ood} (each part at about {half:.2f}). The off-target patches below are macOS
wallpapers, which the regression never saw. Realistic people / helmets / hands / hangars, held out from the regression:
about 5% get through both gates on categories the AoN never saw (runways 14.5%; <code>results/archA/offtarget/rates.md</code>), against 54–70% for the kNN-only gate.</p>
<p class="sum">Normal crops flagged (false reject): <b>{pct(float((pood <= G.arch().alpha_ood).mean()))}</b> per crop,
{pct(fam_mean(pood <= G.arch().alpha_ood))} per family · off-target patches caught: <b>{pct(float(off_caught.mean()))}</b>
(false accept {pct(float(1 - off_caught.mean()))}; with the quality gate too: caught
{pct(float(((off["pq"] <= G.arch().alpha_q) | off_caught).mean()))}).</p>{b64fig(fig)}""")
    S_off = A0.lac_sets(off["p"], q_cp[0.10])
    ofam = np.arange(len(off["imgs"]))
    html.append(gallery("Off-target patches caught", "→ re-aim",
                        [card(off["imgs"][i], [f"distance {off['dist'][i]:.3f}", f"AoN {off['notair'][i]:.2f}", f"OOD p {off['pood'][i]:.3f} ({off['part'][i]})",
                                               f"model's guess: {classes[off['p'][i].argmax()]} {off['p'][i].max() * 100:.0f}%"], "good")
                         for i in pick(np.flatnonzero(off_caught), ofam, rng)]))
    miss = np.flatnonzero(~off_caught & (off["pq"] > G.arch().alpha_q))
    html.append(gallery("Off-target patches that got through both gates", "false accepts: the visor names a defect",
                        [card(off["imgs"][i], [f"distance {off['dist'][i]:.3f}", f"AoN {off['notair'][i]:.2f}", f"OOD p {off['pood'][i]:.3f}",
                                               f"visor: <b>{setstr(S_off[i], off['p'][i], classes)}</b>"], "bad")
                         for i in pick(miss, ofam, rng)]))
    fr = np.flatnonzero(pood <= G.arch().alpha_ood)
    html.append(gallery("Normal defect crops flagged as unfamiliar", f"false rejects ({len(fr)} of {len(pood)})",
                        [card(imgs[i], [f"true: {classes[yt[i]]}", f"distance {dist[i]:.3f}", f"AoN {notair[i]:.2f}", f"OOD p {pood[i]:.3f} ({part[i]})"], "bad")
                         for i in pick(fr, gt, rng)]))
    ok = np.flatnonzero(pood > 0.5)
    html.append(gallery("Typical normal crops (well inside the training data)", "for comparison",
                        [card(imgs[i], [f"true: {classes[yt[i]]}", f"distance {dist[i]:.3f}", f"AoN {notair[i]:.2f}", f"OOD p {pood[i]:.3f}"], "good")
                         for i in pick(ok, gt, rng, 4)]))
    for name in ("noise s2", "jpeg s2"):
        s = sets[name]
        c = np.flatnonzero((s["pood"] <= G.arch().alpha_ood) & (s["pq"] > G.arch().alpha_q))
        html.append(gallery(f"{name}: damaged crops only the OOD gate caught",
                            f"the quality gate passed these ({pct(float(len(c) / len(s['pq'])))} of the set)",
                            [card(s["imgs"][i], [f"true: {classes[y_ev[i]]}", f"distance {s['dist'][i]:.3f}", f"AoN {s['notair'][i]:.2f}"], "good")
                             for i in pick(c, g_ev, rng, 4)]))

    # ================================================================== 2. temperature
    rel = []
    for lab, pp in (("as shipped", Rt), ("with temperature", Pt)):
        conf, right = pp.max(1), pp.argmax(1) == yt
        edges = np.linspace(0, 1, 11)
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                rel.append({"version": lab, "lo": lo, "hi": hi, "n": int(m.sum()), "stated": conf[m].mean(), "right": right[m].mean()})
    rel = pd.DataFrame(rel)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, lab in zip(axs, ("as shipped", "with temperature")):
        r = rel[rel.version == lab]
        ax.bar((r.lo + r.hi) / 2, r.right, width=.09, alpha=.7, label="actually right")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.scatter(r.stated, r.right, s=12, color="C3", zorder=3, label="(stated, right)")
        for _, rr in r.iterrows():
            ax.text((rr.lo + rr.hi) / 2, 0.02, f"{rr.n}", ha="center", fontsize=6, color="w")
        e = A1.ece(Rt if lab == "as shipped" else Pt, yt)
        ax.set_title(f"{lab}: ECE {e:.3f}"); ax.set_xlabel("stated confidence (top class)")
    axs[0].set_ylabel("share actually right"); axs[0].legend(fontsize=7, loc="upper left")
    acc = float((Pt.argmax(1) == yt).mean())
    html.append(f"""<h2 id="temp">2. Temperature scaling: making the % honest</h2>
<p class="lede">One number, T = {T:.3f}, fitted on val: every probability is raised to the power 1/T and renormalised.
It softens the model's overconfidence. The top class never changes, so accuracy stays {pct(acc)}. Bars below the
dashed line = the model claims more than it delivers. White numbers = crops in each bin.</p>
<p class="sum">Mean stated confidence {pct(float(Rt.max(1).mean()))} → {pct(float(Pt.max(1).mean()))} against
{pct(acc)} actually right · ECE {A1.ece(Rt, yt):.3f} → {A1.ece(Pt, yt):.3f} · same top class on
{pct(float((Rt.argmax(1) == Pt.argmax(1)).mean()))} of crops.</p>{b64fig(fig)}""")
    wrong_hi = np.flatnonzero((Rt.max(1) >= 0.95) & (Rt.argmax(1) != yt))
    html.append(gallery("Overconfident mistakes", f"the model said ≥ 95% and was wrong: {len(wrong_hi)} crops "
                        f"({pct(float(len(wrong_hi) / max((Rt.max(1) >= .95).sum(), 1)))} of its ≥ 95% answers)",
                        [card(imgs[i], [f"true: <b>{classes[yt[i]]}</b>", f"said: {classes[Rt[i].argmax()]}",
                                        f"{Rt[i].max() * 100:.1f}% → <b>{Pt[i].max() * 100:.1f}%</b>"], "bad")
                         for i in pick(wrong_hi, gt, rng)]))
    right_hi = np.flatnonzero((Rt.max(1) >= 0.95) & (Rt.argmax(1) == yt))
    html.append(gallery("Confident and right", "the same softening applies; these still get the top shown scores",
                        [card(imgs[i], [f"true: {classes[yt[i]]}", f"{Rt[i].max() * 100:.1f}% → <b>{Pt[i].max() * 100:.1f}%</b>"], "good")
                         for i in pick(right_hi, gt, rng, 4)]))
    unsure = np.flatnonzero(Rt.max(1) < 0.6)
    html.append(gallery("Unsure answers", "low scores barely move: temperature mainly pulls down the top end",
                        [card(imgs[i], [f"true: {classes[yt[i]]}", f"said: {classes[Rt[i].argmax()]}",
                                        f"{Rt[i].max() * 100:.1f}% → <b>{Pt[i].max() * 100:.1f}%</b>"],
                              "good" if Rt[i].argmax() == yt[i] else "bad")
                         for i in pick(unsure, gt, rng, 4)]))

    # ================================================================== 3. CP coverage
    hit = S[np.arange(len(yt)), yt]
    size = S.sum(1)
    cov_rows = []
    for a in ALPHAS:
        Sa = Pt >= 1 - q_cp[a]
        h = Sa[np.arange(len(yt)), yt]
        cov_rows.append({"α": a, "target": 1 - a, "coverage (crops)": h.mean(), "coverage (families)": fam_mean(h),
                         "mean set size": Sa.sum(1).mean(), "single class": (Sa.sum(1) == 1).mean()})
    for k in (1, 2, 3):   # the old fixed displays
        top = np.argsort(-Rt, 1)[:, :k]
        h = (top == yt[:, None]).any(1)
        cov_rows.append({"α": f"fixed top-{k}", "target": np.nan, "coverage (crops)": h.mean(),
                         "coverage (families)": fam_mean(h), "mean set size": float(k), "single class": float(k == 1)})
    cov = pd.DataFrame(cov_rows)
    per = pd.DataFrame({"class": classes, "coverage": [hit[yt == c].mean() for c in range(len(classes))],
                        "crops": [(yt == c).sum() for c in range(len(classes))]})
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    axs[0].barh(per["class"], per.coverage, color=["C3" if v < .9 else "C2" for v in per.coverage])
    axs[0].axvline(.9, color="k", ls="--", lw=1); axs[0].set_xlim(.5, 1); axs[0].set_title("coverage per class (α 0.10)")
    for i, (v, n) in enumerate(zip(per.coverage, per.crops)):
        axs[0].text(v + .005, i, f"{v * 100:.0f}% (n={n})", va="center", fontsize=7)
    fc = pd.Series(hit).groupby(gt).mean()
    axs[1].hist(fc, bins=np.linspace(0, 1, 21), color="C0")
    axs[1].axvline(.9, color="k", ls="--", lw=1)
    axs[1].set_title(f"coverage per photo family ({len(fc)} families)"); axs[1].set_xlabel("share of the family's crops covered")
    html.append(f"""<h2 id="cp">3. Conformal prediction: does the set hold the true class?</h2>
<p class="lede">The set is every class whose (temperature-scaled) probability is at least 1 − q, with q chosen on the
calibration half so the true class is inside about 90% of the time (α 0.10). Coverage = how often it really is. Below,
CP alone (no gates) on all 2,417 test-half crops, then real examples of each outcome.</p>
{table(cov, {"target": pct, "coverage (crops)": pct, "coverage (families)": pct, "single class": pct,
            "mean set size": lambda v: f"{v:.2f}", "α": lambda v: v if isinstance(v, str) else f"{v:.2f}"})}
{b64fig(fig)}""")
    cases = [("Single class, right", "shown as e.g. CRACK 88%", (size == 1) & hit, "good"),
             ("Single class, wrong", "false accept: wrong answer shown with no warning", (size == 1) & ~hit, "bad"),
             ("Ambiguous set holding the truth", "inspector picks from a short list", (size >= 2) & hit, "good"),
             ("Ambiguous set missing the truth", "even the list is wrong", (size >= 2) & ~hit, "bad")]
    for t, note, m, tone in cases:
        html.append(gallery(f"{t}", f"{note} · {pct(float(m.mean()))} of crops",
                            [card(imgs[i], [f"true: <b>{classes[yt[i]]}</b>", f"set: {setstr(S[i], Pt[i], classes)}"], tone)
                             for i in pick(np.flatnonzero(m), gt, rng)]))

    # ================================================================== 4. rates everywhere
    rows = [{"stage": "quality gate", "set": "normal crops (all 2,417)", "kind": "false reject",
             "per crop": (pq <= G.arch().alpha_q).mean(), "per family": fam_mean(pq <= G.arch().alpha_q)},
            {"stage": "OOD gate", "set": "normal crops (all 2,417)", "kind": "false reject",
             "per crop": (pood <= G.arch().alpha_ood).mean(), "per family": fam_mean(pood <= G.arch().alpha_ood)},
            {"stage": "both gates", "set": "normal crops (all 2,417)", "kind": "false reject",
             "per crop": (~passed).mean(), "per family": fam_mean(~passed)}]
    for name, s in sets.items():
        qf, of = s["pq"] <= G.arch().alpha_q, s["pood"] <= G.arch().alpha_ood
        rows.append({"stage": "quality / OOD / both", "set": name, "kind": "false accept (passed)",
                     "per crop": f"{pct(float((~qf).mean()))} / {pct(float((~of).mean()))} / {pct(float((~(qf | of)).mean()))}",
                     "per family": "-"})
    gate_t = pd.DataFrame(rows)
    fmt_any = lambda v: v if isinstance(v, str) else pct(float(v))

    cp_rows = []
    for a in ALPHAS:
        Sa = Pt >= 1 - q_cp[a]
        Sa[~passed] = False
        h = Sa[np.arange(len(yt)), yt]
        n = Sa.sum(1)
        top_right = Pt.argmax(1) == yt
        cp_rows.append({"α": f"{a:.2f}",
                        "truth not in set (miss)": float((passed & ~h).sum() / passed.sum()),
                        "false accept: single wrong shown": float(((n == 1) & ~h).mean()),
                        "false reject: ambiguous although top-1 was right": float(((n >= 2) & top_right).mean()),
                        "stopped by a gate": float((~passed).mean()),
                        "inspector needed": float(1 - (n == 1).mean())})
    cp_rows.append({"α": "old top-1 display", "truth not in set (miss)": float((Rt.argmax(1) != yt).mean()),
                    "false accept: single wrong shown": float((Rt.argmax(1) != yt).mean()),
                    "false reject: ambiguous although top-1 was right": 0.0, "stopped by a gate": 0.0,
                    "inspector needed": 0.0})
    cp_t = pd.DataFrame(cp_rows)

    d5 = pd.read_csv(D5 / "visor_actions.csv") if (D5 / "visor_actions.csv").exists() else None
    d5_html = ""
    if d5 is not None:
        d5 = d5[["alpha", "set", "bad image: re-take", "can't assess: re-aim", "single class, wrong",
                 "ambiguous: review", "human attention"]].rename(columns={
                     "bad image: re-take": "false reject: re-take", "can't assess: re-aim": "false reject: re-aim",
                     "single class, wrong": "false accept: single wrong", "ambiguous: review": "ambiguous"})
        d5_html = ("<h3>New data source: D5 outside test (calibrated on D1 only)</h3>"
                   "<p class='lede'>Normal defect crops from another dataset (dent + missing-head only): every gate "
                   "stop here is a false reject.</p>" + table(d5, {c: pct for c in d5.columns if c not in ("alpha", "set")}
                                                         | {"alpha": lambda v: f"{v:.2f}"}))
    html.append(f"""<h2 id="rates">4. False accepts and false rejects, every stage</h2>
<h3>The gates</h3>
<p class="lede">Normal crops: what the gate wrongly stops (false reject). Damaged / off-target sets (553 sample crops
each, synthetic damage; 400 off-target patches): what it wrongly lets through (false accept), shown as quality gate
alone / OOD gate alone / both. Per family = each photo family's rate, averaged (what the α budget controls).</p>
{table(gate_t, {"per crop": fmt_any, "per family": fmt_any})}
<h3>The conformal set (after the gates, all 2,417 normal test-half crops)</h3>
<p class="lede">Miss = the true class is not in the set (among crops that passed the gates). False accept = a single
class is shown and it is wrong. False reject = the set is ambiguous although the model's top answer was right (extra
inspector work that was not needed). The old display always showed one answer and never asked for help.</p>
{table(cp_t, {c: pct for c in cp_t.columns if c != "α"})}
{d5_html}""")

    css = """:root{--bg:#fff;--fg:#1c1c1e;--mut:#6b6b70;--card:#f5f5f7;--line:#e2e2e6;--good:#1e8449;--bad:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#ececef;--mut:#9a9aa0;--card:#232326;--line:#333338;
--good:#58d68d;--bad:#ff6b5e}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif;max-width:1180px;margin:0 auto;padding:24px 16px}
h1{font-size:26px}h2{margin-top:44px;border-bottom:1px solid var(--line);padding-bottom:6px}h4{margin:18px 0 6px}
h4 small{color:var(--mut);font-weight:400}.lede,.sum{color:var(--mut);max-width:900px}.sum{color:var(--fg)}
nav{margin:10px 0 0}nav a{color:inherit}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.card{background:var(--card);border-radius:8px;padding:6px;border-top:3px solid transparent}
.card.good{border-top-color:var(--good)}.card.bad{border-top-color:var(--bad)}
.card img{width:100%;border-radius:4px;display:block}.cap{font-size:11.5px;margin-top:4px}
.fig{max-width:100%;background:#fff;border-radius:6px;margin:8px 0}
.tw{overflow-x:auto}table{border-collapse:collapse;font-size:12.5px;margin:8px 0}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right}th:first-child,td:first-child,
th:nth-child(2),td:nth-child(2){text-align:left}.none{color:var(--mut);font-style:italic}"""
    page = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
            f'initial-scale=1"><title>Architecture A Gallery</title><style>{css}</style></head><body>'
            + "".join(html) + "</body></html>")
    (OUT / "archA_gallery.html").write_text(page)
    print("wrote", OUT / "archA_gallery.html", f"({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
