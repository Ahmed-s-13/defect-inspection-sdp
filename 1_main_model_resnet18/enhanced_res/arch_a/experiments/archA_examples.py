"""Example sheet for the quality gate: blur, glare, over- and underexposure crops that the gates caught vs let through.

From 1_main_model_resnet18/:
  python enhanced_res/arch_a/experiments/archA_examples.py       # reuses archA_gates.py's cached corrupted-crop features

Same settings as architecture A (quality gate alpha 0.10 -> OOD gate 0.10 -> prob-T + family-weighted LAC, alpha 0.10).
Each example shows the clean crop next to its corrupted copy, the four quality statistics of both, the gate p-values
and, for crops that got through, the CP set the visor would show. Also shows clean crops that were stopped (false
alarms). Examples are drawn at random (seeded), at most one per photo family per panel.
Writes enhanced_res/arch_a/results/archA/examples/quality_examples.html (self-contained, images embedded).
"""
import base64
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd

import arch0_cp as A0
import archA_cp as A1
import archA_gates as G

OUT = HERE.parent / "results" / "archA" / "examples"
N_SHOW, SEED = 6, 0
GROUPS = [("Blur", ["blur s1", "blur s2", "motion blur s1", "motion blur s2"]),
          ("Glare", ["glare s1", "glare s2"]),
          ("Overexposure", ["overexposed s1", "overexposed s2"]),
          ("Underexposure (whole crop darkened)", ["dark s1", "dark s2"])]
HOW = {"blur s1": "Gaussian blur, radius 2", "blur s2": "Gaussian blur, radius 4",
       "motion blur s1": "motion blur, 9 px streak", "motion blur s2": "motion blur, 21 px streak",
       "glare s1": "bright spot added, 70% strength", "glare s2": "bright spot added, full strength",
       "overexposed s1": "brightness x1.8", "overexposed s2": "brightness x3.0",
       "dark s1": "brightness x0.35", "dark s2": "brightness x0.12"}


def b64(im, size=150):
    buf = io.BytesIO()
    im.resize((size, size)).save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def pick(idx, fam, rng):
    """Up to N_SHOW random indices, one per family first."""
    idx = rng.permutation(idx)
    seen, first, rest = set(), [], []
    for i in idx:
        (rest if fam[i] in seen else first).append(i)
        seen.add(fam[i])
    return (first + rest)[:N_SHOW]


def stats_line(s):
    return (f"sharp {s.sharpness:.0f} · bright {s.brightness:.0f} · "
            f"white {s.clipped_hi * 100:.0f}% · black {s.clipped_lo * 100:.0f}%")


def card(D, name, i, clean_im, clean_st, im, st, pq, why, pood, S, classes, y):
    true = classes[y[i]]
    if pq[i] <= G.ALPHA_Q:
        verdict = f'<b class="stop">Re-take</b> (quality p = {pq[i]:.3f}, worst: {why[i]})'
    elif pood[i] <= G.ALPHA_OOD:
        verdict = f'<b class="aim">Re-aim</b> (OOD p = {pood[i]:.3f}; quality p = {pq[i]:.3f})'
    else:
        st_set = [classes[k] for k in np.flatnonzero(S[i])]
        ok = y[i] in np.flatnonzero(S[i])
        shown = st_set[0] if len(st_set) == 1 else "Ambiguous: " + " or ".join(st_set) if st_set else "Unclear"
        verdict = (f'<b class="pass">Passed</b> (quality p = {pq[i]:.3f}) → visor: <i>{shown}</i> '
                   f'<span class="{"ok" if ok else "bad"}">{"holds truth" if ok else "WRONG"}</span>')
    pair = "" if name == "clean" else (
        f'<figure><img src="{b64(clean_im)}"><figcaption>clean<br>{stats_line(clean_st)}</figcaption></figure>')
    return (f'<div class="card"><div class="imgs">{pair}<figure><img src="{b64(im)}"><figcaption>'
            f'{"clean" if name == "clean" else name}<br>{stats_line(st)}</figcaption></figure></div>'
            f'<p>true class: <b>{true}</b><br>{verdict}</p></div>')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    D = G.prepare()
    te, is_cal, ev, classes = D["te"], D["is_cal"], D["ev"], D["classes"]
    y, fam = D["y"][ev], D["g"][ev]
    q_cp = A1.family_threshold(D["P"][is_cal], D["y"][is_cal], D["g"][is_cal], G.ALPHA_CP)
    clean_ims, clean_st = D["sets"]["clean"], G.quality_stats(D["sets"]["clean"])
    ref = D["q_ref"]

    def panel(name):
        p, pq, why, _, pv = G.gate_pvalues(D, name)
        pood = pv["kNN distance"]
        S = A0.lac_sets(p, q_cp)
        ims, st = D["sets"][name], G.quality_stats(D["sets"][name])
        stopped = (pq <= G.ALPHA_Q) | (pood <= G.ALPHA_OOD)
        passed = ~stopped
        cov = S[passed][np.arange(passed.sum()), y[passed]].mean() if passed.any() else np.nan
        mk = lambda ids: "".join(card(D, name, i, clean_ims[i], clean_st.iloc[i], ims[i], st.iloc[i], pq, why,
                                      pood, S, classes, y) for i in ids) or '<p class="none">none</p>'
        caught_lbl, missed_lbl = ("Stopped (false alarms)", "Passed") if name == "clean" else ("Caught", "Not caught")
        return (f'<h3>{name} <small>{HOW.get(name, "no corruption")}</small></h3>'
                f'<p class="sum">{stopped.mean() * 100:.1f}% stopped of {len(ims)} '
                f'(quality {(pq <= G.ALPHA_Q).mean() * 100:.1f}%, OOD only '
                f'{((pq > G.ALPHA_Q) & (pood <= G.ALPHA_OOD)).mean() * 100:.1f}%) · CP coverage on the crops that '
                f'passed: {"-" if np.isnan(cov) else f"{cov * 100:.1f}%"} (target 90%)</p>'
                f'<h4>{caught_lbl}</h4><div class="grid">{mk(pick(np.flatnonzero(stopped), fam, rng))}</div>'
                f'<h4>{missed_lbl}</h4><div class="grid">{mk(pick(np.flatnonzero(passed), fam, rng))}</div>')

    med = ref.median()
    body = [f'<h1>Quality gate: caught vs not caught</h1>',
            '<p class="lede">Architecture A on d1_grouped (enhanced_res, no retraining). Synthetic corruptions of '
            f'{len(ev)} test-half crops (up to 8 per photo family, {len(set(fam))} families). Pipeline: quality gate '
            f'(α {G.ALPHA_Q}) → OOD gate (α {G.ALPHA_OOD}) → prob-T + family-weighted LAC (α {G.ALPHA_CP}). '
            'Examples are random, one per family where possible.</p>',
            '<p class="lede"><b>Reading the numbers.</b> sharp = variance of the Laplacian (low = blurry); bright = '
            'mean grey 0–255; white / black = share of pixels ≥ 250 / ≤ 5. Typical normal (val) crop: '
            f'sharp {med.sharpness:.0f}, bright {med.brightness:.0f}, white {med.clipped_hi * 100:.1f}%, '
            f'black {med.clipped_lo * 100:.1f}%. A crop is stopped for quality when its quality p-value ≤ '
            f'{G.ALPHA_Q}: its most extreme statistic is rarer than in all but ~10% of normal calibration families. '
            '"Holds truth" / "WRONG" says whether the visor message contains the true class.</p>',
            '<p class="lede">Synthetic only: no real helmet footage, and no partial shadows (the dark test darkens '
            'the whole crop).</p>']
    for title, names in GROUPS:
        body.append(f'<h2>{title}</h2>' + "".join(panel(n) for n in names))
    body.append('<h2>Normal crops the gate stopped anyway</h2>' + panel("clean"))

    css = """:root{--bg:#fff;--fg:#1c1c1e;--mut:#6b6b70;--card:#f5f5f7;--line:#e2e2e6;--stop:#c0392b;--aim:#b9770e;
--pass:#1e8449;--bad:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#ececef;--mut:#9a9aa0;--card:#232326;--line:#333338;
--stop:#ff6b5e;--aim:#f5b041;--pass:#58d68d;--bad:#ff6b5e}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif;max-width:1180px;
margin:0 auto;padding:24px 16px}
h1{font-size:26px}h2{margin-top:40px;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{margin:24px 0 2px}h3 small{color:var(--mut);font-weight:400}h4{margin:12px 0 6px;color:var(--mut)}
.lede,.sum{color:var(--mut);max-width:880px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));
gap:10px}.card{background:var(--card);border-radius:8px;padding:8px}.imgs{display:flex;gap:6px}
figure{margin:0;flex:1}img{width:100%;max-width:150px;border-radius:4px;display:block}
figcaption{font-size:11px;color:var(--mut)}.card p{margin:6px 0 0;font-size:12.5px}
.stop{color:var(--stop)}.aim{color:var(--aim)}.pass,.ok{color:var(--pass)}.bad{color:var(--bad);font-weight:600}
.none{color:var(--mut);font-style:italic}"""
    html = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
            f'initial-scale=1"><title>Quality Gate Examples</title><style>{css}</style></head><body>'
            + "".join(body) + "</body></html>")
    (OUT / "quality_examples.html").write_text(html)
    print("wrote", OUT / "quality_examples.html")


if __name__ == "__main__":
    main()
