"""The photo families whose normal crops architecture A stops most often (false rejects), as a web page.

From 1_main_model_resnet18/ (after calibrate.py --model v2 and archA_pooled_cal.py, whose caches it reuses):
  python enhanced_res/arch_a/experiments/archA_reject_families.py

Report only: the test half (87 families) with the deployed calibration.json; nothing is chosen from it.
Shows the TOP families by number of stopped crops: every stopped crop, plus a few that passed, with what the visor shows.
Writes enhanced_res/arch_a/results/archA/examples/reject_families.html (self-contained, images embedded).
"""
import base64
import io
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE))

import numpy as np
from PIL import Image

import archA_pooled_cal as PC
from arch_a import calibrate as C
from arch_a import runtime as R
from model import load_split

TOP, PASSED_SHOWN = 5, 12
OUT = HERE.parent / "results" / "archA" / "examples" / "reject_families.html"


def b64(path, size=150):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    a = R.ArchA(R.CALIBRATIONS["v2"])
    cr = load_split("d1_grouped")
    va = cr[cr.split == "val"].reset_index(drop=True)
    tr = cr[cr.split == "train"].reset_index(drop=True)
    te = cr[cr.split == "test"].reset_index(drop=True)
    te["half"] = C.halves(te)
    cal = te[te.half == "cal"].reset_index(drop=True)
    tst = te[te.half == "test"].reset_index(drop=True)
    paths = list(va.crop_path) + list(cal.crop_path) + list(tst.crop_path)
    nv, nc, n3 = len(va), len(cal), len(paths)
    z = np.load(C.CACHE["v2"], allow_pickle=True)
    sl = slice(nv + nc, n3)
    st = PC.quality_stats(paths + list(tr.crop_path))
    S = {s: st[s][sl] for s in R.STATS}
    sc, why = R.quality_score(a.ref, S, a.sides)
    pq = R.pvalue(a.q_cal, a.q_groups, sc)
    po, part = a.ood(z["knn_dist"][sl], z["feats"][sl])
    p, sets = a.sets(z["probs"][sl])
    stop = (pq <= a.alpha_q) | (po <= a.alpha_ood)
    tst["stop"] = stop
    counts = tst.groupby("src_group").stop.agg(["sum", "size"]).sort_values(["sum", "size"], ascending=[False, True])
    fams = counts.head(TOP)
    total = int(stop.sum())

    cards = []
    for rank, (g, row) in enumerate(fams.iterrows(), 1):
        idx = np.flatnonzero(tst.src_group.to_numpy() == g)
        idx = np.r_[idx[stop[idx]], idx[~stop[idx]][:PASSED_SHOWN]]       # every stopped crop, then a few that passed
        cls = tst.cls[idx].mode()[0]
        items = []
        for i in idx:
            true = tst.cls[i]
            if po[i] <= a.alpha_ood:
                verdict = (f'<b class="aim">Re-aim</b> · {"not aircraft skin (AoN)" if part[i] == "aon" else "unlike any training crop (kNN)"}'
                           f'<br>OOD p {po[i]:.3f}')
            elif pq[i] <= a.alpha_q:
                verdict = f'<b class="stop">Re-take</b> · {a._reason(why[i], S[why[i]][i])}<br>quality p {pq[i]:.3f}'
            else:
                k = [c for c in np.argsort(-p[i]) if sets[i, c]]
                shown = (f"{a.classes[k[0]]} {p[i, k[0]] * 100:.0f}%" if len(k) == 1
                         else "ambiguous: " + " / ".join(a.classes[c] for c in k))
                ok = cid(a, true) in k
                verdict = (f'<b class="pass">Passed</b> · {shown} '
                           f'<span class="{"ok" if ok else "bad"}">{"✓" if ok else "wrong"}</span>')
            stats = (f"sharp {S['sharpness'][i]:.0f} · bright {S['brightness'][i]:.0f} · "
                     f"white {S['clipped_hi'][i] * 100:.0f}% · black {S['clipped_lo'][i] * 100:.0f}%")
            items.append(f'<figure class="{"s" if stop[i] else "p"}"><img src="{b64(tst.crop_path[i])}" alt="{true} crop">'
                         f'<figcaption><span class="true">{true}</span><br>{verdict}<br><small>{stats}</small>'
                         f'</figcaption></figure>')
        cards.append(f'<section><h2>{rank}. Family {g} <small>{cls} · {int(row["sum"])} of {int(row["size"])} crops '
                     f'stopped ({row["sum"] / row["size"] * 100:.0f}%); showing all stopped and '
                     f'{min(PASSED_SHOWN, int(row["size"] - row["sum"]))} that passed</small></h2>'
                     f'<div class="grid">{"".join(items)}'
                     f'</div></section>')

    share = fams["sum"].sum() / total * 100
    med = {s: float(np.median(a.ref[s])) for s in R.STATS}
    css = """:root{--bg:#fff;--fg:#1c1c1e;--mut:#6b6b70;--card:#f5f5f7;--line:#e2e2e6;--stop:#c0392b;--aim:#b9770e;
--pass:#1e8449;--bad:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#161618;--fg:#ececef;--mut:#9a9aa0;--card:#232326;--line:#333338;
--stop:#ff6b5e;--aim:#f5b041;--pass:#58d68d;--bad:#ff6b5e}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 -apple-system,system-ui,sans-serif;max-width:1180px;
margin:0 auto;padding:24px 16px}h1{font-size:26px;margin-bottom:4px}h2{margin-top:36px;border-bottom:1px solid var(--line);
padding-bottom:6px;font-size:19px}h2 small{color:var(--mut);font-weight:400;font-size:14px}.lede{color:var(--mut);max-width:880px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:10px}
figure{margin:0;background:var(--card);border-radius:8px;padding:8px;border-top:3px solid var(--line)}
figure.s{border-top-color:var(--stop)}img{width:100%;border-radius:4px;display:block}
figcaption{font-size:12px;margin-top:6px}figcaption small{color:var(--mut)}.true{font-weight:600}
.stop{color:var(--stop)}.aim{color:var(--aim)}.pass,.ok{color:var(--pass)}.bad{color:var(--bad);font-weight:600}"""
    html = (f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,'
            f'initial-scale=1"><title>False Reject Families</title><style>{css}</style></head><body>'
            f'<h1>Where the false rejects are</h1>'
            f'<p class="lede">Architecture A (enhanced_res v2, quality α {a.alpha_q}, OOD α {a.alpha_ood}) on the test '
            f'half: {len(tst)} normal defect crops from {tst.src_group.nunique()} photo families, report only. '
            f'{total} crops are stopped; these {TOP} families hold {int(fams["sum"].sum())} of them ({share:.0f}%). '
            'For each family: every stopped crop (red line on top), then up to 12 that passed, for comparison.</p>'
            f'<p class="lede">Typical normal crop: sharp {med["sharpness"]:.0f}, bright {med["brightness"]:.0f}, '
            f'white {med["clipped_hi"] * 100:.1f}%, black {med["clipped_lo"] * 100:.1f}%. '
            'white / black = share of pixels ≥ 250 / ≤ 5.</p>' + "".join(cards) + "</body></html>")
    OUT.write_text(html)
    print("wrote", OUT, f"({OUT.stat().st_size / 1e6:.1f} MB)")
    print(fams.to_string())


def cid(a, c):
    return a.classes.index(c)


if __name__ == "__main__":
    main()
