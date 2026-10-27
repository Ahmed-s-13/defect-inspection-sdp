"""Does random static (and other images that have nothing to do with aircraft) get through architecture A's gates,
and which stricter setting stops it at what cost in false rejects of normal crops?

From 1_main_model_resnet18/:
  python enhanced_res/arch_a/experiments/archA_static.py

Nonsense set (seeded, 224 px, 40 of each): colour static, grey static, gaussian static, salt & pepper, blocky static,
smooth noise blobs, solid colour, two-colour gradient. Compared with the off-target patches and heavy noise / JPEG
damage from archA_gates.py, and the false-reject rate on all 2,417 normal test-half crops (per crop and per family).

Options (every one recalibrated on the clean cal half, so its false reject budget stays alpha per family):
  previous          quality alpha 0.10 (low sharpness only)       + OOD alpha 0.025 (OOD 0.10 is the default now)
  OOD 0.05 / 0.10   looser OOD threshold
  sharpness both    the quality gate also flags sharpness far ABOVE normal (static is all edges)
  combinations
Writes enhanced_res/arch_a/results/archA/static/ (README.md, rates.csv, examples.png).
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

import archA_cp as A1
import archA_gates as G
from model import EnhancedRes

OUT = HERE.parent / "results" / "archA" / "static"
N_EACH, SEED = 40, 0


def nonsense(rng):
    s = 224
    up = lambda a, k, how: np.asarray(Image.fromarray(a).resize((s, s), how))
    kinds = {
        "colour static": lambda: rng.integers(0, 256, (s, s, 3), dtype=np.uint8),
        "grey static": lambda: np.repeat(rng.integers(0, 256, (s, s, 1), dtype=np.uint8), 3, 2),
        "gaussian static": lambda: np.clip(rng.normal(128, 50, (s, s, 3)), 0, 255).astype(np.uint8),
        "salt & pepper": lambda: np.repeat((rng.random((s, s, 1)) < .5).astype(np.uint8) * 255, 3, 2),
        "blocky static": lambda: up(rng.integers(0, 256, (28, 28, 3), dtype=np.uint8), 28, Image.NEAREST),
        "smooth noise blobs": lambda: up(rng.integers(0, 256, (7, 7, 3), dtype=np.uint8), 7, Image.BICUBIC),
        "solid colour": lambda: np.broadcast_to(rng.integers(0, 256, 3, dtype=np.uint8), (s, s, 3)).copy(),
        "gradient": lambda: (np.linspace(0, 1, s)[None, :, None] * rng.integers(0, 256, 3)
                             + np.linspace(1, 0, s)[None, :, None] * rng.integers(0, 256, 3)).repeat(s, 0).astype(np.uint8),
    }
    return {k: [Image.fromarray(f()) for _ in range(N_EACH)] for k, f in kinds.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    D = G.prepare()
    te, is_cal, g, T = D["te"], D["is_cal"], D["g"], D["T_prob"]
    tm = ~is_cal
    m = EnhancedRes()

    # ---- every evaluation set: quality statistics + kNN distance
    sets = {}
    imgs_n = [Image.open(p).convert("RGB") for p in te.crop_path[tm]]
    sets["normal crops (all 2,417)"] = dict(stats=G.quality_stats(imgs_n), dist=G.scores(
        D["bank"], D["ft"][tm], D["Lt"][:, tm], D["P"][tm])["kNN distance"], groups=g[tm])
    for name in ("off-target", "noise s2", "jpeg s2"):
        f, L, K = D["out"][name]
        sets[name] = dict(stats=G.quality_stats(D["sets"][name]),
                          dist=G.scores(D["bank"], f, L, A1.temper(A1.blend(L, K), T))["kNN distance"])
    junk = nonsense(rng)
    for name, ims in junk.items():
        f, L, K = G.model_outputs(m, ims)
        sets[name] = dict(stats=G.quality_stats(ims), dist=G.scores(D["bank"], f, L, A1.temper(A1.blend(L, K), T))["kNN distance"],
                          imgs=ims, top=[m.classes[i] for i in A1.temper(A1.blend(L, K), T).argmax(1)])
    cal_dist, g_cal = D["s_cal"]["kNN distance"], D["g_cal"]

    # ---- options
    options = {"previous (quality 0.10, OOD 0.025)": ("low", 0.10, 0.025),
               "OOD 0.05": ("low", 0.10, 0.05),
               "OOD 0.10 (default since 2026-10-27)": ("low", 0.10, 0.10),
               "sharpness both sides": ("both", 0.10, 0.025),
               "sharpness both sides + OOD 0.05": ("both", 0.10, 0.05)}
    rows, stop = [], {}
    for opt, (side, aq, ao) in options.items():
        G.SIDES["sharpness"] = side
        q_cal, _ = G.quality_score(D["q_ref"], D["q_cal"])
        for name, s in sets.items():
            sq, why = G.quality_score(D["q_ref"], s["stats"])
            pq = G.pvalue(q_cal, g_cal, sq)
            po = G.pvalue(cal_dist, g_cal, s["dist"])
            stopped = (pq <= aq) | (po <= ao)
            stop[(opt, name)] = (pq <= aq, po <= ao)
            r = {"option": opt, "set": name}
            if "groups" in s:
                r["false reject (per crop)"] = stopped.mean()
                r["false reject (per family)"] = pd.Series(stopped).groupby(s["groups"]).mean().mean()
            else:
                r["false accept (passes both gates)"] = 1 - stopped.mean()
            rows.append(r)
    G.SIDES["sharpness"] = "low"
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "rates.csv", index=False)

    # ---- examples figure: current setting
    kinds = list(junk)
    fig, axs = plt.subplots(2, len(kinds), figsize=(2.1 * len(kinds), 4.9))
    cur = list(options)[0]
    for j, k in enumerate(kinds):
        qf, of = stop[(cur, k)]
        for r, i in enumerate((0, 1)):
            ax = axs[r, j]
            ax.imshow(junk[k][i]); ax.set_xticks([]); ax.set_yticks([])
            verdict = "re-take" if qf[i] else "re-aim" if of[i] else f"PASSED: {sets[k]['top'][i]}"
            ax.set_title(f"{k}\n{verdict}" if r == 0 else verdict, fontsize=7, color="C2" if (qf[i] or of[i]) else "C3")
    fig.suptitle("Nonsense images under the previous gates (OOD 0.025) (green = stopped, red = shown as a defect)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "examples.png", dpi=120); plt.close(fig)

    # ---- report
    pct = lambda v: "-" if pd.isna(v) else f"{v * 100:.1f}%"
    piv = res.pivot(index="set", columns="option", values="false accept (passes both gates)").reindex(
        [n for n in sets if "groups" not in sets[n]])[list(options)]
    fr = res[res.set.str.startswith("normal")].set_index("option").loc[list(options)]
    md = ["# Random static and other nonsense images vs architecture A's gates", "",
          f"{N_EACH} images of each kind (224 px, seeded). Every option is recalibrated on the clean cal half. "
          "False reject = normal test-half crops stopped; false accept = nonsense that passes both gates and is "
          "shown as a defect or an ambiguous set.", "",
          "## False rejects on normal crops (the cost)", "",
          "| option | per crop | per family |", "|---|---:|---:|"]
    for o, r in fr.iterrows():
        md.append(f"| {o} | {pct(r['false reject (per crop)'])} | {pct(r['false reject (per family)'])} |")
    md += ["", "## False accepts (passes both gates)", "",
           "| set | " + " | ".join(options) + " |", "|---|" + "---:|" * len(options)]
    for n, r in piv.iterrows():
        md.append(f"| {n} | " + " | ".join(pct(v) for v in r) + " |")
    md += ["", "![examples](examples.png)", ""]
    (OUT / "README.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
