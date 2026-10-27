"""Architecture A, step 2: the two gates that run before CP, tested on synthetic corruptions and off-target images.

From 1_main_model_resnet18/:
  python enhanced_res/arch_a/experiments/archA_gates.py           # features of corrupted / off-target crops are cached after the first run

Gate 1, image quality ("bad image: re-take"). Four numbers per crop, no model needed:
    sharpness   variance of the Laplacian of the grey image (low = blurry; Pech-Pacheco et al. 2000)
    brightness  mean grey level (too dark or too bright)
    clipped_hi  share of pixels >= 250 (blown out / glare)
    clipped_lo  share of pixels <= 5 (black)
  Each number is compared with the clean val crops (its tail probability: low sharpness, very low or very high
  brightness, blown-out pixels, black pixels), and the most extreme of the four tails is the crop's quality score.
  A single family-weighted conformal p-value of that score against the clean cal-half crops is the gate (flag when
  p <= alpha_q). One combined score instead of four Bonferroni tests: with 88 calibration families the smallest
  possible p-value is 1/89 = 0.011, above a four-way Bonferroni cut-off of 0.025/4.
  Tried for glare and dropped (2026-10-26): the brightest smoothed spot minus the median grey level (AUROC 0.63-0.73
  on glare; it diluted the gate) and the largest near-white share in any 56 x 56 window (as good as clipped_hi on
  glare alone, but replacing clipped_hi lost overexposure detection, 46% -> 24%). Every extra statistic makes normal
  crops look extreme more often, which raises the bar for all of them. Synthetic glare stays mostly undetected at a
  strict budget because clean white-skin crops often contain near-white patches too.

Gate 2, unfamiliar input ("can't assess: re-aim"). Score = 1 - mean cosine similarity to the 5 nearest of the
  20,179 training-crop features already stored in enhanced_res (Sun et al. 2022). Conformal p-value against the
  clean cal-half crops; flag when p <= alpha_ood. Baselines, same p-value recipe: energy of the heads' averaged logits
  (Liu et al. 2020) and 1 - top probability (MSP; Hendrycks & Gimpel 2017).

p-values are family-weighted like A step 1: every calibration photo family counts once. alpha_q = 0.10 (at 0.025 the
quality gate missed almost all glare; at 0.10 it catches most of it). alpha_ood = 0.10 since 2026-10-27 (was 0.025: 40% of
off-target patches got through both gates, 17.5% now); together roughly 20% of normal
crops are stopped. operating_points.csv shows other budgets.

Test sets
  clean       the test half's crops (sample: up to 8 per photo family)
  14 corruptions of those same crops: gaussian blur, motion blur, dark, overexposed, glare, noise, JPEG x 2 severities
  off-target  random patches from 138 non-aircraft images (macOS wallpapers + scikit-learn's two sample photos);
              landscapes, sky, water, rock and graphic patterns, not hangar floors or gloves
Then the full pipeline (quality gate -> OOD gate -> CP set with prob-T and family-weighted LAC at alpha 0.10) is run on
every set: how much is stopped, and how CP coverage changes on what gets through.

Writes enhanced_res/arch_a/results/archA/step2/ and data/work/enhanced_res/calibration/archA_gates.json.
"""
import json
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import roc_auc_score

import arch0_cp as A0
import archA_cp as A1
from model import MAIN, EnhancedRes, load_split
from klshift.common import eval_transform
from klshift.mlp import knn_proba, l2n

CACHE = A0.CACHE
OUT = HERE.parent / "results" / "archA" / "step2"
OFFTARGET_DIR = CACHE / "offtarget_src"
ALPHA_Q, ALPHA_OOD = 0.10, 0.10      # quality 0.10 so glare is caught (2026-10-26); OOD 0.10 for fewer false accepts (2026-10-27)
ALPHA_CP = 0.10
PER_FAMILY, N_OFF, SEED = 8, 400, 0

CORRUPTIONS = {
    "blur s1": lambda im, r: im.filter(ImageFilter.GaussianBlur(2)),
    "blur s2": lambda im, r: im.filter(ImageFilter.GaussianBlur(4)),
    "motion blur s1": lambda im, r: motion(im, 9, r),
    "motion blur s2": lambda im, r: motion(im, 21, r),
    "dark s1": lambda im, r: ImageEnhance.Brightness(im).enhance(0.35),
    "dark s2": lambda im, r: ImageEnhance.Brightness(im).enhance(0.12),
    "overexposed s1": lambda im, r: ImageEnhance.Brightness(im).enhance(1.8),
    "overexposed s2": lambda im, r: ImageEnhance.Brightness(im).enhance(3.0),
    "glare s1": lambda im, r: glare(im, 0.7, r),
    "glare s2": lambda im, r: glare(im, 1.0, r),
    "noise s1": lambda im, r: noise(im, 20, r),
    "noise s2": lambda im, r: noise(im, 45, r),
    "jpeg s1": lambda im, r: jpeg(im, 12),
    "jpeg s2": lambda im, r: jpeg(im, 4),
}


def motion(im, k, rng):
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - .5, k / 2 - .5), float(rng.uniform(0, 180)), 1.0)
    ker = cv2.warpAffine(ker, M, (k, k))
    ker /= ker.sum()
    return Image.fromarray(cv2.filter2D(np.asarray(im), -1, ker))


def glare(im, strength, rng):
    a = np.asarray(im).astype(np.float32)
    h, w = a.shape[:2]
    cy, cx = rng.uniform(0.2, 0.8, 2) * (h, w)
    yy, xx = np.mgrid[:h, :w]
    blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (0.25 * h) ** 2)))[..., None]
    return Image.fromarray(np.clip(a + strength * 255 * blob, 0, 255).astype(np.uint8))


def noise(im, sigma, rng):
    a = np.asarray(im).astype(np.float32) + rng.normal(0, sigma, np.asarray(im).shape)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def jpeg(im, q):
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return Image.fromarray(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))


def offtarget_patches(n, rng):
    files = sorted(OFFTARGET_DIR.glob("*.jpg"))
    out = []
    for i in range(n):
        im = Image.open(files[i % len(files)]).convert("RGB")
        s = int(min(im.size) * rng.uniform(0.35, 0.9))
        x, y = rng.integers(0, im.width - s + 1), rng.integers(0, im.height - s + 1)
        out.append(im.crop((x, y, x + s, y + s)).resize((224, 224), Image.BILINEAR))
    return out


# ------------------------------------------------------------------ quality statistics
def quality_stats(images):
    rows = []
    for im in images:
        g = np.asarray(im.convert("L"), dtype=np.float32)
        rows.append({"sharpness": float(cv2.Laplacian(g, cv2.CV_32F).var()), "brightness": float(g.mean()),
                     "clipped_hi": float((g >= 250).mean()), "clipped_lo": float((g <= 5).mean())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ family-weighted conformal p-values
def pvalue(cal_scores, cal_groups, scores):
    """Upper-tail p-value: large score = unusual. Each calibration family carries total weight 1."""
    _, inv, cnt = np.unique(cal_groups, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    order = np.argsort(cal_scores)
    s, cw = cal_scores[order], np.cumsum(w[order][::-1])[::-1]          # cw[i] = weight of scores >= s[i]
    idx = np.searchsorted(s, scores, side="left")
    above = np.where(idx < len(s), cw[np.minimum(idx, len(s) - 1)], 0.0)
    return (1.0 + above) / (len(cnt) + 1.0)


STATS = ("sharpness", "brightness", "clipped_hi", "clipped_lo")


def tail(ref, v, side):
    """Share of reference crops at least as extreme as v ('low', 'high' or 'both' tail)."""
    r = np.sort(ref)
    lo = np.searchsorted(r, v, side="right") / len(r)
    hi = 1 - np.searchsorted(r, v, side="left") / len(r)
    return {"low": lo, "high": hi, "both": np.minimum(1, 2 * np.minimum(lo, hi))}[side]


SIDES = {"sharpness": "low", "brightness": "both", "clipped_hi": "high", "clipped_lo": "high"}


def quality_score(ref, q):
    """-log of the most extreme tail probability over the statistics; also which statistic it was."""
    t = np.vstack([np.maximum(tail(ref[s].values, q[s].values, SIDES[s]), 1e-6) for s in STATS])
    return -np.log(t.min(0)), np.array(STATS)[t.argmin(0)]


def quality_pvalue(ref, cal, cal_g, q):
    s_cal, _ = quality_score(ref, cal)
    s, why = quality_score(ref, q)
    return pvalue(s_cal, cal_g, s), why


# ------------------------------------------------------------------ model outputs for in-memory images
@torch.no_grad()
def model_outputs(m, images, batch=64):
    tf = eval_transform()
    F = [m.features(torch.stack([tf(im) for im in images[i:i + batch]])) for i in range(0, len(images), batch)]
    f = torch.cat(F)
    logits = torch.stack([h(f) for h in m.heads])
    knn = knn_proba(m.bank, m.bank_y, l2n(f), 5, len(m.classes))
    return f.cpu().numpy(), logits.cpu().numpy(), knn.cpu().numpy()


def scores(m_bank, feats, logits, probs):
    f = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    sim = torch.from_numpy(f) @ m_bank.T
    knn = 1 - sim.topk(5, dim=1).values.mean(1).numpy()
    z = logits.mean(0)
    energy = -(np.log(np.exp(z - z.max(1, keepdims=True)).sum(1)) + z.max(1))
    return {"kNN distance": knn, "energy": energy, "1 - top probability": 1 - probs.max(1)}


def prepare():
    """Everything both the gate evaluation and the visor pipeline need: data, sample, corrupted sets, cached model
    outputs, and the gate calibration (clean cal half; quality reference = clean val crops)."""
    rng = np.random.default_rng(SEED)
    crops = load_split()
    te, ft, Lt, Kt = A1.model_parts(crops, "test")
    va, fv, Lv, Kv = A1.model_parts(crops, "val")
    classes = list(np.load(CACHE / "feats_test.npz", allow_pickle=True)["classes"])
    cid = {c: i for i, c in enumerate(classes)}
    T_prob = json.loads((CACHE / "archA_step1.json").read_text())["T_prob"]

    sp = pd.read_csv(HERE.parent / "cal_split.csv")
    sp["crop_path"] = [str(MAIN / p) for p in sp.crop_path]
    te = te.merge(sp[["crop_path", "split"]].rename(columns={"split": "half"}), on="crop_path", validate="1:1")
    is_cal = (te.half == "cal").to_numpy()
    y, g = te.cls.map(cid).to_numpy(), te.src_group.to_numpy()
    P = A1.temper(A1.blend(Lt, Kt), T_prob)

    # evaluation sample from the test half
    ev = te[~is_cal].groupby("src_group", group_keys=False).apply(
        lambda d: d.sample(min(len(d), PER_FAMILY), random_state=SEED)).index.to_numpy()
    print(f"evaluation sample: {len(ev)} test-half crops from {te.loc[ev].src_group.nunique()} families")

    # images; corruption randomness seeded per corruption name with a stable hash (Python's hash() changes per run)
    cal_imgs = [Image.open(p).convert("RGB") for p in te.crop_path[is_cal]]
    ev_imgs = [Image.open(p).convert("RGB") for p in te.crop_path[ev]]
    sets = {"clean": ev_imgs}
    for name, fn in CORRUPTIONS.items():
        r = np.random.default_rng(zlib.crc32(name.encode()))
        sets[name] = [fn(im, r) for im in ev_imgs]
    sets["off-target"] = offtarget_patches(N_OFF, rng)

    # model outputs (cached)
    m = EnhancedRes()
    cache = CACHE / "gates_feats.npz"
    key = f"v2-crc32-{len(ev)}-{PER_FAMILY}-{N_OFF}-{SEED}-" + "|".join(CORRUPTIONS)
    store = dict(np.load(cache, allow_pickle=True)) if cache.exists() else {}
    if str(store.get("key", "")) != key:
        store = {"key": key}
        for name, ims in sets.items():
            if name == "clean":
                continue
            print(f"  features: {name} ({len(ims)})", flush=True)
            f, L, K = model_outputs(m, ims)
            store[f"{name}|f"], store[f"{name}|L"], store[f"{name}|K"] = f, L, K
        np.savez(cache, **store)
    out = {"clean": (ft[ev], Lt[:, ev], Kt[ev])}
    for name in sets:
        if name != "clean":
            out[name] = (store[f"{name}|f"], store[f"{name}|L"], store[f"{name}|K"])

    bank = m.bank.cpu()
    q_cal = quality_stats(cal_imgs)
    q_ref = quality_stats([Image.open(p).convert("RGB") for p in va.crop_path])
    s_cal = scores(bank, ft[is_cal], Lt[:, is_cal], P[is_cal])
    return dict(te=te, is_cal=is_cal, y=y, g=g, P=P, ft=ft, Lt=Lt, classes=classes, T_prob=T_prob, ev=ev, sets=sets, out=out,
                bank=bank, q_cal=q_cal, q_ref=q_ref, s_cal=s_cal, g_cal=g[is_cal])


def gate_pvalues(D, name):
    """Quality p-value (and its reason), kNN-OOD p-value and tempered probabilities for one evaluation set."""
    f, L, K = D["out"][name]
    p = A1.temper(A1.blend(L, K), D["T_prob"])
    pq, why = quality_pvalue(D["q_ref"], D["q_cal"], D["g_cal"], quality_stats(D["sets"][name]))
    sc = scores(D["bank"], f, L, p)
    pv = {k: pvalue(D["s_cal"][k], D["g_cal"], v) for k, v in sc.items()}
    return p, pq, why, sc, pv


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    D = prepare()
    te, is_cal, y, g, P, ev, sets = D["te"], D["is_cal"], D["y"], D["g"], D["P"], D["ev"], D["sets"]
    q_ref, q_cal, g_cal, T_prob = D["q_ref"], D["q_cal"], D["g_cal"], D["T_prob"]

    # CP threshold (A step 1 default): prob-T + family-weighted LAC
    q_cp = A1.family_threshold(P[is_cal], y[is_cal], g[is_cal], ALPHA_CP)

    rows, auroc, gate_p = [], [], {}
    clean_scores = None
    for name, ims in sets.items():
        p, pq, why, sc, pv = gate_pvalues(D, name)
        if name == "clean":
            clean_scores, clean_pq = sc, pq
        gate_p[name] = (pq, pv["kNN distance"])
        flag_q = pq <= ALPHA_Q
        flag_o = pv["kNN distance"] <= ALPHA_OOD
        retake, reaim = flag_q, ~flag_q & flag_o
        passed = ~flag_q & ~flag_o
        r = {"set": name, "crops": len(ims), "re-take (quality)": flag_q.mean(), "re-aim (kNN OOD)": reaim.mean(),
             "stopped total": (flag_q | flag_o).mean(), "passed": passed.mean(),
             "kNN gate alone": flag_o.mean(), "energy gate alone": (pv["energy"] <= ALPHA_OOD).mean(),
             "MSP gate alone": (pv["1 - top probability"] <= ALPHA_OOD).mean(),
             "top quality reason": pd.Series(why[flag_q]).value_counts().index[0] if flag_q.any() else "-"}
        if name != "off-target":
            yy = y[ev]
            S = A0.lac_sets(p, q_cp)
            hit = S[np.arange(len(yy)), yy]
            r.update({"accuracy (all)": (p.argmax(1) == yy).mean(), "CP coverage (all)": hit.mean(),
                      "CP coverage (passed)": hit[passed].mean() if passed.any() else np.nan,
                      "mean set size (passed)": S[passed].sum(1).mean() if passed.any() else np.nan})
        else:
            S = A0.lac_sets(p, q_cp)
            r.update({"mean set size (passed)": S[passed].sum(1).mean() if passed.any() else np.nan})
        rows.append(r)
        if name != "clean":
            lab = np.r_[np.zeros(len(ev)), np.ones(len(ims))]
            a = {"set": name}
            for k in sc:
                a[k] = roc_auc_score(lab, np.r_[clean_scores[k], sc[k]])
            a["quality score"] = roc_auc_score(lab, np.r_[-clean_pq, -pq])
            auroc.append(a)
    res = pd.DataFrame(rows).set_index("set")
    # operating points: share stopped by each gate (and both together) at other false-alarm budgets
    ops = pd.DataFrame({f"{k} a={a}": {n: (fn(pq, po, a)).mean() for n, (pq, po) in gate_p.items()}
                        for a in (0.025, 0.05, 0.10)
                        for k, fn in [("quality", lambda pq, po, a: pq <= a), ("OOD", lambda pq, po, a: po <= a),
                                      ("both", lambda pq, po, a: (pq <= a) | (po <= a))]})
    au = pd.DataFrame(auroc).set_index("set")

    (CACHE / "archA_gates.json").write_text(json.dumps({
        "alpha_quality": ALPHA_Q, "alpha_ood": ALPHA_OOD, "alpha_cp": ALPHA_CP, "T_prob": T_prob, "q_cp": q_cp,
        "ood_score": "1 - mean cosine similarity to 5 nearest training features",
        "calibration": "clean cal-half crops, family-weighted conformal p-values; quality reference = clean val crops"}, indent=2))

    # ------------------------------------------------------------------ figure: example crops per set
    names = list(sets)
    fig, ax = plt.subplots(3, len(names) // 3 + 1, figsize=(16, 7.5))
    for a, name in zip(ax.ravel(), names):
        a.imshow(sets[name][3]); a.set_title(f"{name}\nstopped {res.loc[name, 'stopped total'] * 100:.0f}%", fontsize=8)
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "examples.png", dpi=120); plt.close(fig)

    # ------------------------------------------------------------------ report
    pct = lambda v: "-" if pd.isna(v) else f"{v * 100:.1f}%"
    md = ["# Architecture A, step 2: quality gate + unfamiliar-input gate", "",
          f"Gates calibrated on the clean cal-half crops (family-weighted conformal p-values), alpha_quality = "
          f"{ALPHA_Q}, alpha_OOD = {ALPHA_OOD}. Evaluated on {len(ev)} test-half crops (up to {PER_FAMILY} per family, "
          f"{te.loc[ev].src_group.nunique()} families), 14 synthetic corruptions of those same crops, and {N_OFF} "
          "off-target patches. CP after the gates: prob-T + family-weighted LAC, alpha 0.10. No retraining.", "",
          "## What happens to each set", "",
          "| set | re-take (quality) | re-aim (OOD) | stopped | passed | top quality reason | accuracy | CP coverage, all | CP coverage, passed | set size, passed |",
          "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"]
    for n, r in res.iterrows():
        md.append(f"| {n} | {pct(r['re-take (quality)'])} | {pct(r['re-aim (kNN OOD)'])} | {pct(r['stopped total'])} | "
                  f"{pct(r['passed'])} | {r['top quality reason']} | {pct(r.get('accuracy (all)'))} | "
                  f"{pct(r.get('CP coverage (all)'))} | {pct(r.get('CP coverage (passed)'))} | "
                  f"{'-' if pd.isna(r['mean set size (passed)']) else f'{r['mean set size (passed)']:.2f}'} |")
    md += ["", "## OOD score comparison: share flagged at the same p-value cut-off", "",
           "| set | kNN distance | energy | 1 - top probability |", "|---|---:|---:|---:|"]
    for n, r in res.iterrows():
        md.append(f"| {n} | {pct(r['kNN gate alone'])} | {pct(r['energy gate alone'])} | {pct(r['MSP gate alone'])} |")
    md += ["", "## Operating points: share stopped at other false-alarm budgets (alpha per gate)", "",
           "The clean row is the false-alarm rate on normal crops; every other row is what gets caught. alpha should "
           "come from GAES's costs: a missed bad image against interrupting the inspector.", "",
           "| set | " + " | ".join(ops.columns) + " |", "|---|" + "---:|" * len(ops.columns)]
    for n, r in ops.iterrows():
        md.append(f"| {n} | " + " | ".join(pct(v) for v in r) + " |")
    md += ["", "## AUROC: how well each score separates clean crops from each set (0.5 = chance, 1 = perfect)", "",
           "| set | " + " | ".join(au.columns) + " |", "|---|" + "---:|" * len(au.columns)]
    for n, r in au.iterrows():
        md.append(f"| {n} | " + " | ".join(f"{v:.3f}" for v in r) + " |")
    md += ["", "![examples](examples.png)", ""]
    (OUT / "README.md").write_text("\n".join(md))
    res.to_csv(OUT / "gates.csv")
    au.to_csv(OUT / "auroc.csv")
    ops.to_csv(OUT / "operating_points.csv")
    print("\n".join(md))


if __name__ == "__main__":
    main()
