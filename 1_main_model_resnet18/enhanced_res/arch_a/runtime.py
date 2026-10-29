"""Architecture A at prediction time: two gates, temperature scaling and a conformal prediction set on top of
enhanced_res. The model is unchanged; everything here reads numbers fitted once by calibrate.py
(calibration.json, in git).

Per crop (both gates are checked; if both fire, the visor says "re-aim"):
  1. quality gate        4 pixel statistics -> conformal p-value; p <= alpha_quality   -> "BAD IMAGE: re-take"
  2. (model)             enhanced_res probabilities and its 768-d features
  3. unfamiliar-input    two scores on the model's 768-d features, each a conformal p-value, combined by
     (OOD) gate          Bonferroni (p = 2 x the smaller one; p <= alpha_ood -> "CAN'T ASSESS: re-aim"):
                           kNN distance       1 - mean cosine similarity to the 5 nearest train features
                           AoN                logistic regression (aircraft or not), D1 crops vs people / hands / vehicles / scenes
  4. temperature         p ** (1 / T), renormalised (T fitted on val)                 -> the shown percentage
  5. CP set (LAC)        every class with tempered p >= 1 - q[alpha]                  -> 1 class: that class
                                                                                         2+: "AMBIGUOUS"
                         an empty set keeps its top class (only at larger alphas, e.g. 0.20; coverage can only
                         rise), so "UNCLEAR" is no longer produced; the action stays defined for older callers
Models with a "no-defect" class (architecture B, gc7): the quality gate can be switched off in the calibration file
(it is legacy there), and the set is read for the visor as
  {no-defect}                -> nothing shown (SAQR: no message when there is no damage)
  {defect(s) + no-defect}    -> "POSSIBLE <DEFECT>: check"
  defects only               -> as above
An optional miss guard (calibration "miss_guard") only allows "nothing shown" when a one-sided conformal test on
the defect calibration crops rejects "this is a defect"; otherwise the crop is shown as "POSSIBLE".
numpy + PIL only (no PyTorch, no OpenCV), so trt_enhanced_res/predict_trt.py and predict_onnx.py (Raspberry Pi) use it too.

  from enhanced_res.arch_a.runtime import ArchA
  a = ArchA()                                   # loads calibration.json (v2); alpha 0.10 by default
  a = ArchA(CALIBRATIONS["gc7"])                # the main model's (gc7, 7 classes)
  out = a.decide(probs, crops=pil_crops, knn_dist=d)   # list of Decision, one per crop
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CALIBRATION = Path(__file__).resolve().parent / "calibration.json"             # v2 (6 classes)
CALIBRATIONS = {"v2": CALIBRATION, "gc7": CALIBRATION.with_name("calibration_gc7.json"),
                "gc7_fix": CALIBRATION.with_name("calibration_gc7_fix.json")}   # one per model version
STATS = ("sharpness", "brightness", "clipped_hi", "clipped_lo")
SIDES = {"sharpness": "low", "brightness": "both", "clipped_hi": "high", "clipped_lo": "high"}

SHOW, AMBIGUOUS, UNCLEAR, RETAKE, REAIM = "show class", "ambiguous", "unclear", "re-take", "re-aim"
NOTHING, POSSIBLE = "nothing shown", "possible defect"     # models with a "no-defect" class only
ACTIONS = (SHOW, AMBIGUOUS, UNCLEAR, RETAKE, REAIM, NOTHING, POSSIBLE)
NO_DEFECT = "no-defect"


# ------------------------------------------------------------------ quality statistics
def laplacian_var(g):
    """Variance of the 3x3 Laplacian of a grey image, = cv2.Laplacian(g, cv2.CV_32F).var() (reflect-101 border)."""
    p = np.pad(g, 1, mode="reflect")
    lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * p[1:-1, 1:-1]
    return float(lap.var())


def quality_stats(crop):
    """The 4 statistics of one crop (PIL image or HxWx3 uint8 array), on the grey image (PIL's L conversion)."""
    from PIL import Image
    im = crop if isinstance(crop, Image.Image) else Image.fromarray(np.asarray(crop, np.uint8))
    g = np.asarray(im.convert("L"), dtype=np.float32)
    return {"sharpness": laplacian_var(g), "brightness": float(g.mean()),
            "clipped_hi": float((g >= 250).mean()), "clipped_lo": float((g <= 5).mean())}


def tail(ref_sorted, v, side):
    """Share of reference crops at least as extreme as v ('low', 'high' or 'both' tail)."""
    lo = np.searchsorted(ref_sorted, v, side="right") / len(ref_sorted)
    hi = 1 - np.searchsorted(ref_sorted, v, side="left") / len(ref_sorted)
    return {"low": lo, "high": hi, "both": np.minimum(1, 2 * np.minimum(lo, hi))}[side]


def quality_score(ref, stats, sides=SIDES):
    """-log of the most extreme tail probability over the 4 statistics, and which statistic it was.
    ref: {stat: sorted reference values}; stats: {stat: array of values}; sides: which tail is bad per statistic."""
    t = np.vstack([np.maximum(tail(ref[s], np.asarray(stats[s]), sides[s]), 1e-6) for s in STATS])
    return -np.log(t.min(0)), np.array(STATS)[t.argmin(0)]


# ------------------------------------------------------------------ conformal pieces
def pvalue(cal_scores, cal_groups, scores):
    """Family-weighted conformal p-value (upper tail: large score = unusual). Every calibration family has weight 1."""
    cal_scores = np.asarray(cal_scores, float)
    _, inv, cnt = np.unique(cal_groups, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    order = np.argsort(cal_scores)
    s, cw = cal_scores[order], np.cumsum(w[order][::-1])[::-1]          # cw[i] = weight of scores >= s[i]
    idx = np.searchsorted(s, np.asarray(scores, float), side="left")
    above = np.where(idx < len(s), cw[np.minimum(idx, len(s) - 1)], 0.0)
    return (1.0 + above) / (len(cnt) + 1.0)


def family_threshold(p_cal, y_cal, g_cal, alpha):
    """LAC threshold q where every calibration family carries total weight 1 (its crops share it)."""
    s = 1 - p_cal[np.arange(len(y_cal)), y_cal]
    _, inv, cnt = np.unique(g_cal, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    order = np.argsort(s)
    k = np.searchsorted(np.cumsum(w[order]), (1 - alpha) * (len(cnt) + 1) - 1e-9)
    return 1.0 if k >= len(s) else float(s[order][k])


def temper(p, T):
    """Temperature scaling on probabilities: p ** (1/T), renormalised. Keeps each crop's class ranking."""
    q = np.clip(np.asarray(p, float), 1e-12, None) ** (1.0 / T)
    return q / q.sum(-1, keepdims=True)


def knn_distance(feats, bank, k=5):
    """1 - mean cosine similarity to the k nearest rows of bank (L2-normalised train features). numpy version of
    EnhancedRes.knn_distance, for callers without PyTorch."""
    f = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    sim = f @ bank.T
    return 1 - np.sort(sim, axis=1)[:, -k:].mean(1)


# ------------------------------------------------------------------ the pipeline
@dataclass
class Decision:
    action: str                         # one of ACTIONS
    classes: list = field(default_factory=list)   # the CP set (empty when a gate stopped the crop)
    scores: list = field(default_factory=list)    # their tempered probabilities, same order
    p_quality: float = float("nan")
    p_ood: float = float("nan")
    reason: str = ""                    # why the quality gate fired, in words

    def text(self):
        if self.action == RETAKE:
            return f"BAD IMAGE: re-take ({self.reason})"
        if self.action == REAIM:
            return f"CAN'T ASSESS: re-aim ({self.reason or 'unlike any training crop'})"
        if self.action == UNCLEAR:
            return "UNCLEAR: re-inspect"
        if self.action == NOTHING:
            return ""
        if self.action == POSSIBLE:
            return "POSSIBLE " + " / ".join(c.upper() for c in self.classes if c != NO_DEFECT) + ": check"
        ranked = "   ".join(f"{c} {s * 100:.1f}%" for c, s in zip(self.classes, self.scores))
        return ranked.upper() if self.action == SHOW else f"AMBIGUOUS: {ranked}"


class ArchA:
    """Calibrated decision layer for enhanced_res. alpha: CP error rate (one of the calibrated values)."""

    def __init__(self, path=CALIBRATION, alpha=None):
        path = Path(path)
        if not path.exists():
            raise SystemExit(f"missing {path}: run `python enhanced_res/arch_a/calibrate.py`")
        c = json.loads(path.read_text())
        self.cal = c
        self.classes = c["classes"]
        self.T = c["temperature"]["T"]
        self.alpha = float(alpha if alpha is not None else c["cp"]["alpha"])
        key = f"{self.alpha:.2f}"
        if key not in c["cp"]["q"]:
            raise SystemExit(f"alpha {self.alpha} is not calibrated; choose one of {sorted(c['cp']['q'])}")
        self.q = c["cp"]["q"][key]
        qg, og = c.get("quality_gate") or {"enabled": False}, c["ood_gate"]
        self.use_quality = qg.get("enabled", True)          # legacy in architecture B: off unless re-enabled
        self.alpha_ood = og["alpha"]
        self.alpha_q = qg["alpha"] if self.use_quality else None
        if self.use_quality:
            self.sides = qg.get("sides", SIDES)
            self.ref = {s: np.asarray(qg["reference"][s]) for s in STATS}
            self.median = {s: float(np.median(self.ref[s])) for s in STATS}
            self.q_cal, self.q_groups = np.asarray(qg["cal_scores"]), np.asarray(qg["cal_groups"])
        self.nd = self.classes.index(NO_DEFECT) if NO_DEFECT in self.classes else None
        mg = c.get("miss_guard")                            # optional; see the module docstring
        self.guard = None if mg is None else (float(mg["alpha"]), np.asarray(mg["cal_scores"]), np.asarray(mg["cal_groups"]))
        self.o_cal, self.o_groups = np.asarray(og["cal_scores"]), np.asarray(og["cal_groups"])
        self.k = og["k"]
        lr = og.get("aon", og.get("aircraft_or_not"))   # optional second score; older files call it "aircraft_or_not"
        self.lr = None if lr is None else {k: np.asarray(v) for k, v in lr.items() if k != "about"}
        self.n_parts = 2 if self.lr is not None else 1

    # ---- the pieces
    def quality(self, crops):
        """Quality p-values and the deciding statistic for a list of crops."""
        st = [quality_stats(c) for c in crops]
        stats = {s: np.array([d[s] for d in st]) for s in STATS}
        sc, why = quality_score(self.ref, stats, self.sides)
        return pvalue(self.q_cal, self.q_groups, sc), why, stats

    def aon_score(self, feats):
        """Logistic AoN score (probability of NOT aircraft skin) from the 768-d features."""
        z = (np.asarray(feats, float) - self.lr["mu"]) / self.lr["sd"]
        return 1 / (1 + np.exp(-(z @ self.lr["coef"] + self.lr["intercept"])))

    def ood(self, knn_dist=None, feats=None):
        """Combined OOD p-value and which part decided ('knn' / 'aon'). Bonferroni over the configured parts,
        so the false-reject guarantee holds even when a part is unavailable (it then simply cannot fire)."""
        parts = {}
        if knn_dist is not None:
            parts["knn"] = pvalue(self.o_cal, self.o_groups, np.asarray(knn_dist, float))
        if feats is not None and self.lr is not None:
            parts["aon"] = pvalue(self.lr["cal_scores"], self.o_groups, self.aon_score(feats))
        names = list(parts)
        P = np.vstack([parts[k] for k in names])
        return np.minimum(1, self.n_parts * P.min(0)), np.array(names)[P.argmin(0)]

    def sets(self, probs):
        """Tempered probabilities and the boolean CP sets [N, n_classes]. A set that would be empty keeps its top
        class (the usual non-empty LAC set; it can only raise coverage)."""
        p = temper(probs, self.T)
        S = p >= 1 - self.q
        S[np.arange(len(p)), p.argmax(1)] |= ~S.any(1)
        return p, S

    def _reason(self, stat, value):  # quality gate only
        if stat == "sharpness":
            return "blurry" if value < self.median["sharpness"] else "noisy / grainy"
        if stat == "brightness":
            return "too dark" if value < self.median["brightness"] else "too bright"
        return "glare / blown-out highlights" if stat == "clipped_hi" else "black areas"

    # ---- everything
    def silence_allowed(self, p):
        """Miss guard: True where "nothing shown" is allowed. p: tempered probabilities. Without a guard, always.
        With one: the conformal p-value of "this is a defect" (defect calibration crops scored by their tempered
        p(no-defect); a high p(no-defect) is unusual for a defect) must be <= alpha_miss."""
        if self.guard is None:
            return np.ones(len(p), bool)
        a, cal, g = self.guard
        return pvalue(cal, g, p[:, self.nd]) <= a

    def decide(self, probs, crops=None, knn_dist=None, feats=None):
        """One Decision per crop. probs: [N, n_classes] enhanced_res probabilities (untempered). crops: the 224 px
        crops (quality gate skipped when None). knn_dist: EnhancedRes.knn_distance(f); feats: EnhancedRes.features(x)
        as numpy (the OOD gate uses whichever of the two is given; skipped when both are None)."""
        probs = np.asarray(probs, float)
        n = len(probs)
        if not self.use_quality:
            crops = None
        pq = np.ones(n) if crops is None else None
        why, stats = None, None
        if crops is not None:
            pq, why, stats = self.quality(crops)
        use_ood = knn_dist is not None or (feats is not None and self.lr is not None)
        po, part = self.ood(knn_dist, feats) if use_ood else (np.ones(n), None)
        p, S = self.sets(probs)
        silent_ok = self.silence_allowed(p) if self.nd is not None else None
        out = []
        for i in range(n):
            d = Decision(action="", p_quality=float(pq[i]) if crops is not None else float("nan"),
                         p_ood=float(po[i]) if use_ood else float("nan"))
            # both gates are checked; when both fire, "re-aim" wins: an unrelated photo is often also "unusual" in its
            # pixel statistics (far sharper than D1 crops), but the inspector needs to point at the aircraft, not
            # re-take the same view. Which crops are stopped does not change, only the message.
            if use_ood and po[i] <= self.alpha_ood:
                d.action = REAIM
                d.reason = "not aircraft skin" if part[i] == "aon" else "unlike any training crop"
            elif crops is not None and pq[i] <= self.alpha_q:
                d.action, d.reason = RETAKE, self._reason(why[i], stats[why[i]][i])
            else:
                idx = [k for k in np.argsort(-p[i]) if S[i, k]]
                d.classes, d.scores = [self.classes[k] for k in idx], [float(p[i, k]) for k in idx]
                d.action = SHOW if len(idx) == 1 else AMBIGUOUS if idx else UNCLEAR
                if self.nd is not None and self.nd in idx:          # architecture B: the no-defect class
                    if len(idx) == 1:
                        if silent_ok[i]:
                            d.action = NOTHING
                        else:                                       # the miss guard keeps it on the visor
                            k = int(np.argmax(np.where(np.arange(p.shape[1]) == self.nd, -1, p[i])))
                            d.classes, d.scores = [self.classes[k], NO_DEFECT], [float(p[i, k]), float(p[i, self.nd])]
                            d.action = POSSIBLE
                    else:
                        d.action = POSSIBLE
            out.append(d)
        return out


def summarise(decisions, arch, probs, y, groups):
    """Labelled crops: CP coverage / set size without gates, and the visor actions with them (as decided).
    probs are untempered model probabilities; y class indices; groups photo families."""
    p, S = arch.sets(probs)
    y = np.asarray(y)
    hit = S[np.arange(len(y)), y]
    _, inv = np.unique(groups, return_inverse=True)
    fam_cov = np.bincount(inv, hit) / np.bincount(inv)
    act = np.array([d.action for d in decisions])
    shown = act == SHOW
    right = np.array([arch.classes[t] in d.classes for d, t in zip(decisions, y)])
    extra = {}
    if arch.nd is not None:                                 # architecture B: what "nothing shown" costs and gains
        defect, clean = y != arch.nd, y == arch.nd
        extra = {"silent_miss": float((act[defect] == NOTHING).mean()),
                 "silent_miss_by_class": {arch.classes[k]: float((act[y == k] == NOTHING).mean())
                                          for k in range(len(arch.classes)) if k != arch.nd and (y == k).any()},
                 "no_defect_nothing_shown": float((act[clean] == NOTHING).mean()) if clean.any() else float("nan"),
                 "no_defect_false_alarm": float((act[clean] != NOTHING).mean()) if clean.any() else float("nan")}
    return {
        "alpha": arch.alpha,
        "accuracy": float((p.argmax(1) == y).mean()),
        "cp_coverage": float(hit.mean()),
        "cp_family_coverage": float(fam_cov.mean()),
        "cp_mean_set_size": float(S.sum(1).mean()),
        "cp_single_class": float((S.sum(1) == 1).mean()),
        "actions": {k: float((act == k).mean()) for k in ACTIONS},
        "human_attention": float(1 - (shown | (act == NOTHING)).mean()),
        "gates_false_reject": float(np.isin(act, (RETAKE, REAIM)).mean()),
        "gates_false_reject_family": float((np.bincount(inv, np.isin(act, (RETAKE, REAIM))) / np.bincount(inv)).mean()),
        "single_class_wrong": float((shown & ~right).mean()),
        **extra,
    }
