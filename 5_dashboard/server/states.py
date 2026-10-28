"""Map a prediction to the helmet's own states. For the main model these are architecture B's answers
(enhanced_res/arch_a, calibration_gc7.json): its unfamiliar-input gate, temperature and conformal set (no quality
gate). "Nothing shown" (the set is only no-defect) is the state "clear"; "possible" (a defect and no-defect) shows as
"ambiguous"."""
from server import paths  # noqa: F401  (puts the repo on sys.path)
from enhanced_res.arch_a.runtime import AMBIGUOUS, NOTHING, POSSIBLE, REAIM, RETAKE, SHOW, UNCLEAR, temper

# the decision layer's action -> the state the UI shows (NOTHING / POSSIBLE: architecture B, models with no-defect)
STATE = {SHOW: "detected", AMBIGUOUS: "ambiguous", UNCLEAR: "unclear", RETAKE: "retake", REAIM: "cannot_assess",
         NOTHING: "clear", POSSIBLE: "ambiguous"}
NO_DEFECT = "no-defect"


def describe(classes, probs, decision=None, arch=None):
    """One crop's answer. `probs` are the model's own probabilities. With the decision layer they are shown
    tempered (the calibrated %), `answer` is the conformal set, and a gate can stop the crop before any class
    is shown. Without it (the legacy model) the raw ranking is shown, unrated."""
    shown = list(temper([probs], arch.T)[0]) if arch else list(probs)
    order = sorted(range(len(classes)), key=lambda i: -shown[i])
    out = {"classes": list(classes), "probs": [float(p) for p in shown],
           "ranked": [{"cls": classes[i], "p": float(shown[i])} for i in order],
           "top": classes[order[0]], "confidence": float(shown[order[0]]), "state": "unrated",
           "answer": [], "reason": None, "p_quality": None, "p_ood": None, "set_threshold": None, "alpha": None,
           "alpha_quality": None, "alpha_ood": None}
    if arch is None or decision is None:
        return out
    nan = lambda v: None if v != v else float(v)  # noqa: E731  (NaN: that gate was skipped)
    state = "clear" if decision.action == SHOW and list(decision.classes) == [NO_DEFECT] else STATE[decision.action]
    return out | {"state": state, "answer": list(decision.classes), "reason": decision.reason or None,
                  "p_quality": nan(decision.p_quality), "p_ood": nan(decision.p_ood),
                  "set_threshold": float(1 - arch.q), "alpha": arch.alpha,
                  "alpha_quality": arch.alpha_q, "alpha_ood": arch.alpha_ood}
