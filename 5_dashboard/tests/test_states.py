import numpy as np
import pytest

from server.report import ece, reliability_bins
from server.states import STATE, describe
from enhanced_res.arch_a.runtime import AMBIGUOUS, REAIM, RETAKE, SHOW, UNCLEAR, ArchA, Decision, temper


@pytest.fixture(scope="module")
def arch():
    return ArchA()


def test_every_architecture_a_answer_has_a_state():
    assert set(STATE) == {SHOW, AMBIGUOUS, UNCLEAR, RETAKE, REAIM}
    assert len(set(STATE.values())) == 5


def test_describe_shows_tempered_percentages_and_the_set(arch):
    classes = arch.classes
    p = np.array([0.05, 0.80, 0.10, 0.02, 0.02, 0.01])
    (d,) = arch.decide(p[None])
    out = describe(classes, list(p), d, arch)
    assert np.allclose(out["probs"], temper(p[None], arch.T)[0])        # calibrated %, same ranking
    assert out["top"] == classes[1] and out["confidence"] == max(out["probs"])
    assert out["answer"] == d.classes and out["set_threshold"] == pytest.approx(1 - arch.q)
    assert all(r["p"] >= out["set_threshold"] for r in out["ranked"] if r["cls"] in out["answer"])
    assert out["state"] == ("detected" if len(d.classes) == 1 else "ambiguous")


def test_gate_answers_carry_no_classes(arch):
    p = [0.9, 0.02, 0.02, 0.02, 0.02, 0.02]
    retake = describe(arch.classes, p, Decision(action=RETAKE, reason="blurry", p_quality=0.01), arch)
    assert retake["state"] == "retake" and retake["answer"] == [] and retake["reason"] == "blurry"
    assert retake["p_ood"] is None                                       # that gate never ran
    assert describe(arch.classes, p, Decision(action=REAIM, p_ood=0.02), arch)["state"] == "cannot_assess"


def test_no_decision_layer_is_unrated():
    out = describe(["a", "b", "c"], [0.1, 0.7, 0.2])
    assert out["state"] == "unrated" and out["top"] == "b" and [r["cls"] for r in out["ranked"]] == ["b", "c", "a"]
    assert out["probs"] == [0.1, 0.7, 0.2]


def test_bins_and_ece():
    conf = np.array([0.15, 0.85, 0.95, 1.0])
    correct = np.array([False, True, True, False])
    assert [(b["lo"], b["n"]) for b in reliability_bins(conf, correct)] == [(0.1, 1), (0.8, 1), (0.9, 2)]
    assert ece(np.array([0.9] * 10), np.array([True] * 9 + [False])) == pytest.approx(0.0)
    assert ece(np.array([1.0] * 4), np.array([True, False, True, False])) == pytest.approx(0.5)
