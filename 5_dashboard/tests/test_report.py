import numpy as np
import pytest

from server.registry import Registry
from server.report import build_report


@pytest.fixture(scope="module")
def rep_and_parts():
    r = Registry()
    entry, adapter = r.entry(), r.active()
    return build_report(entry, adapter), entry, adapter


def test_reproduces_published_numbers(rep_and_parts):
    # gc7 on d1_gc7 test: 4,255 defect crops + 615 no-defect crops (enhanced_res/results/gc7/summary.json)
    rep, _, _ = rep_and_parts
    assert rep["test"]["n"] == 4870 and rep["test"]["groups"] == 175
    assert abs(rep["test"]["accuracy"] - rep["published"]["accuracy"]) < 5e-4
    assert abs(rep["test"]["macro_f1"] - rep["published"]["macro_f1"]) < 5e-4


def test_no_defect_numbers_match_the_teams_report(rep_and_parts):
    nd = rep_and_parts[0]["no_defect"]
    assert nd["n"] == 615 and nd["n_defects"] == 4255
    assert abs(nd["recall"] - 0.9106) < 5e-4 and abs(nd["defects_called_clear"] - 0.0385) < 5e-4
    assert abs(nd["defects_best"]["accuracy"] - 0.8047) < 5e-4 and abs(nd["defects_best"]["macro_f1"] - 0.7423) < 5e-4
    assert abs(nd["recall_by_source"]["healed"] - 0.88) < 5e-4 and abs(nd["recall_by_source"]["clean_cropper"] - 0.9315) < 5e-4


def test_architecture_a_matches_the_teams_own_check(rep_and_parts):
    # same test half, same gates and sets as enhanced_res/arch_a/calibrate.py's expected_test_half
    a = rep_and_parts[0]["arch_a"]
    got, exp = a["by_alpha"][f"{a['alpha']:.2f}"], a["expected"]
    assert a["test_half"] == {"n": 2752, "families": 87}
    for k in ("accuracy", "cp_coverage", "cp_family_coverage", "cp_mean_set_size", "human_attention", "single_class_wrong"):
        assert abs(got[k] - exp[k]) < 2e-3, k
    for k, v in exp["actions"].items():
        assert abs(got["actions"][k] - v) < 2e-3, k


def test_temperature_lowers_calibration_error(rep_and_parts):
    e = rep_and_parts[0]["arch_a"]["ece"]
    assert e["tempered"] < e["raw"]


def test_smaller_alpha_gives_larger_sets(rep_and_parts):
    by = rep_and_parts[0]["arch_a"]["by_alpha"]
    sizes = [by[k]["cp_mean_set_size"] for k in sorted(by)]          # 0.05, 0.10, 0.20
    assert sizes == sorted(sizes, reverse=True)


def test_silent_errors_are_wrong_single_answers(rep_and_parts):
    errs = rep_and_parts[0]["arch_a"]["silent_errors"]
    assert 0 < len(errs) <= 12 and all(e["true"] != e["pred"] for e in errs)
    assert [e["confidence"] for e in errs] == sorted((e["confidence"] for e in errs), reverse=True)


def test_unreadable_cache_is_a_miss_and_key_has_version(tmp_path, monkeypatch):
    from server import report as rmod
    monkeypatch.setattr(rmod, "CACHE", tmp_path)
    rmod._mem.clear()
    r = Registry()
    (tmp_path / "report_enhanced_res.json").write_text('{"key": "trunc')  # half-written file
    rep = rmod.get_report(r)
    assert rep["arch_a"]["test_half"]["n"] == 2752
    assert rmod.REPORT_VERSION in rep["key"]
