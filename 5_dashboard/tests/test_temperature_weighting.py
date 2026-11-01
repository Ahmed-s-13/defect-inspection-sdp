"""Family weighting must not let copies of one family change its influence."""
import sys
import numpy as np
import pytest
from server.paths import REPO_MAIN
sys.path.insert(0, str(REPO_MAIN))
from enhanced_res.arch_a.calibrate import fit_temperature


def test_repeating_a_family_does_not_change_fitted_temperature():
    probs = np.array([[.95,.05],[.7,.3],[.8,.2],[.4,.6],[.3,.7]])
    labels = np.array([0,0,1,1,1])
    families = np.array([10,10,20,30,30])
    original = fit_temperature(probs,labels,families)
    # Replicate every observation in family 10 equally: its total weight must stay one.
    idx = np.r_[np.arange(5),np.tile([0,1],20)]
    repeated = fit_temperature(probs[idx],labels[idx],families[idx])
    assert original.success and repeated.success
    assert np.exp(original.x) == pytest.approx(np.exp(repeated.x),abs=1e-6)
    assert original.fun == pytest.approx(repeated.fun,abs=1e-10)
