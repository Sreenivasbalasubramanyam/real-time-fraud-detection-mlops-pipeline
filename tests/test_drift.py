import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.monitoring.drift_monitor import (
    check_drift,
    ks_test,
    population_stability_index,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def test_psi_is_near_zero_for_identical_distributions(rng):
    reference = rng.normal(loc=0, scale=1, size=5000)
    current = rng.normal(loc=0, scale=1, size=5000)
    psi = population_stability_index(reference, current)
    assert psi < 0.05


def test_psi_is_large_for_shifted_distribution(rng):
    reference = rng.normal(loc=0, scale=1, size=5000)
    current = rng.normal(loc=3, scale=1, size=5000)  # large mean shift
    psi = population_stability_index(reference, current)
    assert psi > 0.25


def test_ks_test_no_drift_case(rng):
    reference = rng.normal(loc=0, scale=1, size=3000)
    current = rng.normal(loc=0, scale=1, size=3000)
    d_stat, p_value = ks_test(reference, current)
    assert d_stat < 0.05
    assert p_value > 0.05  # fail to reject null of same distribution


def test_ks_test_drift_case(rng):
    reference = rng.normal(loc=0, scale=1, size=3000)
    current = rng.normal(loc=1.5, scale=1.5, size=3000)
    d_stat, p_value = ks_test(reference, current)
    assert d_stat > 0.2
    assert p_value < 0.05  # reject null: distributions differ significantly


def test_check_drift_detects_no_drift_when_stable(rng):
    n = 4000
    reference_df = pd.DataFrame({
        "feat_a": rng.normal(0, 1, n),
        "feat_b": rng.exponential(2.0, n),
    })
    current_df = pd.DataFrame({
        "feat_a": rng.normal(0, 1, n),
        "feat_b": rng.exponential(2.0, n),
    })
    report = check_drift(reference_df, current_df, ["feat_a", "feat_b"])
    assert report.overall_drift_detected is False
    assert report.n_drifted_features == 0


def test_check_drift_detects_drift_when_shifted(rng):
    n = 4000
    reference_df = pd.DataFrame({
        "feat_a": rng.normal(0, 1, n),
        "feat_b": rng.exponential(2.0, n),
    })
    current_df = pd.DataFrame({
        "feat_a": rng.normal(4, 1.5, n),   # drifted
        "feat_b": rng.exponential(2.0, n),  # stable
    })
    report = check_drift(reference_df, current_df, ["feat_a", "feat_b"])
    assert report.overall_drift_detected is True
    assert report.n_drifted_features == 1
    drifted_features = [r.feature for r in report.feature_results if r.is_drifted]
    assert drifted_features == ["feat_a"]


def test_drift_report_serializes_to_json(rng):
    n = 500
    reference_df = pd.DataFrame({"x": rng.normal(0, 1, n)})
    current_df = pd.DataFrame({"x": rng.normal(0, 1, n)})
    report = check_drift(reference_df, current_df, ["x"])
    payload = report.to_dict()
    assert "overall_drift_detected" in payload
    assert "features" in payload
    assert payload["n_reference"] == n
    assert payload["n_current"] == n
