import pytest
import pandas as pd
import numpy as np
from analytics_eda.core.numeric.normality_assessment import normality_assessment


def test_empty_series_returns_empty():
    s = pd.Series([], dtype=float, name="nums")
    result = normality_assessment(s)
    assert result == {}


def test_non_series_input_raises_type_error():
    with pytest.raises(TypeError, match="Input must be a pandas Series."):
        normality_assessment([1, 2, 3])


def test_missing_name_raises_value_error():
    s = pd.Series([1, 2, 3], dtype=float)
    with pytest.raises(ValueError, match="Series must have a non-empty 'name'"):
        normality_assessment(s)


def test_small_sample_only_shapiro_anderson():
    # n < 20
    s = pd.Series(range(10), name="nums")
    result = normality_assessment(s, alpha=0.05)
    # Expect n, shapiro, anderson, jarque_bera, reject_normality
    assert result["n"] == 10
    assert "shapiro" in result
    assert "dagostino_pearson" not in result
    assert "anderson" in result
    assert "jarque_bera" not in result
    assert isinstance(result["reject_normality"], bool)


def test_medium_sample_includes_shapiro_and_dagostino():
    # 20 <= n < 50
    data = np.arange(25)
    s = pd.Series(data, name="nums")
    result = normality_assessment(s)
    assert result["n"] == 25
    assert "shapiro" in result
    assert "dagostino_pearson" in result
    assert "anderson" in result
    assert "jarque_bera" not in result


def test_medium_sample_excludes_shapiro():
    # n >= 50
    data = np.arange(60)
    s = pd.Series(data, name="nums")
    result = normality_assessment(s)
    assert result["n"] == 60
    assert "shapiro" not in result
    assert "dagostino_pearson" in result
    assert "anderson" in result
    assert "jarque_bera" not in result


def test_large_sample_includes_jarque_bera():
    # n > 2000
    data = np.arange(2001)
    s = pd.Series(data, name="nums")
    result = normality_assessment(s)
    assert result["n"] == 2001
    assert "shapiro" not in result
    assert "dagostino_pearson" in result
    assert "anderson" in result
    assert "jarque_bera" in result


def test_nans_are_dropped_before_count():
    data = [1.0, 2.0, np.nan, 3.0]
    s = pd.Series(data, name="nums")
    result = normality_assessment(s)
    # n should count only non-NA values
    assert result["n"] == 3


def test_statistic_values_types():
    # Use a small sample to get all keys
    s = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], name="nums")
    result = normality_assessment(s, alpha=0.5)
    # Check each test entry has correct types
    for key in ("shapiro", "dagostino_pearson", "anderson", "jarque_bera"):
        if key in result:
            entry = result[key]
            assert isinstance(entry.get("statistic"), float), f"Expected float statistic for {key}, got {type(entry.get('statistic'))}"
            # p_value exists for all except Anderson
            if key != "anderson":
                assert isinstance(entry.get("p_value"), float), f"Expected float p_value for {key}, got {type(entry.get('p_value'))}"
            # reject may be numpy.bool_ or bool
            reject_val = entry.get("reject")
            assert isinstance(reject_val, (bool, np.bool_)), (
                f"Expected boolean reject flag for {key}, got {type(reject_val)} with value {reject_val}"
            )

def test_reject_normality_flag_true_for_non_normal_distribution():
    # Uniformly spaced integers should fail normality
    data = np.arange(30)
    s = pd.Series(data, name="nums")
    result = normality_assessment(s)
    assert result["reject_normality"] is True


def test_alpha_edge_pick_closest_anderson():
    # Test alpha not exactly in significance_level list picks closest
    # Use a sample size to include Anderson
    data = np.arange(30)
    s = pd.Series(data, name="nums")
    # Use an uncommon alpha
    result = normality_assessment(s, alpha=0.123)
    assert "anderson" in result
    ad = result["anderson"]
    # critical_values and significance_levels lengths match
    assert len(ad["critical_values"]) == len(ad["significance_levels"]) 
