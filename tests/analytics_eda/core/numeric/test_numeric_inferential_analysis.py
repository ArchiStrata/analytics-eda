import pytest
import pandas as pd
import numpy as np
from analytics_eda.core.numeric.numeric_inferential_analysis import numeric_inferential_analysis


def test_empty_series_returns_empty():
    s = pd.Series([], dtype=float, name="x")
    result = numeric_inferential_analysis(s)
    assert result == {}


def test_non_series_input_raises_type_error():
    with pytest.raises(TypeError, match="Input must be a pandas Series."):
        numeric_inferential_analysis([1, 2, 3])


def test_missing_name_raises_value_error():
    s = pd.Series([1, 2, 3], dtype=float)
    with pytest.raises(ValueError, match="Series must have a non-empty 'name'"):
        numeric_inferential_analysis(s)


def test_basic_ci_and_gof():
    # Basic inferential outputs without optional parameters
    s = pd.Series([1, 2, 3, 4, 5], name="x")
    result = numeric_inferential_analysis(s, alpha=0.05, bootstrap_samples=10)

    # Confidence intervals
    assert "ci" in result
    ci = result["ci"]
    assert isinstance(ci["mean_t"], list) and len(ci["mean_t"]) == 2
    assert all(isinstance(v, float) for v in ci["mean_t"])
    assert isinstance(ci["median_boot"], list) and len(ci["median_boot"]) == 2
    assert all(isinstance(v, float) for v in ci["median_boot"])

    # Goodness-of-fit via KS
    assert "gof" in result and "ks" in result["gof"]
    ks = result["gof"]["ks"]
    assert set(ks.keys()) == {"statistic", "p_value", "reject"}
    assert isinstance(ks["statistic"], float)
    assert isinstance(ks["p_value"], float)
    assert isinstance(ks["reject"], bool)

    # Bootstrap section
    assert "bootstrap" in result
    bs = result["bootstrap"]
    assert set(bs.keys()) == {"mean", "median"}
    assert len(bs["mean"]) == 2 and len(bs["median"]) == 2


def test_popvariance_only():
    s = pd.Series([2, 4, 6, 8, 10], name="x")
    result = numeric_inferential_analysis(s, popvariance=4)
    assert "variance" in result
    chi2 = result["variance"]["chi2"]
    assert set(chi2.keys()) == {"statistic", "p_value", "reject"}
    # No t_test or z_test without popmean
    assert "t_test" not in result
    assert "z_test" not in result


def test_popmean_only():
    s = pd.Series([1, 2, 3, 4, 5], name="x")
    result = numeric_inferential_analysis(s, popmean=3)
    assert "t_test" in result
    t = result["t_test"]
    assert set(t.keys()) == {"statistic", "p_value", "reject"}
    assert "effect_size" in result and "cohens_d" in result["effect_size"]
    assert result["z_test"] is None if "z_test" in result else True  # z_test only with popvariance

    # z_test should not appear without popvariance
    assert "z_test" not in result


def test_popmean_and_variance():
    s = pd.Series([1, 2, 3, 4, 5], name="x")
    result = numeric_inferential_analysis(s, popmean=3, popvariance=2)
    assert "t_test" in result
    assert "z_test" in result
    z = result["z_test"]
    assert set(z.keys()) == {"statistic", "p_value", "reject"}


def test_popmedian_only():
    s = pd.Series([10, 12, 14, 16, 18], name="x")
    result = numeric_inferential_analysis(s, popmedian=14)
    assert "wilcoxon_signed_rank" in result
    wr = result["wilcoxon_signed_rank"]
    assert set(wr.keys()) == {"statistic", "p_value", "reject"}
    assert "sign_test" in result
    st = result["sign_test"]
    assert set(st.keys()) == {"num_positive", "num_negative", "n", "p_value", "reject"}
