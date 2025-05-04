import pytest
import pandas as pd
from analytics_eda.core.numeric.report_binning_rules import report_binning_rules


def test_non_series_input_raises_type_error():
    with pytest.raises(TypeError, match="Input must be a pandas Series."):
        report_binning_rules([1, 2, 3], is_discrete=True)


def test_non_numeric_series_raises_type_error():
    series = pd.Series(["a", "b", "c"], name="letters")
    with pytest.raises(TypeError, match="Series must be numeric"):
        report_binning_rules(series, is_discrete=False)


def test_missing_name_raises_value_error():
    series = pd.Series([1, 2, 3])  # name is None
    with pytest.raises(ValueError, match="Series must have a non-empty 'name' attribute."):
        report_binning_rules(series, is_discrete=True)


def test_discrete_value_counts():
    series = pd.Series([1, 2, 2, 3, None, 2], name="nums")
    result = report_binning_rules(series, is_discrete=True)
    expected = {'value_counts': {1: 1, 2: 3, 3: 1}}
    assert result == expected


def test_continuous_sturges_binning():
    # For n=8, Sturges' rule gives k = ceil(log2(8) + 1) = 4
    series = pd.Series(list(range(1, 9)), name="nums")
    report = report_binning_rules(series, is_discrete=False, rules=["sturges"])
    info = report["sturges"]
    assert info["n_bins"] == 4
    assert len(info["edges"]) == info["n_bins"] + 1
    assert sum(info["counts"]) == 8


def test_continuous_scott_binning():
    series = pd.Series([1, 2, 3, 4, 5], name="nums")
    report = report_binning_rules(series, is_discrete=False, rules=["scott"])
    info = report["scott"]
    k = info["n_bins"]
    assert isinstance(k, int) and k > 0
    assert len(info["edges"]) == k + 1
    assert sum(info["counts"]) == 5


def test_continuous_freedman_diaconis_binning():
    series = pd.Series([1, 2, 3, 4, 5, 6], name="nums")
    report = report_binning_rules(series, is_discrete=False, rules=["freedman-diaconis"])
    info = report["freedman-diaconis"]
    k = info["n_bins"]
    assert isinstance(k, int) and k > 0
    assert len(info["edges"]) == k + 1
    assert sum(info["counts"]) == 6


def test_continuous_doane_binning():
    series = pd.Series([1, 2, 3, 4, 5, 6, 7], name="nums")
    report = report_binning_rules(series, is_discrete=False, rules=["doane"])
    info = report["doane"]
    k = info["n_bins"]
    assert isinstance(k, int) and k > 0
    assert len(info["edges"]) == k + 1
    assert sum(info["counts"]) == 7


def test_scott_insufficient_non_na_raises_error():
    series = pd.Series([1.0], name="nums")
    report = report_binning_rules(series, is_discrete=False, rules=["scott"])
    info = report["scott"]
    assert "error" in info
    assert "at least two non-NA values" in info["error"]


def test_scott_zero_variance_raises_error():
    series = pd.Series([5.0, 5.0], name="nums")
    report = report_binning_rules(series, is_discrete=False, rules=["scott"])
    info = report["scott"]
    assert "error" in info
    assert "non-zero variance" in info["error"]


def test_freedman_insufficient_non_na_raises_error():
    series = pd.Series([1.0], name="nums")
    report = report_binning_rules(series, is_discrete=False, rules=["freedman-diaconis"])
    info = report["freedman-diaconis"]
    assert "error" in info, f"Expected an 'error' key but got: {info!r}"

    error_msg = info["error"]
    assert "at least two non-NA values" in error_msg, (
        f"Expected substring 'at least two non-NA values' in error message.\n"
        f"Actual message: {error_msg!r}\n"
        f"Full info dict: {info!r}"
    )


def test_freedman_zero_iqr_raises_error():
    series = pd.Series([3.0, 3.0, 3.0], name="nums")
    report = report_binning_rules(series, is_discrete=False, rules=["freedman-diaconis"])
    info = report["freedman-diaconis"]
    assert "error" in info
    assert "IQR must be positive" in info["error"]


def test_doane_insufficient_non_na_raises_error():
    series = pd.Series([1.0, 2.0], name="nums")
    report = report_binning_rules(series, is_discrete=False, rules=["doane"])
    info = report["doane"]
    assert "error" in info
    assert "at least three values" in info["error"]


def test_unknown_rule_reports_error():
    series = pd.Series([1, 2, 3, 4], name="nums")
    report = report_binning_rules(series, is_discrete=False, rules=["invalid_rule"])
    assert "invalid_rule" in report
    error_info = report["invalid_rule"]
    assert "error" in error_info
    assert "Unknown binning rule" in error_info["error"]


def test_default_rules_presence():
    series = pd.Series(list(range(1, 10)), name="nums")
    report = report_binning_rules(series, is_discrete=False)
    expected_rules = {"sturges", "scott", "freedman-diaconis", "doane"}
    assert set(report.keys()) == expected_rules

