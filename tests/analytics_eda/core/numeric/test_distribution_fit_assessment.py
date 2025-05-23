import pandas as pd
import logging
from analytics_eda.core.numeric.distribution_fit_assessment import distribution_fit_assessment

def test_fit_exception_from_invalid_data(caplog):
    caplog.set_level(logging.DEBUG)
    # Series with only NaNs will cause real scipy.stats.<dist>.fit to fail
    series = pd.Series([float('nan'), float('nan')], name="invalid_data")

    result = distribution_fit_assessment(series, distributions=["norm"])

    # Check that the error was captured for 'norm'
    assert "norm" in result["alternative_fits"]
    assert "error" in result["alternative_fits"]["norm"]
    assert isinstance(result["alternative_fits"]["norm"]["error"], str)
    assert "error" in result["alternative_fits"]["norm"]

    # Assert error was captured for 'norm'
    assert "norm" in result["alternative_fits"]
    norm_result = result["alternative_fits"]["norm"]
    assert "error" in norm_result
    assert isinstance(norm_result["error"], str)

    # Assert start log exists
    start_log = next((r for r in caplog.records if "Starting distribution_fit_assessment" in r.message), None)
    assert start_log is not None
    assert start_log.series_name == "invalid_data"
    assert hasattr(start_log, "report_log_id")

    # Assert error log exists and matches
    error_log = next((r for r in caplog.records if "distribution_fit_assessment failed" in r.message), None)
    assert error_log is not None
    assert error_log.series_name == "invalid_data"
    assert error_log.report_log_id == start_log.report_log_id

    # Assert completion log exists
    complete_log = next((r for r in caplog.records if "Completed distribution_fit_assessment" in r.message), None)
    assert complete_log is not None
    assert complete_log.series_name == "invalid_data"
    assert complete_log.report_log_id == start_log.report_log_id


def test_positive_series_all_fits_successful():
    s = pd.Series([1, 2, 3, 4, 5], name="x")
    result = distribution_fit_assessment(s, alpha=0.05)

    alt = result.get("alternative_fits", {})
    # All distributions should fit without error
    for dist in ("norm", "lognorm", "gamma", "expon"):
        assert dist in alt, f"Missing distribution: {dist}"
        entry = alt[dist]
        assert "error" not in entry, f"Unexpected error for {dist}: {entry.get('error')}"
        assert isinstance(entry.get("params"), tuple), f"Params not found for {dist}"
        ks = entry.get("ks", {})
        for key in ("statistic", "p_value", "reject"):
            assert key in ks, f"KS key '{key}' missing for {dist}"


def test_non_positive_series_errors_for_positive_only_distributions():
    s = pd.Series([-1, 0, 1, 2, 3], name="x")
    result = distribution_fit_assessment(s)
    alt = result["alternative_fits"]

    # lognorm and gamma require positive data
    assert alt["lognorm"] == {"error": "requires positive data"}
    assert alt["gamma"] == {"error": "requires positive data"}
    # expon requires non-negative
    assert alt["expon"] == {"error": "requires non-negative data"}
    # norm should still fit
    assert "norm" in alt and "params" in alt["norm"]
