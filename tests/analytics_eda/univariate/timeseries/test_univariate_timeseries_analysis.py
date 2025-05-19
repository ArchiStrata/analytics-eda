import pandas as pd
import json

from analytics_eda.univariate.timeseries.univariate_timeseries_analysis import univariate_timeseries_analysis

def test_univariate_timeseries_analysis_creates_and_populates_report(tmp_path):
    # 1. Prepare sample data
    dates = pd.date_range("2021-01-01", periods=12, freq="ME")
    values = list(range(12))
    df = pd.DataFrame({"date": dates, "value": values})

    # 2. Define a temporary report_root
    report_root = tmp_path / "reports"

    # 3. Run the analysis
    report_path = univariate_timeseries_analysis(
        df,
        numeric_col="value",
        time_col="date",
        report_root=str(report_root),
        rolling_window=3
    )

    # 4. Assert report_path exists
    assert report_path.exists(), f"Report file was not created at {report_path}"

    # 5. Assert report_path is exactly where we expect
    expected_dir = report_root / "date_value"
    expected_path = expected_dir / "value_univariate_analysis_report.json"
    assert report_path == expected_path, (
        f"Expected report_path {expected_path}, got {report_path}"
    )

    # 6. Load and inspect the JSON report
    with open(report_path, "r") as f:
        report = json.load(f)

    # 7. Assert top‐level report structure
    assert report is not None, "Loaded report is None"
    assert "visuals" in report, "'visuals' key missing in report"

    # 8. Assert visuals dict is non‐empty
    visuals = report["visuals"]
    assert isinstance(visuals, dict), f"'visuals' should be a dict, got {type(visuals)}"
    assert visuals, "Visuals dictionary is empty"
