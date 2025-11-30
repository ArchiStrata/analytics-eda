import pandas as pd
import pytest

from analytics_eda.core.numeric.central_tendency import (
    CentralTendencyAnalysis,
    CentralTendencyAnalysisContext,
)


@pytest.mark.parametrize(
    "make_series, expected",
    [
        (
            lambda: pd.Series([1.0, 2.0, 3.0], name="metric"),
            {
                "histogram": {
                    "chart_metadata": {"title": lambda v: isinstance(v, str)},
                },
                "mean_point_ci": {
                    "chart_metadata": {"title": lambda v: isinstance(v, str)},
                },
                "median_point_ci": {
                    "chart_metadata": {"title": lambda v: isinstance(v, str)},
                },
            },
        ),
    ],
    ids=["basic_central_tendency"],
)
def test_central_tendency_analysis(tmp_path, assert_report_data, make_series, expected):
    ctx = CentralTendencyAnalysisContext(
        base_dir=tmp_path,
        data_source="UnitTest",
        filter_desc="UnitTest",
        save_json_report=False,
        return_full_report=True,
    )
    analysis = CentralTendencyAnalysis(ctx)

    report = analysis.run(make_series())

    assert_report_data(report, expected, tmp_path / "central_tendency")
