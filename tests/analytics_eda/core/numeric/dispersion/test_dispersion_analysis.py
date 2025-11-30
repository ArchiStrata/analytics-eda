import pandas as pd
import pytest

from analytics_eda.core.numeric.dispersion import (
    DispersionAnalysis,
    DispersionAnalysisContext,
)


@pytest.mark.parametrize(
    "make_series, expected",
    [
        (
            lambda: pd.Series([1, 2, 3, 4], name="metric"),
            {
                "boxplot": {"chart_metadata": {"title": lambda v: isinstance(v, str)}},
                "sigma_bands": {"chart_metadata": {"title": lambda v: isinstance(v, str)}},
                "deciles": {"chart_metadata": {"title": lambda v: isinstance(v, str)}},
            },
        ),
    ],
    ids=["basic_dispersion"],
)
def test_dispersion_analysis(tmp_path, assert_report_data, make_series, expected):
    ctx = DispersionAnalysisContext(
        base_dir=tmp_path,
        data_source="UnitTest",
        filter_desc="UnitTest",
        save_json_report=False,
        return_full_report=True,
    )
    analysis = DispersionAnalysis(ctx)

    report = analysis.run(make_series())

    assert_report_data(report, expected, tmp_path / "dispersion")
