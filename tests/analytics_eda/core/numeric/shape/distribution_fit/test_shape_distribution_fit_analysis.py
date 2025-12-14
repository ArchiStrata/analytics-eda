import pandas as pd
import pytest

from analytics_eda.core.numeric.shape import (
    ShapeDistributionFitAnalysis,
    ShapeDistributionFitAnalysisContext,
)


@pytest.mark.parametrize(
    "make_series, expected",
    [
        (
            lambda: pd.Series([1.0, 2.0, 3.0], name="metric"),
            {"distribution_fits": lambda v: (isinstance(v, dict) and "norm" in v and all(k in v["norm"] for k in ("ecdf_vs_cdf", "qq_fit")))},
        ),
    ],
    ids=["basic_distribution_fit"],
)
def test_shape_distribution_fit_analysis(tmp_path, assert_report_data, make_series, expected):
    ctx = ShapeDistributionFitAnalysisContext(
        base_dir=tmp_path,
        data_source="UnitTest",
        filter_desc="UnitTest",
        distribution_names=("norm",),
        save_json_report=False,
        return_full_report=True,
    )
    analysis = ShapeDistributionFitAnalysis(ctx)

    report = analysis.run(make_series())

    assert_report_data(report, expected, tmp_path / "shape")
