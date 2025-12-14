import pandas as pd
import pytest

from analytics_eda.core.numeric.shape import (
    ShapeAnalysis,
    ShapeAnalysisContext,
)


@pytest.mark.parametrize(
    "make_series, expected",
    [
        (
            lambda: pd.Series([1.0, 2.0, 3.0, 4.0], name="metric"),
            {
                "density": {"chart_metadata": {"title": lambda v: isinstance(v, str)}},
                "distribution_fits": lambda v: (
                    isinstance(v, dict) and isinstance(v.get("data"), dict) and "distribution_fits" in v["data"] and "norm" in v["data"]["distribution_fits"] and all(k in v["data"]["distribution_fits"]["norm"] for k in ("ecdf_vs_cdf", "qq_fit"))
                ),
            },
        ),
    ],
    ids=["basic_shape"],
)
def test_shape_analysis(tmp_path, assert_report_data, make_series, expected):
    ctx = ShapeAnalysisContext(
        base_dir=tmp_path,
        data_source="UnitTest",
        filter_desc="UnitTest",
        save_json_report=False,
        return_full_report=True,
    )
    analysis = ShapeAnalysis(ctx)

    report = analysis.run(make_series())

    assert_report_data(report, expected, tmp_path / "shape")
