import numpy as np
import pandas as pd
import pytest

from analytics_eda.analysis.univariate.univariate_numeric_analysis import (
    UnivariateNumericAnalysis,
    UnivariateNumericAnalysisContext,
)


@pytest.mark.parametrize(
    "make_series, expected",
    [
        (
            lambda: pd.Series(np.random.default_rng(0).normal(size=20), name="metric", dtype="float64"),
            {
                "data_quality": {"data": {}},
                "distribution": {
                    "data": {
                        "central_tendency": {},
                        "dispersion": {},
                        "shape": {},
                    }
                },
                "transforms": {"data": {}},
            },
        ),
    ],
    ids=["basic_numeric"],
)
def test_univariate_numeric_analysis(tmp_path, assert_report_data, make_series, expected):
    ctx = UnivariateNumericAnalysisContext(
        base_dir=tmp_path,
        data_source="UnitTest",
        filter_desc="UnitTest",
        save_json_report=False,
        return_full_report=True,
    )
    analysis = UnivariateNumericAnalysis(ctx)

    report = analysis.run(make_series())

    assert_report_data(report, expected, tmp_path / "univariate" / "numeric")


@pytest.mark.parametrize(
    "series_factory, expected_exc, match",
    [
        (lambda: [1, 2, 3], TypeError, r"Input must be a pandas Series."),
        (lambda: pd.Series([1, 2, 3]), ValueError, r"must have a non-empty 'name'"),
        (lambda: pd.Series([1, 2, 3], name="   "), ValueError, r"must have a non-empty 'name'"),
    ],
)
def test_validate_numeric_named_series_errors(series_factory, expected_exc, match, tmp_path):
    s = series_factory()
    with pytest.raises(expected_exc, match=match):
        ctx = UnivariateNumericAnalysisContext(base_dir=tmp_path)
        analysis = UnivariateNumericAnalysis(ctx)
        analysis.run(s)
