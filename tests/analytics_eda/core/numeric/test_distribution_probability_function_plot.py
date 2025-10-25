import math
import pytest

import pandas as pd
import numpy as np

from analytics_eda.core.numeric import DistributionProbabilityFunctionContext, DistributionProbabilityFunctionPlot

@pytest.mark.parametrize(
    "series_factory, expected_exc, match",
    [
        # Not a Series
        (lambda: [1, 2, 3], TypeError, r"data must be a pandas Series or DataFrame"),
        # Non-numeric Series
        (lambda: pd.Series(["a", "b", "c"], name="letters"), TypeError, r"Series must be numeric"),
        # Missing name
        (lambda: pd.Series([1, 2, 3]), ValueError, r"must have a non-empty 'name'"),
        # Blank/whitespace name
        (lambda: pd.Series([1, 2, 3], name="   "), ValueError, r"must have a non-empty 'name'"),
    ],
    ids=["not_series", "bad_dtype", "missing_name", "blank_name"],
)
def test_validate_numeric_named_series_errors(series_factory, expected_exc, match):
    s = series_factory()
    with pytest.raises(expected_exc, match=match):
        ctx = DistributionProbabilityFunctionContext(is_discrete=True)
        plot = DistributionProbabilityFunctionPlot(ctx)

        plot.run(s)


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) EMPTY (discrete) → NaN stats, PMF title, default ylabel, no file
        (
            lambda: pd.Series([], dtype="float64", name="nums"),
            {"is_discrete": True},
            {
                "chart_metadata": {
                    "title": "PMF of nums",
                    "xlabel": "Value",
                    "ylabel": "Probability P(X = x)",
                    "data_source": None,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "n": 0,
                    "mean": None,
                    "median": None,
                    "mode": None,
                    "variance": None,
                    "std": None,
                    "iqr": None,
                    "skewness": None,
                    "kurtosis": None,
                    "min": None,
                    "max": None,
                },
            },
        ),

        # 1) EMPTY (continuous) → NaN stats, PDF title via template, density ylabel
        (
            lambda: pd.Series([], dtype="float64", name="x"),
            {"is_discrete": False, "title_template": "PDF estimate of {name}{modifiers}"},
            {
                "chart_metadata": {
                    "title": "PDF estimate of x (bw=scott)",
                    "xlabel": "Value",
                    "ylabel": "Density f(x)",
                    "data_source": None,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "n": 0,
                    "mean": None,
                    "median": None,
                    "mode": None,
                    "variance": None,
                    "std": None,
                    "iqr": None,
                    "skewness": None,
                    "kurtosis": None,
                    "min": None,
                    "max": None,
                },
            },
        ),

        # 2) Discrete integers with data_source
        (
            lambda: pd.Series([1, 1, 2, 2, 2, 3], name="ids"),
            {"is_discrete": True, "data_source": "UnitTest"},
            {
                "chart_metadata": {
                    "title": "PMF of ids",
                    "xlabel": "Value",
                    "ylabel": "Probability P(X = x)",
                    "data_source": "UnitTest",
                },
                "descriptive_stats": {
                    "n": 6,
                    "min": 1,
                    "max": 3,
                },
            },
        ),

        # 3) Continuous simple series + explicit PDF title_template + save
        (
            lambda: pd.Series([0.0, 0.5, 1.0, 1.5, 2.0], name="nums"),
            {
                "is_discrete": False,
                "title_template": "PDF estimate of {name}{modifiers}",
                "file_name": "pdf.png",
            },
            {
                "chart_metadata": {
                    "title": "PDF estimate of nums (bw=scott)",
                    "xlabel": "Value",
                    "ylabel": "Density f(x)",
                    "file_name": "pdf.png",
                },
                "descriptive_stats": {
                    "n": 5,
                    "min": 0.0,
                    "max": 2.0,
                },
            },
        ),

        # 4) NaNs present → n counts non‑NaN; stats from cleaned data
        (
            lambda: pd.Series([1.0, np.nan, 2.0, np.nan, 5.0], name="with_nans"),
            {"is_discrete": True},
            {
                "chart_metadata": {"title": "PMF of with_nans"},
                "descriptive_stats": {
                    "n": 3,
                    "min": 1.0,
                    "max": 5.0,
                },
            },
        ),

        # 5) Title building with name override + modifiers (discrete)
        (
            lambda: pd.Series([10, 10, 20], name="ignored"),
            {
                "is_discrete": True,
                "name": "Price",
                "filter_desc": "NY only",
                "transform_desc": "winsorized",
            },
            {
                "chart_metadata": {
                    "title": "PMF of Price (NY only, winsorized)",
                },
                "descriptive_stats": {"n": 3},
            },
        ),

        # 6) Axis label overrides (continuous) + source
        (
            lambda: pd.Series([0.2, 0.4, 0.6, 0.8], name="x"),
            {
                "is_discrete": False,
                "title_template": "PDF estimate of {name}{modifiers}",
                "xlabel": "Score",
                "ylabel": "Density",
                "data_source": "UnitTest",
            },
            {
                "chart_metadata": {
                    "title": "PDF estimate of x (bw=scott)",
                    "xlabel": "Score",
                    "ylabel": "Density",
                    "data_source": "UnitTest",
                },
                "descriptive_stats": {"n": 4},
            },
        ),

        # 7) Discrete + explicit save filename
        (
            lambda: pd.Series([1, 1, 2, 3, 3, 3], name="save_me"),
            {"is_discrete": True, "file_name": "pmf.png"},
            {
                "chart_metadata": {"file_name": "pmf.png"},
                "descriptive_stats": {"n": 6},
            },
        ),

        # 8) Continuous + different bw_method (silverman)
        (
            lambda: pd.Series(np.linspace(-1, 1, 25), name="grid"),
            {
                "is_discrete": False,
                "title_template": "PDF estimate of {name}{modifiers}",
                "bw_method": "silverman",
            },
            {
                "chart_metadata": {"title": "PDF estimate of grid (bw=silverman)"},
                "descriptive_stats": {
                    "n": 25,
                    "mean": (lambda v: isinstance(v, float)),
                    "std": (lambda v: isinstance(v, float)),
                },
            },
        ),

        # 9) Discrete custom xlabel/ylabel overrides
        (
            lambda: pd.Series([2, 2, 2, 3, 4], name="k"),
            {
                "is_discrete": True,
                "xlabel": "Category",
                "ylabel": "Prob",
            },
            {
                "chart_metadata": {
                    "title": "PMF of k",
                    "xlabel": "Category",
                    "ylabel": "Prob",
                },
                "descriptive_stats": {"n": 5},
            },
        ),

        # 10) Continuous min/max & basic moments (approx) on small set
        (
            lambda: pd.Series([1, 2, 3, 4, 5], name="basic"),
            {"is_discrete": False, "title_template": "PDF estimate of {name}{modifiers}"},
            {
                "chart_metadata": {"title": "PDF estimate of basic (bw=scott)"},
                "descriptive_stats": {
                    "n": 5,
                    "min": 1,
                    "max": 5,
                    "mean": (lambda v: math.isclose(v, 3.0, rel_tol=1e-12, abs_tol=1e-12)),
                    "median": (lambda v: math.isclose(v, 3.0, rel_tol=1e-12, abs_tol=1e-12)),
                    "iqr": (lambda v: math.isclose(v, 2.0, rel_tol=1e-12, abs_tol=1e-12)),
                },
            },
        ),
    ],
    ids=[
        "0_empty_discrete",
        "1_empty_continuous",
        "2_discrete_with_source",
        "3_continuous_pdf_with_save",
        "4_nans_cleaning",
        "5_title_with_modifiers_discrete",
        "6_axis_overrides_continuous",
        "7_discrete_explicit_filename",
        "8_continuous_bw_silverman",
        "9_discrete_label_overrides",
        "10_continuous_basic_moments",
    ],
)
def test_distribution_probability_function_plot_data_driven(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    # If a file_name is provided, also set save_path to tmp_path
    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["save_path"] = tmp_path

    ctx = DistributionProbabilityFunctionContext(**kwargs)
    plot = DistributionProbabilityFunctionPlot(ctx)

    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
