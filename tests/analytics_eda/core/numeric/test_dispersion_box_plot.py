import math

import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric import DispersionBoxPlot, DispersionBoxPlotContext


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
        ctx = DispersionBoxPlotContext()
        plot = DispersionBoxPlot(ctx)

        plot.run(s)


test_series_1_to_5 = pd.Series([1, 2, 3, 4, 5])


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) EMPTY → all stats NaN/0; params carries std_outlier_multiplier
        (
            lambda: pd.Series([], dtype="float64", name="nums"),
            {},
            {
                "chart_metadata": {
                    "title": "Dispersion of nums (IQR & Outliers)",
                    "ylabel": "Value",
                    "data_source": None,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "n": 0,
                    "mean": None,
                    "min": None,
                    "max": None,
                    "range": None,
                    "pct_25": None,
                    "median": None,
                    "pct_75": None,
                    "iqr": None,
                    "iqr_lower_bound": None,
                    "iqr_upper_bound": None,
                    "outlier_lower_count": 0,
                    "outlier_upper_count": 0,
                },
                "draft_descriptive_findings": {
                    "context": "No non-null observations.",
                    "primary_finding": None,
                    "secondary_finding": None,
                },
            },
        ),
        # 1) SIMPLE DEFAULTS → check n, key percentiles & IQR
        (
            lambda: pd.Series([1, 2, 3, 4], name="simple"),
            {},
            {
                "chart_metadata": {
                    "title": "Dispersion of simple (IQR & Outliers)",
                    "ylabel": "Value",
                },
                "descriptive_stats": {
                    "n": 4,
                    "min": 1,
                    "max": 4,
                    "range": 3,
                    "pct_25": (lambda v: math.isclose(v, 1.75, rel_tol=1e-12, abs_tol=1e-12)),
                    "median": (lambda v: math.isclose(v, 2.5, rel_tol=1e-12, abs_tol=1e-12)),
                    "pct_75": (lambda v: math.isclose(v, 3.25, rel_tol=1e-12, abs_tol=1e-12)),
                    "iqr": (lambda v: math.isclose(v, 1.5, rel_tol=1e-12, abs_tol=1e-12)),
                },
                "draft_descriptive_findings": {
                    "context": "n = 4 • Q1 1.75 • Q3 3.25",
                    "primary_finding": "Median 2.50 with middle 50% spanning 1.75–3.25 (IQR 1.50).",
                    "secondary_finding": "Overall range extends from 1.00 to 4.00.",
                },
            },
        ),
        # 2) FULL STATS DEFAULTS → match full baseline computation
        (
            lambda: pd.Series([1, 2, 3, 4, 5], name="numeric_series"),
            {},
            {
                "chart_metadata": {
                    "title": "Dispersion of numeric_series (IQR & Outliers)",
                    "ylabel": "Value",
                    "data_source": None,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "n": 5,
                    "mean": (lambda v, s=test_series_1_to_5: math.isclose(v, s.mean(), rel_tol=1e-12, abs_tol=1e-12)),
                    "min": 1,
                    "max": 5,
                    "range": 4,
                    "pct_25": (lambda v, s=test_series_1_to_5: math.isclose(v, s.quantile(0.25), rel_tol=1e-12, abs_tol=1e-12)),
                    "median": (lambda v, s=test_series_1_to_5: math.isclose(v, s.median(), rel_tol=1e-12, abs_tol=1e-12)),
                    "pct_75": (lambda v, s=test_series_1_to_5: math.isclose(v, s.quantile(0.75), rel_tol=1e-12, abs_tol=1e-12)),
                    "iqr": (lambda v, s=test_series_1_to_5: math.isclose(v, s.quantile(0.75) - s.quantile(0.25), rel_tol=1e-12, abs_tol=1e-12)),
                    "iqr_lower_bound": (lambda v, s=test_series_1_to_5: math.isclose(v, s.quantile(0.25) - 1.5 * (s.quantile(0.75) - s.quantile(0.25)), rel_tol=1e-12, abs_tol=1e-12)),
                    "iqr_upper_bound": (lambda v, s=test_series_1_to_5: math.isclose(v, s.quantile(0.75) + 1.5 * (s.quantile(0.75) - s.quantile(0.25)), rel_tol=1e-12, abs_tol=1e-12)),
                },
            },
        ),
        # 3) NaNs CLEANING → n counts non‑NaN; min/max/range on cleaned data
        (
            lambda: pd.Series([1.0, np.nan, 2.0, 5.0, np.nan, -1.0], name="with_nans"),
            {},
            {
                "chart_metadata": {"title": "Dispersion of with_nans (IQR & Outliers)"},
                "descriptive_stats": {
                    "n": 4,  # [1.0, 2.0, 5.0, -1.0]
                    "min": -1.0,
                    "max": 5.0,
                    "range": 6.0,
                },
            },
        ),
        # 4) MEAN WHEN ZERO → still reported even when CV undefined before
        (
            lambda: pd.Series([-1, 0, 1], name="zero_mean"),
            {},
            {
                "descriptive_stats": {
                    "n": 3,
                    "mean": 0.0,
                },
            },
        ),
        # 5) OUTLIER UPPER FLAG → IQR fence catches single spike
        (
            lambda: pd.Series([0, 0, 0, 0, 10], name="spike"),
            {},
            {
                "descriptive_stats": {
                    "n": 5,
                    "outlier_lower_count": 0,
                    "outlier_upper_count": 1,
                },
            },
        ),
        # 6) TITLE WITH MODIFIERS → name override + filter/transform in title
        (
            lambda: pd.Series([10, 20, 30], name="ignored"),
            {"name": "Price", "filter_desc": "NY only", "transform_desc": "winsorized"},
            {
                "chart_metadata": {
                    "title": "Dispersion of Price (NY only, winsorized) (IQR & Outliers)",
                },
                "descriptive_stats": {"n": 3},
            },
        ),
        # 7) LABELS/SOURCE & SAVE → override labels/source, save with filename
        (
            lambda: pd.Series([2, 4, 6, 8, 10], name="even"),
            {"ylabel": "Score", "data_source": "UnitTest", "file_name": "box.png"},
            {
                "chart_metadata": {
                    "ylabel": "Score",
                    "data_source": "UnitTest",
                    "file_name": "box.png",
                },
                "descriptive_stats": {"n": 5},
            },
        ),
        # 8) BOTH‑TAIL EXTREMES → IQR fences flag both tails; also save
        (
            lambda: pd.Series([-10, 0, 1, 2, 100], name="x"),
            {"file_name": "dispersion.png"},
            {
                "chart_metadata": {"file_name": "dispersion.png"},
                "descriptive_stats": {
                    "n": 5,
                    "outlier_lower_count": 1,
                    "outlier_upper_count": 1,
                },
                "draft_descriptive_findings": {
                    "context": "n = 5 • Q1 0.00 • Q3 2.00",
                    "primary_finding": "Median 1.00 with middle 50% spanning 0.00–2.00 (IQR 2.00).",
                    "secondary_finding": "2 observations sit outside the IQR fences (1 low / 1 high).",
                },
            },
        ),
        # 9) VIOLIN OUTLIERS & SAVE → outliers on both sides; iqr is float; saved
        (
            lambda: pd.Series(np.concatenate([np.random.default_rng(0).normal(loc=0, scale=1, size=100), [5, -5]]), name="x"),
            {
                "file_name": "violin_dispersion.png",
                "title_template": "Violin Dispersion Test",
                "ylabel": "Units",
                "data_source": "UnitTest",
            },
            {
                "chart_metadata": {
                    "title": "Violin Dispersion Test",
                    "ylabel": "Units",
                    "data_source": "UnitTest",
                    "file_name": "violin_dispersion.png",
                },
                "descriptive_stats": {
                    "n": (lambda v, s_len=102: v == s_len),
                    "iqr": (lambda v: isinstance(v, float)),
                    "outlier_lower_count": (lambda v: v >= 1),
                    "outlier_upper_count": (lambda v: v >= 1),
                },
            },
        ),
    ],
    ids=[
        "0_empty",
        "1_defaults_simple_series",
        "2_defaults_full_stats_simple_series",
        "3_nans_cleaning",
        "4_cv_nan_when_mean_zero",
        "5_outlier_upper_flag_k1",
        "6_title_with_modifiers",
        "7_labels_source_and_save",
        "8_both_tail_extremes_small_k",
        "9_violin_outliers_and_save",
    ],
)
def test_dispersion_box_plot_data_driven(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    # If a file_name is provided, also set save_path to tmp_path
    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["save_path"] = tmp_path

    ctx = DispersionBoxPlotContext(**kwargs)
    plot = DispersionBoxPlot(ctx)

    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
