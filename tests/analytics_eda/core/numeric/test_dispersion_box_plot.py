import math
import pytest
import numpy as np
import pandas as pd

from analytics_eda.core.numeric import DispersionBoxplotContext, DispersionBoxPlot


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
        ctx = DispersionBoxplotContext()
        plot = DispersionBoxPlot(ctx)

        plot.run(s)


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
                    "params": {"std_outlier_multiplier": 4.0},
                    "n": 0,
                    "mean": (lambda v: np.isnan(v)),
                    "std":  (lambda v: np.isnan(v)),
                    "var":  (lambda v: np.isnan(v)),
                    "min":  (lambda v: np.isnan(v)),
                    "max":  (lambda v: np.isnan(v)),
                    "range":(lambda v: np.isnan(v)),
                    "mad":  (lambda v: np.isnan(v)),
                    "cv":   (lambda v: np.isnan(v)),
                    "pct_10": (lambda v: np.isnan(v)),
                    "pct_25": (lambda v: np.isnan(v)),
                    "pct_75": (lambda v: np.isnan(v)),
                    "pct_90": (lambda v: np.isnan(v)),
                    "iqr":    (lambda v: np.isnan(v)),
                    "extreme_lower_count": 0,
                    "extreme_upper_count": 0,
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
                    "pct_75": (lambda v: math.isclose(v, 3.25, rel_tol=1e-12, abs_tol=1e-12)),
                    "iqr":   (lambda v: math.isclose(v, 1.5,  rel_tol=1e-12, abs_tol=1e-12)),
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
                    "std": (lambda v, s=pd.Series([1,2,3,4,5]): math.isclose(v, s.std(), rel_tol=1e-12, abs_tol=1e-12)),
                    "var": (lambda v, s=pd.Series([1,2,3,4,5]): math.isclose(v, s.var(), rel_tol=1e-12, abs_tol=1e-12)),
                    "min": 1,
                    "max": 5,
                    "range": 4,
                    "mad": (lambda v, s=pd.Series([1,2,3,4,5]): math.isclose(v, (s - s.mean()).abs().mean(), rel_tol=1e-12, abs_tol=1e-12)),
                    "cv":  (lambda v, s=pd.Series([1,2,3,4,5]): math.isclose(v, s.std()/s.mean(), rel_tol=1e-12, abs_tol=1e-12)),
                    "pct_10": (lambda v, s=pd.Series([1,2,3,4,5]): math.isclose(v, s.quantile(0.10), rel_tol=1e-12, abs_tol=1e-12)),
                    "pct_25": (lambda v, s=pd.Series([1,2,3,4,5]): math.isclose(v, s.quantile(0.25), rel_tol=1e-12, abs_tol=1e-12)),
                    "pct_75": (lambda v, s=pd.Series([1,2,3,4,5]): math.isclose(v, s.quantile(0.75), rel_tol=1e-12, abs_tol=1e-12)),
                    "pct_90": (lambda v, s=pd.Series([1,2,3,4,5]): math.isclose(v, s.quantile(0.90), rel_tol=1e-12, abs_tol=1e-12)),
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

        # 4) CV WHEN MEAN==0 → cv is NaN
        (
            lambda: pd.Series([-1, 0, 1], name="zero_mean"),
            {},
            {
                "descriptive_stats": {
                    "n": 3,
                    "mean": 0.0,
                    "cv": (lambda v: np.isnan(v)),
                },
            },
        ),

        # 5) OUTLIER UPPER FLAG → custom k=1.0 should flag the single upper extreme
        #    s=[0,0,0,0,10], mean=2, std≈4.472; upper bound≈6.47 → one upper outlier
        (
            lambda: pd.Series([0, 0, 0, 0, 10], name="spike"),
            {"std_outlier_multiplier": 1.0},
            {
                "descriptive_stats": {
                    "params": {"std_outlier_multiplier": 1.0},
                    "n": 5,
                    "extreme_lower_count": 0,
                    "extreme_upper_count": 1,
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

        # 8) BOTH‑TAIL EXTREMES → tiny k flags both tails; also save
        (
            lambda: pd.Series([-10, 0, 1, 2, 100], name="x"),
            {"std_outlier_multiplier": 0.3, "file_name": "dispersion.png"},
            {
                "chart_metadata": {"file_name": "dispersion.png"},
                "descriptive_stats": {
                    "n": 5,
                    "params": {"std_outlier_multiplier": 0.3},
                    "extreme_lower_count": 4,
                    "extreme_upper_count": 1,
                },
            },
        ),

        # 9) VIOLIN OUTLIERS & SAVE → outliers on both sides; iqr is float; saved
        (
            lambda: pd.Series(
                np.concatenate([np.random.default_rng(0).normal(loc=0, scale=1, size=100), [5, -5]]),
                name="x"
            ),
            {
                "std_outlier_multiplier": 2.0,
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
                    "extreme_lower_count": (lambda v: v >= 1),
                    "extreme_upper_count": (lambda v: v >= 1),
                    "params": {"std_outlier_multiplier": 2.0},
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
def test_plot_dispersion_boxplot_param(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    # If a file_name is provided, also set save_path to tmp_path
    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["save_path"] = tmp_path

    ctx = DispersionBoxplotContext(**kwargs)
    plot = DispersionBoxPlot(ctx)

    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
