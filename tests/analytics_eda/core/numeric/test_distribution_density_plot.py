import math
import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric import DistributionDensityContext, DistributionDensityPlot

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
        ctx = DistributionDensityContext()
        plot = DistributionDensityPlot(ctx)

        plot.run(s)


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) EMPTY → NaN stats, 0 modes; params echo bins/bin_method (both None)
        (
            lambda: pd.Series([], dtype="float64", name="nums"),
            {},
            {
                "chart_metadata": {
                    "title": "Distribution Density of nums",
                    "xlabel": "Value",
                    "ylabel": "Density",
                    "data_source": None,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "n": 0,
                    "modes_count": 0,
                    "entropy_bits": (lambda v: np.isnan(v)),
                    "skewness":     (lambda v: np.isnan(v)),
                    "kurtosis":     (lambda v: np.isnan(v)),
                    "quartile_skew":(lambda v: np.isnan(v)),
                    "pct_10": (lambda v: np.isnan(v)),
                    "pct_25": (lambda v: np.isnan(v)),
                    "pct_50": (lambda v: np.isnan(v)),
                    "pct_75": (lambda v: np.isnan(v)),
                    "pct_90": (lambda v: np.isnan(v)),
                    "params": {"bins": None, "bin_method": None},
                },
            },
        ),

        # 1) Defaults with no kwargs → bins=30 fallback; basic shape stats present
        (
            lambda: pd.Series([1, 2, 2, 3, 4, 4, 5], name="nums"),
            {},
            {
                "chart_metadata": {
                    "title": "Distribution Density of nums",
                    "xlabel": "Value",
                    "ylabel": "Density",
                    "data_source": None,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "n": 7,
                    "params": {"bins": 30, "bin_method": None},
                    "entropy_bits": (lambda v: isinstance(v, float)),
                    "skewness":     (lambda v: isinstance(v, float)),
                    "kurtosis":     (lambda v: isinstance(v, float)),
                    "modes_count":  (lambda v: isinstance(v, int) and v >= 1),
                    "quartile_skew":(lambda v: isinstance(v, float)),
                    "pct_10": (lambda v: isinstance(v, float)),
                    "pct_25": (lambda v: isinstance(v, float)),
                    "pct_50": (lambda v: isinstance(v, float)),
                    "pct_75": (lambda v: isinstance(v, float)),
                    "pct_90": (lambda v: isinstance(v, float)),
                },
            },
        ),

        # 2) Explicit integer bins
        (
            lambda: pd.Series(np.arange(20), name="x"),
            {"bins": 10},
            {
                "descriptive_stats": {
                    "n": 20,
                    "params": {"bins": 10, "bin_method": None},
                },
            },
        ),

        # 3) Explicit bin edges sequence
        (
            lambda: pd.Series([0, 1, 2, 3, 4, 5], name="edges"),
            {"bins": [0, 2, 4, 6]},
            {
                "descriptive_stats": {
                    "n": 6,
                    "params": {"bins": [0, 2, 4, 6], "bin_method": None},
                },
            },
        ),

        # 4) bin_method='sturges' → params carry method; bins is an int
        (
            lambda: pd.Series(np.random.default_rng(0).normal(size=50), name="rng"),
            {"bin_method": "sturges"},
            {
                "descriptive_stats": {
                    "n": 50,
                    "params": {
                        "bin_method": "sturges",
                        "bins": 7,
                    },
                },
            },
        ),

        # 5) bin_method='scott'
        (
            lambda: pd.Series(np.random.default_rng(1).normal(size=60), name="rng"),
            {"bin_method": "scott"},
            {
                "descriptive_stats": {
                    "n": 60,
                    "params": {
                        "bin_method": "scott",
                        "bins": 7,
                    },
                },
            },
        ),

        # 6) bin_method='freedman_diaconis'
        (
            lambda: pd.Series(np.random.default_rng(2).normal(size=80), name="rng"),
            {"bin_method": "freedman_diaconis"},
            {
                "descriptive_stats": {
                    "n": 80,
                    "params": {
                        "bin_method": "freedman_diaconis",
                        "bins": 8,
                    },
                },
            },
        ),

        # 7) bin_method='doane'
        (
            lambda: pd.Series(np.random.default_rng(3).normal(size=100), name="rng"),
            {"bin_method": "doane"},
            {
                "descriptive_stats": {
                    "n": 100,
                    "params": {
                        "bin_method": "doane",
                        "bins": 9,
                    },
                },
            },
        ),

        # 8) NaNs present → n counts non‑NaN; percentiles computed on cleaned
        (
            lambda: pd.Series([1.0, np.nan, 2.0, np.nan, 5.0], name="with_nans"),
            {},
            {
                "chart_metadata": {"title": "Distribution Density of with_nans"},
                "descriptive_stats": {
                    "n": 3,
                    "pct_50": (lambda v: isinstance(v, float)),
                },
            },
        ),

        # 9) Title with name override + modifiers (filter + transform)
        (
            lambda: pd.Series([10, 20, 20, 30], name="ignored"),
            {"name": "Price", "filter_desc": "NY only", "transform_desc": "log-scaled"},
            {
                "chart_metadata": {
                    "title": "Distribution Density of Price (NY only, log-scaled)",
                },
                "descriptive_stats": {"n": 4},
            },
        ),

        # 10) Custom title template ignores modifiers
        (
            lambda: pd.Series([1, 2, 3, 4], name="nums"),
            {"title_template": "My Density: {name}", "filter_desc": "ignored", "transform_desc": "ignored"},
            {
                "chart_metadata": {"title": "My Density: nums"},
                "descriptive_stats": {"n": 4},
            },
        ),

        # 11) Axis labels + data_source overrides + explicit save filename
        (
            lambda: pd.Series([0, 0, 1, 1, 2, 3, 5], name="fib"),
            {"xlabel": "Score", "ylabel": "PDF", "data_source": "UnitTest", "file_name": "density.png"},
            {
                "chart_metadata": {
                    "xlabel": "Score",
                    "ylabel": "PDF",
                    "data_source": "UnitTest",
                    "file_name": "density.png",
                },
                "descriptive_stats": {"n": 7},
            },
        ),

        # 12) Strict quartile-skew & percentile checks (unimodal small series)
        (
            lambda: pd.Series([1, 2, 2, 3, 4], name="test_series"),
            {},
            {
                "chart_metadata": {
                    "title": "Distribution Density of test_series",
                    "xlabel": "Value",
                    "ylabel": "Density",
                    "data_source": None,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "n": 5,
                    "quartile_skew": (
                        lambda v, s=pd.Series([1,2,2,3,4]):
                            math.isclose(
                                v,
                                (s.quantile(0.75) + s.quantile(0.25) - 2*s.median())
                                / (s.quantile(0.75) - s.quantile(0.25)),
                                rel_tol=1e-12, abs_tol=1e-12
                            )
                    ),
                    "pct_10": (lambda v, s=pd.Series([1,2,2,3,4]): math.isclose(v, s.quantile(0.10), rel_tol=1e-12, abs_tol=1e-12)),
                    "pct_25": (lambda v, s=pd.Series([1,2,2,3,4]): math.isclose(v, s.quantile(0.25), rel_tol=1e-12, abs_tol=1e-12)),
                    "pct_50": (lambda v, s=pd.Series([1,2,2,3,4]): math.isclose(v, s.quantile(0.50), rel_tol=1e-12, abs_tol=1e-12)),
                    "pct_75": (lambda v, s=pd.Series([1,2,2,3,4]): math.isclose(v, s.quantile(0.75), rel_tol=1e-12, abs_tol=1e-12)),
                    "pct_90": (lambda v, s=pd.Series([1,2,2,3,4]): math.isclose(v, s.quantile(0.90), rel_tol=1e-12, abs_tol=1e-12)),
                },
            },
        ),

        # 13) Right‑skewed (exponential) → skewness > 0, quartile_skew > 0
        (
            lambda: pd.Series(np.random.default_rng(4).exponential(scale=1.0, size=200), name="exp"),
            {},
            {
                "descriptive_stats": {
                    "n": 200,
                    "skewness":      (lambda v: v > 0),
                    "quartile_skew": (lambda v: v > 0),
                },
            },
        ),

        # 14) Left‑skewed (negative exponential) → skewness < 0, quartile_skew < 0
        (
            lambda: pd.Series(-np.random.default_rng(5).exponential(scale=1.0, size=200), name="negexp"),
            {},
            {
                "descriptive_stats": {
                    "n": 200,
                    "skewness":      (lambda v: v < 0),
                    "quartile_skew": (lambda v: v < 0),
                },
            },
        ),

        # 15) Bimodal mixture → modes_count >= 2
        (
            lambda: pd.Series(
                np.concatenate([
                    np.random.default_rng(6).normal(loc=-2, scale=0.5, size=150),
                    np.random.default_rng(7).normal(loc=+2, scale=0.5, size=150),
                ]),
                name="bimodal"
            ),
            {},
            {
                "descriptive_stats": {
                    "n": 300,
                    "modes_count": (lambda v: isinstance(v, int) and v >= 2),
                },
            },
        ),

        # 16) Save with defaults (only save_path/file_name)
        (
            lambda: pd.Series(range(10), name="nums"),
            {"file_name": "out.png"},
            {
                "chart_metadata": {
                    "title": "Distribution Density of nums",
                    "xlabel": "Value",
                    "ylabel": "Density",
                    "data_source": None,
                    "file_name": "out.png",
                },
            },
        ),

        # 17) Custom title + labels + data_source + save
        (
            lambda: pd.Series([0,0,1,1,2,3,5,5,5], name="s"),
            {
                "title_template": "Custom KDE",
                "xlabel": "X-axis",
                "ylabel": "Y-axis",
                "data_source": "UnitTest",
                "file_name": "shape.png",
            },
            {
                "chart_metadata": {
                    "title": "Custom KDE",
                    "xlabel": "X-axis",
                    "ylabel": "Y-axis",
                    "data_source": "UnitTest",
                    "file_name": "shape.png",
                },
                "descriptive_stats": {
                    "n": 9,
                    "modes_count": (lambda v: isinstance(v, int) and v >= 1),
                },
            },
        ),
    ],
    ids=[
        "0_empty",
        "1_defaults_bins30",
        "2_explicit_bins_int",
        "3_explicit_bins_edges",
        "4_binmethod_sturges",
        "5_binmethod_scott",
        "6_binmethod_freedman_diaconis",
        "7_binmethod_doane",
        "8_nans_cleaning",
        "9_title_with_modifiers",
        "10_custom_title_template_no_mods",
        "11_labels_source_and_save",
        "12_strict_quartile_skew_and_percentiles",
        "13_right_skew_positive",
        "14_left_skew_negative",
        "15_bimodal_modes_ge2",
        "16_save_with_defaults",
        "17_custom_title_labels_source_and_save",
    ],
)
def test_plot_distribution_density_param(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()
    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["save_path"] = tmp_path

    ctx = DistributionDensityContext(**kwargs)
    plot = DistributionDensityPlot(ctx)

    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
