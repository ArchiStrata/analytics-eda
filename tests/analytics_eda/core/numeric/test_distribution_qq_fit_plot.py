import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric import DistributionQqFitContext, DistributionQqFitPlot

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
        ctx = DistributionQqFitContext(distribution_name="norm")
        plot = DistributionQqFitPlot(ctx)

        plot.run(s)


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) EMPTY → minimal stats NaN; inferential params carry alpha; title includes dist
        (
            lambda: pd.Series([], dtype="float64", name="nums"),
            {"distribution_name": "norm"},
            {
                "chart_metadata": {
                    "title": "Q–Q Plot Fit Assessment of nums (fitted to norm)",
                    "xlabel": "Theoretical Quantiles",
                    "ylabel": "Sample Quantiles",
                    "data_source": None,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "intercept": (lambda v: np.isnan(v)),
                    "slope": (lambda v: np.isnan(v)),
                    "r_squared": (lambda v: np.isnan(v)),
                    "median_residual": (lambda v: np.isnan(v)),
                    "iqr_residual": (lambda v: np.isnan(v)),
                    "max_abs_residual": (lambda v: np.isnan(v)),
                    "skewness": (lambda v: np.isnan(v)),
                    "kurtosis": (lambda v: np.isnan(v)),
                    "min": (lambda v: np.isnan(v)),
                },
                "inferential_stats": {
                    "params": {"alpha": 0.05, "distribution_name": "norm"},
                },
            },
        ),

        # 1) norm, small n (<50) → Shapiro present, no D’Agostino (since n<20), no JB
        (
            lambda: pd.Series(np.random.default_rng(0).normal(size=10), name="x"),
            {"distribution_name": "norm", "alpha": 0.05},
            {
                "chart_metadata": {"title": "Q–Q Plot Fit Assessment of x (fitted to norm)"},
                "descriptive_stats": {
                    "r_squared": (lambda v: isinstance(v, float) and 0.0 <= v <= 1.0),
                },
                "inferential_stats": {
                    "params": {"alpha": 0.05, "distribution_name": "norm"},
                    "shapiro": {
                        "statistic": (lambda v: isinstance(v, float)),
                        "p_value": (lambda v: isinstance(v, float)),
                        "reject": (lambda v: isinstance(v, bool)),
                    },
                },
            },
        ),

        # 2) norm, mid n (≥20 and <50) → Shapiro + D’Agostino present, no JB
        (
            lambda: pd.Series(np.random.default_rng(1).normal(size=30), name="mid"),
            {"distribution_name": "norm"},
            {
                "chart_metadata": {"title": "Q–Q Plot Fit Assessment of mid (fitted to norm)"},
                "inferential_stats": {
                    "params": {"alpha": 0.05, "distribution_name": "norm"},
                    "shapiro": {
                        "statistic": (lambda v: isinstance(v, float)),
                        "p_value": (lambda v: isinstance(v, float)),
                        "reject": (lambda v: isinstance(v, bool)),
                    },
                    "dagostino_pearson": {
                        "statistic": (lambda v: isinstance(v, float)),
                        "p_value": (lambda v: isinstance(v, float)),
                        "reject": (lambda v: isinstance(v, bool)),
                    },
                },
            },
        ),

        # 3) norm, huge n (>2000) → Shapiro excluded (n>=50), D’Agostino present, JB present
        (
            lambda: pd.Series(np.random.default_rng(2).normal(size=2100), name="huge"),
            {"distribution_name": "norm"},
            {
                "chart_metadata": {"title": "Q–Q Plot Fit Assessment of huge (fitted to norm)"},
                "descriptive_stats": {
                    "r_squared": (lambda v: isinstance(v, float) and 0.0 <= v <= 1.0),
                },
                "inferential_stats": {
                    "params": {"alpha": 0.05, "distribution_name": "norm"},
                    "dagostino_pearson": {
                        "statistic": (lambda v: isinstance(v, float)),
                        "p_value": (lambda v: isinstance(v, float)),
                        "reject": (lambda v: isinstance(v, bool)),
                    },
                    "jarque_bera": {
                        "statistic": (lambda v: isinstance(v, float)),
                        "p_value": (lambda v: isinstance(v, float)),
                        "reject": (lambda v: isinstance(v, bool)),
                    },
                    "reject_normality": (lambda v: isinstance(v, bool)),
                },
            },
        ),

        # 4) Custom title/labels/source + save; title_template respected and (dist) appended
        (
            lambda: pd.Series([0, 0.5, 1.2, 1.7, 2.1], name="s"),
            {
                "distribution_name": "norm",
                "title_template": "QQ Fit of {name}{modifiers}",
                "xlabel": "Theo Q",
                "ylabel": "Sample Q",
                "data_source": "UnitTest",
                "file_name": "qq.png",
            },
            {
                "chart_metadata": {
                    "title": "QQ Fit of s (fitted to norm)",
                    "xlabel": "Theo Q",
                    "ylabel": "Sample Q",
                    "data_source": "UnitTest",
                    "file_name": "qq.png",
                },
            },
        ),

        # 5) Alpha override reflected in inferential params
        (
            lambda: pd.Series(np.linspace(-1, 1, 40), name="alpha"),
            {"distribution_name": "norm", "alpha": 0.10},
            {
                "inferential_stats": {"params": {"alpha": 0.10, "distribution_name": "norm"}},
                "chart_metadata": {"title": "Q–Q Plot Fit Assessment of alpha (fitted to norm)"},
            },
        ),

        # 6) lognorm (positive support) → no normality tests; stats types present
        (
            lambda: pd.Series(np.random.default_rng(3).lognormal(mean=0.0, sigma=0.5, size=80), name="logpos"),
            {"distribution_name": "lognorm"},
            {
                "chart_metadata": {"title": "Q–Q Plot Fit Assessment of logpos (fitted to lognorm)"},
                "descriptive_stats": {
                    "slope": (lambda v: isinstance(v, float)),
                    "intercept": (lambda v: isinstance(v, float)),
                    "r_squared": (lambda v: isinstance(v, float)),
                },
                "inferential_stats": {
                    "params": {"alpha": 0.05, "distribution_name": "lognorm"},
                },
            },
        ),

        # 7) gamma (positive support) → no normality tests
        (
            lambda: pd.Series(np.random.default_rng(4).gamma(shape=2.0, scale=2.0, size=100), name="g"),
            {"distribution_name": "gamma"},
            {
                "chart_metadata": {"title": "Q–Q Plot Fit Assessment of g (fitted to gamma)"},
                "inferential_stats": {"params": {"alpha": 0.05, "distribution_name": "gamma"}},
            },
        ),

        # 8) expon (non-negative support) → no normality tests
        (
            lambda: pd.Series(np.random.default_rng(5).exponential(scale=1.0, size=120), name="e"),
            {"distribution_name": "expon"},
            {
                "chart_metadata": {"title": "Q–Q Plot Fit Assessment of e (fitted to expon)"},
                "inferential_stats": {"params": {"alpha": 0.05, "distribution_name": "expon"}},
            },
        ),

        # 9) Title with name override + modifiers
        (
            lambda: pd.Series([1, 2, 3, 4, 5], name="ignored"),
            {
                "distribution_name": "norm",
                "name": "Price",
                "filter_desc": "NY only",
                "transform_desc": "winsorized",
            },
            {
                "chart_metadata": {
                    "title": "Q–Q Plot Fit Assessment of Price (NY only, winsorized, fitted to norm)",
                },
            },
        ),

        # 10) NaNs present → cleaned n used; still produces valid stats
        (
            lambda: pd.Series([1.0, np.nan, 2.0, np.nan, 5.0], name="with_nans"),
            {"distribution_name": "norm"},
            {
                "chart_metadata": {"title": "Q–Q Plot Fit Assessment of with_nans (fitted to norm)"},
                "descriptive_stats": {
                    "slope": (lambda v: isinstance(v, float)),
                    "r_squared": (lambda v: isinstance(v, float) and 0.0 <= v <= 1.0),
                },
            },
        ),

        # 11) Explicit file_name triggers save to tmp_path
        (
            lambda: pd.Series(np.random.default_rng(6).normal(size=25), name="save_me"),
            {"distribution_name": "norm", "file_name": "qq_fit.png"},
            {
                "chart_metadata": {"file_name": "qq_fit.png"},
            },
        ),
    ],
    ids=[
        "0_empty_norm",
        "1_norm_small_shapiro_only",
        "2_norm_mid_shapiro_and_dp",
        "3_norm_huge_includes_jb",
        "4_custom_title_labels_source_save",
        "5_alpha_override_param",
        "6_lognorm_positive",
        "7_gamma_positive",
        "8_expon_nonnegative",
        "9_title_with_modifiers",
        "10_nans_cleaning",
        "11_explicit_filename_saves",
    ],
)
def test_plot_distribution_qq_fit_param(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    # If a file_name is provided, also set save_path to tmp_path
    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["save_path"] = tmp_path

    ctx = DistributionQqFitContext(**kwargs)
    plot = DistributionQqFitPlot(ctx)

    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
