import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric import plot_distribution_ecdf_vs_cdf

@pytest.mark.parametrize(
    "series_factory, expected_exc, match",
    [
        # Not a Series
        (lambda: [1, 2, 3], TypeError, r"Input must be a pandas Series\."),
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
    obj = series_factory()
    with pytest.raises(expected_exc, match=match):
        plot_distribution_ecdf_vs_cdf(obj, distribution_name="norm")

def test_invalid_distribution_name_raises_value_error():
    s = pd.Series([1, 2, 3], dtype=float, name="x")
    with pytest.raises(ValueError):
        plot_distribution_ecdf_vs_cdf(s, distribution_name="invalid")


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) EMPTY → minimal metadata; inferential params carry alpha; title includes dist
        (
            lambda: pd.Series([], dtype="float64", name="nums"),
            {"distribution_name": "norm"},
            {
                "chart_metadata": {
                    "title": "ECDF vs. Theoretical CDF of nums (norm)",
                    "xlabel": "Value",
                    "ylabel": "CDF",
                    "data_source": None,
                    "file_name": None,
                },
                "descriptive_stats": {"n": 0},
                "inferential_stats": {"params": {"alpha": 0.05}},
            },
        ),

        # 1) Defaults on small normal-like data (norm): presence/type checks for tests & params
        (
            lambda: pd.Series([0.0, 0.5, 1.0, 1.5, 2.0], name="x"),
            {"distribution_name": "norm"},
            {
                "chart_metadata": {
                    "title": "ECDF vs. Theoretical CDF of x (norm)",
                    "xlabel": "Value",
                    "ylabel": "CDF",
                    "data_source": None,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "n": 5,
                    "params": {
                        "distribution_name": "norm",
                        "distribution_fit": (lambda v: isinstance(v, tuple) and len(v) >= 2),
                    },
                },
                "inferential_stats": {
                    "params": {"alpha": 0.05},
                    "ks": {
                        "statistic": (lambda v: isinstance(v, float)),
                        "p_value": (lambda v: isinstance(v, float)),
                        "reject": (lambda v: isinstance(v, bool)),
                    },
                    "anderson": {
                        "statistic": (lambda v: isinstance(v, float)),
                        "critical_value": (lambda v: isinstance(v, float)),
                        "reject": (lambda v: isinstance(v, bool)),
                    },
                    "cvm": {
                        "statistic": (lambda v: isinstance(v, float)),
                        "p_value": (lambda v: isinstance(v, float)),
                        "reject": (lambda v: isinstance(v, bool)),
                    },
                },
            },
        ),

        # 2) Custom title/labels/source + save; title_template is respected and '(norm)' appended
        (
            lambda: pd.Series([0, 0, 1, 2, 3, 5], name="s"),
            {
                "distribution_name": "norm",
                "title_template": "My ECDF vs CDF",
                "xlabel": "X",
                "ylabel": "Y",
                "data_source": "UnitTest",
                "file_name": "ecdfcdf.png",
            },
            {
                "chart_metadata": {
                    "title": "My ECDF vs CDF (norm)",
                    "xlabel": "X",
                    "ylabel": "Y",
                    "data_source": "UnitTest",
                    "file_name": "ecdfcdf.png",
                },
                "descriptive_stats": {"n": 6},
            },
        ),

        # 3) Alpha override flows to inferential_stats.params
        (
            lambda: pd.Series([0.2, 0.4, 0.6, 0.8], name="alpha_test"),
            {"distribution_name": "expon", "alpha": 0.1},
            {
                "descriptive_stats": {"n": 4},
                "inferential_stats": {"params": {"alpha": 0.1}},
                "chart_metadata": {"title": "ECDF vs. Theoretical CDF of alpha_test (expon)"},
            },
        ),

        # 4) lognorm requires positive data → error noted, still returns default-ish metadata
        (
            lambda: pd.Series([0.0, 1.0, 2.0], name="pos_req"),
            {"distribution_name": "lognorm"},
            {
                "descriptive_stats": {
                    "n": 3,
                    "params": {"distribution_name": "lognorm"},
                    "error": "requires positive data",
                },
                "chart_metadata": {
                    "title": "ECDF vs. Theoretical CDF of pos_req (lognorm)",
                },
            },
        ),

        # 5) gamma requires positive data → error noted
        (
            lambda: pd.Series([-1.0, 0.0, 2.0], name="gamma_req"),
            {"distribution_name": "gamma"},
            {
                "descriptive_stats": {
                    "n": 3,
                    "params": {"distribution_name": "gamma"},
                    "error": "requires positive data",
                },
                "chart_metadata": {
                    "title": "ECDF vs. Theoretical CDF of gamma_req (gamma)",
                },
            },
        ),

        # 6) expon requires non-negative data → error noted
        (
            lambda: pd.Series([-1.0, 0.0, 1.0], name="expon_req"),
            {"distribution_name": "expon"},
            {
                "descriptive_stats": {
                    "n": 3,
                    "params": {"distribution_name": "expon"},
                    "error": "requires non-negative data",
                },
                "chart_metadata": {
                    "title": "ECDF vs. Theoretical CDF of expon_req (expon)",
                },
            },
        ),

        # 7) Name override + modifiers in title (filter + transform)
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
                    "title": "ECDF vs. Theoretical CDF of Price (NY only, winsorized) (norm)",
                },
                "descriptive_stats": {"n": 5},
            },
        ),

        # 8) lognormal on strictly positive data → checks for params + test fields present (no AD)
        (
            lambda: pd.Series([0.5, 1.2, 2.4, 4.8, 9.6], name="logpos"),
            {"distribution_name": "lognorm"},
            {
                "descriptive_stats": {
                    "n": 5,
                    "params": {
                        "distribution_name": "lognorm",
                        "distribution_fit": (lambda v: isinstance(v, tuple) and len(v) >= 2),
                    },
                },
                "inferential_stats": {
                    "ks": {
                        "statistic": (lambda v: isinstance(v, float)),
                        "p_value": (lambda v: isinstance(v, float)),
                        "reject": (lambda v: isinstance(v, bool)),
                    },
                    "cvm": {
                        "statistic": (lambda v: isinstance(v, float)),
                        "p_value": (lambda v: isinstance(v, float)),
                        "reject": (lambda v: isinstance(v, bool)),
                    },
                },
            },
        ),

        # 9) gamma positive types
        (
            lambda: pd.Series(np.random.default_rng(42).gamma(shape=2.0, scale=2.0, size=150), name="gamma_ok"),
            {"distribution_name": "gamma"},
            {
                "descriptive_stats": {
                    "n": 150,
                    "params": {
                        "distribution_name": "gamma",
                        "distribution_fit": (lambda v: isinstance(v, tuple) and len(v) >= 2),
                    },
                },
                "inferential_stats": {
                    "ks":  {
                        "statistic": (lambda v: isinstance(v, float)),
                        "p_value":  (lambda v: isinstance(v, float)),
                        "reject":   (lambda v: isinstance(v, bool)),
                    },
                    "cvm": {
                        "statistic": (lambda v: isinstance(v, float)),
                        "p_value":  (lambda v: isinstance(v, float)),
                        "reject":   (lambda v: isinstance(v, bool)),
                    },
                    # note: no 'anderson' expected for gamma
                },
                "chart_metadata": {
                    "title": "ECDF vs. Theoretical CDF of gamma_ok (gamma)",
                },
            },
        ),
    ],
    ids=[
        "0_empty_norm",
        "1_defaults_norm_types",
        "2_custom_title_labels_source_save",
        "3_alpha_override_expon",
        "4_support_check_lognorm_positive",
        "5_support_check_gamma_positive",
        "6_support_check_expon_nonnegative",
        "7_title_with_modifiers_norm",
        "8_lognorm_positive_types",
        "9_gamma_positive_types",
    ],
)
def test_plot_distribution_ecdf_vs_cdf_param(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    # If a file_name is provided, also set save_path to tmp_path
    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["save_path"] = tmp_path

    payload = plot_distribution_ecdf_vs_cdf(s, **kwargs)

    assert_plot_metadata(payload, expect, tmp_path)
