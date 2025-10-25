import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric import CentralTendencyViolinContext, CentralTendencyViolinPlot


def test_mean_ci_method_invalid():
    s = pd.Series([1, 2, 3], name="C")
    with pytest.raises(ValueError):
        ctx = CentralTendencyViolinContext(mean_ci_method="invalid")
        plot = CentralTendencyViolinPlot(ctx)

        plot.run(s)


def test_median_ci_method_invalid():
    s = pd.Series([1, 2, 3], name="E")
    with pytest.raises(ValueError):
        ctx = CentralTendencyViolinContext(median_ci_method="invalid")
        plot = CentralTendencyViolinPlot(ctx)

        plot.run(s)


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
        ctx = CentralTendencyViolinContext()
        plot = CentralTendencyViolinPlot(ctx)

        plot.run(s)


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) EMPTY numeric series
        (
            lambda: pd.Series([], dtype="float64", name="nums"),
            {},
            {
                "chart_metadata": {
                    "title": "Distribution of nums: Central Tendency (Violin)",
                    "xlabel": "Value",
                    "ylabel": "Density",
                    "data_source": None,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "n": 0,
                    "mean": None,
                    "median": None,
                    "mean_ci": [None, None],
                    "median_ci": [None, None],
                    "params": {
                        "mean_ci_method": "t",
                        "median_ci_method": "bootstrap",
                    },
                },
            },
        ),

        # 1) Defaults on small integer data (mean CI via t, median CI via bootstrap)
        (
            lambda: pd.Series([1, 2, 2, 3, 3, 3], name="nums"),
            {},
            {
                "chart_metadata": {
                    "title": "Distribution of nums: Central Tendency (Violin)",
                    "xlabel": "Value",
                    "ylabel": "Density",
                },
                "descriptive_stats": {
                    "n": 6,
                    "mean": pytest.approx(7/3, rel=1e-9, abs=1e-10),
                    "median": 2.5,
                    # Just check structure of CIs (don’t pin numeric values)
                    "mean_ci": (lambda v: isinstance(v, tuple) and len(v) == 2 and all(np.isfinite(x) for x in v)),
                    "median_ci": (lambda v: isinstance(v, tuple) and len(v) == 2 and all(np.isfinite(x) for x in v)),
                    "params": {
                        "mean_ci_method": "t",
                        "median_ci_method": "bootstrap",
                    },
                },
            },
        ),

        # 2) Override mean CI to 'bootstrap' (structure checks only)
        (
            lambda: pd.Series([10, 20, 20, 30, 30, 30, 40], name="vals"),
            {"mean_ci_method": "bootstrap"},
            {
                "descriptive_stats": {
                    "n": 7,
                    "params": {
                        "mean_ci_method": "bootstrap",
                        "median_ci_method": "bootstrap",
                    },
                    "mean_ci": (lambda v: isinstance(v, tuple) and len(v) == 2 and all(np.isfinite(x) for x in v)),
                    "median_ci": (lambda v: isinstance(v, tuple) and len(v) == 2 and all(np.isfinite(x) for x in v)),
                },
            },
        ),

        # 3) NaNs present → n counts non-NaN; CI tuples exist
        (
            lambda: pd.Series([1.0, np.nan, 2.0, 2.0, 3.0, np.nan, 4.0], name="with_nans"),
            {},
            {
                "chart_metadata": {
                    "title": "Distribution of with_nans: Central Tendency (Violin)",
                },
                "descriptive_stats": {
                    "n": 5,
                    "mean": (lambda v: np.isfinite(v)),
                    "median": (lambda v: np.isfinite(v)),
                    "mean_ci": (lambda v: isinstance(v, tuple) and len(v) == 2),
                    "median_ci": (lambda v: isinstance(v, tuple) and len(v) == 2),
                },
            },
        ),

        # 4) Title building with name override + modifiers
        (
            lambda: pd.Series([5, 6, 7, 8], name="ignored"),
            {"name": "Price", "filter_desc": "NY only", "transform_desc": "standardized"},
            {
                "chart_metadata": {
                    "title": "Distribution of Price (NY only, standardized): Central Tendency (Violin)",
                },
                "descriptive_stats": {"n": 4},
            },
        ),

        # 5) Custom labels + data_source + explicit save filename
        (
            lambda: pd.Series([0, 1, 1, 2, 3, 5, 8], name="fib"),
            {
                "xlabel": "Score",
                "ylabel": "Density",
                "data_source": "UnitTest",
                "file_name": "violin.png",
            },
            {
                "chart_metadata": {
                    "xlabel": "Score",
                    "ylabel": "Density",
                    "data_source": "UnitTest",
                    "file_name": "violin.png",
                },
                "descriptive_stats": {"n": 7},
            },
        ),

        # 6) Deterministic small set to check mean/median precisely (t & bootstrap structures)
        (
            lambda: pd.Series([2, 2, 2, 2], name="const"),
            {},
            {
                "descriptive_stats": {
                    "n": 4,
                    "mean": 2.0,
                    "median": 2.0,
                    "mean_ci": (lambda v: isinstance(v, tuple) and len(v) == 2),   # sem=0 → t.interval may return nan; structure is enough
                    "median_ci": (lambda v: isinstance(v, tuple) and len(v) == 2),
                    "params": {
                        "mean_ci_method": "t",
                        "median_ci_method": "bootstrap",
                    },
                },
            },
        ),
        # X1) Empty: also assert inferential params block
        (
            lambda: pd.Series([], dtype=float, name="empty"),
            {},
            {
                "chart_metadata": {
                    "title": "Distribution of empty: Central Tendency (Violin)",
                    "xlabel": "Value",
                    "ylabel": "Density",
                    "data_source": None,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "n": 0,
                    "mean": None,
                    "median": None,
                    "mean_ci": [None, None],
                    "median_ci": [None, None],
                    "params": {"mean_ci_method": "t", "median_ci_method": "bootstrap"},
                },
                "inferential_stats": {
                    "params": {
                        "alpha": 0.05,
                        "bootstrap_samples": 1000,
                        "popmean": None,
                        "popmedian": None,
                        "popvariance": None,
                    }
                },
            },
        ),

        # X2) Combined: custom title template + name + modifiers + labels/source (no save)
        (
            lambda: pd.Series([1, 2, 3], dtype=float, name="X"),
            {
                "xlabel": "My X",
                "ylabel": "My Y",
                "data_source": "DataSrc",
                "title_template": "Test {name}: Central {modifiers}",
                "name": "OverrideName",
                "filter_desc": "filtered",
                "transform_desc": "transformed",
            },
            {
                "chart_metadata": {
                    "xlabel": "My X",
                    "ylabel": "My Y",
                    "data_source": "DataSrc",
                    # Title must include name and both modifiers
                    "title": (lambda t: "OverrideName" in t and "filtered" in t and "transformed" in t),
                },
                "descriptive_stats": {"n": 3},
            },
        ),

        # X3) Save & assert mean CI method = 't'
        (
            lambda: pd.Series([1, 2, 3, 4, 5], name="A"),
            {"mean_ci_method": "t", "file_name": "mean_ci_t.png"},
            {
                "chart_metadata": {"file_name": "mean_ci_t.png"},
                "descriptive_stats": {
                    "n": 5,
                    "mean_ci": (lambda v: isinstance(v, tuple) and len(v) == 2),
                    "params": {"mean_ci_method": "t", "median_ci_method": "bootstrap"},
                },
            },
        ),

        # X4) Save & assert mean CI method = 'bootstrap'
        (
            lambda: pd.Series([1, 2, 3, 4, 5], name="A"),
            {"mean_ci_method": "bootstrap", "file_name": "mean_ci_bootstrap.png"},
            {
                "chart_metadata": {"file_name": "mean_ci_bootstrap.png"},
                "descriptive_stats": {
                    "n": 5,
                    "mean_ci": (lambda v: isinstance(v, tuple) and len(v) == 2),
                    "params": {"mean_ci_method": "bootstrap", "median_ci_method": "bootstrap"},
                },
            },
        ),

        # X5) Save & assert median CI method = 'bootstrap'
        (
            lambda: pd.Series([5, 6, 7, 8, 9], name="D"),
            {"median_ci_method": "bootstrap", "bootstrap_samples": 100, "file_name": "median_ci_bootstrap.png"},
            {
                "chart_metadata": {"file_name": "median_ci_bootstrap.png"},
                "descriptive_stats": {
                    "n": 5,
                    "median_ci": (lambda v: isinstance(v, tuple) and len(v) == 2),
                    "params": {"mean_ci_method": "t", "median_ci_method": "bootstrap"},
                },
            },
        ),

        # X6) popmean + popvariance inferential tests + save
        (
            lambda: pd.Series(np.random.default_rng(0).normal(loc=10, scale=2, size=50), name="F"),
            {"popmean": 10.0, "popvariance": 5.0, "file_name": "popmean.png"},
            {
                "chart_metadata": {"file_name": "popmean.png"},
                "inferential_stats": {
                    "params": (lambda p: p["popmean"] == 10.0 and p["popvariance"] == 5.0),
                    # presence/type checks for tests
                    "popmean": (lambda d:
                        isinstance(d.get("cohens_d"), float) and
                        isinstance(d["t_test"]["statistic"], float) and
                        isinstance(d["t_test"]["p_value"], float) and
                        isinstance(d["t_test"]["reject"], bool) and
                        isinstance(d["z_test"]["statistic"], float) and
                        isinstance(d["z_test"]["p_value"], float) and
                        isinstance(d["z_test"]["reject"], bool)
                    ),
                },
            },
        ),

        # X7) popmedian inferential tests + save
        (
            lambda: pd.Series([0.0, 1.0, -1.0, 2.5, 0.0], dtype=float, name="G"),
            {"popmedian": 0.0, "file_name": "popmedian.png"},
            {
                "chart_metadata": {"file_name": "popmedian.png"},
                "inferential_stats": {
                    "params": (lambda p: p["popmedian"] == 0.0),
                    "popmedian": (lambda d:
                        isinstance(d["wilcoxon"]["statistic"], float) and
                        isinstance(d["wilcoxon"]["p_value"], float) and
                        isinstance(d["wilcoxon"]["reject"], bool) and
                        isinstance(d["sign_test"]["n"], int) and
                        isinstance(d["sign_test"]["num_positive"], int) and
                        isinstance(d["sign_test"]["num_negative"], int) and
                        isinstance(d["sign_test"]["p_value"], float) and
                        isinstance(d["sign_test"]["reject"], bool)
                    ),
                },
            },
        ),
    ],
    ids=[
        "empty",
        "defaults_t_for_mean_bootstrap_for_median",
        "override_mean_ci_to_bootstrap",
        "nans_present",
        "title_with_modifiers",
        "labels_source_and_save",
        "deterministic_small_set",
        "empty_with_inferential_params",
        "combined_title_labels_source",
        "save_mean_ci_t",
        "save_mean_ci_bootstrap",
        "save_median_ci_bootstrap",
        "save_popmean_tests",
        "save_popmedian_tests",
    ],
)
def test_central_tendency_violin_plot_data_driven(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    # If a file_name is provided, also set save_path to tmp_path
    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["save_path"] = tmp_path

    ctx = CentralTendencyViolinContext(**kwargs)
    plot = CentralTendencyViolinPlot(ctx)

    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
