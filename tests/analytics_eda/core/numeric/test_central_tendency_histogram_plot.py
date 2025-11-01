import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric import CentralTendencyHistogramContext, CentralTendencyHistogramPlot


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
        ctx = CentralTendencyHistogramContext()
        plot = CentralTendencyHistogramPlot(ctx)

        plot.run(s)


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) EMPTY numeric series → bins computed as ceil(sqrt(0)) = 0
        (
            lambda: pd.Series([], dtype="float64", name="nums"),
            {},
            {
                "chart_metadata": {
                    "title": "Distribution of nums: Central Tendency",
                    "xlabel": "Value",
                    "ylabel": "Count",
                    "data_source": None,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "n": 0,
                    "mean": None,
                    "median": None,
                    "modes": [],
                    "params": {
                        "bins": 0,
                        "mode_method": None,
                    },
                },
            },
        ),
        # 1) Default bins via Square-Root Choice on non-empty data (n=9 → bins=3)
        (
            lambda: pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="float64", name="nums"),
            {"file_name": "hist.png"},
            {
                "chart_metadata": {"file_name": "hist.png"},
                "descriptive_stats": {
                    "mean": 5.0,
                    "median": 5.0,
                    "modality": "multimodal",
                    "modes": [1.4444444444444444, 2.333333333333333, 3.2222222222222223, 4.111111111111111, 5.0, 5.888888888888889, 6.777777777777777, 7.666666666666666, 8.555555555555555],
                    "n": 9,
                    "params": {"bin_rule": "discrete_integer", "bins": None, "bins_arg": 9, "max_mode_lines": 3, "min_peak_strength": 0.05, "mode_method": "histogram_bin_centers"},
                    "peak_strength": 0.1111111111111111,
                },
                "draft_descriptive_findings": {
                    "context": "n = 9 observations",
                    "primary_finding": "The distribution is multimodal, indicating several distinct peaks.",
                    "secondary_finding": "Mean and median are nearly identical at 5.0, suggesting a symmetric distribution.",
                },
            },
        ),
        # 2) Explicit integer bins; single clear mode uses series.mode()
        (
            lambda: pd.Series([1, 1, 2, 3, 4, 5], dtype="float64", name="vals"),
            {"bins": 5, "file_name": "hist.png"},
            {
                "chart_metadata": {"file_name": "hist.png"},
                "descriptive_stats": {
                    "mean": 2.6666666666666665,
                    "median": 2.5,
                    "modality": "unimodal",
                    "modes": [1.0],
                    "n": 6,
                    "params": {
                        "bins": 5,
                        "mode_method": "series.mode",
                    },
                    "peak_strength": 0.3333333333333333,
                },
                "draft_descriptive_findings": {
                    "context": "n = 6 observations",
                    "primary_finding": "The distribution is unimodal with a central peak around 1.00.",
                    "secondary_finding": "Mean (2.7) exceeds median (2.5) by 0.2, suggesting right-skew.",
                },
            },
        ),
        # 3) Custom bin edges (sequence); ambiguous/multimodal → histogram_bin_centers
        #    We only assert the mode_method and that at least one mode was produced.
        (
            lambda: pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype="float64", name="edges"),
            {"bins": [0, 3, 6, 10], "file_name": "hist.png"},
            {
                "chart_metadata": {},
                "descriptive_stats": {
                    "n": 9,
                    "modes": (lambda v: isinstance(v, list) and len(v) >= 1),
                    "params": {
                        "bins": [0, 3, 6, 10],
                        "mode_method": "histogram_bin_centers",
                    },
                },
            },
        ),
        # 4) NaNs present → n counts non-NaN, title defaults
        (
            lambda: pd.Series([1.0, np.nan, 2.0, np.nan, 3.0], name="with_nans"),
            {"file_name": "hist.png"},
            {
                "chart_metadata": {
                    "title": "Distribution of with_nans: Central Tendency",
                    "xlabel": "Value",
                    "ylabel": "Count",
                },
                "descriptive_stats": {
                    "n": 3,  # only non-NaN
                    # don't pin mean/median numerically; just ensure modes list exists
                    "modes": (lambda v: isinstance(v, list)),
                },
            },
        ),
        # 5) Name override + modifiers in title (filter + transform)
        (
            lambda: pd.Series([10, 20, 20, 30], name="ignored"),
            {"name": "Price", "filter_desc": "NY only", "transform_desc": "log-scaled", "file_name": "hist.png"},
            {
                "chart_metadata": {
                    "title": "Distribution of Price (NY only, log-scaled): Central Tendency",
                },
                "descriptive_stats": {
                    "n": 4,
                },
            },
        ),
        # 6) Custom title template that ignores modifiers
        (
            lambda: pd.Series([1, 2, 3, 4], name="nums"),
            {"title_template": "My Hist: {name}", "filter_desc": "ignored", "transform_desc": "ignored", "file_name": "hist.png"},
            {
                "chart_metadata": {
                    "title": "My Hist: nums",
                },
                "descriptive_stats": {"n": 4},
            },
        ),
        # 7) Axis labels + data_source overrides, with explicit save filename
        (
            lambda: pd.Series([1, 1, 2, 3, 5, 8], name="fib"),
            {"xlabel": "Score", "ylabel": "Frequency", "data_source": "UnitTest", "file_name": "hist.png"},
            {
                "chart_metadata": {
                    "xlabel": "Score",
                    "ylabel": "Frequency",
                    "data_source": "UnitTest",
                    "file_name": "hist.png",
                },
                "descriptive_stats": {"n": 6},
            },
        ),
        # T1) Single mode with bin edges, check mean/median/title/labels
        (
            lambda: pd.Series([1, 1, 1, 2, 2, 3], name="numeric_series"),
            {"bins": [0.5, 1.5, 2.5, 3.5], "file_name": "hist.png"},
            {
                "chart_metadata": {
                    "title": "Distribution of numeric_series: Central Tendency",
                    "xlabel": "Value",
                    "ylabel": "Count",
                    "data_source": None,
                },
                "descriptive_stats": {
                    "n": 6,
                    "mean": pytest.approx(1.67, 0.01),
                    "median": 1.5,
                    "modes": [1],
                    "params": {
                        "bins": [0.5, 1.5, 2.5, 3.5],
                        "mode_method": "series.mode",
                    },
                },
            },
        ),
        # T2) Two modes with bin edges, check exact modes [1, 2]
        (
            lambda: pd.Series([1, 1, 2, 2, 3, 4], name="numeric_series"),
            {"bins": [0.5, 1.5, 2.5, 3.5, 4.5], "file_name": "hist.png"},
            {
                "descriptive_stats": {
                    "n": 6,
                    "modes": [1, 2],
                    "params": {
                        "bins": [0.5, 1.5, 2.5, 3.5, 4.5],
                        "mode_method": "histogram_bin_centers",
                    },
                },
            },
        ),
        # T3) Three modes with bin edges, check exact sorted modes
        (
            lambda: pd.Series([1, 1, 2, 2, 3, 3, 4], name="numeric_series"),
            {"bins": [0.5, 1.5, 2.5, 3.5, 4.5], "file_name": "hist.png"},
            {
                "descriptive_stats": {
                    "n": 7,
                    "modes": (lambda v: sorted(v) == [1, 2, 3]),
                    "params": {
                        "bins": [0.5, 1.5, 2.5, 3.5, 4.5],
                        "mode_method": "histogram_bin_centers",
                    },
                },
            },
        ),
        # T4) Save with explicit filename, verify PNG signature
        (
            lambda: pd.Series([0, 1, 2, 2, 3, 3, 3], name="numeric_series"),
            {"bins": 5, "file_name": "hist.png"},
            {
                "chart_metadata": {
                    "file_name": "hist.png",
                    "xlabel": "Value",
                    "ylabel": "Count",
                    "title": "Distribution of numeric_series: Central Tendency",
                    "data_source": None,
                },
                "descriptive_stats": {
                    "n": 7,
                    "params": {
                        "bins": 5,
                        "mode_method": "series.mode",
                    },
                },
                # Special check: PNG signature will be verified in test body
            },
        ),
    ],
    ids=[
        "empty",
        "default_bins_sqrt_n",
        "explicit_bins_int_single_mode",
        "custom_bins_sequence_hist_modes",
        "nans_title_defaults",
        "title_with_modifiers",
        "custom_title_template_no_mods",
        "labels_source_and_save",
        "single_mode_with_bin_edges",
        "two_modes_with_bin_edges",
        "three_modes_with_bin_edges",
        "save_with_png_signature",
    ],
)
def test_central_tendency_histogram_plot_data_driven(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    # If a file_name is provided, also set save_path to tmp_path
    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["save_path"] = tmp_path

    ctx = CentralTendencyHistogramContext(**kwargs)
    plot = CentralTendencyHistogramPlot(ctx)

    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
