import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric import DistributionECDFGapContext, DistributionECDFGapPlot

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
    s = series_factory()
    with pytest.raises(expected_exc, match=match):
        ctx = DistributionECDFGapContext()
        plot = DistributionECDFGapPlot(ctx)

        plot.run(s)


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) EMPTY → safe defaults; params echoes threshold=None
        (
            lambda: pd.Series([], dtype="float64", name="nums"),
            {},
            {
                "chart_metadata": {
                    "title": "ECDF Gap Analysis of nums",
                    "xlabel": "Value",
                    "ylabel": "ECDF",
                    "data_source": None,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "params": {"threshold": None},
                    "n": 0,
                    "n_unique": 0,
                    "gaps": [],
                    "max_gap": (lambda v: np.isnan(v)),
                    "median_gap": (lambda v: np.isnan(v)),
                    "pct10_gap": (lambda v: np.isnan(v)),
                    "pct50_gap": (lambda v: np.isnan(v)),
                    "pct90_gap": (lambda v: np.isnan(v)),
                    "n_gaps_above_thr": None,
                    "total_gap_prop": (lambda v: np.isnan(v)),
                    "max_gap_loc": (lambda v: np.isnan(v)),
                },
            },
        ),

        # 1) Simple increasing ints → verify n/n_unique and a few gap facts
        (
            lambda: pd.Series([1, 2, 4, 7], name="g"),
            {},
            {
                "chart_metadata": {
                    "title": "ECDF Gap Analysis of g",
                    "xlabel": "Value",
                    "ylabel": "ECDF",
                },
                "descriptive_stats": {
                    "n": 4,
                    "n_unique": 4,
                    "gaps": (lambda v: isinstance(v, list) and len(v) == 3),
                    "max_gap": 3.0,          # (4→7)
                    "max_gap_loc": 5.5,      # midpoint of (4,7)
                    # don't pin percentiles/median; small-sample interpolation varies
                },
            },
        ),

        # 2) NaNs present → cleaning affects n/n_unique
        (
            lambda: pd.Series([1.0, np.nan, 2.0, np.nan, 5.0, 5.0], name="with_nans"),
            {},
            {
                "chart_metadata": {
                    "title": "ECDF Gap Analysis of with_nans",
                },
                "descriptive_stats": {
                    "n": 4,          # [1,2,5,5]
                    "n_unique": 3,   # [1,2,5]
                },
            },
        ),

        # 3) Threshold counting → n_gaps_above_thr computed on unique-value gaps
        #    unique: [0,1,4,10] → gaps [1,3,6], threshold=2 → 2 gaps > 2
        (
            lambda: pd.Series([0, 1, 4, 10], name="thr"),
            {"threshold": 2.0},
            {
                "descriptive_stats": {
                    "params": {"threshold": 2.0},
                    "n": 4,
                    "n_unique": 4,
                    "n_gaps_above_thr": 2,
                },
            },
        ),

        # 4) Title building with name override + modifiers
        (
            lambda: pd.Series([10, 20, 30], name="ignored"),
            {"name": "Price", "filter_desc": "NY", "transform_desc": "scaled"},
            {
                "chart_metadata": {
                    "title": "ECDF Gap Analysis of Price (NY, scaled)",
                },
                "descriptive_stats": {"n": 3, "n_unique": 3},
            },
        ),

        # 5) Label/source overrides + explicit save filename
        (
            lambda: pd.Series([2, 3, 5, 8, 13], name="fib"),
            {"xlabel": "Score", "ylabel": "ECDF%", "data_source": "UnitTest", "file_name": "ecdf_gap.png"},
            {
                "chart_metadata": {
                    "xlabel": "Score",
                    "ylabel": "ECDF%",
                    "data_source": "UnitTest",
                    "file_name": "ecdf_gap.png",
                },
                "descriptive_stats": {"n": 5, "n_unique": 5},
            },
        ),

        # 6) All values identical (single unique) → gaps empty; n_gaps_above_thr = 0 when threshold provided
        (
            lambda: pd.Series([5, 5, 5, 5], name="const"),
            {"threshold": 1.0},
            {
                "descriptive_stats": {
                    "params": {"threshold": 1.0},
                    "n": 4,
                    "n_unique": 1,
                    "gaps": [],
                    "max_gap": (lambda v: np.isnan(v)),
                    "median_gap": (lambda v: np.isnan(v)),
                    "n_gaps_above_thr": 0,
                    "total_gap_prop": (lambda v: np.isnan(v)),
                    "max_gap_loc": (lambda v: np.isnan(v)),
                },
            },
        ),

        # 7) Known largest gap & location with half-steps
        #    unique: [0, 0.5, 5, 5.5] → gaps [0.5, 4.5, 0.5]; max=4.5 at (0.5,5) → loc=2.75
        (
            lambda: pd.Series([0, 0.5, 5, 5.5], name="loc"),
            {},
            {
                "descriptive_stats": {
                    "n": 4,
                    "n_unique": 4,
                    "max_gap": 4.5,
                    "max_gap_loc": 2.75,
                },
            },
        ),

        # 8) Threshold present but no gap exceeds it → count is 0
        (
            lambda: pd.Series([0, 1, 2, 3], name="tight"),
            {"threshold": 2.5},
            {
                "descriptive_stats": {
                    "params": {"threshold": 2.5},
                    "n": 4,
                    "n_unique": 4,
                    "n_gaps_above_thr": 0,  # gaps are [1,1,1]
                },
            },
        ),

        # 9) Save with defaults (no label overrides); ensure metadata echoes filename
        (
            lambda: pd.Series([1, 2, 2, 3, 5], name="s"),
            {"file_name": "ecdf.png"},
            {
                "chart_metadata": {
                    "title": "ECDF Gap Analysis of s",
                    "xlabel": "Value",
                    "ylabel": "ECDF",
                    "data_source": None,
                    "file_name": "ecdf.png",
                },
                "descriptive_stats": {"n": 5},
            },
        ),

        # 10) total_gap_prop is a valid fraction when at least 2 uniques
        (
            lambda: pd.Series([0, 1, 3, 6, 10], name="prop"),
            {},
            {
                "descriptive_stats": {
                    "n_unique": 5,
                    "total_gap_prop": (lambda v: isinstance(v, float) and 0.0 <= v <= 1.0),
                },
            },
        ),
    ],
    ids=[
        "0_empty",
        "1_simple_increasing_ints",
        "2_nans_cleaning",
        "3_threshold_counting",
        "4_title_with_modifiers",
        "5_labels_source_and_save",
        "6_single_unique_with_threshold",
        "7_known_max_gap_location",
        "8_threshold_no_exceeders",
        "9_save_with_filename",
        "10_total_gap_prop_fraction",
    ],
)
def test_plot_distribution_ecdf_gap_param(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    # If a file_name is provided, also set save_path to tmp_path
    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["save_path"] = tmp_path

    ctx = DistributionECDFGapContext(**kwargs)
    plot = DistributionECDFGapPlot(ctx)

    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
