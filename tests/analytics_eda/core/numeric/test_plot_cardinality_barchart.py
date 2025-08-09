import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric import plot_cardinality_barchart

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
def test_validate_categorical_named_series_errors(series_factory, expected_exc, match):
    obj = series_factory()
    with pytest.raises(expected_exc, match=match):
        plot_cardinality_barchart(obj)


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) EMPTY numeric series
        (
            lambda: pd.Series([], dtype="float64", name="nums"),
            {},
            {
                "chart_metadata": {
                    "title": "Value Counts (Top 10) of nums for Cardinality",
                    "xlabel": "Value",
                    "ylabel": "Count",
                    "top_k": 10,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "total": 0,
                    "nunique": 0,
                    "uniqueness_ratio": 0.0,
                    "is_discrete": None,
                    "params": {
                        "max_unique_fraction": 0.05,
                        "max_unique_values": 20,
                        "integer_tolerance": 1e-8,
                    },
                },
            },
        ),
        # 1) Low-cardinality integer data is discrete; include data_source
        (
            lambda: pd.Series([1, 1, 2, 2, 2, 3], name="ids", dtype="int64"),
            {"data_source": "UnitTest"},
            {
                "chart_metadata": {
                    "data_source": "UnitTest",
                    "xlabel": "Value",
                    "ylabel": "Count",
                    "top_k": 10,
                },
                "descriptive_stats": {
                    "total": 6,
                    "nunique": 3,
                    "is_discrete": True,
                },
            },
        ),
        # 2) Float values that are whole numbers (within tolerance) => discrete
        (
            lambda: pd.Series([1.0, 2.0, 2.0, 3.0], name="vals", dtype="float64"),
            {},
            {
                "descriptive_stats": {
                    "total": 4,
                    "nunique": 3,
                    "is_discrete": True,
                },
            },
        ),
        # 3) Many unique floats => continuous (not discrete)
        (
            lambda: pd.Series(np.linspace(0, 0.99, 100), name="x", dtype="float64"),
            {},
            {
                "descriptive_stats": {
                    "total": 100,
                    "nunique": 100,
                    "is_discrete": False,
                },
            },
        ),
        # 4) NaNs present; uniqueness_ratio uses len(series) (not len(clean))
        (
            lambda: pd.Series([1, 1, 2, np.nan, np.nan], name="with_nans"),
            {},
            {
                "descriptive_stats": {
                    "total": 3,  # clean size
                    "nunique": 2,
                    "uniqueness_ratio": 2 / 5,  # nunique / original length
                },
            },
        ),
        # 5) name override + custom top_k reflected in title
        (
            lambda: pd.Series([5, 5, 4, 4, 4, 3], name="ignored"),
            {"name": "Age", "top_k": 5},
            {
                "chart_metadata": {
                    "title": "Value Counts (Top 5) of Age for Cardinality",
                    "top_k": 5,
                },
            },
        ),
        # 6) Explicit file_name triggers save to tmp_path
        (
            lambda: pd.Series([1, 1, 2, 3, 3, 3], name="save_me"),
            {"file_name": "cardinality.png"},
            {
                "chart_metadata": {"file_name": "cardinality.png"},
                "descriptive_stats": {"total": 6},
            },
        ),
        # A) Non-empty, no kwargs, assert all defaults
        (
            lambda: pd.Series([1, 2, 2, 3, 4, 4, 5], name="nums"),
            {},
            {
                "chart_metadata": {
                    "title": "Value Counts (Top 10) of nums for Cardinality",
                    "xlabel": "Value",
                    "ylabel": "Count",
                    "data_source": None,
                    "top_k": 10,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "nunique": 5
                },
            },
        ),

        # B) Custom title_template + axis labels + data_source + save
        (
            lambda: pd.Series([10, 20, 20, 30, 30, 30], name="vals"),
            {
                "top_k": 3,
                "title_template": "Top 3 Frequencies",
                "xlabel": "Category",
                "ylabel": "Frequency",
                "data_source": "UnitTest",
                "file_name": "card.png",
            },
            {
                "chart_metadata": {
                    "title": "Top 3 Frequencies",
                    "xlabel": "Category",
                    "ylabel": "Frequency",
                    "data_source": "UnitTest",
                    "top_k": 3,
                    "file_name": "card.png",
                },
                "descriptive_stats": {
                    "nunique": 3
                },
            },
        ),

        # C) Override max_unique_fraction to 1.0 on high-cardinality → discrete
        (
            lambda: pd.Series(range(100), name="nums"),
            {
                "max_unique_fraction": 1.0,
                "file_name": "frac.png",
            },
            {
                "descriptive_stats": {
                    "is_discrete": True
                },
                "chart_metadata": {
                    "file_name": "frac.png"
                },
            },
        ),

        # D) Floats not integer-like, but unique < max_unique_values → discrete (path 2b)
        (
            lambda: pd.Series([i + 0.1 for i in range(10)], name="floats"),
            {
                "file_name": "low_card.png",
            },
            {
                "descriptive_stats": {
                    "is_discrete": True
                },
                "chart_metadata": {
                    "file_name": "low_card.png"
                },
            },
        ),

        # E) High-cardinality floats, not integer-like, tolerance flips discrete from False → True
        #    We'll check only the 'True' case here; the 'False' case is already covered in many_unique_floats_continuous.
        (
            lambda: pd.Series([i + 1e-6 for i in range(30)], name="floats"),
            {
                "integer_tolerance": 1e-5,
                "file_name": "flt_tol.png",
            },
            {
                "descriptive_stats": {
                    "is_discrete": True
                },
                "chart_metadata": {
                    "file_name": "flt_tol.png"
                },
            },
        ),
    ],
    ids=[
        "empty",
        "discrete_integers_with_source",
        "whole_like_floats_are_discrete",
        "many_unique_floats_continuous",
        "nans_affect_uniqueness_ratio_denominator",
        "name_override_and_topk_in_title",
        "explicit_filename_saves",
        "defaults_non_empty",
        "custom_title_and_labels_with_save",
        "override_max_unique_fraction_discrete",
        "float_low_cardinality_path_2b",
        "float_high_cardinality_tol_flip",
    ],
)
def test_plot_cardinality_barchart_param(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    # If a file_name is provided, also set save_path to tmp_path
    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["save_path"] = tmp_path

    payload = plot_cardinality_barchart(s, **kwargs)

    assert_plot_metadata(payload, expect, tmp_path)
