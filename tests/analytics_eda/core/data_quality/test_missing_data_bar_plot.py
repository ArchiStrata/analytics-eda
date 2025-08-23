import pytest
import pandas as pd
import numpy as np

from analytics_eda.core.data_quality import MissingDataBarContext, MissingDataBarPlot


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) Empty series → zeros across stats; default labels and title
        (
            lambda: pd.Series([], dtype="float64", name="empty_col"),
            {},
            {
                "chart_metadata": {
                    "title": "Missing Data for empty_col",
                    "xlabel": "Status",
                    "ylabel": "Percentage of Total",
                    "data_source": None,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "total": 0,
                    "missing": 0,
                    "pct_missing": 0.0,
                },
            },
        ),

        # 1) No missing values
        (
            lambda: pd.Series([1, 2, 3], dtype="float64", name="no_nans"),
            {},
            {
                "descriptive_stats": {
                    "total": 3,
                    "missing": 0,
                    "pct_missing": 0.0,
                },
            },
        ),

        # 2) Some missing values (counts use full length including NaNs)
        (
            lambda: pd.Series([1, np.nan, 2, np.nan], name="some_nans"),
            {},
            {
                "descriptive_stats": {
                    "total": 4,
                    "missing": 2,
                    "pct_missing": 0.5,
                },
            },
        ),

        # 3) All missing values
        (
            lambda: pd.Series([np.nan, np.nan, np.nan], name="all_nans"),
            {},
            {
                "descriptive_stats": {
                    "total": 3,
                    "missing": 3,
                    "pct_missing": 1.0,
                },
            },
        ),

        # 4) Title uses name override via context kwargs
        (
            lambda: pd.Series([1, np.nan, 2], name="ignored"),
            {"name": "Revenue"},
            {
                "chart_metadata": {
                    "title": "Missing Data for Revenue",
                },
            },
        ),

        # 5) Axis labels + data_source overrides
        (
            lambda: pd.Series([1, 2, np.nan], name="labeled"),
            {"xlabel": "Status", "ylabel": "Percent", "data_source": "UnitTest"},
            {
                "chart_metadata": {
                    "xlabel": "Status",
                    "ylabel": "Percent",
                    "data_source": "UnitTest",
                },
            },
        ),

        # 6) Save with explicit filename, verify PNG signature
        (
            lambda: pd.Series([1, np.nan, 2, 3, np.nan], name="save_me"),
            {"file_name": "missing.png"},
            {
                "chart_metadata": {
                    "file_name": "missing.png",
                    "title": "Missing Data for save_me",
                    "xlabel": "Status",
                    "ylabel": "Percentage of Total",
                    "data_source": None,
                },
                "descriptive_stats": {
                    "total": 5,
                    "missing": 2,
                    "pct_missing": 2 / 5,
                },
            },
        ),
    ],
    ids=[
        "empty",
        "no_missing",
        "some_missing",
        "all_missing",
        "title_name_override",
        "labels_and_source_override",
        "save_with_png_signature",
    ],
)
def test_plot_missing_data_barchart_param(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    # If a file_name is provided, also set save_path to tmp_path
    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["save_path"] = tmp_path

    ctx = MissingDataBarContext(**kwargs)
    plot = MissingDataBarPlot(ctx)

    payload = plot.run(s)
    
    # Reuse shared helper to assert only the fields specified in `expect`
    assert_plot_metadata(payload, expect, tmp_path)
