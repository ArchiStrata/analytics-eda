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
                "version": "1.0.0",
                "file_name": None,
            },
            "descriptive_stats": {
                "total": 0,
                "missing": 0,
                "pct_missing": 0.0,
            },
            "inferential_stats": {},
            "draft_inferential_findings": {},
            "draft_descriptive_findings": {},  # empty base ⇒ no finding
            },
        ),

        # 1) No missing values
        (
            lambda: pd.Series([1, 2, 3], dtype="float64", name="no_nans"),
            {"file_name": "missing.png"},
            {
            "descriptive_stats": {"total": 3, "missing": 0, "pct_missing": 0.0},
            "inferential_stats": {},
            "draft_inferential_findings": {},
            "draft_descriptive_findings": lambda d: "summary" in d and "0.0% missing" in d["summary"],
            "chart_metadata": {"file_name": "missing.png", "version": "1.0.0"},
            },
        ),

        # 2) Some missing values (counts use full length including NaNs)
        (
            lambda: pd.Series([1, np.nan, 2, np.nan], name="some_nans"),
            {"file_name": "missing.png"},
            {
            "descriptive_stats": {"total": 4, "missing": 2, "pct_missing": 0.5},
            "inferential_stats": {},
            "draft_inferential_findings": {},
            "draft_descriptive_findings": lambda d: "summary" in d and "50.0% missing" in d["summary"],
            "chart_metadata": {"file_name": "missing.png", "version": "1.0.0"},
            },
        ),

        # 3) All missing values
        (
            lambda: pd.Series([np.nan, np.nan, np.nan], name="all_nans"),
            {"file_name": "missing.png", "show_count_in_label": True},
            {
            "descriptive_stats": {"total": 3, "missing": 3, "pct_missing": 1.0},
            "inferential_stats": {},
            "draft_inferential_findings": {},
            "draft_descriptive_findings": lambda d: "summary" in d and "100.0% missing" in d["summary"],
            "chart_metadata": {"file_name": "missing.png", "version": "1.0.0"},
            },
        ),

        # 4) Title uses name override via context kwargs
        (
            lambda: pd.Series([1, np.nan, 2], name="ignored"),
            {"file_name": "missing.png", "name": "Revenue"},
            {
                "chart_metadata": {
                    "file_name": "missing.png",
                    "title": "Missing Data for Revenue",
                },
            },
        ),

        # 5) Axis labels + data_source overrides
        (
            lambda: pd.Series([1, 2, np.nan], name="labeled"),
            {"file_name": "missing.png", "xlabel": "Status", "ylabel": "Percent", "data_source": "UnitTest"},
            {
                "chart_metadata": {
                    "file_name": "missing.png",
                    "xlabel": "Status",
                    "ylabel": "Percent",
                    "data_source": "UnitTest",
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
    ],
)
def test_missing_data_bar_plot_data_driven(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
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
