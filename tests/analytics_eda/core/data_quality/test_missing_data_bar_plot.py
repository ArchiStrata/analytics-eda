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
                    "bars": {},
                    "subset_count": 0,
                    "total": 0,
                    "total_nonnull": 0
                },
                "draft_descriptive_findings": {
                    "context": "0 values",
                    "primary_finding": "The series is empty.",
                    "secondary_finding": None
                },
                "draft_inferential_findings": {},
                "inferential_stats": {}
            },
        ),

        # 1) No missing values
        (
            lambda: pd.Series([1, 2, 3], dtype="float64", name="no_nans"),
            {"file_name": "missing.png"},
            {
                "chart_metadata": {
                    "data_source": None,
                    "file_name": "missing.png",
                    "title": "Missing Data for no_nans",
                    "version": "1.0.0",
                    "xlabel": "Status",
                    "ylabel": "Percentage of Total"
                },
                "descriptive_stats": {
                    "bars": {
                        "Missing": {
                            "count": 0,
                            "pct_of_total": 0.0
                        },
                        "Present": {
                            "count": 3,
                            "pct_of_total": 1.0
                        }
                    },
                    "denominator_key": "pct_of_total",
                    "input_categories": 2,
                    "input_nonzero_categories": 1,
                    "n_bars_rendered": 2,
                    "nonzero_categories": 1,
                    "params": {
                        "bar_height_source": "values",
                        "bar_sort_descending": True,
                        "max_display_bars": 15,
                        "other_label": "Other",
                        "show_count_in_bar_label": False,
                        "show_value_in_bar_label": True
                    },
                    "pct_subset": 1.0,
                    "subset_count": 3,
                    "total": 3,
                    "total_nonnull": 3,
                    "unique_categories_total": 3
                },
                "draft_descriptive_findings": {
                    "context": "3 values",
                    "primary_finding": "0.0% of 3 values are missing.",
                    "secondary_finding": "No missing values detected."
                },
                "draft_inferential_findings": {},
                "inferential_stats": {}
            }
        ),

        # 2) Some missing values (counts use full length including NaNs)
        (
            lambda: pd.Series([1, np.nan, 2, np.nan], name="some_nans"),
            {"file_name": "missing.png"},
            {
                "chart_metadata": {
                    "data_source": None,
                    "file_name": "missing.png",
                    "title": "Missing Data for some_nans",
                    "version": "1.0.0",
                    "xlabel": "Status",
                    "ylabel": "Percentage of Total"
                },
                "descriptive_stats": {
                    "bars": {
                        "Missing": {
                            "count": 2,
                            "pct_of_total": 0.5
                        },
                        "Present": {
                            "count": 2,
                            "pct_of_total": 0.5
                        }
                    },
                    "denominator_key": "pct_of_total",
                    "input_categories": 2,
                    "input_nonzero_categories": 2,
                    "n_bars_rendered": 2,
                    "nonzero_categories": 2,
                    "params": {
                        "bar_height_source": "values",
                        "bar_sort_descending": True,
                        "max_display_bars": 15,
                        "other_label": "Other",
                        "show_count_in_bar_label": False,
                        "show_value_in_bar_label": True
                    },
                    "pct_subset": 1.0,
                    "subset_count": 4,
                    "total": 4,
                    "total_nonnull": 2,
                    "unique_categories_total": 2
                },
                "draft_descriptive_findings": {
                    "context": "4 values",
                    "primary_finding": "50.0% of 4 values are missing.",
                    "secondary_finding": "Missing and present values are evenly split."
                },
                "draft_inferential_findings": {},
                "inferential_stats": {}
            },
        ),

        # 3) All missing values
        (
            lambda: pd.Series([np.nan, np.nan, np.nan], name="all_nans"),
            {"file_name": "missing.png", "show_count_in_bar_label": True},
            {
                "chart_metadata": {
                    "data_source": None,
                    "file_name": "missing.png",
                    "title": "Missing Data for all_nans",
                    "version": "1.0.0",
                    "xlabel": "Status",
                    "ylabel": "Percentage of Total"
                },
                "descriptive_stats": {
                    "bars": {
                        "Missing": {
                            "count": 3,
                            "pct_of_total": 1.0
                        },
                        "Present": {
                            "count": 0,
                            "pct_of_total": 0.0
                        }
                    },
                    "denominator_key": "pct_of_total",
                    "input_categories": 2,
                    "input_nonzero_categories": 1,
                    "n_bars_rendered": 2,
                    "nonzero_categories": 1,
                    "params": {
                        "bar_height_source": "values",
                        "bar_sort_descending": True,
                        "max_display_bars": 15,
                        "other_label": "Other",
                        "show_count_in_bar_label": True,
                        "show_value_in_bar_label": True
                    },
                    "pct_subset": 1.0,
                    "subset_count": 3,
                    "total": 3,
                    "total_nonnull": 0,
                    "unique_categories_total": 0
                },
                "draft_descriptive_findings": {
                    "context": "3 values",
                    "primary_finding": "100.0% of 3 values are missing.",
                    "secondary_finding": "All values are missing."
                },
                "draft_inferential_findings": {},
                "inferential_stats": {}
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
