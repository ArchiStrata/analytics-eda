import pytest
import pandas as pd

from analytics_eda.core.data_quality import (
    CategoricalCleanlinessBarContext,
    CategoricalCleanlinessBarPlot,
)

@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) Empty series → no non-null, skip; defaults for labels/axes/title
        (
            lambda: pd.Series([], dtype="float64", name="empty_cat"),
            {},
            {
                "chart_metadata": {
                    "title": "Categorical Cleanliness for empty_cat",
                    "xlabel": "Percent of non‑null",
                    "ylabel": "Issue Type",
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
                "inferential_stats": {},
                "draft_inferential_findings": {},
                "draft_descriptive_findings": {
                    "context": "N (non‑null) = 0",
                    "primary_finding": "The series is empty.",
                    "secondary_finding": None
                },
            },
        ),

        # 1) No issues at all → skip (even if filename set later in another case)
        #    Clean labels, consistent casing, allowed chars, in-allowlist; one NA.
        (
            lambda: pd.Series(["Apple", "Banana", "Pear", None], name="clean"),
            {"allowed_categories": ["Apple", "Banana", "Pear"], "case_sensitive_allowed": True},
            {
                "chart_metadata": {
                    "title": "Categorical Cleanliness for clean",
                    "file_name": None,
                    "version": "1.0.0",
                    "xlabel": "Percent of non‑null",
                    "ylabel": "Issue Type"
                },
                "descriptive_stats": {
                    "bars": {
                        "Invalid Category": {
                            "count": 0,
                            "pct_of_nonnull": 0.0
                        },
                        "Leading/Trailing Whitespace": {
                            "count": 0,
                            "pct_of_nonnull": 0.0
                        },
                        "Mixed Casing": {
                            "count": 0,
                            "pct_of_nonnull": 0.0
                        },
                        "Non-Standard Characters": {
                            "count": 0,
                            "pct_of_nonnull": 0.0
                        }
                    },
                    "denominator_key": "pct_of_nonnull",
                    "error": "no categories to display",
                    "input_categories": 4,
                    "input_nonzero_categories": 0,
                    "n_bars_rendered": 4,
                    "nonzero_categories": 0,
                    "params": {
                        "allowed_categories_count": 3,
                        "allowed_char_pattern": "^[\\w\\s\\-\\_/.,&()']*$",
                        "bar_height_source": "values",
                        "bar_sort_descending": False,
                        "case_sensitive_allowed": True,
                        "max_display_bars": 15,
                        "other_label": "Other",
                        "show_count_in_bar_label": False,
                        "show_value_in_bar_label": True,
                        "treat_empty_as_invalid": True
                    },
                    "pct_subset": 0.0,
                    "skip_plot": True,
                    "subset_count": 0,
                    "total": 4,
                    "total_nonnull": 3,
                    "unique_categories_total": 3
                },
                "inferential_stats": {},
                "draft_inferential_findings": {},
                "draft_descriptive_findings": {
                    "context": "N (non‑null) = 3",
                    "primary_finding": "No cleanliness issues detected.",
                    "secondary_finding": None
                },
            },
        ),

        # 2) Mixed real-world issues (whitespace, mixed casing, nonstandard chars, invalids)
        #    Series (non-null base = 10, total = 11):
        #      " Apple" [ws+mc], "apple", "apple", "Banana" [mc], "banana", "banana",
        #      "Pear","Pear", "ban@na" [nonstd + invalid], "" [invalid], None
        (
            lambda: pd.Series(
                [" Apple", "apple", "apple", "Banana", "banana", "banana",
                 "Pear", "Pear", "ban@na", "", None],
                name="messy"
            ),
            {
                "file_name": "cleanliness.png",
                "allowed_categories": ["Apple", "Banana", "Pear"],  # case-insensitive by default
                # keep default allowed_char_pattern
                "show_count_in_bar_label": True,
            },
            {
                "chart_metadata": {
                    "file_name": "cleanliness.png",
                    "title": "Categorical Cleanliness for messy",
                },
                "descriptive_stats": {
                    "bars": {
                        "Invalid Category": {
                            "count": 2,
                            "pct_of_nonnull": 0.2
                        },
                        "Leading/Trailing Whitespace": {
                            "count": 1,
                            "pct_of_nonnull": 0.1
                        },
                        "Mixed Casing": {
                            "count": 2,
                            "pct_of_nonnull": 0.2
                        },
                        "Non-Standard Characters": {
                            "count": 1,
                            "pct_of_nonnull": 0.1
                        }
                    },
                    "denominator_key": "pct_of_nonnull",
                    "input_categories": 4,
                    "input_nonzero_categories": 4,
                    "n_bars_rendered": 4,
                    "nonzero_categories": 4,
                    "params": {
                        "allowed_categories_count": 3,
                        "allowed_char_pattern": "^[\\w\\s\\-\\_/.,&()']*$",
                        "bar_height_source": "values",
                        "bar_sort_descending": False,
                        "case_sensitive_allowed": False,
                        "max_display_bars": 15,
                        "other_label": "Other",
                        "show_count_in_bar_label": True,
                        "show_value_in_bar_label": True,
                        "treat_empty_as_invalid": True
                    },
                    "pct_subset": 0.6,
                    "subset_count": 6,
                    "total": 11,
                    "total_nonnull": 10,
                    "unique_categories_total": 7
                },
                "inferential_stats": {},
                "draft_inferential_findings": {},
                "draft_descriptive_findings": {
                    "context": "N (non‑null) = 10",
                    "primary_finding": "60.0% of values show at least one cleanliness issue (6 rows).",
                    "secondary_finding": "Most frequent issue: Mixed Casing at 20.0% (2 rows)."
                },
            },
        ),

        # 3) Custom title via context name override; axis labels + data_source overrides
        (
            lambda: pd.Series([" a ", "A", "a", "ok"], name="ignored"),
            {"file_name": "cleanliness.png", "name": "Customers", "xlabel": "Rows", "ylabel": "Issue", "data_source": "UnitTest"},
            {
                "chart_metadata": {
                    "file_name": "cleanliness.png",
                    "title": "Categorical Cleanliness for Customers",
                    "xlabel": "Rows",
                    "ylabel": "Issue",
                    "data_source": "UnitTest",
                    "version": "1.0.0",
                },
                "descriptive_stats": {
                    "bars": {
                        "Invalid Category": {
                            "count": 0,
                            "pct_of_nonnull": 0.0
                        },
                        "Leading/Trailing Whitespace": {
                            "count": 1,
                            "pct_of_nonnull": 0.25
                        },
                        "Mixed Casing": {
                            "count": 1,
                            "pct_of_nonnull": 0.25
                        },
                        "Non-Standard Characters": {
                            "count": 0,
                            "pct_of_nonnull": 0.0
                        }
                    },
                    "denominator_key": "pct_of_nonnull",
                    "input_categories": 4,
                    "input_nonzero_categories": 2,
                    "n_bars_rendered": 4,
                    "nonzero_categories": 2,
                    "params": {
                        "allowed_categories_count": 0,
                        "allowed_char_pattern": "^[\\w\\s\\-\\_/.,&()']*$",
                        "bar_height_source": "values",
                        "bar_sort_descending": False,
                        "case_sensitive_allowed": False,
                        "max_display_bars": 15,
                        "other_label": "Other",
                        "show_count_in_bar_label": False,
                        "show_value_in_bar_label": True,
                        "treat_empty_as_invalid": True
                    },
                    "pct_subset": 0.5,
                    "subset_count": 2,
                    "total": 4,
                    "total_nonnull": 4,
                    "unique_categories_total": 4
                },
                "draft_descriptive_findings": {
                    "context": "N (non‑null) = 4",
                    "primary_finding": "50.0% of values show at least one cleanliness issue (2 rows).",
                    "secondary_finding": "Most frequent issue: Leading/Trailing Whitespace at 25.0% (1 rows)."
                },
                "draft_inferential_findings": {},
                "inferential_stats": {}
            },
        ),
    ],
    ids=[
        "empty_skips",
        "no_issues_skips",
        "mixed_issues",
        "labels_and_source_override",
    ],
)
def test_categorical_cleanliness_bar_plot_data_driven(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    # If a file_name is provided, also set save_path to tmp_path
    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["save_path"] = tmp_path

    ctx = CategoricalCleanlinessBarContext(**kwargs)
    plot = CategoricalCleanlinessBarPlot(ctx)

    payload = plot.run(s)

    # Reuse shared helper to assert only the fields specified in `expect`
    assert_plot_metadata(payload, expect, tmp_path)
