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
                    "total": 0,
                    "total_nonnull": 0,
                    "bars": {},
                },
                "inferential_stats": {},
                "draft_inferential_findings": {},
                "draft_descriptive_findings": {},
            },
        ),

        # 1) No issues at all → skip (even if filename set later in another case)
        #    Clean labels, consistent casing, allowed chars, in-allowlist; one NA.
        (
            lambda: pd.Series(["Apple", "Banana", "Pear", None], name="clean"),
            {"allowed_categories": ["Apple", "Banana", "Pear"], "case_sensitive_allowed": True},
            {
                "chart_metadata": {"file_name": None, "version": "1.0.0"},
                "descriptive_stats": {
                    "total": 4,
                    "total_nonnull": 3,
                    "bars": {},
                },
                "inferential_stats": {},
                "draft_inferential_findings": {},
                "draft_descriptive_findings": lambda d: isinstance(d, dict) and "summary" in d and "No cleanliness issues" in d["summary"],
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
                    "total": 11,
                    "total_nonnull": 10,
                    "bars": {
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
                        },
                        "Invalid Category": {
                            "count": 2,
                            "pct_of_nonnull": 0.2
                        }
                    }
                },
                "inferential_stats": {},
                "draft_inferential_findings": {},
                # Allow either top order due to tie; require both top issues appear
                "draft_descriptive_findings": lambda d: (
                    isinstance(d, dict)
                    and "summary" in d
                    and "coverage" in d
                    and (
                    ("Invalid Category" in d.get("summary","") and "Mixed Casing" in d.get("secondary",""))
                    or ("Mixed Casing" in d.get("summary","") and "Invalid Category" in d.get("secondary",""))
                    )
                ),
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
