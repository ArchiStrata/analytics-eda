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
                    "xlabel": "Count",
                    "ylabel": "Issue Type",
                    "data_source": None,
                    "file_name": None,   # skip means nothing saved
                },
                "descriptive_stats": {
                    "total": 0,
                    "total_nonnull": 0,
                    "issue_counts": [0, 0, 0, 0],
                    "issue_pcts": [0.0, 0.0, 0.0, 0.0],
                },
            },
        ),

        # 1) No issues at all → skip (even if filename set later in another case)
        #    Clean labels, consistent casing, allowed chars, in-allowlist; one NA.
        (
            lambda: pd.Series(["Apple", "Banana", "Pear", None], name="clean"),
            {"allowed_categories": ["Apple", "Banana", "Pear"], "case_sensitive_allowed": True},
            {
                "chart_metadata": {"file_name": None},
                "descriptive_stats": {
                    "total": 4,
                    "total_nonnull": 3,
                    "issue_counts": lambda xs: sum(xs) == 0,
                    "issue_pcts":   lambda xs: all(abs(v) < 1e-12 for v in xs),
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
                "allowed_categories": ["Apple", "Banana", "Pear"],  # case-insensitive by default
                # keep default allowed_char_pattern
            },
            {
                "chart_metadata": {
                    "title": "Categorical Cleanliness for messy",
                },
                "descriptive_stats": {
                    "total": 11,
                    "total_nonnull": 10,
                    # We assert via a dict of label -> count so ordering doesn't matter
                    "issue_labels": lambda labels: set(labels) == {
                        "Leading/Trailing Whitespace", "Mixed Casing",
                        "Non-Standard Characters", "Invalid Category"
                    },
                    "issue_counts": lambda counts: isinstance(counts, list) and len(counts) == 4
                },
            },
        ),

        # 3) Custom title via context name override; axis labels + data_source overrides
        (
            lambda: pd.Series([" a ", "A", "a", "ok"], name="ignored"),
            {"name": "Customers", "xlabel": "Rows", "ylabel": "Issue", "data_source": "UnitTest"},
            {
                "chart_metadata": {
                    "title": "Categorical Cleanliness for Customers",
                    "xlabel": "Rows",
                    "ylabel": "Issue",
                    "data_source": "UnitTest",
                },
            },
        ),

        # 4) Save with explicit filename (and at least one issue so it actually saves)
        (
            lambda: pd.Series(["ok", "OK", "ok "], name="save_me"),  # mixed casing + whitespace
            {"file_name": "cleanliness.png"},
            {
                "chart_metadata": {
                    "file_name": "cleanliness.png",
                    "title": "Categorical Cleanliness for save_me",
                    "xlabel": "Count",
                    "ylabel": "Issue Type",
                    "data_source": None,
                },
                "descriptive_stats": {
                    "total": 3,
                    "total_nonnull": 3,
                    # At least one issue should be nonzero (exact split depends on canonical choice);
                    # just ensure there is signal.
                    "issue_counts": lambda xs: sum(xs) >= 1,
                },
            },
        ),
    ],
    ids=[
        "empty_skips",
        "no_issues_skips",
        "mixed_issues",
        "labels_and_source_override",
        "save_with_png_signature",
    ],
)
def test_plot_categorical_cleanliness_bar_param(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
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
