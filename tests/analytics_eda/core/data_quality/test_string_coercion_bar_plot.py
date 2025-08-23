import pytest
import pandas as pd

from analytics_eda.core.data_quality import (
    StringCoercionBarContext,
    StringCoercionBarPlot,
)


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) Empty series → zeros across stats; default labels and title
        (
            lambda: pd.Series([], dtype="float64", name="empty_col"),
            {},
            {
            "chart_metadata": {
                "title": "Non-Numeric (String) Values in empty_col",
                "xlabel": "Percent of non‑null",
                "ylabel": "Category",
                "data_source": None,
                "version": "1.0.0",
                "file_name": None,
            },
            "descriptive_stats": {
                "total": 0,
                "total_nonnull": 0,
                "n_non_numeric": 0,
                "pct_non_numeric": 0.0,
                "k_non_numeric": 0,
                "category_labels": [],
                "category_counts": [],
            },
            "inferential_stats": {},
            "draft_inferential_findings": {},
            "draft_descriptive_findings": {},  # base=0 → {}
            },
        ),

        # 1) All numeric → no string-coercion issues
        (
            lambda: pd.Series([1, 2, 3], dtype="float64", name="all_numeric"),
            {"file_name": "strings.png"},
            {
            "descriptive_stats": {
                "total": 3,
                "total_nonnull": 3,
                "n_non_numeric": 0,
                "pct_non_numeric": 0,
                "k_non_numeric": 0,
                "category_labels": [],
                "category_counts": [],
            },
            "chart_metadata": {"file_name": None, "version": "1.0.0"},
            "inferential_stats": {},
            "draft_inferential_findings": {},
            "draft_descriptive_findings": lambda d: "summary" in d and "No non‑numeric tokens" in d["summary"],
            },
        ),

        # 2) Some non-numeric strings (numeric-like strings should not count as issues)
        #    Series: [1, 'x', 2, 'y', 'x'] → total(non-null)=5 → n_strings=3 ('x','y','x'), unique_strings=2
        (
            lambda: pd.Series([1, "x", 2, "y", "x"], name="mixed"),
            {"file_name": "strings.png", "show_count_in_label": True},
            {
            "descriptive_stats": {
                "total": 5,
                "total_nonnull": 5,
                "n_non_numeric": 3,
                "pct_non_numeric": 3/5,
                "k_non_numeric": 2,
                "category_labels": lambda xs: set(xs) == {"x", "y"},
                "category_counts": lambda xs: sorted(list(xs)) == [1, 2],
            },
            "chart_metadata": {"file_name": "strings.png", "version": "1.0.0"},
            "inferential_stats": {},
            "draft_inferential_findings": {},
            "draft_descriptive_findings": lambda d: "summary" in d and "60.0% of non‑null" in d["summary"],  # overall share
            },
        ),

        # 3) All strings but numeric-like → zero issues (coerces fine)
        (
            lambda: pd.Series(["1", "2", "3"], name="numeric_like_strings"),
            {"file_name": "strings.png"},
            {
            "descriptive_stats": {
                "total": 3, "total_nonnull": 3,
                "n_non_numeric": 0, "pct_non_numeric": 0, "k_non_numeric": 0
            },
            "chart_metadata": {"file_name": None, "version": "1.0.0"},
            "inferential_stats": {},
            "draft_inferential_findings": {},
            "draft_descriptive_findings": lambda d: "summary" in d and "No non‑numeric tokens" in d["summary"],
            },
        ),

        # 4) Custom title via context name override; axis labels + data_source overrides
        (
            lambda: pd.Series(["a", "b", "c", "bad", "bad"], name="ignored"),
            {"name": "Revenue", "xlabel": "String Value", "ylabel": "Frequency", "data_source": "UnitTest"},
            {
                "chart_metadata": {
                    "title": "Non-Numeric (String) Values in Revenue",
                    "xlabel": "String Value",
                    "ylabel": "Frequency",
                    "data_source": "UnitTest",
                },
            },
        ),

        # literal NA strings
        (
            lambda: pd.Series(["NaN", "None", "3", "4"], name="incl_na_lit"),
            {"include_na_literal": True},
            {
                "chart_metadata": {"version": "1.0.0"},
                "descriptive_stats": {
                "total": 4, "total_nonnull": 4,
                "n_non_numeric": 2, "k_non_numeric": 2,
                "category_labels": lambda xs: set(xs) == {"NaN","None"},
                "category_counts": lambda xs: sorted(xs) == [1,1],
                },
                "inferential_stats": {},
                "draft_inferential_findings": {},
            },
        ),

        # descending order
        (
        lambda: pd.Series(["a","a","b","c","c","c"], name="descending"),
        {"file_name": "strings.png", "sort_ascending": False},
        {
            "chart_metadata": {"file_name": "strings.png", "version": "1.0.0"},
            "descriptive_stats": {
            "category_counts": lambda xs: all(xs[i] >= xs[i+1] for i in range(len(xs)-1)),
            },
            "inferential_stats": {},
            "draft_inferential_findings": {},
        },
        ),
    ],
    ids=[
        "empty",
        "all_numeric",
        "some_non_numeric_strings",
        "numeric_like_strings",
        "labels_and_source_override",
        "include_na_literal",
        "descending_order"
    ],
)
def test_string_coercion_bar_plot_data_driven(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    # If a file_name is provided, also set save_path to tmp_path
    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["save_path"] = tmp_path

    ctx = StringCoercionBarContext(**kwargs)
    plot = StringCoercionBarPlot(ctx)

    payload = plot.run(s)

    # Reuse shared helper to assert only the fields specified in `expect`
    assert_plot_metadata(payload, expect, tmp_path)
