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
                    "xlabel": "Count",
                    "ylabel": "Category",
                    "data_source": None,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "total": 0,
                    "total_nonnull": 0,
                    "n_non_numeric": 0,
                    "pct_non_numeric": 0.0,
                    "k_non_numeric": 0,
                    "labels": [],
                    "counts": [],
                },
            },
        ),

        # 1) All numeric → no string-coercion issues
        (
            lambda: pd.Series([1, 2, 3], dtype="float64", name="all_numeric"),
            {},
            {
                "descriptive_stats": {
                    "total": 3,
                    "total_nonnull": 3,
                    "n_non_numeric": 0,
                    "pct_non_numeric": 0,
                    "k_non_numeric": 0,
                    "labels": [],
                    "counts": []
                },
            },
        ),

        # 2) Some non-numeric strings (numeric-like strings should not count as issues)
        #    Series: [1, 'x', 2, 'y', 'x'] → total(non-null)=5 → n_strings=3 ('x','y','x'), unique_strings=2
        (
            lambda: pd.Series([1, "x", 2, "y", "x"], name="mixed"),
            {},
            {
                "descriptive_stats": {
                    "total": 5,
                    "total_nonnull": 5,
                    "n_non_numeric": 3,
                    "pct_non_numeric": 3 / 5,
                    "k_non_numeric": 2,
                    # labels/counts can be order-dependent; check content via predicates
                    "labels": lambda xs: set(xs) == {"x", "y"},
                    "counts": lambda xs: sorted(list(xs)) == [1, 2],
                },
            },
        ),

        # 3) All strings but numeric-like → zero issues (coerces fine)
        (
            lambda: pd.Series(["1", "2", "3"], name="numeric_like_strings"),
            {},
            {
                "descriptive_stats": {
                    "total": 3,
                    "total_nonnull": 3,
                    "n_non_numeric": 0,
                    "pct_non_numeric": 0,
                    "k_non_numeric": 0,
                },
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

        # 5) Save with explicit filename, verify basic stats and that a file gets written
        (
            lambda: pd.Series(["ok", "3.14", "bad", "NaN", "ok", "bad"], name="save_me"),
            # Note: '3.14' should coerce; 'NaN' the *string* will not coerce and counts as a non-numeric string
            {"file_name": "strings.png"},
            {
                "chart_metadata": {
                    "file_name": "strings.png",
                    "title": "Non-Numeric (String) Values in save_me",
                    "xlabel": "Count",
                    "ylabel": "Category",
                    "data_source": None,
                },
                "descriptive_stats": {
                    # We count non-null entries for totals; ensure your implementation matches
                    "total": 6,
                    "total_nonnull": 6,
                    # Non-numeric strings here: "ok","bad","NaN","ok","bad" → 5; unique={"ok","bad","NaN"} → 3
                    "n_non_numeric": 4,
                    "pct_non_numeric": 4 / 6,
                    "k_non_numeric": 2,
                },
            },
        ),
    ],
    ids=[
        "empty",
        "all_numeric",
        "some_non_numeric_strings",
        "numeric_like_strings",
        "labels_and_source_override",
        "save_with_png_signature",
    ],
)
def test_plot_string_coercion_bar_param(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
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
