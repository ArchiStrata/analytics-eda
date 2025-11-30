import pandas as pd
import pytest

from analytics_eda.core.data_quality import (
    ConsistencyNumericCoercionBarContext,
    ConsistencyNumericCoercionBarPlot,
)


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) Empty series → zeros across stats; default labels and title
        (
            lambda: pd.Series([], dtype="float64", name="empty_col"),
            {},
            {
                "chart_metadata": {"data_source": None, "file_name": None, "title": "Numeric Coercion Issues for empty_col", "version": "1.0.0", "xlabel": "Percent of non‑null", "ylabel": "Category"},
                "descriptive_stats": {"bars": {}, "subset_count": 0, "total": 0, "total_nonnull": 0},
                "draft_descriptive_findings": {"context": "N (non‑null) = 0", "primary_finding": "The series is empty.", "secondary_finding": None},
                "draft_inferential_findings": {},
                "inferential_stats": {},
            },
        ),
        # 1) All numeric → no string-coercion issues
        (
            lambda: pd.Series([1, 2, 3], dtype="float64", name="all_numeric"),
            {"file_name": "strings.png"},
            {
                "chart_metadata": {"data_source": None, "file_name": None, "title": "Numeric Coercion Issues for all_numeric", "version": "1.0.0", "xlabel": "Percent of non‑null", "ylabel": "Category"},
                "descriptive_stats": {
                    "bars": {},
                    "denominator_key": "pct_of_nonnull",
                    "error": "no categories to display",
                    "input_categories": 0,
                    "input_nonzero_categories": 0,
                    "n_bars_rendered": 0,
                    "nonzero_categories": 0,
                    "params": {"bar_height_source": "values", "bar_sort_descending": False, "include_na_literal": False, "max_display_bars": 15, "other_label": "Other", "show_count_in_bar_label": False, "show_value_in_bar_label": True},
                    "pct_subset": 0.0,
                    "skip_plot": True,
                    "subset_count": 0,
                    "total": 3,
                    "total_nonnull": 3,
                    "unique_categories_total": 3,
                },
                "draft_descriptive_findings": {"context": "N (non‑null) = 3", "primary_finding": "No non-numeric values detected.", "secondary_finding": None},
                "draft_inferential_findings": {},
                "inferential_stats": {},
            },
        ),
        # 2) Some non-numeric strings (numeric-like strings should not count as issues)
        #    Series: [1, 'x', 2, 'y', 'x'] → total(non-null)=5 → n_strings=3 ('x','y','x'), unique_strings=2
        (
            lambda: pd.Series([1, "x", 2, "y", "x"], name="mixed"),
            {"file_name": "strings.png", "show_count_in_bar_label": True},
            {
                "chart_metadata": {"data_source": None, "file_name": "strings.png", "title": "Numeric Coercion Issues for mixed", "version": "1.0.0", "xlabel": "Percent of non‑null", "ylabel": "Category"},
                "descriptive_stats": {
                    "bars": {"x": {"count": 2, "pct_of_nonnull": 0.4}, "y": {"count": 1, "pct_of_nonnull": 0.2}},
                    "denominator_key": "pct_of_nonnull",
                    "input_categories": 2,
                    "input_nonzero_categories": 2,
                    "n_bars_rendered": 2,
                    "nonzero_categories": 2,
                    "params": {"bar_height_source": "values", "bar_sort_descending": False, "include_na_literal": False, "max_display_bars": 15, "other_label": "Other", "show_count_in_bar_label": True, "show_value_in_bar_label": True},
                    "pct_subset": 0.6,
                    "subset_count": 3,
                    "total": 5,
                    "total_nonnull": 5,
                    "unique_categories_total": 4,
                },
                "draft_descriptive_findings": {"context": "N (non‑null) = 5", "primary_finding": "60.0% of values failed numeric coercion (3 rows).", "secondary_finding": "Most frequent token: 'x' (40.0%, 2 rows)."},
                "draft_inferential_findings": {},
                "inferential_stats": {},
            },
        ),
        # 3) All strings but numeric-like → zero issues (coerces fine)
        (
            lambda: pd.Series(["1", "2", "3"], name="numeric_like_strings"),
            {"file_name": "strings.png"},
            {
                "chart_metadata": {"data_source": None, "file_name": None, "title": "Numeric Coercion Issues for numeric_like_strings", "version": "1.0.0", "xlabel": "Percent of non‑null", "ylabel": "Category"},
                "descriptive_stats": {
                    "bars": {},
                    "denominator_key": "pct_of_nonnull",
                    "error": "no categories to display",
                    "input_categories": 0,
                    "input_nonzero_categories": 0,
                    "n_bars_rendered": 0,
                    "nonzero_categories": 0,
                    "params": {"bar_height_source": "values", "bar_sort_descending": False, "include_na_literal": False, "max_display_bars": 15, "other_label": "Other", "show_count_in_bar_label": False, "show_value_in_bar_label": True},
                    "pct_subset": 0.0,
                    "skip_plot": True,
                    "subset_count": 0,
                    "total": 3,
                    "total_nonnull": 3,
                    "unique_categories_total": 3,
                },
                "draft_descriptive_findings": {"context": "N (non‑null) = 3", "primary_finding": "No non-numeric values detected.", "secondary_finding": None},
                "draft_inferential_findings": {},
                "inferential_stats": {},
            },
        ),
        # 4) Custom title via context name override; axis labels + data_source overrides
        (
            lambda: pd.Series(["a", "b", "c", "bad", "bad"], name="ignored"),
            {"name": "Revenue", "xlabel": "String Value", "ylabel": "Frequency", "data_source": "UnitTest"},
            {
                "chart_metadata": {
                    "title": "Numeric Coercion Issues for Revenue",
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
                "chart_metadata": {"data_source": None, "file_name": None, "title": "Numeric Coercion Issues for incl_na_lit", "version": "1.0.0", "xlabel": "Percent of non‑null", "ylabel": "Category"},
                "descriptive_stats": {
                    "bars": {"NaN": {"count": 1, "pct_of_nonnull": 0.25}, "None": {"count": 1, "pct_of_nonnull": 0.25}},
                    "denominator_key": "pct_of_nonnull",
                    "input_categories": 2,
                    "input_nonzero_categories": 2,
                    "n_bars_rendered": 2,
                    "nonzero_categories": 2,
                    "params": {"bar_height_source": "values", "bar_sort_descending": False, "include_na_literal": True, "max_display_bars": 15, "other_label": "Other", "show_count_in_bar_label": False, "show_value_in_bar_label": True},
                    "pct_subset": 0.5,
                    "subset_count": 2,
                    "total": 4,
                    "total_nonnull": 4,
                    "unique_categories_total": 4,
                },
                "draft_descriptive_findings": {"context": "N (non‑null) = 4", "primary_finding": "50.0% of values failed numeric coercion (2 rows).", "secondary_finding": "Most frequent tokens (tie): 'NaN' (25.0%, 1 rows), 'None' (25.0%, 1 rows)."},
                "draft_inferential_findings": {},
                "inferential_stats": {},
            },
        ),
        # descending order
        (
            lambda: pd.Series(["a", "a", "b", "c", "c", "c"], name="descending"),
            {"file_name": "strings.png", "bar_sort_descending": True},
            {
                "chart_metadata": {"data_source": None, "file_name": "strings.png", "title": "Numeric Coercion Issues for descending", "version": "1.0.0", "xlabel": "Percent of non‑null", "ylabel": "Category"},
                "descriptive_stats": {
                    "bars": {"a": {"count": 2, "pct_of_nonnull": 0.3333333333333333}, "b": {"count": 1, "pct_of_nonnull": 0.16666666666666666}, "c": {"count": 3, "pct_of_nonnull": 0.5}},
                    "denominator_key": "pct_of_nonnull",
                    "input_categories": 3,
                    "input_nonzero_categories": 3,
                    "n_bars_rendered": 3,
                    "nonzero_categories": 3,
                    "params": {"bar_height_source": "values", "bar_sort_descending": True, "include_na_literal": False, "max_display_bars": 15, "other_label": "Other", "show_count_in_bar_label": False, "show_value_in_bar_label": True},
                    "pct_subset": 1.0,
                    "subset_count": 6,
                    "total": 6,
                    "total_nonnull": 6,
                    "unique_categories_total": 3,
                },
                "draft_descriptive_findings": {"context": "N (non‑null) = 6", "primary_finding": "100.0% of values failed numeric coercion (6 rows).", "secondary_finding": "Most frequent token: 'c' (50.0%, 3 rows)."},
                "draft_inferential_findings": {},
                "inferential_stats": {},
            },
        ),
    ],
    ids=["empty", "all_numeric", "some_non_numeric_strings", "numeric_like_strings", "labels_and_source_override", "include_na_literal", "descending_order"],
)
def test_consistency_numeric_coercion_bar_plot_data_driven(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    # If a file_name is provided, also set base_dir to tmp_path
    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["base_dir"] = tmp_path

    ctx = ConsistencyNumericCoercionBarContext(**kwargs)
    plot = ConsistencyNumericCoercionBarPlot(ctx)

    payload = plot.run(s)

    # Reuse shared helper to assert only the fields specified in `expect`
    assert_plot_metadata(payload, expect, tmp_path)
