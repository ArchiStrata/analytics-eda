import numpy as np
import pandas as pd
import pytest
from pytest import approx

from analytics_eda.core.data_quality import (
    CompletenessIssuesBarContext,
    CompletenessIssuesBarPlot,
)


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) Empty series → default descriptive + empty findings
        (
            lambda: pd.Series([], dtype="float64", name="empty"),
            {},
            {
                "chart_metadata": {
                    "title": "Completeness Issues for empty",
                    "xlabel": "Percent of total",
                    "ylabel": "Completeness issue",
                    "data_source": None,
                    "version": "1.0.0",
                    "file_name": None,
                },
                "descriptive_stats": {
                    "bars": {},
                    "subset_count": 0,
                    "total": 0,
                    "total_nonnull": 0,
                },
                "draft_descriptive_findings": {
                    "context": "0 values",
                    "primary_finding": "The series is empty.",
                    "secondary_finding": None,
                },
                "draft_inferential_findings": {},
                "inferential_stats": {},
            },
        ),
        # 1) Mix of missing types + artifact save (covers all encoded tokens and blank/empty)
        (
            lambda: pd.Series(
                [
                    1,
                    np.nan,  # Missing
                    None,  # Null
                    " ",  # Blank
                    "",  # Empty
                    "NA",
                    "n/a",
                    "NaN",
                    "none",
                    "Null",
                    "missing",
                    "unknown",
                    "-",
                    "--",
                    "?",
                ],
                name="completeness",
            ),
            {"file_name": "completeness.png"},
            {
                "chart_metadata": {
                    "data_source": None,
                    "file_name": "completeness.png",
                    "title": "Completeness Issues for completeness",
                    "version": "1.0.0",
                    "xlabel": "Percent of total",
                    "ylabel": "Completeness issue",
                },
                "descriptive_stats": {
                    "bars": {
                        "Missing": {"count": 1, "pct_of_total": approx(1 / 15)},
                        "Null": {"count": 1, "pct_of_total": approx(1 / 15)},
                        "Blank/Empty": {"count": 2, "pct_of_total": approx(2 / 15)},
                        "Encoded Missing": {"count": 10, "pct_of_total": approx(10 / 15)},
                    },
                    "denominator_key": "pct_of_total",
                    "input_categories": 4,
                    "input_nonzero_categories": 4,
                    "n_bars_rendered": 4,
                    "nonzero_categories": 4,
                    "params": {
                        "bar_height_source": "values",
                        "bar_sort_descending": True,
                        "bar_top_include_ties": True,
                        "bar_top_n": 1,
                        "encoded_missing_tokens": [
                            "na",
                            "n/a",
                            "nan",
                            "none",
                            "null",
                            "missing",
                            "unknown",
                            "-",
                            "--",
                            "?",
                        ],
                        "max_display_bars": 15,
                        "other_label": "Other",
                        "other_label_format": "{label} (k={k_agg})",
                        "other_min_count": None,
                        "other_respect_existing": True,
                        "show_count_in_bar_label": True,
                        "show_value_in_bar_label": True,
                    },
                    "pct_subset": approx(14 / 15),
                    "subset_count": 14,
                    "total": 15,
                    "total_nonnull": 13,
                    "top_labels": ["Encoded Missing"],
                    "unique_categories_total": 13,
                },
                "draft_descriptive_findings": {
                    "context": "15 values",
                    "primary_finding": "93.3% of 15 values are incomplete (14 rows).",
                    "secondary_finding": "Most common issue: Encoded Missing (66.7%, 10 rows).",
                },
                "draft_inferential_findings": {},
                "inferential_stats": {},
            },
        ),
        # 2) All values present → skip plot and retain findings
        (
            lambda: pd.Series(["good", "values", "here"], name="clean"),
            {"file_name": "clean.png"},
            {
                "chart_metadata": {
                    "file_name": None,  # skip plot when no issues
                },
                "descriptive_stats": {
                    "subset_count": 0,
                    "total": 3,
                    "total_nonnull": 3,
                    "bars": {
                        "Missing": {"count": 0, "pct_of_total": 0.0},
                        "Null": {"count": 0, "pct_of_total": 0.0},
                        "Blank/Empty": {"count": 0, "pct_of_total": 0.0},
                        "Encoded Missing": {"count": 0, "pct_of_total": 0.0},
                    },
                    "skip_plot": True,
                    "error": "no categories to display",
                },
                "draft_descriptive_findings": {
                    "context": "3 values",
                    "primary_finding": "All values are present; no completeness gaps detected.",
                    "secondary_finding": None,
                },
            },
        ),
    ],
    ids=["empty", "mixed_missing", "all_present"],
)
def test_completeness_issues_bar_plot_data_driven(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["base_dir"] = tmp_path

    ctx = CompletenessIssuesBarContext(**kwargs)
    plot = CompletenessIssuesBarPlot(ctx)

    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
