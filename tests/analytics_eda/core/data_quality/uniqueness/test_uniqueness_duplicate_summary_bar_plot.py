import pandas as pd
import pytest
from pytest import approx

from analytics_eda.core.data_quality import (
    UniquenessDuplicateSummaryBarContext,
    UniquenessDuplicateSummaryBarPlot,
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
                    "title": "Duplicate Summary for empty",
                    "xlabel": "Percent of non-null",
                    "ylabel": "Uniqueness status",
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
                    "context": "0 non-null values",
                    "primary_finding": "The series is empty.",
                    "secondary_finding": None,
                },
            },
        ),
        # 1) All unique non-null values
        (
            lambda: pd.Series(["a", "b", "c"], name="unique"),
            {},
            {
                "chart_metadata": {
                    "title": "Duplicate Summary for unique",
                    "xlabel": "Percent of non-null",
                    "ylabel": "Uniqueness status",
                    "data_source": None,
                    "version": "1.0.0",
                    "file_name": None,
                },
                "descriptive_stats": {
                    "bars": {
                        "Distinct values": {"count": 3, "pct_of_nonnull": approx(1.0)},
                        "Duplicate entries": {"count": 0, "pct_of_nonnull": approx(0.0)},
                    },
                    "denominator_key": "pct_of_nonnull",
                    "subset_count": 3,
                    "total": 3,
                    "total_nonnull": 3,
                    "params": {
                        "nunique_native": 3,
                        "duplicates": 0,
                        "duplicate_ratio": approx(0.0),
                        "distinct_ratio": approx(1.0),
                        "one_value_column": False,
                    },
                    "top_labels": ["Distinct values"],
                },
                "draft_descriptive_findings": {
                    "context": "3 non-null values",
                    "primary_finding": "All non-null values are unique.",
                    "secondary_finding": None,
                },
            },
        ),
        # 2) Mixed duplicates + alert + artifact
        (
            lambda: pd.Series([1, 1, 1, 2, 2, 3, None], name="dup"),
            {"file_name": "duplicates.png", "high_duplicate_ratio_threshold": 0.4},
            {
                "chart_metadata": {
                    "title": "Duplicate Summary for dup",
                    "xlabel": "Percent of non-null",
                    "ylabel": "Uniqueness status",
                    "data_source": None,
                    "version": "1.0.0",
                    "file_name": "duplicates.png",
                },
                "descriptive_stats": {
                    "bars": {
                        "Distinct values": {"count": 3, "pct_of_nonnull": approx(0.5)},
                        "Duplicate entries": {"count": 3, "pct_of_nonnull": approx(0.5)},
                    },
                    "denominator_key": "pct_of_nonnull",
                    "subset_count": 6,
                    "total": 7,
                    "total_nonnull": 6,
                    "params": {
                        "nunique_native": 3,
                        "duplicates": 3,
                        "duplicate_ratio": approx(0.5),
                        "distinct_ratio": approx(0.5),
                        "one_value_column": False,
                        "high_duplicate_ratio_threshold": 0.4,
                        "low_duplicate_ratio_threshold": None,
                    },
                    "top_labels": ["Distinct values", "Duplicate entries"],
                },
                "draft_descriptive_findings": {
                    "context": "6 non-null values",
                    "primary_finding": "50.0% of 6 non-null values are duplicates (3 entries).",
                    "secondary_finding": "Distinct values: 3 (50.0%). Duplicate ratio exceeds the alert threshold (40%).",
                },
            },
        ),
    ],
    ids=["empty", "all_unique", "mixed_duplicates"],
)
def test_uniqueness_duplicate_summary_bar_plot_data_driven(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["base_dir"] = tmp_path

    ctx = UniquenessDuplicateSummaryBarContext(**kwargs)
    plot = UniquenessDuplicateSummaryBarPlot(ctx)

    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
