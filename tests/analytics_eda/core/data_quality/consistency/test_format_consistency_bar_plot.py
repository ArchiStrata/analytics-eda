import pandas as pd
import pytest
from pytest import approx

from analytics_eda.core.data_quality import (
    ConsistencyFormatConsistencyBarContext,
    ConsistencyFormatConsistencyBarPlot,
)


@pytest.mark.parametrize(
    "make_series, expect",
    [
        (
            lambda: pd.Series(
                ["1", "01", "1.0", "abc", "ABC", "2020-01-01", None],
                name="formats",
            ),
            {
                "chart_metadata": {
                    "title": "Format Consistency for formats",
                    "xlabel": "Percent of total",
                    "ylabel": "Format group",
                    "version": "1.0.0",
                    "file_name": None,
                },
                "descriptive_stats": {
                    "bars": {
                        "Numeric string (int)": {"count": 2, "pct_of_total": approx(2 / 7)},
                        "Numeric string (decimal, 1dp)": {"count": 1, "pct_of_total": approx(1 / 7)},
                        "Alphabetic string": {"count": 2, "pct_of_total": approx(2 / 7)},
                        "Date string (YYYY-MM-DD)": {"count": 1, "pct_of_total": approx(1 / 7)},
                        "Null / Missing": {"count": 1, "pct_of_total": approx(1 / 7)},
                    },
                    "total": 7,
                    "total_nonnull": 6,
                    "dominant_format": "Numeric string (int)",
                    "dominant_ratio": approx(2 / 6),
                },
                "draft_descriptive_findings": {
                    "context": "6 non-null of 7 total",
                    "primary_finding": "Formats are fragmented; no single pattern exceeds 90% of non-null.",
                },
            },
        ),
    ],
    ids=["fragmented_formats"],
)
def test_format_consistency_bar_plot(make_series, expect, assert_plot_metadata, tmp_path):
    ctx = ConsistencyFormatConsistencyBarContext(base_dir=tmp_path)
    plot = ConsistencyFormatConsistencyBarPlot(ctx)

    payload = plot.run(make_series())

    assert_plot_metadata(payload, expect, tmp_path)
