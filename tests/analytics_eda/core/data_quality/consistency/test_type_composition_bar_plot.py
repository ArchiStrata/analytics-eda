import pandas as pd
import pytest
from pytest import approx

from analytics_eda.core.data_quality import (
    ConsistencyTypeCompositionBarContext,
    ConsistencyTypeCompositionBarPlot,
)


@pytest.mark.parametrize(
    "make_series, expect",
    [
        (
            lambda: pd.Series([1, "2", "abc", pd.Timestamp("2020-01-01"), True, None], name="mix"),
            {
                "chart_metadata": {
                    "title": "Type Composition for mix",
                    "xlabel": "Percent of total",
                    "ylabel": "Detected type",
                    "version": "1.0.0",
                    "file_name": None,
                },
                "descriptive_stats": {
                    "bars": {
                        "Numeric": {"count": 2, "pct_of_total": approx(2 / 6)},
                        "Text": {"count": 1, "pct_of_total": approx(1 / 6)},
                        "Datetime": {"count": 1, "pct_of_total": approx(1 / 6)},
                        "Boolean": {"count": 1, "pct_of_total": approx(1 / 6)},
                        "Null / Missing": {"count": 1, "pct_of_total": approx(1 / 6)},
                    },
                    "total": 6,
                    "total_nonnull": 5,
                    "dominant_type": "Numeric",
                    "dominant_ratio": approx(0.4),
                },
                "draft_descriptive_findings": {
                    "context": "5 non-null of 6 total",
                    "primary_finding": "Column is mixed-type; no single type exceeds 90% of non-null.",
                },
            },
        ),
    ],
    ids=["mixed_types"],
)
def test_type_composition_bar_plot(make_series, expect, assert_plot_metadata, tmp_path):
    ctx = ConsistencyTypeCompositionBarContext(base_dir=tmp_path)
    plot = ConsistencyTypeCompositionBarPlot(ctx)

    payload = plot.run(make_series())

    assert_plot_metadata(payload, expect, tmp_path)
