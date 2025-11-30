import pandas as pd
import pytest
from pytest import approx

from analytics_eda.core.data_quality import (
    ConsistencyDecimalPrecisionBarContext,
    ConsistencyDecimalPrecisionBarPlot,
)


@pytest.mark.parametrize(
    "make_series, expect",
    [
        (
            lambda: pd.Series([1, 1.2, 2.34, None], name="precision"),
            {
                "chart_metadata": {
                    "title": "Decimal Precision for precision",
                    "xlabel": "Percent of non-null",
                    "ylabel": "Decimal places",
                    "version": "1.0.0",
                    "file_name": None,
                },
                "descriptive_stats": {
                    "bars": {
                        "0 dp": {"count": 1, "pct_of_nonnull": approx(1 / 3)},
                        "1 dp": {"count": 1, "pct_of_nonnull": approx(1 / 3)},
                        "2 dp": {"count": 1, "pct_of_nonnull": approx(1 / 3)},
                    },
                    "total_nonnull": 3,
                    "dominant_precision_dp": 0,
                    "dominant_ratio": approx(1 / 3),
                },
                "draft_descriptive_findings": {
                    "context": "3 non-null values",
                    "primary_finding": "Precision is fragmented; top bucket 0dp covers 33.3% of non-null.",
                },
            },
        ),
    ],
    ids=["fragmented_precision"],
)
def test_consistency_decimal_precision_bar_plot(make_series, expect, assert_plot_metadata, tmp_path):
    ctx = ConsistencyDecimalPrecisionBarContext(base_dir=tmp_path)
    plot = ConsistencyDecimalPrecisionBarPlot(ctx)

    payload = plot.run(make_series())

    assert_plot_metadata(payload, expect, tmp_path)
