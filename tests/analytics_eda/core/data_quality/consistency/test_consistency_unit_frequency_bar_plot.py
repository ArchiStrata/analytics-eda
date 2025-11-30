import pandas as pd
import pytest
from pytest import approx

from analytics_eda.core.data_quality import (
    ConsistencyUnitFrequencyBarContext,
    ConsistencyUnitFrequencyBarPlot,
)


@pytest.mark.parametrize(
    "make_series, expect",
    [
        (
            lambda: pd.Series(["1 kg", "2kg", "3 lbs", 4], name="units"),
            {
                "chart_metadata": {
                    "title": "Unit Frequency for units",
                    "xlabel": "Percent of non-null",
                    "ylabel": "Detected unit",
                    "version": "1.0.0",
                    "file_name": None,
                },
                "descriptive_stats": {
                    "bars": {
                        "KG": {"count": 2, "pct_of_nonnull": approx(0.5)},
                        "LBS": {"count": 1, "pct_of_nonnull": approx(0.25)},
                        "Unitless": {"count": 1, "pct_of_nonnull": approx(0.25)},
                    },
                    "dominant_unit": "KG",
                    "dominant_ratio": approx(0.5),
                    "total_nonnull": 4,
                },
                "draft_descriptive_findings": {
                    "context": "4 non-null values",
                    "primary_finding": "Units are mixed; top unit KG covers 50.0% of non-null.",
                },
            },
        ),
    ],
    ids=["mixed_units"],
)
def test_consistency_unit_frequency_bar_plot(make_series, expect, assert_plot_metadata, tmp_path):
    ctx = ConsistencyUnitFrequencyBarContext(base_dir=tmp_path)
    plot = ConsistencyUnitFrequencyBarPlot(ctx)

    payload = plot.run(make_series())

    assert_plot_metadata(payload, expect, tmp_path)
