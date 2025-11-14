import pandas as pd
import pytest

from analytics_eda.core.numeric import DispersionPercentilePlot, DispersionPercentilePlotContext


@pytest.mark.parametrize(
    "series_factory, kwargs, expect",
    [
        (
            lambda: pd.Series([], dtype="float64", name="empty"),
            {},
            {
                "descriptive_stats": {"n": 0, "percentile_ranks": [], "percentile_values": {}},
                "draft_descriptive_findings": {"context": "No non-null observations.", "primary_finding": None, "secondary_finding": None},
            },
        ),
        (
            lambda: pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype="float64", name="seq"),
            {"percentile_step": 25},
            {
                "descriptive_stats": {
                    "n": 10,
                    "percentile_ranks": [25, 50, 75],
                    "percentile_values": {"p25": pytest.approx(3.25), "p50": pytest.approx(5.5), "p75": pytest.approx(7.75)},
                    "iqr": pytest.approx(4.5),
                    "ninety_ten_spread": pytest.approx(7.2, rel=1e-2),
                },
                "chart_metadata": {"title": "Percentile Dispersion of seq", "ylabel": "Value"},
            },
        ),
    ],
)
def test_dispersion_percentile_plot_run(series_factory, kwargs, expect, tmp_path, assert_plot_metadata):
    series = series_factory()
    ctx = DispersionPercentilePlotContext(save_path=tmp_path, **kwargs)
    plot = DispersionPercentilePlot(ctx)

    payload = plot.run(series)

    assert_plot_metadata(payload, expect, tmp_path)
