import pandas as pd
import pytest

from analytics_eda.core.numeric.dispersion import DispersionDecilePlot, DispersionDecilePlotContext


@pytest.mark.parametrize(
    "series_factory, kwargs, expect",
    [
        (
            lambda: pd.Series([], dtype="float64", name="empty"),
            {},
            {
                "descriptive_stats": {"n": 0, "percentile_ranks": [], "decile_values": {}},
                "draft_descriptive_findings": {"context": "No non-null observations.", "primary_finding": None, "secondary_finding": None},
            },
        ),
        (
            lambda: pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype="float64", name="seq"),
            {},
            {
                "descriptive_stats": {
                    "n": 10,
                    "percentile_ranks": [10, 20, 30, 40, 50, 60, 70, 80, 90],
                    "decile_values": {
                        "D1": pytest.approx(1.9),
                        "D2": pytest.approx(2.8),
                        "D3": pytest.approx(3.7),
                        "D4": pytest.approx(4.6),
                        "D5": pytest.approx(5.5),
                        "D6": pytest.approx(6.4),
                        "D7": pytest.approx(7.3),
                        "D8": pytest.approx(8.2),
                        "D9": pytest.approx(9.1),
                    },
                    "iqr": pytest.approx(4.5),
                    "ninety_ten_spread": pytest.approx(7.2, rel=1e-2),
                    "decile_spread": pytest.approx(7.2, rel=1e-2),
                },
                "chart_metadata": {"title": "Decile Dispersion of seq", "xlabel": "Decile", "ylabel": "Value"},
            },
        ),
    ],
)
def test_dispersion_decile_plot_run(series_factory, kwargs, expect, tmp_path, assert_plot_metadata):
    series = series_factory()
    ctx = DispersionDecilePlotContext(base_dir=tmp_path, **kwargs)
    plot = DispersionDecilePlot(ctx)

    payload = plot.run(series)

    assert_plot_metadata(payload, expect, tmp_path)
