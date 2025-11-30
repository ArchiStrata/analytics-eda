import pandas as pd
import pytest

from analytics_eda.core.numeric.dispersion import (
    DispersionSigmaBandsPlot,
    DispersionSigmaBandsPlotContext,
)


@pytest.mark.parametrize(
    "series_factory, expected_exc, match",
    [
        (lambda: [1, 2, 3], TypeError, r"data must be a pandas Series or DataFrame"),
        (lambda: pd.Series(["a", "b", "c"], name="letters"), TypeError, r"Series must be numeric"),
        (lambda: pd.Series([1, 2, 3]), ValueError, r"must have a non-empty 'name'"),
        (lambda: pd.Series([1, 2, 3], name="   "), ValueError, r"must have a non-empty 'name'"),
    ],
    ids=["not_series", "bad_dtype", "missing_name", "blank_name"],
)
def test_validate_numeric_named_series_errors(series_factory, expected_exc, match):
    s = series_factory()
    with pytest.raises(expected_exc, match=match):
        ctx = DispersionSigmaBandsPlotContext()
        plot = DispersionSigmaBandsPlot(ctx)

        plot.run(s)


SIGMA_STORY_SERIES = pd.Series([-6, -3, -1, 0, 1, 2, 3, 6], dtype="float64", name="sigma_story")


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        (
            lambda: pd.Series([], dtype="float64", name="nums"),
            {},
            {
                "chart_metadata": {
                    "title": "Sigma Bands for nums",
                    "ylabel": "Value",
                    "data_source": None,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "n": 0,
                    "mean": None,
                    "std": None,
                    "sigma_1_lower": None,
                    "sigma_1_upper": None,
                    "count_within_1_sigma": 0,
                    "count_between_1_2_sigma": 0,
                    "count_between_2_3_sigma": 0,
                    "count_beyond_3_sigma": 0,
                    "count_beyond_outlier_threshold": 0,
                },
                "draft_descriptive_findings": {
                    "context": "No non-null observations.",
                    "primary_finding": None,
                    "secondary_finding": None,
                },
            },
        ),
        (
            lambda: pd.Series([5, 5, 5, 5], dtype="float64", name="flat"),
            {"file_name": "sigma_flat.png"},
            {
                "chart_metadata": {"file_name": "sigma_flat.png"},
                "descriptive_stats": {
                    "n": 4,
                    "mean": 5.0,
                    "std": 0.0,
                    "sigma_1_lower": 5.0,
                    "sigma_1_upper": 5.0,
                    "sigma_3_lower": 5.0,
                    "sigma_3_upper": 5.0,
                    "count_within_1_sigma": 4,
                    "count_between_1_2_sigma": 0,
                    "count_between_2_3_sigma": 0,
                    "count_beyond_3_sigma": 0,
                    "count_beyond_outlier_threshold": 0,
                },
                "draft_descriptive_findings": {
                    "context": "n = 4 • mean 5.00 • σ 0.00 • Outliers beyond ±3.0σ",
                    "primary_finding": "100.0% of observations fall within ±1σ; 100.0% stay within ±2σ.",
                    "secondary_finding": None,
                },
            },
        ),
        (
            lambda: SIGMA_STORY_SERIES.copy(),
            {"file_name": "sigma_story.png"},
            {
                "chart_metadata": {"file_name": "sigma_story.png"},
                "descriptive_stats": {
                    "n": 8,
                    "mean": pytest.approx(0.25, rel=1e-9),
                    "std": pytest.approx(3.693623849670827, rel=1e-9),
                    "sigma_1_lower": pytest.approx(-3.443623849670827, rel=1e-9),
                    "sigma_1_upper": pytest.approx(3.943623849670827, rel=1e-9),
                    "count_within_1_sigma": 6,
                    "count_between_1_2_sigma": 2,
                    "count_between_2_3_sigma": 0,
                    "count_beyond_3_sigma": 0,
                    "count_beyond_outlier_threshold": 0,
                },
                "draft_descriptive_findings": {
                    "context": "n = 8 • mean 0.25 • σ 3.69 • Outliers beyond ±3.0σ",
                    "primary_finding": "75.0% of observations fall within ±1σ; 100.0% stay within ±2σ.",
                    "secondary_finding": None,
                },
            },
        ),
        (
            lambda: SIGMA_STORY_SERIES.copy(),
            {"std_outlier_multiplier": 1.5, "file_name": "sigma_custom.png"},
            {
                "chart_metadata": {"file_name": "sigma_custom.png"},
                "descriptive_stats": {
                    "params": {"std_outlier_multiplier": 1.5},
                    "count_beyond_outlier_threshold": 2,
                    "extreme_lower_count": 1,
                    "extreme_upper_count": 1,
                    "extreme_lower_bound": pytest.approx(-5.290435774506241, rel=1e-9),
                    "extreme_upper_bound": pytest.approx(5.790435774506241, rel=1e-9),
                },
                "draft_descriptive_findings": {
                    "context": "n = 8 • mean 0.25 • σ 3.69 • Outliers beyond ±1.5σ",
                    "primary_finding": "75.0% of observations fall within ±1σ; 100.0% stay within ±2σ.",
                    "secondary_finding": "25.0% exceed ±1.5σ (1 low / 1 high).",
                },
            },
        ),
    ],
    ids=["empty_series", "degenerate_std", "sigma_story_default", "sigma_story_custom_multiplier"],
)
def test_dispersion_sigma_bands_plot_data_driven(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["base_dir"] = tmp_path

    ctx = DispersionSigmaBandsPlotContext(**kwargs)
    plot = DispersionSigmaBandsPlot(ctx)

    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
