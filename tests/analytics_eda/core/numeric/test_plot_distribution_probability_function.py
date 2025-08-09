import pytest

import pandas as pd
import numpy as np

from analytics_eda.core.numeric import plot_distribution_probability_function

@pytest.mark.parametrize(
    "series_factory, expected_exc, match",
    [
        # Not a Series
        (lambda: [1, 2, 3], TypeError, r"Input must be a pandas Series\."),
        # Non-numeric Series
        (lambda: pd.Series(["a", "b", "c"], name="letters"), TypeError, r"Series must be numeric"),
        # Missing name
        (lambda: pd.Series([1, 2, 3]), ValueError, r"must have a non-empty 'name'"),
        # Blank/whitespace name
        (lambda: pd.Series([1, 2, 3], name="   "), ValueError, r"must have a non-empty 'name'"),
    ],
    ids=["not_series", "bad_dtype", "missing_name", "blank_name"],
)
def test_validate_numeric_named_series_errors(series_factory, expected_exc, match):
    obj = series_factory()
    with pytest.raises(expected_exc, match=match):
        plot_distribution_probability_function(obj, is_discrete=True)


def test_pmf_descriptive_stats_and_chart_metadata_and_file(tmp_path):
    # discrete data with known values
    series = pd.Series([0, 1, 1, 2, 2, 2], name="cnts")
    out = plot_distribution_probability_function(
        series=series,
        is_discrete=True,
        name="cnts",
        data_source="unit_test",
        figsize=(12, 10),
        save_path=str(tmp_path),
        file_name="pmf.png"
    )

    # DESCRIPTIVE STATS
    ds = out["descriptive_stats"]
    # exact checks
    assert ds["n"] == 6
    assert ds["mean"] == pytest.approx(series.mean())
    assert ds["median"] == pytest.approx(series.median())
    assert ds["mode"] == series.mode().iloc[0]
    assert ds["variance"] == pytest.approx(series.var())
    assert ds["std"] == pytest.approx(series.std())
    assert ds["iqr"] == pytest.approx(series.quantile(0.75) - series.quantile(0.25))
    assert ds["skewness"] == pytest.approx(series.skew())
    assert ds["kurtosis"] == pytest.approx(series.kurtosis())
    assert ds["min"] == series.min()
    assert ds["max"] == series.max()

    # CHART METADATA
    cm = out["chart_metadata"]
    assert cm["title"] == "PMF of cnts"
    assert cm["xlabel"] == "cnts"
    assert cm["ylabel"] == "Probability P(X = x)"
    assert cm["data_source"] == "unit_test"
    assert cm["file_name"] == "pmf.png"

    # FILE WAS WRITTEN
    saved = tmp_path / "pmf.png"
    assert saved.exists() and saved.stat().st_size > 0


def test_pdf_descriptive_stats_and_chart_metadata_with_file(tmp_path):
    # continuous data: draw from a normal
    rng = np.random.default_rng(123)
    data = rng.normal(loc=5.0, scale=2.0, size=500)
    series = pd.Series(data, name="vals")

    out = plot_distribution_probability_function(
        series=series,
        is_discrete=False,
        name=None,
        data_source=None,
        figsize=(12, 10),
        save_path=str(tmp_path),
        file_name="pdf.png"
    )

    # DESCRIPTIVE STATS
    ds = out["descriptive_stats"]
    assert ds["n"] == 500
    assert ds["mean"] == pytest.approx(series.mean(), rel=1e-3)
    assert ds["median"] == pytest.approx(series.median(), rel=1e-3)
    assert ds["mode"] == pytest.approx(series.mode().iloc[0], rel=1e-3)
    assert ds["variance"] == pytest.approx(series.var(), rel=1e-3)
    assert ds["std"] == pytest.approx(series.std(), rel=1e-3)
    assert ds["iqr"] == pytest.approx(series.quantile(0.75) - series.quantile(0.25), rel=1e-3)
    # shape metrics
    assert "skewness" in ds and isinstance(ds["skewness"], float)
    assert "kurtosis" in ds and isinstance(ds["kurtosis"], float)
    assert ds["min"] == pytest.approx(series.min())
    assert ds["max"] == pytest.approx(series.max())

    # CHART METADATA
    cm = out["chart_metadata"]
    # name was None, so it should fall back to series.name
    assert cm["title"] == "PDF estimate of vals"
    assert cm["xlabel"] == "vals"
    assert cm["ylabel"] == "Density f(x)"
    assert cm["data_source"] is None
    assert cm["file_name"] == "pdf.png"

    # FILE WAS WRITTEN
    saved = tmp_path / "pdf.png"
    assert saved.exists() and saved.stat().st_size > 0


def test_pdf_descriptive_stats_and_chart_metadata_without_file(tmp_path):
    # continuous data: draw from a normal
    rng = np.random.default_rng(123)
    data = rng.normal(loc=5.0, scale=2.0, size=500)
    series = pd.Series(data, name="vals")

    out = plot_distribution_probability_function(
        series=series,
        is_discrete=False,
        name=None,
        data_source=None
    )

    # DESCRIPTIVE STATS
    ds = out["descriptive_stats"]
    assert ds["n"] == 500
    assert ds["mean"] == pytest.approx(series.mean(), rel=1e-3)
    assert ds["median"] == pytest.approx(series.median(), rel=1e-3)
    assert ds["mode"] == pytest.approx(series.mode().iloc[0], rel=1e-3)
    assert ds["variance"] == pytest.approx(series.var(), rel=1e-3)
    assert ds["std"] == pytest.approx(series.std(), rel=1e-3)
    assert ds["iqr"] == pytest.approx(series.quantile(0.75) - series.quantile(0.25), rel=1e-3)
    # shape metrics
    assert "skewness" in ds and isinstance(ds["skewness"], float)
    assert "kurtosis" in ds and isinstance(ds["kurtosis"], float)
    assert ds["min"] == pytest.approx(series.min())
    assert ds["max"] == pytest.approx(series.max())

    # CHART METADATA
    cm = out["chart_metadata"]
    # name was None, so it should fall back to series.name
    assert cm["title"] == "PDF estimate of vals"
    assert cm["xlabel"] == "vals"
    assert cm["ylabel"] == "Density f(x)"
    assert cm["data_source"] is None
    assert cm["file_name"] is None

    # NO FILE WAS WRITTEN
    # nothing in tmp_path
    assert not any(tmp_path.iterdir())
