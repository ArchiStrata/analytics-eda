import os
import pytest
import numpy as np
import pandas as pd

from analytics_eda.core.numeric import plot_dispersion_boxplot

def test_default_parameters_no_save():
    # A simple series
    series = pd.Series([1, 2, 3, 4, 5], name="numeric_series")
    meta = plot_dispersion_boxplot(series)
    stats = meta['descriptive_stats']
    chart = meta['chart_metadata']

    # Expected values
    expected_n    = series.size
    expected_std  = series.std()
    expected_var  = series.var()
    expected_min  = series.min()
    expected_max  = series.max()
    expected_range= expected_max - expected_min
    expected_mad  = (series - series.mean()).abs().mean()
    expected_cv   = expected_std / series.mean()
    expected_p10  = series.quantile(0.10)
    expected_p25  = series.quantile(0.25)
    expected_p75  = series.quantile(0.75)
    expected_p90  = series.quantile(0.90)

    # Descriptive stats
    assert stats['n']      == expected_n
    assert stats['std']    == pytest.approx(expected_std)
    assert stats['var']    == pytest.approx(expected_var)
    assert stats['min']    == expected_min
    assert stats['max']    == expected_max
    assert stats['range']  == expected_range
    assert stats['mad']    == pytest.approx(expected_mad)
    assert stats['cv']     == pytest.approx(expected_cv)
    assert stats['pct_10'] == pytest.approx(expected_p10)
    assert stats['pct_25'] == pytest.approx(expected_p25)
    assert stats['pct_75'] == pytest.approx(expected_p75)
    assert stats['pct_90'] == pytest.approx(expected_p90)

    # Chart metadata
    assert chart['title']         == "Boxplot with Dispersion Statistics"
    assert chart['ylabel']        == "Value"
    assert chart['data_source'] is None
    assert chart['relative_path'] is None

def test_override_and_save(tmp_path):
    # Prepare a small series
    series = pd.Series([10, 20, 20, 30], name="numeric_series")
    custom_title   = "My Custom Dispersion Plot"
    custom_ylabel  = "Custom Y Label"
    custom_source  = "UnitTest Source"
    filename       = "dispersion.png"

    # Call with overrides and saving enabled
    meta = plot_dispersion_boxplot(
        series,
        title=custom_title,
        ylabel=custom_ylabel,
        data_source=custom_source,
        figsize=(12, 8),
        save_path=str(tmp_path),
        file_name=filename
    )
    stats = meta['descriptive_stats']
    chart = meta['chart_metadata']

    # Expected values
    expected_n     = series.size
    expected_std   = series.std()
    expected_var   = series.var()
    expected_min   = series.min()
    expected_max   = series.max()
    expected_range = expected_max - expected_min
    expected_mad   = (series - series.mean()).abs().mean()
    expected_cv    = expected_std / series.mean()
    expected_p10   = series.quantile(0.10)
    expected_p25   = series.quantile(0.25)
    expected_p75   = series.quantile(0.75)
    expected_p90   = series.quantile(0.90)

    # Descriptive stats
    assert stats['n']      == expected_n
    assert stats['std']    == pytest.approx(expected_std)
    assert stats['var']    == pytest.approx(expected_var)
    assert stats['min']    == expected_min
    assert stats['max']    == expected_max
    assert stats['range']  == expected_range
    assert stats['mad']    == pytest.approx(expected_mad)
    assert stats['cv']     == pytest.approx(expected_cv)
    assert stats['pct_10'] == pytest.approx(expected_p10)
    assert stats['pct_25'] == pytest.approx(expected_p25)
    assert stats['pct_75'] == pytest.approx(expected_p75)
    assert stats['pct_90'] == pytest.approx(expected_p90)

    # Chart metadata overrides
    assert chart['title']        == custom_title
    assert chart['ylabel']       == custom_ylabel
    assert chart['data_source']  == custom_source

    # Saved file exists and is a non-empty PNG
    saved_path = tmp_path / filename
    assert saved_path.exists() and saved_path.stat().st_size > 0

    with open(saved_path, 'rb') as f:
        header = f.read(8)
    assert header == b'\x89PNG\r\n\x1a\n'

    # Metadata path ends with filename
    assert os.path.basename(chart['relative_path']) == filename

def test_save_defaults_and_metadata(tmp_path):
    # Prepare a simple numeric series
    series = pd.Series([5, 10, 15, 20], name="numeric_series")
    filename = "dispersion.png"

    # Call with only save_path and file_name, using all defaults
    meta = plot_dispersion_boxplot(
        series,
        save_path=str(tmp_path),
        file_name=filename
    )
    chart = meta['chart_metadata']

    # Verify default chart metadata
    assert chart['title'] == "Boxplot with Dispersion Statistics"
    assert chart['ylabel'] == "Value"
    assert chart['data_source'] is None

    # Verify that the image file was created and is a valid PNG
    saved_path = tmp_path / filename
    assert saved_path.exists() and saved_path.is_file()
    assert saved_path.stat().st_size > 0

    with open(saved_path, 'rb') as f:
        header = f.read(8)
    assert header == b'\x89PNG\r\n\x1a\n'

    # The returned relative_path should end with the filename
    rel = chart['relative_path']
    assert os.path.basename(rel) == filename

def test_missing_series_name_raises_error():
    missing_name = pd.Series(dtype=float)
    with pytest.raises(ValueError):
        plot_dispersion_boxplot(missing_name)

def test_empty_series_returns_stats_and_defaults_dispersion():
    empty = pd.Series([], dtype=float, name="empty_series")
    meta = plot_dispersion_boxplot(empty)
    stats = meta['descriptive_stats']
    chart = meta['chart_metadata']

    # Descriptive stats for empty series
    assert stats['n'] == 0
    assert np.isnan(stats['std'])
    assert np.isnan(stats['var'])
    assert np.isnan(stats['min'])
    assert np.isnan(stats['max'])
    assert np.isnan(stats['range'])
    assert np.isnan(stats['mad'])
    assert np.isnan(stats['cv'])
    assert np.isnan(stats['pct_10'])
    assert np.isnan(stats['pct_25'])
    assert np.isnan(stats['pct_75'])
    assert np.isnan(stats['pct_90'])

    # Chart metadata defaults
    assert chart['title'] == "Boxplot with Dispersion Statistics"
    assert chart['ylabel'] == "Value"
    assert chart['data_source'] is None
    assert chart['relative_path'] is None

def test_override_and_save_extreme_bounds(tmp_path):
    # Construct a series with clear upper and lower extremes
    data = [-10, 0, 1, 2, 100]
    series = pd.Series(data, name="x")

    # Override parameters: use a small multiplier to flag both tails
    overrides = {
        "std_outlier_multiplier": 0.3,
        "title": "Custom Dispersion",
        "ylabel": "Units",
        "data_source": "UnitTest",
        "file_name": "dispersion.png"
    }

    meta = plot_dispersion_boxplot(
        series,
        **overrides,
        save_path=str(tmp_path)
    )
    desc  = meta["descriptive_stats"]
    chart = meta["chart_metadata"]

    # Basic counts
    assert desc["n"] == len(data)
    # With multiplier=0.3, lower bound ≈ mean−0.3σ flags 4 values, upper bound flags 1 value
    assert desc["extreme_lower_count"] == 4
    assert desc["extreme_upper_count"] == 1

    # Chart metadata
    assert chart["title"]       == overrides["title"]
    assert chart["ylabel"]      == overrides["ylabel"]
    assert chart["data_source"] == overrides["data_source"]
    assert pytest.approx(chart["std_outlier_multiplier"]) == overrides["std_outlier_multiplier"]

    # File saved correctly
    saved = tmp_path / overrides["file_name"]
    assert saved.exists() and saved.stat().st_size > 0
    with open(saved, "rb") as f:
        header = f.read(8)
    assert header == b'\x89PNG\r\n\x1a\n'

    # relative_path ends with the file name
    assert os.path.basename(chart["relative_path"]) == overrides["file_name"]


def test_violin_silhouette_and_save(tmp_path):
    # Prepare a series with some outliers
    rng = np.random.default_rng(0)
    data = np.concatenate([rng.normal(loc=0, scale=1, size=100), [5, -5]])
    series = pd.Series(data, name="x")

    # Override parameters and enable saving
    overrides = {
        "std_outlier_multiplier": 2.0,
        "title": "Violin Dispersion Test",
        "ylabel": "Units",
        "data_source": "UnitTest",
        "file_name": "violin_dispersion.png"
    }
    meta = plot_dispersion_boxplot(
        series,
        **overrides,
        save_path=str(tmp_path)
    )
    desc  = meta["descriptive_stats"]
    chart = meta["chart_metadata"]

    # Descriptive stats metadata
    assert desc["n"] == len(series)
    assert "iqr" in desc and isinstance(desc["iqr"], float)
    assert desc["extreme_lower_count"] >= 1
    assert desc["extreme_upper_count"] >= 1

    # Chart metadata overrides
    assert chart["title"]       == overrides["title"]
    assert chart["ylabel"]      == overrides["ylabel"]
    assert chart["data_source"] == overrides["data_source"]
    assert pytest.approx(chart["std_outlier_multiplier"]) == overrides["std_outlier_multiplier"]

    # Verify file was saved as a non-empty PNG
    saved = tmp_path / overrides["file_name"]
    assert saved.exists() and saved.stat().st_size > 0
    with open(saved, "rb") as f:
        header = f.read(8)
    assert header == b'\x89PNG\r\n\x1a\n'
