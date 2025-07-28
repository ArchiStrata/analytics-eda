import os
import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric import plot_distribution_kde

def test_default_parameters_no_save():
    # simple unimodal series
    series = pd.Series([1, 2, 2, 3, 4], name="test_series")
    meta = plot_distribution_kde(series)
    stats = meta['descriptive_stats']
    chart = meta['chart_metadata']

    # Descriptive stats
    assert stats['n'] == 5
    assert isinstance(stats['skewness'], float)
    assert isinstance(stats['kurtosis'], float)
    assert isinstance(stats['modes_count'], int)
    assert stats['quartile_skew'] == pytest.approx(
        (series.quantile(0.75) + series.quantile(0.25) - 2*series.median())
        / (series.quantile(0.75) - series.quantile(0.25))
    )
    for p in ['pct_10','pct_25','pct_50','pct_75','pct_90']:
        assert stats[p] == pytest.approx(series.quantile({
            'pct_10':0.10, 'pct_25':0.25,
            'pct_50':0.50, 'pct_75':0.75,
            'pct_90':0.90
        }[p]))

    # Chart metadata defaults
    assert chart['title'] == "KDE Plot of Distribution Shape"
    assert chart['xlabel'] == "Value"
    assert chart['ylabel'] == "Density"
    assert chart['data_source'] is None
    assert chart['relative_path'] is None

def test_override_and_save(tmp_path):
    # multimodal series
    series = pd.Series([0,0,1,1,2,3,5,5,5], name="s")
    custom_title  = "Custom KDE"
    custom_xlabel = "X-axis"
    custom_ylabel = "Y-axis"
    custom_source = "UnitTest"
    filename      = "shape.png"

    meta = plot_distribution_kde(
        series,
        title=custom_title,
        xlabel=custom_xlabel,
        ylabel=custom_ylabel,
        data_source=custom_source,
        figsize=(12,10),
        save_path=str(tmp_path),
        file_name=filename
    )
    stats = meta['descriptive_stats']
    chart = meta['chart_metadata']

    # Modes count should match detected peaks
    assert stats['modes_count'] >= 1

    # Overrides applied
    assert chart['title']       == custom_title
    assert chart['xlabel']      == custom_xlabel
    assert chart['ylabel']      == custom_ylabel
    assert chart['data_source'] == custom_source

    # File written correctly
    saved = tmp_path / filename
    assert saved.exists() and saved.stat().st_size > 0
    with open(saved, 'rb') as f:
        assert f.read(8) == b'\x89PNG\r\n\x1a\n'
    assert os.path.basename(chart['relative_path']) == filename

def test_save_defaults_and_metadata(tmp_path):
    series = pd.Series(range(10), name="nums")
    filename = "out.png"

    meta = plot_distribution_kde(
        series,
        save_path=str(tmp_path),
        file_name=filename
    )
    chart = meta['chart_metadata']

    # Defaults preserved
    assert chart['title'] == "KDE Plot of Distribution Shape"
    assert chart['xlabel'] == "Value"
    assert chart['ylabel'] == "Density"
    assert chart['data_source'] is None

    # File exists
    saved = tmp_path / filename
    assert saved.exists()
    assert saved.stat().st_size > 0

def test_missing_series_name_raises_error():
    # Series without a name should trigger validation error
    unnamed = pd.Series([1,2,3])
    with pytest.raises(ValueError):
        plot_distribution_kde(unnamed)

def test_empty_series_returns_stats_and_defaults():
    empty = pd.Series([], dtype=float, name="empty")
    meta = plot_distribution_kde(empty)
    stats = meta['descriptive_stats']
    chart = meta['chart_metadata']

    # Empty stats should be nan or zero
    assert stats['n'] == 0
    assert np.isnan(stats['skewness'])
    assert np.isnan(stats['kurtosis'])
    assert stats['modes_count'] == 0
    assert np.isnan(stats['quartile_skew'])
    for p in ['pct_10','pct_25','pct_50','pct_75','pct_90']:
        assert np.isnan(stats[p])

    # Defaults for chart metadata
    assert chart['relative_path'] is None
    assert chart['data_source'] is None
    assert chart['title'] == "KDE Plot of Distribution Shape"
    assert chart['xlabel'] == "Value"
    assert chart['ylabel'] == "Density"
