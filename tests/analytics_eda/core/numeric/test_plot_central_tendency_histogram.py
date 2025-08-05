import os
import math
import pytest
import pandas as pd

from analytics_eda.core.numeric import plot_central_tendency_histogram

def test_single_mode():
    data = pd.Series([1, 1, 1, 2, 2, 3], name='numeric_series')
    bins = [0.5, 1.5, 2.5, 3.5]
    meta = plot_central_tendency_histogram(data, bins=bins)
    stats = meta['descriptive_stats']
    chart = meta['chart_metadata']
    
    # Descriptive stats
    assert stats['n'] == 6
    assert pytest.approx(stats['mean'], 0.01) == 1.67
    assert stats['median'] == 1.5
    assert stats['mode'] == [1]
    assert isinstance(stats['ci95'], tuple) and len(stats['ci95']) == 2
    
    # Chart metadata
    assert chart['bins'] == bins
    assert chart['title'] == "Distribution of numeric_series: Central Tendency"
    assert chart['xlabel'] == "Value"
    assert chart['ylabel'] == "Count"
    assert chart['data_source'] is None
    assert chart['file_name'] is None

def test_two_modes():
    data = pd.Series([1, 1, 2, 2, 3, 4], name='numeric_series')
    bins = [0.5, 1.5, 2.5, 3.5, 4.5]
    meta = plot_central_tendency_histogram(data, bins=bins)
    stats = meta['descriptive_stats']
    
    assert stats['n'] == 6
    assert stats['mode'] == [1, 2]

def test_three_modes():
    data = pd.Series([1, 1, 2, 2, 3, 3, 4], name='numeric_series')
    bins = [0.5, 1.5, 2.5, 3.5, 4.5]
    meta = plot_central_tendency_histogram(data, bins=bins)
    stats = meta['descriptive_stats']
    
    assert stats['n'] == 7
    assert sorted(stats['mode']) == [1, 2, 3]

def test_bell_shaped_default_bins():
    data = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 5], name='numeric_series')
    meta = plot_central_tendency_histogram(data)
    stats = meta['descriptive_stats']
    chart = meta['chart_metadata']
    
    assert stats['n'] == 9
    assert stats['mode'] == [3]
    assert chart['bins'] == math.ceil(math.sqrt(9))

def test_save_creates_file(tmp_path):
    data = pd.Series([0,1,2,2,3,3,3], name='numeric_series')
    filename = "hist.png"
    
    meta = plot_central_tendency_histogram(
        data,
        bins=5,
        save_path=str(tmp_path),
        file_name=filename
    )
    chart = meta['chart_metadata']
    # File exists
    saved_path = tmp_path / filename
    assert saved_path.exists() and saved_path.is_file()
    # Not empty
    assert saved_path.stat().st_size > 0
    # Check PNG signature
    with open(saved_path, 'rb') as f:
        sig = f.read(8)
    assert sig == b'\x89PNG\r\n\x1a\n'
    # Metadata path is a relative path ending with the filename
    rel = chart['file_name']
    assert os.path.basename(rel) == filename
    # Other metadata
    assert chart['data_source'] == None
    assert chart['bins'] == 5
    assert chart['xlabel'] == "Value"
    assert chart['ylabel'] == "Count"
    assert chart['title'] == "Distribution of numeric_series: Central Tendency"

def test_default_bins_square_root_choice():
    # Verify that when bins=None, it defaults to ceil(sqrt(n))
    data = pd.Series(range(16), name='numeric_series')  # n = 16
    expected_bins = math.ceil(math.sqrt(16))
    meta = plot_central_tendency_histogram(data, bins=None)
    chart = meta['chart_metadata']
    
    assert chart['bins'] == expected_bins
    # Ensure descriptive_stats n matches
    assert meta['descriptive_stats']['n'] == 16

def test_override_chart_labels_and_source(tmp_path):
    # Prepare data
    data = pd.Series([10, 20, 20, 30, 30, 30], name='numeric_series')
    custom_title = "Custom Histogram Title"
    custom_xlabel = "Custom X"
    custom_ylabel = "Custom Y"
    custom_source = "Custom DataSource"
    filename = "custom_hist.png"
    
    # Call with overrides
    meta = plot_central_tendency_histogram(
        data,
        bins=4,
        title_template=custom_title,
        xlabel=custom_xlabel,
        ylabel=custom_ylabel,
        data_source=custom_source,
        save_path=str(tmp_path),
        file_name=filename
    )
    stats = meta['descriptive_stats']
    chart = meta['chart_metadata']
    
    # Check overrides applied
    assert chart['title'] == custom_title
    assert chart['xlabel'] == custom_xlabel
    assert chart['ylabel'] == custom_ylabel
    assert chart['data_source'] == custom_source
    assert chart['bins'] == 4
    
    # Descriptive stats should still be correct
    assert stats['n'] == 6
    assert stats['mode'] == [30]
    
    # File exists and is valid PNG
    saved_path = tmp_path / filename
    assert saved_path.exists() and saved_path.is_file()
    assert saved_path.stat().st_size > 0
    with open(saved_path, 'rb') as f:
        assert f.read(8) == b'\x89PNG\r\n\x1a\n'

    # Metadata path is a relative path ending with the filename
    assert os.path.basename(chart['file_name']) == filename

def test_missing_series_name_raises_error():
    missing_name = pd.Series(dtype=float)
    with pytest.raises(ValueError):
        plot_central_tendency_histogram(missing_name)

def test_empty_series_returns_stats():
    empty = pd.Series([], dtype=float, name="empty_series")
    meta = plot_central_tendency_histogram(empty)
    stats = meta['descriptive_stats']
    chart = meta['chart_metadata']
    
    # Descriptive stats for empty series
    assert stats['n'] == 0
    assert math.isnan(stats['mean'])
    assert math.isnan(stats['median'])
    assert stats['mode'] == []
    ci_low, ci_high = stats['ci95']
    assert isinstance(stats['ci95'], tuple) and len(stats['ci95']) == 2
    assert math.isnan(ci_low) and math.isnan(ci_high)

    # Chart metadata defaults
    assert chart['bins'] == 0
    assert chart['title'] == "Distribution of empty_series: Central Tendency"
    assert chart['xlabel'] == "Value"
    assert chart['ylabel'] == "Count"
    assert chart['data_source'] is None
    assert chart['file_name'] is None
