import os
import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric import plot_distribution_density
from analytics_eda.core.numeric.binning_rules import sturges_bins, scott_bins, freedman_diaconis_bins, doane_bins

def test_default_parameters_no_save():
    # simple unimodal series
    series = pd.Series([1, 2, 2, 3, 4], name="test_series")
    meta = plot_distribution_density(series)
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
    assert chart['title'] == "Distribution Density: Histogram with KDE"
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

    meta = plot_distribution_density(
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

    meta = plot_distribution_density(
        series,
        save_path=str(tmp_path),
        file_name=filename
    )
    chart = meta['chart_metadata']

    # Defaults preserved
    assert chart['title'] == "Distribution Density: Histogram with KDE"
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
        plot_distribution_density(unnamed)

def test_empty_series_returns_stats_and_defaults():
    empty = pd.Series([], dtype=float, name="empty")
    meta = plot_distribution_density(empty)
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
    assert chart['title'] == "Distribution Density: Histogram with KDE"
    assert chart['xlabel'] == "Value"
    assert chart['ylabel'] == "Density"

def test_plot_distribution_density_with_sturges_method(tmp_path):
    """
    Verify that `plot_distribution_kde` respects the 'sturges' bin_method by
    comparing its output to explicitly passing the computed Sturges bin count.
    """
    # Prepare a simple series
    series = pd.Series([0, 0, 1, 1, 2, 3, 5, 5, 5], name="s")
    filename1 = "shape_sturges.png"
    filename2 = "shape_explicit.png"

    # Compute expected bins via Sturges' Rule
    expected_k = sturges_bins(series)

    # Call with bin_method='sturges'
    meta1 = plot_distribution_density(
        series,
        bin_method='sturges',
        save_path=str(tmp_path),
        file_name=filename1
    )

    # Call with explicit bins=expected_k
    meta2 = plot_distribution_density(
        series,
        bins=expected_k,
        save_path=str(tmp_path),
        file_name=filename2
    )

    # Both calls should produce the same descriptive statistics
    assert meta1['descriptive_stats'] == meta2['descriptive_stats']

    # Files should be created and valid PNGs
    path1 = tmp_path / filename1
    path2 = tmp_path / filename2
    for path in (path1, path2):
        assert path.exists(), f"File {path} was not created"
        assert path.stat().st_size > 0, f"File {path} is empty"
        # Check PNG signature
        with open(path, 'rb') as f:
            sig = f.read(8)
        assert sig == b'\x89PNG\r\n\x1a\n'

    # Ensure relative_path in metadata matches filenames
    assert os.path.basename(meta1['chart_metadata']['relative_path']) == filename1
    assert os.path.basename(meta2['chart_metadata']['relative_path']) == filename2

def test_plot_distribution_density__with_scott_method(tmp_path):
    """
    Verify that `plot_distribution_kde` respects the 'scott' bin_method by
    comparing its output against explicitly passing the computed Scott bin count.
    """
    # Prepare a simple multimodal series
    series = pd.Series([0, 0, 1, 1, 2, 3, 5, 5, 5], name="s")
    filename_method = "shape_scott_method.png"
    filename_explicit = "shape_scott_explicit.png"

    # Compute expected bins via Scott's Rule
    expected_k = scott_bins(series)

    # Call with bin_method='scott'
    meta_method = plot_distribution_density(
        series,
        bin_method='scott',
        save_path=str(tmp_path),
        file_name=filename_method
    )

    # Call with explicit bins
    meta_explicit = plot_distribution_density(
        series,
        bins=expected_k,
        save_path=str(tmp_path),
        file_name=filename_explicit
    )

    # Descriptive statistics should match
    assert meta_method['descriptive_stats'] == meta_explicit['descriptive_stats']

    # Validate files created and non-empty
    path_method = tmp_path / filename_method
    path_explicit = tmp_path / filename_explicit
    for path in (path_method, path_explicit):
        assert path.exists(), f"Expected file {path} to exist"
        assert path.stat().st_size > 0, f"File {path} is empty"
        # PNG signature check
        with open(path, 'rb') as f:
            signature = f.read(8)
        assert signature == b'\x89PNG\r\n\x1a\n', "Invalid PNG file signature"

    # Check relative_path metadata
    rel1 = os.path.basename(meta_method['chart_metadata']['relative_path'])
    rel2 = os.path.basename(meta_explicit['chart_metadata']['relative_path'])
    assert rel1 == filename_method
    assert rel2 == filename_explicit

def test_plot_distribution_density_with_fd_method(tmp_path):
    """
    Verify that `plot_distribution_kde` respects the 'freedman_diaconis' bin_method by
    comparing its output against explicitly passing the computed FD bin count.
    """
    # Prepare a sample series
    series = pd.Series([2, 4, 4, 4, 5, 6, 8, 10, 10, 12], name="s")
    filename_method = "shape_fd_method.png"
    filename_explicit = "shape_fd_explicit.png"

    # Compute expected bins via Freedman–Diaconis Rule
    expected_k = freedman_diaconis_bins(series)

    # Call with bin_method='freedman_diaconis'
    meta_method = plot_distribution_density(
        series,
        bin_method='freedman_diaconis',
        save_path=str(tmp_path),
        file_name=filename_method
    )

    # Call with explicit bins
    meta_explicit = plot_distribution_density(
        series,
        bins=expected_k,
        save_path=str(tmp_path),
        file_name=filename_explicit
    )

    # Both calls should yield the same descriptive stats
    assert meta_method['descriptive_stats'] == meta_explicit['descriptive_stats']

    # Ensure files are created and non-empty
    path_method = tmp_path / filename_method
    path_explicit = tmp_path / filename_explicit
    for path in (path_method, path_explicit):
        assert path.exists(), f"Expected file {path} to exist"
        assert path.stat().st_size > 0, f"File {path} is empty"
        # Check PNG signature
        with open(path, 'rb') as f:
            sig = f.read(8)
        assert sig == b'\x89PNG\r\n\x1a\n', "Invalid PNG file signature"

    # Confirm relative_path metadata matches filenames
    rel1 = os.path.basename(meta_method['chart_metadata']['relative_path'])
    rel2 = os.path.basename(meta_explicit['chart_metadata']['relative_path'])
    assert rel1 == filename_method
    assert rel2 == filename_explicit

def test_plot_distribution_density_with_doane_method(tmp_path):
    """
    Verify that `plot_distribution_kde` respects the 'doane' bin_method by
    comparing its output against explicitly passing the computed Doane bin count.
    """
    # Prepare a sample series with sufficient skew
    series = pd.Series([0, 0, 1, 1, 2, 3, 5, 5, 5], name="s")
    filename_method = "shape_doane_method.png"
    filename_explicit = "shape_doane_explicit.png"

    # Compute expected bins via Doane's Rule
    expected_k = doane_bins(series)

    # Call with bin_method='doane'
    meta_method = plot_distribution_density(
        series,
        bin_method='doane',
        save_path=str(tmp_path),
        file_name=filename_method
    )

    # Call with explicit bins
    meta_explicit = plot_distribution_density(
        series,
        bins=expected_k,
        save_path=str(tmp_path),
        file_name=filename_explicit
    )

    # Both calls should yield the same descriptive stats
    assert meta_method['descriptive_stats'] == meta_explicit['descriptive_stats']

    # Ensure files are created and non-empty
    path_method = tmp_path / filename_method
    path_explicit = tmp_path / filename_explicit
    for path in (path_method, path_explicit):
        assert path.exists(), f"Expected file {path} to exist"
        assert path.stat().st_size > 0, f"File {path} is empty"
        # Check PNG signature
        with open(path, 'rb') as f:
            signature = f.read(8)
        assert signature == b'\x89PNG\r\n\x1a\n', "Invalid PNG file signature"

    # Confirm relative_path metadata matches filenames
    rel1 = os.path.basename(meta_method['chart_metadata']['relative_path'])
    rel2 = os.path.basename(meta_explicit['chart_metadata']['relative_path'])
    assert rel1 == filename_method
    assert rel2 == filename_explicit
