import os
import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric import plot_distribution_ecdf

def test_default_parameters_no_save():
    # Simple two-point series → one gap of size 2
    s = pd.Series([1, 3], name="two_point")
    meta = plot_distribution_ecdf(s)
    stats = meta['descriptive_stats']
    chart = meta['chart_metadata']

    # Descriptive stats
    assert stats['n'] == 2
    assert stats['n_unique'] == 2
    assert stats['gaps'] == [2.0]
    assert stats['max_gap'] == 2.0
    assert stats['median_gap'] == 2.0
    assert stats['pct10_gap'] == 2.0
    assert stats['pct50_gap'] == 2.0
    assert stats['pct90_gap'] == 2.0
    assert stats['n_gaps_above_thr'] is None
    assert pytest.approx(stats['total_gap_prop']) == 1.0  # (2)/(3−1)
    assert stats['max_gap_loc'] == 2.0

    # Chart metadata defaults
    assert chart['title'] == "ECDF with Gap Analysis"
    assert chart['xlabel'] == "Value"
    assert chart['ylabel'] == "ECDF"
    assert chart['data_source'] is None
    assert chart['threshold'] is None
    assert chart['relative_path'] is None

def test_override_and_save(tmp_path):
    # Same two-point data, override labels, threshold, and save
    s = pd.Series([1, 3], name="two_point")
    custom = {
        'title': "My ECDF",
        'xlabel': "X-axis",
        'ylabel': "Y-axis",
        'data_source': "UnitTest",
        'threshold': 1.0
    }
    fname = "ecdf.png"

    meta = plot_distribution_ecdf(
        s,
        title=custom['title'],
        xlabel=custom['xlabel'],
        ylabel=custom['ylabel'],
        data_source=custom['data_source'],
        threshold=custom['threshold'],
        save_path=str(tmp_path),
        file_name=fname
    )
    stats = meta['descriptive_stats']
    chart = meta['chart_metadata']

    # Overrides applied
    assert chart['title'] == custom['title']
    assert chart['xlabel'] == custom['xlabel']
    assert chart['ylabel'] == custom['ylabel']
    assert chart['data_source'] == custom['data_source']
    assert chart['threshold'] == custom['threshold']

    # Threshold count: gap=2 > 1 → 1
    assert stats['n_gaps_above_thr'] == 1

    # File checks
    saved = tmp_path / fname
    assert saved.exists() and saved.is_file()
    assert saved.stat().st_size > 0

    # PNG signature
    with saved.open('rb') as f:
        sig = f.read(8)
    assert sig == b'\x89PNG\r\n\x1a\n'

    # relative_path ends with filename
    assert os.path.basename(chart['relative_path']) == fname

def test_save_defaults_and_metadata(tmp_path):
    # Save with defaults only
    s = pd.Series([0, 5, 10], name="three_point")
    fname = "out.png"
    meta = plot_distribution_ecdf(
        s,
        save_path=str(tmp_path),
        file_name=fname
    )
    chart = meta['chart_metadata']

    # File exists
    saved = tmp_path / fname
    assert saved.exists()

    # Metadata for defaults
    assert chart['title'] == "ECDF with Gap Analysis"
    assert chart['threshold'] is None
    assert os.path.basename(chart['relative_path']) == fname

def test_missing_series_name_raises_error():
    # Series without name should trigger validation error
    s = pd.Series([1, 2, 3])  # name is None
    with pytest.raises(ValueError):
        plot_distribution_ecdf(s)

def test_empty_series_returns_stats_and_defaults():
    empty = pd.Series([], dtype=float, name="empty_series")
    meta = plot_distribution_ecdf(empty)
    stats = meta['descriptive_stats']
    chart = meta['chart_metadata']

    # Empty data stats
    assert stats['n'] == 0
    assert stats['n_unique'] == 0
    assert stats['gaps'] == []
    # Other numeric stats should be NaN
    for key in ['max_gap','median_gap','pct10_gap','pct50_gap','pct90_gap','total_gap_prop','max_gap_loc']:
        assert np.isnan(stats[key])
    assert stats['n_gaps_above_thr'] == 0 or stats['n_gaps_above_thr'] is None

    # Chart metadata defaults
    assert chart['relative_path'] is None
    assert chart['threshold'] is None
    assert chart['data_source'] is None
    assert chart['title'] == "ECDF with Gap Analysis"
