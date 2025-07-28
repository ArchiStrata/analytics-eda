import os
import pandas as pd
import pytest

from analytics_eda.core.numeric import plot_cardinality_barchart

def test_default_parameters_no_save():
    # a series with 5 distinct values
    series = pd.Series([1, 2, 2, 3, 4, 4, 5], name="nums")
    meta = plot_cardinality_barchart(series)
    stats = meta['descriptive_stats']
    chart = meta['chart_metadata']

    # descriptive_stats
    assert stats['nunique'] == 5

    # chart_metadata defaults
    assert chart['title'] == "Value Counts (Top k) for Cardinality"
    assert chart['xlabel'] == "Value"
    assert chart['ylabel'] == "Count"
    assert chart['data_source'] is None
    assert chart['top_k'] == 10
    assert chart['relative_path'] is None

def test_override_and_save(tmp_path):
    # series with known values
    series = pd.Series([10, 20, 20, 30, 30, 30], name="vals")
    custom_title   = "Top 3 Frequencies"
    custom_xlabel  = "Category"
    custom_ylabel  = "Frequency"
    custom_source  = "UnitTest"
    top_k          = 3
    filename       = "card.png"

    meta = plot_cardinality_barchart(
        series,
        top_k=top_k,
        title=custom_title,
        xlabel=custom_xlabel,
        ylabel=custom_ylabel,
        data_source=custom_source,
        save_path=str(tmp_path),
        file_name=filename
    )
    stats = meta['descriptive_stats']
    chart = meta['chart_metadata']

    # descriptive_stats
    assert stats['nunique'] == 3  # values 10,20,30

    # chart_metadata overrides
    assert chart['title']       == custom_title
    assert chart['xlabel']      == custom_xlabel
    assert chart['ylabel']      == custom_ylabel
    assert chart['data_source'] == custom_source
    assert chart['top_k']       == top_k

    # file was saved correctly
    saved = tmp_path / filename
    assert saved.exists() and saved.stat().st_size > 0
    with open(saved, 'rb') as f:
        sig = f.read(8)
    assert sig == b'\x89PNG\r\n\x1a\n'
    # relative_path ends with filename
    assert os.path.basename(chart['relative_path']) == filename

def test_save_defaults_and_metadata(tmp_path):
    series = pd.Series(range(5), name="range")
    filename = "out.png"
    meta = plot_cardinality_barchart(
        series,
        save_path=str(tmp_path),
        file_name=filename
    )
    chart = meta['chart_metadata']

    # defaults preserved
    assert chart['title']   == "Value Counts (Top k) for Cardinality"
    assert chart['xlabel']  == "Value"
    assert chart['ylabel']  == "Count"
    assert chart['data_source'] is None
    assert chart['top_k']   == 10

    # file exists and non-empty
    saved = tmp_path / filename
    assert saved.exists()
    assert saved.stat().st_size > 0

def test_missing_series_name_raises_error():
    # series without a name
    unnamed = pd.Series([1, 2, 3])
    with pytest.raises(ValueError):
        plot_cardinality_barchart(unnamed)

def test_empty_series_returns_stats_and_defaults():
    empty = pd.Series([], dtype=float, name="empty")
    meta = plot_cardinality_barchart(empty)
    stats = meta['descriptive_stats']
    chart = meta['chart_metadata']

    # empty descriptive_stats
    assert stats['nunique'] == 0

    # default chart metadata, no save
    assert chart['relative_path'] is None
    assert chart['data_source'] is None
    assert chart['title'] == "Value Counts (Top k) for Cardinality"
    assert chart['xlabel'] == "Value"
    assert chart['ylabel'] == "Count"
    assert chart['top_k'] == 10
