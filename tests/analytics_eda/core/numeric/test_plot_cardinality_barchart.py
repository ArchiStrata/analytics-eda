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

def test_default_is_discrete_low_cardinality():
    # 4 unique ints repeated, default thresholds -> discrete
    series = pd.Series([1, 2, 3, 4] * 10, name="nums")
    meta = plot_cardinality_barchart(series)
    assert meta['descriptive_stats']['is_discrete'] is True

def test_default_is_not_discrete_high_cardinality():
    # 100 unique ints, default thresholds -> not discrete
    series = pd.Series(range(100), name="nums")
    meta = plot_cardinality_barchart(series)
    assert meta['descriptive_stats']['is_discrete'] is False

def test_override_max_unique_fraction(tmp_path):
    # high-cardinality series but override fraction to 1.0 -> discrete
    series = pd.Series(range(100), name="nums")
    meta = plot_cardinality_barchart(
        series,
        max_unique_fraction=1.0,
        save_path=str(tmp_path),
        file_name="frac.png"
    )
    stats = meta['descriptive_stats']
    chart = meta['chart_metadata']

    assert stats['is_discrete'] is True
    saved = tmp_path / "frac.png"
    assert saved.exists()
    assert os.path.basename(chart['relative_path']) == "frac.png"

def test_override_max_unique_values(tmp_path):
    # 25 unique ints, default fraction fails but override max_unique_values=30 -> discrete
    series = pd.Series(range(25), name="nums")
    meta = plot_cardinality_barchart(
        series,
        max_unique_values=30,
        save_path=str(tmp_path),
        file_name="uniq.png"
    )
    stats = meta['descriptive_stats']
    chart = meta['chart_metadata']

    assert stats['is_discrete'] is True
    saved = tmp_path / "uniq.png"
    assert saved.exists()
    assert os.path.basename(chart['relative_path']) == "uniq.png"

def test_integer_tolerance_paths_for_high_cardinality_floats(tmp_path):
    # Create a float series slightly off whole numbers,
    # with length > default max_unique_values (20) so low-cardinality
    # logic does NOT trigger.
    n = 30
    offset = 1e-6
    series = pd.Series([i + offset for i in range(n)], name="floats")

    # 1) default tolerance=1e-8: 
    #    s % 1 = offset (1e-6) is NOT within 1e-8 → path 2a False
    #    nunique=30 >= max_unique_values=20 → path 2b False
    #    → overall is_discrete=False
    meta_def = plot_cardinality_barchart(
        series,
        save_path=str(tmp_path),
        file_name="flt_def.png"
    )
    assert meta_def['descriptive_stats']['is_discrete'] is False

    # 2) override tolerance to 1e-5:
    #    now np.isclose(offset, 0, atol=1e-5) → True → path 2a True
    meta_tol = plot_cardinality_barchart(
        series,
        integer_tolerance=1e-5,
        save_path=str(tmp_path),
        file_name="flt_tol.png"
    )
    assert meta_tol['descriptive_stats']['is_discrete'] is True

    # confirm file save for the override case
    saved = tmp_path / "flt_tol.png"
    assert saved.exists() and saved.stat().st_size > 0
    assert os.path.basename(meta_tol['chart_metadata']['relative_path']) == "flt_tol.png"

def test_float_low_cardinality_discrete_path_2b(tmp_path):
    # Floats that are not integer‐like, but with low cardinality (unique < max_unique_values)
    series = pd.Series([i + 0.1 for i in range(10)], name="floats")
    # default max_unique_values=20 → 10 < 20 triggers path 2b
    meta = plot_cardinality_barchart(
        series,
        save_path=str(tmp_path),
        file_name="low_card.png"
    )
    stats = meta['descriptive_stats']
    chart = meta['chart_metadata']

    # 2b low‐cardinality should yield True
    assert stats['is_discrete'] is True

    # confirm file was saved
    saved = tmp_path / "low_card.png"
    assert saved.exists() and saved.stat().st_size > 0
    assert os.path.basename(chart['relative_path']) == "low_card.png"
