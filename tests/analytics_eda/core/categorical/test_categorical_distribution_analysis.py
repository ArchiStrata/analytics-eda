import pytest
import pandas as pd
from pathlib import Path

from analytics_eda.core.categorical.categorical_distribution_analysis import categorical_distribution_analysis

def test_categorical_distribution_analysis_basic(tmp_path):
    # Setup: 3 items, 2 unique categories, dtype=category
    series = pd.Series(['a', 'b', 'a'], dtype='category', name='test_series')
    save_dir = tmp_path / "plots"
    
    result = categorical_distribution_analysis(series, save_dir)
    assert 'report' in result
    report = result['report']
    
    # Statistics
    stats = report['statistics']
    assert stats['cardinality'] == 2
    assert pytest.approx(stats['imbalance_ratio'], rel=1e-6) == 2.0
    # Category length stats
    max_len = max(len('a'), len('b'))
    min_len = min(len('a'), len('b'))
    assert stats['category_length_stats']['max_length'] == max_len
    assert stats['category_length_stats']['min_length'] == min_len

    # Distribution / Frequency table
    freq_report = report['frequency_report']
    freq_table = freq_report['frequency_table']
    assert freq_table['a']['count'] == 2
    assert pytest.approx(freq_table['a']['proportion'], rel=1e-6) == 2/3
    assert freq_table['b']['count'] == 1
    assert pytest.approx(freq_table['b']['proportion'], rel=1e-6) == 1/3

    # Visualization: top-n plot file exists
    viz = freq_report['visualizations']
    plot_path = Path(viz['top_n_plot'])
    assert plot_path.exists()
    assert plot_path.suffix == '.png'

def test_top_n_adjustment(tmp_path):
    # 4 unique categories, default top_n=10 → adjusted to max(1,4//2)=2
    series = pd.Series(['w','x','y','z'], dtype='category', name='letters')
    save_dir = tmp_path / "plots2"
    result = categorical_distribution_analysis(series, save_dir, top_n=10)
    
    viz = result['report']['frequency_report']['visualizations']
    plot_path = Path(viz['top_n_plot'])
    # filename should reflect the adjusted top_n = 2
    assert '_top_2' in plot_path.name

def test_requires_series_and_name(tmp_path):
    # Not a Series
    with pytest.raises(TypeError):
        categorical_distribution_analysis(['a', 'b'], tmp_path)
    # Series without name
    s = pd.Series(['a','b'], dtype='category')
    with pytest.raises(ValueError):
        categorical_distribution_analysis(s, tmp_path)

def test_requires_categorical_or_object_dtype(tmp_path):
    # Integer dtype is invalid
    s = pd.Series([1,2,3], name='nums')
    with pytest.raises(TypeError):
        categorical_distribution_analysis(s, tmp_path)
