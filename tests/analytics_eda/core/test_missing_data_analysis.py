import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from analytics_eda.core import missing_data_analysis

def test_missing_data_analysis_saves_plot(tmp_path):
    # Series with three present values and two missing
    series = pd.Series([1, 1, 1, np.nan, np.nan], name='test_series')
    report_dir = tmp_path

    result = missing_data_analysis(series, report_dir)

    # Validate summary keys
    assert set(result.keys()) == {'total', 'missing', 'pct_missing', 'missing_data_barplot'}

    # Validate numeric summary
    assert result['total'] == 5
    assert result['missing'] == 2
    assert pytest.approx(result['pct_missing'], rel=1e-3) == 2/5

    # Validate that the PNG was written
    plot_path = Path(result['missing_data_barplot'])
    assert plot_path.exists(), f"Expected plot at {plot_path}"
    assert plot_path.suffix == '.png'
    assert plot_path.stat().st_size > 0  # non-empty file

def test_missing_data_analysis_all_zero_sum(tmp_path):
    # Series sums to zero -> total=3, missing=0, pct_missing=0.0
    series = pd.Series([0, 0, 0], name='zero_series')
    report_dir = tmp_path

    result = missing_data_analysis(series, report_dir)

    assert result['total'] == 3
    assert result['missing'] == 0
    assert result['pct_missing'] == 0.0

    plot_path = Path(result['missing_data_barplot'])
    assert plot_path.exists()
    assert plot_path.suffix == '.png'
    assert plot_path.stat().st_size > 0

def test_missing_data_analysis_two_third_missing(tmp_path):
    # Series with 1 present value and two missing
    series = pd.Series([1, np.nan, np.nan], name='test_series')
    report_dir = tmp_path

    result = missing_data_analysis(series, report_dir)

    # Validate summary keys
    assert set(result.keys()) == {'total', 'missing', 'pct_missing', 'missing_data_barplot'}

    # Validate numeric summary
    assert result['total'] == 3
    assert result['missing'] == 2
    assert pytest.approx(result['pct_missing'], rel=1e-3) == result['missing']/result['total']

    # Validate that the PNG was written
    plot_path = Path(result['missing_data_barplot'])
    assert plot_path.exists(), f"Expected plot at {plot_path}"
    assert plot_path.suffix == '.png'
    assert plot_path.stat().st_size > 0  # non-empty file