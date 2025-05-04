import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path

from analytics_eda.univariate.numeric.univariate_numeric_analysis import univariate_numeric_analysis

@pytest.fixture
def normal_15_series():
    """A small (<20) reproducible normal-distributed Series."""
    rng = np.random.default_rng(0)
    data = rng.normal(loc=0.0, scale=1.0, size=15)
    return pd.Series(data, name="normal_series")

def test_univariate_numeric_analysis_on_normal_series(tmp_path, normal_15_series):
    # Arrange
    report_root = tmp_path / "reports"
    
    # Act
    actual_report_file = univariate_numeric_analysis(
        normal_15_series,
        report_root=str(report_root),
        alpha=0.05
    )

    # Assert JSON report file was written and matches the result
    assert actual_report_file.exists(), "Expected JSON report file to be created"
    report_dir = Path(report_root) / normal_15_series.name
    report_file = report_dir / f"{normal_15_series.name}_univariate_analysis_report.json"
    assert actual_report_file == report_file
    
    result = json.loads(actual_report_file.read_text())
    
    # Assert top-level keys in returned dict
    assert isinstance(result, dict)
    expected_top = {'missing_data', 'distribution', 'outliers', 'inferential'}
    assert set(result.keys()) == expected_top
    
    # Inspect the distribution sub-report
    dist = result['distribution']
    expected_dist_keys = {
        'statistics',
        'normality_report',
        'transformation_report',
        'distribution_fit_report'
    }
    assert set(dist.keys()) == expected_dist_keys

