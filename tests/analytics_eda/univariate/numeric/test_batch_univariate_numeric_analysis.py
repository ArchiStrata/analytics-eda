import pytest
import numpy as np
import pandas as pd
import json

from analytics_eda.univariate.numeric.batch_univariate_numeric_analysis import batch_univariate_numeric_analysis

@pytest.fixture
def df_normal():
    """DataFrame with one column of 15 normally distributed values."""
    rng = np.random.default_rng(0)
    data = rng.normal(loc=0.0, scale=1.0, size=15)
    return pd.DataFrame({'norm': data})

def test_batch_univariate_numeric_analysis_single_column(tmp_path, df_normal):
    # Arrange
    report_root = tmp_path / "reports"
    
    # Act
    result = batch_univariate_numeric_analysis(
        df_normal,
        columns=['norm'],
        report_root=str(report_root),
        alpha=0.05
    )

    assert "norm" in result
    actual_report_file_path = result["norm"]

    # Assert JSON report file was created and contains all sections
    assert actual_report_file_path.exists(), "Expected JSON report file for 'norm' column"

    report_dir = report_root / 'norm'
    report_file = report_dir / 'norm_univariate_analysis_report.json'
    assert actual_report_file_path == report_file

    result = json.loads(actual_report_file_path.read_text())

    # Assert - top‐level mapping
    assert isinstance(result, dict)
    assert 'eda' in result

    expected_sections = {'missing_data', 'distribution', 'outliers', 'inferential'}
    assert set(result['eda'].keys()) == expected_sections
