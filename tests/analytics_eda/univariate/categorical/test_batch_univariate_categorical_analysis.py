import pandas as pd
import pytest

from analytics_eda.univariate.categorical.batch_univariate_categorical_analysis import batch_univariate_categorical_analysis

def test_batch_univariate_categorical_analysis_success(tmp_path):
    # Create a simple DataFrame with two categorical columns
    df = pd.DataFrame({
        'col1': pd.Categorical(['a', 'b', 'a', 'c']),
        'col2': pd.Categorical(['x', 'y', 'x', 'z']),
    })
    report_root = tmp_path / "reports"

    # Run batch analysis
    result = batch_univariate_categorical_analysis(
        df,
        columns=['col1', 'col2'],
        top_n=2,
        report_root=str(report_root)
    )

    # Should return one entry per column
    assert set(result.keys()) == {'col1', 'col2'}

    for col in ['col1', 'col2']:
        actual_report_file_path = result[col]

        # Output file should have been written
        report_file = report_root / col / f"{col}_univariate_analysis_report.json"
        assert actual_report_file_path == report_file, f"Actual report file path doesn't match expected"
        assert actual_report_file_path.exists(), f"Report file for {col} not found"

def test_batch_error_on_non_categorical(tmp_path):
    # DataFrame with one numeric column should trigger error
    df = pd.DataFrame({
        'cat': pd.Categorical(['a', 'b', 'a']),
        'num': [1, 2, 3],
    })
    report_root = tmp_path / "reports"

    with pytest.raises(TypeError):
        batch_univariate_categorical_analysis(
            df,
            columns=['cat', 'num'],
            report_root=str(report_root)
        )
