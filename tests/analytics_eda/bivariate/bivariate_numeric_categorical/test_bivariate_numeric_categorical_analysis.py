import pytest
import pandas as pd
import json

from analytics_eda.bivariate.bivariate_numeric_categorical.bivariate_numeric_categorical_analysis import bivariate_numeric_categorical_analysis

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'category': ['A', 'A', 'B', 'B'],
        'value': [1.0, 2.0, 3.0, 4.0]
    })

def test_missing_categorical_column(sample_df):
    df = sample_df.drop(columns=['category'])
    with pytest.raises(KeyError) as exc:
        bivariate_numeric_categorical_analysis(df, 'value', 'category')
    assert "Categorical column 'category' not found." in str(exc.value)

def test_missing_numeric_column(sample_df):
    df = sample_df.drop(columns=['value'])
    with pytest.raises(KeyError) as exc:
        bivariate_numeric_categorical_analysis(df, 'value', 'category')
    assert "Numeric column 'value' not found." in str(exc.value)

def test_invalid_categorical_dtype(sample_df):
    # category as numeric dtype should fail
    df = sample_df.copy()
    df['category'] = df['category'].map({'A': 1, 'B': 2})
    with pytest.raises(TypeError) as exc:
        bivariate_numeric_categorical_analysis(df, 'value', 'category')
    assert "must be categorical or object" in str(exc.value)

def test_invalid_numeric_dtype(sample_df):
    # value as object dtype should fail
    df = sample_df.copy()
    df['value'] = df['value'].astype(str)
    with pytest.raises(TypeError) as exc:
        bivariate_numeric_categorical_analysis(df, 'value', 'category')
    assert "must be numeric" in str(exc.value)

def test_integration_creates_report(tmp_path, sample_df):
    # Define a temporary report root
    report_root = tmp_path / "reports"

    # Run the analysis; function returns a Path
    report_path = bivariate_numeric_categorical_analysis(
        sample_df,
        numeric_col='value',
        categorical_col='category',
        report_root=str(report_root)
    )

    # 1. Assert the JSON report file exists
    assert report_path.exists(), f"Expected report at {report_path}, but not found."

    # 2. Assert the returned path matches the expected structure
    expected_dir = report_root / "value_by_category"
    expected_file = expected_dir / "value_by_category_bivariate_analysis_report.json"
    assert report_path == expected_file

    # 3. Load and verify JSON structure
    loaded = json.loads(report_path.read_text())

    assert 'metadata' in loaded
    assert 'version' in loaded['metadata']
    assert 'report_name' in loaded['metadata']
    assert 'parameters' in loaded['metadata']

    assert 'eda' in loaded
    eda_report = loaded['eda']

    assert 'statistical_tests' in eda_report, "Missing 'statistical_tests' in report"
    assert 'segments_report' in eda_report, "Missing 'segments_report' in report"
    # Check that each category appears
    assert set(eda_report['segments_report'].keys()) == {'A', 'B'}
