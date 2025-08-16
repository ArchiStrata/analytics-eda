import pytest
import pandas as pd
import json

from analytics_eda.analysis.bivariate.categorical_numeric_relationship_analysis.categorical_numeric_relationship_analysis import categorical_numeric_relationship_analysis

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'category': ['A', 'A', 'B', 'B'],
        'value': [1.0, 2.0, 3.0, 4.0]
    })

def test_missing_categorical_column(sample_df):
    df = sample_df.drop(columns=['category'])
    with pytest.raises(KeyError) as exc:
        categorical_numeric_relationship_analysis(df, 'value', 'category')
    assert "Categorical column 'category' not found." in str(exc.value)

def test_missing_numeric_column(sample_df):
    df = sample_df.drop(columns=['value'])
    with pytest.raises(KeyError) as exc:
        categorical_numeric_relationship_analysis(df, 'value', 'category')
    assert "Numeric column 'value' not found." in str(exc.value)

def test_invalid_categorical_dtype(sample_df):
    # category as numeric dtype should fail
    df = sample_df.copy()
    df['category'] = df['category'].map({'A': 1, 'B': 2})
    with pytest.raises(TypeError) as exc:
        categorical_numeric_relationship_analysis(df, 'value', 'category')
    assert "must be categorical or object" in str(exc.value)

def test_invalid_numeric_dtype(sample_df):
    # value as object dtype should fail
    df = sample_df.copy()
    df['value'] = df['value'].astype(str)
    with pytest.raises(TypeError) as exc:
        categorical_numeric_relationship_analysis(df, 'value', 'category')
    assert "must be numeric" in str(exc.value)

def test_integration_creates_report(tmp_path, caplog, sample_df):
    caplog.set_level("DEBUG")

    # Define a temporary report root
    report_root = tmp_path / "reports"

    # Run the analysis; function returns a Path
    numeric_col = 'value'
    categorical_col = 'category'
    response = categorical_numeric_relationship_analysis(
        sample_df,
        numeric_col=numeric_col,
        categorical_col=categorical_col,
        report_root=str(report_root)
    )

    report_file_path = response['report_file_path']

    # 1. Assert the JSON report file exists
    assert report_file_path.exists(), f"Expected report at {report_file_path}, but not found."

    # 2. Assert the returned path matches the expected structure
    expected_dir = report_root / f"categorical_{categorical_col}_numeric_{numeric_col}_relationship_analysis"
    expected_file = expected_dir / f"categorical_{categorical_col}_numeric_{numeric_col}_relationship_analysis_report.json"
    assert report_file_path == expected_file

    # 3. Load and verify JSON structure
    loaded = json.loads(report_file_path.read_text())

    assert 'metadata' in loaded
    assert 'version' in loaded['metadata']
    assert 'report_name' in loaded['metadata']
    assert 'parameters' in loaded['metadata']

    assert 'data' in loaded
    eda_report = loaded['data']

    assert 'relationship_structure' in eda_report, "Missing 'relationship_structure' in report"

    actual_relationship_structure = eda_report['relationship_structure']
    assert 'numeric_distribution_by_category' in actual_relationship_structure, "Missing 'numeric_distribution_by_category' in relationship_structure"

    actual_numeric_distribution_by_category = actual_relationship_structure['numeric_distribution_by_category']
    # Check that each category appears
    assert set(actual_numeric_distribution_by_category.keys()) == {'A', 'B'}

    assert 'magnitude_of_association' in eda_report, "Missing 'magnitude_of_association' in report"

    assert 'direction_of_association' in eda_report, "Missing 'direction_of_association' in report"

    # Logging assertions
    records = caplog.records

    # a. Start log
    start_log = next((r for r in records if "Starting categorical_numeric_relationship_analysis" in r.message), None)
    assert start_log is not None
    assert start_log.numeric_col == 'value'
    assert start_log.categorical_col == 'category'
    assert start_log.report_root == str(report_root)
    assert hasattr(start_log, 'report_log_id')

    # b. Category logs
    category_logs = [r for r in records if r.message == "Running univariate analysis for category"]
    assert len(category_logs) == 2  # Expect 2 categories: A and B
    for r in category_logs:
        assert r.numeric_col == 'value'
        assert r.categorical_col == 'category'
        assert hasattr(r, 'category')
        assert hasattr(r, 'report_log_id')

    # c. Completion log
    complete_log = next((r for r in records if "Completed categorical_numeric_relationship_analysis" in r.message), None)
    assert complete_log is not None
    assert complete_log.numeric_col == 'value'
    assert complete_log.categorical_col == 'category'
    assert complete_log.report_file_path == str(report_file_path)
    assert hasattr(complete_log, 'report_log_id')
