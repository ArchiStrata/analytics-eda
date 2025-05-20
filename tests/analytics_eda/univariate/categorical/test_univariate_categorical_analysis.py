import json
import pandas as pd
import pytest
from pathlib import Path

from analytics_eda.univariate.categorical.univariate_categorical_analysis import univariate_categorical_analysis

@pytest.fixture
def tmp_report_root(tmp_path):
    return tmp_path / "reports"

@pytest.fixture
def simple_series():
    """A named categorical series with no missing values."""
    return pd.Series(
        ["a", "b", "a", "c", "a", "b", "d", "d", "d", "e"],
        dtype="category",
        name="test_col"
    )

def load_json(path: Path):
    with path.open() as f:
        return json.load(f)

def test_success_return_and_file_written(simple_series, tmp_report_root):
    """
    Basic smoke test: 
    - Function returns a dict with the right top‐level keys.
    - A JSON report file is created with the same content.
    """
    report_file_path = univariate_categorical_analysis(
        series=simple_series,
        top_n=5,
        report_root=str(tmp_report_root),
        rare_threshold=0.2,  # so 'e' is rare at 1/10 = 0.1
        alpha=0.05
    )

    # JSON file should exist
    out_path = tmp_report_root / "test_col" / "test_col_univariate_analysis_report.json"

    assert report_file_path == out_path
    assert report_file_path.exists()

    result = load_json(report_file_path)

    # returned structure
    assert isinstance(result, dict)

    # assert metadata
    assert 'metadata' in result
    for key in ("version", "report_name", "parameters"):
        assert key in result['metadata']

    assert 'eda' in result
    for key in ("missing_data", "distribution", "outliers", "inferential"):
        assert key in result['eda']

    # outliers should list 'e' only
    assert result["eda"]["outliers"]["rare_categories"] == ["c", "e"]


def test_missing_data_counts(simple_series, tmp_report_root):
    """
    Insert some missing values and verify missing_data section.
    """
    s = simple_series.copy()
    s.iloc[0] = None
    s.iloc[3] = None  # now 2 missing of 10
    report_file_path = univariate_categorical_analysis(
        series=s,
        top_n=3,
        report_root=str(tmp_report_root),
        rare_threshold=0.0,  # make none rare
        alpha=0.05
    )

    result = load_json(report_file_path)

    md = result["eda"]["missing_data"]
    assert md["total"] == 10
    assert md["missing"] == 2
    assert pytest.approx(md["pct_missing"], rel=1e-3) == 0.2

def test_inferential_keys_and_types(simple_series, tmp_report_root):
    """
    Verify that the inferential section has the expected numeric keys.
    """
    report_file_path = univariate_categorical_analysis(
        series=simple_series,
        top_n=4,
        report_root=str(tmp_report_root),
        rare_threshold=0.0,
        alpha=0.01
    )

    result = load_json(report_file_path)

    assert "inferential" in result["eda"]
    inferential = result["eda"]["inferential"]

    assert "goodness_of_fit" in inferential
    goodness_of_fit = inferential["goodness_of_fit"]
    
    # should contain these exact keys
    expected_keys = {"chi2_statistic", "p_value", "alpha", "reject_null_uniform"}
    assert set(goodness_of_fit) == expected_keys
    # types
    assert isinstance(goodness_of_fit["chi2_statistic"], float)
    assert isinstance(goodness_of_fit["p_value"], float)
    assert goodness_of_fit["alpha"] == 0.01
    assert isinstance(goodness_of_fit["reject_null_uniform"], bool)

def test_error_on_non_series(tmp_report_root):
    """Passing a non-Series should raise a TypeError."""
    with pytest.raises(TypeError):
        univariate_categorical_analysis(
            series=[1,2,3],
            report_root=str(tmp_report_root)
        )

def test_error_on_unnamed_series(tmp_report_root):
    """Passing a Series without a name should raise ValueError."""
    s = pd.Series(["x","y","z"], dtype="category")  # no name
    with pytest.raises(ValueError):
        univariate_categorical_analysis(
            series=s,
            report_root=str(tmp_report_root)
        )
