import pytest
import pandas as pd
import json
import os
from tempfile import TemporaryDirectory
from analytics_eda.core import explore_data


@pytest.fixture
def sample_dataframe():
    df = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "age": [25, 30, 35, None, 40],
        "gender": pd.Series(["F", "M", "M", "M", "F"], dtype="category"),
        "constant": ["yes"] * 5
    })
    return df


def test_explore_data_returns_expected_keys(sample_dataframe):
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "summary.json")
        summary = explore_data(sample_dataframe, report_path=path)

        expected_keys = {
            "shape", "columns", "dtypes", "missing_values", "duplicate_count",
            "constant_columns", "high_cardinality_columns", "statistical_summary",
            "object_unique_counts", "category_unique_details", "unique_values_preview",
            "memory_usage_bytes", "sample_data"
        }
        assert expected_keys.issubset(summary.keys())


def test_explore_data_creates_json_file(sample_dataframe):
    with TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "eda_summary.json")
        explore_data(sample_dataframe, report_path=path)

        assert os.path.exists(path)

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert isinstance(data, dict)
        assert "shape" in data
        assert "sample_data" in data
