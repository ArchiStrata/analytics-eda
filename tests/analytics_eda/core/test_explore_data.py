import json
import os

import numpy as np
import pandas as pd
import pytest

from analytics_eda.core import explore_data


@pytest.fixture
def sample_dataframe():
    df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "name": ["Alice", "Bob", "Charlie", "David", "Eve"], "age": [25, 30, 35, None, 40], "gender": pd.Series(["F", "M", "M", "M", "F"], dtype="category"), "constant": ["yes"] * 5})
    return df


def test_explore_data_all_branches(tmp_path):
    # Create test DataFrame to trigger all code paths
    np.random.seed(42)
    base_values = np.random.normal(loc=50, scale=5, size=98)

    # Add extreme outliers: one very large and one very small
    extreme_upper = [1_000_000]
    extreme_lower = [-1_000_000]

    # Combine them
    numeric_var = np.concatenate([base_values, extreme_upper, extreme_lower])

    df = pd.DataFrame(
        {
            "numeric_const": [1.0] * 100,  # Constant numeric
            "numeric_var": numeric_var,  # Numeric with outlier
            "category_var": pd.Series((["A", "B", "C", None] * 25)[:100], dtype="category"),
            "object_high_card": [str(i) for i in range(100)],  # Object high cardinality
            "object_const": ["X"] * 100,  # Constant object
            "mixed_nan": [np.nan] * 10 + [1] * 90,  # Missing values
        }
    )

    summary = explore_data(df, report_path=tmp_path)

    # ---- Overview Checks ----
    overview = summary["overview"]
    assert overview["shape"] == {"rows": 100, "columns": 6}
    assert overview["duplicate_rows"] == 0
    assert isinstance(overview["memory_usage_bytes"], dict)
    assert all(isinstance(v, int) for v in overview["memory_usage_bytes"].values())

    # ---- Column Drill-Down Checks ----
    cols = summary["columns"]

    # Constant numeric
    assert cols["numeric_const"]["is_constant"] is True
    assert "numeric_analysis" in cols["numeric_const"]

    # Variable numeric with outlier
    assert cols["numeric_var"]["is_constant"] is False
    assert cols["numeric_var"]["is_high_cardinality"] is True

    assert "numeric_analysis" in cols["numeric_var"]
    outlier_info = cols["numeric_var"]["numeric_analysis"]["extreme_outliers_4sigma"]

    # Check structure of outlier split
    assert set(outlier_info.keys()) == {"total_count", "lower", "upper"}
    assert isinstance(outlier_info["lower"], dict)
    assert isinstance(outlier_info["upper"], dict)

    # Expecting 1 upper bound outlier (1_000_000) and 1 lower bound outlier (-1_000_000)
    assert outlier_info["total_count"] == 2

    # Lower outliers
    assert outlier_info["lower"]["count"] == 1
    assert outlier_info["lower"]["min"] == float(-1_000_000)
    assert outlier_info["lower"]["max"] == float(-1_000_000)

    # Upper outliers
    assert outlier_info["upper"]["count"] == 1
    assert outlier_info["upper"]["min"] == float(1_000_000)
    assert outlier_info["upper"]["max"] == float(1_000_000)

    # Category
    assert cols["category_var"]["dtype"] == "category"
    assert "category_analysis" in cols["category_var"]
    assert cols["category_var"]["category_analysis"]["unique_count"] == 3

    # Object high cardinality
    assert cols["object_high_card"]["is_high_cardinality"] is True
    assert "object_analysis" in cols["object_high_card"]
    assert cols["object_high_card"]["object_analysis"]["unique_count"] == 100

    # Constant object
    assert cols["object_const"]["is_constant"] is True

    # Column with NaNs
    assert cols["mixed_nan"]["missing_values"] == 10

    # verify report saved
    full_report_path = tmp_path / "explore_data_summary.json"
    assert os.path.exists(full_report_path)

    with open(full_report_path, encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, dict)
    assert "overview" in data
    assert "columns" in data
