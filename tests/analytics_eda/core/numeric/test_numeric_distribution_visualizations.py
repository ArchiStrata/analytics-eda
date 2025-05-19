import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from analytics_eda.core.numeric.numeric_distribution_visualizations import numeric_distribution_visualizations

@pytest.fixture
def normal_distribution_series():
    # Generate a reproducible normal distribution
    rng = np.random.default_rng(42)
    data = rng.normal(loc=0, scale=1, size=100)
    return pd.Series(data, name="test_series")

def test_normal_distribution_visualizations_creates_plots(tmp_path, normal_distribution_series):
    # Arrange
    report_dir = tmp_path / "plots"
    report_dir.mkdir()
    transform_label = "raw"
    
    # Act
    viz_paths = numeric_distribution_visualizations(normal_distribution_series, report_dir, transform=transform_label)
    
    # Assert - correct set of visualization keys
    expected_keys = {"hist_counts", "hist_kde", "boxplot", "ecdf", "qq_plot"}
    assert set(viz_paths.keys()) == expected_keys
    
    # Assert - each file exists and is named as expected
    suffix_map = {
        "hist_counts": "hist_counts",
        "hist_kde": "hist_kde",
        "boxplot": "boxplot",
        "ecdf": "ecdf",
        "qq_plot": "qq_plot",
    }
    for key, path_str in viz_paths.items():
        path = Path(path_str)
        assert path.exists(), f"Expected file for {key} to be created"
        assert path.parent == report_dir
        expected_filename = f"{normal_distribution_series.name}_{transform_label}_{suffix_map[key]}.png"
        assert path.name == expected_filename
        assert path.stat().st_size > 0, f"Plot file for '{key}' should not be empty"

def test_distribution_visualizations_empty_series(tmp_path):
    # Arrange
    empty_series = pd.Series([None, None], dtype="float", name="empty")
    report_dir = tmp_path / "plots"
    report_dir.mkdir()
    
    # Act
    viz_paths = numeric_distribution_visualizations(empty_series, report_dir)
    
    # Assert
    assert viz_paths == {}
