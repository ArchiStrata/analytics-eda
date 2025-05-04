import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from analytics_eda.core.numeric.numeric_distribution_analysis import numeric_distribution_analysis

@pytest.fixture
def normal_15_distribution_series():
    """Generate a small (<20) reproducible normal-distributed Series."""
    rng = np.random.default_rng(123)
    data = rng.normal(loc=0, scale=1, size=15)
    return pd.Series(data, name="norm_test")

@pytest.fixture
def skewed_sqrt_15_series():
    """Generate a small (<20) positively skewed Series to trigger sqrt."""
    rng = np.random.default_rng(0)
    data = rng.exponential(scale=1.0, size=15)
    return pd.Series(data, name="skewed")

def test_distribution_analysis_on_normal_15_series(tmp_path, normal_15_distribution_series):
    # Arrange
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    
    # Act
    result = numeric_distribution_analysis(normal_15_distribution_series, report_dir, alpha=0.05)
    
    # Assert top-level structure
    assert isinstance(result, dict)
    assert 'report' in result and 'series' in result
    assert isinstance(result['series'], pd.Series)
    # Series should be unchanged when no transform is applied
    assert result['series'].equals(normal_15_distribution_series)
    
    report = result['report']
    
    # 1. Descriptive statistics
    stats = report['statistics']
    assert stats['count'] == len(normal_15_distribution_series)
    assert 'mean' in stats and isinstance(stats['mean'], float)
    assert not stats.get('is_discrete', False)
    
    # 2. Normality report
    normality = report['normality_report']
    assert 'assessment' in normality and isinstance(normality['assessment'], dict)
    viz = normality['visualizations']
    assert isinstance(viz, dict)
    expected_plots = {"hist_counts", "hist_kde", "boxplot", "ecdf", "qq_plot"}
    assert set(viz.keys()) == expected_plots
    for key, path_str in viz.items():
        path = Path(path_str)
        assert path.exists(), f"Expected {key} plot to exist"
        # File should be non-empty
        assert path.stat().st_size > 0
    
    # 3. Transformation report (no transform expected)
    transform_report = report['transformation_report']
    assert 'assessment' in transform_report and isinstance(transform_report['assessment'], dict)
    # Since data is normal, no best_transform should be set -> visualizations None
    assert transform_report['visualizations'] is None
    
    # 4. Distribution fit report (no alternative fits when normality holds)
    fit_report = report['distribution_fit_report']
    assert 'assessment' in fit_report
    assert fit_report['assessment'] == {}

def test_distribution_analysis_sqrt_transform(tmp_path, skewed_sqrt_15_series):
    # Arrange
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    
    # Act
    result = numeric_distribution_analysis(skewed_sqrt_15_series, report_dir, alpha=0.05)
    
    # Assert top‐level structure
    assert 'report' in result and 'series' in result
    transformed = result['series']
    assert isinstance(transformed, pd.Series)
    assert transformed.name == skewed_sqrt_15_series.name
    # The series should change when sqrt is applied
    assert not transformed.equals(skewed_sqrt_15_series)
    
    report = result['report']
    
    # 1. Check that normality was initially rejected
    normality_assess = report['normality_report']['assessment']
    assert normality_assess.get('reject_normality', False), "Expected initial normality rejection"
    
    # 2. Transformation report
    trans_report = report['transformation_report']
    assessment = trans_report['assessment']
    # Box–Cox should be chosen
    assert assessment['needs_transform'] is True
    assert assessment['best_transform'] == 'sqrt'
    
    # Visualizations for the transformed data should be generated
    viz = trans_report['visualizations']
    assert isinstance(viz, dict) and viz, "Expected transform visualizations"
    expected_keys = {"hist_counts", "hist_kde", "boxplot", "ecdf", "qq_plot"}
    assert set(viz.keys()) == expected_keys
    
    # Each visualization file must exist and be non‐empty
    for key, path_str in viz.items():
        path = Path(path_str)
        assert path.exists(), f"{key} plot was not created"
        assert path.parent == report_dir
        assert path.stat().st_size > 0, f"{key} plot file is empty"
    
    # 3. Alternative fits for sqrt distribution
    fit_report = report['distribution_fit_report']['assessment']
    assert 'alternative_fits' in fit_report
