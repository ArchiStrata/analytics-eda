import numpy as np
from analytics_eda.bivariate.bivariate_numeric_categorical.compute_overlap_metrics import compute_overlap_metrics

def test_identical_distributions():
    """Identical groups should yield high overlap and low distance despite KDE smoothing."""
    group = np.array([0, 1, 2, 3, 4], dtype=float)
    data = {'A': group, 'B': group.copy()}
    result = compute_overlap_metrics(data)

    assert 'A vs B' in result
    metrics = result['A vs B']

    # KDE smoothing may reduce overlap_coef below 1.0, but it should remain high
    assert metrics['overlap_coeff'] > 0.7
    # Bhattacharyya distance should be modest for identical data
    assert metrics['bhattacharyya_dist'] < 1.0

def test_non_overlapping_distributions():
    """Well-separated groups should have near-zero overlap and positive distance."""
    data = {
        'A': np.array([0, 1, 2], dtype=float),
        'B': np.array([100, 101, 102], dtype=float)
    }
    result = compute_overlap_metrics(data)
    metrics = result['A vs B']

    assert metrics['overlap_coeff'] < 1e-3
    assert metrics['bhattacharyya_dist'] > 1.0

def test_three_groups_pairwise_keys():
    """Ensure all unique pairwise combinations are present for three groups."""
    data = {
        'A': np.array([0, 1, 2], dtype=float),
        'B': np.array([1, 2, 3], dtype=float),
        'C': np.array([2, 3, 4], dtype=float)
    }
    result = compute_overlap_metrics(data)
    expected = {'A vs B', 'A vs C', 'B vs C'}
    assert set(result.keys()) == expected

def test_single_group_returns_empty():
    """A single group should yield no pairwise metrics."""
    data = {'A': np.array([1, 2, 3], dtype=float)}
    result = compute_overlap_metrics(data)
    assert result == {}
