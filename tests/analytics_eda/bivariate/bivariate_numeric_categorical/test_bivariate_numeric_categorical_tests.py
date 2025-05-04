import pytest
import numpy as np
import pandas as pd
from analytics_eda.bivariate.bivariate_numeric_categorical.bivariate_numeric_categorical_tests import bivariate_numeric_categorical_tests

@pytest.fixture
def df_identical():
    # Two identical groups A and B
    values = np.array([0, 1, 2, 3, 4], dtype=float)
    return pd.DataFrame({
        'group': ['A']*5 + ['B']*5,
        'value': np.concatenate([values, values])
    })

@pytest.fixture
def df_separated():
    # Two well-separated groups A and B
    return pd.DataFrame({
        'group': ['A']*3 + ['B']*3,
        'value': np.array([0, 1, 2, 100, 101, 102], dtype=float)
    })

@pytest.fixture
def df_three():
    # Three overlapping groups A, B, C
    return pd.DataFrame({
        'group': ['A']*3 + ['B']*3 + ['C']*3,
        'value': np.array([0, 1, 2, 1, 2, 3, 2, 3, 4], dtype=float)
    })

def test_single_group_error():
    df = pd.DataFrame({'group': ['A']*3, 'value': [1, 2, 3]})
    result = bivariate_numeric_categorical_tests(df, 'value', 'group')
    assert 'error' in result
    assert 'Not enough groups' in result['error']

def test_identical_groups(df_identical):
    result = bivariate_numeric_categorical_tests(df_identical, 'value', 'group')
    # Metadata
    assert result['meta']['n_groups'] == 2
    assert set(result['meta']['group_sizes']) == {5}
    # Distribution overlap
    overlap = result['distribution_overlap']['A vs B']
    assert overlap['overlap_coeff'] > 0.7
    assert overlap['bhattacharyya_dist'] < 1.0
    # Global tests non-significant
    assert result['anova']['reject'] is False
    assert result['kruskal']['reject'] is False
    # No post-hoc needed
    assert 'tukey_hsd' not in result or not result['anova']['reject']
    # Effect sizes near zero
    es = result['effect_size']
    assert pytest.approx(es['eta_squared'], abs=0.05) == 0.0
    assert es['omega_squared'] <= 0.0
    assert es['epsilon_squared'] <= 0.0

def test_separated_groups(df_separated):
    result = bivariate_numeric_categorical_tests(df_separated, 'value', 'group')
    # Significant global tests
    assert result['anova']['reject'] is True
    assert result['kruskal']['reject'] is True
    # Post-hoc present
    assert 'tukey_hsd' in result
    pairs = result['tukey_hsd']['pairs']
    assert any(p['group1']=='A' and p['group2']=='B' for p in pairs)
    # Effect sizes positive
    es = result['effect_size']
    assert es['eta_squared'] > 0.5
    assert es['omega_squared'] > 0.5
    assert es['epsilon_squared'] > 0.5

def test_three_groups_structure(df_three):
    result = bivariate_numeric_categorical_tests(df_three, 'value', 'group')

    # Mandatory keys that should always be present
    mandatory = {
        'meta', 'distribution_overlap', 'bartlett', 'levene',
        'anova', 'kruskal', 'effect_size'
    }
    assert mandatory.issubset(result.keys())

    # Tukey’s HSD only if ANOVA is significant
    if result['anova']['reject']:
        assert 'tukey_hsd' in result
    else:
        assert 'tukey_hsd' not in result

    # Distribution overlap pairs
    overlap_keys = set(result['distribution_overlap'].keys())
    assert overlap_keys == {'A vs B', 'A vs C', 'B vs C'}

    # Metadata group count
    assert result['meta']['n_groups'] == 3
