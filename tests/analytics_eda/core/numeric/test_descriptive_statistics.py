import pytest
import pandas as pd
import numpy as np
from analytics_eda.core.numeric.descriptive_statistics import descriptive_statistics


def test_empty_series_returns_empty():
    s = pd.Series([], dtype=float, name="nums")
    result = descriptive_statistics(s)
    assert result == {}


def test_non_series_input_raises_type_error():
    with pytest.raises(TypeError, match="Input must be a pandas Series."):
        descriptive_statistics([1, 2, 3])


def test_missing_name_raises_value_error():
    s = pd.Series([1, 2, 3], dtype=float)
    with pytest.raises(ValueError, match="Series must have a non-empty 'name'"):
        descriptive_statistics(s)


def test_basic_statistics_without_include_type():
    data = [1, 2, 3, 4, 5]
    s = pd.Series(data, name="nums")
    result = descriptive_statistics(s)
    # Core stats
    assert result['count'] == 5
    assert result['mean'] == pytest.approx(3.0)
    assert result['median'] == pytest.approx(3.0)
    assert result['mode'] == pytest.approx(1.0)
    assert result['min'] == pytest.approx(1.0)
    assert result['max'] == pytest.approx(5.0)
    assert result['range'] == pytest.approx(4.0)
    assert result['nunique'] == 5
    # Percentiles
    assert result['pct_10'] == pytest.approx(np.percentile(data, 10))
    assert result['pct_25'] == pytest.approx(np.percentile(data, 25))
    assert result['pct_75'] == pytest.approx(np.percentile(data, 75))
    assert result['pct_90'] == pytest.approx(np.percentile(data, 90))
    # Standard deviation and variance
    assert result['std'] == pytest.approx(pd.Series(data).std())
    assert result['var'] == pytest.approx(pd.Series(data).var())
    # CV = std/mean
    assert result['cv'] == pytest.approx(result['std'] / result['mean'])
    # No is_discrete key by default
    assert 'is_discrete' not in result


def test_multiple_modes_picks_first():
    s = pd.Series([1, 1, 2, 2, 3], name="nums")
    result = descriptive_statistics(s)
    # mode should be first lowest value
    assert result['mode'] == pytest.approx(1.0)


def test_cv_none_when_mean_zero():
    s = pd.Series([0, 0, 0, 0], name="nums")
    result = descriptive_statistics(s)
    assert result['mean'] == pytest.approx(0.0)
    assert result['cv'] is None


def test_include_type_true_and_kwargs():
    # float series with few uniques
    s = pd.Series([1.1, 2.2, 1.1, 2.2, 1.1], name="nums")
    # default max_unique_fraction=0.05 would be False, but override to 1.0 yields True
    result = descriptive_statistics(s, include_type=True, max_unique_fraction=1.0)
    assert 'is_discrete' in result
    assert result['is_discrete'] is True


def test_include_type_false_skips_discrete():
    s = pd.Series([1, 2, 3], name="nums")
    result = descriptive_statistics(s, include_type=False)
    assert 'is_discrete' not in result
