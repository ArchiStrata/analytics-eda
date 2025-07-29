import math
import pytest
import numpy as np
import pandas as pd

from analytics_eda.core.numeric.binning_rules import sturges_bins, scott_bins, freedman_diaconis_bins, doane_bins


def test_sturges_bins_various_lengths():
    """
    Test sturges_bins against known sample sizes and expected bin counts.
    """
    cases = [
        (1, math.ceil(math.log2(1) + 1)),   # 1 observation
        (2, math.ceil(math.log2(2) + 1)),   # 2 observations
        (3, math.ceil(math.log2(3) + 1)),   # 3 observations
        (4, math.ceil(math.log2(4) + 1)),   # 4 observations
        (8, math.ceil(math.log2(8) + 1)),   # 8 observations
        (16, math.ceil(math.log2(16) + 1)), # 16 observations
    ]
    for n_obs, expected in cases:
        series = pd.Series(np.arange(n_obs), name='test')
        assert sturges_bins(series) == expected


def test_sturges_bins_drops_nas():
    """
    Ensure that NaN values are dropped before computation.
    """
    # After dropping NaNs, 3 values remain, so k = ceil(log2(3) + 1)
    series = pd.Series([10, np.nan, 20, None, 30], name='test')
    expected = math.ceil(math.log2(3) + 1)
    assert sturges_bins(series) == expected


def test_sturges_bins_empty_raises():
    """
    An empty series (or one with only NaNs) should raise ValueError.
    """
    empty_series = pd.Series([], dtype=float, name='test')
    with pytest.raises(ValueError):
        sturges_bins(empty_series)

    nan_only_series = pd.Series([np.nan, None], name='test')
    with pytest.raises(ValueError):
        sturges_bins(nan_only_series)


def test_scott_bins_linearly_spaced():
    # For a uniform range 0-99, Scott's rule should yield approximately 5 bins
    s = pd.Series(np.arange(100), name='test')
    expected = 5
    assert scott_bins(s) == expected


def test_scott_bins_drops_nas():
    # After dropping NAs, data = [1, 2, 3] -> sigma=1, h≈2.427 -> k = ceil(2/h) = 1
    s = pd.Series([1, 2, 3, np.nan, None], name='test')
    assert scott_bins(s) == 1


def test_scott_bins_too_few_observations():
    # Should require at least two non-NA values
    with pytest.raises(ValueError):
        scott_bins(pd.Series([42.0], name='test'))


def test_scott_bins_zero_variance():
    # All values identical -> zero variance -> error
    s = pd.Series([5.0, 5.0, 5.0], name='test')
    with pytest.raises(ValueError):
        scott_bins(s)


def test_scott_bins_non_numeric_series():
    # Non-numeric values should raise a type error
    with pytest.raises(TypeError):
        scott_bins(pd.Series(['a', 'b', 'c'], name='test'))

def test_fd_bins_uniform_range():
    # Uniform data 0-9: IQR=Q3-Q1 ≈5.5, h≈11/10^(1/3), k≈2
    data = pd.Series(np.arange(10), name='test')
    expected = math.ceil((data.max() - data.min()) / (2 * (data.quantile(0.75) - data.quantile(0.25)) / (len(data) ** (1/3))))
    assert freedman_diaconis_bins(data) == expected


def test_fd_bins_small_dataset():
    # Small dataset [1,2,3], n=3, Q1=1.5,Q3=2.5, iqr=1, h≈2/³√3 ~1.26, k=ceil(2/1.26)=2
    data = pd.Series([1, 2, 3], name='test')
    assert freedman_diaconis_bins(data) == 2


def test_fd_bins_drops_nas():
    # After dropping NaNs, data=[5,10,15], IQR=10, range=10, h=2*10/3^(1/3), k=ceil(10/h)
    raw = pd.Series([5, np.nan, 10, None, 15], name='test')
    clean = raw.dropna()
    iqr = clean.quantile(0.75) - clean.quantile(0.25)
    h = 2 * iqr / (len(clean) ** (1/3))
    expected = max(math.ceil((clean.max() - clean.min()) / h), 1)
    assert freedman_diaconis_bins(raw) == expected


def test_fd_bins_insufficient_length():
    # Less than 2 non-NA values should raise ValueError
    with pytest.raises(ValueError):
        freedman_diaconis_bins(pd.Series([42.0], name='test'))


def test_fd_bins_zero_iqr():
    # IQR zero (all identical) should raise ValueError
    with pytest.raises(ValueError):
        freedman_diaconis_bins(pd.Series([7.0, 7.0, 7.0], name='test'))

def test_doane_bins_symmetric_reduces_to_sturges():
    """
    For a symmetric dataset (skewness ≈ 0), Doane's rule should equal Sturges' rule.
    """
    data = pd.Series([1.0, 2.0, 3.0], name='test')
    expected = sturges_bins(data)
    assert doane_bins(data) == expected


def test_doane_bins_skewed_data_increases_bins():
    """
    For skewed data, Doane's rule should yield at least as many bins as Sturges'.
    """
    data = pd.Series([1, 1, 1, 1, 5, 10, 100], name='test')
    k_doane = doane_bins(data)
    k_sturges = sturges_bins(data)
    assert isinstance(k_doane, int)
    assert k_doane >= k_sturges


def test_doane_bins_drops_nas_and_computes_correctly():
    """
    Ensure NaNs are dropped before computing bins.
    """
    raw = pd.Series([5, np.nan, 10, None, 15], name='test')
    clean = raw.dropna()
    n = len(clean)
    g1 = clean.skew()
    sigma_g1 = math.sqrt(6 * (n - 2) / ((n + 1) * (n + 3)))
    expected = math.ceil(1 + math.log2(n) + math.log2(1 + abs(g1) / sigma_g1))
    assert doane_bins(raw) == expected


def test_doane_bins_insufficient_length_raises():
    """
    Fewer than three values should raise ValueError.
    """
    with pytest.raises(ValueError):
        doane_bins(pd.Series([42.0, 42.0], name='test'))
