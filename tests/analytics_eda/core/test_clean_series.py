import pytest
import pandas as pd
from analytics_eda.core import clean_series

def test_clean_series_drops_nans_preserves_index():
    series = pd.Series([10.0, None, 20.0, float('nan'), 30.0])
    result = clean_series(series)
    expected = pd.Series([10.0, 20.0, 30.0], index=[0, 2, 4])
    pd.testing.assert_series_equal(result, expected)


def test_clean_series_drops_nans_and_resets_index():
    series = pd.Series([None, 5.0, float('nan'), 15.0])
    result = clean_series(series, reset_index=True)
    expected = pd.Series([5.0, 15.0], index=[0, 1])
    pd.testing.assert_series_equal(result, expected)


def test_clean_series_raises_error_on_empty_series():
    series = pd.Series([None, float('nan')])
    with pytest.raises(ValueError, match="Series is empty after dropping NAs."):
        clean_series(series)


def test_clean_series_returns_empty_series_if_throw_false():
    series = pd.Series([None, float('nan')])
    result = clean_series(series, throw_if_empty=False)
    assert result.empty


def test_clean_series_empty_with_reset_index_false():
    series = pd.Series([None])
    result = clean_series(series, throw_if_empty=False, reset_index=False)
    assert result.empty
    assert result.index.tolist() == []


def test_clean_series_empty_with_reset_index_true():
    series = pd.Series([None])
    result = clean_series(series, throw_if_empty=False, reset_index=True)
    assert result.empty
    assert result.index.tolist() == []
