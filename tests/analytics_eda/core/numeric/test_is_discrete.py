import pytest
import pandas as pd
from analytics_eda.core.numeric.is_discrete import is_discrete


def test_integer_dtype_series_always_discrete():
    series = pd.Series([1, 2, 3, 4], dtype=int)
    assert is_discrete(series) is True


def test_float_whole_numbers_with_tolerance():
    series = pd.Series([1.0, 2.0, 3.0, 4.0], dtype=float)
    # default integer_tolerance=True
    assert is_discrete(series) is True


def test_float_whole_numbers_without_tolerance():
    series = pd.Series([1.0, 2.0, 3.0], dtype=float)
    # disable integer tolerance: falls back to unique fraction
    assert is_discrete(series, integer_tolerance=False) is False


def test_float_few_unique_values_below_default_threshold():
    # 2 unique values out of 100 -> fraction 0.02 < 0.05
    data = [1.1] * 95 + [2.2] * 5
    series = pd.Series(data, dtype=float)
    assert is_discrete(series) is True


def test_float_many_unique_values_above_default_threshold():
    # 100 unique values out of 100 -> fraction 1.0 > 0.05
    data = [i + 0.1 for i in range(100)]
    series = pd.Series(data, dtype=float)
    assert is_discrete(series) is False


def test_max_unique_fraction_override():
    # override threshold to allow more unique values
    data = list(float(i) for i in range(10))
    series = pd.Series(data, dtype=float)
    assert is_discrete(series, max_unique_fraction=1.0) is True


def test_empty_series_raises_value_error():
    series = pd.Series([None, None], dtype=float)
    with pytest.raises(ValueError, match="Series is empty after dropping NA"):
        is_discrete(series)


def test_non_numeric_dtype_raises_type_error():
    series = pd.Series(["a", "b", "c"])
    with pytest.raises(TypeError, match="Series must be int or float dtype."):
        is_discrete(series)
