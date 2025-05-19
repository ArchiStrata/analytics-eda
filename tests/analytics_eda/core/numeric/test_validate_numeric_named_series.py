import pytest
import pandas as pd
from analytics_eda.core.numeric.validate_numeric_named_series import validate_numeric_named_series

def test_non_series_input_raises_type_error():
    with pytest.raises(TypeError, match="Input must be a pandas Series."):
        validate_numeric_named_series([1, 2, 3])


def test_non_numeric_series_raises_type_error():
    series = pd.Series(["a", "b", "c"], name="letters")
    with pytest.raises(TypeError, match="Series must be numeric"):
        validate_numeric_named_series(series)


def test_missing_name_raises_value_error():
    series = pd.Series([1, 2, 3])  # name is None
    with pytest.raises(ValueError, match="Series must have a non-empty 'name' attribute."):
        validate_numeric_named_series(series)


def test_empty_name_raises_value_error():
    series = pd.Series([1, 2, 3], name="   ")  # name is whitespace
    with pytest.raises(ValueError, match="Series must have a non-empty 'name' attribute."):
        validate_numeric_named_series(series)


def test_require_name_false_allows_unnamed_series():
    series = pd.Series([1, 2, 3])
    result = validate_numeric_named_series(series, require_name=False)
    assert result is series


def test_valid_series_returns_series():
    series = pd.Series([1.0, 2.5, 3.2], name="values")
    result = validate_numeric_named_series(series)
    assert result is series
