import pytest
import pandas as pd

from analytics_eda.core.categorical.validate_categorical_named_series import validate_categorical_named_series

def test_valid_categorical_series_with_name():
    s = pd.Series(["a", "b", "c"], dtype="category", name="my_col")
    result = validate_categorical_named_series(s)
    assert result.equals(s)

def test_valid_object_series_with_name():
    s = pd.Series(["x", "y", "z"], dtype="object", name="obj_col")
    result = validate_categorical_named_series(s)
    assert result.equals(s)

def test_valid_series_without_name_when_not_required():
    s = pd.Series(["cat", "dog"], dtype="category")
    result = validate_categorical_named_series(s, require_name=False)
    assert result.equals(s)

def test_raises_type_error_when_not_series():
    with pytest.raises(TypeError, match="Input must be a pandas Series."):
        validate_categorical_named_series(["a", "b", "c"])

def test_raises_type_error_when_not_categorical_or_object():
    s = pd.Series([1, 2, 3], name="numeric")
    with pytest.raises(TypeError, match="must be categorical.*for categorical analysis"):
        validate_categorical_named_series(s)

def test_raises_value_error_for_missing_name():
    s = pd.Series(["x", "y", "z"], dtype="category")
    with pytest.raises(ValueError, match="must have a non-empty 'name'"):
        validate_categorical_named_series(s)

def test_raises_value_error_for_blank_name():
    s = pd.Series(["x", "y", "z"], dtype="object", name=" ")
    with pytest.raises(ValueError, match="must have a non-empty 'name'"):
        validate_categorical_named_series(s)
