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

@pytest.mark.parametrize(
    "series_factory, expected_exc, match",
    [
        # Not a pandas Series
        (lambda: ["a", "b", "c"], TypeError, r"Input must be a pandas Series\."),

        # Not categorical/object dtype
        (lambda: pd.Series([1, 2, 3], name="numeric"), TypeError,
         r"must be categorical.*for categorical analysis"),

        # Missing name (None)
        (lambda: pd.Series(["x", "y", "z"], dtype="category"), ValueError,
         r"must have a non-empty 'name'"),

        # Blank/whitespace name
        (lambda: pd.Series(["x", "y", "z"], dtype="object", name=" "), ValueError,
         r"must have a non-empty 'name'"),
    ],
    ids=["not_series", "bad_dtype", "missing_name", "blank_name"],
)
def test_validate_categorical_named_series_errors(series_factory, expected_exc, match):
    obj = series_factory()
    with pytest.raises(expected_exc, match=match):
        validate_categorical_named_series(obj)
    