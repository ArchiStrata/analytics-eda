import pytest
import pandas as pd
from analytics_eda.core.numeric.validate_numeric_named_series import validate_numeric_named_series

@pytest.mark.parametrize(
    "make_input, exc, pattern",
    [
        # Not a Series
        (lambda: [1, 2, 3], TypeError, r"Input must be a pandas Series\."),
        # Non-numeric Series
        (lambda: pd.Series(["a", "b", "c"], name="letters"), TypeError, r"Series must be numeric"),
        # Missing name
        (lambda: pd.Series([1, 2, 3]), ValueError, r"must have a non-empty 'name'"),
        # Blank/whitespace name
        (lambda: pd.Series([1, 2, 3], name="   "), ValueError, r"must have a non-empty 'name'"),
    ],
    ids=["not_series", "non_numeric", "missing_name", "blank_name"],
)
def test_validate_numeric_named_series_errors(make_input, exc, pattern):
    with pytest.raises(exc, match=pattern):
        validate_numeric_named_series(make_input())


def test_require_name_false_allows_unnamed_series():
    series = pd.Series([1, 2, 3])
    result = validate_numeric_named_series(series, require_name=False)
    assert result is series


def test_valid_series_returns_series():
    series = pd.Series([1.0, 2.5, 3.2], name="values")
    result = validate_numeric_named_series(series)
    assert result is series
