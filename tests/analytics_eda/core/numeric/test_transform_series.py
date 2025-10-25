import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analytics_eda.core.numeric.transform_series import transform_series


@pytest.mark.parametrize(
    "method, data, expected_func",
    [
        ("yeo-johnson", [-1.0, 0.0, 1.0], lambda arr: stats.yeojohnson(arr)[0]),
        ("arcsinh",     [-1.0, 0.0, 1.0], lambda arr: np.arcsinh(arr)),
        ("box-cox",     [0.1, 1.0, 2.0],  lambda arr: stats.boxcox(arr)[0]),
        ("log",         [0.1, 1.0, 2.0],  lambda arr: np.log(arr)),
        ("log1p",       [-1.0, 0.0, 1.0], lambda arr: np.log1p(arr)),
        ("sqrt",        [0.0, 1.0, 4.0],  lambda arr: np.sqrt(arr)),
        ("reciprocal",  [ 1.0, 2.0, -1.0],lambda arr: 1.0 / arr),
    ]
)
def test_transform_series_valid_cases(method, data, expected_func):
    series = pd.Series(data, name="x")
    # Should not raise:
    transformed = transform_series(series, method)
    # Must return a Series of same shape and float dtype
    assert isinstance(transformed, pd.Series)
    assert transformed.shape == series.shape
    assert np.issubdtype(transformed.dtype, np.floating)
    # Values match direct call
    arr = series.to_numpy(dtype=float)
    expected = expected_func(arr)
    np.testing.assert_allclose(transformed.to_numpy(), expected, rtol=1e-6, atol=1e-8)

@pytest.mark.parametrize(
    "method, data, err_substr",
    [
        # unsupported method
        ("unknown",    [1.0, 2.0, 3.0],           "Unsupported transform"),
        # box-cox requires x > 0
        ("box-cox",    [0.0, 1.0, 2.0],           "Box–Cox requires x > 0"),
        # log requires x > 0
        ("log",        [0.0, 1.0, 2.0],           "Log requires x > 0"),
        # log1p requires x ≥ -1
        ("log1p",      [-2.0, 0.0, 1.0],          "Log1p requires x ≥ -1"),
        # sqrt requires x ≥ 0
        ("sqrt",       [-1.0, 0.0, 1.0],          "Sqrt requires x ≥ 0"),
        # reciprocal requires x ≠ 0
        ("reciprocal", [0.0, 1.0, -1.0],          "Reciprocal requires x ≠ 0"),
    ]
)
def test_transform_series_error_cases(method, data, err_substr):
    """
    Data‐driven tests that each invalid input or method
    raises a ValueError with an appropriate message.
    """
    series = pd.Series(data, name="x")
    with pytest.raises(ValueError) as exc:
        transform_series(series, method)
    # Check that the exception message mentions the expected substring
    assert err_substr in str(exc.value)
