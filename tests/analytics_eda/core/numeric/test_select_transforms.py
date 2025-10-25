import pytest

from analytics_eda.core.numeric.select_transforms import select_transforms


@pytest.mark.parametrize(
    "statistics, normality_tests, expected",
    [
        # 1. No min ⇒ only the always-included transforms
        (
            {"skewness": 0.0, "kurtosis": 0.0},
            None,
            ["yeo-johnson", "arcsinh"]
        ),
        # 2. Strictly positive data ⇒ all domain-based + always
        (
            {"min": 1.0, "skewness": 0.0, "kurtosis": 0.0},
            None,
            ["yeo-johnson", "arcsinh", "box-cox", "log", "log1p", "sqrt", "reciprocal"]
        ),
        # 3. Zero‐min data ⇒ log1p/sqrt allowed
        (
            {"min": 0.0, "skewness": 0.0, "kurtosis": 0.0},
            None,
            ["yeo-johnson", "arcsinh", "log1p", "sqrt"]
        ),
        # 4. High right skew ⇒ log only
        (
            {"min": -5.0, "skewness": 2.0, "kurtosis": 0.0},
            None,
            ["yeo-johnson", "arcsinh", "log"]
        ),
        # 5. Heavy tails ⇒ reciprocal only
        (
            {"min": -5.0, "skewness": 0.0, "kurtosis": 2.0},
            None,
            ["yeo-johnson", "arcsinh", "reciprocal"]
        ),
        # 6. Fails normality ⇒ ensure box-cox is added
        (
            {"min": -5.0, "skewness": 0.0, "kurtosis": 0.0},
            {"reject_normality": True},
            ["yeo-johnson", "arcsinh", "box-cox"]
        ),
    ]
)
def test_select_transforms(statistics, normality_tests, expected):
    """
    select_transforms should include only those transforms whose rules pass,
    in the defined order, and append box-cox/yeo-johnson when reject_normality=True.
    """
    result = select_transforms(statistics, normality_tests)
    assert result == expected
