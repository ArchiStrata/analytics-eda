import pytest
import pandas as pd
import numpy as np
from analytics_eda.core.numeric.assess_normality_and_transform import assess_normality_and_transform


# ---------------------------
# Helpers and Fixtures
# ---------------------------


@pytest.fixture
def series_no_transform():
    # Data close to normal
    return pd.Series([10, 11, 9, 10.5, 10.2], name="normal_series")


@pytest.fixture
def stats_normal():
    return {
        'min': 9,
        'skewness': 0.1,
        'cv': 0.05,
        'kurtosis': -0.2
    }


@pytest.fixture
def norm_reject_flag():
    return {
        "reject_normality": True
    }


@pytest.fixture
def norm_accept():
    return {'reject_normality': False}

@pytest.fixture
def series_large_skewed():
    return pd.Series(np.concatenate([np.ones(50), np.full(5, 100)]), name="large_skewed")

@pytest.fixture
def series_small_skewed():
    return pd.Series(np.concatenate([np.ones(20), np.full(5, 100)]), name="small_skewed")


@pytest.fixture
def statistics_large_skewed():
    return {
        "min": 1,
        "skewness": 2.0,
        "cv": 2.5,
        "kurtosis": 5.0
    }


# ---------------------------
# Test Cases
# ---------------------------


def test_transform_not_applied_when_not_needed(series_no_transform, stats_normal, norm_accept):
    result = assess_normality_and_transform(
        s=series_no_transform,
        statistics=stats_normal,
        normality=norm_accept
    )

    assessment = result["assessment"]
    transformed_series = result["series"]

    assert assessment["needs_transform"] is False
    assert assessment["best_transform"] is None
    assert assessment["candidates"] == {}
    assert transformed_series.equals(series_no_transform)

def test_non_series_input_raises_type_error():
    with pytest.raises(TypeError, match="Input must be a pandas Series."):
        assess_normality_and_transform([1, 2, 3], statistics={}, normality={})


def test_missing_name_raises_value_error():
    s = pd.Series([1, 2, 3], dtype=float)
    with pytest.raises(ValueError, match="Series must have a non-empty 'name'"):
        assess_normality_and_transform(s, statistics={}, normality={})

def test_dagostino_pearson_in_transform_candidates(series_large_skewed, statistics_large_skewed, norm_reject_flag):
    result = assess_normality_and_transform(
        s=series_large_skewed,
        statistics=statistics_large_skewed,
        normality=norm_reject_flag,
    )

    assessment = result["assessment"]
    candidates = assessment["candidates"]
    best = assessment["best_transform"]
    assert best is not None, "Expected a best_transform to be selected"

    norm_t = candidates[best]["normality"]

    # Because n>=50, shapiro() should NOT run, but dagostino_pearson should
    assert "shapiro" not in norm_t,     "Shapiro should be skipped for n>=50"
    assert "dagostino_pearson" in norm_t, "Expected the D’Agostino–Pearson branch to fire"

def test_shapiro_in_transform_candidates(series_small_skewed, statistics_large_skewed, norm_reject_flag):
    result = assess_normality_and_transform(
        s=series_small_skewed,
        statistics=statistics_large_skewed,
        normality=norm_reject_flag,
    )

    assessment = result["assessment"]
    candidates = assessment["candidates"]
    best = assessment["best_transform"]
    assert best is not None, "Expected a best_transform to be selected"

    norm_t = candidates[best]["normality"]

    assert "shapiro" in norm_t,     "Shapiro should not be skipped for n < 50"
    assert "dagostino_pearson" in norm_t, "Expected the D’Agostino–Pearson branch to fire"
