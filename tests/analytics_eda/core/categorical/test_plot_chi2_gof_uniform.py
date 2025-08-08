import pytest
import pandas as pd

from analytics_eda.core.categorical import plot_chi2_gof_uniform

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
        plot_chi2_gof_uniform(obj)


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) Empty series returns early with no tests
        (
            lambda: pd.Series(pd.Categorical([], categories=["A", "B"]), name="testvar"),
            {},
            {
                "descriptive_stats": {
                    "total": 0,
                    "k": 0
                },
                "tests": {},
                "chart_metadata": {
                    "file_name": None,
                    "xlabel": "Value",
                    "ylabel": "Frequency"
                }
            }
        ),
        # 1) Balanced distribution (uniform, should not reject H0)
        (
            lambda: pd.Series(["A", "B", "C", "A", "B", "C"], name="letters"),
            {"alpha": 0.05, "file_name": "gof_balanced_distribution.png"},
            {
                "chart_metadata": {
                    "file_name": "gof_balanced_distribution.png"
                },
                "descriptive_stats": {
                    "total": 6,
                    "k": 3
                },
                "tests": {
                    "chi2_gof_null_uniform": {
                        "reject": False,
                        'statistic': 0.0,
                        'p_value': 1.0,
                        'alpha': 0.05,
                        'warning': 'Some expected counts are below 5; chi-square test results may not be reliable.'
                    }
                }
            }
        ),
        # 2) Unbalanced distribution (should reject H0)
        (
            lambda: pd.Series(["X"] * 10 + ["Y"] * 2 + ["Z"] * 1, name="choices"),
            {"alpha": 0.05, "file_name": "gof_unbalanced_distribution.png"},
            {
                "chart_metadata": {
                    "file_name": "gof_unbalanced_distribution.png"
                },
                "descriptive_stats": {
                    "total": 13,
                    "k": 3
                },
                "tests": {
                    "chi2_gof_null_uniform": {
                        'statistic': 11.230769230769232,
                        'p_value': 0.003641408886883208,
                        'alpha': 0.05,
                        'reject': True,
                        'warning': 'Some expected counts are below 5; chi-square test results may not be reliable.'
                    }
                }
            }
        ),
        # 3) Custom chart metadata values
        (
            lambda: pd.Series(["cat", "dog", "dog", "bird"], name="animals"),
            {"xlabel": "Animal", "ylabel": "Count", "data_source": "UnitTest"},
            {
                "chart_metadata": {
                    "xlabel": "Animal",
                    "ylabel": "Count",
                    "data_source": "UnitTest"
                }
            }
        )
    ],
    ids=[
        "empty_series",
        "balanced_uniform",
        "unbalanced_reject_h0",
        "custom_labels"
    ]
)
def test_plot_chi2_gof_uniform_param(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()
    if "file_name" in kwargs:
        kwargs = {**kwargs, "save_path": tmp_path}

    payload = plot_chi2_gof_uniform(s, **kwargs)
    assert_plot_metadata(payload, expect, tmp_path)
