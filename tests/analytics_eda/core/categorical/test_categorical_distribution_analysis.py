import pytest
import pandas as pd

from analytics_eda.core.categorical.categorical_distribution_analysis import categorical_distribution_analysis

@pytest.mark.parametrize(
    "make_series, kwargs, expected_report_data",
    [
        (
            # modestly imbalanced categories to populate all plots
            lambda: pd.Series(
                ["A","A","A","B","B","C","C","C","C","D", None],
                name="pets",
                dtype="object",
            ),
            { "data_source": "UnitTest" },
            {
                "frequency_distribution": {
                    "pareto": {
                        "chart_metadata":  {
                            "title": "Pareto Chart of pets",
                            "xlabel": "Value",
                            "ylabel": "Count",
                            "data_source": "UnitTest",
                            "file_name": "Pareto Chart of pets.png"
                        },
                        "descriptive_stats": {
                            "mode": "C",
                            "total_count": 10,
                            "n_categories": 4,
                            "cumulative_count_at_80pct": 9
                        },
                    },
                },
                "balance": {
                    "density": {
                        "chart_metadata":  {
                            "title": "Distribution Density of count",
                            "xlabel": "Frequency",
                            "ylabel": "Density",
                            "data_source": "UnitTest",
                            "file_name": "Distribution Density of count.png"
                        },
                        "descriptive_stats": {
                            "n": 4,
                            "entropy_bits": 1.9999999999942293,
                            "skewness": 0.0,
                            "kurtosis": -1.1999999999999993,
                            "modes_count": 1,
                            "quartile_skew": 0.0,
                            "pct_10": 1.3,
                            "pct_25": 1.75,
                            "pct_50": 2.5,
                            "pct_75": 3.25,
                            "pct_90": 3.7
                        },
                    },
                    "boxplot": {
                        "chart_metadata":  {
                            "title": "Dispersion of count (IQR & Outliers)",
                            "ylabel": "Frequency",
                            "data_source": "UnitTest",
                            "file_name": "Dispersion of count (IQR & Outliers).png"
                        },
                        "descriptive_stats": {
                            "params": {
                                "std_outlier_multiplier": 4.0
                            },
                            "n": 4,
                            "mean": 2.5,
                            "std": 1.2909944487358056,
                            "var": 1.6666666666666667,
                            "min": 1,
                            "max": 4,
                            "range": 3,
                            "mad": 1.0,
                            "cv": 0.5163977794943222,
                            "pct_10": 1.3,
                            "pct_25": 1.75,
                            "pct_75": 3.25,
                            "pct_90": 3.7,
                            "iqr": 1.5,
                            "extreme_lower_count": 0,
                            "extreme_upper_count": 0
                        },
                    },
                    "chi_square_uniform": {
                        "chart_metadata":  {
                            "title": "Chi-Square Goodness-of-Fit: pets",
                            "xlabel": "Frequency",
                            "ylabel": "Frequency",
                            "data_source": "UnitTest",
                            "file_name": "Chi-Square Goodness-of-Fit: pets.png"
                        },
                        "descriptive_stats": {
                            "total": 10,
                            "k": 4
                        },
                        "inferential_stats": {
                            "chi2_gof_null_uniform": {
                                "statistic": 2.0,
                                "p_value": 0.5724067044708798,
                                "alpha": 0.05,
                                "reject": False,
                                "warning": "Some expected counts are below 5; chi-square test results may not be reliable."
                            }
                        },
                    },
                    "lorenz_curve": {
                        "descriptive_stats": {
                            "total": 10,
                            "k": 4,
                            "gini_index": 0.25
                        },
                        "chart_metadata": {
                            "title": "Lorenz Curve of pets",
                            "xlabel": "Cumulative share of categories",
                            "ylabel": "Cumulative share of counts",
                            "data_source": "UnitTest",
                            "file_name": "Lorenz Curve of pets.png"
                        }
                    },
                    "rare_categories": {
                        "descriptive_stats": {
                            "total": 10,
                            "k": 4,
                            "threshold_type": "proportion",
                            "threshold_value_count": 1,
                            "threshold_value_prop": 0.01,
                            "n_rare": 1,
                            "rare_categories": [
                                "D"
                            ],
                            "rare_counts": [
                                1
                            ]
                        },
                        "inferential_stats": {},
                        "chart_metadata": {
                            "title": "Rare Categories of pets",
                            "xlabel": "Category",
                            "ylabel": "Count",
                            "data_source": "UnitTest",
                            "file_name": "Rare Categories of pets.png"
                        }
                    }
                },
            },
        ),
    ],
    ids=["basic_report"],
)
def test_categorical_distribution_analysis_report_data_driven(
    tmp_path,
    assert_report_data,
    make_series,
    kwargs,
    expected_report_data,
):
    # Arrange
    s = make_series()

    # Act
    out = categorical_distribution_analysis(
        s,
        report_path=tmp_path,
        **kwargs,
    )

    # Assert
    assert_report_data(out, expected_report_data, tmp_path)


@pytest.mark.parametrize(
    "series_factory, expected_exc, match",
    [
        # Not a pandas Series
        (lambda: ["a", "b", "c"], TypeError, r"Input must be a pandas Series."),

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
def test_validate_categorical_named_series_errors(series_factory, expected_exc, match, tmp_path):
    obj = series_factory()
    with pytest.raises(expected_exc, match=match):
        categorical_distribution_analysis(obj, report_path=tmp_path)
