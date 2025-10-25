import pandas as pd
import pytest

from analytics_eda.analysis.univariate.univariate_categorical_analysis import (
    univariate_categorical_analysis,
)


@pytest.mark.parametrize(
    "make_series, kwargs, expected_report_data",
    [
        (
            lambda: pd.Series(
                ["A","A","A","B","B","C","C","C","C","D", None],
                name="pets", dtype="object"
            ),
            {
                "data_source": "UnitTest",
            },
            {
                # Expectations for the data_quality section
                "data_quality": {
                    "missing_data_barchart": {
                        "chart_metadata": {
                            "title": "Missing Data for pets",
                            "xlabel": "Status",
                            "ylabel": "Percentage of Total",
                            "data_source": "UnitTest",
                            "file_name": "Missing Data for pets.png",
                        }
                    },
                    "categorical_cleanliness_barchart": {
                        "chart_metadata": {
                            "title": "Categorical Cleanliness for pets",
                            "xlabel": "Percent of non\u2011null",
                            "ylabel": "Issue Type",
                            "data_source": "UnitTest",
                            "file_name": None
                        },
                        "cleaning_meta": {
                            "n_stripped": 0,
                            "n_case_normalized": 0,
                            "n_nonstandard_replaced": 0,
                            "n_invalid_mapped": 0,
                            "policy": {
                                "allowed_char_pattern": "^[\\w\\s\\-\\_/.,&()']*$",
                                "allowed_categories": None,
                                "case_sensitive_allowed": False,
                                "treat_empty_as_invalid": True
                            }
                        }
                    }
                },
                # Expectations for the cardinality section
                "cardinality": {
                    "barchart": {
                        "chart_metadata": {
                            "title": "Cardinality Check — Discrete vs. Continuous for count",
                            "xlabel": "Number of Records",
                            "ylabel": "Values (Top N)",
                            "data_source": "UnitTest",
                            "file_name": "Cardinality Check — Discrete vs. Continuous for count.png"
                        }
                    }
                },
                # Expectations for the nested distribution report
                "distribution": {
                    "frequency_distribution": {
                        "pareto": {
                            "chart_metadata":  {
                                "title": "Pareto Chart of pets",
                                "xlabel": "Share of total",
                                "ylabel": "Category",
                                "data_source": "UnitTest",
                                "file_name": "Pareto Chart of pets.png"
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
                        },
                        "boxplot": {
                            "chart_metadata":  {
                                "title": "Dispersion of count (IQR & Outliers)",
                                "xlabel": "",
                                "ylabel": "Frequency",
                                "data_source": "UnitTest",
                                "file_name": "Dispersion of count (IQR & Outliers).png"
                            },
                        },
                        "rare_categories": {
                            "chart_metadata": {
                                "title": "Rare Categories of pets",
                                "xlabel": "Percent of total",
                                "ylabel": "Category",
                                "data_source": "UnitTest",
                                "file_name": "Rare Categories of pets.png",
                            }
                        },
                        "chi_square_uniform": {
                            "chart_metadata":  {
                                "title": "Chi-Square Goodness-of-Fit: pets",
                                "xlabel": "Frequency",
                                "ylabel": "Category",
                                "data_source": "UnitTest",
                                "file_name": "Chi-Square Goodness-of-Fit: pets.png"
                            },
                        },
                        "lorenz_curve": {
                            "chart_metadata": {
                                "title": "Lorenz Curve of pets",
                                "xlabel": "Cumulative % of categories",
                                "ylabel": "Cumulative % of values",
                                "data_source": "UnitTest",
                                "file_name": "Lorenz Curve of pets.png"
                            }
                        }
                    },
                },
            },
        ),
    ],
    ids=["basic_report"],
)
def test_univariate_categorical_analysis_report_data_driven(
    tmp_path, assert_report_data,
    make_series, kwargs, expected_report_data
):
    # Arrange
    s = make_series()

    # Act
    out = univariate_categorical_analysis(
        s, report_root=str(tmp_path), **kwargs
    )

    # Assert
    assert_report_data(out, expected_report_data, tmp_path / s.name.replace(' ', '_'))


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
def test_validate_categorical_named_series_errors(series_factory, expected_exc, match):
    s = series_factory()
    with pytest.raises(expected_exc, match=match):
        univariate_categorical_analysis(s)
