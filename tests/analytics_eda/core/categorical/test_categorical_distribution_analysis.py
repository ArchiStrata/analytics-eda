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
                            "xlabel": "Share of total",
                            "ylabel": "Category",
                            "data_source": "UnitTest",
                            "file_name": "Pareto Chart of pets.png"
                        }
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
                            "ylabel": "Frequency",
                            "data_source": "UnitTest",
                            "file_name": "Dispersion of count (IQR & Outliers).png"
                        },
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
                    },
                    "rare_categories": {
                        "chart_metadata": {
                            "title": "Rare Categories of pets",
                            "xlabel": "Percent of total",
                            "ylabel": "Category",
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
