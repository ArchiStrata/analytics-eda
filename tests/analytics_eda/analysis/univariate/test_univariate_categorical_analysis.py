import pandas as pd
import pytest

from analytics_eda.analysis.univariate.univariate_categorical_analysis import (
    UnivariateCategoricalAnalysis,
    UnivariateCategoricalAnalysisContext,
)


@pytest.mark.parametrize(
    "make_series, kwargs, expected_report_data",
    [
        (
            lambda: pd.Series(["A", "A", "A", "B", "B", "C", "C", "C", "C", "D", None], name="pets", dtype="object"),
            {
                "data_source": "UnitTest",
            },
            {
                "data_quality": {
                    "data": {
                        "completeness": {"data": {"completeness_issues": {"chart_metadata": {"title": lambda v: isinstance(v, str)}}}},
                        "validity": {
                            "data": {
                                "allowed_categories": {
                                    "descriptive_stats": {"skip_plot": True},
                                }
                            }
                        },
                        "consistency": {"data": {"type_composition": {"chart_metadata": {"title": lambda v: isinstance(v, str)}}}},
                        "uniqueness": {"data": {"cardinality": {"chart_metadata": {"title": lambda v: isinstance(v, str)}}}},
                    }
                },
                "distribution": {
                    "data": {
                        "frequency_distribution": {
                            "data": {
                                "pareto": {
                                    "chart_metadata": {
                                        "title": lambda v: isinstance(v, str),
                                    },
                                },
                            },
                        },
                        "balance": {
                            "data": {
                                "density": {"chart_metadata": {"title": lambda v: isinstance(v, str)}},
                                "boxplot": {"chart_metadata": {"title": lambda v: isinstance(v, str)}},
                                "rare_categories": {"chart_metadata": {"title": lambda v: isinstance(v, str)}},
                                "chi_square_uniform": {"chart_metadata": {"title": lambda v: isinstance(v, str)}},
                                "lorenz_curve": {"chart_metadata": {"title": lambda v: isinstance(v, str)}},
                            },
                        },
                    }
                },
            },
        ),
    ],
    ids=["basic_report"],
)
def test_univariate_categorical_analysis_report_data_driven(tmp_path, assert_report_data, make_series, kwargs, expected_report_data):
    # Arrange
    s = make_series()

    # Act
    ctx = UnivariateCategoricalAnalysisContext(
        base_dir=tmp_path,
        data_source=kwargs.get("data_source"),
        filter_desc=kwargs.get("filter_desc"),
        save_json_report=True,
        return_full_report=False,
    )
    analysis = UnivariateCategoricalAnalysis(ctx)
    out = analysis.run(s)

    # Assert
    assert_report_data(out, expected_report_data, tmp_path / "univariate" / "categorical")


@pytest.mark.parametrize(
    "series_factory, expected_exc, match",
    [
        # Not a pandas Series
        (lambda: ["a", "b", "c"], TypeError, r"Input must be a pandas Series."),
        # Not categorical/object dtype
        (lambda: pd.Series([1, 2, 3], name="numeric"), TypeError, r"must be categorical.*for categorical analysis"),
        # Missing name (None)
        (lambda: pd.Series(["x", "y", "z"], dtype="category"), ValueError, r"must have a non-empty 'name'"),
        # Blank/whitespace name
        (lambda: pd.Series(["x", "y", "z"], dtype="object", name=" "), ValueError, r"must have a non-empty 'name'"),
    ],
    ids=["not_series", "bad_dtype", "missing_name", "blank_name"],
)
def test_validate_categorical_named_series_errors(series_factory, expected_exc, match, tmp_path):
    s = series_factory()
    with pytest.raises(expected_exc, match=match):
        ctx = UnivariateCategoricalAnalysisContext(base_dir=tmp_path)
        analysis = UnivariateCategoricalAnalysis(ctx)
        analysis.run(s)
