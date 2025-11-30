from dataclasses import replace

import pandas as pd
import pytest

from analytics_eda.core.categorical.balance import CategoricalBalanceAnalysisContext
from analytics_eda.core.categorical.categorical_distribution_analysis import (
    CategoricalDistributionAnalysis,
    CategoricalDistributionAnalysisContext,
)
from analytics_eda.core.categorical.frequency_distribution import (
    CategoricalFrequencyDistributionAnalysisContext,
)


def _build_frequency_context(base_dir) -> CategoricalFrequencyDistributionAnalysisContext:
    proto = CategoricalFrequencyDistributionAnalysisContext(base_dir=base_dir)
    return replace(
        proto,
        data_source="UnitTest",
        filter_desc="UnitTest",
    )


def _build_balance_context(base_dir) -> CategoricalBalanceAnalysisContext:
    proto = CategoricalBalanceAnalysisContext(base_dir=base_dir)
    return replace(
        proto,
        data_source="UnitTest",
        filter_desc="UnitTest",
    )


def _build_distribution_context(base_dir, data_source: str = "UnitTest") -> CategoricalDistributionAnalysisContext:
    freq_ctx = _build_frequency_context(base_dir)
    balance_ctx = _build_balance_context(base_dir)
    proto = CategoricalDistributionAnalysisContext(base_dir=base_dir)

    return replace(
        proto,
        data_source=data_source,
        filter_desc="UnitTest",
        frequency_context=freq_ctx,
        balance_context=balance_ctx,
    )


@pytest.mark.parametrize(
    "make_series, kwargs, expected_report_data",
    [
        (
            lambda: pd.Series(
                ["A", "A", "A", "B", "B", "C", "C", "C", "C", "D", None],
                name="pets",
                dtype="object",
            ),
            {"data_source": "UnitTest"},
            {
                "frequency_distribution": {
                    "data": {
                        "pareto": {
                            "chart_metadata": {
                                "title": "Pareto Chart of pets (UnitTest)",
                                "xlabel": "Share of total",
                                "ylabel": "Category",
                                "data_source": "UnitTest",
                                "file_name": "Pareto Chart of pets (UnitTest).png",
                            }
                        }
                    }
                },
                "balance": {
                    "data": {
                        "density": {
                            "chart_metadata": {
                                "title": "Distribution Density of count (UnitTest)",
                                "xlabel": "Frequency",
                                "ylabel": "Density",
                                "data_source": "UnitTest",
                                "file_name": "Distribution Density of count (UnitTest).png",
                            },
                        },
                        "boxplot": {
                            "chart_metadata": {
                                "title": "Dispersion of count (UnitTest) (IQR & Outliers)",
                                "ylabel": "Frequency",
                                "data_source": "UnitTest",
                                "file_name": "Dispersion of count (UnitTest) (IQR & Outliers).png",
                            },
                        },
                        "chi_square_uniform": {
                            "chart_metadata": {
                                "title": "Chi-Square Goodness-of-Fit: pets (UnitTest)",
                                "xlabel": "Frequency",
                                "ylabel": "Category",
                                "data_source": "UnitTest",
                                "file_name": "Chi-Square Goodness-of-Fit: pets (UnitTest).png",
                            },
                        },
                        "lorenz_curve": {
                            "chart_metadata": {
                                "title": "Lorenz Curve of pets (UnitTest)",
                                "xlabel": "Cumulative % of categories",
                                "ylabel": "Cumulative % of values",
                                "data_source": "UnitTest",
                                "file_name": "Lorenz Curve of pets (UnitTest).png",
                            }
                        },
                        "rare_categories": {
                            "chart_metadata": {
                                "title": "Rare Categories of pets (UnitTest)",
                                "xlabel": "Percent of total",
                                "ylabel": "Category",
                                "data_source": "UnitTest",
                                "file_name": "Rare Categories of pets (UnitTest).png",
                            }
                        },
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
    s = make_series()
    context = _build_distribution_context(tmp_path, data_source=kwargs.get("data_source", "UnitTest"))
    analysis = CategoricalDistributionAnalysis(context)
    report = analysis.run(s)

    assert_report_data(report, expected_report_data, tmp_path, root_key="data")


@pytest.mark.parametrize(
    "series_factory, expected_exc, match",
    [
        (lambda: ["a", "b", "c"], TypeError, r"Input must be a pandas Series."),
        (lambda: pd.Series([1, 2, 3], name="numeric"), TypeError, r"must be categorical.*for categorical analysis"),
        (lambda: pd.Series(["x", "y", "z"], dtype="category"), ValueError, r"must have a non-empty 'name'"),
        (lambda: pd.Series(["x", "y", "z"], dtype="object", name=" "), ValueError, r"must have a non-empty 'name'"),
    ],
    ids=["not_series", "bad_dtype", "missing_name", "blank_name"],
)
def test_validate_categorical_named_series_errors(series_factory, expected_exc, match, tmp_path):
    obj = series_factory()
    context = _build_distribution_context(tmp_path)
    analysis = CategoricalDistributionAnalysis(context)

    with pytest.raises(expected_exc, match=match):
        analysis.run(obj)
