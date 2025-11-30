from dataclasses import replace

import pandas as pd
import pytest

from analytics_eda.core.categorical.balance import (
    CategoricalBalanceAnalysis,
    CategoricalBalanceAnalysisContext,
)


def _sample_series() -> pd.Series:
    return pd.Series(
        ["A", "A", "A", "B", "B", "C", "C", "C", "C", "D", None],
        name="pets",
        dtype="object",
    )


@pytest.mark.parametrize(
    "expected_report_data",
    [
        {
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
    ],
    ids=["balance_plots"],
)
def test_categorical_balance_analysis(tmp_path, assert_report_data, expected_report_data):
    context = replace(
        CategoricalBalanceAnalysisContext(base_dir=tmp_path, save_json_report=True, return_full_report=False),
        data_source="UnitTest",
        filter_desc="UnitTest",
    )
    analysis = CategoricalBalanceAnalysis(context)

    report = analysis.run(_sample_series())

    assert_report_data(report, expected_report_data, tmp_path / "balance")
