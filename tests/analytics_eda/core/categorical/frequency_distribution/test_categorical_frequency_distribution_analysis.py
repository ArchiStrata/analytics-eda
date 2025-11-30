from dataclasses import replace

import pandas as pd
import pytest

from analytics_eda.core.categorical.frequency_distribution import (
    CategoricalFrequencyDistributionAnalysis,
    CategoricalFrequencyDistributionAnalysisContext,
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
    ],
    ids=["pareto_only"],
)
def test_categorical_frequency_distribution_analysis(tmp_path, assert_report_data, expected_report_data):
    context = replace(
        CategoricalFrequencyDistributionAnalysisContext(base_dir=tmp_path),
        data_source="UnitTest",
        filter_desc="UnitTest",
    )
    analysis = CategoricalFrequencyDistributionAnalysis(context)

    report = analysis.run(_sample_series())

    assert_report_data(report, expected_report_data, tmp_path / "frequency_distribution")
