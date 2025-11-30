import pandas as pd
import pytest

from analytics_eda.core.data_quality import (
    ValidityValueComplianceAnalysis,
    ValidityValueComplianceAnalysisContext,
)


def _categorical_series():
    return pd.Series(["apple", "banana", "bad", "APPLE"], name="fruits")


def _numeric_series():
    return pd.Series([1, 2, 3], name="nums")


@pytest.mark.parametrize(
    "series_factory, ctx_kwargs, expected",
    [
        (
            _categorical_series,
            {"allowed_categories_bar_context": {"allowed_categories": ["apple", "banana"]}},
            {
                "allowed_categories": {
                    "chart_metadata": {
                        "title": "Validity: Allowed Categories for fruits",
                    },
                    "descriptive_stats": {
                        "values_with_invalid": 1,
                        "distinct_collapse": 1,
                    },
                    "draft_descriptive_findings": {
                        "primary_finding": "1 values fall outside the allowed categories; distinct categories drop from 3 to 2 (Δ = 1) after filtering.",
                    },
                }
            },
        ),
        (
            _numeric_series,
            {"allowed_categories_bar_context": {"allowed_categories": ["1", "2"]}},
            {
                "allowed_categories": {
                    "descriptive_stats": {
                        "skip_plot": True,
                        "error": "Series is numeric; allowed categories not assessed (numeric validity plots TBD).",
                    },
                }
            },
        ),
    ],
    ids=["categorical_allowed_categories", "non_categorical_skip"],
)
def test_validity_allowed_categories_analysis(tmp_path, assert_report_data, series_factory, ctx_kwargs, expected):
    context = ValidityValueComplianceAnalysisContext(
        base_dir=tmp_path,
        save_json_report=True,
        return_full_report=False,
        **ctx_kwargs,
    )
    analysis = ValidityValueComplianceAnalysis(context)

    report = analysis.run(series_factory())

    assert_report_data(report, expected, tmp_path)
