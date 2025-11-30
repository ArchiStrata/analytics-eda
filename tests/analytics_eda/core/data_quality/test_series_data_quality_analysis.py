import pandas as pd

from analytics_eda.core.data_quality import (
    SeriesDataQualityAnalysis,
    SeriesDataQualityAnalysisContext,
)


def _sample_series():
    return pd.Series([1, None, "x", "bad"], name="dq")


def test_series_data_quality_analysis(tmp_path, assert_report_data):
    from analytics_eda.core.data_quality.validity import (
        ValidityAllowedCategoriesBarContext,
        ValidityValueComplianceAnalysisContext,
    )

    ctx = SeriesDataQualityAnalysisContext(
        base_dir=tmp_path,
        data_source="UnitTest",
        filter_desc="UnitTest",
        save_json_report=False,
        return_full_report=True,
        validity_context=ValidityValueComplianceAnalysisContext(
            allowed_categories_bar_context=ValidityAllowedCategoriesBarContext(allowed_categories=["1", "x"]),
        ),
    )

    analysis = SeriesDataQualityAnalysis(ctx)

    report = analysis.run(_sample_series())

    expected = {
        "completeness": {
            "data": {
                "completeness_issues": {
                    "chart_metadata": {
                        "title": lambda v: isinstance(v, str),
                    },
                }
            }
        },
        "validity": {
            "data": {
                "allowed_categories": {
                    "descriptive_stats": {
                        "values_with_invalid": 1,
                    },
                }
            }
        },
        "consistency": {
            "data": {
                "type_composition": {
                    "chart_metadata": {
                        "title": lambda v: isinstance(v, str),
                    },
                }
            }
        },
        "uniqueness": {
            "data": {
                "cardinality": {
                    "chart_metadata": {
                        "title": lambda v: isinstance(v, str),
                    },
                }
            }
        },
    }

    assert_report_data(report, expected, tmp_path / "data_quality")
