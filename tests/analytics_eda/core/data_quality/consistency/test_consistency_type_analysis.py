from dataclasses import replace

import pandas as pd
import pytest
from pytest import approx

from analytics_eda.core.data_quality import (
    ConsistencyTypeAnalysis,
    ConsistencyTypeAnalysisContext,
)


def _sample_series() -> pd.Series:
    return pd.Series(
        [1, "2", "abc", "01", pd.Timestamp("2020-01-01"), None],
        name="consistency",
    )


def _categorical_series() -> pd.Series:
    return pd.Series(["Red", " red", "BLUE", "blue ", None], name="consistency_cat")


def _numeric_series() -> pd.Series:
    return pd.Series([1.0, 1.2, 2.34, 2.3], name="consistency_num")


@pytest.mark.parametrize(
    "series_factory, expected_report_data",
    [
        (
            _sample_series,
            {
                "type_composition": {
                    "chart_metadata": {
                        "title": "Type Composition for consistency (UnitTest)",
                        "file_name": "Type Composition for consistency (UnitTest).png",
                        "data_source": "UnitTest",
                    },
                    "descriptive_stats": {
                        "total": 6,
                        "total_nonnull": 5,
                        "dominant_type": "Numeric",
                        "dominant_ratio": approx(0.6),
                    },
                },
                "format_consistency": {
                    "chart_metadata": {
                        "title": "Format Consistency for consistency (UnitTest)",
                        "file_name": "Format Consistency for consistency (UnitTest).png",
                        "data_source": "UnitTest",
                    },
                    "descriptive_stats": {
                        "total_nonnull": 5,
                        "dominant_format": "Numeric string (int)",
                        "dominant_ratio": approx(0.4),
                    },
                    "draft_descriptive_findings": {
                        "context": "5 non-null of 6 total",
                    },
                },
                "whitespace_normalization": {
                    "chart_metadata": {
                        "title": "Whitespace Normalization Impact for consistency (UnitTest)",
                    },
                    "descriptive_stats": {
                        "values_with_issues": 0,
                        "distinct_collapse": 0,
                    },
                },
                "casing_normalization": {
                    "chart_metadata": {
                        "title": "Casing Normalization Impact for consistency (UnitTest)",
                    },
                    "descriptive_stats": {
                        "values_with_collisions": 0,
                        "distinct_collapse": 0,
                    },
                },
                "character_hygiene": {
                    "chart_metadata": {
                        "title": "Character Hygiene for consistency (UnitTest)",
                    },
                    "descriptive_stats": {
                        "values_with_issues": 1,
                        "distinct_collapse": 0,
                    },
                },
                "numeric_coercion": {
                    "chart_metadata": {
                        "title": "Numeric Coercion Issues for consistency (UnitTest)",
                    },
                    "descriptive_stats": {
                        "subset_count": 1,
                        "pct_subset": approx(0.2),
                    },
                },
                "decimal_precision": {
                    "descriptive_stats": {
                        "skip_plot": True,
                        "error": "Series is not numeric; decimal precision not assessed.",
                    },
                    "chart_metadata": {
                        "title": "Decimal Precision for consistency (not run)",
                    },
                },
            },
        ),
        (
            _categorical_series,
            {
                "decimal_precision": {
                    "descriptive_stats": {
                        "skip_plot": True,
                    },
                },
                "unit_frequency": {
                    "descriptive_stats": {
                        "skip_plot": True,
                    },
                },
                "whitespace_normalization": {
                    "chart_metadata": {
                        "title": "Whitespace Normalization Impact for consistency_cat (UnitTest)",
                    },
                    "descriptive_stats": {
                        "distinct_collapse": 0,
                        "values_with_issues": 2,
                    },
                },
                "casing_normalization": {
                    "chart_metadata": {
                        "title": "Casing Normalization Impact for consistency_cat (UnitTest)",
                    },
                    "descriptive_stats": {
                        "distinct_collapse": 2,
                        "values_with_collisions": 4,
                    },
                    "draft_descriptive_findings": {
                        "primary_finding": "4 values participate in casing collisions; distinct categories drop from 4 to 2 (Δ = 2).",
                    },
                },
                "character_hygiene": {
                    "chart_metadata": {
                        "title": "Character Hygiene for consistency_cat (UnitTest)",
                    },
                    "descriptive_stats": {
                        "values_with_issues": 0,
                        "distinct_collapse": 0,
                    },
                },
                "numeric_coercion": {
                    "chart_metadata": {
                        "title": "Numeric Coercion Issues for consistency_cat (UnitTest)",
                    },
                    "descriptive_stats": {
                        "subset_count": 4,
                        "pct_subset": approx(1.0),
                    },
                },
            },
        ),
        (
            _numeric_series,
            {
                "unit_frequency": {
                    "chart_metadata": {
                        "title": "Unit Frequency for consistency_num (UnitTest)",
                    },
                    "descriptive_stats": {
                        "dominant_unit": "Unitless",
                    },
                    "draft_descriptive_findings": {
                        "primary_finding": "Units are consistent: Unitless covers 100.0% of non-null.",
                    },
                },
                "decimal_precision": {
                    "chart_metadata": {
                        "title": "Decimal Precision for consistency_num (UnitTest)",
                    },
                    "descriptive_stats": {
                        "dominant_precision_dp": 1,
                    },
                    "draft_descriptive_findings": {
                        "primary_finding": "Precision is fragmented; top bucket 1dp covers 50.0% of non-null.",
                    },
                },
            },
        ),
    ],
    ids=["type_consistency", "categorical_consistency", "decimal_precision_numeric"],
)
def test_type_consistency_analysis(tmp_path, assert_report_data, series_factory, expected_report_data):
    context = replace(
        ConsistencyTypeAnalysisContext(
            base_dir=tmp_path,
            save_json_report=True,
            return_full_report=False,
        ),
        data_source="UnitTest",
        filter_desc="UnitTest",
    )
    analysis = ConsistencyTypeAnalysis(context)

    report = analysis.run(series_factory())

    assert_report_data(
        report,
        expected_report_data,
        tmp_path,
    )
