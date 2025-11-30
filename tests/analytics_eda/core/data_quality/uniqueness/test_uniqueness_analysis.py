from dataclasses import replace

import pandas as pd
import pytest
from pytest import approx

from analytics_eda.core.data_quality import (
    UniquenessAnalysis,
    UniquenessAnalysisContext,
)


def _sample_series() -> pd.Series:
    return pd.Series([1, 1, 2, 3, None], name="unique")


@pytest.mark.parametrize(
    "expected_report_data",
    [
        {
            "cardinality": {
                "chart_metadata": {
                    "title": "Cardinality Check — Discrete vs. Continuous for unique (UnitTest)",
                    "data_source": "UnitTest",
                    "file_name": "Cardinality Check — Discrete vs. Continuous for unique (UnitTest).png",
                },
                "descriptive_stats": {
                    "total_nonnull": 4,
                    "nunique_native": 3,
                    "uniqueness_ratio": approx(0.75),
                    "is_discrete": True,
                    "bars": {
                        "1.0": {"count": 2},
                        "2.0": {"count": 1},
                        "3.0": {"count": 1},
                    },
                },
            },
            "duplicate_summary": {
                "chart_metadata": {
                    "title": "Duplicate Summary for unique (UnitTest)",
                    "data_source": "UnitTest",
                    "file_name": "Duplicate Summary for unique (UnitTest).png",
                },
                "descriptive_stats": {
                    "total_nonnull": 4,
                    "bars": {
                        "Distinct values": {"count": 3},
                        "Duplicate entries": {"count": 1},
                    },
                    "params": {
                        "duplicate_ratio": approx(0.25),
                        "nunique_native": 3,
                    },
                },
                "draft_descriptive_findings": {
                    "context": "4 non-null values",
                    "primary_finding": "25.0% of 4 non-null values are duplicates (1 entries).",
                },
            },
        }
    ],
    ids=["uniqueness_analysis"],
)
def test_uniqueness_analysis(tmp_path, assert_report_data, expected_report_data):
    context = replace(
        UniquenessAnalysisContext(
            base_dir=tmp_path,
            save_json_report=True,
            return_full_report=False,
        ),
        data_source="UnitTest",
        filter_desc="UnitTest",
    )
    analysis = UniquenessAnalysis(context)

    report = analysis.run(_sample_series())

    assert_report_data(
        report,
        expected_report_data,
        tmp_path,
    )
