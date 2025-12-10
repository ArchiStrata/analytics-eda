from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.data_quality import (
    CompletenessIssuesAnalysis,
    CompletenessIssuesAnalysisContext,
)


def _sample_series() -> pd.Series:
    return pd.Series(
        [1, None, np.nan, " ", "", "NA", "good"],
        name="dq",
        dtype="object",
    )


@pytest.mark.parametrize(
    "expected_report_data",
    [
        {
            "completeness_issues": {
                "chart_metadata": {
                    "title": "Completeness Issues for dq (UnitTest)",
                    "xlabel": "Percent of total",
                    "ylabel": "Completeness issue",
                    "data_source": "UnitTest",
                    "file_name": "Completeness Issues for dq (UnitTest).png",
                },
            },
            "ecdf_gap": {
                "chart_metadata": {
                    "title": (lambda v: isinstance(v, str)),
                }
            },
        }
    ],
    ids=["completeness_issues"],
)
def test_completeness_issues_analysis(tmp_path, assert_report_data, expected_report_data):
    context = replace(
        CompletenessIssuesAnalysisContext(
            base_dir=tmp_path,
            save_json_report=True,
            return_full_report=False,
        ),
        data_source="UnitTest",
        filter_desc="UnitTest",
    )
    analysis = CompletenessIssuesAnalysis(context)

    report = analysis.run(_sample_series())

    assert_report_data(
        report,
        expected_report_data,
        tmp_path / "completeness",
    )
