import numpy as np
import pandas as pd

from analytics_eda.core.numeric import (
    NumericDistributionAnalysis,
    NumericDistributionAnalysisContext,
)


def test_numeric_distribution_analysis_runs(tmp_path, assert_report_data):
    s = pd.Series(np.random.default_rng(0).normal(size=20), name="metric", dtype="float64")
    ctx = NumericDistributionAnalysisContext(
        base_dir=tmp_path,
        data_source="UnitTest",
        filter_desc="UnitTest",
        save_json_report=False,
        return_full_report=True,
        distribution_names=("norm",),
    )
    analysis = NumericDistributionAnalysis(ctx)

    report = analysis.run(s)

    expected = {
        "central_tendency": {"data": {"histogram": {"chart_metadata": {"title": lambda v: isinstance(v, str)}}}},
        "dispersion": {"data": {"boxplot": {"chart_metadata": {"title": lambda v: isinstance(v, str)}}}},
        "shape": {
            "data": {
                "density": {"chart_metadata": {"title": lambda v: isinstance(v, str)}},
                "distribution_fits": lambda v: (isinstance(v, dict) and isinstance(v.get("data"), dict) and "distribution_fits" in v["data"] and "norm" in v["data"]["distribution_fits"]),
            },
        },
    }
    assert_report_data(report, expected, tmp_path / "numeric_distribution_analysis")
