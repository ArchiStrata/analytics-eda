import pandas as pd

from analytics_eda.core.numeric import (
    TransformEvaluationAnalysis,
    TransformEvaluationAnalysisContext,
)


def test_transform_evaluation_analysis(tmp_path, assert_report_data):
    s = pd.Series([1.0, 2.0, 3.0, 4.0], name="metric")
    ctx = TransformEvaluationAnalysisContext(
        base_dir=tmp_path,
        data_source="UnitTest",
        filter_desc="UnitTest",
        descriptive_stats={"skewness": 1.0},
        normality_tests={"p_value": 0.01},
        transform_names=["log"],
        save_json_report=False,
        return_full_report=True,
    )
    analysis = TransformEvaluationAnalysis(ctx)

    report = analysis.run(s)

    expected = {
        "transforms": {
            "log": {
                "data": {
                    "central_tendency": {},
                    "dispersion": {},
                    "shape": {},
                }
            }
        }
    }
    assert_report_data(report, expected, tmp_path / "transforms")
