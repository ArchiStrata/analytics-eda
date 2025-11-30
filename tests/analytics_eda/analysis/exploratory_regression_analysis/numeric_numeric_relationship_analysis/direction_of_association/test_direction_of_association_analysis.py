import pandas as pd

from analytics_eda.analysis.exploratory_regression_analysis.numeric_numeric_relationship_analysis import (
    DirectionOfAssociationAnalysis,
    DirectionOfAssociationAnalysisContext,
)


def test_direction_of_association_analysis(tmp_path, assert_report_data):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    ctx = DirectionOfAssociationAnalysisContext(
        base_dir=tmp_path,
        data_source="UnitTest",
        x_col="x",
        y_col="y",
        save_json_report=False,
        return_full_report=True,
    )

    report = DirectionOfAssociationAnalysis(ctx).run(df)

    expected = {
        "scatter_ols_trend": {
            "chart_metadata": {"title": lambda v: isinstance(v, str)},
        },
    }
    assert_report_data(report, expected, tmp_path / "direction_of_association")
