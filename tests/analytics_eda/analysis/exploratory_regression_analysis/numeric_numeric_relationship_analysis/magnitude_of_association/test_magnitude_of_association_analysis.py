import pandas as pd

from analytics_eda.analysis.exploratory_regression_analysis.numeric_numeric_relationship_analysis import (
    MagnitudeOfAssociationAnalysis,
    MagnitudeOfAssociationAnalysisContext,
)


def test_magnitude_of_association_analysis(tmp_path, assert_report_data):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    ctx = MagnitudeOfAssociationAnalysisContext(
        base_dir=tmp_path,
        data_source="UnitTest",
        x_col="x",
        y_col="y",
        save_json_report=False,
        return_full_report=True,
    )

    report = MagnitudeOfAssociationAnalysis(ctx).run(df)

    expected = {
        "scatter_ols": {
            "chart_metadata": {"title": lambda v: isinstance(v, str)},
        },
        "residuals": {
            "chart_metadata": {"title": lambda v: isinstance(v, str)},
        },
    }
    assert_report_data(report, expected, tmp_path / "magnitude_of_association")
