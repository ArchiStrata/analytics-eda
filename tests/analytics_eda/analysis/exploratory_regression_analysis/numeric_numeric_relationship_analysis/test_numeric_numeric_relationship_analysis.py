import pandas as pd

from analytics_eda.analysis.exploratory_regression_analysis.numeric_numeric_relationship_analysis import (
    NumericNumericRelationshipAnalysis,
    NumericNumericRelationshipAnalysisContext,
)


def test_numeric_numeric_relationship_analysis(tmp_path, assert_report_data):
    df = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [2.0, 4.0, 6.0, 8.0],
        }
    )
    ctx = NumericNumericRelationshipAnalysisContext(
        base_dir=tmp_path,
        data_source="UnitTest",
        x_col="x",
        y_col="y",
        save_json_report=False,
        return_full_report=True,
    )

    report = NumericNumericRelationshipAnalysis(ctx).run(df)

    expected = {
        "relationship_structure": {
            "scatter": {"chart_metadata": {"title": lambda v: isinstance(v, str)}},
            "scatter_lowess": {"chart_metadata": {"title": lambda v: isinstance(v, str)}},
        },
        "magnitude_of_association": {
            "scatter_ols": {"chart_metadata": {"title": lambda v: isinstance(v, str)}},
            "residuals": {"chart_metadata": {"title": lambda v: isinstance(v, str)}},
        },
        "direction_of_association": {
            "scatter_ols_trend": {"chart_metadata": {"title": lambda v: isinstance(v, str)}},
        },
    }

    assert_report_data(report, expected, tmp_path / "numeric_numeric_relationship_analysis")
