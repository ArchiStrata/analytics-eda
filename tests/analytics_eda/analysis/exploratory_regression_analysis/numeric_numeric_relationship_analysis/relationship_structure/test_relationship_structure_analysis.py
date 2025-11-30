import pandas as pd

from analytics_eda.analysis.exploratory_regression_analysis.numeric_numeric_relationship_analysis import (
    RelationshipStructureAnalysis,
    RelationshipStructureAnalysisContext,
)


def test_relationship_structure_analysis(tmp_path, assert_report_data):
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    ctx = RelationshipStructureAnalysisContext(
        base_dir=tmp_path,
        data_source="UnitTest",
        x_col="x",
        y_col="y",
        save_json_report=False,
        return_full_report=True,
    )

    report = RelationshipStructureAnalysis(ctx).run(df)

    expected = {
        "scatter": {
            "chart_metadata": {"title": lambda v: isinstance(v, str)},
        },
        "scatter_lowess": {
            "chart_metadata": {"title": lambda v: isinstance(v, str)},
        },
    }
    assert_report_data(report, expected, tmp_path / "relationship_structure")
