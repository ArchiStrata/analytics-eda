import pandas as pd
import pytest

from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis.relationship_structure.relationship_structure_analysis import (  # noqa: E501
    RelationshipStructureAnalysis,
    RelationshipStructureAnalysisContext,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "category": ["A", "A", "B", "B"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )


def _build_analysis(tmp_path, categorical_col="category", numeric_col="value"):
    ctx = RelationshipStructureAnalysisContext(
        base_dir=tmp_path,
        categorical_col=categorical_col,
        numeric_col=numeric_col,
    )
    return RelationshipStructureAnalysis(ctx)


def test_relationship_structure_analysis_validation_error(sample_df, tmp_path):
    analysis = _build_analysis(tmp_path)
    with pytest.raises(KeyError) as exc:
        analysis.run(sample_df.drop(columns=["category"]))
    assert "Categorical column 'category' not found." in str(exc.value)


def test_relationship_structure_analysis_happy_path(tmp_path, assert_report_data, sample_df):
    analysis = _build_analysis(tmp_path)
    report = analysis.run(sample_df)

    expected = {
        "group_size_barchart": {
            "descriptive_stats": {
                "n_groups": 2,
                "total": 10.0,
                "col": "category",
            },
            "chart_metadata": {
                "title": "Group Sizes for value by category",
                "xlabel": "Group",
                "ylabel": "Total",
                "file_name": "Group Sizes for value by category.png",
            },
        },
        "variance_homogeneity_boxplot": {
            "descriptive_stats": {
                "n_groups": 2,
                "group_ns": [2, 2],
            },
            "inferential_stats": {
                "bartlett": {"alpha": 0.05, "reject": lambda v: isinstance(v, bool)},
                "levene": {"alpha": 0.05, "reject": lambda v: isinstance(v, bool)},
            },
            "chart_metadata": {
                "title": "Variance Homogeneity for value by category",
                "xlabel": "Group",
                "ylabel": "Value",
                "file_name": "Variance Homogeneity for value by category.png",
            },
        },
        "numeric_distribution_by_category": {
            "A": {},
            "B": {},
        },
    }

    assert_report_data(report, expected, tmp_path)
