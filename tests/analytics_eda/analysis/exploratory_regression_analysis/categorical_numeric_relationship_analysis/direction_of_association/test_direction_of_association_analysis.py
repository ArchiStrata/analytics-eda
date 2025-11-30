from math import isclose

import pandas as pd
import pytest

from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis.direction_of_association.direction_of_association_analysis import (
    DirectionOfAssociationAnalysis,
    DirectionOfAssociationAnalysisContext,
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
    ctx = DirectionOfAssociationAnalysisContext(
        base_dir=tmp_path,
        categorical_col=categorical_col,
        numeric_col=numeric_col,
    )
    return DirectionOfAssociationAnalysis(ctx)


def test_direction_of_association_analysis_validation_error(sample_df, tmp_path):
    analysis = _build_analysis(tmp_path, categorical_col="category", numeric_col="value")
    df = sample_df.copy()
    df["category"] = df["category"].map({"A": 1, "B": 2})
    with pytest.raises(TypeError) as exc:
        analysis.run(df)
    assert "must be categorical or object" in str(exc.value)


def test_direction_of_association_analysis_happy_path(tmp_path, assert_report_data, sample_df):
    analysis = _build_analysis(tmp_path)
    report = analysis.run(sample_df)

    expected = {
        "posthoc_tukey_hsd": {
            "descriptive_stats": {
                "n_groups": 2,
                "alpha": 0.05,
                "pairs": [
                    {
                        "g1": "A",
                        "g2": "B",
                        "diff": lambda v: isclose(v, 2.0, rel_tol=0, abs_tol=1e-12),
                        "reject": lambda v: isinstance(v, bool),
                    }
                ],
            },
            "chart_metadata": {
                "title": "Post-hoc Mean Differences (Tukey HSD) for value by category",
                "xlabel": "Mean difference",
                "ylabel": "Comparison",
                "file_name": "Post-hoc Mean Differences (Tukey HSD) for value by category.png",
            },
        }
    }

    assert_report_data(report, expected, tmp_path)
