import pandas as pd
import pytest

from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis.magnitude_of_association.magnitude_of_association_analysis import (  # noqa: E501
    MagnitudeOfAssociationAnalysis,
    MagnitudeOfAssociationAnalysisContext,
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
    ctx = MagnitudeOfAssociationAnalysisContext(
        base_dir=tmp_path,
        categorical_col=categorical_col,
        numeric_col=numeric_col,
    )
    return MagnitudeOfAssociationAnalysis(ctx)


def test_magnitude_of_association_analysis_validation_error(sample_df, tmp_path):
    analysis = _build_analysis(tmp_path)
    bad_df = sample_df.copy()
    bad_df["value"] = bad_df["value"].astype(str)
    with pytest.raises(TypeError) as exc:
        analysis.run(bad_df)
    assert "must be numeric" in str(exc.value)


def test_magnitude_of_association_analysis_happy_path(tmp_path, assert_report_data, sample_df):
    analysis = _build_analysis(tmp_path)
    report = analysis.run(sample_df)

    expected = {
        "distribution_overlap_density": {
            "descriptive_stats": {
                "n_groups": 2,
                "group_ns": [2, 2],
                "group_labels": ["A", "B"],
            },
            "chart_metadata": {
                "title": "Distribution Shape & Overlap for value by category",
                "xlabel": "Value",
                "ylabel": "Density",
                "file_name": "Distribution Shape & Overlap for value by category.png",
            },
        },
        "central_tendency_anova_kruskal": {
            "descriptive_stats": {
                "n_groups": 2,
                "group_ns": [2, 2],
                "means": [1.5, 3.5],
            },
            "inferential_stats": {
                "anova": {"alpha": 0.05, "reject": lambda v: isinstance(v, bool)},
                "kruskal": {"alpha": 0.05, "reject": lambda v: isinstance(v, bool)},
            },
            "chart_metadata": {
                "title": "Magnitude of Differences for value by category",
                "xlabel": "Group",
                "ylabel": "Value",
                "file_name": "Magnitude of Differences for value by category.png",
            },
        },
        "effect_size_barchart": {
            "descriptive_stats": {
                "n_groups": 2,
                "group_labels": ["A", "B"],
                "effect_sizes": {
                    "eta_squared": lambda v: 0.0 <= v <= 1.0,
                    "omega_squared": lambda v: 0.0 <= v <= 1.0,
                    "epsilon_squared": lambda v: 0.0 <= v <= 1.0,
                },
            },
            "chart_metadata": {
                "title": "Effect Sizes for value by category",
                "xlabel": "Effect Size",
                "ylabel": "Magnitude (0–1)",
                "file_name": "Effect Sizes for value by category.png",
            },
        },
    }

    assert_report_data(report, expected, tmp_path)
