from math import isclose

import pandas as pd
import pytest

from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis import (
    CategoricalNumericRelationshipAnalysis,
    CategoricalNumericRelationshipAnalysisContext,
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
    ctx = CategoricalNumericRelationshipAnalysisContext(
        base_dir=tmp_path,
        categorical_col=categorical_col,
        numeric_col=numeric_col,
        save_json_report=True,
        return_full_report=False,
    )
    return CategoricalNumericRelationshipAnalysis(ctx)


def test_missing_categorical_column(sample_df, tmp_path):
    analysis = _build_analysis(tmp_path)
    df = sample_df.drop(columns=["category"])
    with pytest.raises(KeyError) as exc:
        analysis.run(df)
    assert "Categorical column 'category' not found." in str(exc.value)


def test_missing_numeric_column(sample_df, tmp_path):
    analysis = _build_analysis(tmp_path)
    df = sample_df.drop(columns=["value"])
    with pytest.raises(KeyError) as exc:
        analysis.run(df)
    assert "Numeric column 'value' not found." in str(exc.value)


def test_invalid_categorical_dtype(sample_df, tmp_path):
    analysis = _build_analysis(tmp_path)
    df = sample_df.copy()
    df["category"] = df["category"].map({"A": 1, "B": 2})
    with pytest.raises(TypeError) as exc:
        analysis.run(df)
    assert "must be categorical or object" in str(exc.value)


def test_invalid_numeric_dtype(sample_df, tmp_path):
    analysis = _build_analysis(tmp_path)
    df = sample_df.copy()
    df["value"] = df["value"].astype(str)
    with pytest.raises(TypeError) as exc:
        analysis.run(df)
    assert "must be numeric" in str(exc.value)


@pytest.mark.parametrize(
    "make_df, expected_report_data",
    [
        (
            lambda: pd.DataFrame(
                {
                    "category": ["A", "A", "B", "B"],
                    "value": [1.0, 2.0, 3.0, 4.0],
                }
            ),
            {
                "relationship_structure": {
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
                },
                "magnitude_of_association": {
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
                },
                "direction_of_association": {
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
                },
            },
        ),
    ],
    ids=["two_groups_small_sample"],
)
def test_categorical_numeric_relationship_analysis_report_data_driven(
    tmp_path,
    assert_report_data,
    make_df,
    expected_report_data,
):
    df = make_df()
    categorical_col = "category"
    numeric_col = "value"

    analysis = _build_analysis(tmp_path, categorical_col, numeric_col)
    out = analysis.run(df)

    asset_root = tmp_path / f"categorical_{categorical_col}_numeric_{numeric_col}_relationship_analysis"
    assert_report_data(out, expected_report_data, asset_root)
