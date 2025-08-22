import pytest
import pandas as pd
from math import isclose

from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis.categorical_numeric_relationship_analysis import categorical_numeric_relationship_analysis

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'category': ['A', 'A', 'B', 'B'],
        'value': [1.0, 2.0, 3.0, 4.0]
    })

def test_missing_categorical_column(sample_df):
    df = sample_df.drop(columns=['category'])
    with pytest.raises(KeyError) as exc:
        categorical_numeric_relationship_analysis(df, 'value', 'category')
    assert "Categorical column 'category' not found." in str(exc.value)

def test_missing_numeric_column(sample_df):
    df = sample_df.drop(columns=['value'])
    with pytest.raises(KeyError) as exc:
        categorical_numeric_relationship_analysis(df, 'value', 'category')
    assert "Numeric column 'value' not found." in str(exc.value)

def test_invalid_categorical_dtype(sample_df):
    # category as numeric dtype should fail
    df = sample_df.copy()
    df['category'] = df['category'].map({'A': 1, 'B': 2})
    with pytest.raises(TypeError) as exc:
        categorical_numeric_relationship_analysis(df, 'value', 'category')
    assert "must be categorical or object" in str(exc.value)

def test_invalid_numeric_dtype(sample_df):
    # value as object dtype should fail
    df = sample_df.copy()
    df['value'] = df['value'].astype(str)
    with pytest.raises(TypeError) as exc:
        categorical_numeric_relationship_analysis(df, 'value', 'category')
    assert "must be numeric" in str(exc.value)

@pytest.mark.parametrize(
    "make_df, kwargs, expected_report_data",
    [
        (
            # Simple, deterministic 2-group example
            lambda: pd.DataFrame(
                {
                    "category": ["A", "A", "B", "B"],
                    "value":    [1.0, 2.0, 3.0, 4.0],
                }
            ),
            {"data_source": None},
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
                            # verify PNG exists
                            "file_name": "Group Sizes for value by category.png",
                        },
                    },
                    "variance_homogeneity_boxplot": {
                        "descriptive_stats": {
                            "n_groups": 2,
                            "group_ns": [2, 2],
                        },
                        "inferential_stats": {
                            # keep statistical expectations light & structural
                            "bartlett": {"alpha": 0.05, "reject": lambda v: isinstance(v, bool)},
                            "levene":   {"alpha": 0.05, "reject": lambda v: isinstance(v, bool)},
                        },
                        "chart_metadata": {
                            "title": "Variance Homogeneity for value by category",
                            "xlabel": "Group",
                            "ylabel": "Value",
                            "file_name": "Variance Homogeneity for value by category.png",
                        },
                    },
                    # Each category should reference a nested univariate numeric report
                    "numeric_distribution_by_category": {
                        "A": {},  # just verify report_file_path exists & is a JSON file
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
                            # means are deterministic for this dataset
                            "means": [1.5, 3.5],
                        },
                        "inferential_stats": {
                            "anova":   {"alpha": 0.05, "reject": lambda v: isinstance(v, bool)},
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
                                # keep the checks tolerant but meaningful
                                "eta_squared":       lambda v: 0.0 <= v <= 1.0,
                                "omega_squared":     lambda v: 0.0 <= v <= 1.0,
                                "epsilon_squared":   lambda v: 0.0 <= v <= 1.0,
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
                                    # mean difference for our data is exactly 2.0
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
    kwargs,
    expected_report_data,
):
    # Arrange
    df = make_df()
    numeric_col="value"
    categorical_col="category"

    # Act
    out = categorical_numeric_relationship_analysis(
        df,
        numeric_col=numeric_col,
        categorical_col=categorical_col,
        report_root=str(tmp_path),
        **kwargs,
    )

    # Assert (start from the response that has 'report_file_path')
    assert_report_data(out, expected_report_data, tmp_path / f"categorical_{categorical_col}_numeric_{numeric_col}_relationship_analysis")
