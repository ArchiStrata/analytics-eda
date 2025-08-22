import pandas as pd
import pytest

from analytics_eda.analysis.exploratory_regression_analysis.numeric_numeric_relationship_analysis import (
    numeric_numeric_relationship_analysis,
)

@pytest.mark.parametrize(
    "make_df, kwargs, expected_report_data",
    [
        (
            # Deterministic linear relation: y = 2x
            lambda: pd.DataFrame(
                {
                    "x": [1.0, 2.0, 3.0, 4.0],
                    "y": [2.0, 4.0, 6.0, 8.0],
                }
            ),
            {"data_source": None},
            {
                "relationship_structure": {
                    # Only assert structure + that PNGs exist (file_name auto‑verified by assert_plot_metadata)
                    "scatter": {
                        "chart_metadata": {"data_source": None},
                        "descriptive_stats": {},
                    },
                    "scatter_lowess": {
                        "chart_metadata": {"data_source": None},
                        "descriptive_stats": {},
                    },
                },
                "magnitude_of_association": {
                    "scatter_ols": {
                        "chart_metadata": {"data_source": None},
                        "descriptive_stats": {},
                        "inferential_stats": {},
                    },
                    "residuals": {
                        "chart_metadata": {"data_source": None},
                        "descriptive_stats": {},
                        "inferential_stats": {},
                    },
                },
                "direction_of_association": {
                    "scatter_ols_trend": {
                        "chart_metadata": {"data_source": None},
                        "descriptive_stats": {},
                        "inferential_stats": {},
                    }
                },
            },
        ),
    ],
    ids=["linear_small_sample"],
)
def test_numeric_numeric_relationship_analysis_report_data_driven(
    tmp_path,
    assert_report_data,
    make_df,
    kwargs,
    expected_report_data,
):
    # Arrange
    df = make_df()
    x_col="x"
    y_col="y"

    # Act
    out = numeric_numeric_relationship_analysis(
        df,
        x_col="x",
        y_col="y",
        report_root=str(tmp_path),
        **kwargs,
    )

    # Assert (start from the response that has 'report_file_path')
    assert_report_data(out, expected_report_data, tmp_path / f"numeric_{x_col}_numeric_{y_col}_relationship_analysis")
