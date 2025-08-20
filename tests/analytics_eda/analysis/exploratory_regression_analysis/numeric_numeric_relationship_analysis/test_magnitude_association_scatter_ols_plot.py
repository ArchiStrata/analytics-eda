import numpy as np
import pandas as pd
import pytest

from analytics_eda.analysis.exploratory_regression_analysis.numeric_numeric_relationship_analysis.magnitude_association_scatter_ols_plot import (
    MagnitudeAssociationScatterOLSContext,
    MagnitudeAssociationScatterOLSPlot,
)


@pytest.mark.parametrize(
    "make_df, x_col, y_col, kwargs, expect",
    [
        # 0) Empty input → validated frame is empty => default_descriptive ({})
        (
            lambda: pd.DataFrame({
                "x": pd.Series([], dtype=float),
                "y": pd.Series([], dtype=float),
            }),
            "x", "y",
            {},
            {
                "descriptive_stats": {},
                "inferential_stats": {},
                "chart_metadata": {
                    "file_name": None,
                    "xlabel": "X",
                    "ylabel": "Y",
                },
            },
        ),
        # 1) Perfectly linear relationship y = x
        (
            lambda: pd.DataFrame({
                "feat_x": np.arange(5, dtype=float),    # 0..4
                "feat_y": np.arange(5, dtype=float),    # 0..4
            }),
            "feat_x", "feat_y",
            {"file_name": "scatter_ols_linear.png"},
            {
                "chart_metadata": {
                    "file_name": "scatter_ols_linear.png",
                },
                "descriptive_stats": {
                    "n_obs": 5,
                    "pearson_r": pytest.approx(1.00, 0.01),
                    "r2": pytest.approx(1.00, 0.01),
                },
            },
        ),
        # 2) Negative relationship y = -x
        (
            lambda: pd.DataFrame({
                "xnum": np.arange(6, dtype=float),
                "ynum": -1.0 * np.arange(6, dtype=float),
            }),
            "xnum", "ynum",
            {"file_name": "scatter_ols_negative.png"},
            {
                "chart_metadata": {
                    "file_name": "scatter_ols_negative.png",
                },
                "descriptive_stats": {
                    "n_obs": 6,
                    "pearson_r": pytest.approx(-1.00, 0.01),
                    "r2": pytest.approx(1.00, 0.01),
                },
            },
        ),
        # 3) Custom labels and metadata
        (
            lambda: pd.DataFrame({
                "xA": [1.0, 2.0, 3.0, 4.0],
                "yB": [2.0, 4.0, 6.0, 8.0],
            }),
            "xA", "yB",
            {
                "xlabel": "Feature A",
                "ylabel": "Feature B",
                "data_source": "UnitTest",
                "file_name": "scatter_ols_custom.png",
            },
            {
                "chart_metadata": {
                    "xlabel": "Feature A",
                    "ylabel": "Feature B",
                    "data_source": "UnitTest",
                    "file_name": "scatter_ols_custom.png",
                },
                "descriptive_stats": {
                    "n_obs": 4,
                },
            },
        ),
    ],
    ids=[
        "empty_df",
        "linear_positive",
        "linear_negative",
        "custom_labels",
    ],
)
def test_magnitude_association_scatter_ols_param(make_df, x_col, y_col, kwargs, expect, tmp_path, assert_plot_metadata):
    df = make_df()

    # If saving is requested, write into tmp_path
    if "file_name" in kwargs:
        kwargs = {**kwargs, "save_path": tmp_path}

    ctx = MagnitudeAssociationScatterOLSContext(**kwargs)
    plot = MagnitudeAssociationScatterOLSPlot(ctx)

    # Run with explicit x/y role mapping
    payload = plot.run(df, cols=[x_col, y_col], role_map={"x": x_col, "y": y_col})

    # Reuse assert helper to check chart_metadata + selected descriptive_stats
    assert_plot_metadata(payload, expect, tmp_path)
