import numpy as np
import pandas as pd
import pytest

from analytics_eda.analysis.exploratory_regression_analysis.numeric_numeric_relationship_analysis.relationship_structure.relationship_structure_scatter_lowess_plot import (
    RelationshipStructureScatterLowessContext,
    RelationshipStructureScatterLowessPlot,
)


@pytest.mark.parametrize(
    "make_df, x_col, y_col, kwargs, expect",
    [
        # 0) Empty input → validated frame is empty => default_descriptive ({})
        (
            lambda: pd.DataFrame(
                {
                    "x": pd.Series([], dtype=float),
                    "y": pd.Series([], dtype=float),
                }
            ),
            "x",
            "y",
            {},
            {
                "descriptive_stats": {},  # BasePlot.default_descriptive()
                "inferential_stats": {},
                "chart_metadata": {
                    "file_name": None,
                    "xlabel": "X",
                    "ylabel": "Y",
                },
            },
        ),
        # 1) Small perfectly linear set
        (
            lambda: pd.DataFrame(
                {
                    "feat_x": np.arange(5, dtype=float),  # 0..4
                    "feat_y": np.arange(5, dtype=float),  # 0..4
                }
            ),
            "feat_x",
            "feat_y",
            {"file_name": "scatter_lowess_linear.png"},
            {
                "chart_metadata": {
                    "file_name": "scatter_lowess_linear.png",
                },
                "descriptive_stats": {
                    "n_obs": 5,
                    "x_min": 0.0,
                    "x_max": 4.0,
                    "y_min": 0.0,
                    "y_max": 4.0,
                },
            },
        ),
        # 2) Noisy positive relationship
        (
            lambda: (lambda n=12: pd.DataFrame({"xnum": np.arange(n, dtype=float), "ynum": 1.8 * np.arange(n, dtype=float) + np.array([0, 1, -1, 0.5, -0.5, 1.5, 0, -1, 0.2, -0.2, 0.7, -0.7])}))(),
            "xnum",
            "ynum",
            {"file_name": "scatter_lowess_noisy.png", "frac": 0.35, "iters": 1},
            {
                "chart_metadata": {
                    "file_name": "scatter_lowess_noisy.png",
                },
                "descriptive_stats": {
                    "n_obs": 12,
                    "x_min": 0.0,
                    "x_max": 11.0,
                },
            },
        ),
        # 3) Custom chart metadata values (labels, data_source)
        (
            lambda: pd.DataFrame(
                {
                    "xA": [1.0, 2.0, 3.0, 4.0],
                    "yB": [4.0, 3.0, 2.0, 1.0],
                }
            ),
            "xA",
            "yB",
            {"xlabel": "Feature A", "ylabel": "Feature B", "data_source": "UnitTest", "file_name": "scatter_lowess_custom.png"},
            {
                "chart_metadata": {
                    "xlabel": "Feature A",
                    "ylabel": "Feature B",
                    "data_source": "UnitTest",
                    "file_name": "scatter_lowess_custom.png",
                },
                "descriptive_stats": {
                    "n_obs": 4,
                    "x_min": 1.0,
                    "x_max": 4.0,
                    "y_min": 1.0,
                    "y_max": 4.0,
                },
            },
        ),
    ],
    ids=[
        "empty_df",
        "linear_small",
        "noisy_positive",
        "custom_labels",
    ],
)
def test_relationship_structure_scatter_lowess_param(make_df, x_col, y_col, kwargs, expect, tmp_path, assert_plot_metadata):
    df = make_df()

    # if saving is requested, write into tmp_path
    if "file_name" in kwargs:
        kwargs = {**kwargs, "base_dir": tmp_path}

    ctx = RelationshipStructureScatterLowessContext(**kwargs)
    plot = RelationshipStructureScatterLowessPlot(ctx)

    # DF-only API; explicitly map x and y roles
    payload = plot.run(df, cols=[x_col, y_col], role_map={"x": x_col, "y": y_col})

    # Reuse assert partial metadata/descriptive expectations
    assert_plot_metadata(payload, expect, tmp_path)
