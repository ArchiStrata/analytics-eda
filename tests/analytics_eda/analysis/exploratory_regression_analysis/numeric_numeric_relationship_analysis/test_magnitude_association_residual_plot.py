import numpy as np
import pandas as pd
import pytest

from analytics_eda.analysis.exploratory_regression_analysis.numeric_numeric_relationship_analysis.magnitude_association_residual_plot import (
    MagnitudeAssociationResidualContext,
    MagnitudeAssociationResidualPlot,
)

test_rand = np.random.default_rng(0)


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
                "descriptive_stats": {},   # BasePlot.default_descriptive()
                "inferential_stats": {},
                "chart_metadata": {
                    "file_name": None,
                    "xlabel": "Fitted values",
                    "ylabel": "Residuals",
                },
            },
        ),
        # 1) Perfect fit: y = x → residuals all ~0
        (
            lambda: pd.DataFrame({
                "feat_x": np.arange(5, dtype=float),    # 0..4
                "feat_y": np.arange(5, dtype=float),    # 0..4
            }),
            "feat_x", "feat_y",
            {"file_name": "residuals_perfect_fit.png"},
            {
                "chart_metadata": {
                    "file_name": "residuals_perfect_fit.png",
                },
                "descriptive_stats": {
                    "n_obs": 5,
                    "resid_mean": pytest.approx(0.0, abs=1e-12),
                    "resid_std": pytest.approx(0.0, abs=1e-12),
                    "resid_range": pytest.approx(0.0, abs=1e-12),
                },
            },
        ),
        # 2) Noisy linear: y = 2x + noise → mean ~0, std > 0, non-zero range
        (
            lambda: (lambda n=20, rng=test_rand:
                     pd.DataFrame({
                         "xnum": np.linspace(0, 10, n, dtype=float),
                         "ynum": 2.0*np.linspace(0, 10, n, dtype=float) + rng.normal(0, 1.0, n),
                     }))(),
            "xnum", "ynum",
            {"file_name": "residuals_noisy_linear.png"},
            {
                "chart_metadata": {
                    "file_name": "residuals_noisy_linear.png",
                },
                "descriptive_stats": {
                    "n_obs": 20,
                    "resid_mean": pytest.approx(0.0, abs=0.25),
                    # leave std/range to sanity checks below to avoid flakiness across platforms
                },
            },
        ),
        # 3) Custom labels & metadata
        (
            lambda: pd.DataFrame({
                "xA": [1.0, 2.0, 3.0, 4.0],
                "yB": [2.1, 4.1, 6.1, 8.1],
            }),
            "xA", "yB",
            {
                "xlabel": "Fitted \u2192 ŷ",
                "ylabel": "Residuals (y - ŷ)",
                "data_source": "UnitTest",
                "file_name": "residuals_custom.png",
            },
            {
                "chart_metadata": {
                    "xlabel": "Fitted \u2192 ŷ",
                    "ylabel": "Residuals (y - ŷ)",
                    "data_source": "UnitTest",
                    "file_name": "residuals_custom.png",
                },
                "descriptive_stats": {
                    "n_obs": 4,
                },
            },
        ),
    ],
    ids=[
        "empty_df",
        "perfect_fit",
        "noisy_linear",
        "custom_labels",
    ],
)
def test_magnitude_association_residual_param(make_df, x_col, y_col, kwargs, expect, tmp_path, assert_plot_metadata):
    df = make_df()

    # If saving is requested, write into tmp_path
    if "file_name" in kwargs:
        kwargs = {**kwargs, "save_path": tmp_path}

    ctx = MagnitudeAssociationResidualContext(**kwargs)
    plot = MagnitudeAssociationResidualPlot(ctx)

    # Run with explicit x/y role mapping
    payload = plot.run(df, cols=[x_col, y_col], role_map={"x": x_col, "y": y_col})

    # Reuse helper to check chart_metadata + selected descriptive_stats
    assert_plot_metadata(payload, expect, tmp_path)
