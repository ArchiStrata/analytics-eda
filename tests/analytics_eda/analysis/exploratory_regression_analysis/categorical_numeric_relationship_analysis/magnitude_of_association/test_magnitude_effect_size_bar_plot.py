import numpy as np
import pandas as pd
import pytest

from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis.magnitude_of_association.magnitude_effect_size_bar_plot import (
    MagnitudeEffectSizeBarContext,
    MagnitudeEffectSizeBarPlot,
)


# --- helper to synthesize grouped normal data ---
def _make_normal_groups(specs, n_per_group=200, seed=0):
    """
    specs: list[tuple(label:str, mean:float, std:float)]
    returns DataFrame with columns ["group", "value"]
    """
    rng = np.random.default_rng(seed)
    frames = []
    for label, mu, sd in specs:
        vals = rng.normal(loc=mu, scale=sd, size=n_per_group).astype(float)
        frames.append(pd.DataFrame({"group": label, "value": vals}))
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize(
    "make_df, cat_col, kwargs, expect",
    [
        # 0) Empty input => validated frame empty -> BasePlot.default_descriptive() => {}
        (
            lambda: pd.DataFrame(
                {
                    "value": pd.Series([], dtype=float),
                    "group": pd.Categorical([], categories=["A", "B"]),
                }
            ),
            "group",
            {},
            {
                "descriptive_stats": {},  # BasePlot.default_descriptive()
                "inferential_stats": {},
                "chart_metadata": {
                    "file_name": None,
                    "xlabel": "Effect Size",
                    "ylabel": "Magnitude (0–1)",
                },
            },
        ),
        # 1) Two groups with very similar distributions -> small effect sizes
        (
            lambda: _make_normal_groups([("A", 0.0, 1.0), ("B", 0.1, 1.0)], n_per_group=400, seed=123),
            "group",
            {},
            {
                "descriptive_stats": {
                    "n_groups": 2,
                    "effect_sizes": {
                        "eta_squared": lambda v: isinstance(v, float) and (0.0 <= v <= 0.1),
                        "omega_squared": lambda v: isinstance(v, float) and (-0.001 <= v <= 0.1),
                        "epsilon_squared": lambda v: isinstance(v, float) and (-0.01 <= v <= 0.2),
                    },
                },
            },
        ),
        # 2) Two groups with well-separated means -> large effect sizes
        (
            lambda: _make_normal_groups([("L", -3.0, 1.0), ("R", 3.0, 1.0)], n_per_group=400, seed=7),
            "group",
            {},
            {
                "descriptive_stats": {
                    "n_groups": 2,
                    "effect_sizes": {
                        "eta_squared": lambda v: isinstance(v, float) and (0.5 <= v <= 1.0),
                        "omega_squared": lambda v: isinstance(v, float) and (0.5 <= v <= 1.0),
                        "epsilon_squared": lambda v: isinstance(v, float) and (0.3 <= v <= 1.0),
                    },
                },
            },
        ),
        # 3) Custom file_name flows through chart_metadata
        (
            lambda: _make_normal_groups([("g1", -1.0, 1.0), ("g2", 1.0, 1.0)], n_per_group=250, seed=99),
            "group",
            {"file_name": "effect_sizes_demo.png"},
            {
                "chart_metadata": {"file_name": "effect_sizes_demo.png"},
                "descriptive_stats": {
                    "n_groups": 2,
                    "effect_sizes": {
                        "eta_squared": lambda v: isinstance(v, float) and (0.0 <= v <= 1.0),
                        "omega_squared": lambda v: isinstance(v, float) and (0.0 <= v <= 1.0),
                        "epsilon_squared": lambda v: isinstance(v, float) and (0.0 <= v <= 1.0),
                    },
                },
            },
        ),
    ],
    ids=[
        "empty_df",
        "small_effect",
        "large_effect",
        "custom_filename",
    ],
)
def test_magnitude_effect_size_bar_param(make_df, cat_col, kwargs, expect, tmp_path, assert_plot_metadata):
    df = make_df()

    # if saving is requested, write into tmp_path
    if "file_name" in kwargs:
        kwargs = {**kwargs, "base_dir": tmp_path}

    ctx = MagnitudeEffectSizeBarContext(**kwargs)
    plot = MagnitudeEffectSizeBarPlot(ctx)

    # DF-only API; explicitly point roles to columns
    payload = plot.run(df, cols=[cat_col, "value"], role_map={"x": cat_col, "y": "value"})

    # Reuse shared helper to check partial structures and predicate expectations
    assert_plot_metadata(payload, expect, tmp_path)
