import numpy as np
import pandas as pd
import pytest

from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis import (
    MagnitudeCentralTendencyAnovaKruskalContext,
    MagnitudeCentralTendencyAnovaKruskalPlot,
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
        # 0) Empty input → default_descriptive {} and basic metadata
        (
            lambda: pd.DataFrame({
                "value": pd.Series([], dtype=float),
                "group": pd.Categorical([], categories=["A", "B"])
            }),
            "group",
            {},
            {
                "descriptive_stats": {},   # BasePlot.default_descriptive()
                "inferential_stats": {},
                "chart_metadata": {
                    "file_name": None,
                    "xlabel": "Group",
                    "ylabel": "Value",
                },
            },
        ),
        # 1) ANOVA mode, equal means -> p should be a float; don't force threshold (data-driven)
        (
            lambda: _make_normal_groups([("A", 0.0, 1.0), ("B", 0.0, 1.0)], n_per_group=400, seed=42),
            "group",
            {"mode": "anova"},
            {
                "descriptive_stats": {
                    "n_groups": 2,
                    "means": (lambda v: isinstance(v, list) and len(v) == 2),
                    "mean_ci_lo": (lambda v: isinstance(v, list) and len(v) == 2),
                    "mean_ci_hi": (lambda v: isinstance(v, list) and len(v) == 2),
                },
                "inferential_stats": {
                    "anova": {
                        "statistic": (lambda v: isinstance(v, float)),
                        "p_value": (lambda v: isinstance(v, float)),
                        "reject": (lambda v: isinstance(v, bool)),
                        "alpha": (lambda v: isinstance(v, float)),
                    },
                    "kruskal": {
                        "statistic": (lambda v: isinstance(v, float)),
                        "p_value": (lambda v: isinstance(v, float)),
                        "reject": (lambda v: isinstance(v, bool)),
                        "alpha": (lambda v: isinstance(v, float)),
                    },
                },
            },
        ),
        # 2) ANOVA mode, separated means -> p very small & reject True
        (
            lambda: _make_normal_groups([("L", -2.0, 1.0), ("R", 2.0, 1.0)], n_per_group=300, seed=7),
            "group",
            {"mode": "anova", "file_name": "anova_sep_means.png"},
            {
                "chart_metadata": {"file_name": "anova_sep_means.png"},
                "descriptive_stats": {"n_groups": 2},
                "inferential_stats": {
                    "anova": {
                        "p_value": (lambda v: isinstance(v, float) and v < 1e-3),
                        "reject": (lambda v: v is True),
                    }
                },
            },
        ),
        # 3) Kruskal mode, shifted medians -> p very small & reject True
        (
            lambda: _make_normal_groups([("G1", 0.0, 1.0), ("G2", 1.0, 1.0)], n_per_group=350, seed=123),
            "group",
            {"mode": "kruskal", "random_state": 123, "bootstrap_iters": 800},
            {
                "descriptive_stats": {
                    "n_groups": 2,
                    "medians": (lambda v: isinstance(v, list) and len(v) == 2),
                    "median_ci_lo": (lambda v: isinstance(v, list) and len(v) == 2),
                    "median_ci_hi": (lambda v: isinstance(v, list) and len(v) == 2),
                },
                "inferential_stats": {
                    "kruskal": {
                        "p_value": (lambda v: isinstance(v, float) and v < 1e-3),
                        "reject": (lambda v: v is True),
                    }
                },
            },
        ),
        # 4) Five groups → just exercise plot path & basic keys
        (
            lambda: _make_normal_groups(
                [("g1", -4, 1.0), ("g2", -2, 1.0), ("g3", 0, 1.0), ("g4", 2, 1.0), ("g5", 4, 1.0)],
                n_per_group=150, seed=999
            ),
            "group",
            {"mode": "anova", "file_name": "magnitude_five_groups.png"},
            {
                "descriptive_stats": {"n_groups": 5},
                "chart_metadata": {"file_name": "magnitude_five_groups.png"},
            },
        ),
    ],
    ids=[
        "empty_df",
        "anova_equal_means",
        "anova_separated_means",
        "kruskal_shifted_medians",
        "five_groups",
    ],
)
def test_magnitude_central_tendency_anova_kruskal_param(make_df, cat_col, kwargs, expect, tmp_path, assert_plot_metadata):
    df = make_df()

    # if saving is requested, write into tmp_path
    if "file_name" in kwargs:
        kwargs = {**kwargs, "save_path": tmp_path}

    ctx = MagnitudeCentralTendencyAnovaKruskalContext(**kwargs)
    plot = MagnitudeCentralTendencyAnovaKruskalPlot(ctx)

    payload = plot.run(df, cols=[cat_col, "value"], role_map={"x": cat_col, "y": "value"})

    # Reuse your helper to assert partial metadata/descriptive/inferential expectations
    assert_plot_metadata(payload, expect, tmp_path)
