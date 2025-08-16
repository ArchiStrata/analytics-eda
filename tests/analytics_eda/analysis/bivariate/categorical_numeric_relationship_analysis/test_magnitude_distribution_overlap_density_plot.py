import numpy as np
import pandas as pd
import pytest

from analytics_eda.analysis.bivariate.categorical_numeric_relationship_analysis.magnitude_distribution_overlap_density_plot import (
    MagnitudeDistributionOverlapDensityContext,
    MagnitudeDistributionOverlapDensityPlot,
)

# --- small helper to synthesize grouped normal data ---
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
    "make_df, cat_col, kwargs, expect_keys",
    [
        # 0) Empty input => validated frame empty -> BasePlot.default_descriptive() => {}
        (
            lambda: pd.DataFrame({
                "value": pd.Series([], dtype=float),
                "group": pd.Categorical([], categories=["A", "B"])
            }),
            "group",
            {},
            {
                "descriptive_stats": {},     # default on empty
                "inferential_stats": {},
                "chart_metadata": {
                    "file_name": None,
                    "xlabel": "Value",
                    "ylabel": "Density",
                },
            },
        ),
        # 1) Two groups, same distribution -> high overlap, small Bhattacharyya
        (
            lambda: _make_normal_groups([("A", 0.0, 1.0), ("B", 0.0, 1.0)], n_per_group=300, seed=42),
            "group",
            {},
            {
                "descriptive_stats": {
                    "n_groups": 2,
                },
            },
        ),
        # 2) Two groups, well separated -> low overlap, large Bhattacharyya
        (
            lambda: _make_normal_groups([("L", -4.0, 1.0), ("R",  4.0, 1.0)], n_per_group=300, seed=7),
            "group",
            {},
            {
                "descriptive_stats": {
                    "n_groups": 2,
                },
            },
        ),
        # 3) Five groups -> faceting path; also check grid size and #pairs
        (
            lambda: _make_normal_groups([("g1", -4, 1.0), ("g2", -2, 1.0), ("g3", 0, 1.0), ("g4", 2, 1.0), ("g5", 4, 1.0)],
                                        n_per_group=200, seed=123),
            "group",
            {"file_name": "overlap_five_groups.png", "grid_size": 300},
            {
                "descriptive_stats": {
                    "n_groups": 5,
                },
                "chart_metadata": {
                    "file_name": "overlap_five_groups.png",
                },
            },
        ),
    ],
    ids=["empty_df", "two_groups_high_overlap", "two_groups_low_overlap", "five_groups_faceted"],
)
def test_magnitude_distribution_overlap_density_param(make_df, cat_col, kwargs, expect_keys, tmp_path, assert_plot_metadata):
    df = make_df()

    if "file_name" in kwargs:
        kwargs = {**kwargs, "save_path": tmp_path}

    ctx = MagnitudeDistributionOverlapDensityContext(**kwargs)
    plot = MagnitudeDistributionOverlapDensityPlot(ctx)

    payload = plot.run(df, cols=[cat_col, "value"], role_map={"x": cat_col, "y": "value"})

    # Basic metadata/key expectations (reuse your helper to match partial dicts)
    assert_plot_metadata(payload, expect_keys, tmp_path)
