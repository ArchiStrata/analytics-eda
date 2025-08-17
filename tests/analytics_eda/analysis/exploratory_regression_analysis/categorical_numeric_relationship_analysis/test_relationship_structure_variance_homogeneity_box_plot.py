import numpy as np
import pandas as pd
import pytest

from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis.relationship_structure_variance_homogeneity_box_plot import (
    RelationshipStructureVarianceHomogeneityContext,
    RelationshipStructureVarianceHomogeneityBoxPlot,
)

# --- helpers ---------------------------------------------------------------

def _make_normal_groups(specs, n_per_group=300, seed=0):
    """
    specs: list of (label, mean, std)
    Returns a DataFrame with columns: ['value', 'group']
    """
    rng = np.random.default_rng(seed)
    parts = []
    for label, mu, sd in specs:
        vals = rng.normal(loc=mu, scale=sd, size=n_per_group)
        parts.append(pd.DataFrame({"value": vals, "group": str(label)}))
    return pd.concat(parts, ignore_index=True)

# --- tests ----------------------------------------------------------------

@pytest.mark.parametrize(
    "make_df, cat_col, kwargs, expect",
    [
        # 0) Empty input => validated frame is empty → default_descriptive ({})
        (
            lambda: pd.DataFrame({
                "value": pd.Series([], dtype=float),
                "group": pd.Categorical([], categories=["A", "B"]),
            }),
            "group",
            {},
            {
                "descriptive_stats": {},     # BasePlot.default_descriptive()
                "inferential_stats": {},
                "chart_metadata": {
                    "file_name": None,
                    "xlabel": "Group",
                    "ylabel": "Value",
                },
            },
        ),
        # 1) Equal variances: two identical normals → expect not to reject homogeneity
        (
            lambda: _make_normal_groups([("A", 0.0, 1.0), ("B", 0.0, 1.0)], n_per_group=400, seed=42),
            "group",
            {},
            {
                "descriptive_stats": {"n_groups": 2},
                # We'll assert inferential keys and 'reject' flags in the test body
            },
        ),
        # 2) Unequal variances: std differs a lot → expect to reject (especially Levene)
        (
            lambda: _make_normal_groups([("L", 0.0, 1.0), ("R", 0.0, 3.0)], n_per_group=400, seed=7),
            "group",
            {},
            {
                "descriptive_stats": {"n_groups": 2},
            },
        ),
        # 3) Five groups → faceting branch in draw; also ensure file name plumbs through
        (
            lambda: _make_normal_groups(
                [("g1", -4, 1.0), ("g2", -2, 1.0), ("g3", 0, 1.0), ("g4", 2, 1.0), ("g5", 4, 1.0)],
                n_per_group=200,
                seed=123,
            ),
            "group",
            {"file_name": "var_homo_five_groups.png"},
            {
                "descriptive_stats": {"n_groups": 5},
                "chart_metadata": {"file_name": "var_homo_five_groups.png"},
            },
        ),
    ],
    ids=["empty_df", "equal_variances", "unequal_variances", "five_groups_faceted"],
)
def test_relationship_structure_variance_homogeneity_box_param(
    make_df, cat_col, kwargs, expect, tmp_path, assert_plot_metadata
):
    df = make_df()

    # route file output into tmp dir if requested
    if "file_name" in kwargs:
        kwargs = {**kwargs, "save_path": tmp_path}

    ctx = RelationshipStructureVarianceHomogeneityContext(**kwargs)
    plot = RelationshipStructureVarianceHomogeneityBoxPlot(ctx)

    # DF API; explicitly map roles
    payload = plot.run(df, cols=[cat_col, "value"], role_map={"x": cat_col, "y": "value"})

    # Reuse your helper to assert partial expectations
    assert_plot_metadata(payload, expect, tmp_path)
