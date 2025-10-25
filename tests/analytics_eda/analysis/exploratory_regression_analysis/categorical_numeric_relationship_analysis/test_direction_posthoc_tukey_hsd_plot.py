import numpy as np
import pandas as pd
import pytest

from analytics_eda.analysis.exploratory_regression_analysis.categorical_numeric_relationship_analysis.direction_posthoc_tukey_hsd_plot import (
    DirectionPosthocTukeyHsdContext,
    DirectionPosthocTukeyHsdPlot,
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


def _pairs_predicate(expected_len=None, expect_reject=None):
    """
    Returns a predicate over the desc['pairs'] list that checks:
      - list type, non-empty (if expected_len provided, checks exact length)
      - each item has keys: diff, ci_low, ci_high, reject
      - optional reject boolean expectation (True/False) for all pairs (useful for 2-group tests)
    """
    def _pred(pairs):
        if not isinstance(pairs, list):
            return False
        if expected_len is not None and len(pairs) != expected_len:
            return False
        if len(pairs) == 0 and expected_len == 0:
            return True
        if len(pairs) == 0:
            return False
        for d in pairs:
            if not isinstance(d, dict):
                return False
            for k in ("diff", "ci_low", "ci_high", "reject"):
                if k not in d:
                    return False
            if not (isinstance(d["diff"], float) and isinstance(d["ci_low"], float) and isinstance(d["ci_high"], float)):
                return False
            if not isinstance(d["reject"], bool):
                return False
            if not (d["ci_low"] <= d["ci_high"]):
                return False
            if expect_reject is not None and d["reject"] is not expect_reject:
                return False
        return True
    return _pred


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
                    "xlabel": "Mean difference",
                    "ylabel": "Comparison",
                },
            },
        ),
        # 1) Two groups with very similar means -> Tukey likely non-significant (reject=False)
        (
            lambda: _make_normal_groups(
                [("A", 0.0, 1.0), ("B", 0.10, 1.0)], n_per_group=400, seed=123
            ),
            "group",
            {},
            {
                "descriptive_stats": {
                    "n_groups": 2,
                    "pairs": _pairs_predicate(expected_len=1, expect_reject=False),
                    "alpha": (lambda v: isinstance(v, float) and 0 < v <= 0.1),  # default 0.05
                },
            },
        ),
        # 2) Two groups with well-separated means -> Tukey significant (reject=True)
        (
            lambda: _make_normal_groups(
                [("L", -3.0, 1.0), ("R", 3.0, 1.0)], n_per_group=300, seed=7
            ),
            "group",
            {},
            {
                "descriptive_stats": {
                    "n_groups": 2,
                    "pairs": _pairs_predicate(expected_len=1, expect_reject=True),
                },
            },
        ),
        # 3) Three groups -> 3 pairwise comparisons; also check file_name propagation
        (
            lambda: _make_normal_groups(
                [("g1", -1.0, 1.0), ("g2", 0.0, 1.0), ("g3", 1.0, 1.0)], n_per_group=250, seed=99
            ),
            "group",
            {"file_name": "tukey_triplet.png"},
            {
                "chart_metadata": {"file_name": "tukey_triplet.png"},
                "descriptive_stats": {
                    "n_groups": 3,
                    "pairs": _pairs_predicate(expected_len=3),  # 3 choose 2 = 3
                },
            },
        ),
    ],
    ids=[
        "empty_df",
        "two_groups_nonsig",
        "two_groups_sig",
        "three_groups_triplet",
    ],
)
def test_direction_posthoc_tukey_hsd_param(make_df, cat_col, kwargs, expect, tmp_path, assert_plot_metadata):
    df = make_df()

    # route file output to tmp_path if requested
    if "file_name" in kwargs:
        kwargs = {**kwargs, "save_path": tmp_path}

    ctx = DirectionPosthocTukeyHsdContext(**kwargs)
    plot = DirectionPosthocTukeyHsdPlot(ctx)

    # DF-only API; explicitly point roles to columns
    payload = plot.run(df, cols=[cat_col, "value"], role_map={"x": cat_col, "y": "value"})

    # Use shared helper to validate partial structures and predicate expectations
    assert_plot_metadata(payload, expect, tmp_path)
