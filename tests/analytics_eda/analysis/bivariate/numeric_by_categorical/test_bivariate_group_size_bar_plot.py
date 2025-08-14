import pandas as pd
import numpy as np
import pytest

from analytics_eda.analysis.bivariate.numeric_by_categorical.bivariate_group_size_bar_plot import (
    BivariateGroupSizeBarContext, BivariateGroupSizeBarPlot
)

@pytest.mark.parametrize(
    "make_df, cat_col, kwargs, expect",
    [
        # 0) Empty input → validated frame is empty => default_descriptive ({})
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
                    "ylabel": "Total",
                },
            },
        ),
        # 1) Balanced distribution: A,B,C each appear twice
        (
            lambda: pd.DataFrame({
                "value": np.arange(6, dtype=float),
                "letters": ["A", "B", "C", "A", "B", "C"],
            }),
            "letters",
            {"file_name": "group_sizes_balanced.png"},
            {
                "chart_metadata": {
                    "file_name": "group_sizes_balanced.png",
                },
                "descriptive_stats": {
                    "total": 15.0,
                    "n_groups": 3,
                },
            },
        ),
        # 2) Unbalanced distribution: X(10), Y(2), Z(1)
        (
            lambda: pd.DataFrame({
                "value": np.arange(13, dtype=float),
                "choices": ["X"] * 10 + ["Y"] * 2 + ["Z"] * 1,
            }),
            "choices",
            {"file_name": "group_sizes_unbalanced.png"},
            {
                "chart_metadata": {
                    "file_name": "group_sizes_unbalanced.png",
                },
                "descriptive_stats": {
                    "total": 78.0,
                    "n_groups": 3,
                },
            },
        ),
        # 3) Custom chart metadata values (labels, data_source)
        (
            lambda: pd.DataFrame({
                "value": [1.0, 2.0, 3.0, 4.0],
                "animals": ["cat", "dog", "dog", "bird"],
            }),
            "animals",
            {"xlabel": "Animal", "ylabel": "Count", "data_source": "UnitTest", "file_name": "group_sizes_custom.png"},
            {
                "chart_metadata": {
                    "xlabel": "Animal",
                    "ylabel": "Count",
                    "data_source": "UnitTest",
                    "file_name": "group_sizes_custom.png"
                },
                "descriptive_stats": {
                    "total": 10.0,
                    "n_groups": 3,
                },
            },
        ),
    ],
    ids=[
        "empty_df",
        "balanced_groups",
        "unbalanced_groups",
        "custom_labels",
    ],
)
def test_bivariate_group_size_bar_param(make_df, cat_col, kwargs, expect, tmp_path, assert_plot_metadata):
    df = make_df()

    # if saving is requested, write into tmp_path
    if "file_name" in kwargs:
        kwargs = {**kwargs, "save_path": tmp_path}

    ctx = BivariateGroupSizeBarContext(**kwargs)
    plot = BivariateGroupSizeBarPlot(ctx)

    # DF-only API; explicitly point x-role to the categorical column
    payload = plot.run(df, cols=[cat_col, "value"], role_map={"x": cat_col})

    # Reuse your helper to assert partial metadata/descriptive expectations
    assert_plot_metadata(payload, expect, tmp_path)
