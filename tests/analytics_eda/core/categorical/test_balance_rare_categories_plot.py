import math
import pytest
import pandas as pd

from analytics_eda.core.categorical import BalanceRareCategoriesPlot, BalanceRareCategoriesContext


@pytest.mark.parametrize(
    "series_factory, expected_exc, match",
    [
        # Not a pandas Series
        (lambda: ["a", "b", "c"], TypeError, r"data must be a pandas Series or DataFrame"),

        # Not categorical/object dtype
        (lambda: pd.Series([1, 2, 3], name="numeric"), TypeError,
         r"must be categorical.*for categorical analysis"),

        # Missing name (None)
        (lambda: pd.Series(["x", "y", "z"], dtype="category"), ValueError,
         r"must have a non-empty 'name'"),

        # Blank/whitespace name
        (lambda: pd.Series(["x", "y", "z"], dtype="object", name=" "), ValueError,
         r"must have a non-empty 'name'"),
    ],
    ids=["not_series", "bad_dtype", "missing_name", "blank_name"],
)
def test_validate_categorical_named_series_errors(series_factory, expected_exc, match):
    s = series_factory()
    with pytest.raises(expected_exc, match=match):
        ctx = BalanceRareCategoriesContext()
        plot = BalanceRareCategoriesPlot(ctx)

        plot.run(s)


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) Empty series -> early skip; no saved file
        (
            lambda: pd.Series(pd.Categorical([], categories=["A", "B"]), name="cats"),
            {},
            {
                "chart_metadata": {
                    "file_name": None,
                    "xlabel": "Category",
                    "ylabel": "Count",
                },
                "descriptive_stats": {
                    "total": 0,
                    "k": 0,
                    "threshold_type": "proportion",
                    "threshold_value_count": 0,
                    "threshold_value_prop": 0.0,
                    "n_rare": 0,
                    "rare_categories": [],
                    "rare_counts": [],
                },
            },
        ),
        # 1) Proportion threshold (10%) → selects categories with count <= floor(10% * total) (at least 1)
        #   series: A×5, B×3, C×1, D×1 (total=10) → thr_count=1 → rare={C,D}
        (
            lambda: pd.Series(["A"] * 5 + ["B"] * 3 + ["C"] * 1 + ["D"] * 1, name="letters"),
            {"extreme_lower_bound": 0.10, "file_name": "rare_prop.png"},
            {
                "chart_metadata": {"file_name": "rare_prop.png"},
                "descriptive_stats": {
                    "total": 10,
                    "k": 4,
                    "threshold_type": "proportion",
                    "threshold_value_count": 1,
                    "threshold_value_prop": lambda p: abs(p - 0.10) < 1e-12,
                    "n_rare": 2,
                    "rare_categories": lambda xs: set(xs) == {"C", "D"},
                    "rare_counts":      lambda xs: sorted(xs) == [1, 1],
                },
            },
        ),
        # 2) Count threshold (<=2) with max_bars=3 → cap smallest three
        #   series: A×4, B×2, C×2, D×1, E×1 (total=10) → rare={B:2,C:2,D:1,E:1} → after cap: three smallest counts
        (
            lambda: pd.Series(["A"] * 4 + ["B"] * 2 + ["C"] * 2 + ["D"] * 1 + ["E"] * 1, name="mix"),
            {"extreme_lower_bound": 2, "max_bars": 3, "file_name": "rare_count_cap.png"},
            {
                "chart_metadata": {"file_name": "rare_count_cap.png"},
                "descriptive_stats": {
                    "total": 10,
                    "k": 5,
                    "threshold_type": "count",
                    "threshold_value_count": 2,
                    "threshold_value_prop": lambda p: abs(p - 0.2) < 1e-12,
                    "n_rare": 3,  # capped from 4 → 3
                    # There are two with count=1 (D,E) and two with count=2 (B,C); after capping to 3 smallest,
                    # we must include both 1s and only one of the 2s. Order may vary → check counts multiset & subset.
                    "rare_counts": lambda xs: sorted(xs) == [1, 1, 2],
                    "rare_categories": lambda xs: set(xs).issubset({"B", "C", "D", "E"}) and len(xs) == 3,
                },
            },
        ),
        # 3) No rare categories under threshold → skip save even if file_name given
        #   series: A×3, B×3 (total=6); threshold count=0 → none <= 0 → n_rare=0
        (
            lambda: pd.Series(["A"] * 3 + ["B"] * 3, name="pairs"),
            {"extreme_lower_bound": 0, "file_name": "should_not_save.png"},
            {
                "chart_metadata": {"file_name": None},
                "descriptive_stats": {
                    "total": 6,
                    "k": 2,
                    "threshold_type": "proportion",
                    "threshold_value_count": 0,
                    "threshold_value_prop": lambda p: p == 0.0,
                    "n_rare": 0,
                    "rare_categories": lambda xs: xs == [],
                    "rare_counts": lambda xs: xs == [],
                },
            },
        ),
        # 4) Custom labels + data_source (no save)
        (
            lambda: pd.Series(["cat", "dog", "dog", "bird", "bird", "bird"], name="animals"),
            {"xlabel": "Category Name", "ylabel": "Frequency", "data_source": "UnitTest", "extreme_lower_bound": 2},
            {
                "chart_metadata": {
                    "xlabel": "Category Name",
                    "ylabel": "Frequency",
                    "data_source": "UnitTest",
                },
                "descriptive_stats": {
                    "total": 6,
                    "k": 3,
                    "threshold_type": "count",
                    "threshold_value_count": 2,
                    "threshold_value_prop": lambda p: abs(p - (2 / 6)) < 1e-12,
                    # counts: bird=3, dog=2, cat=1 → rare: cat(1), dog(2)
                    "n_rare": 2,
                    "rare_categories": lambda xs: set(xs) == {"cat", "dog"},
                    "rare_counts":      lambda xs: sorted(xs) == [1, 2],
                },
            },
        ),
    ],
    ids=["empty_skips", "prop_threshold_saves", "count_threshold_capped_saves", "no_rare_skips", "custom_labels"],
)
def test_plot_balance_rare_categories_param(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    # If saving, route to tmp_path
    if "file_name" in kwargs:
        kwargs = {**kwargs, "save_path": tmp_path}

    ctx = BalanceRareCategoriesContext(**kwargs)
    plot = BalanceRareCategoriesPlot(ctx)

    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
