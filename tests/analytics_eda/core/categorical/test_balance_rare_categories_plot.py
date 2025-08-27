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
                    "xlabel": "Percent of total",
                    "ylabel": "Category",
                },
                "descriptive_stats": {
                    "params": {
                        "threshold_type": "proportion",
                        "threshold_value_count": 0,
                        "threshold_value_prop": 0.0,
                    },
                    "total": 0,
                    "k": 0,
                    "n_rare": 0
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
                    "params": {
                        "threshold_type": "proportion",
                        "threshold_value_count": 1,
                        "threshold_value_prop": lambda p: abs(p - 0.10) < 1e-12,
                    },
                    "total": 10,
                    "k": 4,
                    "n_rare": 2,
                },
            },
        ),
        # 2) Count threshold (<=2) with max_display_bars=3 → cap smallest three
        #   series: A×4, B×2, C×2, D×1, E×1 (total=10) → rare={B:2,C:2,D:1,E:1} → after cap: three smallest counts
        (
            lambda: pd.Series(["A"] * 4 + ["B"] * 2 + ["C"] * 2 + ["D"] * 1 + ["E"] * 1, name="mix"),
            {"extreme_lower_bound": 2, "max_display_bars": 3, "file_name": "rare_count_cap.png"},
            {
                "chart_metadata": {"file_name": "rare_count_cap.png"},
                "descriptive_stats": {
                    "params": {
                        "threshold_type": "count",
                        "threshold_value_count": 2,
                        "threshold_value_prop": lambda p: abs(p - 0.2) < 1e-12,
                    },
                    "total": 10,
                    "k": 5,
                    "n_rare": 4,
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
                    "params": {
                        "threshold_type": "proportion",
                        "threshold_value_count": 0,
                        "threshold_value_prop": lambda p: p == 0.0,
                    },
                    "total": 6,
                    "k": 2,
                    "n_rare": 0,
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
                    "params": {
                        "threshold_type": "count",
                        "threshold_value_count": 2,
                        "threshold_value_prop": lambda p: abs(p - (2 / 6)) < 1e-12,
                    },
                    "total": 6,
                    "k": 3,
                    # counts: bird=3, dog=2, cat=1 → rare: cat(1), dog(2)
                    "n_rare": 2,
                },
            },
        ),
    ],
    ids=["empty_skips", "prop_threshold_saves", "count_threshold_capped_saves", "no_rare_skips", "custom_labels"],
)
def test_balance_rare_categories_plot_data_driven(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    # If saving, route to tmp_path
    if "file_name" in kwargs:
        kwargs = {**kwargs, "save_path": tmp_path}

    ctx = BalanceRareCategoriesContext(**kwargs)
    plot = BalanceRareCategoriesPlot(ctx)

    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
