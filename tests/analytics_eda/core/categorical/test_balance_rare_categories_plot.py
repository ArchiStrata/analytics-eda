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
                    "data_source": None,
                    "file_name": None,
                    "title": "Rare Categories of cats",
                    "version": "1.0.0",
                    "xlabel": "Percent of total",
                    "ylabel": "Category"
                },
                "descriptive_stats": {
                    "bars": {},
                    "params": {
                        "threshold_type": "proportion",
                        "threshold_value_count": 0,
                        "threshold_value_prop": 0.0
                    },
                    "subset_count": 0,
                    "total": 0,
                    "total_nonnull": 0
                },
                "draft_descriptive_findings": {},
                "draft_inferential_findings": {},
                "inferential_stats": {}
            },
        ),
        # 1) Proportion threshold (10%) → selects categories with count <= floor(10% * total) (at least 1)
        #   series: A×5, B×3, C×1, D×1 (total=10) → thr_count=1 → rare={C,D}
        (
            lambda: pd.Series(["A"] * 5 + ["B"] * 3 + ["C"] * 1 + ["D"] * 1, name="letters"),
            {"extreme_lower_bound": 0.10, "file_name": "rare_prop.png"},
            {
                "chart_metadata": {
                    "data_source": None,
                    "file_name": "rare_prop.png",
                    "title": "Rare Categories of letters",
                    "version": "1.0.0",
                    "xlabel": "Percent of total",
                    "ylabel": "Category"
                },
                "descriptive_stats": {
                    "bars": {
                        "C": {
                            "count": 1,
                            "pct_of_nonnull": 0.1
                        },
                        "D": {
                            "count": 1,
                            "pct_of_nonnull": 0.1
                        }
                    },
                    "denominator_key": "pct_of_nonnull",
                    "input_categories": 2,
                    "input_nonzero_categories": 2,
                    "n_bars_rendered": 2,
                    "nonzero_categories": 2,
                    "params": {
                        "bar_height_source": "values",
                        "bar_sort_descending": False,
                        "extreme_lower_bound": 0.1,
                        "max_display_bars": 15,
                        "other_label": "Other",
                        "show_count_in_bar_label": False,
                        "show_value_in_bar_label": True,
                        "threshold_type": "proportion",
                        "threshold_value_count": 1,
                        "threshold_value_prop": 0.1
                    },
                    "pct_subset": 0.2,
                    "subset_count": 2,
                    "total": 10,
                    "total_nonnull": 10,
                    "unique_categories_total": 4
                },
                "draft_descriptive_findings": {
                    "context": "Base = 10 non-null; K = 4 total categories",
                    "primary_finding": "Rare categories (threshold=proportion: ≤ 10.0% or ≤ 1 count) found: 2; they account for 20.0% of rows (2).",
                    "secondary_finding": None
                },
                "draft_inferential_findings": {},
                "inferential_stats": {}
            },
        ),
        # 2) Count threshold (<=2) with max_display_bars=3 → cap smallest three
        #   series: A×4, B×2, C×2, D×1, E×1 (total=10) → rare={B:2,C:2,D:1,E:1} → after cap: three smallest counts
        (
            lambda: pd.Series(["A"] * 4 + ["B"] * 2 + ["C"] * 2 + ["D"] * 1 + ["E"] * 1, name="mix"),
            {"extreme_lower_bound": 2, "max_display_bars": 3, "file_name": "rare_count_cap.png"},
            {
                "chart_metadata": {
                    "data_source": None,
                    "file_name": "rare_count_cap.png",
                    "title": "Rare Categories of mix",
                    "version": "1.0.0",
                    "xlabel": "Percent of total",
                    "ylabel": "Category"
                },
                "descriptive_stats": {
                    "bars": {
                        "D": {
                            "count": 1,
                            "pct_of_nonnull": 0.1
                        },
                        "E": {
                            "count": 1,
                            "pct_of_nonnull": 0.1
                        },
                        "Other (k=2)": {
                            "_is_other": True,
                            "count": 4,
                            "pct_of_nonnull": 0.4
                        }
                    },
                    "denominator_key": "pct_of_nonnull",
                    "input_categories": 4,
                    "input_nonzero_categories": 4,
                    "n_bars_rendered": 3,
                    "nonzero_categories": 3,
                    "params": {
                        "bar_height_source": "values",
                        "bar_sort_descending": False,
                        "extreme_lower_bound": 2,
                        "max_display_bars": 3,
                        "other_label": "Other",
                        "show_count_in_bar_label": False,
                        "show_value_in_bar_label": True,
                        "threshold_type": "count",
                        "threshold_value_count": 2,
                        "threshold_value_prop": 0.2
                    },
                    "pct_subset": 0.6,
                    "subset_count": 6,
                    "total": 10,
                    "total_nonnull": 10,
                    "unique_categories_total": 5
                },
                "draft_descriptive_findings": {
                    "context": "Base = 10 non-null; K = 5 total categories",
                    "primary_finding": "Rare categories (threshold=count: ≤ 20.0% or ≤ 2 count) found: 4; they account for 60.0% of rows (6).",
                    "secondary_finding": None
                },
                "draft_inferential_findings": {},
                "inferential_stats": {}
            },
        ),
        # 3) No rare categories under threshold → skip save even if file_name given
        #   series: A×3, B×3 (total=6); threshold count=0 → none <= 0 → n_rare=0
        (
            lambda: pd.Series(["A"] * 3 + ["B"] * 3, name="pairs"),
            {"extreme_lower_bound": 0, "file_name": "should_not_save.png"},
            {
                "chart_metadata": {
                    "data_source": None,
                    "file_name": None,
                    "title": "Rare Categories of pairs",
                    "version": "1.0.0",
                    "xlabel": "Percent of total",
                    "ylabel": "Category"
                },
                "descriptive_stats": {
                    "bars": {},
                    "denominator_key": "pct_of_nonnull",
                    "error": "no categories to display",
                    "input_categories": 0,
                    "input_nonzero_categories": 0,
                    "n_bars_rendered": 0,
                    "nonzero_categories": 0,
                    "params": {
                        "bar_height_source": "values",
                        "bar_sort_descending": False,
                        "extreme_lower_bound": 0,
                        "max_display_bars": 15,
                        "other_label": "Other",
                        "show_count_in_bar_label": False,
                        "show_value_in_bar_label": True,
                        "threshold_type": "proportion",
                        "threshold_value_count": 0,
                        "threshold_value_prop": 0.0
                    },
                    "pct_subset": 0.0,
                    "skip_plot": True,
                    "subset_count": 0,
                    "total": 6,
                    "total_nonnull": 6,
                    "unique_categories_total": 2
                },
                "draft_descriptive_findings": {
                    "context": "Base = 6 non-null; K = 2 total categories",
                    "primary_finding": "No categories meet the rare threshold.",
                    "secondary_finding": None
                },
                "draft_inferential_findings": {},
                "inferential_stats": {}
            },
        ),
        # 4) Custom labels + data_source (no save)
        (
            lambda: pd.Series(["cat", "dog", "dog", "bird", "bird", "bird"], name="animals"),
            {"xlabel": "Category Name", "ylabel": "Frequency", "data_source": "UnitTest", "extreme_lower_bound": 2},
            {
                "chart_metadata": {
                    "data_source": "UnitTest",
                    "file_name": None,
                    "title": "Rare Categories of animals",
                    "version": "1.0.0",
                    "xlabel": "Category Name",
                    "ylabel": "Frequency"
                },
                "descriptive_stats": {
                    "bars": {
                        "cat": {
                            "count": 1,
                            "pct_of_nonnull": 0.16666666666666666
                        },
                        "dog": {
                            "count": 2,
                            "pct_of_nonnull": 0.3333333333333333
                        }
                    },
                    "denominator_key": "pct_of_nonnull",
                    "input_categories": 2,
                    "input_nonzero_categories": 2,
                    "n_bars_rendered": 2,
                    "nonzero_categories": 2,
                    "params": {
                        "bar_height_source": "values",
                        "bar_sort_descending": False,
                        "extreme_lower_bound": 2,
                        "max_display_bars": 15,
                        "other_label": "Other",
                        "show_count_in_bar_label": False,
                        "show_value_in_bar_label": True,
                        "threshold_type": "count",
                        "threshold_value_count": 2,
                        "threshold_value_prop": 0.3333333333333333
                    },
                    "pct_subset": 0.5,
                    "subset_count": 3,
                    "total": 6,
                    "total_nonnull": 6,
                    "unique_categories_total": 3
                },
                "draft_descriptive_findings": {
                    "context": "Base = 6 non-null; K = 3 total categories",
                    "primary_finding": "Rare categories (threshold=count: ≤ 33.3% or ≤ 2 count) found: 2; they account for 50.0% of rows (3).",
                    "secondary_finding": None
                },
                "draft_inferential_findings": {},
                "inferential_stats": {}
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
