from dataclasses import field
import pytest
import pandas as pd

from analytics_eda.core.categorical import FrequencyParetoPlot, FrequencyParetoContext
from analytics_eda.core.visualization.context.plot_context import AxisFormat

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
        ctx = FrequencyParetoContext()
        plot = FrequencyParetoPlot(ctx)

        plot.run(s)

@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) EMPTY categorical
        (
            lambda: pd.Series(pd.Categorical([], categories=["A", "B"]), name="cats"),
            {},
            {
                "chart_metadata": {
                    "title": "Pareto Chart of cats",
                    "file_name": None,
                    "version": "1.0.0",
                    "xlabel": "Share of total",
                    "ylabel": "Category",
                },
                "descriptive_stats": {
                    "bars": {},
                    "subset_count": 0,
                    "total": 0,
                    "total_nonnull": 0
                },
                "draft_descriptive_findings": {},
                "draft_inferential_findings": {},
                "inferential_stats": {}
            },
        ),
        # 1) other_min_count groups small categories into "Others" (save)
        (
            (lambda vals=["A", "A", "A", "B", "C"]:
                pd.Series(vals, name="cats", dtype="object")),
            {
                "is_orientation_vertical": True, 
                "xlabel": "Category", 
                "x_format": AxisFormat(kind="category"), 
                "ylabel": "Share of total", 
                "y_format": AxisFormat(kind="percent", decimals=1, percent_scale_0to1=True), 
                "other_min_count": 2, 
                "file_name": "grouping_to_others.png"
            },
            {
                "chart_metadata": {
                    "file_name": "grouping_to_others.png",
                    "title": "Pareto Chart of cats",
                    "version": "1.0.0",
                    "xlabel": "Category",
                    "ylabel": "Share of total"
                },
                "descriptive_stats": {
                    "bars": {
                        "A": {
                            "count": 3,
                            "pct_of_total": 0.6
                        },
                        "Other (k=2)": {
                            "_is_other": True,
                            "count": 2,
                            "k_agg": 2,
                            "pct_of_total": 0.4
                        }
                    },
                    "cumulative_count_at_threshold": 5,
                    "denominator_key": "pct_of_total",
                    "input_categories": 3,
                    "input_nonzero_categories": 3,
                    "n_bars_rendered": 2,
                    "nonzero_categories": 2,
                    "params": {
                        "bar_height_source": "values",
                        "bar_sort_descending": True,
                        "bar_top_include_ties": True,
                        "bar_top_n": 1,
                        "max_display_bars": 15,
                        "other_label": "Other",
                        "other_label_format": "{label} (k={k_agg})",
                        "other_display": "Other (k=2)",
                        "other_min_count": 2,
                        "other_respect_existing": True,
                        "pareto_mode": "shared",
                        "show_count_in_bar_label": False,
                        "show_value_in_bar_label": True
                    },
                    "pct_subset": 1.0,
                    "subset_count": 5,
                    "threshold_idx": 1,
                    "threshold_pct": 80.0,
                    "top_labels": [
                        "A"
                    ],
                    "total": 5,
                    "total_nonnull": 5,
                    "unique_categories_total": 3
                },
                "draft_descriptive_findings": {
                    "context": "N = 5 values across 3 categories",
                    "primary_finding": "≈80% of occurrences are covered by 'A' and Other (k=2).",
                    "secondary_finding": "Top category: 'A' at 60.0%. Other (k=2) contributes 40.0% within the threshold."
                },
                "draft_inferential_findings": {},
                "inferential_stats": {}
            },
        ),
        # 2) horizontal orientation (save)
        (
            (lambda vals=["A", "A", "B", "C", "C", "C"]:
                pd.Series(vals, name="cats", dtype="object")),
            {"file_name": "horizontal_orientation.png"},
            {
                "chart_metadata": {"file_name": "horizontal_orientation.png"},
                "descriptive_stats": {
                    "bars": {
                        "A": {
                            "count": 2,
                            "pct_of_total": 0.3333333333333333
                        },
                        "B": {
                            "count": 1,
                            "pct_of_total": 0.16666666666666666
                        },
                        "C": {
                            "count": 3,
                            "pct_of_total": 0.5
                        }
                    },
                    "cumulative_count_at_threshold": 5,
                    "denominator_key": "pct_of_total",
                    "input_categories": 3,
                    "input_nonzero_categories": 3,
                    "n_bars_rendered": 3,
                    "nonzero_categories": 3,
                    "params": {
                        "bar_height_source": "values",
                        "bar_sort_descending": True,
                        "bar_top_include_ties": True,
                        "bar_top_n": 1,
                        "max_display_bars": 15,
                        "other_display": None,
                        "other_label": "Other",
                        "other_label_format": "{label} (k={k_agg})",
                        "other_min_count": None,
                        "other_respect_existing": True,
                        "pareto_mode": "shared",
                        "show_count_in_bar_label": False,
                        "show_value_in_bar_label": True
                    },
                    "pct_subset": 1.0,
                    "subset_count": 6,
                    "threshold_idx": 1,
                    "threshold_pct": 80.0,
                    "top_labels": [
                        "C"
                    ],
                    "total": 6,
                    "total_nonnull": 6,
                    "unique_categories_total": 3
                },
                "draft_descriptive_findings": {
                    "context": "N = 6 values across 3 categories",
                    "primary_finding": "≈80% of occurrences are concentrated in the top 2 categories.",
                    "secondary_finding": "Top category: 'C' at 50.0%."
                },
                "draft_inferential_findings": {},
                "inferential_stats": {}
            },
        ),
        # 3) explicit labels/data_source
        (
            (lambda vals=["m", "m", "n", "o"]:
                pd.Series(vals, name="cats", dtype="object")),
            {"xlabel": "Category", "ylabel": "Frequency", "data_source": "UnitTest"},
            {
                "chart_metadata": {"xlabel": "Category", "ylabel": "Frequency", "data_source": "UnitTest"},
                "descriptive_stats": {
                    "bars": {
                    "m": {
                        "count": 2,
                        "pct_of_total": 0.5
                    },
                    "n": {
                        "count": 1,
                        "pct_of_total": 0.25
                    },
                    "o": {
                        "count": 1,
                        "pct_of_total": 0.25
                    }
                    },
                    "cumulative_count_at_threshold": 4,
                    "denominator_key": "pct_of_total",
                    "input_categories": 3,
                    "input_nonzero_categories": 3,
                    "n_bars_rendered": 3,
                    "nonzero_categories": 3,
                    "params": {
                        "bar_height_source": "values",
                        "bar_sort_descending": True,
                        "bar_top_include_ties": True,
                        "bar_top_n": 1,
                        "max_display_bars": 15,
                        "other_display": None,
                        "other_label": "Other",
                        "other_label_format": "{label} (k={k_agg})",
                        "other_min_count": None,
                        "other_respect_existing": True,
                        "pareto_mode": "shared",
                        "show_count_in_bar_label": False,
                        "show_value_in_bar_label": True
                    },
                    "pct_subset": 1.0,
                    "subset_count": 4,
                    "threshold_idx": 2,
                    "threshold_pct": 80.0,
                    "top_labels": [
                        "m"
                    ],
                    "total": 4,
                    "total_nonnull": 4,
                    "unique_categories_total": 3
                },
                "draft_descriptive_findings": {
                    "context": "N = 4 values across 3 categories",
                    "primary_finding": "≈80% of occurrences are concentrated in the top 3 categories.",
                    "secondary_finding": "Top category: 'm' at 50.0%."
                },
            },
        ),
        # 4) name override
        (
            (lambda vals=["a", "a", "b"]:
                pd.Series(vals, name="cats", dtype="object")),
            {"name": "Products"},
            {
                "chart_metadata": {"title": "Pareto Chart of Products"},
            },
        ),
        # 5) filter_desc only
        (
            (lambda vals=["a", "a", "b"]:
                pd.Series(vals, name="cats", dtype="object")),
            {"filter_desc": "filtered by NY"},
            {
                "chart_metadata": {"title": "Pareto Chart of cats (filtered by NY)"},
            },
        ),
        # 6) transform_desc only
        (
            (lambda vals=["a", "a", "b"]:
                pd.Series(vals, name="cats", dtype="object")),
            {"transform_desc": "log-transformed"},
            {
                "chart_metadata": {"title": "Pareto Chart of cats (log-transformed)"},
            },
        ),
        # 7) both modifiers (order & comma)
        (
            (lambda vals=["a", "a", "b"]:
                pd.Series(vals, name="cats", dtype="object")),
            {"filter_desc": "filtered by NY", "transform_desc": "trimmed"},
            {
                "chart_metadata": {"title": "Pareto Chart of cats (filtered by NY, trimmed)"},
            },
        ),
        # 8) custom template (omits {modifiers})
        (
            (lambda vals=["a", "a", "b"]:
                pd.Series(vals, name="cats", dtype="object")),
            {"title_template": "My Chart: {name}", "filter_desc": "ignored", "transform_desc": "ignored"},
            {
                "chart_metadata": {"title": "My Chart: cats"},
            },
        ),
    ],
    ids=[
        "empty",
        "grouping_to_others_save",
        "horizontal_save",
        "meta_fields",
        "title_name_override",
        "title_filter_only",
        "title_transform_only",
        "title_both_modifiers",
        "title_custom_template_no_mods",
    ],
)
def test_frequency_pareto_plot_data_driven(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    # If a file_name is provided, also set save_path to tmp_path
    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["save_path"] = tmp_path

    ctx = FrequencyParetoContext(**kwargs)
    plot = FrequencyParetoPlot(ctx)

    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
