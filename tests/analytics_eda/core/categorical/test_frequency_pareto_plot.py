import pytest
import pandas as pd

from analytics_eda.core.categorical import FrequencyParetoPlot, FrequencyParetoContext

@pytest.mark.parametrize(
    "series_factory, expected_exc, match",
    [
        # Not a pandas Series
        (lambda: ["a", "b", "c"], TypeError, r"Input must be a pandas Series\."),

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
                "chart_metadata": {"title": "Pareto Chart of cats", "file_name": None},
                "descriptive_stats": {
                    "total_count": 0,
                    "n_categories": 0,
                    "cumulative_count_at_80pct": 0,
                    "mode": None,
                },
            },
        ),
        # 1a) min_value groups small categories into "Others" (no save)
        (
            (lambda vals=["A", "A", "A", "B", "C"]:
                pd.Series(vals, name="cats", dtype="object")),
            {"min_value": 2},
            {
                "chart_metadata": {"xlabel": "Value", "ylabel": "Count"},
                "descriptive_stats": {
                    "total_count": 5,
                    "n_categories": 2,
                    "cumulative_count_at_80pct": 5,
                },
            },
        ),
        # 1b) min_value groups small categories into "Others" (save)
        (
            (lambda vals=["A", "A", "A", "B", "C"]:
                pd.Series(vals, name="cats", dtype="object")),
            {"min_value": 2, "file_name": "grouping_to_others.png"},
            {
                "chart_metadata": {
                    "xlabel": "Value",
                    "ylabel": "Count",
                    "file_name": "grouping_to_others.png",
                },
                "descriptive_stats": {
                    "total_count": 5,
                    "n_categories": 2,
                    "cumulative_count_at_80pct": 5,
                },
            },
        ),
        # 2a) horizontal orientation (no save)
        (
            (lambda vals=["A", "A", "B", "C", "C", "C"]:
                pd.Series(vals, name="cats", dtype="object")),
            {"horizontal": True},
            {
                "chart_metadata": {},  # nothing specific to assert
                "descriptive_stats": {
                    "total_count": (lambda v, vals=["A","A","B","C","C","C"]: v == len(vals)),
                    "n_categories": (lambda v: v >= 1),
                },
            },
        ),
        # 2b) horizontal orientation (save)
        (
            (lambda vals=["A", "A", "B", "C", "C", "C"]:
                pd.Series(vals, name="cats", dtype="object")),
            {"horizontal": True, "file_name": "horizontal_orientation.png"},
            {
                "chart_metadata": {"file_name": "horizontal_orientation.png"},
                "descriptive_stats": {
                    "total_count": (lambda v, vals=["A","A","B","C","C","C"]: v == len(vals)),
                    "n_categories": (lambda v: v >= 1),
                },
            },
        ),
        # 3a) vertical (default) (no save)
        (
            (lambda vals=["x", "y", "y", "z"]:
                pd.Series(vals, name="cats", dtype="object")),
            {},
            {
                "chart_metadata": {},
                "descriptive_stats": {
                    "total_count": (lambda v, vals=["x","y","y","z"]: v == len(vals)),
                    "n_categories": (lambda v: v >= 1),
                },
            },
        ),
        # 3b) vertical (default) (save)
        (
            (lambda vals=["x", "y", "y", "z"]:
                pd.Series(vals, name="cats", dtype="object")),
            {"file_name": "vertical_orientation.png"},
            {
                "chart_metadata": {"file_name": "vertical_orientation.png"},
                "descriptive_stats": {
                    "total_count": (lambda v, vals=["x","y","y","z"]: v == len(vals)),
                    "n_categories": (lambda v: v >= 1),
                },
            },
        ),
        # 4) explicit labels/data_source
        (
            (lambda vals=["m", "m", "n", "o"]:
                pd.Series(vals, name="cats", dtype="object")),
            {"xlabel": "Category", "ylabel": "Frequency", "data_source": "UnitTest"},
            {
                "chart_metadata": {"xlabel": "Category", "ylabel": "Frequency", "data_source": "UnitTest"},
                "descriptive_stats": {
                    "total_count": (lambda v, vals=["m","m","n","o"]: v == len(vals)),
                    "n_categories": (lambda v: v >= 1),
                },
            },
        ),
        # 5) name override
        (
            (lambda vals=["a", "a", "b"]:
                pd.Series(vals, name="cats", dtype="object")),
            {"name": "Products"},
            {
                "chart_metadata": {"title": "Pareto Chart of Products"},
                "descriptive_stats": {
                    "total_count": (lambda v, vals=["a","a","b"]: v == len(vals))
                },
            },
        ),
        # 6) filter_desc only
        (
            (lambda vals=["a", "a", "b"]:
                pd.Series(vals, name="cats", dtype="object")),
            {"filter_desc": "filtered by NY"},
            {
                "chart_metadata": {"title": "Pareto Chart of cats (filtered by NY)"},
                "descriptive_stats": {
                    "total_count": (lambda v, vals=["a","a","b"]: v == len(vals))
                },
            },
        ),
        # 7) transform_desc only
        (
            (lambda vals=["a", "a", "b"]:
                pd.Series(vals, name="cats", dtype="object")),
            {"transform_desc": "log-transformed"},
            {
                "chart_metadata": {"title": "Pareto Chart of cats (log-transformed)"},
                "descriptive_stats": {
                    "total_count": (lambda v, vals=["a","a","b"]: v == len(vals))
                },
            },
        ),
        # 8) both modifiers (order & comma)
        (
            (lambda vals=["a", "a", "b"]:
                pd.Series(vals, name="cats", dtype="object")),
            {"filter_desc": "filtered by NY", "transform_desc": "trimmed"},
            {
                "chart_metadata": {"title": "Pareto Chart of cats (filtered by NY, trimmed)"},
                "descriptive_stats": {
                    "total_count": (lambda v, vals=["a","a","b"]: v == len(vals))
                },
            },
        ),
        # 9) custom template (omits {modifiers})
        (
            (lambda vals=["a", "a", "b"]:
                pd.Series(vals, name="cats", dtype="object")),
            {"title_template": "My Chart: {name}", "filter_desc": "ignored", "transform_desc": "ignored"},
            {
                "chart_metadata": {"title": "My Chart: cats"},
                "descriptive_stats": {
                    "total_count": (lambda v, vals=["a","a","b"]: v == len(vals))
                },
            },
        ),
    ],
    ids=[
        "empty",
        "grouping_to_others",
        "grouping_to_others_save",
        "horizontal",
        "horizontal_save",
        "vertical",
        "vertical_save",
        "meta_fields",
        "title_name_override",
        "title_filter_only",
        "title_transform_only",
        "title_both_modifiers",
        "title_custom_template_no_mods",
    ],
)
def test_plot_frequency_pareto_param(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    # If a file_name is provided, also set save_path to tmp_path
    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["save_path"] = tmp_path

    ctx = FrequencyParetoContext(**kwargs)
    plot = FrequencyParetoPlot(ctx)

    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
