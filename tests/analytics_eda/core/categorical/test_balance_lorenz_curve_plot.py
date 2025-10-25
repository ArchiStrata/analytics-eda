import pandas as pd
import pytest

from analytics_eda.core.categorical import BalanceLorenzCurveContext, BalanceLorenzCurvePlot


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
        ctx = BalanceLorenzCurveContext()
        plot = BalanceLorenzCurvePlot(ctx)

        plot.run(s)


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) Empty series -> early return; gini is NaN; no saved file
        (
            lambda: pd.Series(pd.Categorical([], categories=["A", "B"]), name="cats"),
            {},
            {
                "chart_metadata": {
                    "file_name": None,
                    "version": "1.0.0",
                    "title": "Lorenz Curve of cats",
                    "xlabel": "Cumulative % of categories",
                    "ylabel": "Cumulative % of values",
                },
                "descriptive_stats": {
                    "total": 0,
                    "k": 0,
                    "gini_index": None,
                },
            },
        ),
        # 1) Balanced distribution (uniform) -> Gini ~ 0, and save file
        (
            lambda: pd.Series(["A", "B", "C", "A", "B", "C"], name="letters"),
            {"file_name": "lorenz_balanced.png"},
            {
                "chart_metadata": {"file_name": "lorenz_balanced.png"},
                "descriptive_stats": {
                    "total": 6,
                    "k": 3,
                    "gini_index": lambda v: abs(v - 0.0) < 1e-12,
                },
                "draft_descriptive_findings": {
                    "context": "N = 6 values across 3 categories",
                    "primary_finding": "Category imbalance measured by Gini index = 0.000 (0 = perfectly balanced, 1 = highly imbalanced).",
                    "secondary_finding": "All categories are evenly distributed."
                },
            },
        ),
        # 2) Skewed distribution -> Gini noticeably > 0
        (
            lambda: pd.Series(["X"] * 10 + ["Y"] * 2 + ["Z"] * 1, name="choices"),
            {"file_name": "lorenz_skewed.png"},
            {
                "chart_metadata": {"file_name": "lorenz_skewed.png"},
                "descriptive_stats": {
                    "total": 13,
                    "k": 3,
                    "gini_index": lambda v: 0.35 < v < 0.8,  # allow tolerance across envs
                },
                "draft_descriptive_findings": {
                    "context": "N = 13 values across 3 categories",
                    "primary_finding": "Category imbalance measured by Gini index = 0.462 (0 = perfectly balanced, 1 = highly imbalanced).",
                    "secondary_finding": None
                },
            },
        ),
        # 3) Custom labels + data_source (no save)
        (
            lambda: pd.Series(["cat", "dog", "dog", "bird"], name="animals"),
            {"xlabel": "Categories (cum.)", "ylabel": "Counts (cum.)", "data_source": "UnitTest"},
            {
                "chart_metadata": {
                    "xlabel": "Categories (cum.)",
                    "ylabel": "Counts (cum.)",
                    "data_source": "UnitTest",
                },
                "descriptive_stats": {
                    "total": 4,
                    "k": 3,
                    "gini_index": lambda v: 0 <= v <= 1,
                },
            },
        ),
        # 4) Severely imbalanced distribution -> Gini > 0.8
        (
            lambda: pd.Series(
                ["X"] * 100                                   # dominant category
                + ["A", "B", "C", "D", "E", "F", "G", "H"],   # eight singletons
                name="choices"
            ),
            {"file_name": "lorenz_severe_imbalance.png"},
            {
                "chart_metadata": {"file_name": "lorenz_severe_imbalance.png"},
                "descriptive_stats": {
                    "total": 108,
                    "k": 9,
                    # robust predicate: ensure it's clearly > 0.8 (observed ~0.8096)
                    "gini_index": lambda v: 0.80 < v < 0.90,
                },
                "draft_descriptive_findings": {
                    "context": "N = 108 values across 9 categories",
                    # avoid hard-coding the exact numeric string
                    "primary_finding": lambda s: s.startswith(
                        "Category imbalance measured by Gini index = "
                    ),
                    "secondary_finding": "Severe imbalance: a small number of categories dominate.",
                },
            },
        )
    ],
    ids=["empty", "balanced_saves", "skewed_saves", "custom_labels", "severe_saves"],
)
def test_balance_lorenz_curve_plot_data_driven(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    # If saving, route to tmp_path
    if "file_name" in kwargs:
        kwargs = {**kwargs, "save_path": tmp_path}

    ctx = BalanceLorenzCurveContext(**kwargs)
    plot = BalanceLorenzCurvePlot(ctx)

    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
