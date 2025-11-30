import pandas as pd
import pytest

from analytics_eda.core.categorical.balance import (
    BalanceChiSquareUniformPlot,
    BalanceChiSquareUniformPlotContext,
)


@pytest.mark.parametrize(
    "series_factory, expected_exc, match",
    [
        # Not a pandas Series
        (lambda: ["a", "b", "c"], TypeError, r"data must be a pandas Series or DataFrame"),
        # Not categorical/object dtype
        (lambda: pd.Series([1, 2, 3], name="numeric"), TypeError, r"must be categorical.*for categorical analysis"),
        # Missing name (None)
        (lambda: pd.Series(["x", "y", "z"], dtype="category"), ValueError, r"must have a non-empty 'name'"),
        # Blank/whitespace name
        (lambda: pd.Series(["x", "y", "z"], dtype="object", name=" "), ValueError, r"must have a non-empty 'name'"),
    ],
    ids=["not_series", "bad_dtype", "missing_name", "blank_name"],
)
def test_validate_categorical_named_series_errors(series_factory, expected_exc, match):
    s = series_factory()
    with pytest.raises(expected_exc, match=match):
        ctx = BalanceChiSquareUniformPlotContext()
        plot = BalanceChiSquareUniformPlot(ctx)

        plot.run(s)


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) Empty series returns early with no inferential_stats
        (
            lambda: pd.Series(pd.Categorical([], categories=["A", "B"]), name="testvar"),
            {},
            {
                "descriptive_stats": {"total": 0, "k": 0},
                "inferential_stats": {},
                "chart_metadata": {"file_name": None, "title": "Chi-Square Goodness-of-Fit: testvar", "version": "1.0.0", "xlabel": "Observed − Expected (count)", "ylabel": "Category"},
                "draft_descriptive_findings": {},
                "draft_inferential_findings": {},
            },
        ),
        # 1) Balanced distribution (uniform, should not reject H0)
        (
            lambda: pd.Series(["A", "B", "C", "A", "B", "C"], name="letters"),
            {"alpha": 0.05, "file_name": "gof_balanced_distribution.png"},
            {
                "chart_metadata": {"file_name": "gof_balanced_distribution.png", "title": "Chi-Square Goodness-of-Fit: letters", "version": "1.0.0", "xlabel": "Observed − Expected (count)", "ylabel": "Category"},
                "descriptive_stats": {
                    "bars": {
                        "A": {"delta": 0, "delta_pct_of_expected": 0.0, "expected": 2.0, "observed": 2, "ratio_oe": 1.0, "std_resid": 0.0},
                        "B": {"delta": 0, "delta_pct_of_expected": 0.0, "expected": 2.0, "observed": 2, "ratio_oe": 1.0, "std_resid": 0.0},
                        "C": {"delta": 0, "delta_pct_of_expected": 0.0, "expected": 2.0, "observed": 2, "ratio_oe": 1.0, "std_resid": 0.0},
                    },
                    "k": 3,
                    "max_abs_delta": 0.0,
                    "n_over": 0,
                    "n_under": 0,
                    "params": {"bar_top_include_ties": True, "bar_top_n": 1, "delta_metric": "count", "label_value_format": "±count_and_ratio", "sort_mode": "abs_delta"},
                    "top_labels": [],
                    "total": 6,
                },
                "draft_descriptive_findings": {"context": "N = 6 values across 3 categories", "primary_finding": "Observed frequencies match a uniform expectation.", "secondary_finding": None},
                "draft_inferential_findings": {
                    "context": "Chi-square GOF on 3 categories (N = 6)",
                    "primary_finding": "No statistically significant deviation from uniform (p = 1 vs α = 0.05).",
                    "secondary_finding": "χ²(df = 2) = 0.00. Assumption warning: Assumption caution: 3 of 3 categories have expected counts < 5 (minimum expected = 2.00); chi-square results may be unreliable.",
                },
                "inferential_stats": {
                    "chi2_gof_null_uniform": {
                        "alpha": 0.05,
                        "df": 2,
                        "p_value": 1.0,
                        "reject": False,
                        "statistic": 0.0,
                        "warning": "Assumption caution: 3 of 3 categories have expected counts < 5 (minimum expected = 2.00); chi-square results may be unreliable.",
                    }
                },
            },
        ),
        # 2) Unbalanced distribution (should reject H0)
        (
            lambda: pd.Series(["X"] * 10 + ["Y"] * 2 + ["Z"] * 1, name="choices"),
            {"alpha": 0.05, "file_name": "gof_unbalanced_distribution.png"},
            {
                "chart_metadata": {"file_name": "gof_unbalanced_distribution.png", "title": "Chi-Square Goodness-of-Fit: choices", "version": "1.0.0", "xlabel": "Observed − Expected (count)", "ylabel": "Category"},
                "descriptive_stats": {
                    "bars": {
                        "X": {"delta": 5, "delta_pct_of_expected": 1.307692307692308, "expected": 4.333333333333333, "observed": 10, "ratio_oe": 2.307692307692308, "std_resid": 2.7221786146864817},
                        "Y": {"delta": -2, "delta_pct_of_expected": -0.5384615384615384, "expected": 4.333333333333333, "observed": 2, "ratio_oe": 0.46153846153846156, "std_resid": -1.12089707663561},
                        "Z": {"delta": -3, "delta_pct_of_expected": -0.7692307692307692, "expected": 4.333333333333333, "observed": 1, "ratio_oe": 0.23076923076923078, "std_resid": -1.6012815380508714},
                    },
                    "k": 3,
                    "max_abs_delta": 5.666666666666667,
                    "n_over": 1,
                    "n_under": 2,
                    "params": {"bar_top_include_ties": True, "bar_top_n": 1, "delta_metric": "count", "label_value_format": "±count_and_ratio", "sort_mode": "abs_delta"},
                    "top_labels": ["X"],
                    "total": 13,
                },
                "draft_descriptive_findings": {"context": "N = 13 values across 3 categories", "primary_finding": "Category frequencies deviate from a uniform expectation.", "secondary_finding": "Largest absolute deviation: X (+5, O/E=2.31)."},
                "draft_inferential_findings": {
                    "context": "Chi-square GOF on 3 categories (N = 13)",
                    "primary_finding": "Frequencies differ from a uniform distribution (p = 0.004 vs α = 0.05).",
                    "secondary_finding": "χ²(df = 2) = 11.23. Assumption warning: Assumption caution: 3 of 3 categories have expected counts < 5 (minimum expected = 4.33); chi-square results may be unreliable.",
                },
                "inferential_stats": {
                    "chi2_gof_null_uniform": {
                        "alpha": 0.05,
                        "df": 2,
                        "p_value": 0.003641408886883211,
                        "reject": True,
                        "statistic": 11.23076923076923,
                        "warning": "Assumption caution: 3 of 3 categories have expected counts < 5 (minimum expected = 4.33); chi-square results may be unreliable.",
                    }
                },
            },
        ),
        # 3) Custom chart metadata values
        (lambda: pd.Series(["cat", "dog", "dog", "bird"], name="animals"), {"xlabel": "Animal", "ylabel": "Count", "data_source": "UnitTest"}, {"chart_metadata": {"xlabel": "Animal", "ylabel": "Count", "data_source": "UnitTest"}}),
    ],
    ids=["empty_series", "balanced_uniform", "unbalanced_reject_h0", "custom_labels"],
)
def test_balance_chi_square_uniform_plot_data_driven(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()
    if "file_name" in kwargs:
        kwargs = {**kwargs, "base_dir": tmp_path}

    ctx = BalanceChiSquareUniformPlotContext(**kwargs)
    plot = BalanceChiSquareUniformPlot(ctx)

    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
