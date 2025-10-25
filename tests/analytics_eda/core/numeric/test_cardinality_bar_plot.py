import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric import CardinalityBarContext, CardinalityBarPlot


@pytest.mark.parametrize(
    "series_factory, expected_exc, match",
    [
        # Not a Series
        (lambda: [1, 2, 3], TypeError, r"data must be a pandas Series or DataFrame"),
        # Non-numeric Series
        (lambda: pd.Series(["a", "b", "c"], name="letters"), TypeError, r"Series must be numeric"),
        # Missing name
        (lambda: pd.Series([1, 2, 3]), ValueError, r"must have a non-empty 'name'"),
        # Blank/whitespace name
        (lambda: pd.Series([1, 2, 3], name="   "), ValueError, r"must have a non-empty 'name'"),
    ],
    ids=["not_series", "bad_dtype", "missing_name", "blank_name"],
)
def test_validate_numeric_named_series_errors(series_factory, expected_exc, match):
    s = series_factory()
    with pytest.raises(expected_exc, match=match):
        ctx = CardinalityBarContext()
        plot = CardinalityBarPlot(ctx)

        plot.run(s)


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) EMPTY numeric series
        (
            lambda: pd.Series([], dtype="float64", name="nums"),
            {},
            {
                "chart_metadata": {
                    "title": "Cardinality Check — Discrete vs. Continuous for nums",
                    "version": "1.0.0",
                    "xlabel": "Number of Records",
                    "ylabel": "Values (Top N)",
                    "file_name": None,
                },
                "descriptive_stats": {
                    "bars": {},
                    "coverage_named": 0.0,
                    "is_discrete": None,
                    "nunique_native": 0,
                    "params": {
                        "integer_tolerance": 1e-08,
                        "max_unique_fraction": 0.05,
                        "max_unique_values": 20
                    },
                    "total": 0,
                    "uniqueness_ratio": 0.0
                },
                "draft_descriptive_findings": {},
                "draft_inferential_findings": {},
                "inferential_stats": {}
            },
        ),
        # 1) Low-cardinality integer data is discrete; include data_source
        (
            lambda: pd.Series([1, 1, 2, 2, 2, 3], name="ids", dtype="int64"),
            {"file_name": "card.png","data_source": "UnitTest"},
            {
                "chart_metadata": {
                    "data_source": "UnitTest",
                    "title": "Cardinality Check — Discrete vs. Continuous for ids",
                    "version": "1.0.0",
                    "xlabel": "Number of Records",
                    "ylabel": "Values (Top N)",
                    "file_name": "card.png"
                },
                "descriptive_stats": {
                    "bars": {
                        "1": {
                            "count": 2,
                            "pct_of_total": 0.3333333333333333
                        },
                        "2": {
                            "count": 3,
                            "pct_of_total": 0.5
                        },
                        "3": {
                            "count": 1,
                            "pct_of_total": 0.16666666666666666
                        }
                    },
                    "coverage_named": 1.0,
                    "denominator_key": "pct_of_total",
                    "input_categories": 3,
                    "input_nonzero_categories": 3,
                    "is_discrete": True,
                    "n_bars_rendered": 3,
                    "nonzero_categories": 3,
                    "nunique_native": 3,
                    "params": {
                        "bar_height_source": "counts",
                        "bar_sort_descending": True,
                        "bar_top_include_ties": True,
                        "bar_top_n": 1,
                        "integer_tolerance": 1e-08,
                        "max_display_bars": 15,
                        "max_unique_fraction": 0.05,
                        "max_unique_values": 20,
                        "other_display": None,
                        "other_label": "Other",
                        "other_label_format": "{label} (k={k_agg})",
                        "other_min_count": None,
                        "other_respect_existing": True,
                        "show_count_in_bar_label": False,
                        "show_value_in_bar_label": True
                    },
                    "pct_subset": 1.0,
                    "subset_count": 6,
                    "top_labels": [
                        "2"
                    ],
                    "total": 6,
                    "total_nonnull": 6,
                    "unique_categories_total": 3,
                    "uniqueness_ratio": 0.5
                },
                "draft_descriptive_findings": {
                    "context": "N = 6 non-null | 3 unique (50.0%)",
                    "primary_finding": "Variable behaves discrete.",
                    "secondary_finding": "Most frequent value '2' at 50.0%."
                },
                "draft_inferential_findings": {},
                "inferential_stats": {}
            },
        ),
        # 2) Float values that are whole numbers (within tolerance) => discrete
        (
            lambda: pd.Series([1.0, 2.0, 2.0, 3.0], name="vals", dtype="float64"),
            {"file_name": "card.png"},
            {
                "chart_metadata": {
                    "file_name": "card.png"
                },
                "descriptive_stats": {
                    "bars": {
                        "1.0": {
                            "count": 1,
                            "pct_of_total": 0.25
                        },
                        "2.0": {
                            "count": 2,
                            "pct_of_total": 0.5
                        },
                        "3.0": {
                            "count": 1,
                            "pct_of_total": 0.25
                        }
                    },
                    "coverage_named": 1.0,
                    "denominator_key": "pct_of_total",
                    "input_categories": 3,
                    "input_nonzero_categories": 3,
                    "is_discrete": True,
                    "n_bars_rendered": 3,
                    "nonzero_categories": 3,
                    "nunique_native": 3,
                    "params": {
                        "bar_height_source": "counts",
                        "bar_sort_descending": True,
                        "bar_top_include_ties": True,
                        "bar_top_n": 1,
                        "integer_tolerance": 1e-08,
                        "max_display_bars": 15,
                        "max_unique_fraction": 0.05,
                        "max_unique_values": 20,
                        "other_display": None,
                        "other_label": "Other",
                        "other_label_format": "{label} (k={k_agg})",
                        "other_min_count": None,
                        "other_respect_existing": True,
                        "show_count_in_bar_label": False,
                        "show_value_in_bar_label": True
                    },
                    "pct_subset": 1.0,
                    "subset_count": 4,
                    "top_labels": [
                    "2.0"
                    ],
                    "total": 4,
                    "total_nonnull": 4,
                    "unique_categories_total": 3,
                    "uniqueness_ratio": 0.75
                },
                "draft_descriptive_findings": {
                    "context": "N = 4 non-null | 3 unique (75.0%)",
                    "primary_finding": "Variable behaves discrete.",
                    "secondary_finding": "Most frequent value '2.0' at 50.0%."
                },
                "draft_inferential_findings": {},
                "inferential_stats": {}
            },
        ),
        # 3) Many unique floats => continuous (not discrete)
        (
            lambda: pd.Series(np.linspace(0, 0.99, 100), name="x", dtype="float64"),
            {"file_name": "card.png"},
            {
                "chart_metadata": {
                    "file_name": "card.png",
                    "title": "Cardinality Check — Discrete vs. Continuous for x"
                },
                "descriptive_stats": {
                    "bars": {
                        "0.0": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "0.01": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "0.02": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "0.03": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "0.04": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "0.05": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "0.06": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "0.07": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "0.08": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "0.09": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "0.1": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "0.11": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "0.12": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "0.13": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "Other (k=86)": {
                            "_is_other": True,
                            "count": 86,
                            "k_agg": 86,
                            "pct_of_total": 0.86
                        }
                    },
                    "coverage_named": 0.14,
                    "denominator_key": "pct_of_total",
                    "input_categories": 100,
                    "input_nonzero_categories": 100,
                    "is_discrete": False,
                    "n_bars_rendered": 15,
                    "nonzero_categories": 15,
                    "nunique_native": 100,
                    "params": {
                        "bar_height_source": "counts",
                        "bar_sort_descending": True,
                        "bar_top_include_ties": True,
                        "bar_top_n": 1,
                        "integer_tolerance": 1e-08,
                        "max_display_bars": 15,
                        "max_unique_fraction": 0.05,
                        "max_unique_values": 20,
                        "other_display": "Other (k=86)",
                        "other_label": "Other",
                        "other_label_format": "{label} (k={k_agg})",
                        "other_min_count": None,
                        "other_respect_existing": True,
                        "show_count_in_bar_label": False,
                        "show_value_in_bar_label": True
                    },
                    "pct_subset": 1.0,
                    "subset_count": 100,
                    "top_labels": [
                    "0.0",
                    "0.01",
                    "0.02",
                    "0.03",
                    "0.04",
                    "0.05",
                    "0.06",
                    "0.07",
                    "0.08",
                    "0.09",
                    "0.1",
                    "0.11",
                    "0.12",
                    "0.13"
                    ],
                    "total": 100,
                    "total_nonnull": 100,
                    "unique_categories_total": 100,
                    "uniqueness_ratio": 1.0
                },
                "draft_descriptive_findings": {
                    "context": "N = 100 non-null | 100 unique (100.0%) | Top-15 named coverage: 14.0%",
                    "primary_finding": "Variable behaves continuous.",
                    "secondary_finding": None
                },
                "draft_inferential_findings": {},
                "inferential_stats": {}
            },
        ),
        # 4) NaNs present; uniqueness_ratio uses len(clean); total is non-null count
        (
            lambda: pd.Series([1, 1, 2, np.nan, np.nan], name="with_nans"),
            {"file_name": "card.png"},
            {
                "descriptive_stats": {
                    "bars": {
                    "1.0": {
                        "count": 2,
                        "pct_of_total": 0.6666666666666666
                    },
                    "2.0": {
                        "count": 1,
                        "pct_of_total": 0.3333333333333333
                    }
                    },
                    "coverage_named": 1.0,
                    "denominator_key": "pct_of_total",
                    "input_categories": 2,
                    "input_nonzero_categories": 2,
                    "is_discrete": True,
                    "n_bars_rendered": 2,
                    "nonzero_categories": 2,
                    "nunique_native": 2,
                    "params": {
                        "bar_height_source": "counts",
                        "bar_sort_descending": True,
                        "bar_top_include_ties": True,
                        "bar_top_n": 1,
                        "integer_tolerance": 1e-08,
                        "max_display_bars": 15,
                        "max_unique_fraction": 0.05,
                        "max_unique_values": 20,
                        "other_display": None,
                        "other_label": "Other",
                        "other_label_format": "{label} (k={k_agg})",
                        "other_min_count": None,
                        "other_respect_existing": True,
                        "show_count_in_bar_label": False,
                        "show_value_in_bar_label": True
                    },
                    "pct_subset": 1.0,
                    "subset_count": 3,
                    "top_labels": [
                        "1.0"
                    ],
                    "total": 3,
                    "total_nonnull": 3,
                    "unique_categories_total": 2,
                    "uniqueness_ratio": 0.6666666666666666
                },
                "draft_descriptive_findings": {
                    "context": "N = 3 non-null | 2 unique (66.7%)",
                    "primary_finding": "Variable behaves discrete.",
                    "secondary_finding": "Most frequent value '1.0' at 66.7%."
                },
                "draft_inferential_findings": {},
                "inferential_stats": {}
            },
        ),
        # 5) Override max_unique_fraction to 1.0 on high-cardinality → discrete
        (
            lambda: pd.Series(range(100), name="nums"),
            {
                "max_unique_fraction": 1.0,
                "file_name": "frac.png",
            },
            {
                "chart_metadata": {
                    "file_name": "frac.png"
                },
                "descriptive_stats": {
                    "bars": {
                        "0": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "1": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "10": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "11": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "12": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "13": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "14": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "15": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "16": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "17": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "18": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "19": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "2": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "20": {
                            "count": 1,
                            "pct_of_total": 0.01
                        },
                        "Other (k=86)": {
                            "_is_other": True,
                            "count": 86,
                            "k_agg": 86,
                            "pct_of_total": 0.86
                        }
                    },
                    "coverage_named": 0.14,
                    "denominator_key": "pct_of_total",
                    "input_categories": 100,
                    "input_nonzero_categories": 100,
                    "is_discrete": True,
                    "n_bars_rendered": 15,
                    "nonzero_categories": 15,
                    "nunique_native": 100,
                    "params": {
                        "bar_height_source": "counts",
                        "bar_sort_descending": True,
                        "bar_top_include_ties": True,
                        "bar_top_n": 1,
                        "integer_tolerance": 1e-08,
                        "max_display_bars": 15,
                        "max_unique_fraction": 1.0,
                        "max_unique_values": 20,
                        "other_display": "Other (k=86)",
                        "other_label": "Other",
                        "other_label_format": "{label} (k={k_agg})",
                        "other_min_count": None,
                        "other_respect_existing": True,
                        "show_count_in_bar_label": False,
                        "show_value_in_bar_label": True
                    },
                    "pct_subset": 1.0,
                    "subset_count": 100,
                    "top_labels": [
                        "0",
                        "1",
                        "10",
                        "11",
                        "12",
                        "13",
                        "14",
                        "15",
                        "16",
                        "17",
                        "18",
                        "19",
                        "2",
                        "20"
                    ],
                    "total": 100,
                    "total_nonnull": 100,
                    "unique_categories_total": 100,
                    "uniqueness_ratio": 1.0
                },
                "draft_descriptive_findings": {
                    "context": "N = 100 non-null | 100 unique (100.0%) | Top-15 named coverage: 14.0%",
                    "primary_finding": "Variable behaves discrete.",
                    "secondary_finding": "Top values (tie at 1.0%): '0', '1', '10' +11 more."
                },
                "draft_inferential_findings": {},
                "inferential_stats": {}
            },
        ),
        # 6) Floats not integer-like, but unique < max_unique_values → discrete (path 2b)
        (
            lambda: pd.Series([i + 0.1 for i in range(10)], name="floats"),
            {
                "file_name": "low_card.png",
            },
            {
                "chart_metadata": {
                    "file_name": "low_card.png"
                },
                "descriptive_stats": {
                    "bars": {
                        "0.1": {
                            "count": 1,
                            "pct_of_total": 0.1
                        },
                        "1.1": {
                            "count": 1,
                            "pct_of_total": 0.1
                        },
                        "2.1": {
                            "count": 1,
                            "pct_of_total": 0.1
                        },
                        "3.1": {
                            "count": 1,
                            "pct_of_total": 0.1
                        },
                        "4.1": {
                            "count": 1,
                            "pct_of_total": 0.1
                        },
                        "5.1": {
                            "count": 1,
                            "pct_of_total": 0.1
                        },
                        "6.1": {
                            "count": 1,
                            "pct_of_total": 0.1
                        },
                        "7.1": {
                            "count": 1,
                            "pct_of_total": 0.1
                        },
                        "8.1": {
                            "count": 1,
                            "pct_of_total": 0.1
                        },
                        "9.1": {
                            "count": 1,
                            "pct_of_total": 0.1
                        }
                    },
                    "coverage_named": 1.0,
                    "denominator_key": "pct_of_total",
                    "input_categories": 10,
                    "input_nonzero_categories": 10,
                    "is_discrete": True,
                    "n_bars_rendered": 10,
                    "nonzero_categories": 10,
                    "nunique_native": 10,
                    "params": {
                        "bar_height_source": "counts",
                        "bar_sort_descending": True,
                        "bar_top_include_ties": True,
                        "bar_top_n": 1,
                        "integer_tolerance": 1e-08,
                        "max_display_bars": 15,
                        "max_unique_fraction": 0.05,
                        "max_unique_values": 20,
                        "other_display": None,
                        "other_label": "Other",
                        "other_label_format": "{label} (k={k_agg})",
                        "other_min_count": None,
                        "other_respect_existing": True,
                        "show_count_in_bar_label": False,
                        "show_value_in_bar_label": True
                    },
                    "pct_subset": 1.0,
                    "subset_count": 10,
                    "top_labels": [
                        "0.1",
                        "1.1",
                        "2.1",
                        "3.1",
                        "4.1",
                        "5.1",
                        "6.1",
                        "7.1",
                        "8.1",
                        "9.1"
                    ],
                    "total": 10,
                    "total_nonnull": 10,
                    "unique_categories_total": 10,
                    "uniqueness_ratio": 1.0
                },
                "draft_descriptive_findings": {
                    "context": "N = 10 non-null | 10 unique (100.0%)",
                    "primary_finding": "Variable behaves discrete.",
                    "secondary_finding": "Top values (tie at 10.0%): '0.1', '1.1', '2.1' +7 more."
                },
                "draft_inferential_findings": {},
                "inferential_stats": {}
            },
        ),
        # 7) High-cardinality floats, not integer-like, tolerance flips discrete from False → True
        (
            lambda: pd.Series([i + 1e-6 for i in range(30)], name="floats"),
            {
                "integer_tolerance": 1e-5,
                "file_name": "flt_tol.png",
            },
            {
                "chart_metadata": {
                    "file_name": "flt_tol.png"
                },
                "descriptive_stats": {
                    "bars": {
                        "1.000001": {
                            "count": 1,
                            "pct_of_total": 0.03333333333333333
                        },
                        "10.000001": {
                            "count": 1,
                            "pct_of_total": 0.03333333333333333
                        },
                        "11.000001": {
                            "count": 1,
                            "pct_of_total": 0.03333333333333333
                        },
                        "12.000001": {
                            "count": 1,
                            "pct_of_total": 0.03333333333333333
                        },
                        "13.000001": {
                            "count": 1,
                            "pct_of_total": 0.03333333333333333
                        },
                        "14.000001": {
                            "count": 1,
                            "pct_of_total": 0.03333333333333333
                        },
                        "15.000001": {
                            "count": 1,
                            "pct_of_total": 0.03333333333333333
                        },
                        "16.000001": {
                            "count": 1,
                            "pct_of_total": 0.03333333333333333
                        },
                        "17.000001": {
                            "count": 1,
                            "pct_of_total": 0.03333333333333333
                        },
                        "18.000001": {
                            "count": 1,
                            "pct_of_total": 0.03333333333333333
                        },
                        "19.000001": {
                            "count": 1,
                            "pct_of_total": 0.03333333333333333
                        },
                        "1e-06": {
                            "count": 1,
                            "pct_of_total": 0.03333333333333333
                        },
                        "2.000001": {
                            "count": 1,
                            "pct_of_total": 0.03333333333333333
                        },
                        "20.000001": {
                            "count": 1,
                            "pct_of_total": 0.03333333333333333
                        },
                        "Other (k=16)": {
                            "_is_other": True,
                            "count": 16,
                            "k_agg": 16,
                            "pct_of_total": 0.5333333333333333
                        }
                    },
                    "coverage_named": 0.4666666666666667,
                    "denominator_key": "pct_of_total",
                    "input_categories": 30,
                    "input_nonzero_categories": 30,
                    "is_discrete": True,
                    "n_bars_rendered": 15,
                    "nonzero_categories": 15,
                    "nunique_native": 30,
                    "params": {
                        "bar_height_source": "counts",
                        "bar_sort_descending": True,
                        "bar_top_include_ties": True,
                        "bar_top_n": 1,
                        "integer_tolerance": 1e-05,
                        "max_display_bars": 15,
                        "max_unique_fraction": 0.05,
                        "max_unique_values": 20,
                        "other_display": "Other (k=16)",
                        "other_label": "Other",
                        "other_label_format": "{label} (k={k_agg})",
                        "other_min_count": None,
                        "other_respect_existing": True,
                        "show_count_in_bar_label": False,
                        "show_value_in_bar_label": True
                    },
                    "pct_subset": 1.0,
                    "subset_count": 30,
                    "top_labels": [
                        "1.000001",
                        "10.000001",
                        "11.000001",
                        "12.000001",
                        "13.000001",
                        "14.000001",
                        "15.000001",
                        "16.000001",
                        "17.000001",
                        "18.000001",
                        "19.000001",
                        "1e-06",
                        "2.000001",
                        "20.000001"
                    ],
                    "total": 30,
                    "total_nonnull": 30,
                    "unique_categories_total": 30,
                    "uniqueness_ratio": 1.0
                },
                "draft_descriptive_findings": {
                    "context": "N = 30 non-null | 30 unique (100.0%) | Top-15 named coverage: 46.7%",
                    "primary_finding": "Variable behaves discrete.",
                    "secondary_finding": "Top values (tie at 3.3%): '1.000001', '10.000001', '11.000001' +11 more."
                },
                "draft_inferential_findings": {},
                "inferential_stats": {}
            },
        ),
    ],
    ids=[
        "empty",
        "discrete_integers_with_source",
        "whole_like_floats_are_discrete",
        "many_unique_floats_continuous",
        "nans_affect_uniqueness_ratio_denominator",
        "override_max_unique_fraction_discrete",
        "float_low_cardinality_path_2b",
        "float_high_cardinality_tol_flip",
    ],
)
def test_cardinality_bar_plot_data_driven(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()

    # If a file_name is provided, also set save_path to tmp_path
    if "file_name" in kwargs:
        kwargs = kwargs.copy()
        kwargs["save_path"] = tmp_path

    ctx = CardinalityBarContext(**kwargs)
    plot = CardinalityBarPlot(ctx)

    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
