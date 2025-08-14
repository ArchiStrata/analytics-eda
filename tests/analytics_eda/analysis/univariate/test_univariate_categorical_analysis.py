import pandas as pd
import pytest
from math import isclose

from analytics_eda.analysis.univariate.univariate_categorical_analysis import univariate_categorical_analysis
    
@pytest.mark.parametrize(
    "make_series, kwargs, expected_sections",
    [
        (
            lambda: pd.Series(
                ["A","A","A","B","B","C","C","C","C","D", None],
                name="pets", dtype="object"
            ),
            {
                "data_source": "UnitTest",
            },
            {
                # Expectations for the missing_data section
                "missing_data": {
                    "barchart": {
                        "descriptive_stats": {
                            "total": 11,
                            "missing": 1,
                            "pct_missing": lambda v: isclose(v, 1/11, rel_tol=1e-12, abs_tol=1e-12),
                        }
                    }
                },
                # Expectations for the data_quality section
                "data_quality": {
                    "categorical_cleanliness_barchart": {
                        "descriptive_stats": {
                            "total": 11,
                            "total_nonnull": 10,
                            "issue_labels": [
                                "Invalid Category",
                                "Non-Standard Characters",
                                "Mixed Casing",
                                "Leading/Trailing Whitespace"
                            ],
                            "issue_counts": [
                                0,
                                0,
                                0,
                                0
                            ],
                            "issue_pcts": [
                                0.0,
                                0.0,
                                0.0,
                                0.0
                            ],
                            "n_whitespace": 0,
                            "n_mixed_case": 0,
                            "n_nonstandard_chars": 0,
                            "n_invalid_category": 0,
                            "skip_plot": True,
                            "error": "no cleanliness issues detected"
                        },
                        "inferential_stats": {},
                        "chart_metadata": {
                            "title": "Categorical Cleanliness for pets",
                            "xlabel": "Count",
                            "ylabel": "Issue Type",
                            "data_source": "UnitTest",
                            "file_name": None
                        },
                        "cleaning_meta": {
                            "n_stripped": 0,
                            "n_case_normalized": 0,
                            "n_nonstandard_replaced": 0,
                            "n_invalid_mapped": 0,
                            "policy": {
                                "allowed_char_pattern": "^[\\w\\s\\-\\_/.,&()']*$",
                                "allowed_categories": None,
                                "case_sensitive_allowed": False,
                                "treat_empty_as_invalid": True
                            }
                        }
                    }
                },
                # Expectations for the cardinality section
                "cardinality": {
                    "barchart": {
                        "descriptive_stats": {
                            "params": {
                                "max_unique_fraction": 0.05,
                                "max_unique_values": 20,
                                "integer_tolerance": 1e-08
                            },
                            "total": 4,
                            "nunique": 4,
                            "uniqueness_ratio": 1.0,
                            "is_discrete": True,
                            "labels": [
                                "4",
                                "3",
                                "2",
                                "1"
                            ],
                            "values": "[1. 1. 1. 1.]"
                        },
                        "inferential_stats": {},
                        "chart_metadata": {
                            "title": "Value Counts (Top 10) of count for Cardinality",
                            "xlabel": "Value",
                            "ylabel": "Count",
                            "data_source": "UnitTest",
                            "file_name": "Value Counts (Top 10) of count for Cardinality.png",
                            "top_k": 10
                        }
                    }
                },
                # Expectations for the nested distribution report
                "distribution": {
                    "frequency_distribution": {
                        "pareto": {
                            "chart_metadata":  {
                                "title": "Pareto Chart of pets",
                                "xlabel": "Value",
                                "ylabel": "Count",
                                "data_source": "UnitTest",
                                "file_name": "Pareto Chart of pets.png"
                            },
                            "descriptive_stats": {
                                "mode": "C",
                                "total_count": 10,
                                "n_categories": 4,
                                "cumulative_count_at_80pct": 9
                            },
                        },
                    },
                    "balance": {
                        "density": {
                            "chart_metadata":  {
                                "title": "Distribution Density of count",
                                "xlabel": "Frequency",
                                "ylabel": "Density",
                                "data_source": "UnitTest",
                                "file_name": "Distribution Density of count.png"
                            },
                            "descriptive_stats": {
                                "n": 4,
                                "entropy_bits": 1.9999999999942293,
                                "skewness": 0.0,
                                "kurtosis": -1.1999999999999993,
                                "modes_count": 1,
                                "quartile_skew": 0.0,
                                "pct_10": 1.3,
                                "pct_25": 1.75,
                                "pct_50": 2.5,
                                "pct_75": 3.25,
                                "pct_90": 3.7
                            },
                        },
                        "boxplot": {
                            "chart_metadata":  {
                                "title": "Dispersion of count (IQR & Outliers)",
                                "ylabel": "Frequency",
                                "data_source": "UnitTest",
                                "file_name": "Dispersion of count (IQR & Outliers).png"
                            },
                            "descriptive_stats": {
                                "params": {
                                    "std_outlier_multiplier": 4.0
                                },
                                "n": 4,
                                "mean": 2.5,
                                "std": 1.2909944487358056,
                                "var": 1.6666666666666667,
                                "min": 1,
                                "max": 4,
                                "range": 3,
                                "mad": 1.0,
                                "cv": 0.5163977794943222,
                                "pct_10": 1.3,
                                "pct_25": 1.75,
                                "pct_75": 3.25,
                                "pct_90": 3.7,
                                "iqr": 1.5,
                                "extreme_lower_count": 0,
                                "extreme_upper_count": 0
                            },
                        },
                        "chi2_gof_uniform": {
                            "chart_metadata":  {
                                "title": "Chi-Square Goodness-of-Fit: pets",
                                "xlabel": "Frequency",
                                "ylabel": "Frequency",
                                "data_source": "UnitTest",
                                "file_name": "Chi-Square Goodness-of-Fit: pets.png"
                            },
                            "descriptive_stats": {
                                "total": 10,
                                "k": 4
                            },
                            "inferential_stats": {
                                "chi2_gof_null_uniform": {
                                    "statistic": 2.0,
                                    "p_value": 0.5724067044708798,
                                    "alpha": 0.05,
                                    "reject": False,
                                    "warning": "Some expected counts are below 5; chi-square test results may not be reliable."
                                }
                            },
                        },
                        "lorenz_curve": {
                            "descriptive_stats": {
                                "total": 10,
                                "k": 4,
                                "gini_index": 0.25
                            },
                            "chart_metadata": {
                                "title": "Lorenz Curve of pets",
                                "xlabel": "Cumulative share of categories",
                                "ylabel": "Cumulative share of counts",
                                "data_source": "UnitTest",
                                "file_name": "Lorenz Curve of pets.png"
                            }
                        }
                    },
                },
            },
        ),
    ],
    ids=["basic_report"],
)
def test_univariate_categorical_analysis_report_data_driven(
    tmp_path, load_and_validate_report, assert_plot_metadata,
    make_series, kwargs, expected_sections
):
    s = make_series()
    # point the root to tmp_path so files land where we can check them
    out = univariate_categorical_analysis(
        s, report_root=str(tmp_path), **kwargs
    )
    # load the final univariate report
    full = load_and_validate_report(out, tmp_path)

    # assert metadata
    assert 'metadata' in full
    for key in ("version", "report_name", "parameters"):
        assert key in full['metadata']

    assert "data" in full
    data = full["data"]

    # ---- missing_data assertions ----
    if "missing_data" in expected_sections:
        expected_md = expected_sections["missing_data"]
        actual_md = data["missing_data"]
        for plot_key, expectations in expected_md.items():
            assert plot_key in actual_md, f"missing_data missing key: {plot_key!r}"
            payload = actual_md[plot_key]
            assert_plot_metadata(payload, expectations, tmp_path / s.name.replace(' ', '_'))

    # ---- data_quality assertions ----
    if "data_quality" in expected_sections:
        expected_md = expected_sections["data_quality"]
        actual_md = data["data_quality"]
        for plot_key, expectations in expected_md.items():
            assert plot_key in actual_md, f"data_quality missing key: {plot_key!r}"
            payload = actual_md[plot_key]
            assert_plot_metadata(payload, expectations, tmp_path / s.name.replace(' ', '_'))
    
    # ---- cardinality assertions ----
    if "cardinality" in expected_sections:
        expected_cardinality = expected_sections["cardinality"]
        actual_cardinality = data["cardinality"]
        for plot_key, expectations in expected_cardinality.items():
            assert plot_key in actual_cardinality, f"cardinality missing key: {plot_key!r}"
            payload = actual_cardinality[plot_key]
            assert_plot_metadata(payload, expectations, tmp_path / s.name.replace(' ', '_'))

    # ---- distribution assertions (nested report with plots) ----
    if "distribution" in expected_sections:
        dist_full = load_and_validate_report(data["distribution"], tmp_path / s.name.replace(' ', '_'))
        dist = dist_full["data"]

        # check expected sections/plots
        for section, plots in expected_sections["distribution"].items():
            assert section in dist
            for plot_key, expectations in plots.items():
                payload = dist[section][plot_key]
                assert_plot_metadata(payload, expectations, tmp_path / s.name.replace(' ', '_'))


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
        univariate_categorical_analysis(s)
