import numpy as np
import pandas as pd
import pytest

from analytics_eda.analysis.univariate.univariate_numeric_analysis import (
    univariate_numeric_analysis,
)


@pytest.mark.parametrize(
    "make_series, kwargs, expected_report_data",
    [
        (
            # Slightly noisy numeric data with a single NaN to exercise missing_data
            lambda: pd.Series(
                np.r_[np.random.default_rng(0).normal(loc=10, scale=2, size=120), [np.nan]],
                name="metric",
                dtype="float64",
            ),
            {
                "data_source": "UnitTest",
                # You can also pass plot_*_overrides here later; the test honors them automatically
            },
            {
                # Top-level expectations (from the univariate report)
                "data_quality": {"missing_data_barchart": {"chart_metadata": {"title": "Missing Data for metric", "xlabel": "Status", "ylabel": "Percentage of Total", "data_source": "UnitTest", "file_name": "Missing Data for metric.png"}}},
                # Cardinality section expectations (plot payload in the top-level report)
                "cardinality": {
                    "barchart": {
                        "chart_metadata": {"data_source": "UnitTest"},
                        "descriptive_stats": {"is_discrete": lambda v: isinstance(v, bool)},
                    }
                },
                # Distribution section expectations (these are inside the nested distribution report)
                "distribution": {
                    "central_tendency": {
                        "histogram": {
                            "chart_metadata": {"data_source": "UnitTest"},
                            "descriptive_stats": {},  # nothing specific to assert
                        },
                        "mean_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                        "median_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    },
                    "dispersion": {
                        "boxplot": {
                            "chart_metadata": {"data_source": "UnitTest"},
                            "descriptive_stats": {},
                        },
                        "sigma_bands": {
                            "chart_metadata": {"data_source": "UnitTest"},
                            "descriptive_stats": {},
                        },
                        "percentiles": {
                            "chart_metadata": {"data_source": "UnitTest"},
                            "descriptive_stats": {},
                        },
                    },
                    "shape": {
                        # We'll assert distribution_fits separately (per distribution)
                        "ecdf_gap": {
                            "chart_metadata": {"data_source": "UnitTest"},
                            "descriptive_stats": {},
                        },
                        "density": {
                            "chart_metadata": {"data_source": "UnitTest"},
                            "descriptive_stats": {},
                        },
                        "probability": {
                            "chart_metadata": {"data_source": "UnitTest"},
                            "descriptive_stats": {},
                        },
                        "distribution_fits": {
                            "norm": {
                                "ecdf_vs_cdf": {
                                    "descriptive_stats": {},
                                    "inferential_stats": {},
                                    "chart_metadata": {
                                        "title": "ECDF vs. Theoretical CDF of metric (fitted to norm)",
                                        "xlabel": "Value",
                                        "ylabel": "CDF",
                                        "data_source": "UnitTest",
                                        "file_name": "ECDF vs. Theoretical CDF of metric (fitted to norm).png",
                                    },
                                },
                                "qq_fit": {
                                    "descriptive_stats": {},
                                    "inferential_stats": {},
                                    "chart_metadata": {
                                        "title": "Q\u2013Q Plot Fit Assessment of metric (fitted to norm)",
                                        "xlabel": "Theoretical Quantiles",
                                        "ylabel": "Sample Quantiles",
                                        "data_source": "UnitTest",
                                        "file_name": "Q\u2013Q Plot Fit Assessment of metric (fitted to norm).png",
                                    },
                                },
                            },
                            "lognorm": {
                                "ecdf_vs_cdf": {"descriptive_stats": {}, "inferential_stats": {}, "chart_metadata": {"title": "ECDF vs. Theoretical CDF of metric (fitted to lognorm)", "xlabel": "Value", "ylabel": "CDF", "data_source": "UnitTest"}},
                                "qq_fit": {
                                    "descriptive_stats": {},
                                    "inferential_stats": {},
                                    "chart_metadata": {"title": "Q\u2013Q Plot Fit Assessment of metric (fitted to lognorm)", "xlabel": "Theoretical Quantiles", "ylabel": "Sample Quantiles", "data_source": "UnitTest"},
                                },
                            },
                            "gamma": {
                                "ecdf_vs_cdf": {
                                    "descriptive_stats": {
                                        "params": {
                                            "distribution_name": "gamma",
                                        }
                                    },
                                    "inferential_stats": {},
                                    "chart_metadata": {"title": "ECDF vs. Theoretical CDF of metric (fitted to gamma)", "xlabel": "Value", "ylabel": "CDF", "data_source": "UnitTest"},
                                },
                                "qq_fit": {
                                    "descriptive_stats": {},
                                    "inferential_stats": {},
                                    "chart_metadata": {"title": "Q\u2013Q Plot Fit Assessment of metric (fitted to gamma)", "xlabel": "Theoretical Quantiles", "ylabel": "Sample Quantiles", "data_source": "UnitTest"},
                                },
                            },
                            "expon": {
                                "ecdf_vs_cdf": {"descriptive_stats": {}, "inferential_stats": {}, "chart_metadata": {"title": "ECDF vs. Theoretical CDF of metric (fitted to expon)", "xlabel": "Value", "ylabel": "CDF", "data_source": "UnitTest"}},
                                "qq_fit": {
                                    "descriptive_stats": {},
                                    "inferential_stats": {},
                                    "chart_metadata": {"title": "Q\u2013Q Plot Fit Assessment of metric (fitted to expon)", "xlabel": "Theoretical Quantiles", "ylabel": "Sample Quantiles", "data_source": "UnitTest"},
                                },
                            },
                        },
                    },
                },
            },
        ),
        (
            # 8 values: 3 numeric, 1 numeric-like string ("10"), 1 NaN, and 3 non-numeric strings
            # Non-numeric strings: "x", "bad", "oops"
            lambda: pd.Series([10, "x", 12.5, "bad", np.nan, "10", 8, "oops"], name="metric_strings"),
            {"data_source": "UnitTest"},
            {
                "data_quality": {
                    "missing_data_barchart": {
                        "chart_metadata": {
                            "title": "Missing Data for metric_strings",
                            "data_source": "UnitTest",
                            "file_name": "Missing Data for metric_strings.png",
                        },
                    },
                    "string_coercion_barchart": {
                        "chart_metadata": {
                            "title": "Non-Numeric (String) Values in metric_strings",
                            "xlabel": "Percent of non‑null",
                            "ylabel": "Category",
                            "data_source": "UnitTest",
                            "file_name": "Non-Numeric (String) Values in metric_strings.png",
                        },
                    },
                },
                "cardinality": {
                    "barchart": {
                        "chart_metadata": {"data_source": "UnitTest"},
                    }
                },
                "distribution": {
                    "central_tendency": {
                        "histogram": {"chart_metadata": {"data_source": "UnitTest"}},
                        "mean_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                        "median_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    },
                    "dispersion": {
                        "boxplot": {"chart_metadata": {"data_source": "UnitTest"}},
                        "sigma_bands": {"chart_metadata": {"data_source": "UnitTest"}},
                        "percentiles": {"chart_metadata": {"data_source": "UnitTest"}},
                    },
                    "shape": {
                        "ecdf_gap": {"chart_metadata": {"data_source": "UnitTest"}},
                        "density": {"chart_metadata": {"data_source": "UnitTest"}},
                        "probability": {"chart_metadata": {"data_source": "UnitTest"}},
                    },
                },
            },
        ),
    ],
    ids=["basic_numeric_report", "numeric_with_strings"],
)
def test_univariate_numeric_analysis_report_data_driven(
    tmp_path,
    assert_report_data,
    make_series,
    kwargs,
    expected_report_data,
):
    # Arrange
    s = make_series()

    # Act: run univariate numeric analysis and load the top-level report
    out = univariate_numeric_analysis(s, report_root=str(tmp_path), **kwargs)
    top_dir = tmp_path / s.name.replace(" ", "_")

    # Assert
    assert_report_data(out, expected_report_data, top_dir)


@pytest.mark.parametrize(
    "make_input, exc, pattern",
    [
        # Not a Series
        (lambda: [1, 2, 3], TypeError, r"Input must be a pandas Series."),
        # Missing name
        (lambda: pd.Series([1, 2, 3]), ValueError, r"must have a non-empty 'name'"),
        # Blank/whitespace name
        (lambda: pd.Series([1, 2, 3], name="   "), ValueError, r"must have a non-empty 'name'"),
    ],
    ids=["not_series", "missing_name", "blank_name"],
)
def test_validate_named_series_errors(make_input, exc, pattern, tmp_path):
    with pytest.raises(exc, match=pattern):
        univariate_numeric_analysis(make_input(), report_root=str(tmp_path))
