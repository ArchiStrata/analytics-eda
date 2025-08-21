import pytest
import numpy as np
import pandas as pd
from analytics_eda.core.numeric.numeric_distribution_analysis import numeric_distribution_analysis
from analytics_eda.core.numeric.evaluate_transforms import evaluate_transforms

@pytest.mark.parametrize(
    "make_input, exc, pattern",
    [
        # Not a Series
        (lambda: [1, 2, 3], TypeError, r"Input must be a pandas Series."),
        # Non-numeric Series
        (lambda: pd.Series(["a", "b", "c"], name="letters"), TypeError, r"Series must be numeric"),
        # Missing name
        (lambda: pd.Series([1, 2, 3]), ValueError, r"must have a non-empty 'name'"),
        # Blank/whitespace name
        (lambda: pd.Series([1, 2, 3], name="   "), ValueError, r"must have a non-empty 'name'"),
    ],
    ids=["not_series", "non_numeric", "missing_name", "blank_name"],
)
def test_validate_numeric_named_series_errors(make_input, exc, pattern, tmp_path):
    with pytest.raises(exc, match=pattern):
        numeric_distribution_analysis(make_input(), report_path=tmp_path, is_discrete=True)


@pytest.mark.parametrize(
    "make_series, kwargs, expected_distribution",
    [
        # --------------------------
        # Baseline (no transforms)
        # --------------------------
        # 1) Normal
        (
            lambda: pd.Series(np.random.default_rng(0).normal(loc=0, scale=1, size=150), name="norm", dtype="float64"),
            {"data_source": "UnitTest", "is_discrete": False},
            {
                "central_tendency": {
                    "histogram": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "violin":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap":   {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability":{"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "distribution_fits": {
                        "norm": {
                            "ecdf_vs_cdf": {
                                "descriptive_stats": {
                                    "n": 150,
                                    "params": {
                                        "distribution_name": "norm",
                                        "distribution_fit": [
                                            0.063,
                                            0.959
                                        ]
                                    },
                                    "ks_D": 0.0405234014173278
                                },
                                "inferential_stats": {
                                    "params": {
                                        "alpha": 0.05
                                    },
                                    "ks": {
                                        "statistic": 0.040439623105054445,
                                        "p_value": 0.9585664995494558,
                                        "reject": False
                                    },
                                    "anderson": {
                                        "statistic": 0.40396592474297677,
                                        "critical_value": 0.767,
                                        "critical_values": [
                                            0.562,
                                            0.64,
                                            0.767,
                                            0.895,
                                            1.065
                                        ],
                                        "significance_levels": [
                                            15.0,
                                            10.0,
                                            5.0,
                                            2.5,
                                            1.0
                                        ],
                                        "reject": False
                                    },
                                    "cvm": {
                                        "statistic": 0.0556707443438706,
                                        "p_value": 0.8419518956667803,
                                        "reject": False
                                    }
                                },
                                "chart_metadata": {
                                    "title": "ECDF vs. Theoretical CDF of norm (fitted to norm)",
                                    "xlabel": "Value",
                                    "ylabel": "CDF",
                                    "data_source": "UnitTest",
                                    "file_name": "ECDF vs. Theoretical CDF of norm (fitted to norm).png"
                                }
                            },
                            "qq_fit": {
                                "descriptive_stats": {
                                    "intercept": 0.00010465900704410797,
                                    "slope": 0.9983392874998688,
                                    "r_squared": 0.9882016901688403,
                                    "median_residual": 0.022879929960715417,
                                    "iqr_residual": 0.11265581094199321,
                                    "max_abs_residual": 0.6571515339829359,
                                    "skewness": -0.20357498056930046,
                                    "kurtosis": -0.4148577220304337,
                                    "min": -2.3653039062769743
                                },
                                "inferential_stats": {
                                    "params": {
                                        "alpha": 0.05,
                                        "distribution_name": "norm"
                                    },
                                    "dagostino_pearson": {
                                        "statistic": 2.4918425812847533,
                                        "p_value": 0.28767575302776366,
                                        "reject": False
                                    },
                                    "reject_normality": False
                                },
                                "chart_metadata": {
                                    "title": "Q\u2013Q Plot Fit Assessment of norm (fitted to norm)",
                                    "xlabel": "Theoretical Quantiles",
                                    "ylabel": "Sample Quantiles",
                                    "data_source": "UnitTest",
                                    "file_name": "Q\u2013Q Plot Fit Assessment of norm (fitted to norm).png"
                                }
                            }
                        },
                        "lognorm": {
                            "ecdf_vs_cdf": {
                                "descriptive_stats": {
                                    "n": 150,
                                    "params": {
                                        "distribution_name": "lognorm"
                                    },
                                    "error": "requires positive data",
                                    "skip_plot": True,
                                    "x": "[]",
                                    "ecdf": "[]",
                                    "cdf_theo": "[]"
                                },
                                "inferential_stats": {
                                    "params": {
                                        "alpha": 0.05
                                    }
                                },
                                "chart_metadata": {
                                    "title": "ECDF vs. Theoretical CDF of norm (fitted to lognorm)",
                                    "xlabel": "Value",
                                    "ylabel": "CDF",
                                    "data_source": "UnitTest"
                                }
                            },
                            "qq_fit": {
                                "descriptive_stats": {
                                    "intercept": None,
                                    "slope": None,
                                    "r_squared": None,
                                    "median_residual": None,
                                    "iqr_residual": None,
                                    "max_abs_residual": None,
                                    "skewness": None,
                                    "kurtosis": None,
                                    "min": -2.3653039062769743,
                                    "error": "requires positive data",
                                    "skip_plot": True
                                },
                                "inferential_stats": {
                                    "params": {
                                        "alpha": 0.05,
                                        "distribution_name": "lognorm"
                                    }
                                },
                                "chart_metadata": {
                                    "title": "Q\u2013Q Plot Fit Assessment of norm (fitted to lognorm)",
                                    "xlabel": "Theoretical Quantiles",
                                    "ylabel": "Sample Quantiles",
                                    "data_source": "UnitTest"
                                }
                            }
                        },
                        "gamma": {
                            "ecdf_vs_cdf": {
                                "descriptive_stats": {
                                    "n": 150,
                                    "params": {
                                        "distribution_name": "gamma",
                                    },
                                    "error": "requires positive data",
                                    "skip_plot": True,
                                    "x": "[]",
                                    "ecdf": "[]",
                                    "cdf_theo": "[]"
                                },
                                "inferential_stats": {
                                    "params": {
                                        "alpha": 0.05
                                    }
                                },
                                "chart_metadata": {
                                    "title": "ECDF vs. Theoretical CDF of norm (fitted to gamma)",
                                    "xlabel": "Value",
                                    "ylabel": "CDF",
                                    "data_source": "UnitTest"
                                }
                            },
                            "qq_fit": {
                                "descriptive_stats": {
                                    "intercept": None,
                                    "slope": None,
                                    "r_squared": None,
                                    "median_residual": None,
                                    "iqr_residual": None,
                                    "max_abs_residual": None,
                                    "skewness": None,
                                    "kurtosis": None,
                                    "min": -2.3653039062769743,
                                    "error": "requires positive data",
                                    "skip_plot": True
                                },
                                "inferential_stats": {
                                    "params": {
                                        "alpha": 0.05,
                                        "distribution_name": "gamma"
                                    }
                                },
                                "chart_metadata": {
                                    "title": "Q\u2013Q Plot Fit Assessment of norm (fitted to gamma)",
                                    "xlabel": "Theoretical Quantiles",
                                    "ylabel": "Sample Quantiles",
                                    "data_source": "UnitTest"
                                }
                            }
                        },
                        "expon": {
                            "ecdf_vs_cdf": {
                                "descriptive_stats": {
                                    "n": 150,
                                    "params": {
                                        "distribution_name": "expon",
                                        "distribution_fit": None
                                    },
                                    "error": "requires non-negative data",
                                    "skip_plot": True,
                                    "x": "[]",
                                    "ecdf": "[]",
                                    "cdf_theo": "[]",
                                    "ks_D": None
                                },
                                "inferential_stats": {
                                    "params": {
                                        "alpha": 0.05
                                    }
                                },
                                "chart_metadata": {
                                    "title": "ECDF vs. Theoretical CDF of norm (fitted to expon)",
                                    "xlabel": "Value",
                                    "ylabel": "CDF",
                                    "data_source": "UnitTest"
                                }
                            },
                            "qq_fit": {
                                "descriptive_stats": {
                                    "intercept": None,
                                    "slope": None,
                                    "r_squared": None,
                                    "median_residual": None,
                                    "iqr_residual": None,
                                    "max_abs_residual": None,
                                    "skewness": None,
                                    "kurtosis": None,
                                    "min": -2.3653039062769743,
                                    "error": "requires non-negative data",
                                    "skip_plot": True
                                },
                                "inferential_stats": {
                                    "params": {
                                        "alpha": 0.05,
                                        "distribution_name": "expon"
                                    }
                                },
                                "chart_metadata": {
                                    "title": "Q\u2013Q Plot Fit Assessment of norm (fitted to expon)",
                                    "xlabel": "Theoretical Quantiles",
                                    "ylabel": "Sample Quantiles",
                                    "data_source": "UnitTest"
                                }
                            }
                        }
                    }
                }
            },
        ),
        # 2) Lognormal (strictly positive)
        (
            lambda: pd.Series(np.random.default_rng(1).lognormal(mean=0.0, sigma=0.8, size=150), name="lognorm", dtype="float64"),
            {"data_source": "UnitTest", "is_discrete": False},
            {
                "central_tendency": {
                    "histogram": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "violin":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap":   {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability":{"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                }
            },
        ),
        # 3) Gamma (strictly positive)
        (
            lambda: pd.Series(np.random.default_rng(2).gamma(shape=2.0, scale=2.0, size=150), name="gamma", dtype="float64"),
            {"data_source": "UnitTest", "is_discrete": False},
            {
                "central_tendency": {
                    "histogram": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "violin":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap":   {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability":{"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                }
            },
        ),
        # 4) Exponential (non‑negative)
        (
            lambda: pd.Series(np.random.default_rng(3).exponential(scale=1.5, size=150), name="expon", dtype="float64"),
            {"data_source": "UnitTest", "is_discrete": False},
            {
                "central_tendency": {
                    "histogram": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "violin":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap":   {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability":{"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
            },
        ),
        # -----------------------------------------
        # Same 4 scenarios WITH transforms enabled
        # -----------------------------------------
        # 1) Normal + transforms
        (
            lambda: pd.Series(np.random.default_rng(0).normal(loc=0, scale=1, size=150), name="norm", dtype="float64"),
            {"data_source": "UnitTest", "is_discrete": False, "evaluate_transforms_fn": evaluate_transforms},
            {
                "central_tendency": {
                    "histogram": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "violin":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap":   {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability":{"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "transforms": {"yeo-johnson": {}, "arcsinh": {}}
                },
            },
        ),
        # 2) Lognormal + transforms
        (
            lambda: pd.Series(np.random.default_rng(1).lognormal(mean=0.0, sigma=0.8, size=150), name="lognorm", dtype="float64"),
            {"data_source": "UnitTest", "is_discrete": False, "evaluate_transforms_fn": evaluate_transforms},
            {
                "central_tendency": {
                    "histogram": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "violin":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap":   {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability":{"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
            },
        ),
        # 3) Gamma + transforms
        (
            lambda: pd.Series(np.random.default_rng(2).gamma(shape=2.0, scale=2.0, size=150), name="gamma", dtype="float64"),
            {"data_source": "UnitTest", "is_discrete": False, "evaluate_transforms_fn": evaluate_transforms},
            {
                "central_tendency": {
                    "histogram": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "violin":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap":   {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability":{"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
            },
        ),
        # 4) Exponential + transforms
        (
            lambda: pd.Series(np.random.default_rng(3).exponential(scale=1.5, size=150), name="expon", dtype="float64"),
            {"data_source": "UnitTest", "is_discrete": False, "evaluate_transforms_fn": evaluate_transforms},
            {
                "central_tendency": {
                    "histogram": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "violin":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap":   {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability":{"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
            },
        ),
    ],
    ids=[
        "norm_series", "lognorm_series", "gamma_series", "expon_series",
        "norm_series_with_transforms", "lognorm_series_with_transforms",
        "gamma_series_with_transforms", "expon_series_with_transforms",
    ],
)
def test_numeric_distribution_analysis_param(
    make_series,
    kwargs,
    expected_distribution,
    tmp_path,
    load_and_validate_report,
    assert_plot_metadata,
):
    # Arrange
    s = make_series()

    # Act: run the full numeric distribution analysis (which writes nested reports)
    out = numeric_distribution_analysis(s, report_path=tmp_path, **kwargs)

    # Load the nested "distribution" report
    dist_loaded = load_and_validate_report(response=out, report_dir=tmp_path)["data"]  # fixture reads latest JSON under root

    # Assert the three distribution subsections exist
    assert set(dist_loaded.keys()) == {"central_tendency", "dispersion", "shape"}

    # ---- central_tendency ----
    for plot_key, exp in expected_distribution["central_tendency"].items():
        assert plot_key in dist_loaded["central_tendency"], f"missing central_tendency.{plot_key}"
        payload = dist_loaded["central_tendency"][plot_key]
        assert_plot_metadata(payload, exp, tmp_path)

    # ---- dispersion ----
    for plot_key, exp in expected_distribution["dispersion"].items():
        assert plot_key in dist_loaded["dispersion"], f"missing dispersion.{plot_key}"
        payload = dist_loaded["dispersion"][plot_key]
        assert_plot_metadata(payload, exp, tmp_path)

    # ---- shape ----
    for plot_key, exp in expected_distribution["shape"].items():
        assert plot_key in dist_loaded["shape"], f"missing shape.{plot_key}"

        # Distribution fits (per-named distribution)
        if plot_key == "distribution_fits":
            expected_distribution_fits = exp
            actual_distribution_fits = dist_loaded["shape"][plot_key]

            for dist_name, plots in expected_distribution_fits.items():
                assert dist_name in actual_distribution_fits
                actual_distribution_fit = actual_distribution_fits[dist_name]

                for dist_plot_key, dist_exp in plots.items():
                    assert dist_plot_key in actual_distribution_fit, f"missing distribution.{dist_plot_key}"
                    payload = actual_distribution_fit[dist_plot_key]
                    assert_plot_metadata(payload, dist_exp, tmp_path)
        elif plot_key == "transforms":
            if kwargs.get("evaluate_transforms_fn") is not None:
                expected_transforms = exp
                actual_transforms = dist_loaded["shape"][plot_key]

                for transform_name, expected_data in expected_transforms.items():
                    assert transform_name in actual_transforms
                    actual_transform_report_meta = actual_transforms[transform_name]
                    full_transform_report = load_and_validate_report(actual_transform_report_meta, tmp_path / transform_name)
                    report_t = full_transform_report["data"]
                    assert report_t is not None, f"{transform_name!r} entry missing nested 'data'"

                    # ---- central_tendency ----
                    if "central_tendency" in expected_data:
                        for plot_key, exp in expected_data["central_tendency"].items():
                            assert plot_key in report_t["central_tendency"], f"missing central_tendency.{plot_key}"
                            payload = report_t["central_tendency"][plot_key]
                            assert_plot_metadata(payload, exp, tmp_path)
        else:
            payload = dist_loaded["shape"][plot_key]
            assert_plot_metadata(payload, exp, tmp_path)
