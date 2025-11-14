import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric.evaluate_transforms import evaluate_transforms
from analytics_eda.core.numeric.numeric_distribution_analysis import numeric_distribution_analysis


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
    "make_series, kwargs, expected_report_data",
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
                    "mean_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "median_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "sigma_bands": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "percentiles": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "distribution_fits": {
                        "norm": {
                            "ecdf_vs_cdf": {
                                "descriptive_stats": {"n": 150, "params": {"distribution_name": "norm", "distribution_fit": [0.063, 0.959]}, "ks_D": 0.0405234014173278},
                                "inferential_stats": {
                                    "params": {"alpha": 0.05},
                                    "ks": {"statistic": 0.040439623105054445, "p_value": 0.9585664995494558, "reject": False},
                                    "anderson": {"statistic": 0.40396592474297677, "critical_value": 0.767, "critical_values": [0.562, 0.64, 0.767, 0.895, 1.065], "significance_levels": [15.0, 10.0, 5.0, 2.5, 1.0], "reject": False},
                                    "cvm": {"statistic": 0.0556707443438706, "p_value": 0.8419518956667803, "reject": False},
                                },
                                "chart_metadata": {"title": "ECDF vs. Theoretical CDF of norm (fitted to norm)", "xlabel": "Value", "ylabel": "CDF", "data_source": "UnitTest", "file_name": "ECDF vs. Theoretical CDF of norm (fitted to norm).png"},
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
                                    "min": -2.3653039062769743,
                                },
                                "inferential_stats": {"params": {"alpha": 0.05, "distribution_name": "norm"}, "dagostino_pearson": {"statistic": 2.4918425812847533, "p_value": 0.28767575302776366, "reject": False}, "reject_normality": False},
                                "chart_metadata": {
                                    "title": "Q\u2013Q Plot Fit Assessment of norm (fitted to norm)",
                                    "xlabel": "Theoretical Quantiles",
                                    "ylabel": "Sample Quantiles",
                                    "data_source": "UnitTest",
                                    "file_name": "Q\u2013Q Plot Fit Assessment of norm (fitted to norm).png",
                                },
                            },
                        },
                        "lognorm": {
                            "ecdf_vs_cdf": {
                                "descriptive_stats": {"n": 150, "params": {"distribution_name": "lognorm"}, "error": "requires positive data", "skip_plot": True, "x": "[]", "ecdf": "[]", "cdf_theo": "[]"},
                                "inferential_stats": {"params": {"alpha": 0.05}},
                                "chart_metadata": {"title": "ECDF vs. Theoretical CDF of norm (fitted to lognorm)", "xlabel": "Value", "ylabel": "CDF", "data_source": "UnitTest"},
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
                                    "skip_plot": True,
                                },
                                "inferential_stats": {"params": {"alpha": 0.05, "distribution_name": "lognorm"}},
                                "chart_metadata": {"title": "Q\u2013Q Plot Fit Assessment of norm (fitted to lognorm)", "xlabel": "Theoretical Quantiles", "ylabel": "Sample Quantiles", "data_source": "UnitTest"},
                            },
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
                                    "cdf_theo": "[]",
                                },
                                "inferential_stats": {"params": {"alpha": 0.05}},
                                "chart_metadata": {"title": "ECDF vs. Theoretical CDF of norm (fitted to gamma)", "xlabel": "Value", "ylabel": "CDF", "data_source": "UnitTest"},
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
                                    "skip_plot": True,
                                },
                                "inferential_stats": {"params": {"alpha": 0.05, "distribution_name": "gamma"}},
                                "chart_metadata": {"title": "Q\u2013Q Plot Fit Assessment of norm (fitted to gamma)", "xlabel": "Theoretical Quantiles", "ylabel": "Sample Quantiles", "data_source": "UnitTest"},
                            },
                        },
                        "expon": {
                            "ecdf_vs_cdf": {
                                "descriptive_stats": {"n": 150, "params": {"distribution_name": "expon", "distribution_fit": None}, "error": "requires non-negative data", "skip_plot": True, "x": "[]", "ecdf": "[]", "cdf_theo": "[]", "ks_D": None},
                                "inferential_stats": {"params": {"alpha": 0.05}},
                                "chart_metadata": {"title": "ECDF vs. Theoretical CDF of norm (fitted to expon)", "xlabel": "Value", "ylabel": "CDF", "data_source": "UnitTest"},
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
                                    "skip_plot": True,
                                },
                                "inferential_stats": {"params": {"alpha": 0.05, "distribution_name": "expon"}},
                                "chart_metadata": {"title": "Q\u2013Q Plot Fit Assessment of norm (fitted to expon)", "xlabel": "Theoretical Quantiles", "ylabel": "Sample Quantiles", "data_source": "UnitTest"},
                            },
                        },
                    },
                },
            },
        ),
        # 2) Lognormal (strictly positive)
        (
            lambda: pd.Series(np.random.default_rng(1).lognormal(mean=0.0, sigma=0.8, size=150), name="lognorm", dtype="float64"),
            {"data_source": "UnitTest", "is_discrete": False},
            {
                "central_tendency": {
                    "histogram": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "mean_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "median_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "sigma_bands": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "percentiles": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
            },
        ),
        # 3) Gamma (strictly positive)
        (
            lambda: pd.Series(np.random.default_rng(2).gamma(shape=2.0, scale=2.0, size=150), name="gamma", dtype="float64"),
            {"data_source": "UnitTest", "is_discrete": False},
            {
                "central_tendency": {
                    "histogram": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "mean_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "median_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "sigma_bands": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "percentiles": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
            },
        ),
        # 4) Exponential (non‑negative)
        (
            lambda: pd.Series(np.random.default_rng(3).exponential(scale=1.5, size=150), name="expon", dtype="float64"),
            {"data_source": "UnitTest", "is_discrete": False},
            {
                "central_tendency": {
                    "histogram": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "mean_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "median_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "sigma_bands": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "percentiles": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
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
                    "mean_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "median_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "sigma_bands": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "percentiles": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "transforms": {"yeo-johnson": {}, "arcsinh": {}},
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
                    "mean_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "median_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "sigma_bands": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "percentiles": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
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
                    "mean_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "median_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "sigma_bands": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "percentiles": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
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
                    "mean_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "median_point_ci": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "sigma_bands": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "percentiles": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
            },
        ),
    ],
    ids=[
        "norm_series",
        "lognorm_series",
        "gamma_series",
        "expon_series",
        "norm_series_with_transforms",
        "lognorm_series_with_transforms",
        "gamma_series_with_transforms",
        "expon_series_with_transforms",
    ],
)
def test_numeric_distribution_analysis_param(make_series, kwargs, expected_report_data, tmp_path, assert_report_data):
    # Arrange
    s = make_series()

    # Act: run the full numeric distribution analysis (which writes nested reports)
    out = numeric_distribution_analysis(s, report_path=tmp_path, **kwargs)

    # Assert
    assert_report_data(out, expected_report_data, tmp_path)
