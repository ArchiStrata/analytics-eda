import pytest
import numpy as np
import pandas as pd
from math import isclose

from analytics_eda.analysis.univariate.univariate_numeric_analysis import univariate_numeric_analysis

@pytest.mark.parametrize(
    "make_series, kwargs, expected",
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
                "data_quality": {
                    "missing_data_barchart": {
                        "descriptive_stats": {
                            "total": 121,
                            "missing": 1,
                            "pct_missing": lambda v: isclose(v, 1/121, rel_tol=1e-12, abs_tol=1e-12),
                            "labels": [
                                "Present",
                                "Missing"
                            ],
                            "counts": "[120   1]",
                            "pcts": "[99.17355372  0.82644628]"
                        },
                        "inferential_stats": {},
                        "chart_metadata": {
                            "title": "Missing Data for metric",
                            "xlabel": "",
                            "ylabel": "Percentage of Total",
                            "data_source": "UnitTest",
                            "file_name": "Missing Data for metric.png"
                        }
                    }
                },
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
                        "violin": {
                            "chart_metadata": {"data_source": "UnitTest"},
                            "descriptive_stats": {},
                        },
                    },
                    "dispersion": {
                        "boxplot": {
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
                                        "file_name": "ECDF vs. Theoretical CDF of metric (fitted to norm).png"
                                    }
                                },
                                "qq": {
                                    "descriptive_stats": {},
                                    "inferential_stats": {},
                                    "chart_metadata": {
                                        "title": "Q\u2013Q Plot Fit Assessment of metric (fitted to norm)",
                                        "xlabel": "Theoretical Quantiles",
                                        "ylabel": "Sample Quantiles",
                                        "data_source": "UnitTest",
                                        "file_name": "Q\u2013Q Plot Fit Assessment of metric (fitted to norm).png"
                                    }
                                }
                            },
                            "lognorm": {
                                "ecdf_vs_cdf": {
                                    "descriptive_stats": {},
                                    "inferential_stats": {},
                                    "chart_metadata": {
                                        "title": "ECDF vs. Theoretical CDF of metric (fitted to lognorm)",
                                        "xlabel": "Value",
                                        "ylabel": "CDF",
                                        "data_source": "UnitTest"
                                    }
                                },
                                "qq": {
                                    "descriptive_stats": {},
                                    "inferential_stats": {},
                                    "chart_metadata": {
                                        "title": "Q\u2013Q Plot Fit Assessment of metric (fitted to lognorm)",
                                        "xlabel": "Theoretical Quantiles",
                                        "ylabel": "Sample Quantiles",
                                        "data_source": "UnitTest"
                                    }
                                }
                            },
                            "gamma": {
                                "ecdf_vs_cdf": {
                                    "descriptive_stats": {
                                        "params": {
                                            "distribution_name": "gamma",
                                        }
                                    },
                                    "inferential_stats": {},
                                    "chart_metadata": {
                                        "title": "ECDF vs. Theoretical CDF of metric (fitted to gamma)",
                                        "xlabel": "Value",
                                        "ylabel": "CDF",
                                        "data_source": "UnitTest"
                                    }
                                },
                                "qq": {
                                    "descriptive_stats": {},
                                    "inferential_stats": {},
                                    "chart_metadata": {
                                        "title": "Q\u2013Q Plot Fit Assessment of metric (fitted to gamma)",
                                        "xlabel": "Theoretical Quantiles",
                                        "ylabel": "Sample Quantiles",
                                        "data_source": "UnitTest"
                                    }
                                }
                            },
                            "expon": {
                                "ecdf_vs_cdf": {
                                    "descriptive_stats": {},
                                    "inferential_stats": {},
                                    "chart_metadata": {
                                        "title": "ECDF vs. Theoretical CDF of metric (fitted to expon)",
                                        "xlabel": "Value",
                                        "ylabel": "CDF",
                                        "data_source": "UnitTest"
                                    }
                                },
                                "qq": {
                                    "descriptive_stats": {},
                                    "inferential_stats": {},
                                    "chart_metadata": {
                                        "title": "Q\u2013Q Plot Fit Assessment of metric (fitted to expon)",
                                        "xlabel": "Theoretical Quantiles",
                                        "ylabel": "Sample Quantiles",
                                        "data_source": "UnitTest"
                                    }
                                }
                            }
                        }
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
                        "descriptive_stats": {
                            "total": 8,
                            "missing": 1,
                            "pct_missing": lambda v: abs(v - 1/8) < 1e-12,
                        },
                        "chart_metadata": {
                            "title": "Missing Data for metric_strings",
                            "data_source": "UnitTest",
                            "file_name": "Missing Data for metric_strings.png",
                        },
                    },
                    "string_coercion_barchart": {
                        "descriptive_stats": {
                            # We count over NON-NULL entries (dropna)
                            "total": 8,
                            # order can vary → check via predicates
                            "labels": lambda xs: set(xs) == {"x", "bad", "oops"},
                            "counts": lambda xs: sorted(list(xs)) == [1, 1, 1],
                        },
                        "inferential_stats": {},
                        "chart_metadata": {
                            "title": "Non-Numeric (String) Values in metric_strings",
                            "xlabel": "Count",
                            "ylabel": "Category",
                            "data_source": "UnitTest",
                            # BasePlot typically names files from title when save_path is provided
                            "file_name": "Non-Numeric (String) Values in metric_strings.png",
                        },
                    }
                },
                "cardinality": {
                    "barchart": {
                        "chart_metadata": {"data_source": "UnitTest"},
                    }
                },
                "distribution": {
                    "central_tendency": {
                        "histogram": {"chart_metadata": {"data_source": "UnitTest"}},
                        "violin":    {"chart_metadata": {"data_source": "UnitTest"}},
                    },
                    "dispersion": {
                        "boxplot":   {"chart_metadata": {"data_source": "UnitTest"}},
                    },
                    "shape": {
                        "ecdf_gap":   {"chart_metadata": {"data_source": "UnitTest"}},
                        "density":    {"chart_metadata": {"data_source": "UnitTest"}},
                        "probability":{"chart_metadata": {"data_source": "UnitTest"}},
                    },
                },
            },
        ),
    ],
    ids=["basic_numeric_report", "numeric_with_strings"],
)
def test_univariate_numeric_analysis_report_data_driven(
    tmp_path,
    load_and_validate_report,
    assert_plot_metadata,
    make_series,
    kwargs,
    expected,
):
    # Arrange
    s = make_series()

    # Act: run univariate numeric analysis and load the top-level report
    out = univariate_numeric_analysis(s, report_root=str(tmp_path), **kwargs)
    top_dir = tmp_path / s.name.replace(" ", "_")
    full = load_and_validate_report(out, top_dir)

    # ---- Top-level metadata sanity ----
    assert "metadata" in full and "data" in full
    for key in ("version", "report_name", "parameters"):
        assert key in full["metadata"]

    data = full["data"]

    # ---- Top-level: data_quality ----
    if "data_quality" in expected:
        expected_md = expected["data_quality"]
        assert "data_quality" in data, "Missing data_quality pillar"
        actual_md = data["data_quality"]
        for plot_key, expectations in expected_md.items():
            assert plot_key in actual_md, f"data_quality missing plot key: {plot_key!r}"
            payload = actual_md[plot_key]
            assert_plot_metadata(payload, expectations, tmp_path / s.name.replace(' ', '_'))

    # ---- Top-level: cardinality (plot payload) ----
    if "cardinality" in expected:
        assert "cardinality" in data and "barchart" in data["cardinality"]
        card_payload = data["cardinality"]["barchart"]
        assert_plot_metadata(card_payload, expected["cardinality"]["barchart"], top_dir)

    # ---- Nested distribution report ----
    if "distribution" in expected:
        dist_full = load_and_validate_report(data["distribution"], top_dir)
        dist = dist_full["data"]
        # Expect core sections
        assert set(dist.keys()) == {"central_tendency", "dispersion", "shape"}

        # Central Tendency plots
        for plot_key, exp in expected["distribution"]["central_tendency"].items():
            assert plot_key in dist["central_tendency"], f"Missing central_tendency plot {plot_key!r}"
            assert_plot_metadata(dist["central_tendency"][plot_key], exp, top_dir)

        # Dispersion plots
        for plot_key, exp in expected["distribution"]["dispersion"].items():
            assert plot_key in dist["dispersion"], f"Missing dispersion plot {plot_key!r}"
            assert_plot_metadata(dist["dispersion"][plot_key], exp, top_dir)

        # Shape plots (except distribution_fits, handled below)
        for plot_key, exp in expected["distribution"]["shape"].items():
            assert plot_key in dist["shape"], f"Missing shape plot {plot_key!r}"

            # Distribution fits (per-named distribution)
            if plot_key == "distribution_fits":
                expected_distribution_fits = exp
                actual_distribution_fits = dist["shape"][plot_key]

                for dist_name, plots in expected_distribution_fits.items():
                    assert dist_name in actual_distribution_fits
                    actual_distribution_fit = actual_distribution_fits[dist_name]

                    for dist_plot_key, dist_exp in plots.items():
                        assert dist_plot_key in actual_distribution_fit, f"missing distribution.{dist_plot_key}"
                        payload = actual_distribution_fit[dist_plot_key]
                        assert_plot_metadata(payload, dist_exp, top_dir)
            elif plot_key == "transforms":
                if kwargs.get("evaluate_transforms_fn") is not None:
                    expected_transforms = exp
                    actual_transforms = dist["shape"][plot_key]

                    for transform_name, expected_data in expected_transforms.items():
                        assert transform_name in actual_transforms
                        actual_transform_report_meta = actual_transforms[transform_name]
                        full_transform_report = load_and_validate_report(actual_transform_report_meta, top_dir / transform_name)
                        report_t = full_transform_report["data"]
                        assert report_t is not None, f"{transform_name!r} entry missing nested 'data'"
            else:
                assert_plot_metadata(dist["shape"][plot_key], exp, top_dir)


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
