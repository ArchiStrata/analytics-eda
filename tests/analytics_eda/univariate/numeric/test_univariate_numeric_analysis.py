import pytest
import numpy as np
import pandas as pd
from math import isclose

from analytics_eda.univariate.numeric.univariate_numeric_analysis import univariate_numeric_analysis

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
                "missing_data": {
                    "total": 121,
                    "missing": 1,
                    "pct_missing": lambda v: isclose(v, 1/121, rel_tol=1e-12, abs_tol=1e-12),
                },
                # Cardinality section expectations (plot payload in the top-level report)
                "cardinality": {
                    "plot_cardinality_barchart": {
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
                    },
                    # Expected distribution names for the fit section
                    "distribution_names": ("norm", "lognorm", "gamma", "expon"),
                },
            },
        ),
    ],
    ids=["basic_numeric_report"],
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
    assert set(data.keys()) == {"missing_data", "cardinality", "distribution"}

    # ---- Top-level: missing_data ----
    expected_md = expected["missing_data"]
    actual_md = data["missing_data"]
    for key, exp in expected_md.items():
        assert key in actual_md, f"missing_data missing key: {key!r}"
        if callable(exp):
            assert exp(actual_md[key]), f"Predicate failed for missing_data[{key!r}] = {actual_md[key]!r}"
        else:
            assert actual_md[key] == exp, f"missing_data[{key!r}] expected {exp!r}, got {actual_md[key]!r}"

    # ---- Top-level: cardinality (plot payload) ----
    assert "cardinality" in data and "plot_cardinality_barchart" in data["cardinality"]
    card_payload = data["cardinality"]["plot_cardinality_barchart"]
    assert_plot_metadata(card_payload, expected["cardinality"]["plot_cardinality_barchart"], top_dir)

    # ---- Nested distribution report ----
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
        assert_plot_metadata(dist["shape"][plot_key], exp, top_dir)

    # Distribution fits (per-named distribution)
    assert "distribution_fits" in dist["shape"], "Missing 'distribution_fits' in shape section"
    fits = dist["shape"]["distribution_fits"]
    expected_names = set(expected["distribution"]["distribution_names"])
    assert set(fits.keys()) == expected_names

    # For each distribution: assert ecdf_vs_cdf and qq plots are present & saved
    for dist_name in expected_names:
        assert set(fits[dist_name].keys()) == {"ecdf_vs_cdf", "qq"}
        # ecdf_vs_cdf
        evc_exp = {
            "chart_metadata": {"data_source": kwargs.get("data_source"), "distribution": dist_name},
            "descriptive_stats": {"distribution": lambda v, dn=dist_name: v == dn, "n": lambda v: v > 0},
        }
        assert_plot_metadata(fits[dist_name]["ecdf_vs_cdf"], evc_exp, top_dir)
        # qq
        qq_exp = {
            "chart_metadata": {"data_source": kwargs.get("data_source"), "distribution": dist_name},
            "descriptive_stats": {},  # could assert slope/intercept/etc. exist, but keep concise here
        }
        assert_plot_metadata(fits[dist_name]["qq"], qq_exp, top_dir)

@pytest.mark.parametrize(
    "make_input, exc, pattern",
    [
        # Not a Series
        (lambda: [1, 2, 3], TypeError, r"Input must be a pandas Series\."),
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
        univariate_numeric_analysis(make_input(), report_root=str(tmp_path))
