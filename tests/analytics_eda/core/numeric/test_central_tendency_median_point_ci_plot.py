import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric.central_tendency_median_point_ci_plot import (
    CentralTendencyMedianPointCIContext,
    CentralTendencyMedianPointCIPlot,
)


def _is_finite_interval(v):
    try:
        lo, hi = v
        return np.isfinite(lo) and np.isfinite(hi) and (lo <= hi)
    except Exception:
        return False


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        # 0) EMPTY → defaults, no CI, {} inferential/draft findings implied by assert helper
        (
            lambda: pd.Series([], dtype="float64", name="nums"),
            {},
            {
                "chart_metadata": {
                    "title": "Median ± CI for nums",
                    "xlabel": "Value",
                    "ylabel": "",
                    "data_source": None,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "n": 0,
                    "median": None,
                    "median_ci": (None, None),
                    "params": {
                        "median_ci_method": "bootstrap",
                        "ci_level": 0.95,
                    },
                },
            },
        ),
        # 1) Happy path (bootstrap CI, default alpha=0.05)
        (
            lambda: pd.Series(np.random.default_rng(0).normal(0.0, 1.0, size=120), name="x", dtype="float64"),
            {"data_source": "UnitTest", "file_name": "median_point_ci.png"},
            {
                "chart_metadata": {"title": "Median ± CI for x", "xlabel": "Value", "ylabel": "", "data_source": "UnitTest", "file_name": "median_point_ci.png"},
                "descriptive_stats": {
                    "n": 120,
                    "median": pytest.approx(0.0, abs=0.2),
                    "median_ci": _is_finite_interval,
                    "params": {
                        "median_ci_method": "bootstrap",
                        "ci_level": 0.95,
                    },
                },
            },
        ),
        # 2) No CI (median_ci_method=None) – still returns median, but CI is (None, None)
        (
            lambda: pd.Series(np.random.default_rng(1).normal(2.0, 0.5, size=80), name="y", dtype="float64"),
            {"median_ci_method": None, "alpha": 0.10, "file_name": "median_point_ci.png"},
            {
                "chart_metadata": {"title": "Median ± CI for y", "xlabel": "Value", "ylabel": "", "file_name": "median_point_ci.png"},
                "descriptive_stats": {
                    "n": 80,
                    "median": pytest.approx(2.0, abs=0.15),
                    "median_ci": (None, None),
                    "params": {
                        "median_ci_method": None,
                        "ci_level": 0.90,
                        "alpha": 0.10,
                    },
                },
            },
        ),
        # 3) Inference: popmedian present → Wilcoxon and Sign test available (structure check)
        (
            lambda: pd.Series(np.random.default_rng(2).normal(0.0, 1.0, size=100), name="z", dtype="float64"),
            {"popmedian": 0.0, "alpha": 0.05, "bootstrap_samples": 400, "file_name": "median_point_ci.png"},
            {
                "descriptive_stats": {
                    "n": 100,
                    "median_ci": _is_finite_interval,
                },
                "inferential_stats": {
                    "params": {"alpha": 0.05, "popmedian": 0.0},
                    # ensure both tests are present with expected fields (not asserting values)
                    "popmedian": (
                        lambda d: (
                            all(k in d for k in ["wilcoxon", "sign_test"])
                            and all(k in d["wilcoxon"] for k in ["statistic", "p_value", "alpha", "reject"])
                            and all(k in d["sign_test"] for k in ["n", "num_positive", "num_negative", "p_value", "alpha", "reject"])
                        )
                    ),
                },
            },
        ),
        # 4) Custom chart_metadata
        (
            lambda: pd.Series(np.random.default_rng(3).normal(1.5, 0.4, size=60), name="w", dtype="float64"),
            {"title_template": "My Median Plot: {name}", "filter_desc": "ignored", "transform_desc": "ignored", "xlabel": "Score", "ylabel": "Confidence", "data_source": "UnitTest", "file_name": "median_point_ci.png"},
            {
                "chart_metadata": {
                    "title": "My Median Plot: w",
                    "xlabel": "Score",
                    "ylabel": "Confidence",
                    "data_source": "UnitTest",
                    "file_name": "median_point_ci.png",
                },
                "descriptive_stats": {"n": 60},
            },
        ),
        # 5) Edge: many ties around popmedian → sign test still reports counts; Wilcoxon handles ties
        (
            lambda: pd.Series(
                np.concatenate(
                    [
                        np.zeros(20),  # ties exactly at popmedian
                        np.ones(15) * 0.1,  # slight positives
                        -np.ones(15) * 0.1,  # slight negatives
                    ]
                ),
                name="tied",
                dtype="float64",
            ),
            {"popmedian": 0.0, "alpha": 0.05, "bootstrap_samples": 300, "file_name": "median_point_ci.png"},
            {
                "descriptive_stats": {"n": 50},
                "inferential_stats": {
                    "params": {"alpha": 0.05, "popmedian": 0.0},
                    "popmedian": (
                        lambda d: (
                            "sign_test" in d and all(k in d["sign_test"] for k in ["n", "num_positive", "num_negative", "p_value", "alpha", "reject"]) and "wilcoxon" in d  # wilcoxon present, even if p-value/statistic may reflect ties
                        )
                    ),
                },
            },
        ),
    ],
    ids=[
        "empty",
        "bootstrap_ci_happy",
        "no_ci_method_none",
        "inference_wilcoxon_and_sign",
        "custom_title_template",
        "ties_around_popmedian",
    ],
)
def test_central_tendency_median_point_ci_plot_data_driven(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()
    if "file_name" in kwargs:
        kwargs = {**kwargs, "save_path": tmp_path}

    ctx = CentralTendencyMedianPointCIContext(**kwargs)
    plot = CentralTendencyMedianPointCIPlot(ctx)
    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
