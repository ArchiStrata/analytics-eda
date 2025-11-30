import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric.central_tendency import (
    CentralTendencyMeanPointCIContext,
    CentralTendencyMeanPointCIPlot,
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
        # 0) EMPTY → defaults, no CI
        (
            lambda: pd.Series([], dtype="float64", name="nums"),
            {},
            {
                "chart_metadata": {
                    "title": "Mean ± CI for nums",
                    "xlabel": "Value",
                    "ylabel": "",
                    "data_source": None,
                    "file_name": None,
                },
                "descriptive_stats": {
                    "n": 0,
                    "mean": None,
                    "mean_ci": (None, None),
                    "params": {
                        "mean_ci_method": "t",
                        "ci_level": 0.95,
                    },
                },
            },
        ),
        # 1) Happy path (t CI)
        (
            lambda: pd.Series(np.random.default_rng(0).normal(0, 1, size=120), name="x", dtype="float64"),
            {"data_source": "UnitTest", "file_name": "mean_point_ci.png"},
            {
                "chart_metadata": {
                    "title": "Mean ± CI for x",
                    "xlabel": "Value",
                    "ylabel": "",
                    "data_source": "UnitTest",
                    "file_name": "mean_point_ci.png",
                },
                "descriptive_stats": {
                    "n": 120,
                    "mean": pytest.approx(0.0, abs=0.2),
                    "mean_ci": _is_finite_interval,
                    "params": {
                        "mean_ci_method": "t",
                        "ci_level": 0.95,
                    },
                },
            },
        ),
        # 2) Bootstrap CI
        (
            lambda: pd.Series(np.random.default_rng(1).normal(2.0, 0.5, size=80), name="y", dtype="float64"),
            {"mean_ci_method": "bootstrap", "bootstrap_samples": 500, "alpha": 0.1, "file_name": "mean_point_ci.png"},
            {
                "chart_metadata": {
                    "title": "Mean ± CI for y",
                    "xlabel": "Value",
                    "ylabel": "",
                    "file_name": "mean_point_ci.png",
                },
                "descriptive_stats": {
                    "n": 80,
                    "mean": pytest.approx(2.0, abs=0.15),
                    "mean_ci": _is_finite_interval,
                    "params": {
                        "mean_ci_method": "bootstrap",
                        "ci_level": 0.9,
                        "alpha": 0.1,
                    },
                },
            },
        ),
        # 3) Inference: popmean + popvariance (t and z present)
        (
            lambda: pd.Series(np.random.default_rng(2).normal(0.0, 1.0, size=100), name="z", dtype="float64"),
            {"popmean": 0.0, "popvariance": 1.0, "alpha": 0.05, "file_name": "mean_point_ci.png"},
            {
                "chart_metadata": {"file_name": "mean_point_ci.png"},
                "descriptive_stats": {
                    "n": 100,
                    "mean_ci": _is_finite_interval,
                },
                "inferential_stats": {
                    "params": {"alpha": 0.05, "popmean": 0.0, "popvariance": 1.0},
                    "popmean": (lambda d: all(k in d for k in ["t_test", "z_test", "cohens_d"])),
                },
            },
        ),
        # 4) Custom chart_metadata
        (
            lambda: pd.Series(np.random.default_rng(3).normal(1.5, 0.4, size=60), name="w", dtype="float64"),
            {"title_template": "My Mean Plot: {name}", "filter_desc": "ignored", "transform_desc": "ignored", "xlabel": "Score", "ylabel": "Confidence", "data_source": "UnitTest", "file_name": "mean_point_ci.png"},
            {
                "chart_metadata": {"title": "My Mean Plot: w", "xlabel": "Score", "ylabel": "Confidence", "data_source": "UnitTest", "file_name": "mean_point_ci.png"},
                "descriptive_stats": {
                    "n": 60,
                },
            },
        ),
    ],
    ids=[
        "empty",
        "t_ci_happy",
        "bootstrap_ci",
        "inference_t_and_z",
        "custom_title_template",
    ],
)
def test_central_tendency_mean_point_ci_plot_data_driven(make_series, kwargs, expect, tmp_path, assert_plot_metadata):
    s = make_series()
    if "file_name" in kwargs:
        kwargs = {**kwargs, "base_dir": tmp_path}

    ctx = CentralTendencyMeanPointCIContext(**kwargs)
    plot = CentralTendencyMeanPointCIPlot(ctx)
    payload = plot.run(s)

    assert_plot_metadata(payload, expect, tmp_path)
