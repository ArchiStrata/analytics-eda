import os
import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric import plot_distribution_ecdf_vs_cdf

def test_missing_series_name_raises_error():
    series = pd.Series([1, 2, 3])  # name is None
    with pytest.raises(ValueError):
        plot_distribution_ecdf_vs_cdf(series, "norm")

def test_invalid_distribution_name_raises_value_error():
    s = pd.Series([1, 2, 3], dtype=float, name="x")
    with pytest.raises(ValueError):
        plot_distribution_ecdf_vs_cdf(s, "invalid")

def test_empty_series_returns_default_metadata():
    s = pd.Series([], dtype=float, name="x")
    res = plot_distribution_ecdf_vs_cdf(s, "norm")
    # descriptive_stats and tests
    assert res["descriptive_stats"] == {"n": 0}
    assert res["inferential_stats"] == {}
    # chart metadata defaults
    cm = res["chart_metadata"]
    assert cm["distribution"] == "norm"
    assert cm["file_name"] is None

@pytest.mark.parametrize("dist", ["lognorm", "gamma"])
def test_requires_positive_data_error(dist):
    s = pd.Series([0.0, 0.0, 1.0], dtype=float, name="x")
    res = plot_distribution_ecdf_vs_cdf(s, dist)
    ds = res["descriptive_stats"]
    assert ds["n"] == 3
    assert ds["error"] == "requires positive data"
    assert res["inferential_stats"] == {}


def test_requires_non_negative_data_error_expon():
    s = pd.Series([-1.0, 0.0, 1.0], dtype=float, name="x")
    res = plot_distribution_ecdf_vs_cdf(s, "expon")
    ds = res["descriptive_stats"]
    assert ds["n"] == 3
    assert ds["error"] == "requires non-negative data"
    assert res["inferential_stats"] == {}


@pytest.mark.parametrize(
    "dist_name, rng_func, has_ad",
    [
        ("norm", lambda rng: rng.normal(size=150), True),
        ("lognorm", lambda rng: rng.lognormal(size=150), False),
        ("gamma", lambda rng: rng.gamma(shape=2.0, scale=2.0, size=150), False),
        ("expon", lambda rng: rng.exponential(size=150), True),
    ]
)
def test_all_distributions_and_save(dist_name, rng_func, has_ad, tmp_path):
    rng = np.random.default_rng(42)
    raw = rng_func(rng).astype(float)

    # generate series with valid support
    if dist_name in ("lognorm", "gamma"):
        raw = np.abs(raw) + 1e-6
    elif dist_name == "expon":
        raw = np.abs(raw)

    series = pd.Series(raw, dtype=float, name="x")
    outdir = tmp_path / "plots"
    file_name = f"{dist_name}.png"

    ds, tests, cm = _fit_and_collect(series, dist_name, save_path=str(outdir), file_name=file_name)

    # verify file saved
    saved = outdir / file_name
    assert saved.exists() and saved.stat().st_size > 0
    # PNG signature
    assert saved.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert os.path.basename(cm["file_name"]) == file_name

    # chart metadata defaults
    assert cm["distribution"] == dist_name
    assert cm["alpha"] == 0.05
    assert cm["title"] == f"ECDF vs. Theoretical CDF of {series.name} ({dist_name})"
    assert cm["xlabel"] == "Value"
    assert cm["ylabel"] == "CDF"
    assert os.path.basename(cm['file_name']) == file_name

    # descriptive_stats
    assert ds["n"] == 150
    assert ds["distribution"] == dist_name
    assert isinstance(ds["params"], tuple)

    # which tests should be present?
    expected = {"ks", "cvm"} | ({"anderson"} if has_ad else set())
    assert set(tests) == expected

    # basic types for each test
    assert isinstance(tests["ks"]["statistic"], float)
    assert isinstance(tests["ks"]["p_value"], float)
    assert isinstance(tests["cvm"]["statistic"], float)
    assert isinstance(tests["cvm"]["p_value"], float)
    if has_ad:
        ad = tests["anderson"]
        assert isinstance(ad["statistic"], float)
        assert isinstance(ad["critical_value"], float)


# =======================================
# Helper Functions
# =======================================

def _fit_and_collect(series, dist, **kwargs):
    """Run the function and split out its parts."""
    res = plot_distribution_ecdf_vs_cdf(series, dist, **kwargs)
    return res["descriptive_stats"], res["inferential_stats"], res["chart_metadata"]
