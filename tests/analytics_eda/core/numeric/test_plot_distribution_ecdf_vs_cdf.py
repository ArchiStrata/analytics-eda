import os
import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric import plot_distribution_ecdf_vs_cdf

def test_default_parameters_no_save():
    rng = np.random.default_rng(0)
    series = pd.Series(rng.normal(loc=0, scale=1, size=500), name="x")
    result = plot_distribution_ecdf_vs_cdf(series, "norm")
    desc = result["descriptive_stats"]
    chart = result["chart_metadata"]

    assert desc["n"] == 500
    assert desc["distribution"] == "norm"
    assert isinstance(desc["params"], tuple)
    assert "ks_statistic" in desc and "ks_p_value" in desc and "ks_reject" in desc
    assert chart["relative_path"] is None

def test_default_parameters_and_save(tmp_path):
    rng = np.random.default_rng(1)
    series = pd.Series(rng.normal(size=200), name="x")
    save_path = tmp_path / "plots"
    file_name = "ecdf.png"

    result = plot_distribution_ecdf_vs_cdf(
        series,
        "norm",
        save_path=str(save_path),
        file_name=file_name
    )
    desc = result["descriptive_stats"]
    chart = result["chart_metadata"]

    assert desc["distribution"] == "norm"
    assert os.path.basename(chart['relative_path']) == file_name
    assert (save_path / file_name).exists()

def test_missing_series_name_raises_error():
    series = pd.Series([1, 2, 3])  # name is None
    with pytest.raises(ValueError):
        plot_distribution_ecdf_vs_cdf(series, "norm")

def test_empty_series_returns_stats_and_defaults():
    series = pd.Series([], dtype=float, name="x")
    result = plot_distribution_ecdf_vs_cdf(series, "norm")
    assert "descriptive_stats" in result
    assert result["descriptive_stats"] == {"error": "empty series"}

@pytest.mark.parametrize(
    "dist_name, rng_func",
    [
        ("norm", lambda rng: rng.normal(size=150)),
        ("lognorm", lambda rng: rng.lognormal(size=150)),
        ("gamma", lambda rng: rng.gamma(shape=2.0, scale=2.0, size=150)),
        ("expon", lambda rng: rng.exponential(size=150)),
    ]
)
def test_all_distributions_and_save(dist_name, rng_func, tmp_path):
    rng = np.random.default_rng(42)
    raw = rng_func(rng).astype(float)

    # generate series with valid support
    if dist_name in ("lognorm", "gamma"):
        raw = np.abs(raw) + 1e-6
    elif dist_name == "expon":
        raw = np.abs(raw)

    series = pd.Series(raw, dtype=float, name="x")
    file_name = f"{dist_name}.png"

    result = plot_distribution_ecdf_vs_cdf(
        series,
        dist_name,
        save_path=str(tmp_path),
        file_name=file_name
    )
    desc = result["descriptive_stats"]
    chart = result["chart_metadata"]

    # verify file saved
    assert (tmp_path / file_name).exists()

    # descriptive stats
    assert desc["distribution"] == dist_name
    assert isinstance(desc["ks_statistic"], float)
    assert isinstance(desc["ks_p_value"], float)
    assert isinstance(desc["ks_reject"], bool)

    # chart metadata
    assert chart["title"] == "ECDF vs. Theoretical CDF"
    assert chart["xlabel"] == "Value"
    assert chart["ylabel"] == "CDF"
    assert chart["data_source"] is None
    assert chart["distribution"] == dist_name
    assert chart["alpha"] == 0.05
    assert os.path.basename(chart['relative_path']) == file_name
