import os
import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric import plot_distribution_qq_fit

def test_default_parameters_no_save():
    # small sample for default behavior (n < 50)
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="x")
    meta = plot_distribution_qq_fit(series, 'norm')

    # top‐level structure
    assert "descriptive_stats" in meta
    assert "chart_metadata" in meta

    stats = meta["descriptive_stats"]
    chart = meta["chart_metadata"]

    # descriptive_stats keys
    for key in ["intercept", "slope", "r_squared",
                "median_residual", "iqr_residual", "max_abs_residual",
                "skewness", "kurtosis"]:
        assert key in stats

    # inferential_stats present and correct keys for n < 50
    nt = meta["inferential_stats"]
    assert isinstance(nt, dict)
    assert "shapiro" in nt
    assert "reject_normality" in nt

    # chart_metadata defaults
    assert chart["title"] == f"Q–Q Plot Fit Assessment of {series.name} (norm)"
    assert chart["xlabel"] == "Theoretical Quantiles"
    assert chart["ylabel"] == "Sample Quantiles"
    assert chart["data_source"] is None
    assert chart["file_name"] is None
    assert pytest.approx(chart["alpha"]) == 0.05

def test_override_and_save(tmp_path):
    series = pd.Series(np.random.normal(size=30), name="z")
    custom = {
        "title": "My QQ Plot",
        "xlabel": "Theo Q",
        "ylabel": "Sample Q",
        "data_source": "UnitTest",
        "alpha": 0.1
    }
    file_name = "qq_override.png"

    meta = plot_distribution_qq_fit(
        series,
        distribution_name='norm',
        title_template=custom["title"],
        xlabel=custom["xlabel"],
        ylabel=custom["ylabel"],
        data_source=custom["data_source"],
        save_path=str(tmp_path),
        file_name=file_name,
        alpha=custom["alpha"]
    )

    desc = meta["descriptive_stats"]
    chart = meta["chart_metadata"]

    # all main metrics should be floats
    assert isinstance(desc["intercept"], float)
    assert isinstance(desc["slope"], float)
    assert 0.0 <= desc["r_squared"] <= 1.0
    assert isinstance(desc["median_residual"], float)
    assert isinstance(desc["iqr_residual"], float)
    assert isinstance(desc["max_abs_residual"], float)
    assert isinstance(desc["skewness"], float)
    assert isinstance(desc["kurtosis"], float)

    # inferential_stats dict
    nt = meta["inferential_stats"]
    assert isinstance(nt, dict)

    # for n=30: should have shapiro, dagostino_pearson, anderson but no jarque_bera
    assert "shapiro" in nt
    assert "dagostino_pearson" in nt
    assert "jarque_bera" not in nt

    # each test entry has a boolean 'reject'
    for name in ["shapiro", "dagostino_pearson"]:
        entry = nt[name]
        assert "statistic" in entry
        # p-value may not exist for Anderson, but 'reject' must
        assert entry.get("reject") in (True, False)

    # overall flag
    assert "reject_normality" in nt
    assert isinstance(nt["reject_normality"], bool)

    # chart_metadata overrides
    assert chart["title"] == f"{custom['title']} (norm)"
    assert chart["xlabel"] == custom["xlabel"]
    assert chart["ylabel"] == custom["ylabel"]
    assert chart["data_source"] == custom["data_source"]
    assert pytest.approx(chart["alpha"]) == custom["alpha"]

    # file was saved correctly
    saved = tmp_path / file_name
    assert saved.exists() and saved.stat().st_size > 0
    with open(saved, "rb") as f:
        sig = f.read(8)
    assert sig == b'\x89PNG\r\n\x1a\n'
    assert os.path.basename(chart["file_name"]) == file_name

def test_missing_series_name_raises_error():
    unnamed = pd.Series([0, 1, 2, 3])
    with pytest.raises(ValueError):
        plot_distribution_qq_fit(unnamed, 'norm')

def test_empty_series_returns_stats_and_defaults():
    empty = pd.Series([], dtype=float, name="empty")
    meta = plot_distribution_qq_fit(
        empty, 'norm'
    )

    stats = meta["descriptive_stats"]
    chart = meta["chart_metadata"]

    # descriptive_stats all NaN or empty
    for key in ["intercept", "slope", "r_squared",
                "median_residual", "iqr_residual", "max_abs_residual",
                "skewness", "kurtosis"]:
        assert np.isnan(stats[key])
    
    # formal inferential_stats
    assert meta["inferential_stats"] == {}

    # chart_metadata defaults with alpha and path
    assert chart["title"] == f"Q–Q Plot Fit Assessment of {empty.name} (norm)"
    assert chart["xlabel"] == "Theoretical Quantiles"
    assert chart["ylabel"] == "Sample Quantiles"
    assert chart["data_source"] is None
    assert pytest.approx(chart["alpha"]) == 0.05
    assert chart["file_name"] is None


@pytest.mark.parametrize(
    "dist_name, rng_func, expected_tests",
    [
        # Only Shapiro (n<50), no, no Jarque–Bera
        ("norm",    lambda rng: rng.normal(size=19),  
                    {"shapiro", "reject_normality"}),
        # Shapiro (n<50), D’Agostino (n>=20), no Jarque–Bera
        ("norm",    lambda rng: rng.normal(size=30),  
                    {"shapiro", "dagostino_pearson", "reject_normality"}),
        # Only D’Agostino (n>=20), no Jarque–Bera
        ("norm",    lambda rng: rng.normal(size=150), 
                    {"dagostino_pearson", "reject_normality"}),
        # D’Agostino + Jarque–Bera (n>2000)
        ("norm",    lambda rng: rng.normal(size=3000),
                    {"dagostino_pearson", "jarque_bera", "reject_normality"}),
        ("lognorm",lambda rng: rng.lognormal(size=150),    set()),
        ("gamma",  lambda rng: rng.gamma(shape=2.0, scale=2.0, size=150), set()),
        ("expon",  lambda rng: rng.exponential(size=150),  set()),
    ]
)
def test_plot_distribution_qq_fit_all_distributions(dist_name, rng_func, expected_tests, tmp_path):
    rng = np.random.default_rng(42)
    data = rng_func(rng).astype(float)

    # ensure support
    if dist_name in ("lognorm", "gamma"):
        data = np.abs(data) + 1e-6
    elif dist_name == "expon":
        data = np.abs(data)

    series = pd.Series(data, name="x")
    outdir = tmp_path / "plots"
    file_name = f"{dist_name}.png"

    result = plot_distribution_qq_fit(
        series,
        dist_name,
        save_path=str(outdir),
        file_name=file_name
    )

    ds = result["descriptive_stats"]
    tests = result["inferential_stats"]
    cm = result["chart_metadata"]

    # -- file saved correctly --
    saved = outdir / file_name
    assert saved.exists() and saved.stat().st_size > 0
    assert saved.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert os.path.basename(cm["file_name"]) == file_name

    # -- chart metadata --
    assert cm["title"] == f"Q–Q Plot Fit Assessment of {series.name} ({dist_name})"
    assert cm["xlabel"] == "Theoretical Quantiles"
    assert cm["ylabel"] == "Sample Quantiles"
    assert cm["data_source"] is None
    assert cm["distribution"] == dist_name
    assert cm["alpha"] == pytest.approx(0.05)

    # -- descriptive_stats --
    assert isinstance(ds["intercept"], float)
    assert isinstance(ds["slope"], float)
    assert isinstance(ds["r_squared"], float)
    assert isinstance(ds["median_residual"], float)
    assert isinstance(ds["iqr_residual"], float)
    assert isinstance(ds["max_abs_residual"], float)
    assert isinstance(ds["skewness"], float)
    assert isinstance(ds["kurtosis"], float)

    # -- tests presence/absence --
    assert set(tests.keys()) == expected_tests

    # -- if normality tests ran, check their fields --
    if "dagostino_pearson" in tests:
        dp = tests["dagostino_pearson"]
        assert set(dp.keys()) == {"statistic", "p_value", "reject"}
        assert isinstance(dp["statistic"], float)
        assert isinstance(dp["p_value"], float)
        assert isinstance(dp["reject"], bool)

    if "reject_normality" in tests:
        assert isinstance(tests["reject_normality"], bool)