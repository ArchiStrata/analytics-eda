import os
import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric import plot_distribution_qq_normality

def test_default_parameters_no_save():
    # small sample for default behavior (n < 50)
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="x")
    meta = plot_distribution_qq_normality(series)

    # top‐level structure
    assert "descriptive_stats" in meta
    assert "chart_metadata" in meta

    stats = meta["descriptive_stats"]
    chart = meta["chart_metadata"]

    # descriptive_stats keys
    for key in ["intercept", "slope", "r_squared",
                "median_residual", "iqr_residual", "max_abs_residual",
                "skewness", "kurtosis", "normality_tests"]:
        assert key in stats

    # normality_tests present and correct keys for n < 50
    nt = stats["normality_tests"]
    assert isinstance(nt, dict)
    assert "shapiro" in nt
    assert "anderson" in nt
    assert "reject_normality" in nt

    # chart_metadata defaults
    assert chart["title"] == "Q–Q Plot for Normality Assessment"
    assert chart["xlabel"] == "Theoretical Quantiles"
    assert chart["ylabel"] == "Sample Quantiles"
    assert chart["data_source"] is None
    assert chart["relative_path"] is None
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

    meta = plot_distribution_qq_normality(
        series,
        title=custom["title"],
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

    # normality_tests dict
    nt = desc["normality_tests"]
    assert isinstance(nt, dict)

    # for n=30: should have shapiro, dagostino_pearson, anderson but no jarque_bera
    assert "shapiro" in nt
    assert "dagostino_pearson" in nt
    assert "anderson" in nt
    assert "jarque_bera" not in nt

    # each test entry has a boolean 'reject'
    for name in ["shapiro", "dagostino_pearson", "anderson"]:
        entry = nt[name]
        assert "statistic" in entry
        # p-value may not exist for Anderson, but 'reject' must
        assert entry.get("reject") in (True, False)

    # overall flag
    assert "reject_normality" in nt
    assert isinstance(nt["reject_normality"], bool)

    # chart_metadata overrides
    assert chart["title"] == custom["title"]
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
    assert os.path.basename(chart["relative_path"]) == file_name

def test_default_parameters_and_save(tmp_path):
    series = pd.Series(np.linspace(0, 1, 25), name="u")
    file_name = "qq_default.png"

    meta = plot_distribution_qq_normality(
        series,
        save_path=str(tmp_path),
        file_name=file_name
    )
    chart = meta["chart_metadata"]

    # defaults preserved
    assert chart["title"] == "Q–Q Plot for Normality Assessment"
    assert chart["xlabel"] == "Theoretical Quantiles"
    assert chart["ylabel"] == "Sample Quantiles"
    assert chart["data_source"] is None
    assert pytest.approx(chart["alpha"]) == 0.05

    # file exists and non-empty
    saved = tmp_path / file_name
    assert saved.exists() and saved.stat().st_size > 0

def test_missing_series_name_raises_error():
    unnamed = pd.Series([0, 1, 2, 3])
    with pytest.raises(ValueError):
        plot_distribution_qq_normality(unnamed)

def test_empty_series_returns_stats_and_defaults():
    empty = pd.Series([], dtype=float, name="empty")
    meta = plot_distribution_qq_normality(
        empty
    )

    stats = meta["descriptive_stats"]
    chart = meta["chart_metadata"]

    # descriptive_stats all NaN or empty
    for key in ["intercept", "slope", "r_squared",
                "median_residual", "iqr_residual", "max_abs_residual",
                "skewness", "kurtosis"]:
        assert np.isnan(stats[key])
    assert stats["normality_tests"] == {}

    # chart_metadata defaults with alpha and path
    assert chart["title"] == "Q–Q Plot for Normality Assessment"
    assert chart["xlabel"] == "Theoretical Quantiles"
    assert chart["ylabel"] == "Sample Quantiles"
    assert chart["data_source"] is None
    assert pytest.approx(chart["alpha"]) == 0.05
    assert chart["relative_path"] is None
