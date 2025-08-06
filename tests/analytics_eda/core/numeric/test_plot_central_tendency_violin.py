import os
import numpy as np
import pandas as pd
import pytest

from analytics_eda.core.numeric import plot_central_tendency_violin

def test_empty_series_returns_metadata():
    s = pd.Series([], dtype=float, name="empty")
    result = plot_central_tendency_violin(s)
    ds = result["descriptive_stats"]
    cm = result["chart_metadata"]

    assert ds["n"] == 0
    assert np.isnan(ds["mean"])
    assert np.isnan(ds["median"])
    assert ds["mean_ci"] == (np.nan, np.nan)
    assert ds["median_ci"] == (np.nan, np.nan)
    assert result["tests"] == {}
    assert "title" in cm and isinstance(cm["title"], str)

def test_missing_series_name_raises_error():
    s = pd.Series([1, 2, 3], dtype=float)  # no name
    with pytest.raises(ValueError):
        plot_central_tendency_violin(s)

def test_override_chart_labels_and_source():
    s = pd.Series([1, 2, 3], dtype=float, name="X")
    result = plot_central_tendency_violin(
        s,
        xlabel="My X",
        ylabel="My Y",
        data_source="DataSrc",
        title_template="Test {name}: Central {modifiers}",
        name="OverrideName",
        filter_desc="filtered",
        transform_desc="transformed"
    )
    cm = result["chart_metadata"]
    assert cm["xlabel"] == "My X"
    assert cm["ylabel"] == "My Y"
    assert cm["data_source"] == "DataSrc"
    # title should incorporate OverrideName, filtered, transformed
    assert "OverrideName" in cm["title"]
    assert "filtered" in cm["title"]
    assert "transformed" in cm["title"]

@pytest.mark.parametrize("method", [("t"), ("bootstrap")])
def test_save_mean_ci_methods(method, tmp_path):
    s = pd.Series([1, 2, 3, 4, 5], name="A")
    file_name = f"mean_ci_{method}.png"
    res = plot_central_tendency_violin(s, mean_ci_method=method, save_path=str(tmp_path), file_name=file_name)
    ds = res["descriptive_stats"]
    assert ds["mean_ci_method"] == method
    low, high = ds["mean_ci"]
    assert isinstance(low, float) and isinstance(high, float)
    assert low < ds["mean"] < high

    cm = res["chart_metadata"]
    assert "file_name" in cm
    assert file_name == cm["file_name"]

    saved = tmp_path / file_name
    assert saved.exists() and saved.is_file()
    assert saved.stat().st_size > 0
    # Check PNG signature
    with open(saved, 'rb') as f:
        sig = f.read(8)
    assert sig == b'\x89PNG\r\n\x1a\n'

def test_mean_ci_method_invalid():
    s = pd.Series([1, 2, 3], name="C")
    with pytest.raises(ValueError):
        plot_central_tendency_violin(s, mean_ci_method="invalid")

@pytest.mark.parametrize("method", [("bootstrap")])
def test_save_median_ci_methods(method, tmp_path):
    s = pd.Series([5, 6, 7, 8, 9], name="D")

    file_name = f"median_ci_{method}.png"

    res = plot_central_tendency_violin(s, median_ci_method=method, bootstrap_samples=100, save_path=str(tmp_path), file_name=file_name)
    ds = res["descriptive_stats"]
    assert ds["median_ci_method"] == method
    low, high = ds["median_ci"]
    assert isinstance(low, float) and isinstance(high, float)
    assert low <= ds["median"] <= high

    cm = res["chart_metadata"]
    assert "file_name" in cm
    assert file_name == cm["file_name"]

    saved = tmp_path / file_name
    assert saved.exists() and saved.is_file()
    assert saved.stat().st_size > 0
    # Check PNG signature
    with open(saved, 'rb') as f:
        sig = f.read(8)
    assert sig == b'\x89PNG\r\n\x1a\n'

def test_median_ci_method_invalid():
    s = pd.Series([1, 2, 3], name="E")
    with pytest.raises(ValueError):
        plot_central_tendency_violin(s, median_ci_method="invalid")  # must be 'bootstrap' or None

def test_save_popmean(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.normal(loc=10, scale=2, size=50)
    s = pd.Series(data, name="F")

    file_name = "popmean.png"

    popmean = 10.0
    popvariance = 5.0
    res = plot_central_tendency_violin(s, popmean=popmean, popvariance=popvariance, save_path=str(tmp_path), file_name=file_name)

    assert "descriptive_stats" in res
    assert res["descriptive_stats"]["popmean"] == popmean
    assert res["descriptive_stats"]["popvariance"] == popvariance

    tests = res["tests"]
    assert "t_test" in tests and "cohens_d" in tests
    assert isinstance(tests["t_test"]["statistic"], float)
    assert isinstance(tests["t_test"]["p_value"], float)
    assert isinstance(tests["t_test"]["reject"], bool)

    assert isinstance(tests["cohens_d"], float)

    assert isinstance(tests["z_test"]["statistic"], float)
    assert isinstance(tests["z_test"]["p_value"], float)
    assert isinstance(tests["z_test"]["reject"], bool)

    cm = res["chart_metadata"]
    assert "file_name" in cm
    assert file_name == cm["file_name"]

    saved = tmp_path / file_name
    assert saved.exists() and saved.is_file()
    assert saved.stat().st_size > 0
    # Check PNG signature
    with open(saved, 'rb') as f:
        sig = f.read(8)
    assert sig == b'\x89PNG\r\n\x1a\n'

def test_save_popmedian(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.normal(loc=0, scale=1, size=30)
    s = pd.Series(data, name="G")
    file_name = "popmedian.png"

    popmedian=0.0

    res = plot_central_tendency_violin(s, popmedian=popmedian, save_path=str(tmp_path), file_name=file_name)

    assert "descriptive_stats" in res
    assert res["descriptive_stats"]["popmedian"] == popmedian

    tests = res["tests"]
    assert "wilcoxon" in tests
    # sign test only if there are non-zero diffs
    assert "sign_test" in tests or all(d == 0 for d in (s - 0.0))

    cm = res["chart_metadata"]
    assert "file_name" in cm
    assert file_name == cm["file_name"]

    saved = tmp_path / file_name
    assert saved.exists() and saved.is_file()
    assert saved.stat().st_size > 0
    # Check PNG signature
    with open(saved, 'rb') as f:
        sig = f.read(8)
    assert sig == b'\x89PNG\r\n\x1a\n'
