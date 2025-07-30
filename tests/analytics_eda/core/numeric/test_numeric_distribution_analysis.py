import os
import pytest
import numpy as np
import pandas as pd
from analytics_eda.core.numeric.numeric_distribution_analysis import numeric_distribution_analysis

def make_float_series(data, name="x"):
    # ensure float dtype and proper name
    return pd.Series(data, dtype=float, name=name)

def test_missing_series_name_raises_error(tmp_path):
    # Series without a name should trigger validation error
    series = pd.Series([1.0, 2.0, 3.0], dtype=float)
    with pytest.raises(ValueError):
        numeric_distribution_analysis(series, report_path=tmp_path)

def test_norm_override_parameters_save(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.normal(size=50)
    series = make_float_series(data)

    overrides = {
        "title": "Custom Hist",
        "xlabel": "Custom X",
        "ylabel": "Custom Y",
        "data_source": "UnitTest",
        "bins": 5,
        "file_name": "custom_hist.png"
    }

    result = numeric_distribution_analysis(
        series,
        report_path=tmp_path,
        plot_central_tendency_histogram_overrides=overrides
    )
    hist_meta = result["report"]["central_tendency"]["histogram"]
    chart = hist_meta["chart_metadata"]
    desc  = hist_meta["descriptive_stats"]

    # File is saved under override name
    saved_path = tmp_path / "custom_hist.png"
    assert saved_path.exists()
    assert os.path.basename(chart['relative_path']) == "custom_hist.png"

    # Chart metadata matches overrides
    assert chart["title"]       == overrides["title"]
    assert chart["xlabel"]      == overrides["xlabel"]
    assert chart["ylabel"]      == overrides["ylabel"]
    assert chart["data_source"] == overrides["data_source"]
    assert chart["bins"]        == overrides["bins"]

    # Descriptive stats still valid
    assert desc["n"] == 50
    assert isinstance(desc["mean"], float)
    assert isinstance(desc["median"], float)
    assert isinstance(desc["mode"], list)
    assert isinstance(desc["ci95"], tuple)

@pytest.mark.parametrize(
    "dist_name, rng_func, support_adjust, has_anderson",
    [
        ("norm",    lambda r: r.normal(size=100),     lambda x: x,        True),
        ("lognorm", lambda r: r.lognormal(size=100), lambda x: np.abs(x)+1e-6, False),
        ("gamma",   lambda r: r.gamma(2.0, size=100),  lambda x: np.abs(x)+1e-6, False),
        ("expon",   lambda r: r.exponential(size=100), lambda x: np.abs(x),   True),
    ]
)
def test_numeric_distribution_analysis_basic_structure(
    dist_name, rng_func, support_adjust, has_anderson, tmp_path
):
    rng = np.random.default_rng(0)
    raw = rng_func(rng)
    raw = support_adjust(raw).astype(float)
    series = pd.Series(raw, name="x")

    # Run analysis without transforms
    result = numeric_distribution_analysis(series, report_path=tmp_path)
    report = result["report"]

    # 1) Top-level keys
    assert set(report) == {"central_tendency", "dispersion", "shape"}

    # 2) central_tendency → histogram
    ct = report["central_tendency"]
    assert set(ct) == {"histogram"}
    hist_meta = ct["histogram"]
    assert "descriptive_stats" in hist_meta and "chart_metadata" in hist_meta
    # Chart metadata saved file
    rel = hist_meta["chart_metadata"]["relative_path"]
    assert rel and (tmp_path / rel).exists()

    # 3) dispersion → boxplot
    disp = report["dispersion"]
    assert set(disp) == {"boxplot"}
    bp_meta = disp["boxplot"]
    assert "descriptive_stats" in bp_meta and "chart_metadata" in bp_meta
    rel = bp_meta["chart_metadata"]["relative_path"]
    assert rel and (tmp_path / rel).exists()

    # 4) shape contains ecdf_gap, density, distribution_fits
    shape = report["shape"]
    assert set(shape) == {"ecdf_gap", "density", "distribution_fits"}

    # ecdf_gap saved
    eg = shape["ecdf_gap"]["chart_metadata"]["relative_path"]
    assert eg and (tmp_path / eg).exists()

    # density saved
    dn = shape["density"]["chart_metadata"]["relative_path"]
    assert dn and (tmp_path / dn).exists()

    # distribution_fits for all four dist names
    fits = shape["distribution_fits"]
    assert set(fits) == {"norm", "lognorm", "gamma", "expon"}

    # inspect this distribution’s fit
    fit = fits[dist_name]
    assert set(fit) == {"ecdf_vs_cdf", "qq"}

    # ECDF vs. CDF
    ecdf_meta = fit["ecdf_vs_cdf"]
    desc = ecdf_meta["descriptive_stats"]
    tests = ecdf_meta["tests"]
    ecdf_cm    = ecdf_meta["chart_metadata"]

    # descriptive_stats
    assert desc["distribution"] == dist_name
    assert desc["n"] == series.size
    assert isinstance(desc["params"], tuple)

    # tests: KS and CvM always, Anderson only for norm/expon
    expected = {"ks", "cvm"}
    if has_anderson:
        expected.add("anderson")
    assert set(tests) == expected

    # file exists
    assert ecdf_cm["title"] == f"ECDF vs. Theoretical CDF ({dist_name})"
    rel = ecdf_cm["relative_path"]
    assert rel and (tmp_path / rel).exists()

    # Q–Q
    qq_meta = fit["qq"]
    qq_desc = qq_meta["descriptive_stats"]
    qq_cm   = qq_meta["chart_metadata"]

    for key in (
        "intercept","slope","r_squared",
        "median_residual","iqr_residual","max_abs_residual",
        "skewness","kurtosis"
    ):
        assert isinstance(qq_desc[key], float)

    assert qq_cm["title"] == f"Q–Q Plot Fit Assessment for ({dist_name})"
    assert qq_cm["distribution"] == dist_name
    rel = qq_cm["relative_path"]
    assert rel and (tmp_path / rel).exists()

    # 5) By default no transforms
    assert "transforms" not in shape
