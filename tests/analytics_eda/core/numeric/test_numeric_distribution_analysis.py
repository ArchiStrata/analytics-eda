import json
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from analytics_eda.core.numeric.numeric_distribution_analysis import numeric_distribution_analysis
from analytics_eda.core.numeric.evaluate_transforms import evaluate_transforms

def load_and_validate_report(response: dict, report_dir: Path) -> dict:
    """
    Given the return value of `numeric_distribution_analysis` and the directory
    where reports are written, this will:

      1. Assert that 'report_file_name' is present in the response.
      2. Assert that the file exists and is a regular file.
      3. Load it as JSON (failing if invalid).
      4. Return the parsed JSON.

    Usage in pytest:
        report = load_and_validate_report(out, tmp_path)
        # now you can make assertions about report['metadata'], report['data'], etc.
    """
    # 1. Key present
    assert 'report_file_name' in response, "response must contain 'report_file_name'"
    report_file = response['report_file_name']
    assert isinstance(report_file, str) and report_file, "report_file_name must be a non-empty string"

    # 2. File exists
    path = report_dir / report_file
    assert path.exists() and path.is_file(), f"Report file not found at {path!s}"

    # 3. Load & validate JSON
    try:
        with open(path, 'r', encoding='utf-8') as f:
            full_report = json.load(f)
    except json.JSONDecodeError as e:
        raise AssertionError(f"Report file is not valid JSON: {e}") from e

    # 4. Return parsed report
    return full_report


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
        "name": "Custom",
        "filter_desc": "filtered by New York",
        "xlabel": "Custom X",
        "ylabel": "Custom Y",
        "data_source": "UnitTest",
        "bins": 5,
        "file_name": "custom_hist.png"
    }

    out = numeric_distribution_analysis(
        series,
        report_path=tmp_path,
        plot_central_tendency_histogram_overrides=overrides
    )

    full_report = load_and_validate_report(out, tmp_path)
    report = full_report["data"]

    hist_meta = report["central_tendency"]["histogram"]
    chart = hist_meta["chart_metadata"]
    desc  = hist_meta["descriptive_stats"]

    # File is saved under override name
    saved_path = tmp_path / "custom_hist.png"
    assert saved_path.exists()
    assert chart['file_name'] == "custom_hist.png"

    # Chart metadata matches overrides
    assert chart["title"]       == "Distribution of Custom (filtered by New York): Central Tendency"
    assert chart["xlabel"]      == overrides["xlabel"]
    assert chart["ylabel"]      == overrides["ylabel"]
    assert chart["data_source"] == overrides["data_source"]
    assert chart["bins"]        == overrides["bins"]

    # Descriptive stats still valid
    assert desc["n"] == 50
    assert isinstance(desc["mean"], float)
    assert isinstance(desc["median"], float)
    assert isinstance(desc["modes"], list)

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
    out = numeric_distribution_analysis(series, report_path=tmp_path)

    full_report = load_and_validate_report(out, tmp_path)
    report = full_report["data"]

    # 1) Top-level keys
    assert set(report) == {"central_tendency", "dispersion", "shape"}

    # 2) central_tendency → histogram & violin
    ct = report["central_tendency"]
    assert set(ct) == {"histogram", "violin"}
    
    hist_meta = ct["histogram"]
    assert "descriptive_stats" in hist_meta and "chart_metadata" in hist_meta
    # Chart metadata saved file
    rel = hist_meta["chart_metadata"]["file_name"]
    assert rel and (tmp_path / rel).exists()

    # 3) dispersion → boxplot
    disp = report["dispersion"]
    assert set(disp) == {"boxplot"}
    bp_meta = disp["boxplot"]
    assert "descriptive_stats" in bp_meta and "chart_metadata" in bp_meta
    rel = bp_meta["chart_metadata"]["file_name"]
    assert rel and (tmp_path / rel).exists()

    # 4) shape contains ecdf_gap, density, distribution_fits
    shape = report["shape"]
    assert set(shape) == {"ecdf_gap", "density", "distribution_fits"}

    # ecdf_gap saved
    eg = shape["ecdf_gap"]["chart_metadata"]["file_name"]
    assert eg and (tmp_path / eg).exists()

    # density saved
    dn = shape["density"]["chart_metadata"]["file_name"]
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
    assert isinstance(desc["params"], list)

    # tests: KS and CvM always, Anderson only for norm/expon
    expected = {"ks", "cvm"}
    if has_anderson:
        expected.add("anderson")
    assert set(tests) == expected

    # file exists
    assert ecdf_cm["title"] == f"ECDF vs. Theoretical CDF of {series.name} ({dist_name})"
    rel = ecdf_cm["file_name"]
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

    assert qq_cm["title"] == f"Q–Q Plot Fit Assessment of {series.name} ({dist_name})"
    assert qq_cm["distribution"] == dist_name
    rel = qq_cm["file_name"]
    assert rel and (tmp_path / rel).exists()

    # 5) By default no transforms
    assert "transforms" not in shape

@pytest.mark.parametrize(
    "dist_name, rng_func, support_adjust",
    [
        ("norm",    lambda r: r.normal(size=100),      lambda x: x),
        ("lognorm", lambda r: r.lognormal(size=100),   lambda x: np.abs(x) + 1e-6),
        ("gamma",   lambda r: r.gamma(shape=2.0, size=100), lambda x: np.abs(x) + 1e-6),
        ("expon",   lambda r: r.exponential(size=100), lambda x: np.abs(x)),
    ]
)
def test_numeric_distribution_analysis_with_transforms(dist_name, rng_func, support_adjust, tmp_path):
    rng = np.random.default_rng(0)
    raw = rng_func(rng)
    raw = support_adjust(raw).astype(float)
    series = pd.Series(raw, name=dist_name)

    # Run analysis *with* transforms enabled
    out = numeric_distribution_analysis(
        series,
        report_path=tmp_path,
        evaluate_transforms_fn=evaluate_transforms
    )

    full_report = load_and_validate_report(out, tmp_path)
    report = full_report["data"]

    shape  = report["shape"]

    # transforms key should now be present
    assert "transforms" in shape

    transforms = shape["transforms"]
    # by design, select_transforms always includes at least these two
    expected_base = {"yeo-johnson", "arcsinh"}
    assert expected_base.issubset(transforms.keys())

    # each transform entry should itself be a full analysis dict
    for transform_name, transform_meta in transforms.items():
        assert isinstance(transform_meta, dict), f"{transform_name!r} meta must be a dict"

        full_transform_report = load_and_validate_report(transform_meta, tmp_path / transform_name)
        report_t = full_transform_report["data"]

        assert report_t is not None, f"{transform_name!r} entry missing 'report'"

        # 1) central_tendency → histogram
        hist_meta = report_t["central_tendency"]["histogram"]
        rel = hist_meta["chart_metadata"]["file_name"]
        assert rel, f"No file_name for histogram in transform {transform_name!r}"
        assert (tmp_path / transform_name / rel).exists(), \
            f"{tmp_path/transform_name/rel} missing for transform {transform_name!r}"

        # 2) shape → distribution_fits → norm → ecdf_vs_cdf
        ecdf_meta = report_t["shape"]["distribution_fits"]["norm"]["ecdf_vs_cdf"]
        rel = ecdf_meta["chart_metadata"]["file_name"]
        assert rel, f"No file_name for ECDF vs CDF in transform {transform_name!r}"
        assert (tmp_path / transform_name / rel).exists(), \
            f"{tmp_path/transform_name/rel} missing for transform {transform_name!r}"

