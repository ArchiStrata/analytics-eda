import pytest
import numpy as np
import pandas as pd
from analytics_eda.core.numeric.numeric_distribution_analysis import numeric_distribution_analysis
from analytics_eda.core.numeric.evaluate_transforms import evaluate_transforms


def make_float_series(data, name="x"):
    # ensure float dtype and proper name
    return pd.Series(data, dtype=float, name=name)

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
        numeric_distribution_analysis(make_input(), report_path=tmp_path, is_discrete=True)


@pytest.mark.parametrize(
    "make_series, kwargs, expected_distribution",
    [
        # --------------------------
        # Baseline (no transforms)
        # --------------------------
        # 1) Normal
        (
            lambda: pd.Series(np.random.default_rng(0).normal(loc=0, scale=1, size=150), name="norm", dtype="float64"),
            {"data_source": "UnitTest", "is_discrete": False},
            {
                "central_tendency": {
                    "histogram": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "violin":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap":   {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability":{"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "distribution_names": ("norm", "lognorm", "gamma", "expon"),
            },
        ),
        # 2) Lognormal (strictly positive)
        (
            lambda: pd.Series(np.random.default_rng(1).lognormal(mean=0.0, sigma=0.8, size=150), name="lognorm", dtype="float64"),
            {"data_source": "UnitTest", "is_discrete": False},
            {
                "central_tendency": {
                    "histogram": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "violin":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap":   {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability":{"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "distribution_names": ("norm", "lognorm", "gamma", "expon"),
            },
        ),
        # 3) Gamma (strictly positive)
        (
            lambda: pd.Series(np.random.default_rng(2).gamma(shape=2.0, scale=2.0, size=150), name="gamma", dtype="float64"),
            {"data_source": "UnitTest", "is_discrete": False},
            {
                "central_tendency": {
                    "histogram": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "violin":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap":   {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability":{"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "distribution_names": ("norm", "lognorm", "gamma", "expon"),
            },
        ),
        # 4) Exponential (non‑negative)
        (
            lambda: pd.Series(np.random.default_rng(3).exponential(scale=1.5, size=150), name="expon", dtype="float64"),
            {"data_source": "UnitTest", "is_discrete": False},
            {
                "central_tendency": {
                    "histogram": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "violin":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap":   {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability":{"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "distribution_names": ("norm", "lognorm", "gamma", "expon"),
            },
        ),
        # -----------------------------------------
        # Same 4 scenarios WITH transforms enabled
        # -----------------------------------------
        # 1) Normal + transforms
        (
            lambda: pd.Series(np.random.default_rng(0).normal(loc=0, scale=1, size=150), name="norm", dtype="float64"),
            {"data_source": "UnitTest", "is_discrete": False, "evaluate_transforms_fn": evaluate_transforms},
            {
                "central_tendency": {
                    "histogram": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "violin":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap":   {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability":{"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "distribution_names": ("norm", "lognorm", "gamma", "expon"),
            },
        ),
        # 2) Lognormal + transforms
        (
            lambda: pd.Series(np.random.default_rng(1).lognormal(mean=0.0, sigma=0.8, size=150), name="lognorm", dtype="float64"),
            {"data_source": "UnitTest", "is_discrete": False, "evaluate_transforms_fn": evaluate_transforms},
            {
                "central_tendency": {
                    "histogram": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "violin":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap":   {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability":{"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "distribution_names": ("norm", "lognorm", "gamma", "expon"),
            },
        ),
        # 3) Gamma + transforms
        (
            lambda: pd.Series(np.random.default_rng(2).gamma(shape=2.0, scale=2.0, size=150), name="gamma", dtype="float64"),
            {"data_source": "UnitTest", "is_discrete": False, "evaluate_transforms_fn": evaluate_transforms},
            {
                "central_tendency": {
                    "histogram": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "violin":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap":   {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability":{"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "distribution_names": ("norm", "lognorm", "gamma", "expon"),
            },
        ),
        # 4) Exponential + transforms
        (
            lambda: pd.Series(np.random.default_rng(3).exponential(scale=1.5, size=150), name="expon", dtype="float64"),
            {"data_source": "UnitTest", "is_discrete": False, "evaluate_transforms_fn": evaluate_transforms},
            {
                "central_tendency": {
                    "histogram": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "violin":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "dispersion": {
                    "boxplot": {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "shape": {
                    "ecdf_gap":   {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "density":    {"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                    "probability":{"chart_metadata": {"data_source": "UnitTest"}, "descriptive_stats": {}},
                },
                "distribution_names": ("norm", "lognorm", "gamma", "expon"),
            },
        ),
    ],
    ids=[
        "norm_series", "lognorm_series", "gamma_series", "expon_series",
        "norm_series_with_transforms", "lognorm_series_with_transforms",
        "gamma_series_with_transforms", "expon_series_with_transforms",
    ],
)
def test_numeric_distribution_analysis_param(
    make_series,
    kwargs,
    expected_distribution,
    tmp_path,
    load_and_validate_report,
    assert_plot_metadata,
):
    # Arrange
    s = make_series()

    # Act: run the full numeric distribution analysis (which writes nested reports)
    out = numeric_distribution_analysis(s, report_path=tmp_path, **kwargs)

    # Load the nested "distribution" report
    dist_loaded = load_and_validate_report(response=out, report_dir=tmp_path)["data"]  # fixture reads latest JSON under root

    # Assert the three distribution subsections exist
    assert set(dist_loaded.keys()) == {"central_tendency", "dispersion", "shape"}

    # ---- central_tendency ----
    for plot_key, exp in expected_distribution["central_tendency"].items():
        assert plot_key in dist_loaded["central_tendency"], f"missing central_tendency.{plot_key}"
        payload = dist_loaded["central_tendency"][plot_key]
        assert_plot_metadata(payload, exp, tmp_path)

    # ---- dispersion ----
    for plot_key, exp in expected_distribution["dispersion"].items():
        assert plot_key in dist_loaded["dispersion"], f"missing dispersion.{plot_key}"
        payload = dist_loaded["dispersion"][plot_key]
        assert_plot_metadata(payload, exp, tmp_path)

    # ---- shape ----
    for plot_key, exp in expected_distribution["shape"].items():
        assert plot_key in dist_loaded["shape"], f"missing shape.{plot_key}"
        payload = dist_loaded["shape"][plot_key]
        assert_plot_metadata(payload, exp, tmp_path)

    # Distribution fits (per-named distribution)
    fits = dist_loaded["shape"]["distribution_fits"]
    expected_names = set(expected_distribution["distribution_names"])
    assert set(fits.keys()) == expected_names

    for dist_name in expected_names:
        assert set(fits[dist_name].keys()) == {"ecdf_vs_cdf", "qq"}

        # --- ECDF vs CDF: handle support checks ---
        evc = fits[dist_name]["ecdf_vs_cdf"]
        evc_desc = evc["descriptive_stats"]
        evc_cm   = evc["chart_metadata"]

        if "error" in evc_desc:
            # Support violated → no file is expected
            assert evc_cm.get("file_name") is None, f"expected no file for unsupported {dist_name}"
            # Optional: assert the error message is one of the expected ones
            assert evc_desc["error"] in {"requires positive data", "requires non-negative data"}
        else:
            # Normal path → file must exist
            assert evc_cm.get("file_name"), f"missing file_name for {dist_name}.ecdf_vs_cdf"
            assert (tmp_path / evc_cm["file_name"]).exists(), f"missing saved file for {dist_name}.ecdf_vs_cdf"

        # --- QQ plot: always expect a saved file (no support early-return there) ---
        qq = fits[dist_name]["qq"]
        qq_cm = qq["chart_metadata"]
        assert qq_cm.get("file_name"), f"missing file_name for {dist_name}.qq"
        assert (tmp_path / qq_cm["file_name"]).exists(), f"missing saved file for {dist_name}.qq"

    # ---- transforms sanity (only when evaluate_transforms_fn was provided) ----
    if kwargs.get("evaluate_transforms_fn") is not None:
        shape = dist_loaded["shape"]
        assert "transforms" in shape, "Expected 'transforms' when evaluate_transforms_fn is provided"

        transforms = shape["transforms"]
        expected_base = {"yeo-johnson", "arcsinh"}
        assert expected_base.issubset(transforms.keys()), "Base transforms missing from results"

        for transform_name, transform_meta in transforms.items():
            assert isinstance(transform_meta, dict), f"{transform_name!r} meta must be a dict"

            full_transform_report = load_and_validate_report(transform_meta, tmp_path / transform_name)
            report_t = full_transform_report["data"]
            assert report_t is not None, f"{transform_name!r} entry missing nested 'data'"

            # histogram saved
            hist_meta = report_t["central_tendency"]["histogram"]
            rel = hist_meta["chart_metadata"]["file_name"]
            assert rel, f"No file_name for histogram in transform {transform_name!r}"
            assert (tmp_path / transform_name / rel).exists(), \
                f"{tmp_path/transform_name/rel} missing for transform {transform_name!r}"

            # ecdf_vs_cdf saved for norm
            evc_meta = report_t["shape"]["distribution_fits"]["norm"]["ecdf_vs_cdf"]
            rel2 = evc_meta["chart_metadata"]["file_name"]
            assert rel2, f"No file_name for ECDF vs CDF in transform {transform_name!r}"
            assert (tmp_path / transform_name / rel2).exists(), \
                f"{tmp_path/transform_name/rel2} missing for transform {transform_name!r}"
