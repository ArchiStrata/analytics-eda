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

def test_norm_default_parameters_save(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.normal(size=100)
    series = make_float_series(data)

    result = numeric_distribution_analysis(series, report_path=tmp_path)
    hist_meta = result["report"]["central_tendency"]["plot_central_tendency_histogram"]
    chart = hist_meta["chart_metadata"]
    desc  = hist_meta["descriptive_stats"]

    # The default file_name comes from the default title
    expected_file = "Histogram with Central Tendency.png"
    saved_path = tmp_path / expected_file

    # File is saved
    assert saved_path.exists()
    assert os.path.basename(chart['relative_path']) == expected_file

    # Chart metadata defaults
    assert chart["title"]      == "Histogram with Central Tendency"
    assert chart["xlabel"]     == "Value"
    assert chart["ylabel"]     == "Count"
    assert chart["data_source"] is None
    assert isinstance(chart["bins"], int)

    # Descriptive stats present
    assert desc["n"] == 100
    assert isinstance(desc["mean"], float)
    assert isinstance(desc["median"], float)
    assert isinstance(desc["mode"], list)
    assert isinstance(desc["ci95"], tuple)

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
    hist_meta = result["report"]["central_tendency"]["plot_central_tendency_histogram"]
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
