import pandas as pd
import pytest
from math import isclose

from analytics_eda.univariate.categorical.univariate_categorical_analysis import univariate_categorical_analysis
    
@pytest.mark.parametrize(
    "make_series, kwargs, expected_sections",
    [
        (
            lambda: pd.Series(
                ["A","A","A","B","B","C","C","C","C","D", None],
                name="pets", dtype="object"
            ),
            {
                "data_source": "UnitTest",
            },
            {
                # Expectations for the *top-level* missing_data section
                "missing_data": {
                    "total": 11,
                    "missing": 1,
                    # use a predicate for float comparison
                    "pct_missing": lambda v: isclose(v, 1/11, rel_tol=1e-12, abs_tol=1e-12),
                },
                # Expectations for the nested distribution report
                "distribution": {
                    "frequency_distribution": {
                        "pareto": {
                            "chart_metadata":  {
                                "title": "Pareto Chart of pets",
                                "xlabel": "Value",
                                "ylabel": "Count",
                                "data_source": "UnitTest",
                                "file_name": "Pareto Chart of pets.png"
                            },
                            "descriptive_stats": {
                                "mode": "C",
                                "total_count": 10,
                                "n_categories": 4,
                                "cumulative_count_at_80pct": 9
                            },
                        },
                    },
                    "balance": {
                        "density": {
                            "chart_metadata":  {
                                "title": "Distribution Density of count",
                                "xlabel": "Frequency",
                                "ylabel": "Density",
                                "data_source": "UnitTest",
                                "file_name": "Distribution Density of count.png"
                            },
                            "descriptive_stats": {
                                "n": 4,
                                "entropy_bits": 1.9999999999942293,
                                "skewness": 0.0,
                                "kurtosis": -1.1999999999999993,
                                "modes_count": 1,
                                "quartile_skew": 0.0,
                                "pct_10": 1.3,
                                "pct_25": 1.75,
                                "pct_50": 2.5,
                                "pct_75": 3.25,
                                "pct_90": 3.7
                            },
                        },
                        "boxplot": {
                            "chart_metadata":  {
                                "title": "Dispersion of count (IQR & Outliers)",
                                "ylabel": "Frequency",
                                "data_source": "UnitTest",
                                "file_name": "Dispersion of count (IQR & Outliers).png",
                                "std_outlier_multiplier": 4.0
                            },
                            "descriptive_stats": {
                                "n": 4,
                                "mean": 2.5,
                                "std": 1.2909944487358056,
                                "var": 1.6666666666666667,
                                "min": 1,
                                "max": 4,
                                "range": 3,
                                "mad": 1.0,
                                "cv": 0.5163977794943222,
                                "pct_10": 1.3,
                                "pct_25": 1.75,
                                "pct_75": 3.25,
                                "pct_90": 3.7,
                                "iqr": 1.5,
                                "extreme_lower_count": 0,
                                "extreme_upper_count": 0
                            },
                        },
                        "chi2_gof_uniform": {
                            "chart_metadata":  {
                                "title": "Chi-Square Goodness-of-Fit: pets",
                                "xlabel": "Frequency",
                                "ylabel": "Frequency",
                                "data_source": "UnitTest",
                                "file_name": "Chi-Square Goodness-of-Fit: pets.png"
                            },
                            "descriptive_stats": {
                                "total": 10,
                                "k": 4
                            },
                            "tests": {
                                "chi2_gof_null_uniform": {
                                    "statistic": 2.0,
                                    "p_value": 0.5724067044708798,
                                    "alpha": 0.05,
                                    "reject": False,
                                    "warning": "Some expected counts are below 5; chi-square test results may not be reliable."
                                }
                            },
                        }
                    },
                },
            },
        ),
    ],
    ids=["basic_report"],
)
def test_univariate_categorical_analysis_report_data_driven(
    tmp_path, load_and_validate_report, assert_plot_metadata,
    make_series, kwargs, expected_sections
):
    s = make_series()
    # point the root to tmp_path so files land where we can check them
    out = univariate_categorical_analysis(
        s, report_root=str(tmp_path), **kwargs
    )
    # load the final univariate report
    full = load_and_validate_report(out, tmp_path / s.name.replace(' ', '_'))

    # assert metadata
    assert 'metadata' in full
    for key in ("version", "report_name", "parameters"):
        assert key in full['metadata']

    assert "data" in full
    data = full["data"]

    # structure
    assert set(data.keys()) == {"missing_data", "distribution"}

    # ---- missing_data assertions (top-level, not a plot) ----
    expected_md = expected_sections["missing_data"]
    actual_md = data["missing_data"]
    for key, expected in expected_md.items():
        assert key in actual_md, f"missing_data missing key: {key!r}"
        if callable(expected):
            assert expected(actual_md[key]), f"Predicate failed for missing_data[{key!r}] = {actual_md[key]!r}"
        else:
            assert actual_md[key] == expected, f"missing_data[{key!r}] expected {expected!r}, got {actual_md[key]!r}"

    # ---- distribution assertions (nested report with plots) ----
    dist_full = load_and_validate_report(data["distribution"], tmp_path / s.name.replace(' ', '_'))
    dist = dist_full["data"]

    # check expected sections/plots
    for section, plots in expected_sections["distribution"].items():
        assert section in dist
        for plot_key, expectations in plots.items():
            payload = dist[section][plot_key]
            assert_plot_metadata(payload, expectations, tmp_path / s.name.replace(' ', '_'))


@pytest.mark.parametrize(
    "series_factory, expected_exc, match",
    [
        # Not a pandas Series
        (lambda: ["a", "b", "c"], TypeError, r"Input must be a pandas Series\."),

        # Not categorical/object dtype
        (lambda: pd.Series([1, 2, 3], name="numeric"), TypeError,
         r"must be categorical.*for categorical analysis"),

        # Missing name (None)
        (lambda: pd.Series(["x", "y", "z"], dtype="category"), ValueError,
         r"must have a non-empty 'name'"),

        # Blank/whitespace name
        (lambda: pd.Series(["x", "y", "z"], dtype="object", name=" "), ValueError,
         r"must have a non-empty 'name'"),
    ],
    ids=["not_series", "bad_dtype", "missing_name", "blank_name"],
)
def test_validate_categorical_named_series_errors(series_factory, expected_exc, match):
    obj = series_factory()
    with pytest.raises(expected_exc, match=match):
        univariate_categorical_analysis(obj)
