import pytest
import pandas as pd

from analytics_eda.core.categorical.categorical_distribution_analysis import categorical_distribution_analysis

@pytest.mark.parametrize(
    "make_series, kwargs, expected_sections",
    [
        (
            # modestly imbalanced categories to populate all plots
            lambda: pd.Series(
                ["A","A","A","B","B","C","C","C","C","D", None],
                name="pets",
                dtype="object",
            ),
            { "data_source": "UnitTest" },
            {
                "frequency_distribution": {
                    "pareto": {
                        "chart_metadata":  {"xlabel": "Value", "ylabel": "Count", "data_source": "UnitTest"},
                        "descriptive_stats": {},  # nothing specific to assert
                    },
                },
                "balance": {
                    "density": {
                        "chart_metadata":  {"xlabel": "Frequency", "data_source": "UnitTest"},
                        "descriptive_stats": {},  # nothing specific to assert
                    },
                    "boxplot": {
                        "chart_metadata":  {"ylabel": "Frequency", "data_source": "UnitTest"},
                        "descriptive_stats": {},  # nothing specific to assert
                    },
                },
            },
        ),
    ],
    ids=["basic_report"],
)
def test_categorical_distribution_analysis_report_data_driven(
    tmp_path,
    load_and_validate_report,
    assert_plot_metadata,
    make_series,
    kwargs,
    expected_sections,
):
    # Arrange
    s = make_series()

    # Act
    out = categorical_distribution_analysis(
        s,
        report_path=tmp_path,
        **kwargs,
    )
    full_report = load_and_validate_report(out, tmp_path)

    # Assert
    assert "data" in full_report
    report = full_report["data"]

    # Sections present
    assert set(report.keys()) == set(expected_sections.keys())

    # For each expected section/plot: assert metadata + (optional) stats
    for section, plots in expected_sections.items():
        assert section in report, f"Missing section {section!r}"
        for plot_key, expectations in plots.items():
            assert plot_key in report[section], f"Missing plot {plot_key!r} in section {section!r}"

            payload = report[section][plot_key]
            
            assert_plot_metadata(payload, expectations, tmp_path)


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
def test_validate_categorical_named_series_errors(series_factory, expected_exc, match, tmp_path):
    obj = series_factory()
    with pytest.raises(expected_exc, match=match):
        categorical_distribution_analysis(obj, report_path=tmp_path)
