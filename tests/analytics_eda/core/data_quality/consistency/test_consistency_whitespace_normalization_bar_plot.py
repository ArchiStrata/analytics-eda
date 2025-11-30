import pandas as pd
import pytest

from analytics_eda.core.data_quality import (
    ConsistencyWhitespaceNormalizationBarContext,
    ConsistencyWhitespaceNormalizationBarPlot,
)


@pytest.mark.parametrize(
    "make_series, expect",
    [
        (
            lambda: pd.Series([" Red", "Red", "blue ", "blue", None], name="colors"),
            {
                "chart_metadata": {
                    "title": "Whitespace Normalization Impact for colors",
                    "xlabel": "Count",
                    "ylabel": "Measure",
                    "version": "1.0.0",
                    "file_name": None,
                },
                "descriptive_stats": {
                    "bars": {
                        "Distinct categories (raw)": {"count": 4},
                        "Categories with whitespace issues": {"count": 2},
                        "Distinct after trimming whitespace": {"count": 2},
                    },
                    "distinct_raw": 4,
                    "distinct_after": 2,
                    "values_with_issues": 2,
                    "distinct_collapse": 2,
                    "total_nonnull": 4,
                },
                "draft_descriptive_findings": {
                    "context": "N (non-null) = 4",
                    "primary_finding": "2 values show whitespace that trims away; distinct categories drop from 4 to 2 (Δ = 2).",
                },
            },
        ),
        (
            lambda: pd.Series(["one", "two"], name="clean"),
            {
                "descriptive_stats": {
                    "values_with_issues": 0,
                    "distinct_collapse": 0,
                },
                "draft_descriptive_findings": {
                    "primary_finding": "Whitespace is already normalized; no impacted values.",
                },
            },
        ),
    ],
    ids=["whitespace_present", "already_trimmed"],
)
def test_consistency_whitespace_normalization_bar_plot(make_series, expect, assert_plot_metadata, tmp_path):
    ctx = ConsistencyWhitespaceNormalizationBarContext(base_dir=tmp_path)
    plot = ConsistencyWhitespaceNormalizationBarPlot(ctx)

    payload = plot.run(make_series())

    assert_plot_metadata(payload, expect, tmp_path)
