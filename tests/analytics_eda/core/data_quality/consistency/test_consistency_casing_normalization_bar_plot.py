import pandas as pd
import pytest

from analytics_eda.core.data_quality import (
    ConsistencyCasingNormalizationBarContext,
    ConsistencyCasingNormalizationBarPlot,
)


@pytest.mark.parametrize(
    "make_series, expect",
    [
        (
            lambda: pd.Series(["Red", " red", "BLUE", "blue ", None], name="colors"),
            {
                "chart_metadata": {
                    "title": "Casing Normalization Impact for colors",
                    "xlabel": "Count",
                    "ylabel": "Measure",
                    "version": "1.0.0",
                    "file_name": None,
                },
                "descriptive_stats": {
                    "bars": {
                        "Distinct categories (raw)": {"count": 4},
                        "Categories with casing collisions": {"count": 4},
                        "Distinct after case normalization": {"count": 2},
                    },
                    "distinct_raw": 4,
                    "distinct_after": 2,
                    "values_with_collisions": 4,
                    "distinct_collapse": 2,
                    "total_nonnull": 4,
                },
                "draft_descriptive_findings": {
                    "context": "N (non-null) = 4",
                    "primary_finding": "4 values participate in casing collisions; distinct categories drop from 4 to 2 (Δ = 2).",
                },
            },
        ),
        (
            lambda: pd.Series(["one", "two", "three"], name="colors"),
            {
                "descriptive_stats": {
                    "values_with_collisions": 0,
                    "distinct_collapse": 0,
                },
                "draft_descriptive_findings": {
                    "primary_finding": "Casing is already consistent; no collisions detected.",
                },
            },
        ),
    ],
    ids=["mixed_casing", "already_consistent"],
)
def test_consistency_casing_normalization_bar_plot(make_series, expect, assert_plot_metadata, tmp_path):
    ctx = ConsistencyCasingNormalizationBarContext(base_dir=tmp_path)
    plot = ConsistencyCasingNormalizationBarPlot(ctx)

    payload = plot.run(make_series())

    assert_plot_metadata(payload, expect, tmp_path)
