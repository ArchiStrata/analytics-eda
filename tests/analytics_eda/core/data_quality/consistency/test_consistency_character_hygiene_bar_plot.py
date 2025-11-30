import pandas as pd
import pytest

from analytics_eda.core.data_quality import (
    ConsistencyCharacterHygieneBarContext,
    ConsistencyCharacterHygieneBarPlot,
)


@pytest.mark.parametrize(
    "make_series, expect",
    [
        (
            lambda: pd.Series(["good", "ba$d", "ba@d", None], name="chars"),
            {
                "chart_metadata": {
                    "title": "Character Hygiene for chars",
                    "xlabel": "Count",
                    "ylabel": "Measure",
                    "version": "1.0.0",
                    "file_name": None,
                },
                "descriptive_stats": {
                    "bars": {
                        "Distinct categories (raw)": {"count": 3},
                        "Categories with disallowed characters": {"count": 2},
                        "Distinct after sanitization": {"count": 2},
                    },
                    "distinct_raw": 3,
                    "distinct_after": 2,
                    "values_with_issues": 2,
                    "distinct_collapse": 1,
                    "total_nonnull": 3,
                },
                "draft_descriptive_findings": {
                    "context": "N (non-null) = 3",
                    "primary_finding": "2 values contain disallowed characters; distinct categories drop from 3 to 2 (Δ = 1) after sanitization.",
                },
            },
        ),
        (
            lambda: pd.Series(["alpha", "beta"], name="chars"),
            {
                "descriptive_stats": {
                    "values_with_issues": 0,
                    "distinct_collapse": 0,
                },
                "draft_descriptive_findings": {
                    "primary_finding": "All values comply with allowed characters; no sanitization needed.",
                },
            },
        ),
    ],
    ids=["disallowed_present", "all_allowed"],
)
def test_consistency_character_hygiene_bar_plot(make_series, expect, assert_plot_metadata, tmp_path):
    ctx = ConsistencyCharacterHygieneBarContext(base_dir=tmp_path)
    plot = ConsistencyCharacterHygieneBarPlot(ctx)

    payload = plot.run(make_series())

    assert_plot_metadata(payload, expect, tmp_path)
