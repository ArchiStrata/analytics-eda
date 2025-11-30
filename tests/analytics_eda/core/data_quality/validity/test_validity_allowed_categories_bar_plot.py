import pandas as pd
import pytest

from analytics_eda.core.data_quality import (
    ValidityAllowedCategoriesBarContext,
    ValidityAllowedCategoriesBarPlot,
)


@pytest.mark.parametrize(
    "make_series, kwargs, expect",
    [
        (
            lambda: pd.Series(["apple", "banana", "bad", "APPLE", "orange", "bad "], name="fruits"),
            {"allowed_categories": ["apple", "banana", "orange"], "case_sensitive_allowed": False},
            {
                "chart_metadata": {
                    "title": "Validity: Allowed Categories for fruits",
                    "xlabel": "Count",
                    "ylabel": "Measure",
                    "version": "1.0.0",
                    "file_name": None,
                },
                "descriptive_stats": {
                    "bars": {
                        "Distinct categories (raw)": {"count": 4},
                        "Invalid categories": {"count": 1},
                        "Distinct after removing invalid": {"count": 3},
                    },
                    "values_with_invalid": 2,
                    "distinct_raw": 4,
                    "distinct_after": 3,
                    "distinct_collapse": 1,
                    "subset_count": 2,
                    "pct_subset": pytest.approx(1 / 3),
                    "total_nonnull": 6,
                },
                "draft_descriptive_findings": {
                    "context": "N (non-null) = 6",
                    "primary_finding": "2 values fall outside the allowed categories; distinct categories drop from 4 to 3 (Δ = 1) after filtering.",
                },
            },
        ),
        (
            lambda: pd.Series(["apple", "banana"], name="fruits"),
            {"allowed_categories": ["apple", "banana", "orange"]},
            {
                "descriptive_stats": {
                    "values_with_invalid": 0,
                    "distinct_collapse": 0,
                },
                "draft_descriptive_findings": {
                    "primary_finding": "All values conform to the allowed categories; no invalid categories detected.",
                },
            },
        ),
        (
            lambda: pd.Series(["apple"], name="fruits"),
            {},
            {
                "descriptive_stats": {
                    "skip_plot": True,
                    "error": "allowed_categories not provided",
                },
            },
        ),
    ],
    ids=["invalid_present", "all_allowed", "no_allowlist"],
)
def test_validity_allowed_categories_bar_plot(make_series, kwargs, expect, assert_plot_metadata, tmp_path):
    ctx = ValidityAllowedCategoriesBarContext(base_dir=tmp_path, **kwargs)
    plot = ValidityAllowedCategoriesBarPlot(ctx)

    payload = plot.run(make_series())

    assert_plot_metadata(payload, expect, tmp_path)
