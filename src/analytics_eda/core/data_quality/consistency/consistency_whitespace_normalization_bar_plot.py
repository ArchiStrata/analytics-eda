# Copyright 2025 ArchiStrata, LLC and Andrew Dabrowski
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Consistency pillar: whitespace normalization impact."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from analytics_eda.core.visualization.base_plot import BasePlot
from analytics_eda.core.visualization.context.plot_context import AxisFormat
from analytics_eda.core.visualization.plot_mixins.series_bar_chart_mixin import (
    SeriesBarChartContext,
    SeriesBarChartMixin,
)
from analytics_eda.core.visualization.plot_parts import PlotParts
from analytics_eda.core.visualization.validation import named_only_validator


@dataclass
class ConsistencyWhitespaceNormalizationBarContext(SeriesBarChartContext):
    """Whitespace normalization impact — before vs. after trimming."""

    title_template: str = "Whitespace Normalization Impact for {name}{modifiers}"
    xlabel: str = "Count"
    ylabel: str = "Measure"
    is_orientation_vertical: bool = False
    bar_sort_descending: bool = False
    show_subtitle: bool = True

    x_format: AxisFormat = field(default_factory=lambda: AxisFormat(kind="number", decimals=0))
    y_format: AxisFormat = field(default_factory=lambda: AxisFormat(kind="category"))


class ConsistencyWhitespaceNormalizationBarPlot(SeriesBarChartMixin, BasePlot):
    """
    Show how trimming whitespace changes category cleanliness and cardinality.

    Why this matters:
        Leading/trailing whitespace quietly inflates category cardinality and obscures joins.
        Quantifying the rows and categories impacted makes normalization needs explicit.

    What this plot does:
        Counts distinct categories before trimming, distinct categories after trimming,
        and how many categories exhibit whitespace issues, using a horizontal bar chart.
    """

    def __init__(self, ctx: ConsistencyWhitespaceNormalizationBarContext):
        parts = PlotParts(series_validator=named_only_validator(dropna=False, cast_str=False))
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return the semantic version for this plot implementation."""
        return "1.0.0"

    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute counts for whitespace impact."""
        total = int(s.size)
        nonnull_mask = ~s.isna()
        total_nonnull = int(nonnull_mask.sum())

        desc = self.default_descriptive()
        desc.update({"total": total, "total_nonnull": total_nonnull})

        if total_nonnull == 0:
            desc.update({"skip_plot": True, "error": "no non-null values"})
            return desc

        s_nonnull = s[nonnull_mask].astype("string")
        stripped = s_nonnull.str.strip()

        distinct_raw = int(pd.unique(s_nonnull).size)
        distinct_after = int(pd.unique(stripped).size)

        whitespace_issue_mask = s_nonnull != stripped
        categories_with_issues = int(pd.unique(s_nonnull[whitespace_issue_mask]).size)
        values_with_issues = int(whitespace_issue_mask.sum())

        counts = {
            "Distinct categories (raw)": distinct_raw,
            "Categories with whitespace issues": categories_with_issues,
            "Distinct after trimming whitespace": distinct_after,
        }

        extra_params = {
            "values_with_issues": values_with_issues,
            "distinct_raw": distinct_raw,
            "distinct_after": distinct_after,
        }

        desc = self.build_series_bar_desc(
            s,
            counts,
            denominator_key="pct_of_nonnull",
            extra_params=extra_params,
            skip_plot_if_zero=False,
        )

        # cache delta for findings
        desc["distinct_collapse"] = int(distinct_raw - distinct_after)
        desc["distinct_raw"] = distinct_raw
        desc["distinct_after"] = distinct_after
        desc["values_with_issues"] = values_with_issues
        if total_nonnull:
            desc["subset_count"] = values_with_issues
            desc["pct_subset"] = float(values_with_issues / total_nonnull)
        return desc

    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Summarize whitespace impact on values and cardinality."""
        if not desc:
            return {}

        total_nonnull = int(desc.get("total_nonnull", 0))
        if total_nonnull == 0:
            return {
                "context": "0 non-null values",
                "primary_finding": "The series is empty.",
                "secondary_finding": None,
            }

        values_with_issues = int(desc.get("values_with_issues", 0))
        distinct_raw = int(desc.get("distinct_raw", desc.get("bars", {}).get("Distinct categories (raw)", {}).get("count", 0)))
        distinct_after = int(desc.get("distinct_after", desc.get("bars", {}).get("Distinct after trimming whitespace", {}).get("count", 0)))
        collapse = int(desc.get("distinct_collapse", distinct_raw - distinct_after))

        context = f"N (non-null) = {total_nonnull:,}"

        if values_with_issues == 0:
            return {
                "context": context,
                "primary_finding": "Whitespace is already normalized; no impacted values.",
                "secondary_finding": None,
            }

        primary = f"{values_with_issues:,} values show whitespace that trims away; distinct categories drop from {distinct_raw:,} to {distinct_after:,} (Δ = {collapse:,})."

        return {
            "context": context,
            "primary_finding": primary,
            "secondary_finding": None,
        }

    def subtitle_text(self, desc, inf, chart_metadata) -> str:
        """Short subtitle highlighting collapse after trimming."""
        total_nonnull = int(desc.get("total_nonnull", 0))
        if total_nonnull == 0:
            return ""
        distinct_raw = int(desc.get("distinct_raw", desc.get("bars", {}).get("Distinct categories (raw)", {}).get("count", 0)))
        distinct_after = int(desc.get("distinct_after", desc.get("bars", {}).get("Distinct after trimming whitespace", {}).get("count", 0)))
        return f"Non-null: {total_nonnull:,} • Distinct categories: {distinct_raw:,} → {distinct_after:,}"
