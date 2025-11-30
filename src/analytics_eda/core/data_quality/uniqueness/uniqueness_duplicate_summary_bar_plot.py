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
"""Bar chart summarizing duplicate vs. distinct values for a Series."""

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
class UniquenessDuplicateSummaryBarContext(SeriesBarChartContext):
    """Options for duplicate vs. distinct summary."""

    title_template: str = "Duplicate Summary for {name}{modifiers}"
    xlabel: str = "Percent of non-null"
    ylabel: str = "Uniqueness status"
    is_orientation_vertical: bool = False
    bar_sort_descending: bool = True
    show_count_in_bar_label: bool = True
    show_subtitle: bool = True

    x_format: AxisFormat = field(default_factory=lambda: AxisFormat(kind="percent", decimals=1, percent_scale_0to1=True))
    y_format: AxisFormat = field(default_factory=lambda: AxisFormat(kind="category"))

    # Optional alert thresholds for duplicate ratio (0..1)
    high_duplicate_ratio_threshold: float | None = None
    low_duplicate_ratio_threshold: float | None = None


class UniquenessDuplicateSummaryBarPlot(SeriesBarChartMixin, BasePlot):
    """
    Show the split between distinct values and duplicate entries to assess uniqueness risk.

    Why this matters:
    - Duplicate-heavy columns can signal data quality issues, while all-unique columns may
      indicate identifiers or rare signals.

    What this plot does:
    - Counts non-null values, distinct values, and duplicate entries.
    - Plots a horizontal bar chart using percent of non-null as the base.
    """

    def __init__(self, ctx: UniquenessDuplicateSummaryBarContext):
        parts = PlotParts(series_validator=named_only_validator(dropna=False, cast_str=False))
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return the semantic version of this plot implementation."""
        return "1.0.0"

    # ---- compute -----------------------------------------------------
    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute distinct/duplicate counts and percent-of-non-null bars."""
        total = int(s.size)
        total_nonnull = int((~s.isna()).sum())

        # Guard against divide-by-zero downstream
        if total_nonnull == 0:
            desc = self.default_descriptive()
            desc["total"] = total
            desc["total_nonnull"] = 0
            desc["skip_plot"] = True
            desc["error"] = "no non-null values to summarize"
            return desc

        nunique_native = int(s.dropna().nunique(dropna=False))
        duplicates = max(total_nonnull - nunique_native, 0)
        duplicate_ratio = duplicates / total_nonnull
        distinct_ratio = nunique_native / total_nonnull
        one_value_column = nunique_native == 1

        extra_params = {
            "nunique_native": nunique_native,
            "duplicates": duplicates,
            "duplicate_ratio": duplicate_ratio,
            "distinct_ratio": distinct_ratio,
            "one_value_column": one_value_column,
            "high_duplicate_ratio_threshold": self.ctx.high_duplicate_ratio_threshold,
            "low_duplicate_ratio_threshold": self.ctx.low_duplicate_ratio_threshold,
        }

        counts = {
            "Distinct values": nunique_native,
            "Duplicate entries": duplicates,
        }

        desc = self.build_series_bar_desc(
            s,
            counts,
            denominator_key="pct_of_nonnull",
            extra_params=extra_params,
        )

        return desc

    # ---- draft findings ----------------------------------------------
    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Return a concise narrative about duplicate vs. distinct values."""
        total_nonnull = int(desc.get("total_nonnull", 0))
        total = int(desc.get("total", 0))
        duplicates = int(desc.get("params", {}).get("duplicates", desc.get("subset_count", 0)))
        duplicate_ratio = float(desc.get("params", {}).get("duplicate_ratio", 0.0))
        nunique = int(desc.get("params", {}).get("nunique_native", total_nonnull))

        findings = {
            "context": f"{total_nonnull:,} non-null values",
            "primary_finding": "",
            "secondary_finding": None,
        }

        if total == 0:
            findings["primary_finding"] = "The series is empty."
            return findings

        if total_nonnull == 0:
            findings["primary_finding"] = "No non-null values; duplicates cannot be assessed."
            return findings

        if duplicate_ratio == 0:
            findings["primary_finding"] = "All non-null values are unique."
            return findings

        if nunique == 1:
            findings["primary_finding"] = "All non-null values are identical (one-value column)."
            findings["secondary_finding"] = f"Duplicates account for {self.formatter.format_percent(duplicate_ratio)} of entries."
            return findings

        findings["primary_finding"] = f"{self.formatter.format_percent(duplicate_ratio)} of {total_nonnull:,} non-null values " f"are duplicates ({duplicates:,} entries)."
        findings["secondary_finding"] = f"Distinct values: {nunique:,} ({self.formatter.format_percent(1 - duplicate_ratio)})."

        # Threshold alerting, if configured
        high_thr = self.ctx.high_duplicate_ratio_threshold
        low_thr = self.ctx.low_duplicate_ratio_threshold
        if isinstance(high_thr, int | float) and duplicate_ratio >= float(high_thr):
            findings["secondary_finding"] = findings["secondary_finding"] + f" Duplicate ratio exceeds the alert threshold ({high_thr:.0%})."
        elif isinstance(low_thr, int | float) and duplicate_ratio <= float(low_thr):
            findings["secondary_finding"] = findings["secondary_finding"] + f" Duplicate ratio is below the low threshold ({low_thr:.0%})."

        return findings

    # ---- subtitle ----------------------------------------------------
    def subtitle_text(self, desc, inf, chart_metadata) -> str:
        """Return a short subtitle summarizing uniqueness vs. duplicates."""
        total_nonnull = int(desc.get("total_nonnull", 0))
        if total_nonnull == 0:
            return ""

        nunique = int(desc.get("params", {}).get("nunique_native", 0))
        duplicates = int(desc.get("params", {}).get("duplicates", 0))
        dup_ratio = float(desc.get("params", {}).get("duplicate_ratio", 0.0))

        parts = [
            f"Non-null: {total_nonnull:,}",
            f"Unique: {nunique:,} ({self.formatter.format_percent(1 - dup_ratio)})",
            f"Duplicates: {duplicates:,} ({self.formatter.format_percent(dup_ratio)})",
        ]
        return " • ".join(parts)
