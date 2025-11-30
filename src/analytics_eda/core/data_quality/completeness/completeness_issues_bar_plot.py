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
"""Horizontal bar chart summarizing completeness gaps for a Series."""

from collections.abc import Iterable
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
class CompletenessIssuesBarContext(SeriesBarChartContext):
    """Options for completeness bar chart."""

    title_template: str = "Completeness Issues for {name}{modifiers}"
    xlabel: str = "Percent of total"
    ylabel: str = "Completeness issue"
    is_orientation_vertical: bool = False
    bar_sort_descending: bool = True
    show_count_in_bar_label: bool = True
    show_subtitle: bool = True

    x_format: AxisFormat = field(default_factory=lambda: AxisFormat(kind="percent", decimals=1, percent_scale_0to1=True))
    y_format: AxisFormat = field(default_factory=lambda: AxisFormat(kind="category"))

    # Tokens treated as encoded missing values (case-insensitive, stripped)
    encoded_missing_tokens: Iterable[str] = field(
        default_factory=lambda: (
            "na",
            "n/a",
            "nan",
            "none",
            "null",
            "missing",
            "unknown",
            "-",
            "--",
            "?",
        )
    )


class CompletenessIssuesBarPlot(SeriesBarChartMixin, BasePlot):
    """
    Show how often values are absent or encoded as missing to assess completeness risk.

    Why this matters:
    - Completeness gaps drive biased results and failed joins; quantifying them guides cleanup.

    What this plot does:
    - Tallies missing (NaN), explicit null, blank/empty strings, and encoded missing tokens.
    - Plots a horizontal bar chart using percent of total rows as the base.
    """

    def __init__(self, ctx: CompletenessIssuesBarContext):
        parts = PlotParts(series_validator=named_only_validator(dropna=False, cast_str=False))
        super().__init__(ctx, parts)

    def plot_semantic_version(self) -> str:
        """Return the semantic version of this plot implementation."""
        return "1.0.0"

    # ---- compute -----------------------------------------------------
    def compute_descriptive(self, s: pd.Series) -> dict[str, Any]:
        """Compute completeness issue counts and percent-of-total bars."""
        is_missing = s.isna()
        is_null = s.map(lambda x: x is None)

        missing_only_mask = is_missing & ~is_null
        nonnull_mask = ~is_missing

        # Blank/encoded checks operate only on non-null values
        text_values = s[nonnull_mask].astype("string")
        stripped = text_values.str.strip()

        blank_mask = stripped == ""

        # Preserve caller token order while deduplicating (case-insensitive, stripped)
        encoded_tokens_ordered = []
        for tok in self.ctx.encoded_missing_tokens:
            key = str(tok).strip().lower()
            if key and key not in encoded_tokens_ordered:
                encoded_tokens_ordered.append(key)
        encoded_token_set = set(encoded_tokens_ordered)

        encoded_mask = stripped.str.lower().isin(encoded_token_set) & ~blank_mask

        counts = {
            "Missing": int(missing_only_mask.sum()),
            "Null": int(is_null.sum()),
            "Blank/Empty": int(blank_mask.sum()),
            "Encoded Missing": int(encoded_mask.sum()),
        }

        extra_params = {
            "encoded_missing_tokens": encoded_tokens_ordered,
        }

        desc = self.build_series_bar_desc(
            s,
            counts,
            denominator_key="pct_of_total",
            extra_params=extra_params,
        )

        return desc

    # ---- draft findings ----------------------------------------------
    def draft_descriptive_findings(self, desc: dict[str, Any]) -> dict[str, Any]:
        """Return a concise narrative about completeness gaps."""
        total = desc.get("total", 0)
        findings = {
            "context": f"{total:,} values",
            "primary_finding": "",
            "secondary_finding": None,
        }

        if not desc or total == 0:
            findings["primary_finding"] = "The series is empty."
            return findings

        total_gaps = int(desc.get("subset_count", 0))
        pct_gaps = float(desc.get("pct_subset", 0.0))

        if total_gaps == 0:
            findings["primary_finding"] = "All values are present; no completeness gaps detected."
            return findings

        findings["primary_finding"] = f"{self.formatter.format_percent(pct_gaps)} of {total:,} values are incomplete " f"({total_gaps:,} rows)."

        bars: dict[str, Any] = desc.get("bars", {})
        denom_key = desc.get("denominator_key", "pct_of_total")
        top_labels: list[str] = list(desc.get("top_labels", []))

        if not top_labels:
            return findings

        parts = []
        for lbl in sorted(top_labels):
            data = bars.get(lbl, {})
            pct = float(data.get(denom_key, 0.0))
            cnt = int(data.get("count", 0))
            parts.append(f"{lbl} ({self.formatter.format_percent(pct)}, {cnt:,} rows)")

        if len(parts) == 1:
            findings["secondary_finding"] = f"Most common issue: {parts[0]}."
        else:
            findings["secondary_finding"] = "Most common issues (tie): " + ", ".join(parts) + "."

        return findings

    # ---- subtitle ----------------------------------------------------
    def subtitle_text(self, desc, inf, chart_metadata) -> str:
        """Return a short subtitle summarizing the completeness gap rate."""
        if not desc or desc.get("total", 0) == 0:
            return ""

        total = desc["total"]
        total_gaps = int(desc.get("subset_count", 0))
        pct_gaps = float(desc.get("pct_subset", 0.0))

        if total_gaps == 0:
            return "All values present"

        return f"{self.formatter.format_percent(pct_gaps)} incomplete ({total_gaps:,} of {total:,} rows)"
